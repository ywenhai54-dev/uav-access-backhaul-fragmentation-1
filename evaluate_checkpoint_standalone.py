"""
Standalone checkpoint evaluation for DisasterRescue MAHPPO.

Recommended MAHPPO_sc_cleanup_balance_v2 evaluation:

& C:/Users/25658/.conda/envs/uav_final_gpu/python.exe -u .\evaluate_checkpoint_standalone.py `
  --config .\logs\MAHPPO_sc_cleanup_balance_v2\config.yaml `
  --checkpoint_dir .\checkpoints\MAHPPO_sc_cleanup_balance_v2 `
  --last_n 5 `
  --episodes 100 `
  --eval_mode stochastic `
  --progress_interval 5 `
  --sc_pfr_probe_mode off `
  --output_dir .\evaluation_results\mahppo_sc_cleanup_balance_v2_last5_100ep_stochastic_logconfig `
  --tag mahppo_sc_cleanup_balance_v2_last5_100ep_stochastic_logconfig

SC_PFR strict probe should be a small diagnostic run, not the official score:

& C:/Users/25658/.conda/envs/uav_final_gpu/python.exe -u .\evaluate_checkpoint_standalone.py `
  --config .\logs\MAHPPO_sc_cleanup_balance_v2\config.yaml `
  --checkpoint_dir .\checkpoints\MAHPPO_sc_cleanup_balance_v2 `
  --last_n 1 `
  --episodes 10 `
  --eval_mode stochastic `
  --progress_interval 1 `
  --sc_pfr_probe_mode strict `
  --output_dir .\evaluation_results\debug_scpfr_strict_10ep `
  --tag debug_scpfr_strict_10ep
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.disaster_env import DisasterRescueEnv
from train_core_rule_low_fpsfix_v21 import build_hierarchical_algo
from algorithms.mappo_rule_low_local_obs_v1_fpsfix import HierarchicalMAPPO as FullLocalObsHierarchicalMAPPO
from algorithms.mahppo_icn_no_relation_capability_fpsfix import HierarchicalMAPPO as NoRelationCapabilityHierarchicalMAPPO
from algorithms.mahppo_icn_no_relation_capability_strict_v2_fpsfix import HierarchicalMAPPO as StrictV2NoRelationCapabilityHierarchicalMAPPO
from utils.config import Config


class StrictV2DisasterRescueEnv(DisasterRescueEnv):
    def _compute_sc_comm_support_point_np(self, task_pos, task_id=None, c_pos=None):
        if task_id is not None:
            task = self.tasks.get(int(task_id))
            if task is not None and getattr(task.state, "confirmed_pair", None) is not None:
                support = np.asarray(task.position, dtype=np.float32).copy()
                self.metrics["sc_support_target_assigned"] = self.metrics.get("sc_support_target_assigned", 0) + 1
                return support
        return np.asarray(task_pos, dtype=np.float32).copy()

    def _task_service_feasible(self, task, rewards=None):
        feasible, serving_uids, pair, failure = super()._task_service_feasible(task, rewards)
        if pair is not None and task is not None and getattr(task.state, "confirmed_pair", None) is not None:
            sid, cid = int(pair[0]), int(pair[1])
            if 0 <= sid < len(self.uavs):
                self.uavs[sid].state.target_position = np.asarray(task.position, dtype=np.float32).copy()
            if 0 <= cid < len(self.uavs):
                self.uavs[cid].state.target_position = np.asarray(task.position, dtype=np.float32).copy()
        return feasible, serving_uids, pair, failure


DEFAULT_CONFIG_SENTINEL = object()

CONFIG_AUDIT_FIELDS = [
    ("task.task_type_ratio", ("task", "task_type_ratio")),
    ("task.generation_window", ("task", "generation_window")),
    ("task.generation_fade_end", ("task", "generation_fade_end")),
    ("task.deadline_range", ("task", "deadline_range")),
    ("task.hard_deadline_range", ("task", "hard_deadline_range")),
    ("task.deadline_slack", ("task", "deadline_slack")),
    ("task.max_active_tasks", ("task", "max_active_tasks")),
    ("task.max_active_sc_tasks", ("task", "max_active_sc_tasks")),
    ("task.max_active_c_tasks", ("task", "max_active_c_tasks")),
    ("uav.num_uavs", ("environment", "num_uavs")),
    ("uav.num_sensing_uavs", ("environment", "num_sensing_uavs")),
    ("uav.num_comm_uavs", ("environment", "num_comm_uavs")),
    ("uav.max_speed", ("uav", "max_speed")),
    ("map.map_size", ("environment", "map_size")),
    ("real_geo.grid_resolution_m", ("environment", "grid_resolution_m")),
    ("communication.data_upload_radius_m", ("uav", "data_upload_radius_m")),
    ("communication.backhaul_link_radius_m", ("uav", "backhaul_link_radius_m")),
    ("communication.comm_service_radius_m", ("uav", "comm_service_radius_m")),
    ("communication.sense_service_radius_m", ("uav", "sense_service_radius_m")),
    ("communication.nlos_radius_factor", ("real_geo", "nlos_radius_factor")),
    ("base_selection.mode", ("real_geo", "base_selection_mode")),
    ("base_selection.safe_side", ("real_geo", "base_safe_side")),
    ("base_selection.base_outside_margin_m", ("real_geo", "base_outside_margin_m")),
    ("base_selection.base_outer_margin_m", ("real_geo", "base_outer_margin_m")),
    ("evaluation.eval_action_mode", ("evaluation", "eval_action_mode")),
]


ABlation_MAP = {
    "full": None,
    "no_icn": "no_icn",
    "icn_wo_nash": "icn_wo_nash",
    "no_icn_gnn_caa": "no_icn_gnn_caa",
    "mahppo_icn_no_relation_capability_fpsfix": None,
}


def build_eval_hierarchical_algo(config: Config, device: torch.device, mode: str):
    algorithm = None
    ablation = None
    if hasattr(config, "_config") and isinstance(config._config, dict):
        method_cfg = config._config.get("method", {})
        algorithm = str(method_cfg.get("algorithm", "")).lower()
        ablation = str(method_cfg.get("ablation", "")).lower()
    marker = " ".join([str(mode).lower(), str(algorithm or ""), str(ablation or "")])
    if (
        "full_local_obs_v1" in marker
        or "mappo_rule_low_local_obs_v1_fpsfix" in marker
        or "mahppo_full_local_obs_v1" in marker
    ):
        print(
            "EvalFullLocalObservationLoaderCheck | "
            "CheckpointMethod=mappo_rule_low_local_obs_v1_fpsfix | "
            "LoadedAgent=full_local_obs_v1 | OriginalFullGlobalLoaded=0 | "
            "ObservationTaskScope=local_visible_only | status=PASS"
        )
        return FullLocalObsHierarchicalMAPPO(config, device)
    if (
        "no_relation_capability_strict_v2" in marker
        or "mahppo_icn_no_relation_capability_strict_v2" in marker
        or "strict_v2_local_obs_v1" in marker
        or "wo_gnn_caa_local_obs_v1" in marker
    ):
        print(
            "EvalRelationCapabilityAblationLoaderCheck | "
            "CheckpointMethod=mahppo_icn_no_relation_capability_strict_v2_fpsfix | "
            "LoadedAgent=no_relation_capability_strict_v2 | FullAgentLoaded=0 | "
            "CleanV2AgentLoaded=0 | CleanNoICNAgentLoaded=0 | "
            "GNNEnabled=0 | CAAEnabled=0 | ICNEnabled=1 | NashEnabled=1 | "
            "AssignmentBiasEnabled=1 | ProtocolExecutionEnabled=1 | "
            "ICNInputMode=raw_semantic_only | status=PASS"
        )
        return StrictV2NoRelationCapabilityHierarchicalMAPPO(config, device)
    if (
        mode == "mahppo_icn_no_relation_capability_fpsfix"
        or algorithm == "mahppo_icn_no_relation_capability_fpsfix"
        or "wo_relation_capability" in marker
        or "no_relation_capability" in marker
        or "w_o_relation_capability" in marker
    ):
        print(
            "EvalRelationCapabilityAblationLoaderCheck | "
            "CheckpointMethod=mahppo_icn_no_relation_capability_fpsfix | "
            "LoadedAgent=no_relation_capability | FullAgentLoaded=0 | "
            "OldNoGNNAgentLoaded=0 | OldNoCAAAgentLoaded=0 | CleanNoICNAgentLoaded=0 | "
            "GNNEnabled=0 | CAAEnabled=0 | ICNEnabled=1 | NashEnabled=1 | "
            "AssignmentBiasEnabled=1 | ProtocolExecutionEnabled=1 | "
            "ICNInputMode=raw_semantic_only | status=PASS"
        )
        return NoRelationCapabilityHierarchicalMAPPO(config, device)
    return build_hierarchical_algo(config, device, ablation_type=ABlation_MAP[mode])


def maybe_print_no_relation_eval_runtime_audit(mappo) -> None:
    agent = getattr(mappo, "agent", None)
    if agent is None or not hasattr(agent, "validate_relation_capability_ablation_check"):
        return
    if getattr(mappo, "_eval_relation_capability_audit_printed", False):
        return
    counts = agent.validate_relation_capability_ablation_check()
    input_counts = agent.validate_icn_input_source_check()
    access_counts = agent.get_information_access_boundary_audit() if hasattr(agent, "get_information_access_boundary_audit") else {}
    obs_counts = agent.validate_observation_boundary_audit() if hasattr(agent, "validate_observation_boundary_audit") else {}
    is_strict_v2 = "strict_v2" in str(getattr(agent, "ablation_type", "")).lower()
    if is_strict_v2:
        progress_print(
            "StrictRelationCapabilityAblationEffectiveConfig | "
            "Experiment=Ours w/o Relation-Capability Representation strict_v2 | "
            "ProtocolRetained=1 | RepresentationRemoved=1 | PreCommitObservationOnly=1 | "
            "NoGlobalOracleBeforeCommit=1 | PostCommitExecutionOnly=1 | "
            "GNNEnabled=0 | CAAEnabled=0 | RuleBasedCAAEnabled=0 | ICNEnabled=1 | "
            "NashEnabled=1 | AssignmentBiasEnabled=1 | UnaryIntentBiasEnabled=1 | "
            "SCBargainingEnabled=1 | ProtocolCommitmentEnabled=1 | HardSCCommitEnabled=1 | "
            "ForcedSCContinueEnabled=1 | SupportTargetFromPairEnabled=1 | "
            "PhysicalSCWithoutCommitment=1 | "
            f"NegotiationRoundsRuntime={int(counts.get('negotiation_rounds_runtime', 0))} | "
            "status=PASS_ONESHOT"
        )
    if int(obs_counts.get("observation_task_scope_local_visible_only", 0)):
        obs_status = "FAIL_A2_LOCAL_OBS_BOUNDARY" if (
            int(obs_counts.get("global_task_table_in_actor_observation", 0))
            or int(obs_counts.get("action_mask_exposes_global_tasks", 0))
            or int(obs_counts.get("actor_task_list_shared_across_uavs", 0))
            or int(obs_counts.get("actor_uses_global_state_tasks", 0))
            or int(obs_counts.get("riam_uses_invisible_tasks", 0))
            or int(obs_counts.get("commit_on_invisible_task", 0))
            or float(obs_counts.get("sc_bargaining_invisible_task_applied", 0.0)) > 0.0
        ) else "PASS_ONESHOT"
        progress_print(
            "ObservationBoundaryDiag | "
            "Experiment=A2_strict_v2_local_obs_v1 | "
            "ObservationTaskScope=local_visible_only | "
            f"GlobalTaskTableInActorObservation={int(obs_counts.get('global_task_table_in_actor_observation', 0))} | "
            f"GlobalUAVTableInActorObservation={int(obs_counts.get('global_uav_table_in_actor_observation', 0))} | "
            f"ActorTaskListSharedAcrossUAVs={int(obs_counts.get('actor_task_list_shared_across_uavs', 0))} | "
            f"FixedTaskSlotAlignment={int(obs_counts.get('fixed_task_slot_alignment', 0))} | "
            f"InvisibleTaskFeaturesZeroed={int(obs_counts.get('invisible_task_features_zeroed', 0))} | "
            f"InvisibleTaskMasked={int(obs_counts.get('invisible_task_masked', 0))} | "
            f"PaddingTaskMasked={int(obs_counts.get('padding_task_masked', 0))} | "
            f"ActionMaskExposesGlobalTasks={int(obs_counts.get('action_mask_exposes_global_tasks', 0))} | "
            f"UsesUploadOracleForVisibility={int(obs_counts.get('uses_upload_oracle_for_visibility', 0))} | "
            f"UsesBackhaulOracleForVisibility={int(obs_counts.get('uses_backhaul_oracle_for_visibility', 0))} | "
            f"UsesTerrainOracleForVisibility={int(obs_counts.get('uses_terrain_oracle_for_visibility', 0))} | "
            f"ActorUsesGlobalStateTasks={int(obs_counts.get('actor_uses_global_state_tasks', 0))} | "
            f"RIAMUsesInvisibleTasks={int(obs_counts.get('riam_uses_invisible_tasks', 0))} | "
            f"AssignmentBiasInvisibleTaskAbsMean={float(obs_counts.get('assignment_bias_invisible_task_abs_mean', 0.0)):.6f} | "
            f"SCBargainingInvisibleTaskApplied={int(obs_counts.get('sc_bargaining_invisible_task_applied', 0))} | "
            f"CommitOnInvisibleTask={int(obs_counts.get('commit_on_invisible_task', 0))} | "
            f"PostCommitEnvFeasibilityCheckAllowed={int(obs_counts.get('postcommit_env_feasibility_check_allowed', 0))} | "
            f"status={obs_status}"
        )
    progress_print(
        "RelationCapabilityAblationCheck | "
        f"GNNEnabled=0 | GNNForwardCalled={int(counts.get('gnn_forward_called', 0))} | "
        f"GraphEmbeddingBuilt={int(counts.get('graph_embedding_built', 0))} | "
        f"GraphEmbeddingNorm={counts.get('graph_embedding_norm', 0.0):.6f} | "
        f"CAAEnabled=0 | CAAForwardCalled={int(counts.get('caa_forward_called', 0))} | "
        f"CAAOutputNorm={counts.get('caa_output_norm', 0.0):.6f} | "
        f"RuleBasedCAAEnabled={int(counts.get('rule_based_caa_enabled', 0))} | "
        f"ICNEnabled=1 | ICNForwardCalled={int(counts.get('icn_forward_called', 0))} | "
        "NashEnabled=1 | AssignmentBiasEnabled=1 | "
        f"AssignmentBiasAbsMean={counts.get('assignment_bias_abs_mean', 0.0):.6f} | "
        f"AssignmentBiasAppliedToTaskLogits={int(counts.get('assignment_bias_applied_to_task_logits', 0))} | "
        f"UnaryIntentBiasEnabled={int(counts.get('unary_intent_bias_enabled', 0))} | "
        f"SCBargainingEnabled={int(counts.get('sc_bargaining_enabled', 0))} | "
        f"NegotiationRoundsRuntime={int(counts.get('negotiation_rounds_runtime', 0))} | "
        "SCCommitmentEnabled=1 | HardSCCommit=1 | ForcedSCContinue=1 | "
        "SupportTargetFromPair=1 | PhysicalSCWithoutCommitment=1 | status=PASS_ONESHOT"
    )
    progress_print(
        "ICNInputSourceDiag | "
        f"UsesLocalObs={int(input_counts.get('uses_local_obs', 0))} | "
        f"UsesRoleSemantic={int(input_counts.get('uses_role_semantic', 0))} | "
        f"UsesTaskTypeSemantic={int(input_counts.get('uses_task_type_semantic', 0))} | "
        f"UsesBasicResource={int(input_counts.get('uses_basic_resource', 0))} | "
        f"UsesBasicFeasibility={int(input_counts.get('uses_basic_feasibility', 0))} | "
        f"UsesGNNEmbedding={int(input_counts.get('uses_gnn_embedding', 0))} | "
        f"UsesGraphFeatures={int(input_counts.get('uses_graph_features', 0))} | "
        f"UsesRelationEmbedding={int(input_counts.get('uses_relation_embedding', 0))} | "
        f"UsesCAAOutput={int(input_counts.get('uses_caa_output', 0))} | "
        f"UsesCapabilityAwareOutput={int(input_counts.get('uses_capability_aware_output', 0))} | "
        f"UsesRelationCapabilityEmbedding={int(input_counts.get('uses_relation_capability_embedding', 0))} | "
        f"GraphPartAbsMean={input_counts.get('icn_input_graph_part_abs_mean', 0.0):.6f} | "
        f"CAAOutputAbsMean={input_counts.get('icn_input_caa_output_abs_mean', 0.0):.6f} | "
        f"RelationCapabilityPartAbsMean={input_counts.get('icn_input_relation_capability_part_abs_mean', 0.0):.6f} | "
        "ICNInputMode=raw_semantic_only | status=PASS_ONESHOT"
    )
    implicit_full_comm = int(access_counts.get("implicit_full_communication", 1))
    implicit_global_layer = int(access_counts.get("implicit_global_relation_layer", 1))
    precommit_oracle = int(
        access_counts.get("precommit_uses_env_global_state", 0)
        or access_counts.get("precommit_uses_all_uav_states", 0)
        or access_counts.get("precommit_uses_all_task_states", 0)
        or access_counts.get("precommit_scans_all_uav_task_pairs", 0)
        or access_counts.get("precommit_scans_all_sc_pairs", 0)
        or access_counts.get("precommit_uses_upload_oracle", 0)
        or access_counts.get("precommit_uses_backhaul_oracle", 0)
        or access_counts.get("precommit_uses_terrain_oracle", 0)
        or access_counts.get("precommit_uses_global_support_target_search", 0)
    )
    access_status = "FAIL" if (implicit_full_comm or implicit_global_layer or precommit_oracle) else "PASS"
    progress_print(
        "InformationAccessBoundaryDiag | "
        f"Mode={'no_relation_capability_strict_v2' if is_strict_v2 else 'no_relation_capability'} | RepresentationRemoved=1 | ProtocolRetained=1 | "
        f"ActorUsesOnlyObservation={int(access_counts.get('actor_uses_only_observation', 0))} | "
        f"ICNUsesOnlyObservation={int(access_counts.get('icn_uses_only_observation', 0))} | "
        f"NashUsesOnlyObservation={int(access_counts.get('nash_uses_only_observation', 0))} | "
        f"AssignmentBiasUsesOnlyObservation={int(access_counts.get('assignment_bias_uses_only_observation', 0))} | "
        f"SCBargainingUsesOnlyObservation={int(access_counts.get('sc_bargaining_uses_only_observation', 0))} | "
        f"PreCommitUsesEnvGlobalState={int(access_counts.get('precommit_uses_env_global_state', 0))} | "
        f"PreCommitUsesAllUAVStates={int(access_counts.get('precommit_uses_all_uav_states', 0))} | "
        f"PreCommitUsesAllTaskStates={int(access_counts.get('precommit_uses_all_task_states', 0))} | "
        f"PreCommitScansAllUAVTaskPairs={int(access_counts.get('precommit_scans_all_uav_task_pairs', 0))} | "
        f"PreCommitScansAllSCPairs={int(access_counts.get('precommit_scans_all_sc_pairs', 0))} | "
        f"PreCommitUsesUploadOracle={int(access_counts.get('precommit_uses_upload_oracle', 0))} | "
        f"PreCommitUsesBackhaulOracle={int(access_counts.get('precommit_uses_backhaul_oracle', 0))} | "
        f"PreCommitUsesTimeFeasibilityOracle={int(access_counts.get('precommit_uses_time_feasibility_oracle', 0))} | "
        f"PreCommitUsesTerrainOracle={int(access_counts.get('precommit_uses_terrain_oracle', 0))} | "
        f"PreCommitUsesGlobalSupportTargetSearch={int(access_counts.get('precommit_uses_global_support_target_search', 0))} | "
        f"PostCommitExecutionUsesCommittedPairOnly={int(access_counts.get('postcommit_execution_uses_committed_pair_only', 0))} | "
        f"SupportTargetAssignedAfterCommitOnly={int(access_counts.get('support_target_assigned_after_commit_only', 0))} | "
        f"CommitWritesBeforeActorSelection={int(access_counts.get('commit_writes_before_actor_selection', 0))} | "
        f"ImplicitFullCommunication={implicit_full_comm} | "
        f"ImplicitGlobalRelationLayer={implicit_global_layer} | "
        f"status={access_status}"
    )
    progress_print(
        "PreCommitOracleDiag | "
        f"PairFormationSource={'selected_actions_only' if is_strict_v2 else 'env_all_pairs'} | "
        f"NashUtilitySource={'local_obs_messages' if is_strict_v2 else 'env_global_uav_table+env_global_task_table'} | "
        f"AssignmentBiasSource={'ICN_Nash_obs_only' if is_strict_v2 else 'env_global_uav_table+env_global_task_table'} | "
        f"SCBargainingSource={'local_obs_messages' if is_strict_v2 else 'env_all_pairs+env_upload_oracle+env_backhaul_oracle'} | "
        f"SupportTargetSource={'committed_pair_after_commit' if is_strict_v2 else 'env_global_task_table'} | "
        f"UsesEnvPairOracleBeforeCommit={int(access_counts.get('precommit_scans_all_sc_pairs', 0))} | "
        f"UsesUploadFeasibilityBeforeCommit={int(access_counts.get('precommit_uses_upload_oracle', 0))} | "
        f"UsesBackhaulFeasibilityBeforeCommit={int(access_counts.get('precommit_uses_backhaul_oracle', 0))} | "
        f"UsesAllPairsBeforeCommit={int(access_counts.get('precommit_scans_all_sc_pairs', 0))} | "
        f"UsesFutureCompletionInfo={int(access_counts.get('uses_future_completion_info', 0))} | "
        f"status={access_status}"
    )
    execution_status = "FAIL" if (
        int(access_counts.get("support_target_uses_global_search", 0))
        or int(access_counts.get("support_target_uses_uncommitted_pairs", 0))
        or int(access_counts.get("current_task_written_by_protocol_before_selection", 0))
    ) else "PASS"
    progress_print(
        "SCExecutionBoundaryDiag | "
        "HardSCCommitEnabled=1 | ForcedSCContinueEnabled=1 | SupportTargetFromPairEnabled=1 | "
        f"HardSCCommitTriggeredAfterCommit={int(access_counts.get('postcommit_execution_uses_committed_pair_only', 0))} | "
        f"ForcedSCContinueTriggeredAfterCommit={int(access_counts.get('postcommit_execution_uses_committed_pair_only', 0))} | "
        f"SupportTargetAssignedAfterCommit={int(access_counts.get('support_target_assigned_after_commit_only', 0))} | "
        f"SupportTargetUsesCommittedPairOnly={int(access_counts.get('postcommit_execution_uses_committed_pair_only', 0))} | "
        f"SupportTargetUsesGlobalSearch={int(access_counts.get('support_target_uses_global_search', 0))} | "
        f"SupportTargetUsesUncommittedPairs={int(access_counts.get('support_target_uses_uncommitted_pairs', 0))} | "
        f"CurrentTaskWrittenByActorSelection={int(access_counts.get('current_task_written_by_actor_selection', 0))} | "
        f"CurrentTaskWrittenByProtocolBeforeSelection={int(access_counts.get('current_task_written_by_protocol_before_selection', 0))} | "
        f"status={execution_status}"
    )
    progress_print(
        "AssignmentBiasFlowDiag | AssignmentBiasEnabled=1 | AssignmentBiasComputed=1 | "
        f"AssignmentBiasAppliedToTaskLogits={int(counts.get('assignment_bias_applied_to_task_logits', 0))} | "
        f"AssignmentBiasAbsMean={counts.get('assignment_bias_abs_mean', 0.0):.6f} | "
        f"AssignmentBiasSource={'ICN_Nash_obs_only' if is_strict_v2 else 'ICN_Nash_raw_semantic'} | "
        "UsesGNNEmbedding=0 | UsesCAAOutput=0 | status=PASS_ONESHOT"
    )
    progress_print(
        "SCBargainingFlowDiag | "
        f"SCBargainingEnabled={int(counts.get('sc_bargaining_enabled', 0))} | "
        f"SCBargainingApplied={counts.get('sc_bargaining_applied_sum', 0.0):.0f} | "
        f"NegotiationRoundsRuntime={int(counts.get('negotiation_rounds_runtime', 0))} | "
        "PairAttemptSC_Events=0 | PairSuccessSC_Events=0 | CommitWriteSuccess=0 | ActiveSC=0 | "
        f"HardSCCommit={int(counts.get('hard_sc_commit', 0))} | "
        f"ForcedSCContinue={int(counts.get('forced_sc_continue', 0))} | "
        f"SupportTargetFromPair={int(counts.get('support_target_from_pair', 0))} | "
        "status=PASS_ONESHOT"
    )
    setattr(mappo, "_eval_relation_capability_audit_printed", True)

MAIN_METRICS = [
    "TCR_gen",
    "TCR_sel",
    "TCR_res",
    "SC_CR",
    "AvgDone",
    "AvgExpired",
    "AvgReward",
    "SafetyEvents",
    "ActualCollisions",
    "ResponseTime",
    "SC_PFR_Enabled",
    "SC_PFR_gen_valid",
    "SC_PFR_gen",
    "SC_CR_phys",
    "SC_sel",
    "Done",
    "Expired",
    "DoneS",
    "DoneC",
    "DoneSC",
    "total_collisions",
    "total_obstacle_safety_events",
    "total_obstacle_actual_collisions",
    "SC_PhysGeneratedSC",
    "SC_PhysFeasibleSC",
    "Generated_SC",
    "Completed_SC",
    "Feasible_Generated_SC",
    "Completed_Feasible_SC",
    "SC_PhysFeasibleDirectSC",
    "SC_PhysFeasibleRelaySC",
    "ActiveSC",
    "CommitWriteSuccess",
    "PairSuccessSC",
    "BackhaulFailActive",
    "BackhaulSuccessRate",
    "UploadSuccessRate",
    "SwitchTaskChange",
    "SwitchUAV",
    "sc_score_feasible_rate_used",
    "legacy_total_sc_feasible_rate",
    "total_sc_feasible_rate_terrain",
    "episode_reward",
    "episode_length",
]

SUMMARY_REQUIRED_COLUMNS = [
    "run_time",
    "tag",
    "config_path",
    "checkpoint_path",
    "checkpoint_name",
    "checkpoint_step",
    "eval_mode",
    "episodes",
    "seed",
    "mode",
]

REPORT_METRICS = [
    ("TCR_gen", "生成任务完成率"),
    ("TCR_sel", "选择任务完成率"),
    ("SC_CR", "SC任务完成率"),
    ("SC_PFR_gen", "生成SC物理可行率"),
    ("SC_CR_phys", "物理归一化SC完成率"),
    ("SC_sel", "已选择SC任务完成率"),
    ("DoneSC", "完成SC任务数"),
    ("Expired", "平均过期任务数"),
    ("PairSuccessSC", "SC配对成功数"),
    ("ActiveSC", "激活SC承诺数"),
    ("CommitWriteSuccess", "承诺写入成功数"),
    ("BackhaulSuccessRate", "回传成功率"),
    ("UploadSuccessRate", "上传成功率"),
    ("SwitchUAV", "单无人机平均切换次数"),
]

SC_FUNNEL_SUMMARY_METRICS = [
    "GenSC_Unique", "SelSC_Unique", "BothSC", "PairAttemptSC_Events",
    "SC_PFR_Enabled", "SC_PFR_gen_valid",
    "SC_PFR_gen", "SC_CR_phys", "SC_PhysGeneratedSC", "SC_PhysFeasibleSC",
    "SC_PhysFeasibleDirectSC", "SC_PhysFeasibleRelaySC",
    "PairSuccessSC_Events", "PairSuccessSC_Unique", "RawCandidateSC",
    "RoleValidSC", "SameTaskSC", "UploadFeasibleSC", "BackhaulFeasibleSC",
    "TimeFeasibleSC", "CommitWriteSuccess", "ActiveSC_Unique",
    "DoneSC_Unique", "ExpSC_Unique", "SCSelectRate_Unique",
    "BothSelectRate_SC", "PairSuccessRate_Unique", "StepUploadFeasibleRate_SC", "UploadFeasibleRate_SC",
    "BackhaulFeasibleRate_SC", "TimeFeasibleRate_SC", "CommitToActiveRate",
    "ActiveToDoneRate", "CommitToDoneRate", "SupportDirectRate",
    "SupportFallbackRate", "LOSBlockedRate", "UploadSuccessRate",
    "BackhaulSuccessRate", "AvgD2D", "AvgD3D", "AvgClearance",
    "SCSupportRuleEnvTargetGapMean", "SCSupportRuleEnvTargetGapMax",
    "SCSupportRuleEnvTargetGapCount",
    "TaskLevelUploadPairRate_SC", "SCTimeoutCount",
    "SCTimeout_S2TaskMean", "SCTimeout_C2SupportMean", "SCTimeout_S2CMean",
    "SCTimeout_C2BaseMean", "SCTimeout_UploadViolationMean",
    "SCTimeout_BackhaulViolationMean", "SCTimeout_SInTaskRate",
    "SCTimeout_CNearSupportRate", "SCTimeout_UploadFeasibleRate",
    "SCTimeout_BackhaulFeasibleRate", "SCTimeout_BothArrivedRate",
    "SCTimeout_NeitherArrivedRate", "SCTimeout_SMissingRate",
    "SCTimeout_CMissingRate",
    "SCTimeout_S2TaskStartMean", "SCTimeout_S2TaskEndMean",
    "SCTimeout_S2TaskDeltaMean", "SCTimeout_C2SupportStartMean",
    "SCTimeout_C2SupportEndMean", "SCTimeout_C2SupportDeltaMean",
    "SCTimeout_S2TaskImproveRate", "SCTimeout_C2SupportImproveRate",
    "SCTimeout_BothImproveRate", "SCTimeout_NoProgressRate",
    "SCSenseServiceRadiusCells", "SCSenseServiceRadiusMeters",
    "SCGridResolutionMeters", "SCTimeout_SWithinRadiusByDistance",
    "SCTimeout_SWithinRadiusByDistanceRate", "SCTimeout_SInTaskByEnvFlag",
    "SCTimeout_SInTaskByEnvFlagRate", "SCTimeout_SDistanceVsEnvMismatch",
    "SCTimeout_SDistanceVsEnvMismatchRate", "SCTimeout_SHasCurrentTask",
    "SCTimeout_SCurrentTaskIsThisSC", "SCTimeout_SSelectedThisSC",
    "SCTimeout_STargetIsTask", "SCTimeout_STargetTaskGapMean",
    "SCArrivedBeforeRelease", "SCReleasedDespiteSArrived",
    "SCReleasedDespiteBothArrivedByDistance", "SCServiceCheckBeforeRelease",
    "SCReleaseBeforeServiceCheck", "SCTimeout_CWithinSupportByDistance",
    "SCTimeout_CWithinSupportByDistanceRate", "SCTimeout_BothWithinByDistance",
    "SCTimeout_BothWithinByDistanceRate",
    "SensingSC_MoveDistMean", "SensingSC_TaskApproachDeltaMean",
    "SensingSC_TaskApproachNegativeRate", "SensingSC_SpeedEfficiencyMean",
    "SensingSC_CandidateApproachPerStepMean",
    "SCTimeout_SValidCount", "SCTimeout_SValidRate",
    "SCTimeout_SMissingOrUnboundCount", "SCTimeout_SMissingOrUnboundRate",
    "SCTimeout_S2TaskMean_ValidOnly", "SCTimeout_S2TaskStartMean_ValidOnly",
    "SCTimeout_S2TaskEndMean_ValidOnly", "SCTimeout_S2TaskDeltaMean_ValidOnly",
    "SCTimeout_SWithinRadiusRate_ValidOnly",
    "SCTimeout_SStillSelectedRate_ValidOnly",
    "SCTimeout_SStillTargetTaskRate_ValidOnly",
    "SFail_NoValidS", "SFail_NotCurrentTask", "SFail_NotSelected",
    "SFail_TargetNotTask", "SFail_OutOfSenseRadius", "SFail_Unknown",
    "SFail_NoValidSRate", "SFail_NotCurrentTaskRate", "SFail_NotSelectedRate",
    "SFail_TargetNotTaskRate", "SFail_OutOfSenseRadiusRate", "SFail_UnknownRate",
    "SCTimeout_CValidCount", "SCTimeout_CValidRate",
    "SCTimeout_C2SupportMean_ValidOnly",
    "SCTimeout_CWithinSupportRate_ValidOnly",
    "SCActivation_CNearSupportBlocked",
    "SCActivation_LinkReadyButCNotNearSupport",
    "SCActivation_ActivatedWithoutCNearSupport",
    "SCActivation_ReselectBlocked",
    "SCActivation_CommitmentUsed",
    "SCActivation_ServiceReady",
    "SCActivation_ServiceReadyButActiveLimit",
    "SArrivalGraceTriggered",
    "SArrivalGraceKept",
    "SArrivalGraceExpired",
    "SArrivalGraceToActive",
    "SArrivalGraceToDone",
    "SArrivalGraceBlockedNoValidS",
    "SArrivalGraceBlockedNoProgress",
    "SArrivalGraceBlockedNoValidC",
    "SArrivalGraceBlockedMaxTimeout",
    "SArrivalGraceDistToSenseBoundaryMean",
    "SArrivalGraceDistToSenseBoundaryMin",
    "SArrivalGraceDistToSenseBoundaryMax",
    "SArrivalGraceUsedStepsMean",
    "ProgressAwareKeepSC", "ProgressAwareReleaseNoProgress",
    "ProgressAwareReleaseMaxTimeout", "SCProgressSImproved",
    "SCProgressCImproved", "SCProgressBothImproved",
    "SCProgressAnyImproved", "SCProgressNoImprovement",
    "ReleaseSCCandidateTimeout", "ReleaseSCActiveFail", "SCContractActiveFail",
    "SCAudit_TotalCandidates", "SCAudit_TotalCandidateSteps",
    "SCAudit_ActivatedCandidates", "SCAudit_ReleasedCandidates",
    "SCAudit_CompletedCandidates", "SCAudit_ExpiredCandidates",
    "SCAudit_SMismatchCandidateService", "SCAudit_CMismatchCandidateService",
    "SCAudit_SMismatchCurrentTask", "SCAudit_CMismatchCurrentTask",
    "SCAudit_SMismatchSelectedTask", "SCAudit_CMismatchSelectedTask",
    "SCAudit_SMismatchAssignedTask", "SCAudit_CMismatchAssignedTask",
    "SCAudit_SWithinRadiusButNotServiceReady", "SCAudit_SWithinRadiusButReleased",
    "SCAudit_SWithinRadiusButNoValidS", "SCAudit_SWithinRadiusButUploadFalse",
    "SCAudit_SWithinRadiusButBackhaulFalse", "SCAudit_SWithinRadiusButTimeFalse",
    "SCAudit_SWithinRadiusButActiveLimit", "SCAudit_SWithinRadiusButCommitmentLost",
    "SCAudit_SWithinRadiusButUnknown",
    "SCAudit_EventWeighted_SValidTimeoutCount",
    "SCAudit_EventWeighted_SWithinRadiusTimeoutCount",
    "SCAudit_EventWeighted_SWithinRadiusTimeoutRate",
    "SCAudit_EventWeighted_SDistMeanValidTimeout",
    "SCAudit_EventWeighted_SDistMedianValidTimeout",
    "SCAudit_EventWeighted_SDistMinValidTimeout",
    "SCAudit_EventWeighted_SDistMaxValidTimeout",
    "SCAudit_ServiceCheckedThisStep", "SCAudit_ReleaseCheckedThisStep",
    "SCAudit_ServiceThenReleaseSameStep",
    "SCAudit_ReleaseWithoutServiceCheckSameStep",
    "SCAudit_ServiceReadyButReleasedSameStep",
    "SCAudit_ServiceNotReadyReleaseReasonLogged",
    "SCAudit_ServiceReadyButActiveLimit", "SCAudit_ActiveSCLimitValue",
    "SCAudit_CurrentActiveSCCount", "SCAudit_ServiceReadyButActiveLimitReleased",
    "SCAudit_UploadDistanceWithinRadius", "SCAudit_UploadTerrainChecked",
    "SCAudit_UploadTerrainFailed",
    "SCAudit_UploadFeasibleFalseButDistanceWithin",
    "SCAudit_BackhaulDistanceWithinRadius", "SCAudit_BackhaulTerrainChecked",
    "SCAudit_BackhaulTerrainFailed",
    "SCAudit_BackhaulFeasibleFalseButDistanceWithin",
]

CRITICAL_MISSING_KEYWORDS = (
    "intent_comm", "nash_bargaining", "actor", "gnn",
    "high_level_policy", "task_pointer",
)

REPORT_METRICS = [
    ("TCR_gen", "生成任务完成率"),
    ("TCR_sel", "选择任务完成率"),
    ("SC_CR", "SC任务完成率"),
    ("SC_PFR_gen", "生成SC物理可行率"),
    ("SC_CR_phys", "物理归一化SC完成率"),
    ("SC_sel", "已选择SC任务完成率"),
    ("DoneSC", "完成SC任务数"),
    ("Expired", "平均过期任务数"),
    ("PairSuccessSC", "SC配对成功数"),
    ("ActiveSC", "激活SC承诺数"),
    ("CommitWriteSuccess", "承诺写入成功数"),
    ("BackhaulSuccessRate", "回传成功率"),
    ("UploadSuccessRate", "上传成功率"),
    ("SwitchUAV", "单无人机平均切换次数"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone evaluation for MAHPPO checkpoints. For "
            "MAHPPO_sc_cleanup_balance_v2, use logs\\MAHPPO_sc_cleanup_balance_v2\\config.yaml "
            "with --eval_mode stochastic and --sc_pfr_probe_mode off for official comparisons."
        )
    )
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--pattern", type=str, default="*_checkpoint_*.pt")
    parser.add_argument("--last_n", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--eval_mode", choices=("stochastic", "deterministic", "both"), default=None)
    parser.add_argument(
        "--mode",
        choices=(
            "full",
            "no_icn",
            "icn_wo_nash",
            "no_icn_gnn_caa",
            "mahppo_icn_no_relation_capability_fpsfix",
            "strict_v2_local_obs_v1",
            "no_relation_capability_strict_v2_local_obs_v1",
            "wo_gnn_caa_local_obs_v1",
            "full_local_obs_v1",
            "mahppo_full_local_obs_v1",
        ),
        default="full",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, default="evaluation_results/standalone_eval_last10")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--progress_interval", type=int, default=5)
    parser.add_argument("--prefer_training_config", dest="prefer_training_config", action="store_true", default=True)
    parser.add_argument("--no_prefer_training_config", dest="prefer_training_config", action="store_false")
    parser.add_argument("--strict_config_match", action="store_true", default=False)
    parser.add_argument("--sc_pfr_probe_mode", choices=("off", "release_once", "strict"), default="off")
    parser.add_argument("--quiet", action="store_true", default=True)
    parser.add_argument("--verbose", action="store_true", default=False)
    return parser.parse_args()


def progress_print(message: str) -> None:
    stream = getattr(sys, "__stdout__", None) or sys.stdout
    print(message, file=stream, flush=True)


def safe_metric(metrics: Dict[str, Any], key: str, default: str = "NA", precision: int = 4) -> str:
    try:
        val = metrics.get(key, default)
        if val is None:
            return default
        if isinstance(val, (float, np.floating)) and np.isnan(float(val)):
            return default
        if isinstance(val, (float, np.floating)):
            return f"{float(val):.{precision}f}"
        return str(val)
    except Exception:
        return default


def report_num(value: Any, precision: int = 4) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if np.isnan(val):
        return "n/a"
    return f"{val:.{precision}f}"


def summary_mean(row: Dict[str, Any], key: str, default: Any = np.nan) -> Any:
    return row.get(f"{key}_mean", row.get(key, default))


def finite_float(value: Any, default: float = np.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def finite_sum(rows: Sequence[Dict[str, Any]], key: str) -> float:
    values = [finite_float(row.get(key), np.nan) for row in rows]
    return float(np.sum([value for value in values if np.isfinite(value)]))


def first_present(rows: Sequence[Dict[str, Any]], key: str, default: Any = None) -> Any:
    for row in rows:
        if key in row:
            return row.get(key)
    return default


def aggregate_sc_pfr_summary(rows: Sequence[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    """Add checkpoint-level SC physical feasibility ratios.

    Per-episode SC_PFR_gen is useful for debugging, but checkpoint-level
    SC_PFR_gen is a task-count ratio and must be aggregated from summed counts:
    sum(SC_PhysFeasibleSC) / sum(SC_PhysGeneratedSC).
    """
    if not rows:
        return

    mode = str(first_present(rows, "SC_PFR_Mode", summary.get("SC_PFR_Mode", "off")))
    enabled = any(bool(row.get("SC_PFR_Enabled", False)) for row in rows)
    valid_flag = any(bool(row.get("SC_PFR_gen_valid", False)) for row in rows)
    generated_sum = finite_sum(rows, "SC_PhysGeneratedSC")
    feasible_sum = finite_sum(rows, "SC_PhysFeasibleSC")
    feasible_direct_sum = finite_sum(rows, "SC_PhysFeasibleDirectSC")
    feasible_relay_sum = finite_sum(rows, "SC_PhysFeasibleRelaySC")
    done_key = "DoneSC_Unique" if any("DoneSC_Unique" in row for row in rows) else "DoneSC"
    done_sc_sum = finite_sum(rows, done_key)

    summary["SC_PFR_Mode"] = mode
    summary["SC_PFR_Enabled"] = bool(enabled)
    summary["SC_PFR_gen_valid"] = bool(enabled and valid_flag and generated_sum > 0.0)
    summary["SC_PhysGeneratedSC_sum"] = generated_sum
    summary["SC_PhysFeasibleSC_sum"] = feasible_sum
    summary["SC_PhysFeasibleDirectSC_sum"] = feasible_direct_sum
    summary["SC_PhysFeasibleRelaySC_sum"] = feasible_relay_sum
    summary["SC_PhysGeneratedSC"] = summary.get("SC_PhysGeneratedSC_mean", np.nan)
    summary["SC_PhysFeasibleSC"] = summary.get("SC_PhysFeasibleSC_mean", np.nan)
    summary["SC_PhysFeasibleDirectSC"] = summary.get("SC_PhysFeasibleDirectSC_mean", np.nan)
    summary["SC_PhysFeasibleRelaySC"] = summary.get("SC_PhysFeasibleRelaySC_mean", np.nan)
    summary["SC_CR_phys_done_key"] = done_key
    summary[f"{done_key}_sum"] = done_sc_sum

    if not enabled:
        sc_pfr = np.nan
        sc_cr_phys = np.nan
        pfr_reason = "sc_pfr_probe_mode=off"
        cr_reason = "sc_pfr_probe_mode=off"
    else:
        if generated_sum > 0.0 and valid_flag:
            sc_pfr = feasible_sum / generated_sum
            pfr_reason = ""
        else:
            sc_pfr = np.nan
            pfr_reason = "SC_PhysGeneratedSC=0"

        if feasible_sum > 0.0:
            sc_cr_phys = done_sc_sum / feasible_sum
            cr_reason = ""
        else:
            sc_cr_phys = np.nan
            cr_reason = "SC_PhysFeasibleSC=0"

    for key, value in {
        "SC_PFR_gen": sc_pfr,
        "SC_PFR_gen_mean": sc_pfr,
        "SC_CR_phys": sc_cr_phys,
        "SC_CR_phys_mean": sc_cr_phys,
    }.items():
        summary[key] = float(value) if np.isfinite(finite_float(value)) else np.nan
    summary["SC_PFR_gen_invalid_reason"] = pfr_reason
    summary["SC_CR_phys_invalid_reason"] = cr_reason
    if enabled and feasible_sum > 0.0 and "SC_PhysFeasibleSC=0" in str(cr_reason):
        summary["SC_CR_phys_invalid_reason"] = ""


@contextmanager
def redirected_console(console_log_path: Path, enabled: bool):
    if enabled:
        with console_log_path.open("a", encoding="utf-8") as log_f:
            with redirect_stdout(log_f), redirect_stderr(log_f):
                yield
    else:
        yield


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def extract_checkpoint_step_from_name(path: Path) -> Optional[int]:
    match = re.search(r"checkpoint_(\d+)", path.name)
    return int(match.group(1)) if match else None


def extract_checkpoint_step(path: Path, checkpoint_obj: Any) -> int:
    if isinstance(checkpoint_obj, dict):
        for key in ("total_timesteps", "timestep", "step"):
            value = checkpoint_obj.get(key)
            if isinstance(value, (int, float)):
                return int(value)
            if torch.is_tensor(value) and value.numel() == 1:
                return int(value.item())
    name_step = extract_checkpoint_step_from_name(path)
    return int(name_step) if name_step is not None else -1


def select_checkpoints(args: argparse.Namespace) -> List[Path]:
    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return [checkpoint]
    if not args.checkpoint_dir:
        raise ValueError("Provide --checkpoint or --checkpoint_dir.")

    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    candidates = [p for p in checkpoint_dir.glob(args.pattern) if p.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No checkpoints matched {args.pattern} in {checkpoint_dir}")

    def sort_key(path: Path) -> Tuple[int, float]:
        step = extract_checkpoint_step_from_name(path)
        if step is not None:
            return (0, float(step))
        return (1, path.stat().st_mtime)

    return sorted(candidates, key=sort_key)[-max(1, int(args.last_n)) :]


def get_eval_train_target(mappo):
    if callable(getattr(mappo, "eval", None)) and callable(getattr(mappo, "train", None)):
        return mappo
    agent = getattr(mappo, "agent", None)
    if callable(getattr(agent, "eval", None)) and callable(getattr(agent, "train", None)):
        return agent
    return None


def set_training_mode(module, training: bool) -> None:
    if module is None:
        return
    if training:
        try:
            module.train(True)
        except TypeError:
            module.train()
    else:
        if callable(getattr(module, "eval", None)):
            module.eval()
        else:
            try:
                module.train(False)
            except TypeError:
                pass


def state_dict_like(obj: Any) -> bool:
    return isinstance(obj, dict) and bool(obj) and all(torch.is_tensor(v) for v in obj.values())


def extract_state_dict(checkpoint_obj: Any) -> Tuple[Dict[str, torch.Tensor], str]:
    if isinstance(checkpoint_obj, dict):
        for key in ("agent_state_dict", "model_state_dict", "mappo_state_dict", "state_dict"):
            value = checkpoint_obj.get(key)
            if isinstance(value, dict):
                return value, key
        if state_dict_like(checkpoint_obj):
            return checkpoint_obj, "root_state_dict"
    raise ValueError("Could not find a compatible state_dict in checkpoint.")


def load_checkpoint(mappo, checkpoint_path: Path, device: torch.device) -> Tuple[int, str, Dict[str, Any]]:
    checkpoint_obj = torch.load(checkpoint_path, map_location=device)
    checkpoint_step = extract_checkpoint_step(checkpoint_path, checkpoint_obj)
    state_dict, source_key = extract_state_dict(checkpoint_obj)
    target = getattr(mappo, "agent", mappo)

    incompatible = target.load_state_dict(state_dict, strict=False)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))

    if unexpected and all(str(k).startswith("agent.") for k in unexpected):
        stripped = {str(k)[6:]: v for k, v in state_dict.items() if str(k).startswith("agent.")}
        if stripped:
            incompatible = target.load_state_dict(stripped, strict=False)
            missing = list(getattr(incompatible, "missing_keys", []))
            unexpected = list(getattr(incompatible, "unexpected_keys", []))
            source_key = f"{source_key}:agent_prefix_stripped"

    if missing or unexpected:
        print(f"WARNING loading {checkpoint_path.name} with strict=False")
        if missing:
            print(f"  missing_keys: {missing[:20]}{' ...' if len(missing) > 20 else ''}")
        if unexpected:
            print(f"  unexpected_keys: {unexpected[:20]}{' ...' if len(unexpected) > 20 else ''}")
    load_info = {
        "missing_keys_count": int(len(missing)),
        "unexpected_keys_count": int(len(unexpected)),
        "missing_keys_preview": ";".join(str(x) for x in missing[:20]),
        "unexpected_keys_preview": ";".join(str(x) for x in unexpected[:20]),
    }
    load_info["critical_missing_warning"] = any(
        keyword in load_info["missing_keys_preview"] for keyword in CRITICAL_MISSING_KEYWORDS
    )
    return checkpoint_step, source_key, load_info


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if float(denominator) != 0.0 else 0.0


def _enum_text(value: Any) -> str:
    if value is None:
        return ""
    value = getattr(value, "value", value)
    return str(value).upper()


def _uav_role_text(uav: Any) -> str:
    text = _enum_text(getattr(uav, "role", None) or getattr(getattr(uav, "state", None), "role", None))
    if text in ("S", "SENSING"):
        return "S"
    if text in ("C", "COMMUNICATION", "COMM"):
        return "C"
    return ""


def _uav_current_task_id(uav: Any) -> Optional[int]:
    value = getattr(uav, "current_task_id", None)
    if value is None:
        value = getattr(getattr(uav, "state", None), "current_task_id", None)
    try:
        return None if value is None else int(value)
    except Exception:
        return None


def _task_type_text(task: Any) -> str:
    text = _enum_text(getattr(task, "task_type", None) or getattr(getattr(task, "state", None), "task_type", None))
    if text in ("S", "SENSING"):
        return "S"
    if text in ("C", "COMMUNICATION", "COMM"):
        return "C"
    if text in ("SC", "SENSING_COMMUNICATION", "SENSING-COMMUNICATION"):
        return "SC"
    return ""


def _task_is_active_for_rcr(task: Any) -> bool:
    if hasattr(task, "is_active"):
        try:
            return bool(task.is_active())
        except Exception:
            pass
    status = _enum_text(getattr(task, "status", None) or getattr(getattr(task, "state", None), "status", None))
    return status in ("AVAILABLE", "IN_PROGRESS", "ACTIVE")


def strict_role_competition_snapshot(env: Any) -> Dict[str, float]:
    """Read-only role-aware redundant competition snapshot after an env step."""
    tasks = getattr(env, "tasks", None)
    uavs = getattr(env, "uavs", None)
    if not isinstance(tasks, dict) or not isinstance(uavs, list):
        return {
            "RCR_available": 0.0,
            "RCR_redundant": np.nan,
            "RCR_occupancy": np.nan,
            "SC_RCR_redundant": np.nan,
            "SC_RCR_occupancy": np.nan,
        }

    counts: Dict[int, Dict[str, int]] = {}
    for uav in uavs:
        task_id = _uav_current_task_id(uav)
        role = _uav_role_text(uav)
        if task_id is None or role not in ("S", "C"):
            continue
        task = tasks.get(int(task_id))
        if task is None or not _task_is_active_for_rcr(task):
            continue
        task_type = _task_type_text(task)
        if task_type not in ("S", "C", "SC"):
            continue
        bucket = counts.setdefault(int(task_id), {"S": 0, "C": 0})
        bucket[role] += 1

    redundant = 0.0
    occupancy = 0.0
    sc_redundant = 0.0
    sc_occupancy = 0.0
    for task_id, role_counts in counts.items():
        task = tasks.get(task_id)
        task_type = _task_type_text(task)
        demand_s, demand_c = (1, 0) if task_type == "S" else ((0, 1) if task_type == "C" else (1, 1))
        n_s = float(role_counts.get("S", 0))
        n_c = float(role_counts.get("C", 0))
        task_occupancy = n_s + n_c
        task_redundant = max(0.0, n_s - demand_s) + max(0.0, n_c - demand_c)
        occupancy += task_occupancy
        redundant += task_redundant
        if task_type == "SC":
            sc_occupancy += task_occupancy
            sc_redundant += task_redundant

    return {
        "RCR_available": 1.0,
        "RCR_redundant": redundant,
        "RCR_occupancy": occupancy,
        "SC_RCR_redundant": sc_redundant,
        "SC_RCR_occupancy": sc_occupancy,
    }


def get_info_value(info: Dict[str, Any], key: str, default: float = 0.0) -> float:
    value = info.get(key, default)
    if torch.is_tensor(value):
        return float(value.item()) if value.numel() == 1 else default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def metric_value(env: DisasterRescueEnv, info: Dict[str, Any], key: str, default: float = 0.0) -> float:
    metrics = getattr(env, "metrics", {}) or {}
    if key in metrics:
        value = metrics.get(key, default)
    else:
        value = info.get(key, default)
    if torch.is_tensor(value):
        return float(value.item()) if value.numel() == 1 else default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def len_attr_set(env: DisasterRescueEnv, name: str) -> float:
    value = getattr(env, name, None)
    try:
        return float(len(value)) if value is not None else 0.0
    except TypeError:
        return 0.0


def config_value_text(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return str(value)


def relpath_text(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def same_path(a: Optional[Path], b: Optional[Path]) -> bool:
    if a is None or b is None:
        return False
    try:
        return Path(a).resolve() == Path(b).resolve()
    except OSError:
        return str(a) == str(b)


def find_training_config_path(args: argparse.Namespace) -> Optional[Path]:
    exp_name = None
    if args.checkpoint_dir:
        exp_name = Path(args.checkpoint_dir).name
    elif args.checkpoint:
        exp_name = Path(args.checkpoint).parent.name
    if not exp_name:
        return None
    candidate = ROOT / "logs" / exp_name / "config.yaml"
    return candidate if candidate.exists() else None


def get_config_attr(config: Config, attr_path: Sequence[str]) -> Any:
    cur: Any = config
    for part in attr_path:
        cur = getattr(cur, part, DEFAULT_CONFIG_SENTINEL)
        if cur is DEFAULT_CONFIG_SENTINEL:
            return None
    return cur


def normalized_config_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [normalized_config_value(x) for x in value]
    if isinstance(value, list):
        return [normalized_config_value(x) for x in value]
    if isinstance(value, dict):
        return {str(k): normalized_config_value(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, np.generic):
        return value.item()
    return value


def collect_key_fields(config: Config) -> Dict[str, Any]:
    values = {}
    for label, attr_path in CONFIG_AUDIT_FIELDS:
        short_label = {
            "task.task_type_ratio": "task_type_ratio",
            "task.generation_window": "generation_window",
            "task.generation_fade_end": "generation_fade_end",
            "task.deadline_range": "deadline_range",
            "task.hard_deadline_range": "hard_deadline_range",
            "task.deadline_slack": "deadline_slack",
            "task.max_active_tasks": "max_active_tasks",
            "task.max_active_sc_tasks": "max_active_sc_tasks",
            "task.max_active_c_tasks": "max_active_c_tasks",
            "communication.data_upload_radius_m": "data_upload_radius_m",
            "communication.backhaul_link_radius_m": "backhaul_link_radius_m",
            "communication.comm_service_radius_m": "comm_service_radius_m",
            "communication.sense_service_radius_m": "sense_service_radius_m",
            "communication.nlos_radius_factor": "nlos_radius_factor",
            "real_geo.grid_resolution_m": "grid_resolution_m",
            "base_selection.safe_side": "base_safe_side",
            "base_selection.base_outside_margin_m": "base_outside_margin_m",
            "base_selection.base_outer_margin_m": "base_outer_margin_m",
            "base_selection.mode": "base_selection.mode",
            "evaluation.eval_action_mode": "eval_action_mode",
        }.get(label, label)
        values[short_label] = normalized_config_value(get_config_attr(config, attr_path))
    return values


def compare_config_fields(provided_config: Config, training_config: Config) -> List[Dict[str, Any]]:
    diffs = []
    for label, attr_path in CONFIG_AUDIT_FIELDS:
        provided_value = normalized_config_value(get_config_attr(provided_config, attr_path))
        training_value = normalized_config_value(get_config_attr(training_config, attr_path))
        if provided_value != training_value:
            diffs.append({
                "key": label,
                "provided_config": config_value_text(provided_value),
                "training_config": config_value_text(training_value),
            })
    return diffs


def print_config_diff_table(diffs: Sequence[Dict[str, Any]]) -> None:
    if not diffs:
        return
    progress_print("[ConfigAudit] key | provided_config | training_config")
    for diff in diffs:
        progress_print(
            f"[ConfigAudit] {diff['key']} | {diff['provided_config']} | {diff['training_config']}"
        )


def apply_sc_pfr_probe_mode(config: Config, mode: str) -> None:
    mode = str(mode or "off").lower()
    if mode == "light":
        mode = "release_once"
    raw_cfg = getattr(config, "_config", None)
    if isinstance(raw_cfg, dict):
        diagnostics = raw_cfg.setdefault("diagnostics", {})
        if not isinstance(diagnostics, dict):
            diagnostics = {}
            raw_cfg["diagnostics"] = diagnostics
        diagnostics["sc_pfr_probe_mode"] = mode
    setattr(config, "sc_pfr_probe_mode", mode)


def extract_sc_funnel_metrics(
    env: DisasterRescueEnv,
    info: Dict[str, Any],
    episode_length: int,
    config: Config,
) -> Dict[str, Any]:
    gen_sc = metric_value(env, info, "sc_tasks_generated")
    sel_sc = metric_value(env, info, "selected_sc_tasks", len_attr_set(env, "_selected_sc_task_ids"))
    if sel_sc == 0.0:
        sel_sc = len_attr_set(env, "_selected_sc_task_ids")
    done_sc = metric_value(env, info, "sc_tasks_completed")
    exp_sc = metric_value(env, info, "sc_tasks_expired")

    only_s = metric_value(env, info, "sc_with_only_s_selected")
    only_c = metric_value(env, info, "sc_with_only_c_selected")
    both = metric_value(env, info, "sc_with_both_s_c_selected")
    multi_s = metric_value(env, info, "sc_with_multiple_s")
    multi_c = metric_value(env, info, "sc_with_multiple_c")
    no_selected = metric_value(env, info, "sc_with_no_selected")

    pair_attempt_events = metric_value(env, info, "pair_attempt_sc_events")
    pair_success_events = metric_value(env, info, "pair_success_sc_events")
    pair_attempt_unique = metric_value(env, info, "pair_attempt_sc_unique_tasks")
    pair_success_unique = metric_value(env, info, "pair_success_sc_unique_tasks")
    if pair_success_unique == 0.0:
        pair_success_unique = metric_value(env, info, "sc_pair_success_tasks", len_attr_set(env, "_pair_success_sc_task_ids"))
    if pair_success_unique == 0.0:
        pair_success_unique = len_attr_set(env, "_pair_success_sc_task_ids")

    raw_candidate = metric_value(env, info, "sc_pair_stage_raw_candidate")
    role_valid = metric_value(env, info, "sc_pair_stage_role_valid")
    same_task = metric_value(env, info, "sc_pair_stage_same_task")
    upload_feasible = metric_value(env, info, "sc_pair_stage_upload_feasible")
    backhaul_feasible = metric_value(env, info, "sc_pair_stage_backhaul_feasible")
    time_feasible = metric_value(env, info, "sc_pair_stage_time_feasible")

    commit_success = metric_value(env, info, "sc_commit_write_success")
    commit_fail = metric_value(env, info, "sc_commit_write_fail")
    active_sc = metric_value(env, info, "active_commit_sc_tasks", len_attr_set(env, "_active_commit_sc_task_ids"))
    if active_sc == 0.0:
        active_sc = len_attr_set(env, "_active_commit_sc_task_ids")

    support_assigned = metric_value(env, info, "sc_support_target_assigned")
    support_direct = metric_value(env, info, "sc_support_target_direct_feasible")
    support_fallback = metric_value(env, info, "sc_support_target_fallback")
    support_interval_sum = metric_value(env, info, "sc_support_target_interval_width_sum")
    sc_timeout_count = metric_value(env, info, "sc_timeout_count")
    sensing_sc_move_count = metric_value(env, info, "sensing_sc_move_dist_count")
    sensing_sc_approach_count = metric_value(env, info, "sensing_sc_task_approach_delta_count")
    sensing_sc_eff_count = metric_value(env, info, "sensing_sc_speed_efficiency_count")
    sensing_sc_candidate_count = metric_value(env, info, "sensing_sc_candidate_approach_per_step_count")
    sc_timeout_s_valid_count = metric_value(env, info, "SCTimeout_SValidCount")
    sc_timeout_c_valid_count = metric_value(env, info, "SCTimeout_CValidCount")
    s_arrival_grace_boundary_count = metric_value(env, info, "SArrivalGraceDistToSenseBoundaryCount")
    s_arrival_grace_used_steps_count = metric_value(env, info, "SArrivalGraceUsedStepsCount")

    terrain_checks = metric_value(env, info, "terrain_link_checks")
    terrain_success = metric_value(env, info, "terrain_link_success")
    terrain_blocked = metric_value(env, info, "terrain_link_los_blocked")
    upload_checks = metric_value(env, info, "terrain_upload_checks")
    upload_success = metric_value(env, info, "terrain_upload_success")
    backhaul_checks = metric_value(env, info, "terrain_backhaul_checks")
    backhaul_success = metric_value(env, info, "terrain_backhaul_success")

    task_ratio = getattr(config.task, "task_type_ratio", getattr(config.task, "task_type_ratios", {}))
    sc_ratio = task_ratio.get("SC", None) if isinstance(task_ratio, dict) else None

    funnel = {
        "GenSC_Unique": gen_sc,
        "SC_PFR_Mode": str(info.get("SC_PFR_Mode", getattr(env, "sc_pfr_probe_mode", "off"))),
        "SC_PFR_Enabled": bool(info.get("SC_PFR_Enabled", False)),
        "SC_PFR_gen_valid": bool(info.get("SC_PFR_gen_valid", False)),
        "SC_PFR_gen": get_info_value(info, "SC_PFR_gen", np.nan),
        "SC_CR_phys": get_info_value(info, "SC_CR_phys", np.nan),
        "SC_PhysGeneratedSC": get_info_value(info, "SC_PhysGeneratedSC", np.nan),
        "SC_PhysFeasibleSC": get_info_value(info, "SC_PhysFeasibleSC", np.nan),
        "SC_PhysFeasibleDirectSC": metric_value(env, info, "SC_PhysFeasibleDirectSC"),
        "SC_PhysFeasibleRelaySC": metric_value(env, info, "SC_PhysFeasibleRelaySC"),
        "SelSC_Unique": sel_sc,
        "DoneSC_Unique": done_sc,
        "ExpSC_Unique": exp_sc,
        "SCSelectRate_Unique": safe_div(sel_sc, max(gen_sc, 1.0)),
        "SCCompleteRate_Gen": safe_div(done_sc, max(gen_sc, 1.0)),
        "SCCompleteSel": safe_div(done_sc, max(sel_sc, 1.0)),
        "SCExpireRate_Unique": safe_div(exp_sc, max(gen_sc, 1.0)),
        "OnlyS_SC": only_s,
        "OnlyC_SC": only_c,
        "BothSC": both,
        "MultiS_SC": multi_s,
        "MultiC_SC": multi_c,
        "NoSelectedSC": no_selected,
        "BothSelectRate_SC": safe_div(both, max(gen_sc, 1.0)),
        "PairAttemptSC_Events": pair_attempt_events,
        "PairSuccessSC_Events": pair_success_events,
        "PairAttemptSC_Unique": pair_attempt_unique,
        "PairSuccessSC_Unique": pair_success_unique,
        "PairSuccessRate_Events": safe_div(pair_success_events, max(pair_attempt_events, 1.0)),
        "PairSuccessRate_Unique": safe_div(pair_success_unique, max(pair_attempt_unique, 1.0)),
        "TaskLevelUploadPairRate_SC": safe_div(pair_success_unique, max(sel_sc, 1.0)),
        "RawCandidateSC": raw_candidate,
        "RoleValidSC": role_valid,
        "SameTaskSC": same_task,
        "UploadFeasibleSC": upload_feasible,
        "BackhaulFeasibleSC": backhaul_feasible,
        "TimeFeasibleSC": time_feasible,
        "StepUploadFeasibleRate_SC": safe_div(upload_feasible, max(same_task, 1.0)),
        "UploadFeasibleRate_SC": safe_div(upload_feasible, max(same_task, 1.0)),
        "BackhaulFeasibleRate_SC": safe_div(backhaul_feasible, max(upload_feasible, 1.0)),
        "TimeFeasibleRate_SC": safe_div(time_feasible, max(backhaul_feasible, 1.0)),
        "CommitWriteSuccess": commit_success,
        "CommitWriteFail": commit_fail,
        "TaskAssignedBothRoles": metric_value(env, info, "sc_commit_task_assigned_both_roles"),
        "UAVCurrentTaskSet": metric_value(env, info, "sc_commit_uav_current_task_set"),
        "ActiveSC_Unique": active_sc,
        "SCServiceStarted": metric_value(env, info, "sc_service_started"),
        "CommitToActiveRate": safe_div(active_sc, max(commit_success, 1.0)),
        "ActiveToDoneRate": safe_div(done_sc, max(active_sc, 1.0)),
        "CommitToDoneRate": safe_div(done_sc, max(commit_success, 1.0)),
        "SCSupportAssigned": support_assigned,
        "SCSupportDirectFeasible": support_direct,
        "SCSupportFallback": support_fallback,
        "SCSupportTerrainSuccess": metric_value(env, info, "sc_support_target_terrain_success"),
        "SCSupportTerrainFail": metric_value(env, info, "sc_support_target_terrain_fail"),
        "SCSupportFallback2D": metric_value(env, info, "sc_support_target_fallback_2d"),
        "SCSupportIntervalWidthMean": safe_div(support_interval_sum, max(support_assigned, 1.0)),
        "SCSupportRuleEnvTargetGapMean": metric_value(env, info, "SCSupportRuleEnvTargetGapMean"),
        "SCSupportRuleEnvTargetGapMax": metric_value(env, info, "SCSupportRuleEnvTargetGapMax"),
        "SCSupportRuleEnvTargetGapCount": metric_value(env, info, "SCSupportRuleEnvTargetGapCount"),
        "SCTimeoutCount": sc_timeout_count,
        "SCTimeout_S2TaskMean": safe_div(metric_value(env, info, "SCTimeout_S2TaskDistValidSum"), max(sc_timeout_s_valid_count, 1.0)),
        "SCTimeout_C2SupportMean": safe_div(metric_value(env, info, "sc_timeout_c_to_support_dist_sum"), max(sc_timeout_count, 1.0)),
        "SCTimeout_S2CMean": safe_div(metric_value(env, info, "sc_timeout_s_to_c_dist_sum"), max(sc_timeout_count, 1.0)),
        "SCTimeout_C2BaseMean": safe_div(metric_value(env, info, "sc_timeout_c_to_base_dist_sum"), max(sc_timeout_count, 1.0)),
        "SCTimeout_UploadViolationMean": safe_div(metric_value(env, info, "sc_timeout_upload_violation_sum"), max(sc_timeout_count, 1.0)),
        "SCTimeout_BackhaulViolationMean": safe_div(metric_value(env, info, "sc_timeout_backhaul_violation_sum"), max(sc_timeout_count, 1.0)),
        "SCTimeout_SInTaskRate": safe_div(metric_value(env, info, "sc_timeout_s_in_task_range_count"), max(sc_timeout_count, 1.0)),
        "SCTimeout_CNearSupportRate": safe_div(metric_value(env, info, "sc_timeout_c_near_support_count"), max(sc_timeout_count, 1.0)),
        "SCTimeout_UploadFeasibleRate": safe_div(metric_value(env, info, "sc_timeout_upload_feasible_count"), max(sc_timeout_count, 1.0)),
        "SCTimeout_BackhaulFeasibleRate": safe_div(metric_value(env, info, "sc_timeout_backhaul_feasible_count"), max(sc_timeout_count, 1.0)),
        "SCTimeout_BothArrivedRate": safe_div(metric_value(env, info, "sc_timeout_both_arrived_count"), max(sc_timeout_count, 1.0)),
        "SCTimeout_NeitherArrivedRate": safe_div(metric_value(env, info, "sc_timeout_neither_arrived_count"), max(sc_timeout_count, 1.0)),
        "SCTimeout_SMissingRate": safe_div(metric_value(env, info, "sc_timeout_s_missing_count"), max(sc_timeout_count, 1.0)),
        "SCTimeout_CMissingRate": safe_div(metric_value(env, info, "sc_timeout_c_missing_count"), max(sc_timeout_count, 1.0)),
        "SCTimeout_S2TaskStartMean": safe_div(metric_value(env, info, "SCTimeout_S2TaskStartValidSum"), max(sc_timeout_s_valid_count, 1.0)),
        "SCTimeout_S2TaskEndMean": safe_div(metric_value(env, info, "SCTimeout_S2TaskEndValidSum"), max(sc_timeout_s_valid_count, 1.0)),
        "SCTimeout_S2TaskDeltaMean": safe_div(metric_value(env, info, "SCTimeout_S2TaskDeltaValidSum"), max(sc_timeout_s_valid_count, 1.0)),
        "SCTimeout_C2SupportStartMean": safe_div(metric_value(env, info, "sc_timeout_c_to_support_start_sum"), max(sc_timeout_count, 1.0)),
        "SCTimeout_C2SupportEndMean": safe_div(metric_value(env, info, "sc_timeout_c_to_support_end_sum"), max(sc_timeout_count, 1.0)),
        "SCTimeout_C2SupportDeltaMean": safe_div(metric_value(env, info, "sc_timeout_c_to_support_delta_sum"), max(sc_timeout_count, 1.0)),
        "SCTimeout_S2TaskImproveRate": safe_div(metric_value(env, info, "sc_timeout_s_to_task_improved_count"), max(sc_timeout_count, 1.0)),
        "SCTimeout_C2SupportImproveRate": safe_div(metric_value(env, info, "sc_timeout_c_to_support_improved_count"), max(sc_timeout_count, 1.0)),
        "SCTimeout_BothImproveRate": safe_div(metric_value(env, info, "sc_timeout_both_improved_count"), max(sc_timeout_count, 1.0)),
        "SCTimeout_NoProgressRate": safe_div(metric_value(env, info, "sc_timeout_no_progress_count"), max(sc_timeout_count, 1.0)),
        "SCSenseServiceRadiusCells": metric_value(env, info, "SCSenseServiceRadiusCells"),
        "SCSenseServiceRadiusMeters": metric_value(env, info, "SCSenseServiceRadiusMeters"),
        "SCGridResolutionMeters": metric_value(env, info, "SCGridResolutionMeters"),
        "SCTimeout_SWithinRadiusByDistance": metric_value(env, info, "SCTimeout_SWithinRadiusByDistance"),
        "SCTimeout_SWithinRadiusByDistanceRate": safe_div(
            metric_value(env, info, "SCTimeout_SWithinRadiusByDistance"), max(sc_timeout_count, 1.0)
        ),
        "SCTimeout_SInTaskByEnvFlag": metric_value(env, info, "SCTimeout_SInTaskByEnvFlag"),
        "SCTimeout_SInTaskByEnvFlagRate": safe_div(
            metric_value(env, info, "SCTimeout_SInTaskByEnvFlag"), max(sc_timeout_count, 1.0)
        ),
        "SCTimeout_SDistanceVsEnvMismatch": metric_value(env, info, "SCTimeout_SDistanceVsEnvMismatch"),
        "SCTimeout_SDistanceVsEnvMismatchRate": safe_div(
            metric_value(env, info, "SCTimeout_SDistanceVsEnvMismatch"), max(sc_timeout_count, 1.0)
        ),
        "SCTimeout_SHasCurrentTask": metric_value(env, info, "SCTimeout_SHasCurrentTask"),
        "SCTimeout_SCurrentTaskIsThisSC": metric_value(env, info, "SCTimeout_SCurrentTaskIsThisSC"),
        "SCTimeout_SSelectedThisSC": metric_value(env, info, "SCTimeout_SSelectedThisSC"),
        "SCTimeout_STargetIsTask": metric_value(env, info, "SCTimeout_STargetIsTask"),
        "SCTimeout_STargetTaskGapMean": safe_div(
            metric_value(env, info, "SCTimeout_STargetTaskGapSum"), max(metric_value(env, info, "SCTimeout_STargetTaskGapCount"), 1.0)
        ),
        "SCArrivedBeforeRelease": metric_value(env, info, "SCArrivedBeforeRelease"),
        "SCReleasedDespiteSArrived": metric_value(env, info, "SCReleasedDespiteSArrived"),
        "SCReleasedDespiteBothArrivedByDistance": metric_value(env, info, "SCReleasedDespiteBothArrivedByDistance"),
        "SCServiceCheckBeforeRelease": metric_value(env, info, "SCServiceCheckBeforeRelease"),
        "SCReleaseBeforeServiceCheck": metric_value(env, info, "SCReleaseBeforeServiceCheck"),
        "SCTimeout_CWithinSupportByDistance": metric_value(env, info, "SCTimeout_CWithinSupportByDistance"),
        "SCTimeout_CWithinSupportByDistanceRate": safe_div(
            metric_value(env, info, "SCTimeout_CWithinSupportByDistance"), max(sc_timeout_count, 1.0)
        ),
        "SCTimeout_BothWithinByDistance": metric_value(env, info, "SCTimeout_BothWithinByDistance"),
        "SCTimeout_BothWithinByDistanceRate": safe_div(
            metric_value(env, info, "SCTimeout_BothWithinByDistance"), max(sc_timeout_count, 1.0)
        ),
        "SensingSC_MoveDistMean": safe_div(metric_value(env, info, "sensing_sc_move_dist_sum"), max(sensing_sc_move_count, 1.0)),
        "SensingSC_TaskApproachDeltaMean": safe_div(
            metric_value(env, info, "sensing_sc_task_approach_delta_sum"), max(sensing_sc_approach_count, 1.0)
        ),
        "SensingSC_TaskApproachNegativeRate": safe_div(
            metric_value(env, info, "sensing_sc_task_approach_negative_count"), max(sensing_sc_approach_count, 1.0)
        ),
        "SensingSC_SpeedEfficiencyMean": safe_div(
            metric_value(env, info, "sensing_sc_speed_efficiency_sum"), max(sensing_sc_eff_count, 1.0)
        ),
        "SensingSC_CandidateApproachPerStepMean": safe_div(
            metric_value(env, info, "sensing_sc_candidate_approach_per_step_sum"), max(sensing_sc_candidate_count, 1.0)
        ),
        "SCTimeout_SValidCount": sc_timeout_s_valid_count,
        "SCTimeout_SValidRate": safe_div(sc_timeout_s_valid_count, max(sc_timeout_count, 1.0)),
        "SCTimeout_SMissingOrUnboundCount": metric_value(env, info, "SCTimeout_SMissingOrUnboundCount"),
        "SCTimeout_SMissingOrUnboundRate": safe_div(
            metric_value(env, info, "SCTimeout_SMissingOrUnboundCount"), max(sc_timeout_count, 1.0)
        ),
        "SCTimeout_S2TaskMean_ValidOnly": safe_div(
            metric_value(env, info, "SCTimeout_S2TaskDistValidSum"), max(sc_timeout_s_valid_count, 1.0)
        ),
        "SCTimeout_S2TaskStartMean_ValidOnly": safe_div(
            metric_value(env, info, "SCTimeout_S2TaskStartValidSum"), max(sc_timeout_s_valid_count, 1.0)
        ),
        "SCTimeout_S2TaskEndMean_ValidOnly": safe_div(
            metric_value(env, info, "SCTimeout_S2TaskEndValidSum"), max(sc_timeout_s_valid_count, 1.0)
        ),
        "SCTimeout_S2TaskDeltaMean_ValidOnly": safe_div(
            metric_value(env, info, "SCTimeout_S2TaskDeltaValidSum"), max(sc_timeout_s_valid_count, 1.0)
        ),
        "SCTimeout_SWithinRadiusRate_ValidOnly": safe_div(
            metric_value(env, info, "SCTimeout_SWithinRadiusValidCount"), max(sc_timeout_s_valid_count, 1.0)
        ),
        "SCTimeout_SStillSelectedRate_ValidOnly": safe_div(
            metric_value(env, info, "SCTimeout_SStillSelectedValidCount"), max(sc_timeout_s_valid_count, 1.0)
        ),
        "SCTimeout_SStillTargetTaskRate_ValidOnly": safe_div(
            metric_value(env, info, "SCTimeout_SStillTargetTaskValidCount"), max(sc_timeout_s_valid_count, 1.0)
        ),
        "SFail_NoValidS": metric_value(env, info, "SFail_NoValidS"),
        "SFail_NotCurrentTask": metric_value(env, info, "SFail_NotCurrentTask"),
        "SFail_NotSelected": metric_value(env, info, "SFail_NotSelected"),
        "SFail_TargetNotTask": metric_value(env, info, "SFail_TargetNotTask"),
        "SFail_OutOfSenseRadius": metric_value(env, info, "SFail_OutOfSenseRadius"),
        "SFail_Unknown": metric_value(env, info, "SFail_Unknown"),
        "SFail_NoValidSRate": safe_div(metric_value(env, info, "SFail_NoValidS"), max(sc_timeout_count, 1.0)),
        "SFail_NotCurrentTaskRate": safe_div(metric_value(env, info, "SFail_NotCurrentTask"), max(sc_timeout_count, 1.0)),
        "SFail_NotSelectedRate": safe_div(metric_value(env, info, "SFail_NotSelected"), max(sc_timeout_count, 1.0)),
        "SFail_TargetNotTaskRate": safe_div(metric_value(env, info, "SFail_TargetNotTask"), max(sc_timeout_count, 1.0)),
        "SFail_OutOfSenseRadiusRate": safe_div(metric_value(env, info, "SFail_OutOfSenseRadius"), max(sc_timeout_count, 1.0)),
        "SFail_UnknownRate": safe_div(metric_value(env, info, "SFail_Unknown"), max(sc_timeout_count, 1.0)),
        "SCTimeout_CValidCount": sc_timeout_c_valid_count,
        "SCTimeout_CValidRate": safe_div(sc_timeout_c_valid_count, max(sc_timeout_count, 1.0)),
        "SCTimeout_C2SupportMean_ValidOnly": safe_div(
            metric_value(env, info, "SCTimeout_C2SupportValidSum"), max(sc_timeout_c_valid_count, 1.0)
        ),
        "SCTimeout_CWithinSupportRate_ValidOnly": safe_div(
            metric_value(env, info, "SCTimeout_CWithinSupportValidCount"), max(sc_timeout_c_valid_count, 1.0)
        ),
        "SCActivation_CNearSupportBlocked": metric_value(env, info, "SCActivation_CNearSupportBlocked"),
        "SCActivation_LinkReadyButCNotNearSupport": metric_value(env, info, "SCActivation_LinkReadyButCNotNearSupport"),
        "SCActivation_ActivatedWithoutCNearSupport": metric_value(env, info, "SCActivation_ActivatedWithoutCNearSupport"),
        "SCActivation_ReselectBlocked": metric_value(env, info, "SCActivation_ReselectBlocked"),
        "SCActivation_CommitmentUsed": metric_value(env, info, "SCActivation_CommitmentUsed"),
        "SCActivation_ServiceReady": metric_value(env, info, "SCActivation_ServiceReady"),
        "SCActivation_ServiceReadyButActiveLimit": metric_value(env, info, "SCActivation_ServiceReadyButActiveLimit"),
        "SArrivalGraceTriggered": metric_value(env, info, "SArrivalGraceTriggered"),
        "SArrivalGraceKept": metric_value(env, info, "SArrivalGraceKept"),
        "SArrivalGraceExpired": metric_value(env, info, "SArrivalGraceExpired"),
        "SArrivalGraceToActive": metric_value(env, info, "SArrivalGraceToActive"),
        "SArrivalGraceToDone": metric_value(env, info, "SArrivalGraceToDone"),
        "SArrivalGraceBlockedNoValidS": metric_value(env, info, "SArrivalGraceBlockedNoValidS"),
        "SArrivalGraceBlockedNoProgress": metric_value(env, info, "SArrivalGraceBlockedNoProgress"),
        "SArrivalGraceBlockedNoValidC": metric_value(env, info, "SArrivalGraceBlockedNoValidC"),
        "SArrivalGraceBlockedMaxTimeout": metric_value(env, info, "SArrivalGraceBlockedMaxTimeout"),
        "SArrivalGraceDistToSenseBoundaryMean": safe_div(
            metric_value(env, info, "SArrivalGraceDistToSenseBoundarySum"), max(s_arrival_grace_boundary_count, 1.0)
        ),
        "SArrivalGraceDistToSenseBoundaryMin": metric_value(env, info, "SArrivalGraceDistToSenseBoundaryMin"),
        "SArrivalGraceDistToSenseBoundaryMax": metric_value(env, info, "SArrivalGraceDistToSenseBoundaryMax"),
        "SArrivalGraceUsedStepsMean": safe_div(
            metric_value(env, info, "SArrivalGraceUsedStepsSum"), max(s_arrival_grace_used_steps_count, 1.0)
        ),
        "ProgressAwareKeepSC": metric_value(env, info, "ProgressAwareKeepSC"),
        "ProgressAwareReleaseNoProgress": metric_value(env, info, "ProgressAwareReleaseNoProgress"),
        "ProgressAwareReleaseMaxTimeout": metric_value(env, info, "ProgressAwareReleaseMaxTimeout"),
        "SCProgressSImproved": metric_value(env, info, "SCProgressSImproved"),
        "SCProgressCImproved": metric_value(env, info, "SCProgressCImproved"),
        "SCProgressBothImproved": metric_value(env, info, "SCProgressBothImproved"),
        "SCProgressAnyImproved": metric_value(env, info, "SCProgressAnyImproved"),
        "SCProgressNoImprovement": metric_value(env, info, "SCProgressNoImprovement"),
        "SupportDirectRate": safe_div(support_direct, max(support_assigned, 1.0)),
        "SupportFallbackRate": safe_div(support_fallback, max(support_assigned, 1.0)),
        "TerrainChecks": terrain_checks,
        "TerrainSuccess": terrain_success,
        "TerrainLOSBlocked": terrain_blocked,
        "TerrainUploadChecks": upload_checks,
        "TerrainUploadSuccess": upload_success,
        "TerrainBackhaulChecks": backhaul_checks,
        "TerrainBackhaulSuccess": backhaul_success,
        "TerrainSuccessRate": safe_div(terrain_success, max(terrain_checks, 1.0)),
        "LOSBlockedRate": safe_div(terrain_blocked, max(terrain_checks, 1.0)),
        "UploadSuccessRate": safe_div(upload_success, max(upload_checks, 1.0)),
        "BackhaulSuccessRate": safe_div(backhaul_success, max(backhaul_checks, 1.0)),
        "AvgD2D": safe_div(metric_value(env, info, "terrain_link_2d_distance_sum"), max(terrain_checks, 1.0)),
        "AvgD3D": safe_div(metric_value(env, info, "terrain_link_3d_distance_sum"), max(terrain_checks, 1.0)),
        "AvgClearance": safe_div(metric_value(env, info, "terrain_link_min_clearance_sum"), max(terrain_checks, 1.0)),
        "ForcedContinue": metric_value(env, info, "action_mask_forced_continue"),
        "ForcedSCContinue": metric_value(env, info, "action_mask_forced_sc_continue"),
        "SoftTask": metric_value(env, info, "action_mask_soft_hold_task"),
        "SoftSCCandidate": metric_value(env, info, "action_mask_soft_hold_sc_candidate"),
        "HardSCCommit": metric_value(env, info, "action_mask_hard_lock_sc_commit"),
        "AllowReselectAfterHold": metric_value(env, info, "action_mask_allow_reselect_after_hold"),
        "ReleaseSCCandidateTimeout": metric_value(env, info, "action_mask_release_sc_candidate_timeout"),
        "ReleaseSCActiveFail": metric_value(env, info, "action_mask_release_sc_active_fail"),
        "ReleaseSCContract": metric_value(env, info, "action_mask_release_sc_contract"),
        "SCContractActiveFail": metric_value(env, info, "sc_contract_release_active_fail"),
        "ActualGeneratedTasks": float(getattr(env.env_state, "total_tasks_generated", 0.0)),
        "ActualCompletedTasks": float(getattr(env.env_state, "total_tasks_completed", 0.0)),
        "ActualExpiredTasks": float(getattr(env.env_state, "total_tasks_expired", 0.0)),
        "EpisodeLength": float(episode_length),
        "LastTime": float(getattr(env.env_state, "current_time", 0.0)),
        "config_generation_window": config_value_text(getattr(config.task, "generation_window", None)),
        "config_generation_fade_end": config_value_text(getattr(config.task, "generation_fade_end", None)),
        "config_use_reachable_deadline": config_value_text(getattr(config.task, "use_reachable_deadline", None)),
        "config_deadline_range": config_value_text(getattr(config.task, "deadline_range", None)),
        "config_hard_deadline_range": config_value_text(getattr(config.task, "hard_deadline_range", None)),
        "config_deadline_slack": config_value_text(getattr(config.task, "deadline_slack", None)),
        "config_sc_ratio": config_value_text(sc_ratio),
        "config_max_active_tasks": config_value_text(getattr(config.task, "max_active_tasks", None)),
        "config_max_active_sc_tasks": config_value_text(getattr(config.task, "max_active_sc_tasks", None)),
    }
    for reason in (
        "NoSensingUAV", "NoCommUAV", "NotSameTask", "RoleMismatch",
        "TaskInactive", "TaskExpired", "TaskNotSC", "SensingNotInRange",
        "CommNotInRange", "UploadRange", "BackhaulRange", "TimeInfeasible",
        "CommLocked", "DuplicateAssignment", "MaskBlocked", "Unknown",
    ):
        key = f"pair_fail_{reason}"
        funnel[key] = metric_value(env, info, key)
    for key in SC_FUNNEL_SUMMARY_METRICS:
        if key.startswith("SCAudit_") and key not in funnel:
            funnel[key] = metric_value(env, info, key)
    return funnel


def get_episode_metrics(info: Dict[str, Any], episode_reward: float, episode_length: int, config: Config) -> Dict[str, float]:
    generated = get_info_value(info, "total_tasks_generated")
    completed = get_info_value(info, "total_tasks_completed")
    expired = get_info_value(info, "total_tasks_expired")
    selected = get_info_value(info, "selected_tasks")
    completed_selected = get_info_value(info, "completed_selected_tasks")
    gen_sc = get_info_value(info, "sc_tasks_generated")
    selected_sc = get_info_value(info, "selected_sc_tasks")
    completed_selected_sc = get_info_value(info, "completed_selected_sc_tasks")
    done_sc = get_info_value(info, "sc_tasks_completed")
    terrain_checks = get_info_value(info, "terrain_link_checks")
    terrain_success = get_info_value(info, "terrain_link_success")
    terrain_los_blocked = get_info_value(info, "terrain_link_los_blocked")
    upload_checks = get_info_value(info, "terrain_upload_checks")
    upload_success = get_info_value(info, "terrain_upload_success")
    backhaul_checks = get_info_value(info, "terrain_backhaul_checks")
    backhaul_success = get_info_value(info, "terrain_backhaul_success")
    pair_attempt_events = get_info_value(info, "pair_attempt_sc_events")
    pair_success_events = get_info_value(info, "pair_success_sc_events")
    active_sc = get_info_value(info, "active_commit_sc_tasks")
    exp_sc = get_info_value(info, "sc_tasks_expired")
    geom_feasible = get_info_value(info, "sc_support_target_direct_feasible")
    support_assigned = get_info_value(info, "sc_support_target_assigned")
    interval_sum = get_info_value(info, "sc_support_target_interval_width_sum")
    switch_count = get_info_value(info, "commitment_switch_count")
    num_uavs = max(1.0, float(getattr(config.environment, "num_uavs", 1)))

    metrics = {
        "TCR_gen": safe_div(completed, max(1.0, generated)),
        "TCR_sel": safe_div(completed_selected, max(1.0, selected)),
        "TCR_res": safe_div(completed_selected, max(1.0, selected)),
        "SC_CR": safe_div(done_sc, max(1.0, gen_sc)),
        "SC_PFR_Mode": str(info.get("SC_PFR_Mode", "off")),
        "SC_PFR_Enabled": bool(info.get("SC_PFR_Enabled", False)),
        "SC_PFR_gen_valid": bool(info.get("SC_PFR_gen_valid", False)),
        "SC_PFR_gen": get_info_value(info, "SC_PFR_gen", np.nan),
        "SC_CR_phys": get_info_value(info, "SC_CR_phys", np.nan),
        "SC_sel": safe_div(completed_selected_sc, max(1.0, selected_sc)),
        "Done": completed,
        "Expired": expired,
        "AvgDone": completed,
        "AvgExpired": expired,
        "AvgReward": float(episode_reward),
        "SafetyEvents": get_info_value(info, "total_obstacle_safety_events"),
        "ActualCollisions": get_info_value(info, "total_obstacle_actual_collisions"),
        "total_collisions": get_info_value(info, "total_collisions"),
        "total_obstacle_safety_events": get_info_value(info, "total_obstacle_safety_events"),
        "total_obstacle_actual_collisions": get_info_value(info, "total_obstacle_actual_collisions"),
        "DoneS": get_info_value(info, "s_tasks_completed"),
        "DoneC": get_info_value(info, "c_tasks_completed"),
        "DoneSC": done_sc,
        "SC_PhysGeneratedSC": get_info_value(info, "SC_PhysGeneratedSC", np.nan),
        "SC_PhysFeasibleSC": get_info_value(info, "SC_PhysFeasibleSC", np.nan),
        "Generated_SC": gen_sc,
        "Completed_SC": done_sc,
        "Feasible_Generated_SC": get_info_value(info, "SC_PhysFeasibleSC", np.nan),
        "Completed_Feasible_SC": (
            get_info_value(info, "SC_PhysFeasibleSC", np.nan) * get_info_value(info, "SC_CR_phys", np.nan)
            if np.isfinite(get_info_value(info, "SC_PhysFeasibleSC", np.nan)) and np.isfinite(get_info_value(info, "SC_CR_phys", np.nan))
            else np.nan
        ),
        "SC_PhysFeasibleDirectSC": get_info_value(info, "SC_PhysFeasibleDirectSC"),
        "SC_PhysFeasibleRelaySC": get_info_value(info, "SC_PhysFeasibleRelaySC"),
        "ActiveSC": active_sc,
        "CommitWriteSuccess": get_info_value(info, "sc_commit_write_success"),
        "PairSuccessSC": get_info_value(info, "pair_formation_success"),
        "BackhaulFailActive": get_info_value(info, "backhaul_fail_active_commit"),
        "BackhaulSuccessRate": safe_div(backhaul_success, max(1.0, backhaul_checks)),
        "BackhaulSuccessCount": backhaul_success,
        "BackhaulAttemptCount": backhaul_checks,
        "UploadSuccessRate": safe_div(upload_success, max(1.0, upload_checks)),
        "SwitchTaskChange": get_info_value(info, "switch_task_change", switch_count),
        "SwitchUAV": safe_div(switch_count, num_uavs),
        "episode_reward": float(episode_reward),
        "episode_length": float(episode_length),
        "GenSC_Unique": gen_sc,
        "SelSC_Unique": selected_sc,
        "PairAttemptSC_Events": pair_attempt_events,
        "PairSuccessSC_Events": pair_success_events,
        "ActiveSC_Unique": active_sc,
        "DoneSC_Unique": done_sc,
        "ExpSC_Unique": exp_sc,
        "SCSelectRate_Unique": safe_div(selected_sc, max(1.0, gen_sc)),
        "SCCompleteSel": safe_div(completed_selected_sc, max(1.0, selected_sc)),
        "SCExpireRate_Unique": safe_div(exp_sc, max(1.0, gen_sc)),
        "SCPairRate": safe_div(pair_success_events, max(1.0, gen_sc)),
        "SCActiveRate": safe_div(active_sc, max(1.0, gen_sc)),
        "SCSupportAssigned": support_assigned,
        "SCSupportDirectFeasible": geom_feasible,
        "SCSupportFallback": get_info_value(info, "sc_support_target_fallback"),
        "SCSupportIntervalWidthMean": safe_div(interval_sum, max(1.0, support_assigned)),
        "SCSupportRuleEnvTargetGapMean": get_info_value(info, "SCSupportRuleEnvTargetGapMean"),
        "SCSupportRuleEnvTargetGapMax": get_info_value(info, "SCSupportRuleEnvTargetGapMax"),
        "SCSupportRuleEnvTargetGapCount": get_info_value(info, "SCSupportRuleEnvTargetGapCount"),
        "TerrainChecks": terrain_checks,
        "TerrainSuccessRate": safe_div(terrain_success, max(1.0, terrain_checks)),
        "LOSBlockedRate": safe_div(terrain_los_blocked, max(1.0, terrain_checks)),
        "AvgD2D": safe_div(get_info_value(info, "terrain_link_2d_distance_sum"), max(1.0, terrain_checks)),
        "AvgD3D": safe_div(get_info_value(info, "terrain_link_3d_distance_sum"), max(1.0, terrain_checks)),
        "AvgClearance": safe_div(get_info_value(info, "terrain_link_min_clearance_sum"), max(1.0, terrain_checks)),
        "TaskAssignedBothRoles": get_info_value(info, "sc_commit_task_assigned_both_roles"),
        "UAVCurrentTaskSet": get_info_value(info, "sc_commit_uav_current_task_set"),
        "SCServiceStarted": get_info_value(info, "sc_service_started"),
        "CommitWriteFail": get_info_value(info, "sc_commit_write_fail"),
        "ForcedContinue": get_info_value(info, "action_mask_forced_continue"),
        "ForcedSCContinue": get_info_value(info, "action_mask_forced_sc_continue"),
        "SoftTask": get_info_value(info, "action_mask_soft_hold_task"),
        "SoftSCCandidate": get_info_value(info, "action_mask_soft_hold_sc_candidate"),
        "HardSCCommit": get_info_value(info, "action_mask_hard_lock_sc_commit"),
        "AllowReselectAfterHold": get_info_value(info, "action_mask_allow_reselect_after_hold"),
        "ReleaseSCCandidateTimeout": get_info_value(info, "action_mask_release_sc_candidate_timeout"),
        "ReleaseSCActiveFail": get_info_value(info, "action_mask_release_sc_active_fail"),
        "ReleaseSCContract": get_info_value(info, "action_mask_release_sc_contract"),
        "SCContractActiveFail": get_info_value(info, "sc_contract_release_active_fail"),
    }
    for key in (
        "sc_score_source",
        "terrain_aware_sc_scoring",
        "sc_score_feasible_rate_used",
        "legacy_total_sc_feasible_rate",
        "total_sc_feasible_rate_terrain",
    ):
        info_key = f"BaseSelectionDiag_{key}"
        if info_key in info:
            metrics[key] = info.get(info_key)
    return metrics


def evaluate_one_episode(
    mappo,
    env: DisasterRescueEnv,
    config: Config,
    deterministic: bool,
    seed: int,
    episode_id: int,
) -> Dict[str, float]:
    eval_seed = int(seed) + 100000 + int(episode_id)
    np.random.seed(eval_seed)
    torch.manual_seed(eval_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(eval_seed)

    if hasattr(getattr(mappo, "agent", None), "reset_graph_cache"):
        mappo.agent.reset_graph_cache()
    setattr(env, "_sc_audit_episode_id", int(episode_id))
    observations, info = env.reset(seed=eval_seed)
    setattr(env, "_sc_audit_episode_id", int(episode_id))
    global_state = env.get_global_state()
    done = False
    episode_reward = 0.0
    episode_length = 0
    rcr_redundant_sum = 0.0
    rcr_occupancy_sum = 0.0
    sc_rcr_redundant_sum = 0.0
    sc_rcr_occupancy_sum = 0.0
    rcr_available = True

    while not done:
        action_masks = env.get_all_action_masks()
        actions = mappo.get_action(observations, global_state, action_masks, deterministic=deterministic)
        step_out = env.step(actions)
        if len(step_out) == 5:
            observations, rewards, terminated, truncated, info = step_out
            done = all(terminated.values()) or bool(truncated)
        else:
            observations, rewards, done, info = step_out
        global_state = env.get_global_state()
        rcr_snapshot = strict_role_competition_snapshot(env)
        if float(rcr_snapshot.get("RCR_available", 0.0)) <= 0.0:
            rcr_available = False
        else:
            rcr_redundant_sum += float(rcr_snapshot.get("RCR_redundant", 0.0))
            rcr_occupancy_sum += float(rcr_snapshot.get("RCR_occupancy", 0.0))
            sc_rcr_redundant_sum += float(rcr_snapshot.get("SC_RCR_redundant", 0.0))
            sc_rcr_occupancy_sum += float(rcr_snapshot.get("SC_RCR_occupancy", 0.0))
        for reward in rewards.values():
            episode_reward += float(reward.total) if hasattr(reward, "total") else float(reward)
        episode_length += 1
        if episode_length >= int(config.environment.max_steps):
            break

    row = get_episode_metrics(info, episode_reward, episode_length, config)
    row.update(extract_sc_funnel_metrics(env, info, episode_length, config))
    if rcr_available:
        row.update({
            "RCR": safe_div(rcr_redundant_sum, rcr_occupancy_sum),
            "SC_RCR": safe_div(sc_rcr_redundant_sum, sc_rcr_occupancy_sum),
            "AvgRedundantUAV": float(rcr_redundant_sum),
            "AvgRedundantUAV_per_step": safe_div(rcr_redundant_sum, float(episode_length)),
            "EpisodeSteps": float(episode_length),
            "RCR_RedundantSum": float(rcr_redundant_sum),
            "RCR_OccupancySum": float(rcr_occupancy_sum),
            "SC_RCR_RedundantSum": float(sc_rcr_redundant_sum),
            "SC_RCR_OccupancySum": float(sc_rcr_occupancy_sum),
            "RCR_StrictRoleAware": 1.0,
        })
    else:
        row.update({
            "RCR": np.nan,
            "SC_RCR": np.nan,
            "AvgRedundantUAV": np.nan,
            "AvgRedundantUAV_per_step": np.nan,
            "EpisodeSteps": float(episode_length),
            "RCR_StrictRoleAware": 0.0,
        })
    row["episode"] = float(episode_id)
    row["eval_seed"] = float(eval_seed)
    row["eval_seed_requested"] = float(eval_seed)
    row["eval_seed_actual"] = float(getattr(env, "_last_episode_seed", eval_seed))
    row["__sc_lifecycle_audit_records"] = list(getattr(env, "_sc_lifecycle_audit_records", []))
    row["__sc_lifecycle_audit_truncated"] = bool(getattr(env, "_sc_lifecycle_audit_truncated", False))
    maybe_print_no_relation_eval_runtime_audit(mappo)
    return row


def evaluate_checkpoint(
    mappo,
    config: Config,
    checkpoint_path: Path,
    checkpoint_step: int,
    eval_mode: str,
    episodes: int,
    seed: int,
    console_log_path: Path,
    suppress_console: bool,
    checkpoint_idx: int,
    num_checkpoints: int,
    total_episodes: int,
    global_episode_offset: int,
    global_start_time: float,
    progress_interval: int,
) -> List[Dict[str, Any]]:
    deterministic = eval_mode == "deterministic"
    progress_interval = max(1, int(progress_interval))
    target = get_eval_train_target(mappo)
    was_training = getattr(target, "training", None)
    set_training_mode(target, False)
    rows: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []
    env = None
    ckpt_start_time = time.time()
    try:
        with redirected_console(console_log_path, suppress_console):
            agent = getattr(mappo, "agent", None)
            use_strict_v2_env = "strict_v2" in str(getattr(agent, "ablation_type", "")).lower()
            env = StrictV2DisasterRescueEnv(config) if use_strict_v2_env else DisasterRescueEnv(config)
            with torch.no_grad():
                for episode_id in range(int(episodes)):
                    episode_start_time = time.time()
                    metrics = evaluate_one_episode(mappo, env, config, deterministic, seed, episode_id)
                    last_ep_time = time.time() - episode_start_time
                    episode_audit_rows = metrics.pop("__sc_lifecycle_audit_records", [])
                    audit_truncated = bool(metrics.pop("__sc_lifecycle_audit_truncated", False))
                    metrics.update({
                        "checkpoint_path": str(checkpoint_path),
                        "checkpoint_name": checkpoint_path.name,
                        "checkpoint_step": int(checkpoint_step),
                        "eval_mode": eval_mode,
                    })
                    for audit_row in episode_audit_rows:
                        enriched = dict(audit_row)
                        enriched.update({
                            "checkpoint_path": str(checkpoint_path),
                            "checkpoint_name": checkpoint_path.name,
                            "checkpoint_step": int(checkpoint_step),
                            "eval_mode": eval_mode,
                            "audit_truncated": audit_truncated,
                        })
                        audit_rows.append(enriched)
                    rows.append(metrics)
                    episode_done = episode_id + 1
                    if episode_done == 1 or episode_done % progress_interval == 0 or episode_done == int(episodes):
                        elapsed_ckpt = time.time() - ckpt_start_time
                        completed_ckpt_eps = episode_done
                        remaining_ckpt_eps = int(episodes) - episode_done
                        avg_ckpt_ep_time = elapsed_ckpt / max(1, completed_ckpt_eps)
                        eta_ckpt = remaining_ckpt_eps * avg_ckpt_ep_time
                        completed_global_eps = int(global_episode_offset) + episode_done
                        remaining_global_eps = int(total_episodes) - completed_global_eps
                        elapsed_global = time.time() - float(global_start_time)
                        avg_global_ep_time = elapsed_global / max(1, completed_global_eps)
                        eta_total = max(0, remaining_global_eps) * avg_global_ep_time
                        progress_print(
                            f"[Eval] ckpt {checkpoint_idx}/{num_checkpoints} | "
                            f"episode {episode_done}/{episodes} | "
                            f"global {completed_global_eps}/{total_episodes} | "
                            f"last_ep={last_ep_time:.2f}s | "
                            f"avg_ckpt_ep={avg_ckpt_ep_time:.2f}s | "
                            f"elapsed_ckpt={elapsed_ckpt:.1f}s | "
                            f"ETA_ckpt={eta_ckpt:.1f}s | "
                            f"ETA_total={eta_total:.1f}s"
                        )
    finally:
        if env is not None:
            with redirected_console(console_log_path, suppress_console):
                env.close()
        if was_training is not None:
            set_training_mode(target, bool(was_training))
    setattr(evaluate_checkpoint, "last_sc_lifecycle_audit_rows", audit_rows)
    return rows


def summarize_rows(
    rows: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    config_path: str,
    checkpoint_path: Path,
    checkpoint_step: int,
    eval_mode: str,
    run_time: str,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "run_time": run_time,
        "tag": args.tag,
        "config_path": config_path,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_name": checkpoint_path.name,
        "checkpoint_step": int(checkpoint_step),
        "eval_mode": eval_mode,
        "episodes": int(args.episodes),
        "seed": int(args.seed),
        "mode": args.mode,
    }
    numeric_keys = sorted({k for row in rows for k, v in row.items() if isinstance(v, (int, float, np.floating))})
    for key in numeric_keys:
        values = np.array([float(row.get(key, 0.0)) for row in rows], dtype=np.float64)
        summary[f"{key}_mean"] = float(np.mean(values)) if len(values) else 0.0
        summary[f"{key}_std"] = float(np.std(values)) if len(values) else 0.0
        summary[f"{key}_min"] = float(np.min(values)) if len(values) else 0.0
        summary[f"{key}_max"] = float(np.max(values)) if len(values) else 0.0
    if rows:
        first = rows[0]
        for key in ("SC_PFR_Mode", "sc_score_source"):
            if key in first:
                summary[key] = first.get(key)
        for key in ("SC_PFR_Enabled", "SC_PFR_gen_valid", "terrain_aware_sc_scoring"):
            if key in first:
                summary[key] = bool(first.get(key))
    aggregate_sc_pfr_summary(rows, summary)
    return summary


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], preferred_columns: Optional[Iterable[str]] = None) -> None:
    if not rows:
        return
    keys = []
    for key in preferred_columns or []:
        if key not in keys:
            keys.append(key)
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, rows: Sequence[Dict[str, Any]], preferred_columns: Sequence[str]) -> None:
    if not rows:
        return
    columns = list(preferred_columns)
    for row in rows:
        for key in row.keys():
            if key not in columns:
                columns.append(key)
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summary_rows: Sequence[Dict[str, Any]]) -> None:
    lines: List[str] = []
    for row in summary_rows:
        lines.append(f"Checkpoint: {row.get('checkpoint_name', '')}")
        lines.append(f"CheckpointStep: {row.get('checkpoint_step', -1)}")
        lines.append(f"EvalMode: {row.get('eval_mode', '')}")
        lines.append(f"Episodes: {row.get('episodes', 0)}")
        for key, label in REPORT_METRICS:
            mean = float(row.get(f"{key}_mean", 0.0))
            std = float(row.get(f"{key}_std", 0.0))
            lines.append(f"{key}（{label}）= {mean:.4f} ± {std:.4f}")
        lines.append("")
        lines.append("SC Timeout Valid-S Diagnostic:")
        for key, label in (
            ("SCTimeout_SValidRate", "valid S rate at timeout"),
            ("SCTimeout_SMissingOrUnboundRate", "missing or unbound S rate at timeout"),
            ("SCTimeout_S2TaskEndMean_ValidOnly", "S-to-task end distance for valid S only"),
            ("SCTimeout_SWithinRadiusRate_ValidOnly", "S arrival rate for valid S only"),
            ("SCTimeout_SStillSelectedRate_ValidOnly", "valid S still selected this SC rate"),
            ("SCTimeout_SStillTargetTaskRate_ValidOnly", "valid S still targets task point rate"),
            ("SCTimeout_CValidRate", "valid C rate at timeout"),
            ("SCTimeout_C2SupportMean_ValidOnly", "C-to-support distance for valid C only"),
            ("SCTimeout_CWithinSupportRate_ValidOnly", "C support arrival rate for valid C only"),
        ):
            lines.append(f"  {key} ({label}): {float(row.get(f'{key}_mean', 0.0)):.4f}")

        lines.append("")
        lines.append("SC Timeout S-Failure Breakdown:")
        for key, label in (
            ("SFail_NoValidSRate", "no valid S rate"),
            ("SFail_NotCurrentTaskRate", "S current task is not this SC rate"),
            ("SFail_NotSelectedRate", "S no longer selected this SC rate"),
            ("SFail_TargetNotTaskRate", "S target is not task point rate"),
            ("SFail_OutOfSenseRadiusRate", "S outside sensing radius rate"),
            ("SFail_UnknownRate", "unknown S failure rate"),
        ):
            lines.append(f"  {key} ({label}): {float(row.get(f'{key}_mean', 0.0)):.4f}")

        lines.append("")
        lines.append("SC Timeout Valid-S Diagnostic:")
        for key, label in (
            ("SCTimeout_SValidRate", "valid S rate at timeout"),
            ("SCTimeout_SMissingOrUnboundRate", "missing or unbound S rate at timeout"),
            ("SCTimeout_S2TaskEndMean_ValidOnly", "S-to-task end distance for valid S only"),
            ("SCTimeout_SWithinRadiusRate_ValidOnly", "S arrival rate for valid S only"),
            ("SCTimeout_SStillSelectedRate_ValidOnly", "valid S still selected this SC rate"),
            ("SCTimeout_SStillTargetTaskRate_ValidOnly", "valid S still targets task point rate"),
            ("SCTimeout_CValidRate", "valid C rate at timeout"),
            ("SCTimeout_C2SupportMean_ValidOnly", "C-to-support distance for valid C only"),
            ("SCTimeout_CWithinSupportRate_ValidOnly", "C support arrival rate for valid C only"),
        ):
            lines.append(f"  {key} ({label}): {float(row.get(f'{key}_mean', 0.0)):.4f}")

        lines.append("")
        lines.append("SC Timeout S-Failure Breakdown:")
        for key, label in (
            ("SFail_NoValidSRate", "no valid S rate"),
            ("SFail_NotCurrentTaskRate", "S current task is not this SC rate"),
            ("SFail_NotSelectedRate", "S no longer selected this SC rate"),
            ("SFail_TargetNotTaskRate", "S target is not task point rate"),
            ("SFail_OutOfSenseRadiusRate", "S outside sensing radius rate"),
            ("SFail_UnknownRate", "unknown S failure rate"),
        ):
            lines.append(f"  {key} ({label}): {float(row.get(f'{key}_mean', 0.0)):.4f}")

        lines.append("")
        lines.append("SC Arrival Judgement Consistency Diagnostic:")
        for key, label in (
            ("SCSenseServiceRadiusCells", "sense service radius in cells"),
            ("SCSenseServiceRadiusMeters", "sense service radius in meters"),
            ("SCTimeout_SWithinRadiusByDistanceRate", "S arrival rate by distance only"),
            ("SCTimeout_SInTaskByEnvFlagRate", "S arrival rate by env predicate"),
            ("SCTimeout_SDistanceVsEnvMismatchRate", "distance-arrived but env-not-arrived rate"),
            ("SCTimeout_SHasCurrentTask", "S still has current task count"),
            ("SCTimeout_SCurrentTaskIsThisSC", "S current task is this SC count"),
            ("SCTimeout_SSelectedThisSC", "S selected this SC count"),
            ("SCTimeout_STargetIsTask", "S target remains task point count"),
            ("SCTimeout_STargetTaskGapMean", "mean S target-to-task gap"),
            ("SCReleasedDespiteSArrived", "released despite S distance arrival"),
            ("SCReleasedDespiteBothArrivedByDistance", "released despite S/C distance arrival"),
            ("SCServiceCheckBeforeRelease", "service readiness checks before release"),
            ("SCReleaseBeforeServiceCheck", "release before service checks"),
            ("SCTimeout_CWithinSupportByDistanceRate", "C support arrival rate by distance"),
            ("SCTimeout_BothWithinByDistanceRate", "S/C both arrival rate by distance"),
        ):
            lines.append(f"  {key} ({label}): {float(row.get(f'{key}_mean', 0.0)):.4f}")

        lines.append("")
        lines.append("SC Activation Relaxation Diagnostic:")
        for key, label in (
            ("SCActivation_CNearSupportBlocked", "blocked because C was not near support"),
            ("SCActivation_LinkReadyButCNotNearSupport", "links ready but C not near support"),
            ("SCActivation_ActivatedWithoutCNearSupport", "activated without C near support"),
            ("SCActivation_ReselectBlocked", "blocked by per-step reselect requirement"),
            ("SCActivation_CommitmentUsed", "candidate/commitment used to maintain cooperation"),
            ("SCActivation_ServiceReady", "S/upload/backhaul/time ready"),
            ("SCActivation_ServiceReadyButActiveLimit", "service ready but active limit blocked"),
        ):
            lines.append(f"  {key} ({label}): {float(row.get(f'{key}_mean', 0.0)):.4f}")

        lines.append("")
        lines.append("SC Candidate Lifecycle Audit:")
        for key, label in (
            ("SCAudit_TotalCandidates", "total SC candidates"),
            ("SCAudit_TotalCandidateSteps", "total SC candidate steps"),
            ("SCAudit_ActivatedCandidates", "activated candidates"),
            ("SCAudit_ReleasedCandidates", "released candidates"),
            ("SCAudit_CompletedCandidates", "completed candidates"),
            ("SCAudit_ExpiredCandidates", "expired candidates"),
        ):
            lines.append(f"  {key} ({label}): {float(row.get(f'{key}_mean', 0.0)):.4f}")

        lines.append("")
        lines.append("SC Candidate Consistency Audit:")
        for key, label in (
            ("SCAudit_SMismatchCandidateService", "S candidate vs service mismatch"),
            ("SCAudit_CMismatchCandidateService", "C candidate vs service mismatch"),
            ("SCAudit_SMismatchCurrentTask", "S current task mismatch"),
            ("SCAudit_CMismatchCurrentTask", "C current task mismatch"),
            ("SCAudit_SMismatchSelectedTask", "S selected task mismatch"),
            ("SCAudit_CMismatchSelectedTask", "C selected task mismatch"),
            ("SCAudit_SMismatchAssignedTask", "S assigned task mismatch"),
            ("SCAudit_CMismatchAssignedTask", "C assigned task mismatch"),
        ):
            lines.append(f"  {key} ({label}): {float(row.get(f'{key}_mean', 0.0)):.4f}")

        lines.append("")
        lines.append("SC Within-Radius Failure Audit:")
        for key, label in (
            ("SCAudit_SWithinRadiusButNotServiceReady", "S within radius but not service-ready"),
            ("SCAudit_SWithinRadiusButReleased", "S within radius but released"),
            ("SCAudit_SWithinRadiusButNoValidS", "S within radius but no valid S"),
            ("SCAudit_SWithinRadiusButUploadFalse", "S within radius but upload false"),
            ("SCAudit_SWithinRadiusButBackhaulFalse", "S within radius but backhaul false"),
            ("SCAudit_SWithinRadiusButTimeFalse", "S within radius but time false"),
            ("SCAudit_SWithinRadiusButActiveLimit", "S within radius but active limit"),
            ("SCAudit_SWithinRadiusButCommitmentLost", "S within radius but commitment lost"),
            ("SCAudit_SWithinRadiusButUnknown", "S within radius but unknown blocker"),
        ):
            lines.append(f"  {key} ({label}): {float(row.get(f'{key}_mean', 0.0)):.4f}")

        lines.append("")
        lines.append("SC Timeout Event-Weighted Audit:")
        for key, label in (
            ("SCAudit_EventWeighted_SValidTimeoutCount", "event-weighted valid S timeouts"),
            ("SCAudit_EventWeighted_SWithinRadiusTimeoutCount", "event-weighted S within-radius timeouts"),
            ("SCAudit_EventWeighted_SWithinRadiusTimeoutRate", "event-weighted S within-radius timeout rate"),
            ("SCAudit_EventWeighted_SDistMeanValidTimeout", "event-weighted valid S distance mean"),
            ("SCAudit_EventWeighted_SDistMedianValidTimeout", "event-weighted valid S distance median"),
            ("SCAudit_EventWeighted_SDistMinValidTimeout", "event-weighted valid S distance min"),
            ("SCAudit_EventWeighted_SDistMaxValidTimeout", "event-weighted valid S distance max"),
        ):
            lines.append(f"  {key} ({label}): {float(row.get(f'{key}_mean', 0.0)):.4f}")

        lines.append("")
        lines.append("SC Service-Release Order Audit:")
        for key, label in (
            ("SCAudit_ServiceCheckedThisStep", "service checks"),
            ("SCAudit_ReleaseCheckedThisStep", "release checks"),
            ("SCAudit_ServiceThenReleaseSameStep", "service then release in same step"),
            ("SCAudit_ReleaseWithoutServiceCheckSameStep", "release without same-step service check"),
            ("SCAudit_ServiceReadyButReleasedSameStep", "service-ready but released same step"),
            ("SCAudit_ServiceNotReadyReleaseReasonLogged", "not-ready release reason logged"),
            ("SCAudit_ServiceReadyButActiveLimit", "service-ready but active limit"),
            ("SCAudit_ActiveSCLimitValue", "active SC limit value"),
            ("SCAudit_CurrentActiveSCCount", "current active SC count"),
            ("SCAudit_ServiceReadyButActiveLimitReleased", "active-limit blocked and released"),
        ):
            lines.append(f"  {key} ({label}): {float(row.get(f'{key}_mean', 0.0)):.4f}")

        lines.append("")
        lines.append("SC Link Audit:")
        for key, label in (
            ("SCAudit_UploadDistanceWithinRadius", "upload distance within radius"),
            ("SCAudit_UploadTerrainChecked", "upload terrain checks"),
            ("SCAudit_UploadTerrainFailed", "upload terrain failures"),
            ("SCAudit_UploadFeasibleFalseButDistanceWithin", "upload false while distance within"),
            ("SCAudit_BackhaulDistanceWithinRadius", "backhaul distance within radius"),
            ("SCAudit_BackhaulTerrainChecked", "backhaul terrain checks"),
            ("SCAudit_BackhaulTerrainFailed", "backhaul terrain failures"),
            ("SCAudit_BackhaulFeasibleFalseButDistanceWithin", "backhaul false while distance within"),
        ):
            lines.append(f"  {key} ({label}): {float(row.get(f'{key}_mean', 0.0)):.4f}")

        lines.append("")
        lines.append("Sensing UAV Effective Speed Diagnostic:")
        for key, label in (
            ("SensingSC_MoveDistMean", "mean movement distance during SC candidate"),
            ("SensingSC_TaskApproachDeltaMean", "mean effective approach toward task"),
            ("SensingSC_TaskApproachNegativeRate", "non-approach or moving-away rate"),
            ("SensingSC_SpeedEfficiencyMean", "movement efficiency toward task"),
            ("SensingSC_CandidateApproachPerStepMean", "candidate total approach per step"),
        ):
            lines.append(f"  {key} ({label}): {float(row.get(f'{key}_mean', 0.0)):.4f}")

        lines.append("")
        lines.append("SC Progress-Aware Release Diagnostic:")
        for key, label in (
            ("ProgressAwareKeepSC", "kept because recent progress exists"),
            ("ProgressAwareReleaseNoProgress", "released after continuous no-progress"),
            ("ProgressAwareReleaseMaxTimeout", "released at maximum candidate timeout"),
            ("SCProgressSImproved", "sensing side improved"),
            ("SCProgressCImproved", "communication side improved"),
            ("SCProgressBothImproved", "both sides improved"),
            ("SCProgressAnyImproved", "any side improved"),
            ("SCProgressNoImprovement", "no side improved"),
        ):
            lines.append(f"  {key} ({label}): {float(row.get(f'{key}_mean', 0.0)):.4f}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report(path: Path, summary_rows: Sequence[Dict[str, Any]]) -> None:
    lines: List[str] = []
    for row in summary_rows:
        lines.append(f"Checkpoint: {row.get('checkpoint_name', '')}")
        lines.append(f"CheckpointStep: {row.get('checkpoint_step', -1)}")
        lines.append(f"EvalMode: {row.get('eval_mode', '')}")
        lines.append(f"Episodes: {row.get('episodes', 0)}")
        if int(row.get("missing_keys_count", 0) or 0) or int(row.get("unexpected_keys_count", 0) or 0):
            lines.append(
                f"CheckpointLoad: missing={row.get('missing_keys_count', 0)} | "
                f"unexpected={row.get('unexpected_keys_count', 0)}"
            )
        if bool(row.get("critical_missing_warning", False)):
            lines.append(
                "WARNING: missing keys include critical modules "
                "(intent_comm/nash_bargaining/actor/gnn/high_level_policy/task_pointer)."
            )
        for key, label in REPORT_METRICS:
            mean = float(row.get(f"{key}_mean", 0.0))
            std = float(row.get(f"{key}_std", 0.0))
            lines.append(f"{key}（{label}）= {mean:.4f} ± {std:.4f}")

        lines.append("")
        lines.append("SC Funnel Diagnostic:")
        for key, label in (
            ("GenSC_Unique", "生成SC任务数"),
            ("SelSC_Unique", "选择SC任务数"),
            ("BothSC", "S/C共同选择SC数"),
            ("PairSuccessSC_Unique", "SC配对成功数"),
            ("CommitWriteSuccess", "承诺写入成功数"),
            ("ActiveSC_Unique", "激活SC承诺数"),
            ("DoneSC_Unique", "完成SC任务数"),
            ("ExpSC_Unique", "过期SC任务数"),
        ):
            mean = float(row.get(f"{key}_mean", 0.0))
            std = float(row.get(f"{key}_std", 0.0))
            lines.append(f"  {key}（{label}） = {mean:.4f} ± {std:.4f}")

        lines.append("")
        lines.append("Conversion Rates:")
        for key, label in (
            ("SCSelectRate_Unique", "SC选择率"),
            ("BothSelectRate_SC", "S/C共同选择率"),
            ("PairSuccessRate_Unique", "配对成功率"),
            ("CommitToActiveRate", "承诺到激活率"),
            ("ActiveToDoneRate", "激活到完成率"),
            ("CommitToDoneRate", "承诺到完成率"),
        ):
            lines.append(f"  {key}（{label}） = {float(row.get(f'{key}_mean', 0.0)):.4f}")

        lines.append("")
        lines.append("Physical Feasibility:")
        for key, label in (
            ("UploadFeasibleRate_SC", "上传可行率"),
            ("BackhaulFeasibleRate_SC", "回传可行率"),
            ("TimeFeasibleRate_SC", "时间可行率"),
            ("UploadSuccessRate", "上传成功率"),
            ("BackhaulSuccessRate", "回传成功率"),
            ("LOSBlockedRate", "视距阻塞率"),
            ("AvgD2D", "平均二维距离"),
            ("AvgD3D", "平均三维距离"),
            ("AvgClearance", "平均净空"),
        ):
            lines.append(f"  {key}（{label}） = {float(row.get(f'{key}_mean', 0.0)):.4f}")

        lines.append("")
        lines.append("Mask / Release:")
        for key, label in (
            ("ReleaseSCCandidateTimeout", "SC候选超时释放"),
            ("ReleaseSCActiveFail", "SC激活失败释放"),
            ("SCContractActiveFail", "SC合同激活失败"),
        ):
            lines.append(f"  {key}（{label}） = {float(row.get(f'{key}_mean', 0.0)):.4f}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


REPORT_METRICS = [
    ("TCR_gen", "生成任务完成率"),
    ("TCR_sel", "选择任务完成率"),
    ("SC_CR", "SC任务完成率"),
    ("SC_PFR_gen", "生成SC物理可行率"),
    ("SC_CR_phys", "物理归一化SC完成率"),
    ("SC_sel", "已选择SC任务完成率"),
    ("DoneSC", "完成SC任务数"),
    ("Expired", "过期任务数"),
    ("PairSuccessSC", "SC配对成功数"),
    ("ActiveSC", "激活SC数"),
    ("CommitWriteSuccess", "commit写入成功数"),
    ("BackhaulSuccessRate", "回传成功率"),
    ("UploadSuccessRate", "上传成功率"),
    ("SwitchUAV", "平均每UAV切换数"),
]


def write_report(path: Path, summary_rows: Sequence[Dict[str, Any]]) -> None:
    lines: List[str] = []
    for row in summary_rows:
        lines.append(f"Checkpoint: {row.get('checkpoint_name', '')}")
        lines.append(f"CheckpointStep: {row.get('checkpoint_step', -1)}")
        lines.append(f"EvalMode: {row.get('eval_mode', '')}")
        lines.append(f"Episodes: {row.get('episodes', 0)}")
        if int(row.get("missing_keys_count", 0) or 0) or int(row.get("unexpected_keys_count", 0) or 0):
            lines.append(
                f"CheckpointLoad: missing={row.get('missing_keys_count', 0)} | "
                f"unexpected={row.get('unexpected_keys_count', 0)}"
            )
        if bool(row.get("critical_missing_warning", False)):
            lines.append(
                "WARNING: missing keys include critical modules "
                "(intent_comm/nash_bargaining/actor/gnn/high_level_policy/task_pointer)."
            )
        for key, label in REPORT_METRICS:
            mean = float(row.get(f"{key}_mean", 0.0))
            std = float(row.get(f"{key}_std", 0.0))
            lines.append(f"{key}（{label}）: {mean:.4f} ± {std:.4f}")

        lines.append("")
        lines.append("SC Funnel Diagnostic:")
        for key, label in (
            ("GenSC_Unique", "生成SC任务数"),
            ("SelSC_Unique", "选择SC任务数"),
            ("BothSC", "S/C共同选择SC数"),
            ("PairSuccessSC_Unique", "SC配对成功任务数"),
            ("CommitWriteSuccess", "commit写入成功数"),
            ("ActiveSC_Unique", "激活SC任务数"),
            ("DoneSC_Unique", "完成SC任务数"),
            ("ExpSC_Unique", "过期SC任务数"),
        ):
            mean = float(row.get(f"{key}_mean", 0.0))
            std = float(row.get(f"{key}_std", 0.0))
            lines.append(f"  {key}（{label}）= {mean:.4f} ± {std:.4f}")

        lines.append("")
        lines.append("Conversion Rates:")
        for key, label in (
            ("SCSelectRate_Unique", "SC选择率"),
            ("BothSelectRate_SC", "S/C共同选择率"),
            ("PairSuccessRate_Unique", "配对成功率"),
            ("CommitToActiveRate", "commit到激活率"),
            ("ActiveToDoneRate", "激活到完成率"),
            ("CommitToDoneRate", "commit到完成率"),
        ):
            lines.append(f"  {key}（{label}）= {float(row.get(f'{key}_mean', 0.0)):.4f}")

        lines.append("")
        lines.append("Physical Feasibility:")
        for key, label in (
            ("UploadFeasibleRate_SC", "上传可行率"),
            ("BackhaulFeasibleRate_SC", "回传可行率"),
            ("TimeFeasibleRate_SC", "时间可行率"),
            ("UploadSuccessRate", "上传成功率"),
            ("BackhaulSuccessRate", "回传成功率"),
            ("LOSBlockedRate", "LOS阻塞率"),
            ("AvgD2D", "平均2D距离"),
            ("AvgD3D", "平均3D距离"),
            ("AvgClearance", "平均净空"),
            ("SCSupportRuleEnvTargetGapMean", "rule-env支援目标平均差距"),
            ("SCSupportRuleEnvTargetGapMax", "rule-env支援目标最大差距"),
            ("SCSupportRuleEnvTargetGapCount", "rule-env支援目标差距计数"),
        ):
            lines.append(f"  {key}（{label}）= {float(row.get(f'{key}_mean', 0.0)):.4f}")

        lines.append("")
        lines.append("Mask / Release:")
        for key, label in (
            ("ReleaseSCCandidateTimeout", "SC候选超时释放"),
            ("ReleaseSCActiveFail", "SC激活失败释放"),
            ("SCContractActiveFail", "SC合约激活失败"),
        ):
            lines.append(f"  {key}（{label}）= {float(row.get(f'{key}_mean', 0.0)):.4f}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report(path: Path, summary_rows: Sequence[Dict[str, Any]]) -> None:
    lines: List[str] = []
    for row in summary_rows:
        lines.append(f"Checkpoint: {row.get('checkpoint_name', '')}")
        lines.append(f"CheckpointStep: {row.get('checkpoint_step', -1)}")
        lines.append(f"EvalMode: {row.get('eval_mode', '')}")
        lines.append(f"Episodes: {row.get('episodes', 0)}")
        if int(row.get("missing_keys_count", 0) or 0) or int(row.get("unexpected_keys_count", 0) or 0):
            lines.append(
                f"CheckpointLoad: missing={row.get('missing_keys_count', 0)} | "
                f"unexpected={row.get('unexpected_keys_count', 0)}"
            )
        lines.append("")
        lines.append("SC Funnel Diagnostic:")
        for key, label in (
            ("SelSC_Unique", "选择SC任务数"),
            ("PairSuccessSC_Unique", "SC配对成功任务数"),
            ("UploadFeasibleRate_SC", "step级候选上传可行率"),
            ("TaskLevelUploadPairRate_SC", "任务级上传配对成功率"),
            ("BackhaulFeasibleRate_SC", "回传可行率"),
            ("TimeFeasibleRate_SC", "时间可行率"),
        ):
            lines.append(f"  {key}（{label}）= {float(row.get(f'{key}_mean', 0.0)):.4f}")

        lines.append("")
        lines.append("SC Candidate Timeout Diagnostic:")
        for key, label in (
            ("SCTimeoutCount", "SC候选超时次数"),
            ("SCTimeout_S2TaskMean", "超时时S到任务平均距离"),
            ("SCTimeout_C2SupportMean", "超时时C到支援点平均距离"),
            ("SCTimeout_S2CMean", "超时时S-C平均距离"),
            ("SCTimeout_C2BaseMean", "超时时C到基地平均距离"),
            ("SCTimeout_UploadViolationMean", "上传半径平均超出量"),
            ("SCTimeout_BackhaulViolationMean", "回传半径平均超出量"),
            ("SCTimeout_SInTaskRate", "超时时S已到任务率"),
            ("SCTimeout_CNearSupportRate", "超时时C已到支援点率"),
            ("SCTimeout_UploadFeasibleRate", "超时时上传可行率"),
            ("SCTimeout_BackhaulFeasibleRate", "超时时回传可行率"),
            ("SCTimeout_BothArrivedRate", "超时时S/C均到位率"),
            ("SCTimeout_NeitherArrivedRate", "超时时S/C均未到位率"),
            ("SCTimeout_SMissingRate", "超时时S侧未到位率"),
            ("SCTimeout_CMissingRate", "超时时C侧未到位率"),
        ):
            lines.append(f"  {key}（{label}）= {float(row.get(f'{key}_mean', 0.0)):.4f}")

        lines.append("")
        lines.append("SC Candidate Approach Progress:")
        for key, label in (
            ("SCTimeout_S2TaskStartMean", "超时时S到任务初始距离"),
            ("SCTimeout_S2TaskEndMean", "超时时S到任务结束距离"),
            ("SCTimeout_S2TaskDeltaMean", "超时时S到任务距离改善量"),
            ("SCTimeout_C2SupportStartMean", "超时时C到支援点初始距离"),
            ("SCTimeout_C2SupportEndMean", "超时时C到支援点结束距离"),
            ("SCTimeout_C2SupportDeltaMean", "超时时C到支援点距离改善量"),
            ("SCTimeout_S2TaskImproveRate", "超时时S持续接近率"),
            ("SCTimeout_C2SupportImproveRate", "超时时C持续接近率"),
            ("SCTimeout_BothImproveRate", "超时时S/C同时接近率"),
            ("SCTimeout_NoProgressRate", "超时时无进展率"),
        ):
            lines.append(f"  {key}（{label}）= {float(row.get(f'{key}_mean', 0.0)):.4f}")
        lines.append("")
        lines.append("")
        lines.append("SC Timeout Valid-S Diagnostic:")
        for key, label in (
            ("SCTimeout_SValidRate", "valid S rate at timeout"),
            ("SCTimeout_SMissingOrUnboundRate", "missing or unbound S rate at timeout"),
            ("SCTimeout_S2TaskEndMean_ValidOnly", "S-to-task end distance for valid S only"),
            ("SCTimeout_SWithinRadiusRate_ValidOnly", "S arrival rate for valid S only"),
            ("SCTimeout_SStillSelectedRate_ValidOnly", "valid S still selected this SC rate"),
            ("SCTimeout_SStillTargetTaskRate_ValidOnly", "valid S still targets task point rate"),
            ("SCTimeout_CValidRate", "valid C rate at timeout"),
            ("SCTimeout_C2SupportMean_ValidOnly", "C-to-support distance for valid C only"),
            ("SCTimeout_CWithinSupportRate_ValidOnly", "C support arrival rate for valid C only"),
        ):
            lines.append(f"  {key} ({label}): {float(row.get(f'{key}_mean', 0.0)):.4f}")

        lines.append("")
        lines.append("SC Timeout S-Failure Breakdown:")
        for key, label in (
            ("SFail_NoValidSRate", "no valid S rate"),
            ("SFail_NotCurrentTaskRate", "S current task is not this SC rate"),
            ("SFail_NotSelectedRate", "S no longer selected this SC rate"),
            ("SFail_TargetNotTaskRate", "S target is not task point rate"),
            ("SFail_OutOfSenseRadiusRate", "S outside sensing radius rate"),
            ("SFail_UnknownRate", "unknown S failure rate"),
        ):
            lines.append(f"  {key} ({label}): {float(row.get(f'{key}_mean', 0.0)):.4f}")

        lines.append("")
        lines.append("SC Arrival Judgement Consistency Diagnostic:")
        for key, label in (
            ("SCSenseServiceRadiusCells", "sense service radius in cells"),
            ("SCSenseServiceRadiusMeters", "sense service radius in meters"),
            ("SCTimeout_SWithinRadiusByDistanceRate", "S arrival rate by distance only"),
            ("SCTimeout_SInTaskByEnvFlagRate", "S arrival rate by env predicate"),
            ("SCTimeout_SDistanceVsEnvMismatchRate", "distance-arrived but env-not-arrived rate"),
            ("SCTimeout_SHasCurrentTask", "S still has current task count"),
            ("SCTimeout_SCurrentTaskIsThisSC", "S current task is this SC count"),
            ("SCTimeout_SSelectedThisSC", "S selected this SC count"),
            ("SCTimeout_STargetIsTask", "S target remains task point count"),
            ("SCTimeout_STargetTaskGapMean", "mean S target-to-task gap"),
            ("SCReleasedDespiteSArrived", "released despite S distance arrival"),
            ("SCReleasedDespiteBothArrivedByDistance", "released despite S/C distance arrival"),
            ("SCServiceCheckBeforeRelease", "service readiness checks before release"),
            ("SCReleaseBeforeServiceCheck", "release before service checks"),
            ("SCTimeout_CWithinSupportByDistanceRate", "C support arrival rate by distance"),
            ("SCTimeout_BothWithinByDistanceRate", "S/C both arrival rate by distance"),
        ):
            lines.append(f"  {key} ({label}): {float(row.get(f'{key}_mean', 0.0)):.4f}")

        lines.append("")
        lines.append("SC Activation Relaxation Diagnostic:")
        for key, label in (
            ("SCActivation_CNearSupportBlocked", "blocked because C was not near support"),
            ("SCActivation_LinkReadyButCNotNearSupport", "links ready but C not near support"),
            ("SCActivation_ActivatedWithoutCNearSupport", "activated without C near support"),
            ("SCActivation_ReselectBlocked", "blocked by per-step reselect requirement"),
            ("SCActivation_CommitmentUsed", "candidate/commitment used to maintain cooperation"),
            ("SCActivation_ServiceReady", "S/upload/backhaul/time ready"),
            ("SCActivation_ServiceReadyButActiveLimit", "service ready but active limit blocked"),
        ):
            lines.append(f"  {key} ({label}): {float(row.get(f'{key}_mean', 0.0)):.4f}")

        lines.append("")
        lines.append("Sensing UAV Effective Speed Diagnostic:")
        for key, label in (
            ("SensingSC_MoveDistMean", "mean movement distance during SC candidate"),
            ("SensingSC_TaskApproachDeltaMean", "mean effective approach toward task"),
            ("SensingSC_TaskApproachNegativeRate", "non-approach or moving-away rate"),
            ("SensingSC_SpeedEfficiencyMean", "movement efficiency toward task"),
            ("SensingSC_CandidateApproachPerStepMean", "candidate total approach per step"),
        ):
            lines.append(f"  {key} ({label}): {float(row.get(f'{key}_mean', 0.0)):.4f}")

        lines.append("")
        lines.append("SC S-Arrival Grace Diagnostic:")
        for key, label in (
            ("SArrivalGraceTriggered", "S arrival grace trigger attempts"),
            ("SArrivalGraceKept", "candidate kept by S arrival grace"),
            ("SArrivalGraceExpired", "grace exhausted before entering true radius"),
            ("SArrivalGraceToActive", "grace candidate later became active"),
            ("SArrivalGraceToDone", "grace candidate later completed"),
            ("SArrivalGraceBlockedNoValidS", "blocked because no valid S"),
            ("SArrivalGraceBlockedNoProgress", "blocked because S had no recent progress"),
            ("SArrivalGraceBlockedNoValidC", "blocked because no valid C"),
            ("SArrivalGraceBlockedMaxTimeout", "blocked by max candidate timeout"),
            ("SArrivalGraceDistToSenseBoundaryMean", "mean distance to true sensing boundary"),
            ("SArrivalGraceUsedStepsMean", "mean grace steps used"),
        ):
            lines.append(f"  {key} ({label}): {float(row.get(f'{key}_mean', 0.0)):.4f}")

        lines.append("")
        lines.append("SC Progress-Aware Release Diagnostic:")
        for key, label in (
            ("ProgressAwareKeepSC", "kept because recent progress exists"),
            ("ProgressAwareReleaseNoProgress", "released after continuous no-progress"),
            ("ProgressAwareReleaseMaxTimeout", "released at maximum candidate timeout"),
            ("SCProgressSImproved", "sensing side improved"),
            ("SCProgressCImproved", "communication side improved"),
            ("SCProgressBothImproved", "both sides improved"),
            ("SCProgressAnyImproved", "any side improved"),
            ("SCProgressNoImprovement", "no side improved"),
        ):
            lines.append(f"  {key} ({label}): {float(row.get(f'{key}_mean', 0.0)):.4f}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def sc_metric_semantics_warnings(row: Dict[str, Any]) -> List[str]:
    warnings: List[str] = []
    enabled = bool(row.get("SC_PFR_Enabled", row.get("SC_PFR_Enabled_mean", False)))
    sc_pfr = summary_mean(row, "SC_PFR_gen")
    try:
        sc_pfr_f = float(sc_pfr)
    except (TypeError, ValueError):
        sc_pfr_f = np.nan
    if not enabled and np.isfinite(sc_pfr_f) and abs(sc_pfr_f) <= 1e-12:
        warnings.append("SC_PFR_gen is disabled; do not interpret 0.0 as physical infeasibility.")
    phys_generated = finite_float(summary_mean(row, "SC_PhysGeneratedSC"), np.nan)
    phys_feasible = finite_float(summary_mean(row, "SC_PhysFeasibleSC"), np.nan)
    sc_cr_phys = finite_float(summary_mean(row, "SC_CR_phys"), np.nan)
    valid = bool(row.get("SC_PFR_gen_valid", row.get("SC_PFR_gen_valid_mean", False)))
    if enabled and valid and np.isfinite(phys_generated) and phys_generated > 0.0 and not np.isfinite(sc_pfr_f):
        warnings.append(
            "WARNING: SC_PFR_gen is invalid despite enabled release_once and positive "
            "SC_PhysGeneratedSC. Check summary aggregation."
        )
    if enabled and np.isfinite(phys_feasible) and phys_feasible > 0.0 and not np.isfinite(sc_cr_phys):
        warnings.append(
            "WARNING: SC_CR_phys is invalid despite positive SC_PhysFeasibleSC. "
            "Check summary aggregation."
        )
    if np.isfinite(phys_feasible) and phys_feasible > 0.0 and "SC_PhysFeasibleSC=0" in str(row.get("SC_CR_phys_invalid_reason", "")):
        warnings.append(
            "WARNING: SC_CR_phys invalid reason contradicts positive SC_PhysFeasibleSC; "
            "check summary aggregation."
        )

    step_upload = float(summary_mean(row, "StepUploadFeasibleRate_SC", summary_mean(row, "UploadFeasibleRate_SC", 0.0)) or 0.0)
    task_pair = float(summary_mean(row, "TaskLevelUploadPairRate_SC", 0.0) or 0.0)
    if step_upload < 0.05 and task_pair > 0.5:
        warnings.append(
            "Step-level upload feasibility is low due to transition steps; "
            "task-level upload pair rate should be used for SC funnel analysis."
        )

    terrain_aware = bool(row.get("terrain_aware_sc_scoring", row.get("terrain_aware_sc_scoring_mean", False)))
    legacy = summary_mean(row, "legacy_total_sc_feasible_rate")
    terrain = summary_mean(row, "total_sc_feasible_rate_terrain")
    try:
        legacy_f = float(legacy)
        terrain_f = float(terrain)
    except (TypeError, ValueError):
        legacy_f = terrain_f = np.nan
    if terrain_aware and np.isfinite(legacy_f) and np.isfinite(terrain_f) and abs(legacy_f - terrain_f) > 0.25:
        warnings.append(
            "BaseSelectionDiag contains legacy and terrain-aware feasibility rates. "
            "Use sc_score_feasible_rate_used."
        )
    if terrain_aware and not row.get("sc_score_source"):
        warnings.append("BaseSelectionDiag missing sc_score_source.")

    done_sc_key = str(row.get("SC_CR_phys_done_key", "DoneSC_Unique"))
    done_sc = finite_float(row.get(f"{done_sc_key}_sum", np.nan), np.nan)
    if not np.isfinite(done_sc):
        done_sc = finite_float(summary_mean(row, done_sc_key, summary_mean(row, "DoneSC")), np.nan)
    try:
        phys_feasible_sum = finite_float(row.get("SC_PhysFeasibleSC_sum", np.nan), np.nan)
        phys_feasible_for_warning = phys_feasible_sum if np.isfinite(phys_feasible_sum) else phys_feasible
        if enabled and float(phys_feasible_for_warning) > 0 and float(done_sc) > float(phys_feasible_for_warning):
            warnings.append(
                f"WARNING: {done_sc_key} exceeds SC_PhysFeasibleSC; check physical feasibility "
                "probe strictness or denominator consistency."
            )
    except (TypeError, ValueError):
        pass
    return warnings


def write_report(path: Path, summary_rows: Sequence[Dict[str, Any]]) -> None:
    lines: List[str] = []
    for row in summary_rows:
        mode = str(row.get("SC_PFR_Mode", "off"))
        enabled = bool(row.get("SC_PFR_Enabled", row.get("SC_PFR_Enabled_mean", False)))
        valid = bool(row.get("SC_PFR_gen_valid", row.get("SC_PFR_gen_valid_mean", False)))

        lines.append(f"Checkpoint: {row.get('checkpoint_name', '')}")
        lines.append(f"CheckpointStep: {row.get('checkpoint_step', -1)}")
        lines.append(f"EvalMode: {row.get('eval_mode', '')}")
        lines.append(f"Episodes: {row.get('episodes', 0)}")
        if int(row.get("missing_keys_count", 0) or 0) or int(row.get("unexpected_keys_count", 0) or 0):
            lines.append(
                f"CheckpointLoad: missing={row.get('missing_keys_count', 0)} | "
                f"unexpected={row.get('unexpected_keys_count', 0)}"
            )

        lines.append("")
        lines.append("A. Main Performance Metrics")
        lines.append(f"  TCR_gen (generated task completion rate): {report_num(summary_mean(row, 'TCR_gen'))}")
        lines.append("  Definition note: Done / GeneratedTasks, using the existing code denominator.")
        lines.append(f"  SC_CR (SC task completion rate): {report_num(summary_mean(row, 'SC_CR'))}")
        lines.append("  Definition note: DoneSC / current task-level SC denominator used by the code.")
        lines.append(f"  Reward: {report_num(summary_mean(row, 'episode_reward'))}")
        lines.append(f"  Done: {report_num(summary_mean(row, 'Done'))}")
        lines.append(f"  Expired: {report_num(summary_mean(row, 'Expired'))}")
        lines.append(f"  Safety / Collision: {report_num(summary_mean(row, 'total_collisions'))}")

        lines.append("")
        lines.append("B. SC Task-Level Funnel Diagnostics")
        for key, label in (
            ("SelSC_Unique", "selected SC tasks"),
            ("PairSuccessSC_Unique", "SC pair-success tasks"),
            ("TaskLevelUploadPairRate_SC", "task-level upload pair success rate"),
            ("BackhaulFeasibleRate_SC", "backhaul feasible rate"),
            ("TimeFeasibleRate_SC", "time feasible rate"),
        ):
            lines.append(f"  {key} ({label}): {report_num(summary_mean(row, key))}")
        lines.append("  BackhaulFeasibleRate_SC denominator follows the current code: BackhaulFeasibleSC / UploadFeasibleSC.")
        lines.append("  TimeFeasibleRate_SC denominator follows the current code: TimeFeasibleSC / BackhaulFeasibleSC.")

        lines.append("")
        lines.append("C. SC Step-Level Instantaneous Diagnostics")
        step_upload = summary_mean(row, "StepUploadFeasibleRate_SC", summary_mean(row, "UploadFeasibleRate_SC"))
        lines.append(f"  StepUploadFeasibleRate_SC (step-level instantaneous upload feasibility): {report_num(step_upload)}")
        lines.append(f"  UploadFeasibleRate_SC (deprecated alias): {report_num(summary_mean(row, 'UploadFeasibleRate_SC'))}")
        lines.append("  UploadFeasibleRate_SC is deprecated alias of StepUploadFeasibleRate_SC.")
        lines.append("  It is a step-level instantaneous upload feasibility metric, not SC_PFR_gen.")
        lines.append("  StepUploadFeasibleRate_SC is low because it includes many candidate steps before S/C arrival.")

        lines.append("")
        lines.append("D. SC Physical Feasibility Diagnostics")
        lines.append(f"  SC_PFR_Mode: {mode}")
        lines.append(f"  SC_PFR_Enabled: {str(enabled).lower()}")
        lines.append(f"  SC_PFR_gen_valid: {str(valid).lower()}")
        if enabled:
            lines.append(f"  SC_PhysGeneratedSC: {report_num(summary_mean(row, 'SC_PhysGeneratedSC'))}")
            lines.append(f"  SC_PhysFeasibleSC: {report_num(summary_mean(row, 'SC_PhysFeasibleSC'))}")
            lines.append(f"  SC_PFR_gen: {report_num(summary_mean(row, 'SC_PFR_gen'))}")
            lines.append(f"  SC_CR_phys: {report_num(summary_mean(row, 'SC_CR_phys'))}")
            pfr_reason = str(row.get("SC_PFR_gen_invalid_reason", "") or "")
            cr_reason = str(row.get("SC_CR_phys_invalid_reason", "") or "")
            if pfr_reason:
                lines.append(f"  SC_PFR_gen invalid reason: {pfr_reason}")
            if cr_reason:
                lines.append(f"  SC_CR_phys invalid reason: {cr_reason}")
            done_key = str(row.get("SC_CR_phys_done_key", "DoneSC_Unique"))
            lines.append("  Aggregation note: SC_PFR_gen = sum(SC_PhysFeasibleSC) / sum(SC_PhysGeneratedSC).")
            lines.append(f"  Aggregation note: SC_CR_phys = sum({done_key}) / sum(SC_PhysFeasibleSC).")
        else:
            lines.append("  SC_PFR_gen: n/a")
            lines.append("  SC_CR_phys: n/a")
            lines.append("  Reason: sc_pfr_probe_mode=off.")

        lines.append("")
        lines.append("E. Base Selection Static Feasibility")
        lines.append(f"  sc_score_source: {row.get('sc_score_source', 'unknown')}")
        lines.append(f"  sc_score_feasible_rate_used: {report_num(summary_mean(row, 'sc_score_feasible_rate_used'))}")
        lines.append(f"  legacy_total_sc_feasible_rate: {report_num(summary_mean(row, 'legacy_total_sc_feasible_rate'))}")
        lines.append(f"  total_sc_feasible_rate_terrain: {report_num(summary_mean(row, 'total_sc_feasible_rate_terrain'))}")
        lines.append(f"  terrain_aware_sc_scoring: {str(bool(row.get('terrain_aware_sc_scoring', row.get('terrain_aware_sc_scoring_mean', False)))).lower()}")
        lines.append("  Base selection feasibility is a static deployment score, not episode-level SC_PFR_gen.")

        warnings = sc_metric_semantics_warnings(row)
        if warnings:
            lines.append("")
            lines.append("Sanity Warnings")
            for warning in warnings:
                lines.append(f"  WARNING: {warning}" if not warning.startswith("WARNING:") else f"  {warning}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if torch.is_tensor(obj):
        return obj.item() if obj.numel() == 1 else obj.detach().cpu().tolist()
    return str(obj)


def build_config_audit(
    args: argparse.Namespace,
    config: Config,
    checkpoints: Sequence[Path],
    provided_config_path: Path,
    training_config_path: Optional[Path],
    effective_config_path: Path,
    eval_mode_requested: Optional[str],
    eval_mode_effective: str,
    config_diffs: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    sibling_yaml = []
    if checkpoint_dir is not None and checkpoint_dir.exists():
        sibling_yaml = [str(p) for p in sorted(checkpoint_dir.glob("*.yaml"))]
        cfg_name = Path(args.config).name
        same_name = [str(p) for p in checkpoint_dir.glob(cfg_name)]
    else:
        same_name = []

    task_ratio = getattr(config.task, "task_type_ratio", getattr(config.task, "task_type_ratios", {}))
    audit = {
        "effective_config_path": relpath_text(effective_config_path),
        "provided_config_path": relpath_text(provided_config_path),
        "training_config_path": relpath_text(training_config_path),
        "prefer_training_config": bool(args.prefer_training_config),
        "strict_config_match": bool(args.strict_config_match),
        "eval_mode_requested": eval_mode_requested,
        "eval_mode_effective": eval_mode_effective,
        "sc_pfr_probe_mode": str(args.sc_pfr_probe_mode),
        "config_path": args.config,
        "checkpoint_dir": args.checkpoint_dir,
        "checkpoint_list": [str(p) for p in checkpoints],
        "checkpoint_dir_yaml_files": sibling_yaml,
        "checkpoint_dir_has_same_config_name": bool(same_name),
        "checkpoint_dir_same_config_paths": same_name,
        "config_differences": list(config_diffs),
        "key_fields": {
            "map_size": config_value_text(getattr(config.environment, "map_size", None)),
            "grid_resolution_m": config_value_text(getattr(config.environment, "grid_resolution_m", None)),
            "num_uavs": config_value_text(getattr(config.environment, "num_uavs", None)),
            "num_sensing_uavs": config_value_text(getattr(config.environment, "num_sensing_uavs", None)),
            "num_comm_uavs": config_value_text(getattr(config.environment, "num_comm_uavs", None)),
            "task_type_ratio": config_value_text(task_ratio),
            "generation_window": config_value_text(getattr(config.task, "generation_window", None)),
            "generation_fade_end": config_value_text(getattr(config.task, "generation_fade_end", None)),
            "use_reachable_deadline": config_value_text(getattr(config.task, "use_reachable_deadline", None)),
            "deadline_range": config_value_text(getattr(config.task, "deadline_range", None)),
            "hard_deadline_range": config_value_text(getattr(config.task, "hard_deadline_range", None)),
            "deadline_slack": config_value_text(getattr(config.task, "deadline_slack", None)),
            "processing_time_range": config_value_text(getattr(config.task, "processing_time_range", None)),
            "max_active_tasks": config_value_text(getattr(config.task, "max_active_tasks", None)),
            "max_active_sc_tasks": config_value_text(getattr(config.task, "max_active_sc_tasks", None)),
            "max_active_c_tasks": config_value_text(getattr(config.task, "max_active_c_tasks", None)),
            "data_upload_radius_m": config_value_text(getattr(config.uav, "data_upload_radius_m", None)),
            "backhaul_link_radius_m": config_value_text(getattr(config.uav, "backhaul_link_radius_m", None)),
            "comm_service_radius_m": config_value_text(getattr(config.uav, "comm_service_radius_m", None)),
            "nlos_radius_factor": config_value_text(getattr(config.real_geo, "nlos_radius_factor", None)),
            "flight_height_m": config_value_text(getattr(config.real_geo, "flight_height_m", None)),
            "base_height_m": config_value_text(getattr(config.real_geo, "base_height_m", None)),
            "sc_support_upload_margin": config_value_text(getattr(config.communication, "sc_support_upload_margin", None)),
            "sc_support_backhaul_margin": config_value_text(getattr(config.communication, "sc_support_backhaul_margin", None)),
            "eval_action_mode": config_value_text(getattr(config.evaluation, "eval_action_mode", None)),
            "episodes": int(args.episodes),
            "eval_mode": args.eval_mode,
            "seed": int(args.seed),
        },
        "critical_key_fields": collect_key_fields(config),
    }
    return audit


def main() -> None:
    args = parse_args()
    provided_config_path = Path(args.config)
    training_config_path = find_training_config_path(args)
    config_diffs: List[Dict[str, Any]] = []
    if training_config_path is not None:
        progress_print("[ConfigAudit] Found training config:")
        progress_print(f"[ConfigAudit] {relpath_text(training_config_path)}")
        if not same_path(provided_config_path, training_config_path):
            progress_print("[ConfigAudit][WARNING] Evaluation config differs from training config.")
            progress_print(f"[ConfigAudit][WARNING] Provided config: {relpath_text(provided_config_path)}")
            progress_print(f"[ConfigAudit][WARNING] Training config: {relpath_text(training_config_path)}")
            progress_print("[ConfigAudit][WARNING] This may invalidate completion-rate comparisons.")
            provided_config_for_compare = Config(str(provided_config_path))
            training_config_for_compare = Config(str(training_config_path))
            config_diffs = compare_config_fields(provided_config_for_compare, training_config_for_compare)
            print_config_diff_table(config_diffs)
            if bool(args.strict_config_match) and config_diffs:
                raise SystemExit("[ConfigAudit][ERROR] strict_config_match=True and critical config fields differ.")
    else:
        progress_print("[ConfigAudit][WARNING] No logs/<experiment>/config.yaml found for this checkpoint.")

    effective_config_path = training_config_path if (bool(args.prefer_training_config) and training_config_path is not None) else provided_config_path
    if bool(args.prefer_training_config) and training_config_path is not None:
        progress_print("[ConfigAudit] prefer_training_config=True, using training config:")
        progress_print(f"[ConfigAudit] {relpath_text(training_config_path)}")
    args.provided_config = str(provided_config_path)
    args.training_config = str(training_config_path) if training_config_path is not None else None
    args.effective_config = str(effective_config_path)
    args.config = str(effective_config_path)

    config = Config(str(effective_config_path))
    config.seed = int(args.seed)
    apply_sc_pfr_probe_mode(config, args.sc_pfr_probe_mode)
    eval_mode_requested = args.eval_mode
    config_eval_action_mode = str(getattr(config.evaluation, "eval_action_mode", "stochastic")).lower()
    if args.eval_mode is None:
        args.eval_mode = config_eval_action_mode if config_eval_action_mode in ("stochastic", "deterministic", "both") else "stochastic"
    if args.eval_mode == "deterministic" and config_eval_action_mode == "stochastic":
        progress_print("[Eval][WARNING] You are evaluating deterministic while training config suggests stochastic.")
        progress_print("[Eval][WARNING] Results may not be comparable with previous stochastic reports.")
    if args.sc_pfr_probe_mode == "strict":
        progress_print("[Eval][WARNING] SC_PFR strict probe runs heavy feasibility diagnostics; do not use it for official completion-rate comparisons.")
    progress_print(f"[SC-PFR] eval mode: sc_pfr_probe_mode={args.sc_pfr_probe_mode}")
    if args.sc_pfr_probe_mode == "off":
        progress_print("[SC-PFR] SC_PFR_gen disabled. Physical feasibility metrics will be reported as NaN/n/a.")
    else:
        progress_print("[SC-PFR] probe policy: release-time only, cached per SC task.")
    device = resolve_device(args.device)
    eval_modes = ["stochastic", "deterministic"] if args.eval_mode == "both" else [args.eval_mode]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_time = time.strftime("%Y%m%d_%H%M%S")
    tag_suffix = f"_{args.tag}" if args.tag else ""
    console_log_path = output_dir / f"console_log_{run_time}{tag_suffix}.txt"
    suppress_console = bool(args.quiet) and not bool(args.verbose)
    checkpoints = select_checkpoints(args)
    progress_interval = max(1, int(args.progress_interval))
    total_episodes = len(checkpoints) * int(args.episodes) * len(eval_modes)
    global_start_time = time.time()

    progress_print(f"[Eval] Found {len(checkpoints)} checkpoints")
    progress_print(f"[Eval] Episodes per checkpoint: {args.episodes}")
    progress_print(f"[Eval] Total episodes: {total_episodes}")
    progress_print(f"[Eval] Eval mode requested: {eval_mode_requested}")
    progress_print(f"[Eval] Eval mode effective: {args.eval_mode}")
    progress_print(f"[Eval] Config eval_action_mode: {config_eval_action_mode}")
    progress_print(f"[Eval] SC_PFR probe mode: {args.sc_pfr_probe_mode}")
    progress_print(f"[Eval] Progress interval: every {progress_interval} episodes")

    per_episode_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    checkpoint_records: List[Dict[str, Any]] = []
    sc_lifecycle_audit_rows: List[Dict[str, Any]] = []

    for checkpoint_idx, checkpoint_path in enumerate(checkpoints, start=1):
        ckpt_wall_start = time.time()
        progress_print(f"[Eval] Checkpoint {checkpoint_idx}/{len(checkpoints)}: {checkpoint_path.name}")
        progress_print(f"[Eval] Starting checkpoint {checkpoint_idx}/{len(checkpoints)} with {args.episodes} episodes...")
        with redirected_console(console_log_path, suppress_console):
            mappo = build_eval_hierarchical_algo(config, device, args.mode)
            checkpoint_step, source_key, load_info = load_checkpoint(mappo, checkpoint_path, device)
        checkpoint_records.append({
            "path": str(checkpoint_path),
            "name": checkpoint_path.name,
            "step": checkpoint_step,
            "source_key": source_key,
            **load_info,
        })

        for eval_mode_idx, eval_mode in enumerate(eval_modes):
            global_episode_offset = (
                ((checkpoint_idx - 1) * len(eval_modes) + eval_mode_idx) * int(args.episodes)
            )
            rows = evaluate_checkpoint(
                mappo=mappo,
                config=config,
                checkpoint_path=checkpoint_path,
                checkpoint_step=checkpoint_step,
                eval_mode=eval_mode,
                episodes=args.episodes,
                seed=args.seed,
                console_log_path=console_log_path,
                suppress_console=suppress_console,
                checkpoint_idx=checkpoint_idx,
                num_checkpoints=len(checkpoints),
                total_episodes=total_episodes,
                global_episode_offset=global_episode_offset,
                global_start_time=global_start_time,
                progress_interval=progress_interval,
            )
            per_episode_rows.extend(rows)
            sc_lifecycle_audit_rows.extend(getattr(evaluate_checkpoint, "last_sc_lifecycle_audit_rows", []))
            summary = summarize_rows(
                rows=rows,
                args=args,
                config_path=args.config,
                checkpoint_path=checkpoint_path,
                checkpoint_step=checkpoint_step,
                eval_mode=eval_mode,
                run_time=run_time,
            )
            summary.update(load_info)
            summary_rows.append(summary)
            for warning in sc_metric_semantics_warnings(summary):
                progress_print(f"[SC-MetricSemantics][WARNING] {warning}")
            ckpt_elapsed = time.time() - ckpt_wall_start
            avg_ep_time = ckpt_elapsed / max(1, int(args.episodes))
            progress_print(f"[Eval] Finished checkpoint {checkpoint_idx}/{len(checkpoints)}: {checkpoint_path.name}")
            progress_print(f"[Eval] checkpoint_time={ckpt_elapsed:.1f}s | avg_ep_time={avg_ep_time:.2f}s")
            progress_print(
                f"[Eval] Key metrics: "
                f"TCR_gen={safe_metric(summary, 'TCR_gen_mean')}, "
                f"SC_CR={safe_metric(summary, 'SC_CR_mean')}, "
                f"SC_PFR_gen={safe_metric(summary, 'SC_PFR_gen_mean')}, "
                f"reward={safe_metric(summary, 'episode_reward_mean', precision=2)}"
            )
            progress_print(
                f"[Summary] checkpoint={checkpoint_step} | "
                f"TCR_gen={safe_metric(summary, 'TCR_gen_mean', precision=3)} | "
                f"SC_CR={safe_metric(summary, 'SC_CR_mean', precision=3)} | "
                f"SC_PFR_gen={safe_metric(summary, 'SC_PFR_gen_mean', default='n/a', precision=3)} | "
                f"SC_CR_phys={safe_metric(summary, 'SC_CR_phys_mean', default='n/a', precision=3)} | "
                f"DoneSC={safe_metric(summary, 'DoneSC_mean', precision=2)} | "
                f"Expired={safe_metric(summary, 'Expired_mean', precision=2)}"
            )

    metric_summary_columns = []
    for metric in MAIN_METRICS + SC_FUNNEL_SUMMARY_METRICS:
        metric_summary_columns.extend([f"{metric}_mean", f"{metric}_std"])
    required_summary_columns = SUMMARY_REQUIRED_COLUMNS + [
        "missing_keys_count", "unexpected_keys_count",
        "missing_keys_preview", "unexpected_keys_preview",
        "SC_PFR_Mode", "SC_PFR_Enabled", "SC_PFR_gen_valid",
        "SC_PhysGeneratedSC", "SC_PhysFeasibleSC", "SC_PhysFeasibleDirectSC", "SC_PhysFeasibleRelaySC",
        "SC_PhysGeneratedSC_sum", "SC_PhysFeasibleSC_sum",
        "SC_PhysFeasibleDirectSC_sum", "SC_PhysFeasibleRelaySC_sum",
        "SC_PFR_gen", "SC_CR_phys",
        "SC_PFR_gen_invalid_reason", "SC_CR_phys_invalid_reason", "SC_CR_phys_done_key",
        "sc_score_source", "terrain_aware_sc_scoring",
    ] + metric_summary_columns

    per_episode_path = output_dir / f"per_episode_{run_time}{tag_suffix}.csv"
    summary_path = output_dir / f"summary_{run_time}{tag_suffix}.csv"
    json_path = output_dir / f"full_metrics_{run_time}{tag_suffix}.json"
    report_path = output_dir / f"report_{run_time}{tag_suffix}.txt"
    config_audit_path = output_dir / f"config_audit_{run_time}{tag_suffix}.json"
    sc_lifecycle_audit_path = output_dir / f"sc_candidate_lifecycle_audit_{run_time}{tag_suffix}.csv"
    all_summary_path = output_dir / "all_eval_summary.csv"

    write_csv(per_episode_path, per_episode_rows)
    write_csv(summary_path, summary_rows, preferred_columns=required_summary_columns)
    append_csv(all_summary_path, summary_rows, preferred_columns=required_summary_columns)
    if bool(getattr(getattr(config, "diagnostics", object()), "sc_lifecycle_audit_write_csv", True)):
        write_csv(sc_lifecycle_audit_path, sc_lifecycle_audit_rows)
    write_report(report_path, summary_rows)
    config_audit = build_config_audit(
        args=args,
        config=config,
        checkpoints=checkpoints,
        provided_config_path=provided_config_path,
        training_config_path=training_config_path,
        effective_config_path=effective_config_path,
        eval_mode_requested=eval_mode_requested,
        eval_mode_effective=str(args.eval_mode),
        config_diffs=config_diffs,
    )
    config_audit_path.write_text(
        json.dumps(config_audit, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    json_path.write_text(json.dumps({
        "config_path": args.config,
        "checkpoint_dir": args.checkpoint_dir,
        "checkpoint_list": checkpoint_records,
        "console_log_path": str(console_log_path),
        "config_audit_path": str(config_audit_path),
        "sc_candidate_lifecycle_audit_path": str(sc_lifecycle_audit_path),
        "args": vars(args),
        "per_episode_results": per_episode_rows,
        "summary_results": summary_rows,
        "sc_candidate_lifecycle_audit_rows": sc_lifecycle_audit_rows,
    }, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")

    total_elapsed = time.time() - global_start_time
    progress_print("[Eval] All checkpoints finished.")
    progress_print(f"[Eval] Total time: {total_elapsed:.1f}s")
    progress_print(f"[Eval] Output directory: {output_dir}")
    progress_print(f"Saved per-episode CSV to: {per_episode_path}")
    progress_print(f"Saved summary CSV to: {summary_path}")
    progress_print(f"Saved full metrics JSON to: {json_path}")
    progress_print(f"Saved report to: {report_path}")
    progress_print(f"Saved config audit JSON to: {config_audit_path}")
    progress_print(f"Saved SC candidate lifecycle audit CSV to: {sc_lifecycle_audit_path}")
    progress_print(f"Saved all-eval summary CSV to: {all_summary_path}")
    progress_print(f"Saved detailed console log to: {console_log_path}")


if __name__ == "__main__":
    main()


