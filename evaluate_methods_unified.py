"""
Unified evaluation runner for the eight SC-comparison methods.

This script evaluates learned checkpoint methods and non-learning heuristic
methods under the same DisasterRescueEnv metrics pipeline.  It is intentionally
evaluation-only; training remains in the existing train_*.py entrypoints.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from utils.config import Config, yaml
from envs.disaster_env import DisasterRescueEnv

LEARNED_METHOD_TYPES = {
    "mahppo",
    "mahppo_no_icn",
    "mahppo_icn_wo_nash",
    "mappo",
    "mappo_rule_low_baseline_semantic",
    "mappo_gnn_caa_clean_no_icn_protocol",
    "happo",
    "maddpg",
    "maddpg_rule_low_baseline_semantic",
}


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if abs(float(den)) > 1e-12 else 0.0


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


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if torch.is_tensor(obj):
        return obj.item() if obj.numel() == 1 else obj.detach().cpu().tolist()
    return str(obj)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], preferred_columns: Optional[Iterable[str]] = None) -> None:
    if not rows:
        return
    columns: List[str] = []
    for key in preferred_columns or []:
        if key not in columns:
            columns.append(key)
    for row in rows:
        for key in row.keys():
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def metric_value(info: Dict[str, Any], key: str, default: float = 0.0) -> float:
    value = info.get(key, default)
    if isinstance(value, (int, float, np.number)):
        return float(value)
    return float(default)


def get_episode_metrics(info: Dict[str, Any], episode_reward: float, episode_length: int, config: Config) -> Dict[str, Any]:
    generated = metric_value(info, "total_tasks_generated")
    completed = metric_value(info, "total_tasks_completed")
    expired = metric_value(info, "total_tasks_expired")
    selected = metric_value(info, "selected_tasks")
    completed_selected = metric_value(info, "completed_selected_tasks")
    gen_sc = metric_value(info, "sc_tasks_generated")
    selected_sc = metric_value(info, "selected_sc_tasks")
    completed_selected_sc = metric_value(info, "completed_selected_sc_tasks")
    done_sc = metric_value(info, "sc_tasks_completed")
    upload_checks = metric_value(info, "terrain_upload_checks")
    upload_success = metric_value(info, "terrain_upload_success")
    backhaul_checks = metric_value(info, "terrain_backhaul_checks")
    backhaul_success = metric_value(info, "terrain_backhaul_success")
    num_uavs = max(1.0, float(getattr(config.environment, "num_uavs", 1)))
    switch_count = metric_value(info, "commitment_switch_count")

    row: Dict[str, Any] = {
        "TCR_gen": safe_div(completed, max(generated, 1.0)),
        "TCR_sel": safe_div(completed_selected, max(selected, 1.0)),
        "TCR_res": safe_div(completed_selected, max(selected, 1.0)),
        "SC_CR": safe_div(done_sc, max(gen_sc, 1.0)),
        "SC_sel": safe_div(completed_selected_sc, max(selected_sc, 1.0)),
        "Done": completed,
        "Expired": expired,
        "DoneS": metric_value(info, "s_tasks_completed"),
        "DoneC": metric_value(info, "c_tasks_completed"),
        "DoneSC": done_sc,
        "ActiveSC": metric_value(info, "active_commit_sc_tasks"),
        "CommitWriteSuccess": metric_value(info, "sc_commit_write_success"),
        "PairSuccessSC": metric_value(info, "pair_formation_success"),
        "BackhaulFailActive": metric_value(info, "backhaul_fail_active_commit"),
        "BackhaulSuccessRate": safe_div(backhaul_success, max(backhaul_checks, 1.0)),
        "BackhaulSuccessCount": backhaul_success,
        "BackhaulAttemptCount": backhaul_checks,
        "UploadSuccessRate": safe_div(upload_success, max(upload_checks, 1.0)),
        "SwitchTaskChange": metric_value(info, "switch_task_change", switch_count),
        "SwitchUAV": safe_div(switch_count, num_uavs),
        "episode_reward": float(episode_reward),
        "episode_length": float(episode_length),
        "AvgDone": completed,
        "AvgExpired": expired,
        "AvgReward": float(episode_reward),
        "SafetyEvents": metric_value(info, "total_obstacle_safety_events"),
        "ActualCollisions": metric_value(info, "total_obstacle_actual_collisions", metric_value(info, "total_collisions")),
        "Generated_SC": gen_sc,
        "Completed_SC": done_sc,
        "Feasible_Generated_SC": metric_value(info, "SC_PhysFeasibleSC", np.nan),
        "Completed_Feasible_SC": np.nan,
        "SC_PFR_gen": metric_value(info, "SC_PFR_gen", np.nan),
        "SC_CR_phys": metric_value(info, "SC_CR_phys", np.nan),
        "GenSC_Unique": gen_sc,
        "SelSC_Unique": selected_sc,
        "DoneSC_Unique": done_sc,
        "ExpSC_Unique": metric_value(info, "sc_tasks_expired"),
        "ReleaseSCCandidateTimeout": metric_value(info, "action_mask_release_sc_candidate_timeout"),
    }
    feasible_sc = row.get("Feasible_Generated_SC")
    sc_cr_phys = row.get("SC_CR_phys")
    if np.isfinite(feasible_sc) and np.isfinite(sc_cr_phys):
        row["Completed_Feasible_SC"] = float(feasible_sc) * float(sc_cr_phys)
    for key, value in info.items():
        if isinstance(value, (int, float, np.number)) and key not in row:
            row[key] = float(value)
    return row


def extract_sc_funnel_metrics(env: DisasterRescueEnv, info: Dict[str, Any], episode_length: int, config: Config) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    for key, value in info.items():
        if isinstance(value, (int, float, np.number)):
            row[key] = float(value)
    row.setdefault("SCSelectRate_Unique", safe_div(row.get("selected_sc_tasks", 0.0), max(row.get("sc_tasks_generated", 0.0), 1.0)))
    row.setdefault("SCCompleteSel", safe_div(row.get("completed_selected_sc_tasks", 0.0), max(row.get("selected_sc_tasks", 0.0), 1.0)))
    row.setdefault("PairAttemptSC_Events", row.get("pair_attempt_sc_events", 0.0))
    row.setdefault("PairSuccessSC_Events", row.get("pair_success_sc_events", 0.0))
    return row


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
        "tag": getattr(args, "tag", ""),
        "config_path": config_path,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_name": checkpoint_path.name,
        "checkpoint_step": int(checkpoint_step),
        "eval_mode": eval_mode,
        "episodes": int(getattr(args, "episodes", len(rows))),
        "seed": int(getattr(args, "seed", 42)),
        "mode": getattr(args, "mode", "unified"),
    }
    numeric_keys = sorted({k for row in rows for k, v in row.items() if isinstance(v, (int, float, np.number))})
    for key in numeric_keys:
        values = np.asarray([float(row.get(key, 0.0)) for row in rows], dtype=np.float64)
        summary[f"{key}_mean"] = float(np.mean(values)) if values.size else 0.0
        summary[f"{key}_std"] = float(np.std(values)) if values.size else 0.0
        summary[f"{key}_min"] = float(np.min(values)) if values.size else 0.0
        summary[f"{key}_max"] = float(np.max(values)) if values.size else 0.0
    return summary

RULE_METHOD_TYPES = {"hungarian", "genetic"}

DEFAULT_SUMMARY_COLUMNS = [
    "method_id",
    "method_name",
    "method_type",
    "checkpoint_name",
    "checkpoint_step",
    "eval_mode",
    "episodes",
    "TCR_gen_mean",
    "TCR_sel_mean",
    "SC_CR_mean",
    "AvgDone_mean",
    "AvgExpired_mean",
    "AvgReward_mean",
    "SafetyEvents_mean",
    "ActualCollisions_mean",
    "ResponseTime_mean",
    "SC_PFR_gen_mean",
    "SC_CR_phys_mean",
    "Generated_SC_mean",
    "Completed_SC_mean",
    "Feasible_Generated_SC_mean",
    "Completed_Feasible_SC_mean",
    "SC_sel_mean",
    "DoneSC_mean",
    "Expired_mean",
    "BackhaulSuccessRate_mean",
    "UploadSuccessRate_mean",
    "ReleaseSCCandidateTimeout_mean",
    "SCAudit_TotalCandidates_mean",
    "SCAudit_SWithinRadiusButReleased_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate all SC comparison methods with one metric schema.")
    parser.add_argument("--matrix", type=str, default="configs/experiment_matrix_sc_8methods.yaml")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--last_n", type=int, default=None)
    parser.add_argument("--eval_mode", choices=("stochastic", "deterministic", "both"), default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--methods", type=str, default=None, help="Comma-separated method ids to run.")
    parser.add_argument("--allow_missing", action="store_true", help="Skip missing checkpoint dirs/files instead of failing.")
    parser.add_argument("--progress_interval", type=int, default=10, help="Print evaluation progress every N episodes.")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_matrix(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Experiment matrix must be a YAML mapping: {path}")
    return data


def checkpoint_step_from_name(path: Path) -> int:
    match = re.search(r"checkpoint_(\d+)", path.name)
    return int(match.group(1)) if match else -1


def select_checkpoints(method: Dict[str, Any], last_n: int, allow_missing: bool) -> List[Path]:
    checkpoint = method.get("checkpoint")
    if checkpoint:
        path = Path(str(checkpoint))
        if not path.exists():
            if allow_missing:
                print(f"[Skip] Missing checkpoint for {method.get('id')}: {path}")
                return []
            raise FileNotFoundError(path)
        return [path]

    checkpoint_dir = method.get("checkpoint_dir")
    if not checkpoint_dir:
        return []
    root = Path(str(checkpoint_dir))
    if not root.exists():
        if allow_missing:
            print(f"[Skip] Missing checkpoint_dir for {method.get('id')}: {root}")
            return []
        raise FileNotFoundError(root)
    pattern = str(method.get("pattern", "*_checkpoint_*.pt"))
    candidates = [p for p in root.glob(pattern) if p.is_file()]
    if not candidates:
        if allow_missing:
            print(f"[Skip] No checkpoints matched {pattern} for {method.get('id')}: {root}")
            return []
        raise FileNotFoundError(f"No checkpoints matched {pattern} in {root}")
    candidates = sorted(candidates, key=lambda p: (checkpoint_step_from_name(p), p.name))
    return candidates[-max(1, int(last_n)) :]


def apply_method_overrides(config: Config, method: Dict[str, Any], seed: int) -> None:
    config.seed = int(seed)
    overrides = method.get("config_overrides", {}) or {}
    if not isinstance(overrides, dict):
        return
    raw = getattr(config, "_config", None)
    if not isinstance(raw, dict):
        return
    for dotted_key, value in overrides.items():
        parts = str(dotted_key).split(".")
        cursor = raw
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
            if not isinstance(cursor, dict):
                break
        else:
            cursor[parts[-1]] = value
    if overrides and hasattr(config, "_parse_config"):
        config._parse_config()


def build_policy(method_type: str, config: Config, device: torch.device, checkpoint: Optional[Path]):
    if method_type in ("mahppo", "mahppo_no_icn", "mahppo_icn_wo_nash"):
        from algorithms.mappo_rule_low_fpsfix_v21 import HierarchicalMAPPO

        ablation = {
            "mahppo": None,
            "mahppo_no_icn": "no_icn",
            "mahppo_icn_wo_nash": "icn_wo_nash",
        }[method_type]
        policy = HierarchicalMAPPO(config, device, ablation_type=ablation)
        if checkpoint is not None:
            load_torch_policy_checkpoint(policy, checkpoint, device)
        return PolicyAdapter(policy)

    if method_type == "mappo":
        from algorithms.mappo_standard import MAPPOStandard

        policy = MAPPOStandard(config, device)
        if checkpoint is not None:
            policy.load(str(checkpoint))
        return PolicyAdapter(policy)

    if method_type == "mappo_rule_low_baseline_semantic":
        from algorithms.mappo_rule_low_baseline_semantic_fpsfix import MAPPORuleLowBaselineSemanticFPSFix

        policy = MAPPORuleLowBaselineSemanticFPSFix(config, device)
        if checkpoint is not None:
            policy.load(str(checkpoint))
        return PolicyAdapter(policy)

    if method_type == "mappo_gnn_caa_clean_no_icn_protocol":
        from algorithms.mappo_gnn_caa_clean_no_icn_protocol_fpsfix import HierarchicalMAPPO

        policy = HierarchicalMAPPO(config, device)
        if checkpoint is not None:
            policy.load(str(checkpoint))
        return PolicyAdapter(policy)

    if method_type == "happo":
        from algorithms.happo_rule_fast import HAPPORuleFast

        policy = HAPPORuleFast(config, device)
        if checkpoint is not None:
            policy.load(str(checkpoint))
        return HAPPOAdapter(policy)

    if method_type == "maddpg":
        from algorithms.maddpg import MADDPG

        policy = MADDPG(config, device)
        if checkpoint is not None:
            policy.load(str(checkpoint))
        return MADDPGAdapter(policy, config)

    if method_type == "maddpg_rule_low_baseline_semantic":
        from algorithms.maddpg_rule_low_baseline_semantic_fpsfix import MADDPGRuleLowBaselineSemanticFPSFix

        policy = MADDPGRuleLowBaselineSemanticFPSFix(config, device)
        if checkpoint is not None:
            policy.load(str(checkpoint))
        return MADDPGAdapter(policy, config)

    if method_type == "hungarian":
        from algorithms.hungarian import HungarianAlgorithm

        return PolicyAdapter(HungarianAlgorithm(config, device))

    if method_type == "genetic":
        from algorithms.genetic_algorithm import GeneticAlgorithm

        return PolicyAdapter(GeneticAlgorithm(config, device))

    raise ValueError(f"Unknown method_type: {method_type}")


def extract_state_dict(checkpoint_obj: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint_obj, dict):
        for key in (
            "model_state_dict",
            "state_dict",
            "agent_state_dict",
            "agent",
            "policy_state_dict",
            "mappo_state_dict",
        ):
            value = checkpoint_obj.get(key)
            if isinstance(value, dict):
                return value
        if all(isinstance(k, str) for k in checkpoint_obj.keys()) and any(torch.is_tensor(v) for v in checkpoint_obj.values()):
            return checkpoint_obj
    raise ValueError("Could not find a compatible state_dict in checkpoint.")


def load_torch_policy_checkpoint(policy: Any, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint_obj = torch.load(checkpoint_path, map_location=device)
    state_dict = extract_state_dict(checkpoint_obj)
    target = getattr(policy, "agent", policy)
    incompatible = target.load_state_dict(state_dict, strict=False)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    if missing or unexpected:
        print(
            f"[LoadWarning] {checkpoint_path.name}: "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )


class PolicyAdapter:
    def __init__(self, policy: Any):
        self.policy = policy

    def reset(self, env: DisasterRescueEnv, observations: Dict[int, Dict]) -> None:
        target = getattr(self.policy, "agent", self.policy)
        if callable(getattr(target, "eval", None)):
            target.eval()

    def act(self, observations, global_state, action_masks, deterministic: bool):
        return self.policy.get_action(observations, global_state, action_masks, deterministic=deterministic)


class HAPPOAdapter(PolicyAdapter):
    def act(self, observations, global_state, action_masks, deterministic: bool):
        self.policy.agent.eval()
        with torch.no_grad():
            actions, _, _ = self.policy.agent.act(observations, global_state, action_masks, deterministic=deterministic)
        return actions


class MADDPGAdapter(PolicyAdapter):
    def __init__(self, policy: Any, config: Config):
        super().__init__(policy)
        self.config = config
        self.target_obs_dim: Optional[int] = None

    def reset(self, env: DisasterRescueEnv, observations: Dict[int, Dict]) -> None:
        super().reset(env, observations)
        if self.target_obs_dim is None:
            sample = next(iter(observations.values()))
            self.target_obs_dim = len(flatten_observation(sample))

    def act(self, observations, global_state, action_masks, deterministic: bool):
        flat_observations = {
            agent_id: flatten_observation(obs, self.target_obs_dim)
            for agent_id, obs in observations.items()
        }
        return self.policy.get_action(flat_observations, global_state, action_masks, deterministic=deterministic)


def flatten_observation(obs: Any, target_dim: Optional[int] = None) -> np.ndarray:
    def flatten_item(item: Any) -> List[float]:
        if isinstance(item, dict):
            values: List[float] = []
            for key in sorted(item.keys()):
                values.extend(flatten_item(item[key]))
            return values
        if isinstance(item, (list, tuple)):
            values = []
            for child in item:
                values.extend(flatten_item(child))
            return values
        if isinstance(item, np.ndarray):
            return np.asarray(item, dtype=np.float32).reshape(-1).tolist()
        if isinstance(item, (int, float, bool, np.number)):
            return [float(item)]
        if item is None:
            return [0.0]
        return []

    flat = np.asarray(flatten_item(obs), dtype=np.float32)
    if target_dim is not None:
        if flat.shape[0] < target_dim:
            flat = np.pad(flat, (0, target_dim - flat.shape[0]), mode="constant")
        elif flat.shape[0] > target_dim:
            flat = flat[:target_dim]
    return flat.astype(np.float32)


def evaluate_one_episode(
    env: DisasterRescueEnv,
    adapter: PolicyAdapter,
    config: Config,
    seed: int,
    episode_id: int,
    deterministic: bool,
) -> Dict[str, Any]:
    eval_seed = int(seed) + 100000 + int(episode_id)
    np.random.seed(eval_seed)
    torch.manual_seed(eval_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(eval_seed)

    observations, info = env.reset(seed=eval_seed)
    adapter.reset(env, observations)
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
        actions = adapter.act(observations, global_state, action_masks, deterministic=deterministic)
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
    return row


def evaluate_method_checkpoint(
    method: Dict[str, Any],
    config: Config,
    device: torch.device,
    checkpoint: Optional[Path],
    episodes: int,
    seed: int,
    eval_mode: str,
    progress_interval: int,
) -> List[Dict[str, Any]]:
    deterministic = eval_mode == "deterministic"
    adapter = build_policy(str(method["type"]), config, device, checkpoint)
    env = DisasterRescueEnv(config)
    rows: List[Dict[str, Any]] = []
    try:
        with torch.no_grad():
            for episode_id in range(int(episodes)):
                row = evaluate_one_episode(env, adapter, config, seed, episode_id, deterministic)
                row.update({
                    "method_id": method["id"],
                    "method_name": method.get("name", method["id"]),
                    "method_type": method["type"],
                    "checkpoint_path": "" if checkpoint is None else str(checkpoint),
                    "checkpoint_name": "heuristic" if checkpoint is None else checkpoint.name,
                    "checkpoint_step": -1 if checkpoint is None else checkpoint_step_from_name(checkpoint),
                    "eval_mode": eval_mode,
                })
                rows.append(row)
                if (episode_id + 1) == 1 or (episode_id + 1) % max(1, int(progress_interval)) == 0 or episode_id + 1 == int(episodes):
                    print(f"[Eval] {method['id']} {row['checkpoint_name']} {eval_mode}: {episode_id + 1}/{episodes}")
    finally:
        env.close()
    return rows


def summarize_by_group(rows: Sequence[Dict[str, Any]], matrix_args: Dict[str, Any], matrix_path: str) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("method_id", "")), str(row.get("checkpoint_name", "")), str(row.get("eval_mode", "")))
        groups.setdefault(key, []).append(row)

    summaries: List[Dict[str, Any]] = []
    dummy_args = argparse.Namespace(
        tag=matrix_args.get("tag", ""),
        seed=int(matrix_args.get("seed", 42)),
        mode="unified",
        episodes=0,
    )
    for (_, _, _), group_rows in groups.items():
        first = group_rows[0]
        dummy_args.episodes = len(group_rows)
        summary = summarize_rows(
            rows=group_rows,
            args=dummy_args,
            config_path=matrix_path,
            checkpoint_path=Path(str(first.get("checkpoint_path") or first.get("checkpoint_name", "heuristic"))),
            checkpoint_step=int(first.get("checkpoint_step", -1)),
            eval_mode=str(first.get("eval_mode", "")),
            run_time=str(matrix_args.get("run_time", "")),
        )
        summary.update({
            "method_id": first.get("method_id", ""),
            "method_name": first.get("method_name", ""),
            "method_type": first.get("method_type", ""),
        })
        summaries.append(summary)
    return summaries


def write_text_report(path: Path, summaries: Sequence[Dict[str, Any]]) -> None:
    lines = ["Unified 8-Method Evaluation", ""]
    for row in summaries:
        lines.append(f"Method: {row.get('method_name')} ({row.get('method_id')})")
        lines.append(f"Checkpoint: {row.get('checkpoint_name')} | Step: {row.get('checkpoint_step')} | Mode: {row.get('eval_mode')}")
        for key in ("TCR_gen", "TCR_sel", "SC_CR", "SC_sel", "DoneSC", "Expired", "BackhaulSuccessRate", "UploadSuccessRate"):
            lines.append(f"  {key}: {float(row.get(f'{key}_mean', 0.0)):.4f} +/- {float(row.get(f'{key}_std', 0.0)):.4f}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_method_commands(path: Path, matrix_path: Path, matrix: Dict[str, Any], output_dir: Path) -> None:
    default_config = str(matrix.get("default_config", "logs/MAHPPO_sc_cleanup_balance_v2/config.yaml"))
    lines = [
        "cd D:\\MAHPPO",
        f"& C:/Users/25658/.conda/envs/uav_final_gpu/python.exe evaluate_methods_unified.py --matrix {matrix_path.as_posix()} --output_dir {output_dir.as_posix()}",
        "",
        "# Individual learned-method examples:",
    ]
    for method in matrix.get("methods", []):
        if method.get("type") in RULE_METHOD_TYPES:
            continue
        lines.append(
            f"& C:/Users/25658/.conda/envs/uav_final_gpu/python.exe evaluate_methods_unified.py "
            f"--matrix {matrix_path.as_posix()} --methods {method.get('id')} --episodes {matrix.get('episodes', 100)}"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    matrix_path = Path(args.matrix)
    matrix = load_matrix(matrix_path)
    run_time = time.strftime("%Y%m%d_%H%M%S")
    matrix["run_time"] = run_time

    episodes = int(args.episodes if args.episodes is not None else matrix.get("episodes", 100))
    last_n = int(args.last_n if args.last_n is not None else matrix.get("last_n", 1))
    eval_mode = str(args.eval_mode if args.eval_mode is not None else matrix.get("eval_mode", "stochastic"))
    seed = int(args.seed if args.seed is not None else matrix.get("seed", 42))
    progress_interval = max(1, int(args.progress_interval))
    output_dir = Path(args.output_dir or matrix.get("output_dir", f"evaluation_results/sc_8methods_{run_time}"))
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    method_filter = None if not args.methods else {x.strip() for x in args.methods.split(",") if x.strip()}
    eval_modes = ["stochastic", "deterministic"] if eval_mode == "both" else [eval_mode]

    all_rows: List[Dict[str, Any]] = []
    run_records: List[Dict[str, Any]] = []
    methods = matrix.get("methods", [])
    if not isinstance(methods, list) or not methods:
        raise ValueError("Matrix must contain a non-empty methods list.")

    for method in methods:
        if method_filter is not None and method.get("id") not in method_filter:
            continue
        if not bool(method.get("enabled", True)):
            continue
        method_type = str(method.get("type", ""))
        if method_type not in LEARNED_METHOD_TYPES and method_type not in RULE_METHOD_TYPES:
            raise ValueError(f"Unsupported method type for {method.get('id')}: {method_type}")

        config_path = Path(str(method.get("config", matrix.get("default_config"))))
        config = Config(str(config_path))
        apply_method_overrides(config, method, seed)

        checkpoints = [None] if method_type in RULE_METHOD_TYPES else select_checkpoints(method, last_n, args.allow_missing)
        if not checkpoints:
            continue
        for checkpoint in checkpoints:
            for mode in eval_modes:
                print(f"[Start] {method.get('id')} | {checkpoint.name if checkpoint else 'heuristic'} | {mode} | episodes={episodes}")
                rows = evaluate_method_checkpoint(method, config, device, checkpoint, episodes, seed, mode, progress_interval)
                all_rows.extend(rows)
                run_records.append({
                    "method_id": method.get("id"),
                    "method_name": method.get("name"),
                    "method_type": method_type,
                    "checkpoint": None if checkpoint is None else str(checkpoint),
                    "eval_mode": mode,
                    "episodes": episodes,
                    "config": str(config_path),
                })

    summaries = summarize_by_group(all_rows, {**matrix, "episodes": episodes, "seed": seed}, str(matrix_path))

    per_episode_path = output_dir / f"per_episode_{run_time}.csv"
    summary_path = output_dir / f"summary_{run_time}.csv"
    report_path = output_dir / f"report_{run_time}.txt"
    json_path = output_dir / f"full_metrics_{run_time}.json"
    commands_path = output_dir / "rerun_commands.ps1"

    write_csv(per_episode_path, all_rows)
    write_csv(summary_path, summaries, preferred_columns=DEFAULT_SUMMARY_COLUMNS)
    write_text_report(report_path, summaries)
    write_method_commands(commands_path, matrix_path, matrix, output_dir)
    json_path.write_text(
        json.dumps(
            {
                "matrix_path": str(matrix_path),
                "run_time": run_time,
                "episodes": episodes,
                "last_n": last_n,
                "eval_mode": eval_mode,
                "seed": seed,
                "device": str(device),
                "runs": run_records,
                "summary_results": summaries,
                "per_episode_results": all_rows,
            },
            ensure_ascii=False,
            indent=2,
            default=json_default,
        ),
        encoding="utf-8",
    )

    print(f"Saved per-episode CSV to: {per_episode_path}")
    print(f"Saved summary CSV to: {summary_path}")
    print(f"Saved report to: {report_path}")
    print(f"Saved full metrics JSON to: {json_path}")
    print(f"Saved rerun commands to: {commands_path}")


if __name__ == "__main__":
    main()
