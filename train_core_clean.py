"""
干净消融/主模型训练核心（支持高层 BC + 共享模块加载）
"""
import argparse
import inspect
import os
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from utils.config import Config
from utils.training_diagnostics import force_training_safe_diagnostics
from utils.logger import Logger, MetricsTracker
from envs.disaster_env import DisasterRescueEnv
from envs.vec_env import make_vec_env
from algorithms.mappo_clean_ablation import MAPPO, HierarchicalMAPPO

MODE_LABELS = {
    'full': 'MAHPPO 主模型',
    'no_icn': 'A2: w/o ICN',
    'no_icn_gnn_caa': 'A4: w/o ICN + GNN + CAA',
}


def parse_args(description: str):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--config', type=str, default='configs/default_config.yaml')
    parser.add_argument('--experiment_name', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    return parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def safe_mean_reward_from_episode_rewards(episode_rewards, use_hierarchical: bool):
    if use_hierarchical:
        high_rewards = episode_rewards.get('high', {}) if isinstance(episode_rewards, dict) else {}
        low_rewards = episode_rewards.get('low', {}) if isinstance(episode_rewards, dict) else {}
        high_total = sum(high_rewards.values()) / len(high_rewards) if high_rewards else 0.0
        low_total = sum(low_rewards.values()) / len(low_rewards) if low_rewards else 0.0
        total_reward = high_total + low_total
    else:
        high_total = 0.0
        low_total = 0.0
        total_reward = sum(episode_rewards.values()) / len(episode_rewards) if episode_rewards else 0.0
    return high_total, low_total, total_reward


def build_hierarchical_algo(config: Config, device: torch.device, ablation_type: str = None):
    sig = inspect.signature(HierarchicalMAPPO)
    if 'ablation_type' in sig.parameters:
        return HierarchicalMAPPO(config, device, ablation_type=ablation_type)
    if ablation_type is None:
        return HierarchicalMAPPO(config, device)
    raise RuntimeError('当前 clean 版 HierarchicalMAPPO 不支持 ablation_type 参数。')


def _safe_get_config_value(config: Config, key: str, default=None):
    if hasattr(config, '_config') and isinstance(config._config, dict):
        return config._config.get(key, default)
    return default


def _extract_state_dict_payload(pretrained_obj: Any, candidate_keys) -> Tuple[Dict[str, torch.Tensor], str]:
    if isinstance(pretrained_obj, dict):
        for key in candidate_keys:
            state_dict = pretrained_obj.get(key)
            if isinstance(state_dict, dict):
                return state_dict, key
        if all(isinstance(k, str) for k in pretrained_obj.keys()):
            tensor_like = [v for v in pretrained_obj.values() if torch.is_tensor(v)]
            if tensor_like:
                return pretrained_obj, 'root_state_dict'
    raise ValueError(f'无法从预训练文件中解析 state_dict，候选键: {candidate_keys}')


def _log_incompatible_keys(logger: Logger, missing, unexpected, loaded_count: int, total_count: int, tag: str):
    logger.log(f'{tag} 成功加载 {loaded_count}/{total_count} 个预训练权重')
    if missing:
        logger.log(f'  {tag} 缺失的权重: {list(missing)}')
    if unexpected:
        logger.log(f'  {tag} 未预期的权重: {list(unexpected)}')


def maybe_enable_compile(mappo, config: Config, logger: Logger):
    use_compile = bool(_safe_get_config_value(config, 'use_compile', False))
    has_compile = hasattr(torch, 'compile')
    logger.log(f'torch.compile 配置: use_compile={use_compile}, has_compile={has_compile}')
    if use_compile and has_compile:
        try:
            logger.log('启用 torch.compile 加速.')
            if hasattr(mappo, 'agent') and hasattr(mappo.agent, 'gnn'):
                mappo.agent.gnn = torch.compile(mappo.agent.gnn, mode='reduce-overhead')
            if hasattr(mappo, 'agent') and hasattr(mappo.agent, 'actor'):
                mappo.agent.actor = torch.compile(mappo.agent.actor, mode='reduce-overhead')
            logger.log('torch.compile 加速已启用')
        except Exception as e:
            logger.log(f'torch.compile 启用失败: {e}')
    elif use_compile and not has_compile:
        logger.log('警告: use_compile=true 但当前 PyTorch 不支持 torch.compile')


def maybe_load_pretrained_high_level(mappo, config: Config, device: torch.device, logger: Logger):
    pretrain_high_level_path = _safe_get_config_value(config, 'pretrain_high_level', None)
    if not pretrain_high_level_path:
        logger.log('未配置 pretrain_high_level，按从零开始训练高层策略。')
        return
    if not os.path.exists(pretrain_high_level_path):
        logger.log(f'警告: pretrain_high_level 指向的文件不存在: {pretrain_high_level_path}')
        return
    if not hasattr(mappo, 'agent') or not hasattr(mappo.agent, 'actor') or not hasattr(mappo.agent.actor, 'high_level_policy'):
        logger.log('警告: 当前模型不包含 high_level_policy，跳过高层 BC 预训练加载')
        return

    logger.log(f'加载高层 BC 预训练模型: {pretrain_high_level_path}')
    try:
        pretrained = torch.load(pretrain_high_level_path, map_location=device)
        high_level_state_dict, source_key = _extract_state_dict_payload(
            pretrained,
            candidate_keys=(
                'high_level_state_dict',
                'high_level_policy_state_dict',
                'actor_high_level_state_dict',
                'state_dict',
                'model_state_dict',
            )
        )
        high_level_policy = mappo.agent.actor.high_level_policy
        incompatible = high_level_policy.load_state_dict(high_level_state_dict, strict=False)
        loaded_count = len(high_level_state_dict) - len(incompatible.missing_keys)
        _log_incompatible_keys(
            logger,
            incompatible.missing_keys,
            incompatible.unexpected_keys,
            loaded_count,
            len(high_level_state_dict),
            f'高层预训练({source_key})'
        )

        shared = pretrained.get('shared_modules_state_dict') if isinstance(pretrained, dict) else None
        if isinstance(shared, dict):
            if 'gnn' in shared and shared['gnn'] is not None and hasattr(mappo.agent, 'gnn'):
                inc = mappo.agent.gnn.load_state_dict(shared['gnn'], strict=False)
                _log_incompatible_keys(logger, inc.missing_keys, inc.unexpected_keys,
                                       len(shared['gnn']) - len(inc.missing_keys), len(shared['gnn']), '共享模块(gnn)')
            if 'resource_predictor' in shared and shared['resource_predictor'] is not None and hasattr(mappo.agent, 'resource_predictor'):
                inc = mappo.agent.resource_predictor.load_state_dict(shared['resource_predictor'], strict=False)
                _log_incompatible_keys(logger, inc.missing_keys, inc.unexpected_keys,
                                       len(shared['resource_predictor']) - len(inc.missing_keys), len(shared['resource_predictor']), '共享模块(resource_predictor)')
            if 'intent_comm' in shared and shared['intent_comm'] is not None and getattr(mappo.agent, 'intent_comm', None) is not None:
                inc = mappo.agent.intent_comm.load_state_dict(shared['intent_comm'], strict=False)
                _log_incompatible_keys(logger, inc.missing_keys, inc.unexpected_keys,
                                       len(shared['intent_comm']) - len(inc.missing_keys), len(shared['intent_comm']), '共享模块(intent_comm)')
            if 'capability_attention' in shared and shared['capability_attention'] is not None and getattr(mappo.agent, 'capability_attention', None) is not None:
                inc = mappo.agent.capability_attention.load_state_dict(shared['capability_attention'], strict=False)
                _log_incompatible_keys(logger, inc.missing_keys, inc.unexpected_keys,
                                       len(shared['capability_attention']) - len(inc.missing_keys), len(shared['capability_attention']), '共享模块(capability_attention)')

        high_level_lr_scale = float(_safe_get_config_value(config, 'high_level_lr_scale', 1.0))
        if high_level_lr_scale < 1.0 and hasattr(mappo, 'high_level_optimizer'):
            for param_group in mappo.high_level_optimizer.param_groups:
                param_group['lr'] *= high_level_lr_scale
            logger.log(f'高层学习率缩放: {high_level_lr_scale}')

        if hasattr(config, '_config') and isinstance(config._config, dict):
            config._config['bc_initialized'] = True
            config._config['bc_source'] = str(pretrain_high_level_path)
    except Exception as e:
        logger.log(f'加载高层 BC 预训练模型失败: {e}')


def evaluate(mappo, config: Config, device: torch.device, n_episodes: int = None) -> dict:
    if n_episodes is None:
        n_episodes = config.logging.num_eval_episodes
    env = DisasterRescueEnv(config)
    episode_rewards, episode_lengths = [], []
    tasks_completed_list, tasks_expired_list, tasks_generated_list = [], [], []
    for ep in range(n_episodes):
        observations, info = env.reset(seed=config.seed + ep)
        global_state = env.get_global_state()
        episode_reward, episode_length = 0.0, 0
        done = False
        info = {}
        while not done:
            action_masks = env.get_all_action_masks()
            actions = mappo.get_action(observations, global_state, action_masks, deterministic=True)
            observations, rewards, terminated, truncated, info = env.step(actions)
            global_state = env.get_global_state()
            for r in rewards.values():
                episode_reward += float(r.total) if hasattr(r, 'total') else float(r)
            episode_length += 1
            done = all(terminated.values()) or bool(truncated)
            if episode_length >= int(config.environment.max_steps):
                break
        completed = float(info.get('total_tasks_completed', 0.0))
        expired = float(info.get('total_tasks_expired', 0.0))
        generated = float(info.get('total_tasks_generated', getattr(env, '_next_task_id', max(1.0, completed + expired))))
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        tasks_completed_list.append(completed)
        tasks_expired_list.append(expired)
        tasks_generated_list.append(generated)
    env.close()
    mean_generated = float(np.mean(tasks_generated_list)) if tasks_generated_list else 1.0
    mean_completed = float(np.mean(tasks_completed_list)) if tasks_completed_list else 0.0
    mean_expired = float(np.mean(tasks_expired_list)) if tasks_expired_list else 0.0
    return {
        'mean_reward': float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        'std_reward': float(np.std(episode_rewards)) if episode_rewards else 0.0,
        'mean_episode_length': float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        'mean_tasks_completed': mean_completed,
        'mean_tasks_expired': mean_expired,
        'mean_tasks_generated': mean_generated,
        'mean_completion_rate': float(mean_completed / max(1.0, mean_generated)),
        'mean_time_window_satisfaction': float(1.0 - mean_expired / max(1.0, mean_generated)),
    }




def _get_curriculum_stages(config: Config):
    if hasattr(config, '_config') and isinstance(config._config, dict):
        curriculum = config._config.get('curriculum', {})
        if isinstance(curriculum, dict) and curriculum.get('enabled', False):
            stages = curriculum.get('stages', [])
            if isinstance(stages, list):
                return stages
    return []


def _set_runtime_value(section_obj, section_dict, key: str, value):
    try:
        setattr(section_obj, key, value)
    except Exception:
        pass
    if isinstance(section_dict, dict):
        section_dict[key] = value


def _apply_curriculum_stage(config: Config, env, stage_cfg: dict, stage_index: int, logger: Logger):
    reward_cfg_dict = {}
    task_cfg_dict = {}
    if hasattr(config, '_config') and isinstance(config._config, dict):
        reward_cfg_dict = config._config.setdefault('reward', {})
        task_cfg_dict = config._config.setdefault('task', {})

    applied_items = []

    reward_updates = stage_cfg.get('reward', {}) if isinstance(stage_cfg, dict) else {}
    if isinstance(reward_updates, dict):
        for key, value in reward_updates.items():
            _set_runtime_value(config.reward, reward_cfg_dict, key, value)
            if env is not None and hasattr(env, 'reward_config'):
                _set_runtime_value(env.reward_config, reward_cfg_dict, key, value)
            applied_items.append(f"reward.{key}={value}")

    task_updates = stage_cfg.get('task', {}) if isinstance(stage_cfg, dict) else {}
    if isinstance(task_updates, dict):
        for key, value in task_updates.items():
            _set_runtime_value(config.task, task_cfg_dict, key, value)
            if env is not None and hasattr(env, 'config') and hasattr(env.config, 'task'):
                _set_runtime_value(env.config.task, task_cfg_dict, key, value)
            applied_items.append(f"task.{key}={value}")

    if env is not None and hasattr(env, '_invalidate_derived_caches'):
        try:
            env._invalidate_derived_caches()
        except Exception:
            pass

    max_ts = stage_cfg.get('max_timesteps', 'N/A') if isinstance(stage_cfg, dict) else 'N/A'
    logger.log(f'Curriculum 切换到阶段 {stage_index + 1} (<= {max_ts} steps): ' + ', '.join(applied_items))


def _maybe_apply_curriculum(config: Config, env, total_timesteps: int, current_stage_idx: int, logger: Logger):
    stages = _get_curriculum_stages(config)
    if not stages:
        return current_stage_idx

    target_idx = current_stage_idx
    for idx, stage in enumerate(stages):
        max_ts = int(stage.get('max_timesteps', 10**18))
        if total_timesteps <= max_ts:
            target_idx = idx
            break
        target_idx = idx

    if target_idx != current_stage_idx:
        _apply_curriculum_stage(config, env, stages[target_idx], target_idx, logger)
        return target_idx
    return current_stage_idx


def train(config: Config, logger: Logger, device: torch.device, mode: str, run_name: str):
    use_hierarchical = mode != 'standard'
    ablation_type = None if mode == 'full' else mode
    num_envs = config.training.num_envs
    use_vec_env = num_envs > 1 and not use_hierarchical
    if use_vec_env:
        vec_env = make_vec_env(DisasterRescueEnv, config, num_envs, use_subproc=True)
        env = None
    else:
        env = DisasterRescueEnv(config)
        env._debug_low_level = False
        vec_env = None

    if use_hierarchical:
        logger.log(f'实验模式：{MODE_LABELS[mode]}')
        mappo = build_hierarchical_algo(config, device, ablation_type=ablation_type)
    else:
        logger.log('实验模式：标准 MAPPO')
        mappo = MAPPO(config, device)

    maybe_enable_compile(mappo, config, logger)
    if use_hierarchical:
        maybe_load_pretrained_high_level(mappo, config, device, logger)

    total_timesteps = int(config.training.total_timesteps)
    log_interval = int(config.logging.log_interval)
    save_interval = int(config.logging.save_interval)
    rollout_length = int(config.training.rollout_length)
    metrics = MetricsTracker(window_size=100)
    start_time = time.time()
    n_rollouts = 0
    eval_every_timesteps = int(config.logging.eval_interval)
    next_eval_step = eval_every_timesteps
    next_save_step = max(1, save_interval)

    curriculum_stages = _get_curriculum_stages(config)
    current_curriculum_stage = -1
    if not use_vec_env and curriculum_stages:
        current_curriculum_stage = _maybe_apply_curriculum(
            config, env, total_timesteps=0, current_stage_idx=current_curriculum_stage, logger=logger
        )
    elif use_vec_env and curriculum_stages:
        logger.log('警告: 当前为 vec_env 模式，暂不对多环境动态切换 curriculum。')

    try:
        while mappo.total_timesteps < total_timesteps:
            if not use_vec_env and curriculum_stages:
                current_curriculum_stage = _maybe_apply_curriculum(
                    config, env, total_timesteps=mappo.total_timesteps,
                    current_stage_idx=current_curriculum_stage, logger=logger
                )
            if use_vec_env:
                _, rollout_info = mappo.collect_rollouts_vec(vec_env, rollout_length)
            else:
                _, rollout_info = mappo.collect_rollouts(env, rollout_length)
            n_rollouts += 1
            _, _, total_reward = safe_mean_reward_from_episode_rewards(rollout_info['episode_rewards'], use_hierarchical)
            metrics.add('episode_reward', total_reward)
            metrics.add('episode_length', rollout_info.get('episode_length', 0))
            train_info = mappo.train()
            logger.log_scalar('rollout/reward_mean', metrics.get_mean('episode_reward'), mappo.total_timesteps)
            logger.log_scalar('rollout/reward_std', metrics.get_std('episode_reward'), mappo.total_timesteps)
            logger.log_scalar('rollout/episode_length', metrics.get_mean('episode_length'), mappo.total_timesteps)
            if isinstance(train_info, dict):
                for k, v in train_info.items():
                    if isinstance(v, (int, float)):
                        logger.log_scalar(f'train/{k}', float(v), mappo.total_timesteps)
            if n_rollouts % max(1, log_interval) == 0:
                elapsed_time = max(time.time() - start_time, 1e-6)
                fps = mappo.total_timesteps / elapsed_time
                logger.log(
                    f"Rollout {n_rollouts} | Timesteps: {mappo.total_timesteps}/{total_timesteps} | FPS: {fps:.0f} | "
                    f"Reward: {metrics.get_mean('episode_reward'):.2f} +/- {metrics.get_std('episode_reward'):.2f}"
                )
            while mappo.total_timesteps >= next_eval_step:
                eval_info = evaluate(mappo, config, device)
                logger.log_evaluation(
                    step=next_eval_step,
                    mean_reward=eval_info['mean_reward'],
                    std_reward=eval_info['std_reward'],
                    mean_tasks_completed=eval_info['mean_tasks_completed'],
                    mean_completion_rate=eval_info['mean_completion_rate'],
                    mean_time_window_satisfaction=eval_info['mean_time_window_satisfaction'],
                )
                logger.log(
                    f"  EvalDetail: mean_episode_length={eval_info['mean_episode_length']:.2f}, "
                    f"mean_tasks_generated={eval_info['mean_tasks_generated']:.2f}, "
                    f"mean_tasks_completed={eval_info['mean_tasks_completed']:.2f}, "
                    f"mean_tasks_expired={eval_info['mean_tasks_expired']:.2f}"
                )
                next_eval_step += eval_every_timesteps
            while mappo.total_timesteps >= next_save_step:
                ckpt_name = f'{run_name}_checkpoint_{next_save_step}.pt'
                save_path = Path(config.logging.save_dir) / ckpt_name
                save_path.parent.mkdir(parents=True, exist_ok=True)
                mappo.save(str(save_path))
                logger.log(f'保存检查点: {save_path}')
                next_save_step += max(1, save_interval)
        final_name = f'{run_name}_final.pt'
        final_path = Path(config.logging.save_dir) / final_name
        final_path.parent.mkdir(parents=True, exist_ok=True)
        mappo.save(str(final_path))
        logger.log(f'训练完成! 最终模型保存至: {final_path}')
    finally:
        logger.close()
        if use_vec_env and vec_env is not None:
            vec_env.close()
        elif env is not None:
            env.close()


def run_main(mode: str, description: str):
    args = parse_args(description)
    config = Config(args.config)
    if args.seed is not None:
        config.seed = args.seed
    if args.device is not None:
        config.device = args.device
    if args.resume is not None:
        config.resume_path = args.resume
    if hasattr(config, '_config') and isinstance(config._config, dict):
        config._config['debug_low_level'] = False
    force_training_safe_diagnostics(config)
    set_seed(config.seed)
    device = torch.device('cuda' if config.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    env_name_suffix = (
        f"u{config.environment.num_uavs}_t{config.environment.max_tasks}_so{config.environment.num_static_obstacles}_do{config.environment.num_dynamic_obstacles}"
    )
    algorithm_name = 'mahppo' if mode == 'full' else mode
    experiment_name = args.experiment_name or f'{algorithm_name}_{env_name_suffix}'
    config.logging.save_dir = str(Path(config.logging.save_dir) / experiment_name)
    logger = Logger(
        log_dir=config.logging.log_dir,
        experiment_name=experiment_name,
        use_tensorboard=config.logging.use_tensorboard,
        use_wandb=config.logging.use_wandb,
        wandb_config=config.to_dict() if config.logging.use_wandb else None,
    )
    config.save(str(Path(logger.log_dir) / 'config.yaml'))
    print(f'实验角色：{MODE_LABELS[mode]}')
    print(f'种子: {config.seed}  保存目录: {config.logging.save_dir}')
    train(config, logger, device, mode=mode, run_name=experiment_name)


if __name__ == '__main__':
    run_main('full', '训练 MAHPPO / clean 消融主训练脚本（支持高层 BC + 共享模块加载）')
