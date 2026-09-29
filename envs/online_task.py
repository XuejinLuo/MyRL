"""Training-only primitive reward and task-boundary adapter.

Observations/actions are passed through unchanged. Install INSIDE ChunkActionWrapper.
The benchmark must never install this adapter.
"""
import math
import gymnasium as gym


def task_protocol(cfg):
    options = cfg.get('online_task', {})
    mode = options.get('reward_mode', 'env')
    finite = options.get('timeout_semantics', 'continuing_bootstrap') == 'finite_horizon'
    protocol = dict(version=1, reward_mode=mode,
        success_reward=float(options.get('success_reward', 1.0)),
        end_on_success=bool(options.get('end_on_success', False)),
        timeout_semantics=options.get('timeout_semantics', 'continuing_bootstrap'),
        critic_use_remaining_time=bool(options.get('critic_use_remaining_time', False)),
        horizon=int(cfg.env.get('max_episode_steps', 300)),
        environment_reward_mode=options.get('environment_reward_mode'),
        gamma=float(cfg.algo.gamma), reward_scale=float(cfg.algo.reward_scale),
        selection=options.get('selection', 'success_then_env_reward'),
        critic_input_dim=int(cfg.model.cond_dim) + int(options.get('critic_use_remaining_time', False)),
        critic_input_format='frozen_features_remaining_time_v1' if options.get('critic_use_remaining_time', False) else 'frozen_features_v1')
    if mode not in ('env', 'success_once'):
        raise ValueError('online_task.reward_mode must be env or success_once')
    if protocol['timeout_semantics'] not in ('finite_horizon', 'continuing_bootstrap'):
        raise ValueError('Invalid online_task.timeout_semantics')
    if finite and not protocol['critic_use_remaining_time']:
        raise ValueError('finite_horizon requires critic_use_remaining_time=true')
    if mode == 'success_once' and not protocol['end_on_success']:
        raise ValueError('success_once requires end_on_success=true')
    if protocol['horizon'] < 1 or not math.isfinite(protocol['success_reward']) or protocol['success_reward'] <= 0:
        raise ValueError('Invalid task horizon/success_reward')
    if protocol['selection'] not in ('success_only', 'success_then_env_reward'):
        raise ValueError('Invalid online_task.selection')
    return protocol


def check_resume_protocol(checkpoint, current):
    from omegaconf import OmegaConf
    saved = checkpoint.get('online_protocol')
    if saved is None:
        # Unversioned online checkpoints have exactly the historical defaults.
        saved = task_protocol(OmegaConf.create(checkpoint['config']))
    differences = [k for k in set(saved) | set(current) if saved.get(k) != current.get(k)]
    if differences:
        raise ValueError('Online resume protocol mismatch: ' + ', '.join(sorted(differences)) +
            '. Set stages.online.resume=null and stages.online.initial_ckpt=<actor checkpoint> '
            'to start with a NEW Critic and optimizers.')


def critic_features(features, elapsed, protocol):
    if not protocol['critic_use_remaining_time']:
        return features
    import torch
    remaining = max(0., (protocol['horizon'] - elapsed) / protocol['horizon'])
    return torch.cat((features, features.new_full((features.shape[0], 1), remaining)), dim=-1)


def boundary_masks(terminated, truncated, protocol):
    terminal = terminated or (truncated and protocol['timeout_semantics'] == 'finite_horizon')
    return float(not terminal), float(not (terminated or truncated))


def task_phase(info):
    """Only reported task flags; missing flags stay unknown, not inferred from motion."""
    keys = ('success', 'is_grasped', 'is_cubeA_on_cubeB', 'is_cubeA_static')
    flags = {key: bool(info[key]) if key in info else None for key in keys}
    if flags['success']:
        phase = 'success'
    elif flags['is_cubeA_on_cubeB']:
        phase = 'stacked_grasped' if flags['is_grasped'] else 'stacked'
    elif flags['is_grasped']:
        phase = 'grasped'
    elif flags['is_grasped'] is not None:
        phase = 'ungrasped'
    else:
        phase = 'unknown'
    return phase, flags


class OnlineTaskWrapper(gym.Wrapper):
    def __init__(self, env, protocol):
        super().__init__(env)
        self.protocol = protocol
        self.elapsed = 0
        self.ended = False

    def reset(self, **kwargs):
        self.elapsed, self.ended = 0, False
        obs, info = self.env.reset(**kwargs)
        self.phase, _ = task_phase(info)
        return obs, info

    def step(self, action):
        if self.ended:
            raise RuntimeError('Reset required after online task boundary')
        obs, raw_reward, raw_terminated, raw_truncated, info = self.env.step(action)
        info = dict(info)
        if 'success' not in info and self.protocol['reward_mode'] == 'success_once':
            raise ValueError('success_once requires an explicit environment info[success]')
        self.elapsed += 1
        success = bool(info.get('success', False))
        finite_timeout = (self.protocol['timeout_semantics'] == 'finite_horizon'
                          and self.elapsed >= self.protocol['horizon'])
        terminated = bool(raw_terminated) or (success and self.protocol['end_on_success'])
        truncated = bool(raw_truncated) or finite_timeout
        self.ended = terminated or truncated
        reward = (float(success) * self.protocol['success_reward']
                  if self.protocol['reward_mode'] == 'success_once' else float(raw_reward))
        reason = ('success' if success else 'failure' if raw_terminated else
                  'timeout' if truncated else None) if self.ended else None
        phase, flags = task_phase(info)
        info['online_transition'] = dict(step=self.elapsed, raw_reward=float(raw_reward),
            training_reward=reward * self.protocol['reward_scale'],
            raw_terminated=bool(raw_terminated), raw_truncated=bool(raw_truncated),
            terminated=terminated, truncated=truncated, success=success,
            termination_reason=reason, phase=phase, phase_before=self.phase, flags=flags)
        self.phase = phase
        return obs, reward, terminated, truncated, info


def wrap_training_env(env, protocol):
    from envs.chunk_wrapper import ChunkActionWrapper
    cursor = env
    while not isinstance(cursor, ChunkActionWrapper):
        if not isinstance(cursor, gym.Wrapper):
            raise ValueError('Online training requires a ChunkActionWrapper')
        cursor = cursor.env
    if isinstance(cursor.env, OnlineTaskWrapper):
        raise ValueError('Online task adapter already installed')
    cursor.env = OnlineTaskWrapper(cursor.env, protocol)
    return env
