"""Online collector adapters. GPU simulation uses explicit partial resets.

The final observation is returned before reset. Within a chunk, finished lanes
are masked until all lanes reach the chunk boundary; padding is never training
data. Observations reuse the CPU point sampler to preserve the Actor protocol.
"""
import numpy as np
import torch
from envs.online_task import task_phase
from envs.stackcube_training import stackcube_state, shaped_reward


def numpy_tree(value):
    if isinstance(value, dict):
        return {k: numpy_tree(v) for k, v in value.items()}
    return value.detach().cpu().numpy() if torch.is_tensor(value) else value


def lane(value, index):
    if isinstance(value, dict):
        return {k: lane(v, index) for k, v in value.items()}
    return value[index]


class SingleOnlineEnv:
    num_envs = 1

    def __init__(self, env):
        self.env = env

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        return [obs], [info]

    def step(self, actions):
        obs, reward, term, trunc, info = self.env.step(actions[0])
        return [obs], np.array([reward]), np.array([term]), np.array([trunc]), [info]

    def reset_done(self, done, observations, infos):
        if done[0]:
            observations[0], infos[0] = self.env.reset()
        return observations, infos

    def close(self):
        self.env.close()


class GPUOnlineEnv:
    def __init__(self, cfg, protocol, raw_env=None):
        self.cfg, self.protocol = cfg, protocol
        self.num_envs = int(cfg.online_rollout.num_envs)
        if raw_env is None:
            import gymnasium as gym
            import mani_skill.envs  # noqa: F401
            raw_env = gym.make(cfg.env.env_id, num_envs=self.num_envs,
                sim_backend='physx_cuda', obs_mode=cfg.env.obs_mode,
                control_mode=cfg.env.control_mode, render_mode=cfg.env.render_mode,
                max_episode_steps=protocol['horizon'], reconfiguration_freq=0,
                reward_mode=protocol['environment_reward_mode'] or 'normalized_dense')
        self.env = raw_env
        from envs.maniskill_bridge import ManiSkillToRL100Wrapper
        from envs.pointcloud_wrapper import PointCloudObservationWrapper
        from data.observations import sampling_config, validate_observation_config
        objects = validate_observation_config(cfg.env)
        sampling = sampling_config(cfg.env)
        self.objects = list(objects)
        self.bridge = ManiSkillToRL100Wrapper(raw_env, cfg.model.state_dim,
                                            sampling=sampling, objects=objects)
        self.points = PointCloudObservationWrapper(self.bridge, cfg.env.num_points,
            np.asarray(cfg.env.workspace_bounds), cfg.env.use_color, sampling=sampling)
        self.elapsed = np.zeros(self.num_envs, dtype=np.int64)
        self.pending = np.zeros(self.num_envs, dtype=bool)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def observations(self, raw, indices=None):
        raw = numpy_tree(raw)
        # GPU actors have one segmentation ID per environment; never use H5 IDs.
        ids = {o['name']: getattr(self.unwrapped, o['name']).per_scene_id.cpu().numpy()
               for o in self.objects}
        indices = range(self.num_envs) if indices is None else indices
        return {i: self.points.observation(self.bridge.observation(lane(raw, i),
            target_ids={name: int(values[i]) for name, values in ids.items()})) for i in indices}

    def state(self):
        state, potential, flags = stackcube_state(self.env)
        return numpy_tree(state), numpy_tree(potential), numpy_tree(flags)

    def reset(self, seed=None):
        raw, _ = self.env.reset(seed=seed)
        self.elapsed[:] = 0
        self.pending[:] = False
        states, self.potential, flags = self.state()
        self.phases = [task_phase(lane(flags, i))[0] for i in range(self.num_envs)]
        observations = self.observations(raw)
        return [observations[i] for i in range(self.num_envs)], [
            dict(privileged_state=states[i]) for i in range(self.num_envs)]

    def step(self, actions):
        if self.pending.any():
            raise RuntimeError('Call reset_done before collecting another chunk')
        if actions.shape != (self.num_envs, self.cfg.model.chunk_size, self.cfg.model.action_dim):
            raise ValueError('Incorrect vector action shape')
        n = self.num_envs
        term, trunc = np.zeros(n, bool), np.zeros(n, bool)
        infos = [dict(primitive_rewards=[], online_transitions=[]) for _ in range(n)]
        observations = [None] * n
        for step in range(self.cfg.env.exec_steps):
            active = ~(term | trunc)
            action = torch.as_tensor(actions[:, step], device=self.unwrapped.device, dtype=torch.float32)
            space = self.unwrapped.single_action_space
            action = torch.maximum(torch.minimum(action, torch.as_tensor(space.high, device=action.device)),
                                   torch.as_tensor(space.low, device=action.device))
            action[torch.as_tensor(~active, device=action.device)] = 0
            raw, rewards, raw_term, raw_trunc, _ = self.env.step(action)
            rewards, raw_term, raw_trunc = map(numpy_tree, (rewards, raw_term, raw_trunc))
            states, potential, flags = self.state()
            self.elapsed += active
            just_finished = []
            for i in np.flatnonzero(active):
                info = lane(flags, i)
                success = bool(info['success'])
                term[i] = bool(raw_term[i]) or success
                trunc[i] = bool(raw_trunc[i]) or self.elapsed[i] >= self.protocol['horizon']
                ended = term[i] or trunc[i]
                reward = float(success) * self.protocol['success_reward']
                if self.protocol['reward_mode'] == 'success_potential':
                    reward = shaped_reward(success, self.potential[i], potential[i], ended, self.protocol)
                phase, task_flags = task_phase(info)
                infos[i]['primitive_rewards'].append(reward)
                infos[i]['online_transitions'].append(dict(step=int(self.elapsed[i]),
                    raw_reward=float(rewards[i]), training_reward=reward * self.protocol['reward_scale'],
                    raw_terminated=bool(raw_term[i]), raw_truncated=bool(raw_trunc[i]),
                    terminated=bool(term[i]), truncated=bool(trunc[i]), success=success,
                    termination_reason=('success' if success else 'failure' if term[i] else 'timeout') if ended else None,
                    phase=phase, phase_before=self.phases[i], flags=task_flags))
                infos[i].update(privileged_state=states[i].copy(), success=success, success_any=success)
                self.phases[i] = phase
                self.potential[i] = potential[i]
                if ended or step == self.cfg.env.exec_steps - 1:
                    just_finished.append(i)
            # Save each final observation at its actual boundary, before padding/reset.
            if just_finished:
                for i, obs in self.observations(raw, just_finished).items():
                    observations[i] = obs
            if (term | trunc).all():
                break
        for info in infos:
            info['actual_steps'] = len(info['primitive_rewards'])
        self.pending = term | trunc
        return observations, np.array([sum(i['primitive_rewards']) for i in infos]), term, trunc, infos

    def reset_done(self, done, observations, infos):
        indices = np.flatnonzero(done)
        if len(indices):
            raw, _ = self.env.reset(options={'env_idx': torch.as_tensor(indices, device=self.unwrapped.device)})
            states, potential, flags = self.state()
            reset_obs = self.observations(raw, indices)
            for i in indices:
                observations[i] = reset_obs[i]
                infos[i] = dict(privileged_state=states[i])
                self.elapsed[i] = 0
                self.potential[i] = potential[i]
                self.phases[i] = task_phase(lane(flags, i))[0]
        self.pending[:] = False
        return observations, infos

    def close(self):
        self.env.close()
