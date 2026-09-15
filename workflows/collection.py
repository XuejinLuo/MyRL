"""Record below ChunkActionWrapper, after its clipping, before observation normalization."""
import numpy as np
import gymnasium as gym


class PrimitiveRecorder(gym.Wrapper):
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.rows = {k: [] for k in ('pc', 'state', 'action', 'reward', 'success', 'terminated', 'truncated')}
        self._obs(obs)
        return obs, info

    def _obs(self, obs):
        self.rows['pc'].append(np.array(obs['point_cloud'], dtype=np.float32, copy=True))
        self.rows['state'].append(np.array(obs['state'], dtype=np.float32, copy=True))

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if 'success' not in info:
            raise ValueError('Collection requires per-step info[success]')
        success = bool(info['success'])
        self._obs(obs)
        for k, v in dict(action=np.array(action, dtype=np.float32, copy=True),
                         reward=float(success), success=success,
                         terminated=bool(terminated), truncated=bool(truncated)).items():
            self.rows[k].append(v)
        return obs, reward, terminated, truncated, info

    def episode(self):
        return {k: np.asarray(v) for k, v in self.rows.items()}
