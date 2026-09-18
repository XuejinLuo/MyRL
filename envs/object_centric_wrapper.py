"""Live adapter using exactly the same object builder as H5 ingestion."""
import gymnasium as gym
import numpy as np
from data.object_centric import build_object_observation, object_shapes


class ObjectCentricObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, config, workspace_bounds, use_color=True):
        super().__init__(env)
        self.config, self.bounds, self.use_color = config, workspace_bounds, use_color
        spaces = {}
        for key, shape in object_shapes(config, 6 if use_color else 3).items():
            if 'mask' in key or key == 'object_valid':
                spaces[key] = gym.spaces.Box(0, 1, shape, dtype=np.bool_)
            elif key == 'object_roles':
                from data.object_centric import ROLES
                spaces[key] = gym.spaces.Box(0, len(ROLES)-1, shape, dtype=np.int64)
            else:
                spaces[key] = gym.spaces.Box(-np.inf, np.inf, shape, dtype=np.float32)
        spaces['state'] = env.observation_space['state']
        self.observation_space = gym.spaces.Dict(spaces)

    def observation(self, obs):
        return dict(**build_object_observation(obs['xyz'], obs.get('rgb'),
            obs.get('segmentation'), self.bounds, self.config, self.use_color,
            target_ids=obs.get('target_ids')), state=obs['state'])
