# envs/maniskill_bridge.py
import gymnasium as gym
import numpy as np

class ManiSkillToRL100Wrapper(gym.ObservationWrapper):
    """
    将 ManiSkill 原生的点云字典转换为你的 Infra 所需的极简字典格式
    """
    def __init__(self, env):
        super().__init__(env)
        # 你可以根据真实的 action_dim (比如 8) 和 state_dim (比如 18) 修改下面
        self.observation_space = gym.spaces.Dict({
            'xyz': gym.spaces.Box(-np.inf, np.inf, shape=(100000, 3), dtype=np.float32),
            'rgb': gym.spaces.Box(0, 1, shape=(100000, 3), dtype=np.float32),
            'state': gym.spaces.Box(-np.inf, np.inf, shape=(18,), dtype=np.float32)
        })

    def observation(self, obs):
        # 辅助函数：防御性兼容 (ManiSkill GPU 模式返回 Tensor，CPU 模式返回 numpy)
        def to_np(x):
            return x.cpu().numpy() if hasattr(x, 'cpu') else np.array(x)

        # 1. 提取点云 XYZ 和 RGB
        # ManiSkill 默认点云结构可能按相机分组，我们把所有相机的点云压平合并
        if 'pointcloud' in obs:
            pc_dict = obs['pointcloud']
            # ManiSkill 3 的点云通常在 xyzw 中，前三维是 xyz
            if 'xyzw' in pc_dict:
                xyz = to_np(pc_dict['xyzw'][..., :3]).reshape(-1, 3)
            elif 'xyz' in pc_dict:
                xyz = to_np(pc_dict['xyz']).reshape(-1, 3)
            else:
                xyz = np.zeros((0, 3))
            
            if 'rgb' in pc_dict:
                # 归一化到 0~1
                rgb = to_np(pc_dict['rgb']).reshape(-1, 3) / 255.0 
            else:
                rgb = np.zeros_like(xyz)
        else:
            xyz, rgb = np.zeros((0, 3)), np.zeros((0, 3))

        # 2. 提取机器人本体状态 (Proprioception)
        qpos = to_np(obs['agent']['qpos']).reshape(-1)
        state = qpos.astype(np.float32) 

        return {
            'xyz': xyz, 
            'rgb': rgb, 
            'state': state
        }