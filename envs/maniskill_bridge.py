# envs/maniskill_bridge.py
import gymnasium as gym
import numpy as np

class ManiSkillToRL100Wrapper(gym.ObservationWrapper):
    """
    将 ManiSkill 原生的点云字典转换为你的 Infra 所需的极简字典格式
    """
    def __init__(self, env, state_dim=16, sampling=None, objects=None, require_rgb=False):
        super().__init__(env)
        self.require_rgb = require_rgb
        self.sampling_objects = list((sampling or {}).get('objects', [])) if (
            (sampling or {}).get('mode', 'random') == 'object_budget') else []
        if objects is not None:
            self.sampling_objects = list(objects)
        # 你可以根据真实的 action_dim (比如 8) 和 state_dim (比如 16) 修改下面
        self.observation_space = gym.spaces.Dict({
            'xyz': gym.spaces.Box(-np.inf, np.inf, shape=(100000, 3), dtype=np.float32),
            'rgb': gym.spaces.Box(0, 1, shape=(100000, 3), dtype=np.float32),
            'state': gym.spaces.Box(-np.inf, np.inf, shape=(state_dim,), dtype=np.float32)
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
                xyzw = to_np(pc_dict['xyzw']).reshape(-1, 4)
                xyz = xyzw[:, :3]
            elif 'xyz' in pc_dict:
                xyz = to_np(pc_dict['xyz']).reshape(-1, 3)
            else:
                xyz = np.zeros((0, 3))
            
            if 'rgb' in pc_dict:
                # 归一化到 0~1
                rgb = to_np(pc_dict['rgb']).reshape(-1, 3) / 255.0 
            else:
                if self.require_rgb:
                    raise ValueError('object_centric with use_color requires live RGB')
                rgb = np.zeros_like(xyz)
        else:
            xyz, rgb = np.zeros((0, 3)), np.zeros((0, 3))

        segmentation = None
        if self.sampling_objects:
            if 'segmentation' not in obs.get('pointcloud', {}):
                raise ValueError('Segmented observation requires live pointcloud segmentation')
            segmentation = to_np(obs['pointcloud']['segmentation']).reshape(-1)
            if len(segmentation) != len(xyz):
                raise ValueError('Live segmentation and xyz lengths differ')

        if 'xyzw' in obs.get('pointcloud', {}):
            valid = xyzw[:, 3] > 0
            xyz, rgb = xyz[valid], rgb[valid]
            if segmentation is not None:
                segmentation = segmentation[valid]

        # 2. 提取机器人本体状态 (Proprioception)
        qpos = to_np(obs['agent']['qpos']).reshape(-1)
        # 提取 TCP Pose (具体键名需根据你的 ManiSkill 环境确定，通常是 'tcp_pose' 或 'ee_pose')
        # TCP Pose 通常包含 7 维：[x, y, z, qw, qx, qy, qz]
        if 'extra' in obs and 'tcp_pose' in obs['extra']:
            tcp_pose = to_np(obs['extra']['tcp_pose']).reshape(-1)
        elif 'tcp_pose' in obs['agent']:
            tcp_pose = to_np(obs['agent']['tcp_pose']).reshape(-1)
        else:
            raise ValueError('Observation missing tcp_pose')
            
        # 将 qpos 和 tcp_pose 拼接
        state = np.concatenate([qpos, tcp_pose]).astype(np.float32) 

        result = {
            'xyz': xyz, 
            'rgb': rgb, 
            'state': state
        }
        if self.sampling_objects:
            from data.pointcloud import resolve_target_ids
            result['segmentation'] = segmentation
            # Scene IDs can change after reconfiguration/reset.
            result['target_ids'] = resolve_target_ids(
                self.sampling_objects, self.unwrapped.segmentation_id_map)
        return result
