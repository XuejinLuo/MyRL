# envs/pointcloud_wrapper.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Optional, Tuple, Any

class PointCloudObservationWrapper(gym.ObservationWrapper):
    """
    点云观测封装器 (适配 RL-100 / 3D Diffusion 架构)
    功能：
    1. 接收原始的 dict 观测 (包含无序点云 xyz, 颜色 rgb, 本体状态 state)
    2. 工作空间裁剪 (Workspace Cropping) - 剔除无关背景，极大提升 3D Diffusion 效率
    3. 点云降采样 (Downsampling) - 固定输出 N 个点，支持 PointNeXt/PointNet++ 批处理
    4. 特征拼接 - 统一输出 {'point_cloud': [N, 3+C], 'state': [S]}
    """
    def __init__(
        self, 
        env: gym.Env, 
        num_points: int = 1024, 
        workspace_bounds: Optional[np.ndarray] = None, 
        use_color: bool = False,
        sampling=None,
    ):
        """
        :param env: 原始 Gymnasium 环境
        :param num_points: 降采样后的目标点云数量 (常用 1024 或 2048)
        :param workspace_bounds: 形状为 (2, 3) 的 numpy 数组, [[xmin, ymin, zmin], [xmax, ymax, zmax]]
        :param use_color: 是否在 point_cloud 维度中拼接 RGB 特征 (输出 Nx6 或 Nx3)
        """
        super().__init__(env)
        self.num_points = num_points
        self.workspace_bounds = workspace_bounds
        self.use_color = use_color
        self.sampling = sampling

        # 校验原始环境是否包含我们需要的键
        assert isinstance(self.env.observation_space, spaces.Dict), "基础环境观测必须是 Dict 空间"
        assert 'state' in self.env.observation_space.spaces, "基础环境必须包含本体状态 'state'"

        # 计算特征维度
        pc_feature_dim = 6 if self.use_color else 3
        state_dim = self.env.observation_space.spaces['state'].shape[0]

        # 重写 Observation Space (给后端的 Buffer 和 Neural Network 明确形状)
        self.observation_space = spaces.Dict({
            'point_cloud': spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(self.num_points, pc_feature_dim), dtype=np.float32
            ),
            'state': spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(state_dim,), dtype=np.float32
            )
        })

    def observation(self, obs):
        from data.pointcloud import preprocess_points
        return {'point_cloud': preprocess_points(obs['xyz'], obs.get('rgb'),
                    self.workspace_bounds, self.num_points, self.use_color,
                    sampling=self.sampling, segmentation=obs.get('segmentation'),
                    target_ids=obs.get('target_ids')),
                'state': obs['state']}
