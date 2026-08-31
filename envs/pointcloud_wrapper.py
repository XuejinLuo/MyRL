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
        use_color: bool = False
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

    def observation(self, obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        每次 env.step() 或 env.reset() 后会自动调用此方法
        """
        # 1. 提取原始数据 (兼容不同环境的命名习惯)
        xyz = obs.get('xyz', np.zeros((0, 3)))
        rgb = obs.get('rgb', None)
        state = obs['state']

        # [修复] 如果要求用颜色，但环境没给，补零对齐维度，防止后续报 Shape Mismatch
        if self.use_color and rgb is None:
            rgb = np.zeros_like(xyz)

        # 2. 工作空间裁剪 (剔除机器人本体基座、远处背景等无关信息)
        if self.workspace_bounds is not None and xyz.shape[0] > 0:
            mask = (
                (xyz[:, 0] >= self.workspace_bounds[0, 0]) & (xyz[:, 0] <= self.workspace_bounds[1, 0]) &
                (xyz[:, 1] >= self.workspace_bounds[0, 1]) & (xyz[:, 1] <= self.workspace_bounds[1, 1]) &
                (xyz[:, 2] >= self.workspace_bounds[0, 2]) & (xyz[:, 2] <= self.workspace_bounds[1, 2])
            )
            xyz = xyz[mask]
            if rgb is not None:
                rgb = rgb[mask]

        # 3. 点云降采样 / 补齐
        num_current_pts = xyz.shape[0]
        if num_current_pts == 0:
            # 极端情况：视野内什么点都没有，返回全零以防止网络报错
            xyz = np.zeros((self.num_points, 3))
            if rgb is not None:
                rgb = np.zeros((self.num_points, 3))
        elif num_current_pts >= self.num_points:
            # 使用随机采样 (FPS Farthest Point Sampling 在 Python 端做会较慢，随机采样在大多数场景足够好)
            indices = np.random.choice(num_current_pts, self.num_points, replace=False)
            xyz = xyz[indices]
            if rgb is not None:
                rgb = rgb[indices]
        else:
            # 点数不够，允许重复采样 (Padding)
            indices = np.random.choice(num_current_pts, self.num_points, replace=True)
            xyz = xyz[indices]
            if rgb is not None:
                rgb = rgb[indices]

        # 4. 拼接点云特征
        if self.use_color and rgb is not None:
            # 假设 RGB 已经是 0-1 或 -1~1 归一化的浮点数
            point_cloud = np.concatenate([xyz, rgb], axis=-1)
        else:
            point_cloud = xyz

        return {
            'point_cloud': point_cloud.astype(np.float32),
            'state': state.astype(np.float32)
        }


# =========================================================================
# 测试模块：用来验证代码是否可行
# =========================================================================
if __name__ == '__main__':
    class DummyEmbodiedEnv(gym.Env):
        """ 模拟一个底层环境（例如 ManiSkill3, MuJoCo 或是真机相机的返回流） """
        def __init__(self):
            super().__init__()
            # 定义原始空间：含有数万个无序点的 xyz 和 rgb，以及 14 维的本体状态 (如关节角+末端位姿)
            self.observation_space = spaces.Dict({
                'xyz': spaces.Box(-10, 10, shape=(20000, 3), dtype=np.float32),
                'rgb': spaces.Box(0, 1, shape=(20000, 3), dtype=np.float32),
                'state': spaces.Box(-1, 1, shape=(14,), dtype=np.float32)
            })
            self.action_space = spaces.Box(-1, 1, shape=(7,), dtype=np.float32)

        def reset(self, seed=None, options=None):
            # 模拟相机生成一团包含桌面、目标物和背景的点云
            raw_xyz = np.random.uniform(-2, 2, size=(15000, 3))
            raw_rgb = np.random.uniform(0, 1, size=(15000, 3))
            state = np.random.uniform(-1, 1, size=(14,))
            obs = {'xyz': raw_xyz, 'rgb': raw_rgb, 'state': state}
            return obs, {}

        def step(self, action):
            obs, _ = self.reset()
            reward = 1.0
            done = False
            return obs, reward, done, False, {}

    print("=== 初始化原始虚拟环境 ===")
    base_env = DummyEmbodiedEnv()
    
    # 设定机械臂桌面操作的工作空间 (单位：米)
    # 例如：x:[0.1, 0.8] (前方), y:[-0.5, 0.5] (左右), z:[0.0, 0.6] (高度)
    ws_bounds = np.array([
        [0.1, -0.5, 0.0],
        [0.8,  0.5, 0.6]
    ])

    print("=== 接入 PointCloudObservationWrapper ===")
    env = PointCloudObservationWrapper(
        env=base_env,
        num_points=1024,                # 降采样到 1024 个点
        workspace_bounds=ws_bounds,     # 剔除框外干扰点
        use_color=True                  # 采用 XYZ + RGB (6维特征)
    )

    print("\n[环境观测空间]")
    print(env.observation_space)

    obs, info = env.reset()
    print("\n[Reset后 - Wrapper 输出的数据维度]")
    print(f"Point Cloud Shape: {obs['point_cloud'].shape}  (应为 1024, 6)")
    print(f"Robot State Shape: {obs['state'].shape}        (应为 14,)")
    print(f"Point Cloud Sample [0]: \n{obs['point_cloud'][0]}")

    print("\n[Step 测试]")
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step后提取的 Point Cloud Shape: {obs['point_cloud'].shape}")
    print("================ 测试通过 ✅ ================")