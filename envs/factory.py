import gymnasium as gym
import mani_skill.envs
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper
import numpy as np
from envs.maniskill_bridge import ManiSkillToRL100Wrapper
from envs.pointcloud_wrapper import PointCloudObservationWrapper
from envs.chunk_wrapper import ChunkActionWrapper

def make_env(cfg, primitive_wrapper=None, video=None):
    """ 创建并包装 ManiSkill 真实仿真环境 """

    env_id = cfg.env.get("env_id", "PushCube-v1")
    obs_mode = cfg.env.get("obs_mode", "pointcloud")
    control_mode = cfg.env.get("control_mode", "pd_ee_delta_pose")
    render_mode = cfg.env.get("render_mode", "rgb_array")
    
    # 1. 实例化 ManiSkill 环境 (以最经典的抓取方块任务为例)
    env = gym.make(
        env_id,
        num_envs=1,
        sim_backend="physx_cpu",
        obs_mode=obs_mode,
        control_mode=control_mode,
        render_mode=render_mode,
        max_episode_steps=cfg.env.get("max_episode_steps", 300)
    )
    
    # 2. 接入 ManiSkill 数据适配器 (转换为 {'xyz', 'rgb', 'state'})
    env = CPUGymWrapper(env)
    env = ManiSkillToRL100Wrapper(env, state_dim=cfg.model.state_dim)
    
    # 3. 接入你原来写好的 PointCloud Wrapper
    bounds = cfg.env.get("workspace_bounds", [[-0.5, -0.5, 0.0], [0.5, 0.5, 0.5]])
    ws_bounds = np.array(bounds)
    
    env = PointCloudObservationWrapper(
        env=env,
        num_points=cfg.env.num_points,  # 比如 1024
        workspace_bounds=ws_bounds,
        use_color=cfg.env.use_color
    )
    
    if video is not None:
        from evaluation.video import EvalVideo
        env = EvalVideo(env, **video)

    if primitive_wrapper is not None:
        env = primitive_wrapper(env)

    # 4. 接入你写好的 Action Chunk Wrapper
    env = ChunkActionWrapper(
        env=env,
        chunk_size=cfg.model.chunk_size,
        exec_steps=cfg.env.exec_steps,
        exp_weight=cfg.env.exp_weight
    )
    
    return env



