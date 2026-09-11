# evaluate.py

import os
import random
import time # 用于测试高频推理延迟
import torch
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import gymnasium as gym
from gymnasium.wrappers import RecordVideo # 视频录制
import mani_skill.envs
from datetime import datetime

# 导入你的核心架构
from models.policy import EmbodiedGenPolicy
from envs.pointcloud_wrapper import PointCloudObservationWrapper
from envs.chunk_wrapper import ChunkActionWrapper
from envs.maniskill_bridge import ManiSkillToRL100Wrapper 
from utils.normalizer import MinMaxNormalizer
from utils.ppo_checks import configure_ppo_numerics
from utils.eval_utils import RenderToNumpyWrapper 

# =========================================================================
# 核心评估逻辑
# =========================================================================
def make_env_dummy(cfg: DictConfig):
    """ 创建并包装环境 """
    # 1. 初始化你的真实环境 (这里你需要根据真实情况替换)
    # env = gym.make(cfg.env.env_id) 
    
    # [Mock] 为了防止报错，如果你还没有真实环境，用 Dummy 替代
    class DummyEmbodiedEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = gym.spaces.Dict({
                'xyz': gym.spaces.Box(-10, 10, shape=(2000, 3), dtype=np.float32),
                'rgb': gym.spaces.Box(0, 1, shape=(2000, 3), dtype=np.float32),
                'state': gym.spaces.Box(-1, 1, shape=(cfg.model.state_dim,), dtype=np.float32)
            })
            self.action_space = gym.spaces.Box(-1, 1, shape=(cfg.model.action_dim,), dtype=np.float32)
            self.step_count = 0
            # 模拟渲染接口，供 RecordVideo 调用
            self.render_mode = "rgb_array" 
            
        def reset(self, seed=None, options=None):
            self.step_count = 0
            return self.observation_space.sample(), {}
            
        def step(self, action):
            self.step_count += 1
            done = self.step_count >= 100
            reward = 1.0 if done else 0.0
            return self.observation_space.sample(), reward, done, False, {"success": done}
            
        # 假渲染
        def render(self):
            return np.zeros((240, 320, 3), dtype=np.uint8)

    env = DummyEmbodiedEnv()

    # 包装 RecordVideo (按需开启，这里默认每 5 个 episode 录制一次)
    if cfg.eval.get("record_video", False):
        video_dir = os.path.join(cfg.eval.get("log_dir", "./logs"), "videos")
        env = RecordVideo(env, video_folder=video_dir, episode_trigger=lambda x: x % 5 == 0)

    # 2. 包装 PointCloud 处理器
    bounds = cfg.env.get("workspace_bounds", [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
    ws_bounds = np.array(bounds)
    env = PointCloudObservationWrapper(
        env=env,
        num_points=cfg.env.num_points,
        workspace_bounds=ws_bounds,
        use_color=cfg.env.use_color
    )

    # 3. 包装 Action Chunk 处理器 (自动处理 Temporal Ensembling)
    env = ChunkActionWrapper(
        env=env,
        chunk_size=cfg.model.chunk_size,
        exec_steps=cfg.env.exec_steps,
        exp_weight=cfg.env.exp_weight
    )
    return env

def make_env_ManiSkill(cfg):
    """ 创建并包装 ManiSkill 真实仿真环境 """

    env_id = cfg.env.get("env_id", "PushCube-v1")
    obs_mode = cfg.env.get("obs_mode", "pointcloud")
    control_mode = cfg.env.get("control_mode", "pd_ee_delta_pose")
    render_mode = cfg.env.get("render_mode", "rgb_array")

    # 1. 实例化 ManiSkill 环境 (以最经典的抓取方块任务为例)
    env = gym.make(
        env_id,
        obs_mode=obs_mode,
        control_mode=control_mode,
        render_mode=render_mode,
        max_episode_steps=int(cfg.env.get("max_episode_steps", 300))
    )

    if cfg.eval.get("record_video", False):
        # 必须在 RecordVideo 之前转换图像数据类型
        env = RenderToNumpyWrapper(env)
        video_dir = os.path.join(cfg.eval.get("log_dir", "./logs"), "videos")
        os.makedirs(video_dir, exist_ok=True)
        # episode_trigger=lambda x: True 表示每个 Episode 都强制录像
        env = RecordVideo(env, video_folder=video_dir, episode_trigger=lambda x: True, disable_logger=True)
    
    # 2. 接入 ManiSkill 数据适配器 (转换为 {'xyz', 'rgb', 'state'})
    env = ManiSkillToRL100Wrapper(env)
    
    # 3. 接入你原来写好的 PointCloud Wrapper
    bounds = cfg.env.get("workspace_bounds", [[-0.5, -0.5, 0.0], [0.5, 0.5, 0.5]])
    ws_bounds = np.array(bounds)
    
    env = PointCloudObservationWrapper(
        env=env,
        num_points=cfg.env.num_points,  # 比如 1024
        workspace_bounds=ws_bounds,
        use_color=cfg.env.use_color
    )
    
    # 4. 接入你写好的 Action Chunk Wrapper
    env = ChunkActionWrapper(
        env=env,
        chunk_size=cfg.model.chunk_size,
        exec_steps=cfg.env.exec_steps,
        exp_weight=cfg.env.exp_weight
    )
    
    return env

@hydra.main(version_base=None, config_path="configs", config_name="eval")
def main(cfg: DictConfig):
    configure_ppo_numerics()
    print("=" * 60)
    print("🚀 开始具身策略评估 (Evaluation)")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OmegaConf.set_struct(cfg, False)
    base_log_dir = cfg.eval.get("log_dir", "./outputs/eval")
    cfg.eval.log_dir = os.path.join(base_log_dir, f"run_{timestamp}")
    OmegaConf.set_struct(cfg, True)

    device = torch.device(cfg.eval.device)
    
    # 提取基础配置 (防止用户在使用蒸馏模型时忘了改 step)
    num_inference_steps = cfg.eval.get("num_inference_steps", 10)
    is_distilled = cfg.eval.get("is_distilled", False)
    if is_distilled and num_inference_steps != 1:
        print(f"⚠️ 警告: 你声明了这是蒸馏模型，但 num_inference_steps 为 {num_inference_steps}。自动强制覆写为 1！")
        num_inference_steps = 1

    # 1. 准备环境
    # env = make_env_dummy(cfg)
    env = make_env_ManiSkill(cfg)
    print(f"✅ 环境初始化完成! Action Space: {env.action_space}")

    # 2. 准备策略模型
    policy = EmbodiedGenPolicy(
        in_channels=cfg.model.get("in_channels", 3),
        action_dim=cfg.model.action_dim,
        chunk_size=cfg.model.chunk_size,
        use_state=cfg.model.use_state,
        state_dim=cfg.model.state_dim,
        encoder_type=cfg.model.encoder_type,
        backbone_type=cfg.model.backbone_type,
        cond_dim=cfg.model.cond_dim,
        algo_type=cfg.model.algo_type
    ).to(device)

    # 3. 加载 Checkpoint
    if os.path.exists(cfg.eval.ckpt_path):
        state_dict = torch.load(cfg.eval.ckpt_path, map_location=device)
        # 兼容性处理：如果保存的是包含 model / ema_model 的大字典
        if 'ema_model_state_dict' in state_dict:
            policy.load_state_dict(state_dict['ema_model_state_dict'])
            print("📥 成功加载 EMA 权重!")
        elif 'model_state_dict' in state_dict:
            policy.load_state_dict(state_dict['model_state_dict'])
            print("📥 成功加载 Model 权重!")
        else:
            policy.load_state_dict(state_dict)
            print("📥 成功加载纯 State Dict 权重!")
    else:
        print(f"⚠️ 警告: 找不到权重文件 {cfg.eval.ckpt_path}，将使用随机初始化进行测试!")

    policy.eval()

    # 4. 加载 Normalizer (假设你的 stats 与权重存在一起，或者有单独的文件)
    stats_path = os.path.join(os.path.dirname(cfg.eval.ckpt_path), "dataset_stats.json")
    normalizer = MinMaxNormalizer()
    normalizer.load(stats_path)
    print("✅ 成功加载环境归一化参数！")


    # 5. 评估循环
    all_rewards = []
    all_success = []
    latencies = [] # 用于记录推理耗时
    bounds = cfg.env.get("workspace_bounds", [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
    ws_bounds = np.array(bounds)

    print(f"\n🏃 开始进行 {cfg.eval.num_episodes} 个 Episode 的测试...")
    saved_first_pc = False # 用于保存第一个点云
    for ep in tqdm(range(cfg.eval.num_episodes), desc="Evaluating"):
        # 传入固定的 base_seed 以确保测试环境的一致性和可复现性
        base_seed = cfg.eval.get("seed", 42)
        random.seed(base_seed + ep)
        np.random.seed(base_seed + ep)
        torch.manual_seed(base_seed + ep)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(base_seed + ep)
        obs, info = env.reset(seed=base_seed + ep)

        if not saved_first_pc:
            pc_data = obs['point_cloud'] 
            
            # 保存为 .npy 文件
            pc_save_path = os.path.join(cfg.eval.log_dir, "first_frame_pc.npy")
            np.save(pc_save_path, pc_data)
            
            print(f"\n📸 第一帧 3D 点云已保存至: {pc_save_path}")
            print(f"👉 请将其下载到本地，使用可视化脚本查看。")
            saved_first_pc = True
        
        done, truncated = False, False
        ep_reward = 0.0
        success = False

        while not (done or truncated):
            # (A) 准备观测数据: 增加 Batch 维度并发送到 Device
            # 1. 对环境传出的状态进行归一化 (网络期望 [-1,1] 的输入)
            obs_state = normalizer.normalize(obs['state'], 'state')
            pc_centered = normalizer.center_point_cloud(obs['point_cloud'], ws_bounds)
            pc_tensor = torch.from_numpy(pc_centered).unsqueeze(0).to(device)
            state_tensor = torch.from_numpy(obs_state).unsqueeze(0).to(device)
            
            # (B) 策略推理: 返回形状为 [1, chunk_size, action_dim]
            t_start = time.time() 
            with torch.no_grad():
                action_chunk = policy.sample(
                    obs=pc_tensor, 
                    state=state_tensor, 
                    num_steps=num_inference_steps,
                    cfg_weight=cfg.eval.get("cfg_weight", 1.0)
                )
            t_end = time.time()
            latencies.append((t_end - t_start) * 1000.0)

            # (C) 转换为 NumPy 并去掉 Batch 维度 -> [chunk_size, action_dim]
            action_chunk_np = action_chunk.squeeze(0).cpu().to(torch.float32).numpy()
            action_chunk_np = np.clip(action_chunk_np, -1.0, 1.0)

            # (D) 反归一化动作, 将动作反归一化回 ManiSkill 物理引擎的真实增量范围
            real_action_chunk = normalizer.unnormalize(action_chunk_np, 'action')

            # (E) 在 Wrapper 环境中步进 
            # (ChunkActionWrapper 内部会自动做滑动窗口集成并步进 exec_steps 步)
            obs, reward, done, truncated, info = env.step(real_action_chunk)
            ep_reward += reward
            
            # 判断是否成功 (根据具体环境调整，比如 info['success'])
            if info.get('success', False):
                success = True

        all_rewards.append(ep_reward)
        all_success.append(success)
        # print(f"Episode {ep} | Reward: {ep_reward:.2f} | Success: {success}")

    # 6. 统计结果
    mean_reward = np.mean(all_rewards)
    success_rate = np.mean(all_success) * 100.0
    # 推理延迟统计 (去除前 5 个 warmup 样本以保证准确)
    valid_latencies = latencies[5:] if len(latencies) > 5 else latencies
    mean_latency = np.mean(valid_latencies) if valid_latencies else 0.0
    
    print("\n" + "=" * 40)
    print("📊 评估完成 (Evaluation Summary)")
    print("=" * 40)
    print(f"Total Episodes    : {cfg.eval.num_episodes}")
    print(f"Mean Reward       : {mean_reward:.2f} ± {np.std(all_rewards):.2f}")
    print(f"Success Rate      : {success_rate:.1f} %")
    print(f"Inference Latency : {mean_latency:.2f} ms/step (≈ {1000/mean_latency if mean_latency > 0 else 0:.1f} FPS)")
    print("=" * 40)
    env.close()

if __name__ == "__main__":
    main()
