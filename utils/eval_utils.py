# utils/eval_utils.py
import os
import torch
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import mani_skill.envs

# 导入你原有的基础环境 Wrapper
from envs.maniskill_bridge import ManiSkillToRL100Wrapper
from envs.pointcloud_wrapper import PointCloudObservationWrapper
from envs.chunk_wrapper import ChunkActionWrapper

class RenderToNumpyWrapper(gym.Wrapper):
    """
    拦截 ManiSkill 渲染出的 PyTorch Tensor
    将其转换为 numpy.uint8 数组，防止 Gymnasium 的 RecordVideo 崩溃。
    """
    def render(self):
        frame = self.env.render()
        
        # 1. Tensor 转 Numpy
        if hasattr(frame, 'cpu'):
            frame = frame.cpu().numpy()
            
        # 如果画面形状是 (1, H, W, 3)，去掉前面的 1，变成 (H, W, 3)
        if frame.ndim == 4 and frame.shape[0] == 1:
            frame = frame.squeeze(0)
            
        # [防御性编程] 防止某些时候环境返回 [0, 1] 的浮点数导致画面纯黑
        if frame.dtype in [np.float32, np.float64] and frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)
            
        # 2. 确保是图像标准的 uint8 类型
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
            
        return frame
def evaluate_and_record_video(
    cfg, 
    policy, 
    epoch: int, 
    device: torch.device, 
    normalizer=None, 
    base_seed: int = 2000, 
    num_eval_episodes: int = 5, 
    max_steps: int = 300
):
    """
    独立且解耦的验证与录像接口
    Args:
        cfg: Hydra 传入的全局配置
        policy: 当前训练的策略网络
        epoch: 当前训练的 Epoch
        device: 运行设备
        normalizer: 状态和动作的归一化器
        base_seed: 评估起始种子 (默认 2000，避免与训练种子重合)
        num_eval_episodes: 评估并录制的 Episode 数量
        max_steps: 每个 episode 强制截断步数
    """
    print(f"\n🎬 正在为 Epoch {epoch} 进行 ManiSkill 验证测试并录制 {num_eval_episodes} 个视频 (Base Seed: {base_seed})...")
    
    # 记录模型原本的模式，并在测试后还原
    original_training_mode = policy.training 
    policy.eval()
    
    # 清理显存，防止 ManiSkill 开启 RGB 渲染相机时导致 OOM
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        env_id = cfg.env.get("env_id", "PushCube-v1") if "env" in cfg else "PushCube-v1"
        obs_mode = cfg.env.get("obs_mode", "pointcloud") if "env" in cfg else "pointcloud"
        control_mode = cfg.env.get("control_mode", "pd_ee_delta_pose") if "env" in cfg else "pd_ee_delta_pose"

        # 1. 创建 ManiSkill 环境并强制开启录像支持 (rgb_array)
        env = gym.make(
            env_id,
            obs_mode=obs_mode,
            control_mode=control_mode, 
            render_mode="rgb_array",
            max_episode_steps=500
        )
        
        # 2. 先挂载渲染类型转换 Wrapper
        env = RenderToNumpyWrapper(env)
        
        # 3. 再挂载录制 Wrapper (episode_trigger 设为逢 episode 必录)
        video_folder = os.path.join(cfg.save_dir, "eval_videos", f"epoch_{epoch}")
        os.makedirs(video_folder, exist_ok=True)
        env = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda x: True, disable_logger=True)
        
        # 4. 挂载数据对齐与状态 Wrapper
        env = ManiSkillToRL100Wrapper(env)
        
        # 动态提取点云和动作 Chunk 配置
        num_points = cfg.env.num_points if "env" in cfg else cfg.dataset.n_points
        exec_steps = cfg.env.exec_steps if "env" in cfg else 2
        exp_weight = cfg.env.exp_weight if "env" in cfg else 0.01

        if "env" in cfg and "workspace_bounds" in cfg.env:
            bounds = cfg.env.workspace_bounds
        elif "dataset" in cfg and "workspace_bounds" in cfg.dataset:
            bounds = cfg.dataset.workspace_bounds
        else:
            bounds = [[-0.5, -0.5, 0.0], [0.5, 0.5, 0.5]]
        ws_bounds = np.array(bounds)

        use_color = cfg.env.use_color if "env" in cfg else cfg.dataset.get("use_color", False)
        env = PointCloudObservationWrapper(
            env=env, num_points=num_points, workspace_bounds=ws_bounds, use_color=use_color
        )
        env = ChunkActionWrapper(
            env=env, chunk_size=cfg.model.chunk_size, exec_steps=exec_steps, exp_weight=exp_weight, use_ensembling=False
        )
        
        num_infer_steps = cfg.model.get("num_inference_steps", 10)
        success_list = []
        reward_list = []

        # 5. 循环执行多个 Episode 测试
        for ep_idx in range(num_eval_episodes):
            current_seed = base_seed + ep_idx
            obs, _ = env.reset(seed=current_seed)
            done, truncated = False, False
            ep_reward = 0.0
            is_success = False
            step_count = 0
            
            while not (done or truncated) and step_count < max_steps:
                # 点云零均值化
                if normalizer is not None and hasattr(normalizer, 'center_point_cloud'):
                    pc_centered = normalizer.center_point_cloud(obs['point_cloud'], ws_bounds)
                else:
                    pc_centered = obs['point_cloud']
                    
                # State 归一化
                if normalizer is not None:
                    obs_state = normalizer.normalize(obs['state'], 'state')
                else:
                    obs_state = obs['state']

                # 观测转 Tensor
                pc_tensor = torch.from_numpy(
                    np.ascontiguousarray(pc_centered)
                ).float().unsqueeze(0).to(device)

                state_tensor = torch.from_numpy(
                    np.ascontiguousarray(obs_state)
                ).float().unsqueeze(0).to(device)

                # 模型推断
                with torch.no_grad():
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16) if device.type == 'cuda' else torch.no_grad():
                        action_chunk = policy.sample(
                            obs=pc_tensor, state=state_tensor, num_steps=num_infer_steps
                        )
                
                # 环境执行
                action_np = action_chunk.squeeze(0).cpu().to(torch.float32).numpy()
                if normalizer is not None:
                    real_action = normalizer.unnormalize(action_np, 'action')
                else:
                    real_action = action_np
                    
                obs, reward, done, truncated, info = env.step(real_action)
                
                if hasattr(reward, 'item'):
                    reward = reward.item()
                ep_reward += float(reward)
                
                if hasattr(done, 'item'):
                    done = done.item()
                if hasattr(truncated, 'item'):
                    truncated = truncated.item()
                    
                _succ = info.get('success', False)
                if hasattr(_succ, 'item'):
                    _succ = _succ.item()
                if _succ:
                    is_success = True
                step_count += 1

            success_list.append(is_success)
            reward_list.append(ep_reward)
            print(f"  └─ Episode {ep_idx + 1}/{num_eval_episodes} (Seed: {current_seed}) | Reward: {ep_reward:.2f} | Success: {is_success} | Steps: {step_count}")

        mean_reward = np.mean(reward_list)
        success_rate = np.mean(success_list) * 100.0
        print(f"✅ Epoch {epoch} 测试完成 | 平均奖励: {mean_reward:.2f} | 成功率: {success_rate:.1f}%")
        print(f"🎞️ 视频已全部保存至: {video_folder}\n")

    finally:
        # 6. 安全清理
        if 'env' in locals():
            env.close()
        policy.train(original_training_mode)