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
def evaluate_and_record_video(cfg, policy, epoch: int, device: torch.device, seed: int = 42, max_steps: int = 100):
    """
    独立且解耦的验证与录像接口
    Args:
        cfg: Hydra 传入的全局配置
        policy: 当前训练的策略网络 (无需手动切换 eval 模式，内部会自动处理并还原)
        epoch: 当前训练的 Epoch
        device: 运行设备
        seed: 环境随机种子，固定种子可以更好地观察模型在同一初始状态下的演进
        max_steps: 强制截断步数，防止策略在早期未收敛时陷入死循环
    """
    print(f"\n🎬 正在为 Epoch {epoch} 进行 ManiSkill 验证测试并录制视频...")
    
    # 记录模型原本的模式，并在测试后还原
    original_training_mode = policy.training 
    policy.eval()
    
    # 清理显存，防止 ManiSkill 开启 RGB 渲染相机时导致 OOM
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        # 1. 创建 ManiSkill 环境并强制开启录像支持 (rgb_array)
        env = gym.make(
            "PickCube-v1",
            obs_mode="pointcloud",
            control_mode="pd_ee_delta_pose", 
            render_mode="rgb_array",
        )

        # 2. 先挂载渲染类型转换 Wrapper
        env = RenderToNumpyWrapper(env)
        
        # 3. 再挂载录制 Wrapper
        video_folder = os.path.join(cfg.save_dir, "eval_videos", f"epoch_{epoch}")
        os.makedirs(video_folder, exist_ok=True)
        # episode_trigger=lambda x: True 表示逢 Episode 必录制
        env = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda x: True, disable_logger=True)
        
        # 4. 最后挂载数据对齐与状态 Wrapper
        env = ManiSkillToRL100Wrapper(env)
        
        ws_bounds = np.array([[-0.5, -0.5, 0.0], [0.5, 0.5, 0.5]])
        
        # 动态提取配置 (兼容 offline 和 online 两套参数树结构)
        num_points = cfg.env.num_points if "env" in cfg else cfg.dataset.n_points
        exec_steps = cfg.env.exec_steps if "env" in cfg else 2
        exp_weight = cfg.env.exp_weight if "env" in cfg else 0.01

        env = PointCloudObservationWrapper(
            env=env, num_points=num_points, workspace_bounds=ws_bounds, use_color=False
        )
        env = ChunkActionWrapper(
            env=env, chunk_size=cfg.model.chunk_size, exec_steps=exec_steps, exp_weight=exp_weight
        )
        
        # 4. 执行测试环境 Rollout
        obs, _ = env.reset(seed=seed)
        done, truncated = False, False
        ep_reward = 0.0
        success = False
        step_count = 0
        num_infer_steps = cfg.model.get("num_inference_steps", 10)
        
        while not (done or truncated) and step_count < max_steps:
            # 观测转 Tensor
            pc_tensor = torch.from_numpy(obs['point_cloud']).unsqueeze(0).to(device)
            state_tensor = torch.from_numpy(obs['state']).unsqueeze(0).to(device)

            # 模型推断 (使用 AMP 自动混合精度加速)
            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16) if device.type == 'cuda' else torch.no_grad():
                    action_chunk = policy.sample(
                        obs=pc_tensor, state=state_tensor, num_steps=num_infer_steps
                    )
            
            # 环境执行
            action_np = action_chunk.squeeze(0).cpu().to(torch.float32).numpy()
            obs, reward, done, truncated, info = env.step(action_np)
            
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
                success = True
            step_count += 1

        print(f"✅ Epoch {epoch} 测试完成 | 总奖励: {ep_reward:.2f} | 是否成功: {success}")
        print(f"🎞️ 视频已保存至: {video_folder}\n")

    finally:
        # 6. 安全清理 (确保发生异常也会关闭环境并还原模型状态)
        if 'env' in locals():
            env.close()
        policy.train(original_training_mode)