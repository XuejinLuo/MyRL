# utils/eval_utils.py
import os
import random
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
def evaluate_and_record_video(cfg, policy, epoch: int, device: torch.device,
                              normalizer=None, seed: int = 42, max_steps=None):
    """Evaluate fixed environment/NumPy/Torch seed pairs; record a subset.

    max_steps, when explicitly supplied, limits chunk decisions per episode.
    Normally termination is controlled only by cfg.env.max_episode_steps.
    Online training sets eval.num_episodes=20; legacy offline callers default
    to one episode. All global RNG states are restored, even on exceptions.
    Evaluation uses the deployment Flow policy without extra PPO Gaussian noise.
    """
    eval_cfg = cfg.get('eval', {})
    episodes = int(eval_cfg.get('num_episodes', 1))
    if episodes < 1:
        raise ValueError('eval.num_episodes must be positive')
    record = bool(eval_cfg.get('record_video', True))
    video_episodes = max(0, int(eval_cfg.get('video_episodes', 1)))
    original_training_mode = policy.training
    np_state, py_state = np.random.get_state(), random.getstate()
    policy.eval()
    env = None
    print(f'\nEvaluating epoch {epoch}: {episodes} fixed-seed episodes...')
    try:
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            env = gym.make(
                cfg.env.get('env_id', 'PushCube-v1'),
                obs_mode=cfg.env.get('obs_mode', 'pointcloud'),
                control_mode=cfg.env.get('control_mode', 'pd_ee_delta_pose'),
                render_mode='rgb_array',
                max_episode_steps=int(cfg.env.get('max_episode_steps', 300)))
            if record and video_episodes:
                env = RenderToNumpyWrapper(env)
                folder = os.path.join(cfg.save_dir, 'eval_videos', f'epoch_{epoch}')
                # RecordVideo creates its own folder; avoid misleading overwrite warnings.
                env = RecordVideo(
                    env, video_folder=folder,
                    episode_trigger=lambda ep: ep < video_episodes,
                    disable_logger=True)
            env = ManiSkillToRL100Wrapper(env)
            ws_bounds = np.asarray(cfg.env.workspace_bounds)
            env = PointCloudObservationWrapper(
                env=env, num_points=cfg.env.num_points,
                workspace_bounds=ws_bounds, use_color=cfg.env.use_color)
            env = ChunkActionWrapper(
                env=env, chunk_size=cfg.model.chunk_size,
                exec_steps=cfg.env.exec_steps, exp_weight=cfg.env.exp_weight,
                use_ensembling=False)
            rewards, successes = [], []
            for ep in range(episodes):
                ep_seed = int(seed) + ep
                random.seed(ep_seed)
                np.random.seed(ep_seed)
                torch.manual_seed(ep_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(ep_seed)
                obs, _ = env.reset(seed=ep_seed)
                done, truncated, success = False, False, False
                ep_reward, decisions = 0.0, 0
                while not (done or truncated):
                    if max_steps is not None and decisions >= max_steps:
                        break
                    pc, state = obs['point_cloud'], obs['state']
                    if normalizer is not None:
                        pc = normalizer.center_point_cloud(pc, ws_bounds)
                        state = normalizer.normalize(state, 'state')
                    pc_t = torch.as_tensor(np.ascontiguousarray(pc), dtype=torch.float32,
                                           device=device).unsqueeze(0)
                    state_t = torch.as_tensor(np.ascontiguousarray(state), dtype=torch.float32,
                                              device=device).unsqueeze(0)
                    with torch.no_grad():
                        action = policy.sample(
                            obs=pc_t, state=state_t,
                            num_steps=cfg.model.get('num_inference_steps', 10))
                    action = np.clip(action[0].cpu().float().numpy(), -1.0, 1.0)
                    if normalizer is not None:
                        action = normalizer.unnormalize(action, 'action')
                    obs, reward, done, truncated, info = env.step(action)
                    ep_reward += float(reward.item() if hasattr(reward, 'item') else reward)
                    done, truncated = bool(done), bool(truncated)
                    success |= bool(info.get('success', False))
                    decisions += 1
                rewards.append(ep_reward)
                successes.append(float(success))
            result = {
                'Eval/RewardMean': float(np.mean(rewards)),
                'Eval/RewardStd': float(np.std(rewards)),
                'Eval/SuccessRate': float(np.mean(successes)),
                'Eval/Episodes': episodes,
            }
            print(f'Epoch {epoch} eval | Reward: {result["Eval/RewardMean"]:.2f} '
                  f'+/- {result["Eval/RewardStd"]:.2f} | '
                  f'Success: {int(sum(successes))}/{episodes} '
                  f'({result["Eval/SuccessRate"]:.1%})')
            return result
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            np.random.set_state(np_state)
            random.setstate(py_state)
            policy.train(original_training_mode)
