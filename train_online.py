# train_online.py
import os
import torch
import torch.nn as nn
import numpy as np
import hydra
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import wandb
import gymnasium as gym
import mani_skill.envs

# =========================================================================
# 导入你的核心架构组件
# =========================================================================
from models.policy import EmbodiedGenPolicy
from models.encoders.pointnext import PointNeXtEncoder
from models.critics.q_v_network import VNetwork
from algos.pg import FlowPolicyGradient
from envs.pointcloud_wrapper import PointCloudObservationWrapper
from envs.chunk_wrapper import ChunkActionWrapper
from envs.maniskill_bridge import ManiSkillToRL100Wrapper 
from utils.eval_utils import evaluate_and_record_video
from utils.debug_logger import DebugLogger
from utils.normalizer import MinMaxNormalizer

# =========================================================================
# 桥接层 (Bridge Wrappers)
# PG 更新要求 Tensor 输入，而我们的环境吐出的是 Dict。这里用 Wrapper 进行解包对齐。
# =========================================================================

class CriticFeatureExtractor(nn.Module):
    """ 为 Critic 专门提供 3D 点云特征提取 """
    def __init__(self, cfg):
        super().__init__()
        self.encoder = PointNeXtEncoder(
            in_channels=cfg.model.get("in_channels", 3),
            output_dim=cfg.model.cond_dim,
            use_state=cfg.model.use_state,
            state_dim=cfg.model.state_dim
        )
    def forward(self, obs_dict):
        pn_dict = {
            'point_cloud': obs_dict['pc'],
            'state': obs_dict['state']
        }
        return self.encoder(pn_dict)

class Critic_PG_Wrapper(nn.Module):
    """ 将 Encoder 和 VNetwork 组合，对外暴露单一的 forward 接口 """
    def __init__(self, encoder, v_net):
        super().__init__()
        self.encoder = encoder
        self.v_net = v_net
        
    def forward(self, obs_dict):
        # PG 会调用 self.critic(states)，这里 states 实际上就是 obs_dict
        feat = self.encoder(obs_dict)
        return self.v_net(feat)

class Policy_PG_Wrapper(nn.Module):
    """ 将 EmbodiedGenPolicy 包装，对齐 evaluate_actions 和 sample 的字典解包 """
    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        
    def evaluate_actions(self, obs_dict, actions, noise=None):
        # 匹配 algos/pg.py 中 trainer.update_step 的调用
        log_prob_proxy, entropy = self.policy.evaluate_actions(
            obs=obs_dict['pc'], 
            actions=actions, 
            state=obs_dict['state'],
            noise=noise
        )
        # 这里加入一个经验放缩系数，将 MSE 转化为合理的似然差异，激活 PPO 裁剪机制。
        scale_factor = 10.0 
        scaled_log_prob = log_prob_proxy * scale_factor
        
        return scaled_log_prob, entropy
        
    def sample(self, obs_dict, num_steps=10):
        # 环境 Rollout 交互调用
        return self.policy.sample(
            obs=obs_dict['pc'], 
            state=obs_dict['state'], 
            num_steps=num_steps
        )

# =========================================================================
# 环境构建
# =========================================================================

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
        max_episode_steps=300
    )
    
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

# =========================================================================
# Rollout Buffer (On-policy 数据收集器)
# =========================================================================
class RolloutBuffer:
    def __init__(self, device):
        self.device = device
        self.clear()
        
    def clear(self):
        self.obs_pc, self.obs_state = [], []
        self.actions, self.rewards, self.dones, self.values = [], [], [], []
        
    def add(self, pc, state, action, reward, done, value):
        self.obs_pc.append(pc)
        self.obs_state.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        
    def get_tensors(self):
        # 转化为 Tensor, 维度: [Time_Steps, Batch_Size(1), ...]
        return {
            'pc': torch.FloatTensor(np.array(self.obs_pc)).unsqueeze(1).to(self.device),
            'state': torch.FloatTensor(np.array(self.obs_state)).unsqueeze(1).to(self.device),
            'actions': torch.FloatTensor(np.array(self.actions)).unsqueeze(1).to(self.device),
            'rewards': torch.FloatTensor(np.array(self.rewards)).unsqueeze(1).to(self.device),
            'dones': torch.FloatTensor(np.array(self.dones)).unsqueeze(1).to(self.device),
            'values': torch.FloatTensor(np.array(self.values)).unsqueeze(1).to(self.device),
        }

# =========================================================================
# Main 训练循环
# =========================================================================
@hydra.main(version_base=None, config_path="configs", config_name="train_online")
def main(cfg: DictConfig):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{cfg.run_name}_{timestamp}"
    OmegaConf.set_struct(cfg, False)
    cfg.run_name = run_name
    cfg.save_dir = os.path.join(cfg.save_dir, run_name)
    OmegaConf.set_struct(cfg, True)
    debug_logger = DebugLogger(cfg.save_dir)

    # 1. 实验追踪
    if cfg.wandb.enable:
        wandb.init(project=cfg.wandb.project, name=cfg.run_name, config=OmegaConf.to_container(cfg, resolve=True))
        
    device = torch.device(cfg.device)
    os.makedirs(cfg.save_dir, exist_ok=True)

    # 2. 初始化环境
    env = make_env_ManiSkill(cfg)
    normalizer = MinMaxNormalizer()
    if cfg.algo.pretrained_ckpt and os.path.exists(cfg.algo.pretrained_ckpt):
        stats_path = os.path.join(os.path.dirname(cfg.algo.pretrained_ckpt), "dataset_stats.json")
        if os.path.exists(stats_path):
            normalizer.load(stats_path)
            print(f"✅ 成功加载环境归一化参数: {stats_path}")
        else:
            print(f"⚠️ 警告: 找不到归一化文件 {stats_path}，使用默认空 Normalizer!")
        critic_encoder.encoder.load_state_dict(base_policy.encoder.state_dict())
        print("✅ 成功将 Actor 的预训练 3D 编码器权重同步给 Critic！")
    else:
        print("⚠️ 警告: 未提供预训练权重路径(pretrained_ckpt)，Normalizer将为空!")
    bounds = cfg.env.get("workspace_bounds", [[-0.5, -0.5, 0.0], [0.5, 0.5, 0.5]])
    ws_bounds = np.array(bounds)
    
    # 3. 初始化模型组件
    print("🧠 初始化 Actor (Policy) 与 Critic (Value Network)...")
    base_policy = EmbodiedGenPolicy(
        in_channels=cfg.model.get("in_channels", 3),
        action_dim=cfg.model.action_dim, chunk_size=cfg.model.chunk_size,
        use_state=cfg.model.use_state, state_dim=cfg.model.state_dim,
        encoder_type=cfg.model.encoder_type, backbone_type=cfg.model.backbone_type,
        cond_dim=cfg.model.cond_dim, algo_type=cfg.model.algo_type
    ).to(device)
    actor = Policy_PG_Wrapper(base_policy)
    
    # 适配含 EMA/Model 嵌套大字典的 Checkpoint 格式
    if cfg.algo.pretrained_ckpt and os.path.exists(cfg.algo.pretrained_ckpt):
        state_dict = torch.load(cfg.algo.pretrained_ckpt, map_location=device)
        if 'ema_model_state_dict' in state_dict:
            base_policy.load_state_dict(state_dict['ema_model_state_dict'])
            print("✅ 成功加载 IDQL 离线预训练权重 (EMA)，在此基础上启动 PG 微调！")
        elif 'model_state_dict' in state_dict:
            base_policy.load_state_dict(state_dict['model_state_dict'])
            print("✅ 成功加载 IDQL 离线预训练权重 (Model)，在此基础上启动 PG 微调！")
        else:
            base_policy.load_state_dict(state_dict)
            print("✅ 成功加载 IDQL 离线预训练权重 (Raw)，在此基础上启动 PG 微调！")

    critic_encoder = CriticFeatureExtractor(cfg).to(device)
    base_v_net = VNetwork(state_dim=cfg.model.cond_dim).to(device)
    critic = Critic_PG_Wrapper(critic_encoder, base_v_net)
    
    # 4. 初始化 PG Trainer
    trainer = FlowPolicyGradient(
        policy=actor, critic=critic,
        actor_lr=cfg.algo.actor_lr,
        critic_lr=cfg.algo.critic_lr,
        gamma=cfg.algo.gamma, 
        gae_lambda=cfg.algo.gae_lambda, 
        clip_ratio=cfg.algo.clip_ratio,
        entropy_coef=cfg.algo.get("entropy_coef", 0.01),
        v_loss_coef=cfg.algo.get("v_loss_coef", 0.5),
        max_grad_norm=cfg.algo.get("max_grad_norm", 1.0),
        device=device
    )

    buffer = RolloutBuffer(device)
    obs, _ = env.reset()

    # 5. 主循环 (Epochs)
    print(f"🚀 开始在线强化微调 (Total Epochs: {cfg.epochs})")
    
    for epoch in range(1, cfg.epochs + 1):
        buffer.clear()
        epoch_reward = 0.0
        successes = 0             # 累计成功次数
        episodes_completed = 0    # 发生 done 或 truncate 的完整回合数
        step_values = []          # 收集每一步 Critic 的预期价值预测

        # --- A. 轨迹收集阶段 (Rollout) ---
        actor.eval()
        critic.eval()
        step = 0
        for _ in tqdm(range(cfg.algo.steps_per_epoch), desc=f"Epoch {epoch} Rollout", leave=False):
            # 观测喂给网络前必须使用 Normalizer 进行点云中心化和归一化
            pc_centered = normalizer.center_point_cloud(obs['point_cloud'], ws_bounds)
            obs_state = normalizer.normalize(obs['state'], 'state')

            obs_dict = {
                'pc': torch.FloatTensor(pc_centered).unsqueeze(0).to(device),
                'state': torch.FloatTensor(obs_state).unsqueeze(0).to(device)
            }
            
            with torch.no_grad():
                # 计算当前状态的 Value
                value = critic(obs_dict).item()
                # 策略前向生成动作 Chunk
                action_chunk = actor.sample(obs_dict, num_steps=cfg.model.num_inference_steps)

            step_values.append(value)
            debug_logger.log_io(epoch, step, obs_dict, action_chunk)

            # 环境执行 (转换为 numpy)
            action_np = action_chunk.squeeze(0).cpu().numpy()
            real_action = normalizer.unnormalize(action_np, 'action')
            next_obs, reward, done, truncated, info = env.step(real_action)

            reward_val = float(reward.item() if hasattr(reward, 'item') else reward)
            done_val = bool(done.item() if hasattr(done, 'item') else done)
            trunc_val = bool(truncated.item() if hasattr(truncated, 'item') else truncated)
            epoch_reward += reward_val  
            
            # 注意：Buffer 中必须存入的是归一化后的数据！以供后续 PPO 用原比例计算 Loss
            buffer.add(pc_centered, obs_state, action_np, reward_val, done_val or trunc_val, value)
            
            obs = next_obs
            if done_val or trunc_val:
                obs, _ = env.reset()

            _succ = info.get('success', False)
            if hasattr(_succ, 'item'): _succ = _succ.item()
            if _succ:
                successes += 1
            if done_val or trunc_val:
                episodes_completed += 1
                
        # --- B. 计算 GAE 与 Returns ---
        rollout_data = buffer.get_tensors()
        
        # 获取下一个状态的 value (用于 Bootstrapping)
        with torch.no_grad():
            next_pc_centered = normalizer.center_point_cloud(obs['point_cloud'], ws_bounds)
            next_obs_state = normalizer.normalize(obs['state'], 'state')
            next_obs_dict = {
                'pc': torch.FloatTensor(next_pc_centered).unsqueeze(0).to(device),
                'state': torch.FloatTensor(next_obs_state).unsqueeze(0).to(device)
            }
            next_value = critic(next_obs_dict).squeeze(-1)
            
        advantages, returns = trainer.compute_gae(
            rollout_data['rewards'], rollout_data['values'], rollout_data['dones'], next_value
        )
        mean_return = returns.mean().item()
        
        # PPO Trick: Advantage 归一化 (极大提升微调稳定性)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --- C. PPO 网络更新阶段 ---
        actor.train()
        for module in actor.modules():
            if isinstance(module, torch.nn.BatchNorm1d) or isinstance(module, torch.nn.BatchNorm2d):
                module.eval()
        critic.train()
        
        # 将 [Time, Batch, ...] 拍扁为 [Batch_size, ...]
        actual_channels = rollout_data['pc'].shape[-1]  # <--- 获取真实的通道数
        flat_obs_dict = {
            'pc': rollout_data['pc'].view(-1, cfg.env.num_points, actual_channels),
            'state': rollout_data['state'].view(-1, cfg.model.state_dim)
        }
        flat_actions = rollout_data['actions'].view(-1, cfg.model.chunk_size, cfg.model.action_dim)
        flat_adv = advantages.view(-1)
        flat_returns = returns.view(-1)

        fixed_noise = torch.randn_like(flat_actions)

        with torch.no_grad():
            old_log_probs, _ = actor.evaluate_actions(flat_obs_dict, flat_actions, noise=fixed_noise)

        # PPO 循环微调
        epoch_losses = {}
        for ppo_epoch in range(cfg.algo.update_epochs):
            loss_dict = trainer.update_step(
                states=flat_obs_dict, 
                actions=flat_actions, 
                old_log_probs=old_log_probs, 
                returns=flat_returns, 
                advantages=flat_adv,
                noise=fixed_noise
            )
            
            for k, v in loss_dict.items():
                epoch_losses[k] = epoch_losses.get(k, 0) + v
                
        # 平均 Loss
        for k in epoch_losses:
            epoch_losses[k] /= cfg.algo.update_epochs

        # --- D. 打印与保存 ---
        log_metrics = {
            "Env/Reward": epoch_reward,
            "Env/Success_Count": successes,
            "Env/Episodes_Done": episodes_completed,
            "Value/Mean_V_Pred": np.mean(step_values),
            "Value/Mean_Return": mean_return,
            **epoch_losses  # 这里自动包含了 ppo/approx_kl 等指标
        }
        if cfg.wandb.enable:
            wandb.log(log_metrics, step=epoch)
        debug_logger.log_metrics(epoch, log_metrics)
            
        print(f"Epoch {epoch:03d} | Rew: {epoch_reward:.1f} | Succ: {successes} | "
              f"V_Pred: {np.mean(step_values):.2f} | KL: {epoch_losses.get('ppo/approx_kl',0):.4f} | "
              f"EV: {epoch_losses.get('ppo/explained_var',0):.3f}")
        
        if epoch % cfg.save_epoch == 0 or epoch == cfg.epochs:
            ckpt_dir = os.path.join(cfg.save_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"pg_finetuned_ep{epoch}.pth")
            
            torch.save(base_policy.state_dict(), ckpt_path)
            print(f"   💾 Saved Checkpoint to {ckpt_path}")
            # 调用外置的评估接口
            evaluate_and_record_video(cfg, base_policy, epoch, device, normalizer=normalizer)

    if cfg.wandb.enable:
        wandb.finish()
    print("🎉 在线 PG 微调完成！可以接入 distill_policy.py 开始进行单步蒸馏！")


if __name__ == "__main__":
    main()