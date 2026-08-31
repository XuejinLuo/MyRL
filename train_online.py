# train_online.py
import os
import torch
import torch.nn as nn
import numpy as np
import hydra
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

# =========================================================================
# 桥接层 (Bridge Wrappers)
# PG 更新要求 Tensor 输入，而我们的环境吐出的是 Dict。这里用 Wrapper 进行解包对齐。
# =========================================================================

class CriticFeatureExtractor(nn.Module):
    """ 为 Critic 专门提供 3D 点云特征提取 """
    def __init__(self, cfg):
        super().__init__()
        self.encoder = PointNeXtEncoder(
            in_channels=3,
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
        
    def evaluate_actions(self, obs_dict, actions):
        # 匹配 algos/pg.py 中 trainer.update_step 的调用
        return self.policy.evaluate_actions(
            obs=obs_dict['pc'], 
            actions=actions, 
            state=obs_dict['state']
        )
        
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
def make_env_dummy(cfg: DictConfig):
    # 替换为你自己真实的 Gym Env
    class DummyEmbodiedEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = gym.spaces.Dict({
                'xyz': gym.spaces.Box(-10, 10, shape=(cfg.env.n_original_points, 3), dtype=np.float32),
                'rgb': gym.spaces.Box(0, 1, shape=(cfg.env.n_original_points, 3), dtype=np.float32),
                'state': gym.spaces.Box(-1, 1, shape=(cfg.model.state_dim,), dtype=np.float32)
            })
            self.action_space = gym.spaces.Box(-1, 1, shape=(cfg.model.action_dim,), dtype=np.float32)
            self.step_count = 0
            
        def reset(self, seed=None, options=None):
            self.step_count = 0
            return self.observation_space.sample(), {}
            
        def step(self, action):
            self.step_count += 1
            done = self.step_count >= 50
            reward = 1.0 if done else np.random.rand() * 0.1
            return self.observation_space.sample(), reward, done, False, {"success": done}
            
    env = DummyEmbodiedEnv()

    # 包装点云处理器
    ws_bounds = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
    env = PointCloudObservationWrapper(
        env=env, num_points=cfg.env.num_points, workspace_bounds=ws_bounds, use_color=cfg.env.use_color
    )

    # 包装动作 Chunk 处理器 (含 Temporal Ensembling)
    env = ChunkActionWrapper(
        env=env, chunk_size=cfg.model.chunk_size, exec_steps=cfg.env.exec_steps, exp_weight=cfg.env.exp_weight
    )
    return env

def make_env_ManiSkill(cfg):
    """ 创建并包装 ManiSkill 真实仿真环境 """
    
    # 1. 实例化 ManiSkill 环境 (以最经典的抓取方块任务为例)
    env = gym.make(
        "PickCube-v1",
        obs_mode="pointcloud",           # 必须开启点云模式
        control_mode="pd_joint_pos",    # 控制模式为关节位置
        render_mode="rgb_array",         # 用于 evaluation 录制视频
    )
    
    # 2. 接入 ManiSkill 数据适配器 (转换为 {'xyz', 'rgb', 'state'})
    env = ManiSkillToRL100Wrapper(env)
    
    # 3. 接入你原来写好的 PointCloud Wrapper
    # 注意：这里的 workspace_bounds 需要根据 PickCube 任务的实际桌面范围做调整
    # 例如只保留桌面上的物体和机械臂部分点云，剔除背景
    ws_bounds = np.array([
        [-0.5, -0.5, 0.0],  # [X_min, Y_min, Z_min] (Z > 0 保留桌面以上)
        [ 0.5,  0.5, 0.5]   # [X_max, Y_max, Z_max]
    ])
    
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
    # 1. 实验追踪
    if cfg.wandb.enable:
        wandb.init(project=cfg.wandb.project, name=cfg.run_name, config=OmegaConf.to_container(cfg, resolve=True))
        
    device = torch.device(cfg.device)
    os.makedirs(cfg.save_dir, exist_ok=True)

    # 2. 初始化环境
    # env = make_env_dummy(cfg)
    env = make_env_ManiSkill(cfg)
    
    # 3. 初始化模型组件
    print("🧠 初始化 Actor (Policy) 与 Critic (Value Network)...")
    base_policy = EmbodiedGenPolicy(
        action_dim=cfg.model.action_dim, chunk_size=cfg.model.chunk_size,
        use_state=cfg.model.use_state, state_dim=cfg.model.state_dim,
        encoder_type=cfg.model.encoder_type, backbone_type=cfg.model.backbone_type,
        cond_dim=cfg.model.cond_dim, algo_type=cfg.model.algo_type
    ).to(device)
    actor = Policy_PG_Wrapper(base_policy)
    
    # 假设我们在此处加载通过 IDQL 预训练好的权重作为起点 (极其重要)
    if cfg.pg.pretrained_ckpt and os.path.exists(cfg.pg.pretrained_ckpt):
        base_policy.load_state_dict(torch.load(cfg.pg.pretrained_ckpt, map_location=device))
        print("✅ 成功加载 IDQL 离线预训练权重，在此基础上启动 PG 微调！")

    critic_encoder = CriticFeatureExtractor(cfg).to(device)
    base_v_net = VNetwork(state_dim=cfg.model.cond_dim).to(device)
    critic = Critic_PG_Wrapper(critic_encoder, base_v_net)
    
    # 4. 初始化 PG Trainer
    trainer = FlowPolicyGradient(
        policy=actor, critic=critic,
        actor_lr=cfg.algo.actor_lr,        # 统一使用 cfg.algo
        critic_lr=cfg.algo.critic_lr,
        gamma=cfg.algo.gamma, 
        gae_lambda=cfg.algo.gae_lambda, 
        clip_ratio=cfg.algo.clip_ratio,
        device=device
    )

    buffer = RolloutBuffer(device)
    obs, _ = env.reset()

    # 5. 主循环 (Epochs)
    print(f"🚀 开始在线强化微调 (Total Epochs: {cfg.epochs})")
    
    for epoch in range(1, cfg.epochs + 1):
        buffer.clear()
        epoch_reward = 0.0
        
        # --- A. 轨迹收集阶段 (Rollout) ---
        actor.eval()
        critic.eval()
        
        for _ in tqdm(range(cfg.pg.steps_per_epoch), desc=f"Epoch {epoch} Rollout", leave=False):
            # 将 numpy 观测转为单 Batch Tensor
            obs_dict = {
                'pc': torch.FloatTensor(obs['point_cloud']).unsqueeze(0).to(device),
                'state': torch.FloatTensor(obs['state']).unsqueeze(0).to(device)
            }
            
            with torch.no_grad():
                # 计算当前状态的 Value
                value = critic(obs_dict).item()
                # 策略前向生成动作 Chunk
                action_chunk = actor.sample(obs_dict, num_steps=cfg.model.num_inference_steps)
            
            # 环境执行 (转换为 numpy)
            action_np = action_chunk.squeeze(0).cpu().numpy()
            next_obs, reward, done, truncated, info = env.step(action_np)
            
            epoch_reward += reward
            
            # 存储轨迹
            buffer.add(obs['point_cloud'], obs['state'], action_np, reward, done or truncated, value)
            
            obs = next_obs
            if done or truncated:
                obs, _ = env.reset()
                
        # --- B. 计算 GAE 与 Returns ---
        rollout_data = buffer.get_tensors()
        
        # 获取下一个状态的 value (用于 Bootstrapping)
        with torch.no_grad():
            next_obs_dict = {
                'pc': torch.FloatTensor(obs['point_cloud']).unsqueeze(0).to(device),
                'state': torch.FloatTensor(obs['state']).unsqueeze(0).to(device)
            }
            next_value = critic(next_obs_dict).squeeze(-1)
            
        advantages, returns = trainer.compute_gae(
            rollout_data['rewards'], rollout_data['values'], rollout_data['dones'], next_value
        )
        
        # PPO Trick: Advantage 归一化 (极大提升微调稳定性)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --- C. PPO 网络更新阶段 ---
        actor.train()
        critic.train()
        
        # 将 [Time, Batch, ...] 拍扁为 [Batch_size, ...]
        flat_obs_dict = {
            'pc': rollout_data['pc'].view(-1, cfg.env.num_points, 3),
            'state': rollout_data['state'].view(-1, cfg.model.state_dim)
        }
        flat_actions = rollout_data['actions'].view(-1, cfg.model.chunk_size, cfg.model.action_dim)
        flat_adv = advantages.view(-1)
        flat_returns = returns.view(-1)

        # 事先计算 old_log_probs
        with torch.no_grad():
            old_log_probs, _ = actor.evaluate_actions(flat_obs_dict, flat_actions)

        # PPO 循环微调
        epoch_losses = {}
        for ppo_epoch in range(cfg.pg.update_epochs):
            # 这里简单起见做整批更新（如果 GPU 显存不够，可在此加一个 Mini-Batch for 循环切片）
            loss_dict = trainer.update_step(
                states=flat_obs_dict, 
                actions=flat_actions, 
                old_log_probs=old_log_probs, 
                returns=flat_returns, 
                advantages=flat_adv
            )
            
            for k, v in loss_dict.items():
                epoch_losses[k] = epoch_losses.get(k, 0) + v
                
        # 平均 Loss
        for k in epoch_losses:
            epoch_losses[k] /= cfg.pg.update_epochs

        # --- D. 打印与保存 ---
        log_metrics = {**epoch_losses, "Reward/Epoch": epoch_reward}
        if cfg.wandb.enable:
            wandb.log(log_metrics, step=epoch)
            
        print(f"Epoch {epoch:03d} | Avg Reward: {epoch_reward:.2f} | Actor Loss: {epoch_losses['actor_loss']:.4f} | Critic Loss: {epoch_losses['critic_loss']:.4f}")

        if epoch % 10 == 0 or epoch == cfg.epochs:
            ckpt_path = os.path.join(cfg.save_dir, f"pg_finetuned_ep{epoch}.pth")
            torch.save(base_policy.state_dict(), ckpt_path)
            print(f"   💾 Saved Checkpoint to {ckpt_path}")

    if cfg.wandb.enable:
        wandb.finish()
    print("🎉 在线 PG 微调完成！可以接入 distill_policy.py 开始进行单步蒸馏！")

# =========================================================================
# 用于直接运行代码的虚拟配置 (Dummy Config)
# 实际运行请将这些放入 configs/train_online.yaml 中交由 Hydra 解析
# =========================================================================
if __name__ == "__main__":
    from omegaconf import OmegaConf
    dummy_yaml = """
    run_name: "Flow-PG-Finetune"
    device: "cuda"
    save_dir: "./checkpoints/online"
    epochs: 200
    
    env:
      n_original_points: 2048
      num_points: 1024
      use_color: false
      exec_steps: 2
      exp_weight: 0.01

    model:
      action_dim: 7
      state_dim: 14
      chunk_size: 8
      use_state: true
      encoder_type: "pointnext"
      backbone_type: "transformer"
      algo_type: "flow"
      cond_dim: 256
      num_inference_steps: 10
      
    pg:
      steps_per_epoch: 64
      update_epochs: 4
      actor_lr: 1e-5
      critic_lr: 3e-4
      gamma: 0.99
      gae_lambda: 0.95
      clip_ratio: 0.2
      pretrained_ckpt: "" # "./checkpoints/offline/idql_policy_best.pth"
      
    wandb:
      enable: false
      project: "RL-100-Infra"
    """
    
    # 覆盖 Hydra 劫持逻辑，直接解析字符串执行
    cfg = OmegaConf.create(dummy_yaml)
    main(cfg)