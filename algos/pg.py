# algos/pg.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class FlowPolicyGradient:
    """
    针对 3D Flow/Diffusion + Chunk Action 优化的 Policy Gradient / PPO 算法。
    抛弃了复杂的跨模态兼容，专心服务于 PointCloud -> Action Chunk 的更新。
    """
    def __init__(
        self,
        policy: nn.Module,
        critic: nn.Module,
        actor_lr: float = 1e-4,
        critic_lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.01,
        v_loss_coef: float = 0.5,
        max_grad_norm: float = 1.0,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        self.policy = policy.to(device)
        self.critic = critic.to(device)
        
        self.optimizer_policy = optim.AdamW(self.policy.parameters(), lr=actor_lr, weight_decay=1e-4)
        self.optimizer_critic = optim.AdamW(self.critic.parameters(), lr=critic_lr, weight_decay=1e-4)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.v_loss_coef = v_loss_coef
        self.max_grad_norm = max_grad_norm
        self.device = device

    def compute_gae(self, rewards, values, dones, next_value):
        """
        计算广义优势估计 (Generalized Advantage Estimation)
        注意：在 Chunk Action 环境中，一步 step 可能执行了 k 步环境动作，
        这里的 rewards 应该是这 k 步的总和或折现和。
        
        【修正】: 假设输入张量的形状为 [T, B] (Time_Steps, Num_Envs)
        T 是轨迹的时间步数，B 是并行环境的数量。
        """
        # 获取时间步 T
        T = rewards.shape[0]
        advantages = torch.zeros_like(rewards).to(self.device)
        last_gae_lam = 0
        
        for t in reversed(range(T)):
            if t == T - 1:
                next_non_terminal = 1.0 - dones[t]
                next_values = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_values = values[t + 1]
                
            delta = rewards[t] + self.gamma * next_values * next_non_terminal - values[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            
        returns = advantages + values
        return advantages, returns

    def update_step(self, states, actions, old_log_probs, returns, advantages, noise=None):
        """
        执行一次 PG/PPO 网络更新
        states: 3D Point Cloud [B, N, C]
        actions: Chunk Actions [B, Chunk_Size, Action_Dim]
        """
        # 1. 评估当前策略下该动作的 log_prob 和 熵 (Entropy)
        # 注意: 对于 Flow/Diffusion, 这里内部可能会使用 Hutchinson 迹估计(CNF) 
        # 或者沿着去噪轨迹计算 per-step log_prob 累加 (类似 DDPO)。
        log_probs, entropy = self.policy.evaluate_actions(states, actions, noise=noise)
        
        # 2. 评估当前状态的 Value
        values = self.critic(states).squeeze(-1)
        
        # 3. 计算 Actor Loss (PPO Clip机制)
        log_ratio = log_probs - old_log_probs
        log_ratio = torch.clamp(log_ratio, min=-20.0, max=5.0) 
        ratio = torch.exp(log_ratio)

        with torch.no_grad():
            # 1. 近似 KL 散度 (使用更稳健的近似法)
            approx_kl = torch.mean((ratio - 1.0) - log_ratio).item()
            # 2. 截断率
            clip_fraction = torch.mean((torch.abs(ratio - 1.0) > self.clip_ratio).float()).item()
            # 3. 解释方差 (Explained Variance)
            var_y = torch.var(returns)
            explained_var = 1.0 - torch.var(returns - values) / (var_y + 1e-8)
            explained_var = explained_var.item()
        
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
        
        # 将 Actor 的 Loss 分开处理，如果是导致发散的负优势，限制其对 MSE 的无限推远
        actor_loss_raw = -torch.min(surr1, surr2)

        # 彻底丢弃负优势的样本，只保留好的行为进行梯度更新
        mask = (advantages >= 0).float()
        # 使用 sum 除以有效样本数，防止全负 batch 导致 0 除问题
        actor_loss = (actor_loss_raw * mask).sum() / (mask.sum() + 1e-8)
        
        # 4. 计算 Critic Loss (MSE)
        critic_loss = F.mse_loss(values, returns)
        
        # 5. 计算 Entropy Bonus (鼓励探索)
        entropy_loss = entropy.mean()
        
        # 6. 总 Loss
        total_loss = actor_loss + self.v_loss_coef * critic_loss - self.entropy_coef * entropy_loss
        
        # --- 梯度反向传播与更新 ---
        self.optimizer_policy.zero_grad()
        self.optimizer_critic.zero_grad()
        
        total_loss.backward()
        
        # 梯度裁剪防爆 (Flow 模型的梯度容易出现尖峰)
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        
        self.optimizer_policy.step()
        self.optimizer_critic.step()
        
        return {
            "loss/actor": actor_loss.item(),
            "loss/critic": critic_loss.item(),
            "loss/entropy": entropy_loss.item(),
            "ppo/approx_kl": approx_kl,            # <-- 监控: 新旧策略偏差
            "ppo/clip_frac": clip_fraction,        # <-- 监控: 截断比例
            "ppo/explained_var": explained_var,    # <-- 监控: Critic 拟合度
        }


# ==============================================================================
# 本地测试模块 (Mock Data & Models) 
# ==============================================================================
if __name__ == "__main__":
    print("🚀 开始测试 Flow/Diffusion + Chunk Action 的 PG 更新流...")
    
    # 模拟超参数
    TIME_STEPS = 32    # PPO Rollout 收集的时间步长 (Trajectory Length)
    NUM_ENVS = 4       # 并行环境数量 (Batch Size的一种体现)
    NUM_POINTS = 1024
    POINT_DIM = 3      # x, y, z
    CHUNK_SIZE = 8     # Action Chunk 长度
    ACTION_DIM = 7     # 7 DoF (比如 6D位姿 + 1D夹爪)
    
    # 1. Mock 策略网络 (代替复杂的 PointNeXt + Flow/Diffusion ODE)
    class MockFlowPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            # 简单的特征提取模拟
            self.net = nn.Sequential(
                nn.Linear(NUM_POINTS * POINT_DIM, 256),
                nn.ReLU(),
                nn.Linear(256, CHUNK_SIZE * ACTION_DIM * 2) # 输出 mu 和 log_std 用于模拟分布
            )
            
        def evaluate_actions(self, states, actions):
            """
            在真实的 Flow/Diffusion 中，这里会是计算 ODE 似然或去噪步骤的概率。
            为了测试 Infra 是否通顺，这里用高斯分布替代模拟。
            """
            B = states.shape[0]
            # Flatten point cloud for dummy MLP
            flat_states = states.view(B, -1)
            out = self.net(flat_states).view(B, CHUNK_SIZE, ACTION_DIM, 2)
            mu, log_std = out[..., 0], out[..., 1]
            std = torch.exp(torch.clamp(log_std, min=-20, max=2))
            
            dist = torch.distributions.Normal(mu, std)
            # 动作空间独立，将 chunk_size 和 action_dim 的对数概率求和
            log_prob = dist.log_prob(actions).sum(dim=(-1, -2)) 
            entropy = dist.entropy().sum(dim=(-1, -2))
            return log_prob, entropy

    # 2. Mock 价值网络 (Critic)
    class MockCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(NUM_POINTS * POINT_DIM, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )
        def forward(self, states):
            B = states.shape[0]
            return self.net(states.view(B, -1))

    # 3. 初始化环境、数据与 Trainer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    mock_policy = MockFlowPolicy()
    mock_critic = MockCritic()
    
    trainer = FlowPolicyGradient(
        policy=mock_policy,
        critic=mock_critic,
        device=device
    )
    
    # 4. 生成 Dummy Trajectory 数据 (模拟 On-policy 收集的轨迹序列)
    # [T, B, N_points, C]
    traj_states = torch.randn(TIME_STEPS, NUM_ENVS, NUM_POINTS, POINT_DIM, device=device)
    # [T, B, Chunk_Size, Act_Dim]
    traj_actions = torch.randn(TIME_STEPS, NUM_ENVS, CHUNK_SIZE, ACTION_DIM, device=device)
    
    # 模拟环境的奖励和结束标志 [T, B]
    traj_rewards = torch.randn(TIME_STEPS, NUM_ENVS, device=device)
    traj_dones = torch.zeros(TIME_STEPS, NUM_ENVS, device=device) # 假设没有 done
    traj_dones[-1, :] = 1.0 # 最后一个 step 强行 done

    # 获取 Values [T, B] 并计算 GAE
    # 注意: Critic 往往逐帧或将其 flatten 计算以利用并行
    with torch.no_grad():
        flat_states = traj_states.view(TIME_STEPS * NUM_ENVS, NUM_POINTS, POINT_DIM)
        flat_values = mock_critic(flat_states).squeeze(-1)
        traj_values = flat_values.view(TIME_STEPS, NUM_ENVS)
        
        # 假设下一个最终状态的 value [B]
        dummy_next_value = torch.zeros(NUM_ENVS, device=device) 
        
    advantages, returns = trainer.compute_gae(traj_rewards, traj_values, traj_dones, dummy_next_value)
    
    # Advantage 归一化 (PG标准操作，有助于稳定) - 在整个 Trajectory 上做归一化
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # 5. 打平数据并执行一次网络更新 (Flat T and B dimensions into a single Batch)
    flat_states_update = traj_states.view(-1, NUM_POINTS, POINT_DIM)
    flat_actions_update = traj_actions.view(-1, CHUNK_SIZE, ACTION_DIM)
    flat_advantages = advantages.view(-1)
    flat_returns = returns.view(-1)
    
    with torch.no_grad():
        flat_old_log_probs, _ = mock_policy.evaluate_actions(flat_states_update, flat_actions_update)

    print("执行更新前, Policy weight norm:", torch.norm(mock_policy.net[0].weight).item())
    
    # 真实 PPO 训练中，这里往往会将 flat 数据进一步做 Mini-batch 切片，这里为了演示直接整批更新
    loss_dict = trainer.update_step(
        states=flat_states_update,
        actions=flat_actions_update,
        old_log_probs=flat_old_log_probs,
        returns=flat_returns,
        advantages=flat_advantages
    )
    
    print("执行更新后, Policy weight norm:", torch.norm(mock_policy.net[0].weight).item())
    print("✅ 更新成功! Loss 数据如下:")
    for k, v in loss_dict.items():
        print(f"   - {k}: {v:.4f}")