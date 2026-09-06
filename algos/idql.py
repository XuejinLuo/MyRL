# algos/idql.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

# ============================================================================
# 1. Mock 模块 (在你的真实 Infra 中，这些将被 pointnext.py 和 policy.py 替换)
# ============================================================================

class MockPointEncoder(nn.Module):
    """ 简易的点云编码器 (替代 PointNeXt/PointNet++) """
    def __init__(self, in_channels=3, feature_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim)
        )
    def forward(self, pc):
        # pc shape: [B, N, C]
        features = self.net(pc)           # [B, N, feature_dim]
        global_feature = features.max(dim=1)[0] # Max Pooling -> [B, feature_dim]
        return global_feature

class MockFlowDiffusionPolicy(nn.Module):
    """ 简易的 Flow/Diffusion 策略 (用于占位) """
    def __init__(self, pc_feature_dim=256, chunk_size=16, action_dim=7):
        super().__init__()
        self.encoder = MockPointEncoder(3, pc_feature_dim)
        # 假设这是一个条件生成模型，这里只 Mock 它的 Loss 计算过程
        self.mock_net = nn.Sequential(
            nn.Linear(pc_feature_dim + chunk_size * action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def compute_loss(self, states, actions):
        """ Flow/Diffusion 的前向 Loss 计算 (如 Flow Matching Loss 或 DDPM Denoise Loss) """
        state_feat = self.encoder(states)       # [B, F]
        action_flat = actions.view(actions.size(0), -1) # [B, Chunk * A_dim]
        x = torch.cat([state_feat, action_flat], dim=-1)
        # Mock loss: 让网络输出逼近 0
        pred_noise = self.mock_net(x)
        loss = F.mse_loss(pred_noise, torch.zeros_like(pred_noise))
        return loss

# ============================================================================
# 2. IDQL 核心网络组件 (Critic & Value)
# ============================================================================

class ValueNetwork(nn.Module):
    def __init__(self, pc_dim=3, feature_dim=256):
        super().__init__()
        self.encoder = MockPointEncoder(pc_dim, feature_dim)
        self.v_net = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state):
        feat = self.encoder(state)
        return self.v_net(feat)

class TwinQNetwork(nn.Module):
    def __init__(self, pc_dim=3, chunk_size=16, action_dim=7, feature_dim=256):
        super().__init__()
        self.encoder = MockPointEncoder(pc_dim, feature_dim)
        
        # Q1 Network
        self.q1_net = nn.Sequential(
            nn.Linear(feature_dim + chunk_size * action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Q2 Network
        self.q2_net = nn.Sequential(
            nn.Linear(feature_dim + chunk_size * action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action_chunk):
        feat = self.encoder(state)
        action_flat = action_chunk.view(action_chunk.size(0), -1)
        x = torch.cat([feat, action_flat], dim=-1)
        
        q1 = self.q1_net(x)
        q2 = self.q2_net(x)
        return q1, q2

# ============================================================================
# 3. IDQL 算法核心类
# ============================================================================

class IDQL:
    """ 
    Implicit Diffusion Q-Learning (基于 3D Point Cloud 与 Action Chunk) 
    支持通过 Reject Sampling (拒绝采样) 训练 Flow/Diffusion Actor。
    """
    def __init__(
        self,
        actor: nn.Module,
        q_network: nn.Module,
        v_network: nn.Module,
        device="cuda",
        tau=0.7,             # Expectile 回归系数 (IQL 核心参数)
        discount=0.99,
        beta=3.0,            # Reject Sampling 温度系数
        actor_lr=1e-4,
        critic_lr=3e-4,
        tau_target=0.005,     # Target 网络软更新系数
        use_bc_only=False
    ):
        self.actor = actor.to(device)
        self.q_net = q_network.to(device)
        self.v_net = v_network.to(device)
        self.q_target = copy.deepcopy(self.q_net).to(device)
        
        self.device = device
        self.tau = tau
        self.discount = discount
        self.beta = beta
        self.tau_target = tau_target
        self.use_bc_only = use_bc_only
        
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.q_opt = torch.optim.Adam(self.q_net.parameters(), lr=critic_lr)
        self.v_opt = torch.optim.Adam(self.v_net.parameters(), lr=critic_lr)

    def expectile_loss(self, diff, expectile):
        """ Asymmetric L2 Loss (IQL 用于拟合 V 网络) """
        weight = torch.where(diff > 0, expectile, (1 - expectile))
        return weight * (diff ** 2)

    def update_critic(self, states, actions, rewards, next_states, dones):
        """ 离线阶段：更新 Q 和 V 网络 """
        with torch.no_grad():
            target_q1, target_q2 = self.q_target(states, actions)
            target_q = torch.minimum(target_q1, target_q2)
            next_v = self.v_net(next_states)
            
        # 1. 更新 V 网络 (Expectile Regression)
        v = self.v_net(states)
        adv = target_q - v
        v_loss = self.expectile_loss(adv, self.tau).mean()
        
        self.v_opt.zero_grad()
        v_loss.backward()
        self.v_opt.step()
        
        # 2. 更新 Q 网络 (TD Learning)
        q_target_value = rewards + self.discount * (1.0 - dones) * next_v.detach()
        q1, q2 = self.q_net(states, actions)
        
        q1_loss = F.mse_loss(q1, q_target_value)
        q2_loss = F.mse_loss(q2, q_target_value)
        q_loss = q1_loss + q2_loss
        
        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()
        
        # 3. Soft update Target Q Network
        for param, target_param in zip(self.q_net.parameters(), self.q_target.parameters()):
            target_param.data.copy_(self.tau_target * param.data + (1 - self.tau_target) * target_param.data)
            
        return {"v_loss": v_loss.item(), "q_loss": q_loss.item()}

    def update_actor(self, states, actions):
        """ 提取阶段：使用 Reject Sampling 训练 Flow/Diffusion 策略 """
        with torch.no_grad():
            v = self.v_net(states)
            q1, q2 = self.q_target(states, actions)
            q = torch.minimum(q1, q2)
            
            # 计算 Advantage
            adv = q - v
            
            # [🔥核心修复] 减去 advantage 的最大值，防止 torch.exp 指数爆炸变为 NaN
            adv_stable = adv - adv.max() 
            
            # Reject Sampling 核心概率计算: exp(beta * adv) 归一化到 [0, 1]
            weights = torch.exp(self.beta * adv_stable)
            max_weight = weights.max()
            accept_prob = (weights / max_weight).squeeze(-1) # [B]
            
            # 生成均匀分布 U(0,1) 进行过滤
            random_u = torch.rand_like(accept_prob)
            keep_mask = random_u < accept_prob
            
            # 安全校验: 确保至少保留一个样本，避免 Loss 变成 NaN
            if keep_mask.sum() == 0:
                keep_mask[torch.argmax(accept_prob)] = True

        # 仅对被保留的高质量样本计算 Diffusion/Flow Loss
        filtered_states = states[keep_mask]
        filtered_actions = actions[keep_mask]
        
        actor_loss = self.actor.compute_loss(filtered_states, filtered_actions)
        
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        
        return {
            "actor_loss": actor_loss.item(), 
            "accept_ratio": keep_mask.float().mean().item(),
            "adv_mean": adv.mean().item()
        }

# ============================================================================
# 4. 测试与验证 (直接运行本文件)
# ============================================================================
if __name__ == "__main__":
    print("🚀 启动 IDQL (Point Cloud + Action Chunk) 架构验证...")
    
    # --- 参数设定 ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 32
    NUM_POINTS = 1024
    PC_DIM = 3            # (X, Y, Z)
    CHUNK_SIZE = 16       # Action Chunk 长度 (Temporal Window)
    ACTION_DIM = 7        # 机器人动作维度 (如 6 DoF + 1 Gripper)
    FEATURE_DIM = 128     # 点云提取后的特征维度
    
    # --- 初始化网络组件 ---
    actor = MockFlowDiffusionPolicy(FEATURE_DIM, CHUNK_SIZE, ACTION_DIM)
    v_net = ValueNetwork(PC_DIM, FEATURE_DIM)
    q_net = TwinQNetwork(PC_DIM, CHUNK_SIZE, ACTION_DIM, FEATURE_DIM)
    
    # --- 初始化 IDQL Agent ---
    agent = IDQL(
        actor=actor,
        q_network=q_net,
        v_network=v_net,
        device=DEVICE
    )
    
    # --- 生成 Dummy 数据 ---
    print("\n📦 生成 Batch 数据...")
    states = torch.randn(BATCH_SIZE, NUM_POINTS, PC_DIM).to(DEVICE)
    next_states = torch.randn(BATCH_SIZE, NUM_POINTS, PC_DIM).to(DEVICE)
    actions = torch.randn(BATCH_SIZE, CHUNK_SIZE, ACTION_DIM).to(DEVICE)
    rewards = torch.rand(BATCH_SIZE, 1).to(DEVICE)
    dones = torch.zeros(BATCH_SIZE, 1).to(DEVICE)
    
    print(f"  > States Shape: {states.shape}")
    print(f"  > Actions Shape: {actions.shape}")
    
    # --- 运行一次离线 Q/V 网络更新 (Critic Step) ---
    print("\n🔄 [Step 1] 测试 Critic & Value (IQL Expectile Regression) 更新...")
    critic_info = agent.update_critic(states, actions, rewards, next_states, dones)
    print(f"  ✅ Q Loss: {critic_info['q_loss']:.4f}")
    print(f"  ✅ V Loss: {critic_info['v_loss']:.4f}")
    
    # --- 运行一次策略提取更新 (Actor Step with Reject Sampling) ---
    print("\n🔄 [Step 2] 测试 Actor 拒绝采样 (Reject Sampling) 与 Policy 训练...")
    actor_info = agent.update_actor(states, actions)
    print(f"  ✅ Actor Loss: {actor_info['actor_loss']:.4f}")
    print(f"  ✅ 样本保留率 (Accept Ratio): {actor_info['accept_ratio']*100:.1f}%")
    print(f"  ✅ 平均 Advantage: {actor_info['adv_mean']:.4f}")
    
    print("\n🎉 架构测试通过！代码已修复数值稳定性，可直接用于精简版 Infra！")