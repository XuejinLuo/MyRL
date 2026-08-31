# models/critics/q_v_network.py
import torch
import torch.nn as nn
import torch.nn.functional as F
def weight_init(m):
    """用于 RL 的正交初始化"""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)
            
class MLP(nn.Module):
    """
    基础的多层感知机 (MLP) 模块，用于构建 Q 和 V 网络。
    """
    def __init__(self, input_dim, hidden_dims, output_dim, activation=nn.GELU):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim)) # 使用 LayerNorm 提升离线 RL 训练稳定性
            layers.append(activation())
            in_dim = h_dim
        # 最后一层
        last_layer = nn.Linear(in_dim, output_dim)
        # 针对 RL Critic 的 Trick：最后一层权重初始化为一个极小的值 (如 3e-3)
        # 这样网络刚开始输出的 Q/V 值接近 0，防止启动时 advantage 爆炸
        nn.init.uniform_(last_layer.weight, -3e-3, 3e-3)
        nn.init.uniform_(last_layer.bias, -3e-3, 3e-3)
        
        layers.append(last_layer)
        self.net = nn.Sequential(*layers)
        
        # 对除最后一层外的其他层应用正交初始化
        self.net[:-1].apply(weight_init)

    def forward(self, x):
        return self.net(x)


class VNetwork(nn.Module):
    """
    V(s) 价值网络：用于 IDQL 中计算 Advantage (A = Q - V) 和 Expectile Regression。
    输入: state_embedding (Point Cloud Feature + Proprioception Feature 融合后的一维向量)
    输出: V-value (标量)
    """
    def __init__(self, state_dim, hidden_dims=[256, 256, 256]):
        super().__init__()
        self.v_net = MLP(input_dim=state_dim, hidden_dims=hidden_dims, output_dim=1)

    def forward(self, state_embedding):
        """
        Args:
            state_embedding: Tensor of shape [B, state_dim]
        Returns:
            v_value: Tensor of shape [B, 1]
        """
        return self.v_net(state_embedding)


class TwinQNetwork(nn.Module):
    """
    双 Q(s, a) 网络：用于缓解离线 RL (IDQL) 和在线 PG 中的 Q 值高估问题。
    输入: state_embedding, action_chunk
    输出: Q1, Q2 (标量)
    """
    def __init__(self, state_dim, action_dim, chunk_size, hidden_dims=[256, 256, 256]):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        
        # 因为输入的是 Chunk Action，我们将其 Flatten 为一维向量
        # 实际输入 Q 网络的维度是: state_dim + (chunk_size * action_dim)
        input_dim = state_dim + (chunk_size * action_dim)

        self.q1_net = MLP(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=1)
        self.q2_net = MLP(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=1)

    def forward(self, state_embedding, action_chunk):
        """
        Args:
            state_embedding: Tensor of shape [B, state_dim]
            action_chunk: Tensor of shape [B, chunk_size, action_dim]
        Returns:
            q1, q2: Tensors of shape [B, 1]
        """
        batch_size = state_embedding.shape[0]
        
        # 将 Chunk Action 展平 [B, chunk_size, action_dim] -> [B, chunk_size * action_dim]
        action_flat = action_chunk.view(batch_size, -1)
        
        # 拼接状态与动作 [B, state_dim + chunk_size * action_dim]
        sa_cat = torch.cat([state_embedding, action_flat], dim=1)
        
        q1 = self.q1_net(sa_cat)
        q2 = self.q2_net(sa_cat)
        return q1, q2

    def get_q1(self, state_embedding, action_chunk):
        """仅在某些策略更新时(如PG/Distill)需要单Q值时调用"""
        batch_size = state_embedding.shape[0]
        action_flat = action_chunk.view(batch_size, -1)
        sa_cat = torch.cat([state_embedding, action_flat], dim=1)
        return self.q1_net(sa_cat)


# ==========================================
# 下方为 Mock 的状态编码器，仅用于测试展示整体的数据流
# 实际项目中，你需要从 models/encoders 导入你的 PointNeXt
# ==========================================
class MockStateEncoder(nn.Module):
    """
    模拟一个 3D 点云 + 本体感觉的编码器
    用于将 Point Cloud [B, N, 3+C] 和 Proprioception [B, P] 映射为 State Embedding
    """
    def __init__(self, proprio_dim, state_dim=512):
        super().__init__()
        # 假设点云经过 PointNeXt 提取出 256 维特征
        self.point_feature_dim = 256
        self.proprio_proj = nn.Linear(proprio_dim, 128)
        self.fusion_mlp = MLP(input_dim=self.point_feature_dim + 128, 
                              hidden_dims=[256], 
                              output_dim=state_dim)

    def forward(self, point_cloud, proprio):
        # 模拟点云特征提取 (假装用 PointNeXt 提取了 Global Feature)
        batch_size = point_cloud.shape[0]
        mock_point_features = torch.randn(batch_size, self.point_feature_dim).to(point_cloud.device)
        
        # 本体特征映射
        proprio_feat = F.relu(self.proprio_proj(proprio))
        
        # 拼接并融合
        fused = torch.cat([mock_point_features, proprio_feat], dim=1)
        state_embedding = self.fusion_mlp(fused)
        return state_embedding


if __name__ == "__main__":
    print("="*50)
    print("Testing Q and V Networks for IDQL & PG with Chunk Action")
    print("="*50)

    # 1. 定义超参数 (假设针对桌面机械臂抓取任务)
    BATCH_SIZE = 16
    NUM_POINTS = 1024       # 降采样后的点云数量
    POINT_DIM = 3           # [X, Y, Z]
    PROPRIO_DIM = 14        # 本体感觉 (7DoF 关节角 + 7DoF 速度等)
    CHUNK_SIZE = 8          # Action Chunk 大小 (未来 8 步轨迹)
    ACTION_DIM = 7          # 动作维度 (6DoF 位姿 + 1DoF 夹爪)
    STATE_EMBED_DIM = 512   # 融合后的 State Embedding 维度

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # 2. 初始化网络
    state_encoder = MockStateEncoder(proprio_dim=PROPRIO_DIM, state_dim=STATE_EMBED_DIM).to(DEVICE)
    v_network = VNetwork(state_dim=STATE_EMBED_DIM).to(DEVICE)
    twin_q_network = TwinQNetwork(
        state_dim=STATE_EMBED_DIM, 
        action_dim=ACTION_DIM, 
        chunk_size=CHUNK_SIZE
    ).to(DEVICE)

    # 3. 构造 Dummy Data (模拟从 Replay Buffer 中取出的一个 Batch)
    dummy_points = torch.randn(BATCH_SIZE, NUM_POINTS, POINT_DIM).to(DEVICE)
    dummy_proprio = torch.randn(BATCH_SIZE, PROPRIO_DIM).to(DEVICE)
    dummy_action_chunk = torch.randn(BATCH_SIZE, CHUNK_SIZE, ACTION_DIM).to(DEVICE)

    print("\n[Input Shapes]")
    print(f"Point Cloud   : {dummy_points.shape}")
    print(f"Proprioception: {dummy_proprio.shape}")
    print(f"Action Chunk  : {dummy_action_chunk.shape}")

    # 4. 前向传播测试
    # Step 4.1: 提取 State Embedding (在 RL 训练 step 中只需要提取一次！)
    state_embedding = state_encoder(dummy_points, dummy_proprio)
    print(f"\n[Intermediate] State Embedding shape: {state_embedding.shape}")

    # Step 4.2: 计算 V(s) (用于 IDQL Advantage 计算)
    v_values = v_network(state_embedding)
    print(f"[Output] V_values shape: {v_values.shape}")

    # Step 4.3: 计算 Q1(s, a_chunk), Q2(s, a_chunk) (用于 Critic 更新)
    q1, q2 = twin_q_network(state_embedding, dummy_action_chunk)
    print(f"[Output] Q1 shape: {q1.shape}")
    print(f"[Output] Q2 shape: {q2.shape}")

    # 5. 模拟一个简单的 IDQL Loss 计算过程 (验证梯度流通)
    print("\n[Testing Backprop / Gradient flow (Mock IDQL logic)]")
    # 假设我们要更新 V 网络 (Expectile Regression)
    # Advantage = Q - V
    # 这里用 Q1 模拟 Target Q
    with torch.no_grad():
        target_q, _ = twin_q_network(state_embedding, dummy_action_chunk)
    
    adv = target_q - v_values
    expectile = 0.7
    # Expectile loss: |tau - I(adv < 0)| * adv^2
    weight = torch.where(adv > 0, expectile, (1 - expectile))
    v_loss = (weight * (adv ** 2)).mean()
    
    v_loss.backward()
    print(f"V_loss: {v_loss.item():.4f}")
    
    # 检查梯度是否正常生成
    v_has_grad = any(p.grad is not None for p in v_network.parameters())
    print(f"VNetwork has gradients after backward: {v_has_grad}")
    
    print("\n=== Test Passed Successfully! ===")