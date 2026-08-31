# models/encoders/pointnext.py
"""
PointNeXt Encoder (Pure PyTorch Version v1.0)

【当前版本的优点】
1. 开箱即用：完全由纯 PyTorch 实现，摒弃了复杂的 C++/CUDA 扩展编译（如 pointnet2_ops）。
2. 高度便携：跨平台兼容性极强，适合前期快速跑通 Behavior Cloning (BC) 和 IDQL 等算法的训练流。

【🚨 当前版本的缺点与隐患】
1. 致命的推理延迟 (FPS Bottleneck)：
   `farthest_point_sample` 包含纯 Python 的 for 循环。在 GPU 上运行时，会导致大量的 Kernel 启动与显存通信开销。
   如果用于 One-step Distillation 的高频真机部署（如 50Hz），这里的耗时（可能达 20~50ms）将抹杀蒸馏带来的速度优势。
2. OOM 显存爆炸风险：
   `square_distance` 的空间复杂度为 O(B * N * M)。如果直接传入未降采样的高密度点云（如 >4096 点），极易导致显存溢出。
3. 空间拓扑信息的丢失：
   网络末端使用了 Global Average Pooling，将 3D 点云压缩成了 1D 向量。这在传统的 Diffusion Policy 中可用，
   但在追求极致精度的 3D Diffusion Policy (DP3) 中，会丢失部分精细的三维拓扑特征。

【🚀 未来部署与优化的改进方向】
1. 算子替换 (Speedup)：在真机部署或追求极限推理速度时，务必将 `farthest_point_sample` 和 `query_ball_point`
   替换为 C++/CUDA 预编译版本（如使用 `torch-cluster` 库的 fps，或第三方 `pointnet2_ops`）。
2. 前置降采样 (Memory)：在 envs/pointcloud_wrapper.py 中强制约束，输入前必须通过 Open3D 等进行 Voxel Downsample，
   将点数控制在 1024 或 2048 以下。
3. 保留 Token 序列 (Architecture)：如果抓取任务对三维空间精度要求极高，考虑去掉 `torch.mean`，
   将 `l3_points` 展平为 token 序列 (例如 [B, 16, 256])，通过 Cross-Attention 注入到后续的 Transformer/UNet 中。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# =====================================================================
# 1. 基础点云算子 (Pure PyTorch Implementation)
# =====================================================================

def square_distance(src, dst):
    """计算两组点之间的平方距离"""
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.transpose(1, 2))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """根据索引提取点"""
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    """最远点采样 (FPS)"""
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
    """球查询 (Ball Query)"""
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def sample_and_group(npoint, radius, nsample, xyz, points):
    """组合采样与局部特征提取"""
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, 3)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
    return new_xyz, new_points

# =====================================================================
# 2. PointNeXt 核心模块
# =====================================================================

class SetAbstraction(nn.Module):
    """PointNet++ 风格的 Set Abstraction 层，用于降采样和空间特征聚合"""
    def __init__(self, npoint, radius, nsample, in_channel, mlp):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        new_points = new_points.permute(0, 3, 2, 1) # [B, C, nsample, npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        new_points = torch.max(new_points, 2)[0]    # [B, C, npoint]
        return new_xyz, new_points.permute(0, 2, 1) # 返回 [B, npoint, C]

class InvResMLP(nn.Module):
    """PointNeXt 核心贡献：Inverted Residual MLP (倒置残差结构)"""
    def __init__(self, channels, expansion=4):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels * expansion, 1)
        self.bn1 = nn.BatchNorm1d(channels * expansion)
        self.conv2 = nn.Conv1d(channels * expansion, channels, 1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        
    def forward(self, x):
        # x shape: [B, C, N]
        identity = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.act(x + identity)

# =====================================================================
# 3. 供 Flow/Diffusion Policy 使用的 PointNeXt Encoder 主类
# =====================================================================

class PointNeXtEncoder(nn.Module):
    def __init__(
        self, 
        in_channels=3,       # 如果只有 XYZ 则是 3；如果带 RGB 则是 6
        output_dim=256,      # 最终输出的全局特征维度 (供 Diffusion/Flow 条件使用)
        use_state=False,     # 是否拼接机器人本体状态 (Proprioception)
        state_dim=0          # 本体状态的维度
    ):
        """
        专为 Flow / 3D Diffusion + Chunk Action 打造的点云编码器
        使用 PointNeXt-S 的轻量化设计，保证推理速度。
        """
        super().__init__()
        self.in_channels = in_channels
        self.use_state = use_state
        self.state_dim = state_dim
        
        # Initial MLP
        self.conv1 = nn.Conv1d(in_channels, 32, 1)
        self.bn1 = nn.BatchNorm1d(32)
        
        # Stage 1: SA + InvResMLP (1024 -> 256 points)
        self.sa1 = SetAbstraction(npoint=256, radius=0.1, nsample=32, in_channel=32+3, mlp=[32, 32, 64])
        self.inv_res1 = InvResMLP(64)
        
        # Stage 2: SA + InvResMLP (256 -> 64 points)
        self.sa2 = SetAbstraction(npoint=64, radius=0.2, nsample=32, in_channel=64+3, mlp=[64, 64, 128])
        self.inv_res2 = InvResMLP(128)
        
        # Stage 3: SA + InvResMLP (64 -> 16 points)
        self.sa3 = SetAbstraction(npoint=16, radius=0.4, nsample=32, in_channel=128+3, mlp=[128, 128, 256])
        self.inv_res3 = InvResMLP(256)
        
        # Global Pooling + Final Projection
        self.fc = nn.Sequential(
            nn.Linear(256 + (state_dim if use_state else 0), 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, obs_dict):
        """
        输入 obs_dict 必须包含:
            - 'point_cloud': Tensor, shape [B, N, C]
            - 'state' (可选): Tensor, shape [B, state_dim]
        返回:
            - global_feat: Tensor, shape [B, output_dim]
        """
        pc = obs_dict['point_cloud'] 
        B, N, C_in = pc.shape
        
        # 【防御性编程】防止纯 PyTorch 算子导致 OOM 和 FPS 逻辑崩坏
        assert N >= 256, f"输入点云数量 N={N} 必须 >= 256，否则 Stage 1 无法正常降采样。"
        assert N <= 4096, f"输入点云数量 N={N} 必须 <= 4096，否则纯 PyTorch 矩阵乘法极易引发显存 OOM。"

        # 如果输入带有比 in_channels 更多的特征（防止报错），只取前 in_channels 维
        xyz = pc[:, :, :3].contiguous()
        features = pc[:, :, :self.in_channels].permute(0, 2, 1).contiguous() # [B, in_channels, N]
        
        # Stem
        features = F.relu(self.bn1(self.conv1(features))) # [B, 32, N]
        features = features.permute(0, 2, 1).contiguous() # [B, N, 32]
        
        # Stage 1
        l1_xyz, l1_points = self.sa1(xyz, features)     # [B, 256, 64]
        l1_points = l1_points.permute(0, 2, 1)          # [B, 64, 256]
        l1_points = self.inv_res1(l1_points)
        l1_points = l1_points.permute(0, 2, 1)          # [B, 256, 64]
        
        # Stage 2
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points) # [B, 64, 128]
        l2_points = l2_points.permute(0, 2, 1)          # [B, 128, 64]
        l2_points = self.inv_res2(l2_points)
        l2_points = l2_points.permute(0, 2, 1)          # [B, 64, 128]
        
        # Stage 3
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points) # [B, 16, 256]
        l3_points = l3_points.permute(0, 2, 1)          # [B, 256, 16]
        l3_points = self.inv_res3(l3_points)            # [B, 256, 16]
        
        # Global Average Pooling
        global_feat = torch.mean(l3_points, dim=-1)     # [B, 256]
        
        # 融合机器人本体状态 (Proprioception)
        if self.use_state and 'state' in obs_dict:
            state = obs_dict['state']                   # [B, state_dim]
            global_feat = torch.cat([global_feat, state], dim=-1)
            
        # 最终映射为 Flow/Diffusion 接受的条件向量维度
        out = self.fc(global_feat)                      # [B, output_dim]
        
        return out


# =====================================================================
# 本地测试脚本
# =====================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("🚀 测试启动: PointNeXtEncoder (Pure PyTorch V1.0)")
    print("=" * 60)

    # 1. 自动选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] 运行设备: {device}")

    # 2. 模拟超参数与输入数据 (例如 1024 个点, 本体状态 14 维)
    B, N, C = 2, 1024, 3 
    state_dim = 14
    output_dim = 256

    print(f"[*] 生成模拟数据: Batch Size={B}, 点云数={N}, 坐标通道={C}, 状态维数={state_dim}")
    # 注意：真实环境的点云坐标通常在 (-1, 1) 左右，我们随机生成标准正态分布
    dummy_pc = torch.randn(B, N, C, device=device) 
    dummy_state = torch.randn(B, state_dim, device=device)
    
    obs_dict = {
        'point_cloud': dummy_pc,
        'state': dummy_state
    }

    # 3. 初始化模型
    model = PointNeXtEncoder(
        in_channels=C, 
        output_dim=output_dim, 
        use_state=True, 
        state_dim=state_dim
    ).to(device)
    model.eval() # 切换到评估模式

    # 4. 前向传播并记录耗时
    print("\n[*] 开始前向传播测试 (首次运行可能包含 CUDA 初始化开销)...")
    
    # 预热一次 (Warm-up)
    with torch.no_grad():
        _ = model(obs_dict)
    
    # 正式计时
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    start_time = time.time()
    with torch.no_grad():
        output = model(obs_dict)
        
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.time()

    elapsed_ms = (end_time - start_time) * 1000

    # 5. 打印结果
    print(f"✅ 前向传播成功！")
    print(f"   输入点云维度 : {obs_dict['point_cloud'].shape}")
    print(f"   输入状态维度 : {obs_dict['state'].shape}")
    print(f"   模型输出维度 : {output.shape} (预期为 [{B}, {output_dim}])")
    print(f"⏱️  单次推理耗时 : {elapsed_ms:.2f} ms")
    print("\n【性能提示】如果上述推理时间 > 10ms，请在未来部署时务必参考代码顶部的优化方案！")
    print("=" * 60)