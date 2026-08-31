# models/backbones/unet1d.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple

class SinusoidalPosEmb(nn.Module):
    """时间步 (Timestep) 的正弦位置编码，用于告诉模型当前处于 Diffusion/Flow 的哪一步"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Downsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class Conv1dBlock(nn.Module):
    """基础 1D 卷积块 (带有 GroupNorm)"""
    def __init__(self, inp_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            # GroupNorm 要求 out_channels 必须是 8 的倍数 (隐藏维度 128/256/512 都满足)
            nn.GroupNorm(8, out_channels),
            nn.SiLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class ConditionalResidualBlock1D(nn.Module):
    """
    带有 FiLM (Feature-wise Linear Modulation) 条件注入的残差块。
    """
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels),
            Conv1dBlock(out_channels, out_channels)
        ])
        self.dropout = nn.Dropout(dropout)

        self.cond_encoder = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, out_channels * 2)
        )
        
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()


    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out = self.blocks[0](x)
        
        # FiLM 注入
        cond_embed = self.cond_encoder(cond)
        cond_embed = cond_embed.unsqueeze(-1)
        scale, shift = cond_embed.chunk(2, dim=1)
        out = out * (scale + 1.0) + shift
        out = self.dropout(out)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)

class ConditionalUnet1D(nn.Module):
    """
    针对 Action Chunking 设计的 1D U-Net 主干网络。
    可直接作为 Diffusion 或 Flow Matching 的 Policy backbone。
    """
    def __init__(
        self, 
        action_dim: int, 
        global_cond_dim: int, 
        diffusion_step_embed_dim: int = 256,
        down_dims: Tuple[int, ...] = (128, 256, 512)
    ):
        super().__init__()
        self.action_dim = action_dim
        
        # 时间步的编码器
        self.step_emb = SinusoidalPosEmb(diffusion_step_embed_dim)
        self.step_mlp = nn.Sequential(
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim)
        )
        
        cond_dim = diffusion_step_embed_dim + global_cond_dim
        
        # 1. 初始卷积：把不规则的 action_dim (如14) 投影到隐藏维度(如128)，避开 GroupNorm 报错
        self.init_conv = nn.Conv1d(action_dim, down_dims[0], kernel_size=5, padding=2)
        
        # 2. 规划 U-Net 中间层的通道流动 (只在 down_dims 这种 8 的倍数的维度之间流动)
        in_out_channels = []
        dim_in = down_dims[0]
        for dim_out in down_dims:
            in_out_channels.append((dim_in, dim_out))
            dim_in = dim_out
        # 例如对于 (128, 256, 512)，得到：[(128, 128), (128, 256), (256, 512)]

        # 构建 Encoder (Down)
        self.down_modules = nn.ModuleList([])
        for i, (dim_in, dim_out) in enumerate(in_out_channels):
            is_last = i == (len(in_out_channels) - 1)
            self.down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_in, dim_out, cond_dim=cond_dim),
                ConditionalResidualBlock1D(dim_out, dim_out, cond_dim=cond_dim),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))
            
        # 中间块 (Mid)
        mid_dim = down_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim=cond_dim),
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim=cond_dim)
        ])
        
        # 构建 Decoder (Up)
        self.up_modules = nn.ModuleList([])
        for i, (dim_in, dim_out) in enumerate(reversed(in_out_channels)):
            is_last = i == (len(in_out_channels) - 1)
            self.up_modules.append(nn.ModuleList([
                # U-Net Skip Connection 导致输入通道数翻倍
                ConditionalResidualBlock1D(dim_out * 2, dim_in, cond_dim=cond_dim),
                ConditionalResidualBlock1D(dim_in, dim_in, cond_dim=cond_dim),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))
            
        # 最后的输出投影层 (从 down_dims[0] 无缝恢复到 action_dim)
        self.final_conv = nn.Sequential(
            Conv1dBlock(down_dims[0], down_dims[0]),
            nn.Conv1d(down_dims[0], action_dim, 1) # 1x1卷积，不带 GroupNorm
        )

    def forward(
        self, 
        sample: torch.Tensor, 
        timestep: Union[torch.Tensor, float, int], 
        global_cond: torch.Tensor
    ) -> torch.Tensor:
        # 1. 处理输入维度 (B, Chunk_Size, Action_Dim) -> (B, Action_Dim, Chunk_Size)
        if sample.shape[-1] == self.action_dim:
            x = sample.permute(0, 2, 1)
            permuted = True
        else:
            x = sample
            permuted = False

        # 2. 初始通道投影 (规避不规则维度)
        x = self.init_conv(x)

        # 3. 处理时间步编码
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.float32, device=x.device)
        elif torch.is_tensor(timestep) and len(timestep.shape) == 0:
            timestep = timestep[None].to(x.device)
            
        timestep = timestep.expand(x.shape[0]).float()
        step_embed = self.step_mlp(self.step_emb(timestep))
        
        # 4. 拼接全局特征和时间步特征
        cond = torch.cat([step_embed, global_cond], dim=-1)

        # 5. Encoder 阶段
        h = []
        for resnet1, resnet2, downsample in self.down_modules:
            x = resnet1(x, cond)
            x = resnet2(x, cond)
            h.append(x)  # 保存 Skip Connection
            x = downsample(x)

        # 6. Mid 阶段
        for resnet in self.mid_modules:
            x = resnet(x, cond)

        # 7. Decoder 阶段
        for resnet1, resnet2, upsample in self.up_modules:
            # 兼容奇数尺寸的 Chunk Size
            h_pop = h.pop()
            if x.shape[-1] != h_pop.shape[-1]:
                x = F.interpolate(x, size=h_pop.shape[-1], mode='linear', align_corners=False)
            x = torch.cat([x, h_pop], dim=1)
            x = resnet1(x, cond)
            x = resnet2(x, cond)
            x = upsample(x)

        # 8. 输出层
        x = self.final_conv(x)

        # 9. 维度复原
        if permuted:
            x = x.permute(0, 2, 1)

        return x

# ==========================================
# 调试与测试模块 
# ==========================================
if __name__ == '__main__':
    batch_size = 32
    chunk_size = 16
    action_dim = 14     # 即使是奇奇怪怪的数字也不会报错了
    global_cond_dim = 256
    
    model = ConditionalUnet1D(
        action_dim=action_dim,
        global_cond_dim=global_cond_dim,
        down_dims=(128, 256, 512)
    )
    
    noisy_action_chunk = torch.randn(batch_size, chunk_size, action_dim) 
    timesteps = torch.randint(0, 100, (batch_size,))
    pointcloud_feats = torch.randn(batch_size, global_cond_dim)
    
    out = model(noisy_action_chunk, timesteps, pointcloud_feats)
    
    print(f"输入动作形状: {noisy_action_chunk.shape}")
    print(f"输出预测形状: {out.shape}")
    assert noisy_action_chunk.shape == out.shape
    print("测试通过")