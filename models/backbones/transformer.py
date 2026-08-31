# models/backbones/transformer.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPosEmb(nn.Module):
    """
    经典的正弦波位置编码，用于将连续的 Timestep 转化为高维特征
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # x: [Batch]
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class AdaLN(nn.Module):
    """
    Adaptive Layer Normalization (AdaLN)
    用于将环境特征 (Cond) 和 时间步 (Timestep) 调制并注入到 Transformer 块中
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.silu = nn.SiLU()
        # 输出 6 倍维度，分别对应 Self-Attention 和 FFN 的 (shift, scale, gate)
        self.linear = nn.Linear(embed_dim, embed_dim * 6)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, c):
        # c: [Batch, embed_dim]
        c = self.silu(c)
        c = self.linear(c)
        # 将输出切分为 6 个部分，形状均为 [Batch, embed_dim]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = c.chunk(6, dim=1)
        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

class DiTBlock(nn.Module):
    """
    Diffusion Transformer Block
    带有 AdaLN 调制的标准 Transformer Encoder Layer
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        
        # FFN层 (使用GELU)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
        self.adaLN_modulation = AdaLN(embed_dim)

    def forward(self, x, c):
        # x: [Batch, Chunk_size, embed_dim]
        # c: [Batch, embed_dim] (Timestep + 3D/State Feature 融合后的特征)
        
        # 1. 获取调制参数，并增加序列维度以匹配 x: [Batch, 1, embed_dim]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = [
            param.unsqueeze(1) for param in self.adaLN_modulation(c)
        ]

        # 2. Attention 模块 (带调制)
        norm_x1 = self.norm1(x) * (1 + scale_msa) + shift_msa
        attn_out, _ = self.attn(norm_x1, norm_x1, norm_x1)
        x = x + gate_msa * attn_out

        # 3. FFN 模块 (带调制)
        norm_x2 = self.norm2(x) * (1 + scale_mlp) + shift_mlp
        ffn_out = self.ffn(norm_x2)
        x = x + gate_mlp * ffn_out

        return x

class ActionDiffusionTransformer(nn.Module):
    """
    主网络 Backbone：专为 Flow / 3D Diffusion 优化的 Chunk Action Transformer
    """
    def __init__(
        self, 
        action_dim, 
        cond_dim, 
        chunk_size, 
        embed_dim=256, 
        depth=6, 
        num_heads=8
    ):
        super().__init__()
        self.action_dim = action_dim
        self.cond_dim = cond_dim
        self.chunk_size = chunk_size
        self.embed_dim = embed_dim
        
        # 1. Timestep Embedding
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(embed_dim),
            nn.Linear(embed_dim, embed_dim * 4),
            nn.Mish(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
        # 2. Condition (3D features + state) Embedding
        self.cond_emb = nn.Sequential(
            nn.Linear(cond_dim, embed_dim * 4),
            nn.Mish(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
        # 3. Action / Chunk Embedding
        self.action_proj = nn.Linear(action_dim, embed_dim)
        # Learnable Positional Embedding for Action Chunk Sequence
        self.pos_embed = nn.Parameter(torch.randn(1, chunk_size, embed_dim) * 0.02)
        
        # 4. Transformer Blocks
        self.blocks = nn.ModuleList([
            DiTBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        
        # 5. Output Projector (Predict Noise or Vector Field)
        self.final_norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Linear(embed_dim, embed_dim * 2)
        self.action_out = nn.Linear(embed_dim, action_dim)

        self.cond_time_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Mish(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # 针对 Diffusion 模型的零初始化技巧 (Zero-initialization for outputs)
        nn.init.zeros_(self.action_out.weight)
        nn.init.zeros_(self.action_out.bias)
        nn.init.zeros_(self.final_adaLN.weight)
        nn.init.zeros_(self.final_adaLN.bias)

    def forward(self, x, timestep, cond):
        """
        x: [B, Chunk_Size, Action_Dim] (Noisy actions at current step)
        timestep: [B] (Current diffusion/flow step)
        cond: [B, Cond_Dim] (Output from PointNeXt encoder + Robot proprioception)
        
        Returns:
        [B, Chunk_Size, Action_Dim] (Predicted noise or vector field dt)
        """
        B, seq_len, _ = x.shape
        assert seq_len == self.chunk_size, f"Input sequence length must be {self.chunk_size}"
        if timestep.max() <= 1.0:
            timestep = timestep * 1000.0 

        # 1. Embed Timestep & Condition
        t_feat = self.time_emb(timestep)      # [B, embed_dim]
        c_feat = self.cond_emb(cond)          # [B, embed_dim]
        mod_feat = self.cond_time_mlp(torch.cat([t_feat, c_feat], dim=-1))
        
        # 2. Embed Action and add Positional Embedding
        x = self.action_proj(x)               # [B, Chunk_Size, embed_dim]
        x = x + self.pos_embed                # 添加序列位置信息
        
        # 3. Pass through Transformer Blocks
        for block in self.blocks:
            x = block(x, mod_feat)
            
        # 4. Final output layer with AdaLN modulation
        shift_final, scale_final = self.final_adaLN(F.silu(mod_feat)).chunk(2, dim=1)
        shift_final = shift_final.unsqueeze(1)
        scale_final = scale_final.unsqueeze(1)
        
        x = self.final_norm(x) * (1 + scale_final) + shift_final
        out = self.action_out(x)              # [B, Chunk_Size, Action_Dim]
        
        return out


if __name__ == "__main__":
    # ==========================================
    # Main 测试模块：验证 Backbone 是否可跑通
    # ==========================================
    print("🚀 开始测试 ActionDiffusionTransformer...")
    
    # 设定超参数 (可以根据你的任务修改)
    BATCH_SIZE = 4
    CHUNK_SIZE = 16        # 每次预测连续 16 步动作
    ACTION_DIM = 8         # 机器人动作维度 (如 7-DOF机械臂 + 1个夹爪)
    COND_DIM = 512         # 从 PointNeXt 提取的 3D 特征 + 本体特征 (如关节角) 拼起来的维度
    EMBED_DIM = 256
    
    # 初始化模型
    model = ActionDiffusionTransformer(
        action_dim=ACTION_DIM,
        cond_dim=COND_DIM,
        chunk_size=CHUNK_SIZE,
        embed_dim=EMBED_DIM,
        depth=4,           # 4层 Transformer (轻量级测试)
        num_heads=8
    )
    
    # 模拟输入 Tensor
    # 1. 加噪后的 Action Chunk (Noisy Action)
    noisy_actions = torch.randn(BATCH_SIZE, CHUNK_SIZE, ACTION_DIM)
    
    # 2. 时间步 Timestep (Diffusion 常用 0-1000，Flow 常用 0-1 的浮点数)
    # 这里模拟 Flow Matching 的浮点时间 (或 DDPM/DDIM 的整数步)
    timesteps = torch.rand(BATCH_SIZE) * 1000 
    
    # 3. 视觉与本体状态特征 (Condition)
    conditions = torch.randn(BATCH_SIZE, COND_DIM)
    
    print(f"✅ 模型初始化完成. 参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")
    print(f"👉 输入 Action Shape: {noisy_actions.shape}")
    print(f"👉 输入 Timestep Shape: {timesteps.shape}")
    print(f"👉 输入 Condition Shape: {conditions.shape}")
    
    # 前向传播 (Forward)
    try:
        predicted_noise_or_vf = model(noisy_actions, timesteps, conditions)
        print(f"🎉 前向传播成功!")
        print(f"✅ 输出 Shape (应该与输入 Action 保持一致): {predicted_noise_or_vf.shape}")
        
        # 简单验证输出形状是否一致
        assert predicted_noise_or_vf.shape == noisy_actions.shape, "❌ 输出形状与输入动作不匹配！"
        
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")