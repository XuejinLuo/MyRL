# algos/distill.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from typing import Tuple

class OneStepDistiller:
    """
    单步蒸馏核心逻辑 (One-step Distillation for Flow-matching / Diffusion)
    将多步 ODE 的 Teacher Policy 蒸馏为一个只需 1 步推理的 Student Policy。
    适用于 PointCloud + Action Chunk 架构。
    """
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        lr: float = 1e-4,
        teacher_steps: int = 10,
        use_ema: bool = True  # 强推开启 EMA
    ):
        self.device = device
        self.teacher_steps = teacher_steps
        
        # 1. 初始化 Teacher 模型 (冻结参数)
        self.teacher = teacher_model.to(self.device).eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        # 2. 初始化 Student 模型 (可训练)
        self.student = student_model.to(self.device).train()
        
        # 3. 优化器
        self.optimizer = optim.AdamW(self.student.parameters(), lr=lr, weight_decay=1e-4)

        # 4. EMA Model (Optional but recommended)
        self.use_ema = use_ema
        if self.use_ema:
            self.ema_student = copy.deepcopy(self.student).eval()
            for param in self.ema_student.parameters():
                param.requires_grad = False

    def update_ema(self, decay=0.995):
        with torch.no_grad():
            for ema_param, param in zip(self.ema_student.parameters(), self.student.parameters()):
                ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)
            for ema_buffer, buffer in zip(self.ema_student.buffers(), self.student.buffers()):
                ema_buffer.data.copy_(buffer.data)

    @torch.no_grad()
    def teacher_generate(self, obs: torch.Tensor, state: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        使用 Teacher 模型通过 Euler ODE 求解器多步生成 Action Chunk
        Flow Matching 约定: t=0 为纯噪声, t=1 为目标真实动作
        """
        dt = 1.0 / self.teacher_steps
        x_t = noise.clone() # 初始状态: 纯噪声 [B, Chunk_Size, Action_Dim]
        
        for i in range(self.teacher_steps):
            # 构造当前的时间步 t
            t = torch.ones((obs.shape[0], 1), device=self.device) * (i * dt)
            
            # 预测向量场 Vector Field (速度 v)
            v_pred = self.teacher(obs, state, x_t, t) 
            
            # Euler 步进: x_{t+dt} = x_t + v * dt
            x_t = x_t + v_pred * dt
            
        return x_t # 返回生成的 Target Action Chunk

    def update_step(self, obs: torch.Tensor, state: torch.Tensor, 
                    precomputed_noise=None, precomputed_target=None) -> float:
        """
        单步蒸馏训练逻辑，支持在线蒸馏 (慢) 和离线缓存数据集蒸馏 (快)
        Args:
            obs: 3D Point Cloud 观测, shape: [B, N_points, 3] (或经过 Encoder 后的特征)
        """
        B, chunk_size, action_dim = obs.shape[0], self.student.chunk_size, self.student.action_dim

        # --- 数据准备 ---
        if precomputed_noise is not None and precomputed_target is not None:
            # 高效模式：直接使用事先跑好 teacher_generate 的离线数据
            z_noise = precomputed_noise
            target_action = precomputed_target
        else:
            # 在线模式：每次前向都要跑 10 步 (极慢，仅供验证和小规模微调)
            z_noise = torch.randn((B, chunk_size, action_dim), device=self.device) # 采样相同的初始随机噪声 [B, Chunk_Size, Action_Dim]
            target_action = self.teacher_generate(obs, state, z_noise) # Teacher 多步生成伪标签 (Pseudo-target Action)
        
        # --- Student 单步预测 ---
        t_zero = torch.zeros((B, 1), device=self.device) # 对于单步蒸馏，Student 的输入是同样的噪声，但 t 固定为 0
        student_v = self.student(obs, state, z_noise, t_zero)
            
        # --- Loss 计算 (Velocity Matching) ---
        target_v = target_action - z_noise  # dt=1 的理想速度
        loss = F.mse_loss(student_v, target_v)
        
        # --- 优化 ---
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
        self.optimizer.step()
        
        if self.use_ema:
            self.update_ema()
            
        return loss.item()


# ==============================================================================
# 以下为测试与验证模块 (__main__)，仅在直接运行该脚本时执行
# 用于验证你的模型管道 (Model Pipeline) 是否能够正常完成维度对齐和蒸馏闭环
# ==============================================================================
if __name__ == "__main__":
    import numpy as np
    
    print("🚀 正在启动 One-step Distillation 测试...\n")
    
    # 模拟超参数 (Hyperparams)
    BATCH_SIZE = 16
    N_POINTS = 1024         # 3D 点云点数
    OBS_DIM = 3             # 3D 坐标
    STATE_DIM = 7           # 本体状态维度 (比如 6DoF 关节角 + 1 Gripper 状态)
    CHUNK_SIZE = 16         # Action Chunk 长度 (Temporal Ensembling)
    ACTION_DIM = 7          # 机器臂动作维度 (6DoF + 1 Gripper)
    
    class DummyPolicyNetwork(nn.Module):
        """
        模拟你的 3D Flow/Diffusion 策略网络 (PointNeXt + Transformer)
        仅作测试用，确保张量维度流转正确
        """
        def __init__(self, chunk_size, action_dim, state_dim):
            super().__init__()
            self.chunk_size = chunk_size
            self.action_dim = action_dim
            self.state_dim = state_dim
            
            # 假装这是一个 PointNet/PointNeXt 特征提取器
            self.point_encoder = nn.Sequential(
                nn.Linear(3, 64),
                nn.ReLU(),
                nn.Linear(64, 128)
            )
            # 假装这是一个 Transformer/UNet 处理时序和噪声
            self.action_decoder = nn.Sequential(
                nn.Linear(128 + self.state_dim + action_dim * chunk_size + 1, 256), 
                nn.ReLU(),
                nn.Linear(256, action_dim * chunk_size)
            )

        def forward(self, obs: torch.Tensor, state: torch.Tensor, noisy_action: torch.Tensor, time: torch.Tensor):
            # obs: [B, N, 3] -> 简易 PointNet 的 MaxPool
            obs_feats = self.point_encoder(obs).max(dim=1)[0] # [B, 128]
            
            # 展平 noisy_action [B, Chunk, Dim] -> [B, Chunk*Dim]
            act_feats = noisy_action.view(noisy_action.shape[0], -1) 
            
            # 拼接所有条件 (Conditioning)
            fused = torch.cat([obs_feats, state, act_feats, time], dim=-1)
            
            # 预测向量场 (Vector Field)
            v_pred = self.action_decoder(fused)
            
            return v_pred.view(noisy_action.shape[0], self.chunk_size, self.action_dim)

    # 1. 实例化假模型
    print("1️⃣ 初始化 Teacher 和 Student 网络...")
    teacher_net = DummyPolicyNetwork(CHUNK_SIZE, ACTION_DIM, STATE_DIM)
    # Student 通常以 Teacher 为初始化基座 (可以极大加速收敛)
    student_net = copy.deepcopy(teacher_net) 
    
    # 2. 实例化蒸馏器
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"2️⃣ 初始化 Distiller (设备: {device})...")
    distiller = OneStepDistiller(
        teacher_model=teacher_net,
        student_model=student_net,
        device=device,
        teacher_steps=10, # Teacher 用 10 步 ODE 求解
        use_ema=True
    )
    
    # 3. 模拟离线或在线环境给过来的 3D 点云 Batch
    print("3️⃣ 生成模拟的 3D 点云观测数据...\n")
    dummy_obs_batch = torch.randn((BATCH_SIZE, N_POINTS, OBS_DIM), device=device)
    dummy_state_batch = torch.randn((BATCH_SIZE, STATE_DIM), device=device)
    
    # 4. 执行蒸馏训练循环
    print("🔥 开始蒸馏训练 (模拟 10 个 Step):")
    for step in range(1, 11):
        loss = distiller.update_step(dummy_obs_batch, dummy_state_batch)
        print(f"   Step {step:02d} | Distillation Loss: {loss:.6f}")
        
    print("\n✅ 测试通过！你现在可以安全地将真实的 PointNeXt 和 Transformer 替换 DummyNetwork 了。")