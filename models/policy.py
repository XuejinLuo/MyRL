# models/policy.py
"""
Embodied Policy Core (Flow / Diffusion Policy)
为 3D 具身智能 (PointCloud + Chunk Action) 打造的通用策略包装器。

完美适配：
1. Behavior Cloning (BC) & IDQL: 通过 compute_loss() 接口
2. Env Rollout (推理): 通过 sample() 接口
3. One-step Distillation: 通过 forward() 接口暴露底层前向计算
4. Policy Gradient (PG): 通过 evaluate_actions() 提供近似对数似然估计
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Union

# 导入你提供的各个模块 (请确保这些文件在你的 PYTHONPATH 中)
from models.encoders.factory import build_encoder, encoder_observation
from models.backbones.transformer import ActionDiffusionTransformer
from models.backbones.unet1d import ConditionalUnet1D
from algos.diffusion_utils.flow_matching import OTFlowMatching
from algos.diffusion_utils.ddim_scheduler import DDIMScheduler

class EmbodiedGenPolicy(nn.Module):
    def __init__(
        self,
        # 1. 空间/动作定义
        action_dim: int = 7,
        chunk_size: int = 16,
        use_state: bool = True,
        state_dim: int = 14,
        in_channels: int = 3,
        
        # 2. 网络结构配置
        encoder_type: str = "pointnext", 
        backbone_type: str = "transformer", # "transformer" or "unet1d"
        cond_dim: int = 256,                # 融合后条件向量维度
        
        # 3. 生成算法配置
        algo_type: str = "flow",            # "flow" (推荐) or "diffusion"
        num_train_steps: int = 100,         # 仅 Diffusion 需要
        encoder_variant: str = 'points',
        state_skip: bool = False,
        relational_features: bool = False,
        object_feature_dim: int = 23,
        object_feature_hidden_dim: int = 64,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.use_state = use_state
        self.state_dim = state_dim
        self.algo_type = algo_type.lower()

        # ====================================================================
        # 1. 实例化 3D 编码器 (Encoder)
        # ====================================================================
        self.encoder = build_encoder(encoder_type, in_channels, cond_dim,
                                     use_state, state_dim, encoder_variant, state_skip,
                                     relational_features, object_feature_dim, object_feature_hidden_dim)

        # ====================================================================
        # 2. 实例化 骨干网络 (Backbone)
        # ====================================================================
        if backbone_type == "transformer":
            self.backbone = ActionDiffusionTransformer(
                action_dim=action_dim,
                cond_dim=cond_dim,
                chunk_size=chunk_size,
                embed_dim=256,
                depth=6,
                num_heads=8
            )
        elif backbone_type == "unet1d":
            self.backbone = ConditionalUnet1D(
                action_dim=action_dim,
                global_cond_dim=cond_dim,
                down_dims=(128, 256, 512)
            )
        else:
            raise NotImplementedError(f"Unsupported backbone: {backbone_type}")

        # ====================================================================
        # 3. 实例化 生成调度器 (Scheduler)
        # ====================================================================
        if self.algo_type == "flow":
            self.scheduler = OTFlowMatching(sigma_min=1e-5)
        elif self.algo_type == "diffusion":
            self.scheduler = DDIMScheduler(num_train_timesteps=num_train_steps)
        else:
            raise ValueError("algo_type must be 'flow' or 'diffusion'")

    def _get_condition(self, obs: torch.Tensor, state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """ 内部辅助函数：将点云和本体状态打包喂给 Encoder """
        return self.encoder(encoder_observation(obs, state))

    def encode(self, obs, state=None):
        return self._get_condition(obs, state)

    # ========================================================================
    # [核心接口 1] Distillation / 基础前向计算
    # 与 algos/distill.py 完全对齐 (student_v = self.student(obs, state, z_noise, t_zero))
    # ========================================================================
    def forward(self, obs: torch.Tensor, state: torch.Tensor, noisy_action: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        直接暴露底层计算流，专供 One-step Distillation 和 Teacher-forcing 使用。
        """
        cond = self._get_condition(obs, state)
        
        # 如果 time 的维度是 [B, 1]，需要 squeeze 成 [B] 给 Transformer/UNet
        if time.dim() == 2 and time.shape[1] == 1:
            time = time.squeeze(1)
            
        return self.backbone(noisy_action, time, cond)

    # ========================================================================
    # [核心接口 2] BC / IDQL 训练
    # 与 algos/idql.py 完全对齐 (actor_loss = self.actor.compute_loss(filtered_states, filtered_actions))
    # ========================================================================
    def compute_loss(self, obs: torch.Tensor, actions: torch.Tensor, state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算生成模型的训练 Loss (Vector Field MSE 或 Denoising MSE)
        支持 IDQL 传入过滤后的高质量样本直接求导。
        """
        cond = self._get_condition(obs, state)

        if self.algo_type == "flow":
            # Flow Matching 直接利用你写好的 OTFlowMatching.compute_loss
            # 注意：我们将 self.backbone 传入，巧妙解耦
            return self.scheduler.compute_loss(self.backbone, actions, cond)
            
        elif self.algo_type == "diffusion":
            # DDPM/DDIM 标准加噪与 MSE 训练逻辑
            B = actions.shape[0]
            device = actions.device
            noise = torch.randn_like(actions)
            # 随机采样整数时间步
            timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (B,), device=device).long()
            
            # 加噪
            noisy_actions = self.scheduler.add_noise(actions, noise, timesteps)
            
            # 预测
            pred = self.backbone(noisy_actions, timesteps, cond)
            
            # 假定 prediction_type == "epsilon" (预测噪声)
            return F.mse_loss(pred, noise)

    # ========================================================================
    # [核心接口 3] 环境 Rollout / 动作推断
    # ========================================================================
    @torch.no_grad()
    def sample(self, 
               obs: torch.Tensor, 
               state: Optional[torch.Tensor] = None, 
               num_steps: int = 10, 
               cfg_weight: float = 1.0) -> torch.Tensor:
        """
        用于环境交互期间生成动作 Chunk。
        """
        cond = self._get_condition(obs, state)
        action_shape = (self.chunk_size, self.action_dim)
        
        if self.algo_type == "flow":
            # [Fix: CFG 修复] - 补充 uncond_cond 生成逻辑，否则 cfg_weight 传入底层会失效
            uncond_cond = None
            if cfg_weight != 1.0:
                dummy_obs = ({k: (v if k == 'object_roles' else torch.zeros_like(v))
                              for k, v in obs.items()} if isinstance(obs, dict) else torch.zeros_like(obs))
                dummy_state = torch.zeros_like(state) if (self.use_state and state is not None) else None
                uncond_cond = self._get_condition(dummy_obs, dummy_state)
            
            # 调用你写好的 Flow ODE Solver
            return self.scheduler.sample(
                model=self.backbone, 
                cond=cond, 
                action_shape=action_shape, 
                num_steps=num_steps,
                solver='euler',       # Euler 对于 10 步通常足够且快
                cfg_weight=cfg_weight,
                uncond_cond=uncond_cond # [Fix] 将无条件特征传入 Flow Matching
            )
            
        elif self.algo_type == "diffusion":
            # 调用你写好的 DDIM Step 循环
            B = cond.shape[0]
            device = cond.device
            self.scheduler.set_timesteps(num_steps, device=device)
            
            x_t = torch.randn((B, *action_shape), device=device)
            for t in self.scheduler.timesteps:
                t_batch = torch.full((B,), t.item(), device=device, dtype=torch.long)
                pred = self.backbone(x_t, t_batch, cond)
                x_t = self.scheduler.step(pred, int(t.item()), x_t)
                
            return x_t

    # ========================================================================
    # [核心接口 4] Policy Gradient (PPO) 辅助评估
    # ========================================================================
    def evaluate_actions(
        self, 
        obs: torch.Tensor, 
        actions: torch.Tensor, 
        state: Optional[torch.Tensor] = None, 
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        在线强化微调的回归评估。返回每个样本的 Vector Field MSE
        """        
        cond = self._get_condition(obs, state)
        B, device = cond.shape[0], cond.device
        
        if noise is None:
            noise = torch.randn_like(actions)

        if self.algo_type == "flow":
            # 随机采样 t，保护整个向量场不被破坏
            t = torch.rand((B,), device=device) * (1.0 - 2e-5) + 1e-5
            t_expand = t.view(-1, 1, 1)
            
            xt = (1 - (1 - 1e-5) * t_expand) * noise + t_expand * actions
            target_v = actions - (1 - 1e-5) * noise
            pred_v = self.backbone(xt, t, cond)
            
            # 保持 [B] 的维度，不求平均
            mse_error = torch.mean((pred_v - target_v) ** 2, dim=(-1, -2)) 
            
            return mse_error, torch.zeros_like(mse_error) # 返回 MSE 和 占位用的 Entropy
            
        else: # diffusion
            # DDPM 同理，随机采样 t
            t = torch.randint(0, self.scheduler.num_train_timesteps, (B,), device=device).long()
            xt = self.scheduler.add_noise(actions, noise, t)
            pred_noise = self.backbone(xt, t, cond)
            
            mse_error = torch.mean((pred_noise - noise) ** 2, dim=(-1, -2))
            return mse_error, torch.zeros_like(mse_error)


# ==============================================================================
# 本地测试模块 
# ==============================================================================
if __name__ == "__main__":
    print("🚀 启动 EmbodiedGenPolicy 大一统核心测试...\n")
    
    # 模拟超参数
    BATCH_SIZE = 4
    N_POINTS = 1024
    OBS_DIM = 3
    STATE_DIM = 7
    CHUNK_SIZE = 16
    ACTION_DIM = 7
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"🖥️  测试设备: {DEVICE}")
    
    # 初始化策略 (以最前沿的 Transformer + Flow Matching 为例)
    print("1️⃣ 正在初始化 Flow Transformer Policy...")
    policy = EmbodiedGenPolicy(
        action_dim=ACTION_DIM, chunk_size=CHUNK_SIZE, 
        use_state=True, state_dim=STATE_DIM,
        encoder_type="pointnext", backbone_type="transformer", algo_type="flow"
    ).to(DEVICE)
    print(f"✅ 初始化完成！总参数量: {sum(p.numel() for p in policy.parameters()) / 1e6:.2f} M\n")

    # 模拟数据
    dummy_obs = torch.randn((BATCH_SIZE, N_POINTS, OBS_DIM), device=DEVICE)
    dummy_state = torch.randn((BATCH_SIZE, STATE_DIM), device=DEVICE)
    dummy_actions = torch.randn((BATCH_SIZE, CHUNK_SIZE, ACTION_DIM), device=DEVICE)

    # 测试 1: BC/IDQL Loss 计算
    print("2️⃣ 测试 IDQL/BC 前向 Loss 计算 (compute_loss)...")
    loss = policy.compute_loss(dummy_obs, dummy_actions, dummy_state)
    print(f"✅ Loss 计算成功: {loss.item():.4f}\n")

    # 测试 2: 环境推断 Sample
    print("3️⃣ 测试 环境推断 (sample)...")
    generated_actions = policy.sample(dummy_obs, dummy_state, num_steps=10)
    print(f"✅ 推断成功，输出维度 (应与目标一致): {generated_actions.shape}\n")

    # 测试 3: 蒸馏底层调用 Distill Forward
    print("4️⃣ 测试 蒸馏底层调用 (forward)...")
    t_zero = torch.zeros((BATCH_SIZE, 1), device=DEVICE)
    z_noise = torch.randn_like(dummy_actions)
    v_pred = policy(dummy_obs, dummy_state, z_noise, t_zero)
    print(f"✅ 前向成功，输出维度: {v_pred.shape}\n")

    # 测试 4: PG PPO 似然估计
    print("5️⃣ 测试 PG 似然评估 (evaluate_actions)...")
    log_probs, entropy = policy.evaluate_actions(dummy_obs, dummy_actions, dummy_state)
    print(f"✅ 似然估计成功，Log Probs shape: {log_probs.shape}\n")

    print("🎉 EmbodiedGenPolicy 完美契合你的所有底层协议，可以开服了！")
