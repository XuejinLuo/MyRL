# algos/diffusion_utils/flow_matching.py

import torch
import torch.nn as nn
from typing import Callable, Tuple, Optional
import contextlib

class OTFlowMatching:
    """
    Optimal Transport Conditional Flow Matching (OT-CFM)
    适用于 Chunk Action 和 Continuous Normalizing Flows 的流匹配核心逻辑。
    """
    def __init__(self, sigma_min: float = 1e-5):
        """
        初始化 Flow Matching 调度器
        :param sigma_min: 防止数值不稳定的微小噪声扰动，通常在 OT Flow 中设为极小值
        """
        self.sigma_min = sigma_min

    def compute_loss(self, 
                     model: nn.Module, 
                     x1: torch.Tensor, 
                     cond: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
        """
        计算 Flow Matching 的目标函数 (Vector Field MSE Loss)
        
        :param model: 预测向量场的神经网络模型 v_theta(x_t, t, cond)
        :param x1: 目标数据，即 Expert Action Chunk，形状 [B, Chunk_Size, Action_Dim]
        :param cond: 条件特征 (如 PointCloud 的全局/局部特征)，形状 [B, Cond_Dim]
        :return: MSE 损失值
        """
        B = x1.shape[0]
        device = x1.device

        # 1. 采样初始噪声 x0 ~ N(0, I)
        x0 = torch.randn_like(x1)

        # 2. 在 [0, 1] 之间均匀采样时间步 t
        # 为了避免 t 恰好为 0 或 1 带来的奇异性，通常在 (0, 1) 的微小截断内采样，或者直接均匀采样
        # 为了避免极值，通常限制在 [1e-5, 1 - 1e-5] 增加数值稳定性
        t = torch.rand((B,), device=device) * (1.0 - 2e-5) + 1e-5

        # 将 t 扩展到与 x1 相同的维度，以便于广播计算: [B, 1, 1]
        t_expand = t.view(-1, *([1] * (x1.dim() - 1)))

        # 3. 构建严格一致的 OT 直线路径
        # 数学公式: x_t = (1 - (1 - sigma_min)*t) * x0 + t * x1
        # 求导: d(x_t)/dt = x1 - (1 - sigma_min) * x0
        xt = (1 - (1 - self.sigma_min) * t_expand) * x0 + t_expand * x1

        # 4. 计算真实的目标向量场 (Target Vector Field)
        ut = x1 - (1 - self.sigma_min) * x0

        # 5. 模型预测当前时间的向量场
        vt = model(xt, t, cond)

        # 6. 计算 MSE Loss (匹配向量场)
        squared_error = (vt - ut) ** 2
        if reduction == 'none':
            return squared_error.flatten(1).mean(1)
        if reduction != 'mean':
            raise ValueError('reduction must be mean or none')
        return torch.mean(squared_error)

    def sample(self, 
               model: nn.Module, 
               cond: torch.Tensor, 
               action_shape: Tuple[int, ...], 
               num_steps: int = 20, 
               solver: str = 'euler',
               cfg_weight: float = 1.0,         # 新增: CFG 权重 (1.0表示不使用CFG)
               uncond_cond: Optional[torch.Tensor] = None, # 新增: 无条件向量 (用于CFG)
               requires_grad: bool = False      # 新增: 是否允许梯度回传 (用于PG和蒸馏)
               ) -> torch.Tensor:
        """
        使用 ODE 求解器从纯噪声生成 Action Chunk
        
        :param model: 预测向量场的模型
        :param cond: 状态/点云条件特征 [B, Cond_Dim]
        :param action_shape: 单个样本的 Action 形状，如 (Chunk_Size, Action_Dim)
        :param num_steps: 推理步数 (Flow 一般只需要 10-20 步)
        :param solver: ODE 求解器，支持 'euler' (欧拉法) 或 'heun' (改进欧拉/Heun法)
        :return: 生成的 Action Chunk [B, Chunk_Size, Action_Dim]
        """
        B = cond.shape[0]
        device = cond.device

        # 根据 requires_grad 决定是否开启梯度追踪
        context = contextlib.nullcontext() if requires_grad else torch.no_grad()

        with context:
            # t=0 时刻，从纯标准高斯噪声出发
            x = torch.randn((B, *action_shape), device=device)
            
            # 步长
            dt = 1.0 / num_steps
            
            # 时间步列表: 0, dt, 2dt, ..., 1.0-dt
            t_steps = torch.linspace(0, 1.0 - dt, num_steps, device=device)

            for t_val in t_steps:
                # 构造当前批次的时间张量 [B,]
                t_batch = torch.full((B,), t_val, device=device)

                # 辅助函数: 计算当前的向量场 (带有 CFG 逻辑)
                def get_vt(x_curr, t_curr):
                    if cfg_weight != 1.0 and uncond_cond is not None:
                        # 拼接 batch 以便一次 forward 算出 cond 和 uncond，节省时间
                        x_in = torch.cat([x_curr, x_curr], dim=0)
                        t_in = torch.cat([t_curr, t_curr], dim=0)
                        c_in = torch.cat([cond, uncond_cond], dim=0)
                        
                        vt_out = model(x_in, t_in, c_in)
                        vt_cond, vt_uncond = torch.chunk(vt_out, chunks=2, dim=0)
                        
                        # CFG 外推公式
                        return vt_uncond + cfg_weight * (vt_cond - vt_uncond)
                    else:
                        return model(x_curr, t_curr, cond)

                if solver == 'euler':
                    vt = get_vt(x, t_batch)
                    x = x + vt * dt
                    
                elif solver == 'heun':
                    vt = get_vt(x, t_batch)
                    x_next_euler = x + vt * dt
                    
                    if t_val == t_steps[-1]:
                        x = x_next_euler
                    else:
                        t_next = t_batch + dt
                        vt_next = get_vt(x_next_euler, t_next)
                        x = x + (vt + vt_next) / 2.0 * dt
                else:
                    raise ValueError(f"不支持的 ODE 求解器: {solver}")

            return x


# =========================================================================
# 测试逻辑 (Main) - 验证前向 Loss 与 ODE 采样是否闭环
# =========================================================================
if __name__ == "__main__":
    print("🚀 开始测试 Flow Matching 模块...")

    # 1. 模拟环境与模型的超参数
    BATCH_SIZE = 16
    CHUNK_SIZE = 8       # Action Chunk 长度
    ACTION_DIM = 7       # 机器人自由度 (如 6 DoF + 1 Gripper)
    COND_DIM = 128       # 假设从 PointNeXt 提取出来的点云特征维度

    # 2. 构建一个非常简单的 Dummy 网络来替代真实的 UNet1D/Transformer
    class DummyVectorFieldNet(nn.Module):
        def __init__(self):
            super().__init__()
            # 输入: Action_Flat(8*7=56) + Time(1) + Cond(128) = 185
            self.net = nn.Sequential(
                nn.Linear(CHUNK_SIZE * ACTION_DIM + 1 + COND_DIM, 256),
                nn.Mish(),
                nn.Linear(256, 256),
                nn.Mish(),
                nn.Linear(256, CHUNK_SIZE * ACTION_DIM)
            )
        
        def forward(self, x, t, cond):
            B = x.shape[0]
            # Flatten 动作序列
            x_flat = x.view(B, -1)
            # t 扩展维度
            t_exp = t.view(B, 1)
            # 拼接输入
            inputs = torch.cat([x_flat, t_exp, cond], dim=-1)
            # 预测 Vector Field
            vt_flat = self.net(inputs)
            # 还原形状返回
            return vt_flat.view(B, CHUNK_SIZE, ACTION_DIM)

    # 实例化组件
    model = DummyVectorFieldNet()
    fm_scheduler = OTFlowMatching()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 3. 模拟离线数据集中的输入 (Expert Data)
    expert_actions = torch.randn(BATCH_SIZE, CHUNK_SIZE, ACTION_DIM) # 真实的 Chunk Action
    state_conds = torch.randn(BATCH_SIZE, COND_DIM)                  # 真实的 PointCloud 条件特征

    print(f"📊 模型输入维度: Actions {expert_actions.shape}, Conds {state_conds.shape}")

    # 4. 测试计算 Loss 并反向传播 (模拟训练过程)
    model.train()
    loss = fm_scheduler.compute_loss(model, expert_actions, state_conds)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"✅ 训练单步成功! 当前 Flow Matching Loss: {loss.item():.4f}")

    # 5. 测试生成过程 (ODE 采样 / 推理过程)
    model.eval()
    action_shape = (CHUNK_SIZE, ACTION_DIM)
    
    # 测试 Euler 采样 (假设单步推断需要 10 步)
    print("⏳ 测试 Euler ODE 采样 (10步)...")
    sampled_actions_euler = fm_scheduler.sample(
        model=model, 
        cond=state_conds, 
        action_shape=action_shape, 
        num_steps=10, 
        solver='euler'
    )
    print(f"✅ Euler 采样成功! 生成动作维度: {sampled_actions_euler.shape}")

    # 测试 Heun 采样 (更精确)
    print("⏳ 测试 Heun ODE 采样 (10步)...")
    sampled_actions_heun = fm_scheduler.sample(
        model=model, 
        cond=state_conds, 
        action_shape=action_shape, 
        num_steps=10, 
        solver='heun'
    )
    print(f"✅ Heun 采样成功! 生成动作维度: {sampled_actions_heun.shape}")
    print("🎉 模块全部测试通过！可以直接移植到 my_rl100_infra 中。")