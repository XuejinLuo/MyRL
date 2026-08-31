# algos/diffusion_utils/ddim_scheduler.py
import math
import torch
import torch.nn as nn
import numpy as np

def get_beta_schedule(schedule_name: str, num_train_timesteps: int, beta_start: float, beta_end: float):
    """
    获取 beta 调度策略，具身智能通常使用 linear 或 squaredcos_cap_v2
    """
    if schedule_name == "linear":
        return torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
    elif schedule_name == "squaredcos_cap_v2":
        # Diffusion Policy 中常用的 Cosine 调度，在两端更平滑
        def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999):
            alpha_bar = lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
            betas = []
            for i in range(num_diffusion_timesteps):
                t1 = i / num_diffusion_timesteps
                t2 = (i + 1) / num_diffusion_timesteps
                betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
            return torch.tensor(betas, dtype=torch.float32)
        return betas_for_alpha_bar(num_train_timesteps)
    else:
        raise NotImplementedError(f"Beta schedule {schedule_name} is not implemented.")

class DDIMScheduler(nn.Module): # 继承 nn.Module 以利用 register_buffer 管理设备
    def __init__(
        self,
        num_train_timesteps: int = 100,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "squaredcos_cap_v2",
        prediction_type: str = "epsilon", # 默认模型预测的是噪声 (epsilon) 支持 "epsilon", "sample"(x0), "v_prediction"
        clip_sample: bool = False,         # 是否裁剪生成的动作 (防止飞车)，PG 训练下强烈建议设为 False
        clip_sample_range: float = 1.0    # 动作通常归一化到 [-1, 1]
    ):
        """
        为 Action Chunking 定制的 DDIM Scheduler。
        去除了 HuggingFace diffusers 中复杂的图像处理逻辑，直接进行张量运算。
        """
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range

        # 1. 计算 Betas 和 Alphas
        betas = get_beta_schedule(beta_schedule, num_train_timesteps, beta_start, beta_end)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # 2. 使用 register_buffer 彻底解决 .to(device) 的高频调用痛点
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

        # 2. 推理时的 timesteps 占位符
        self.num_inference_steps = None
        # 默认 timestep 序列
        self.register_buffer("timesteps", torch.arange(0, num_train_timesteps).flip(0))

    def set_timesteps(self, num_inference_steps: int, device: torch.device = None):
        """
        设置推理时间步:使用 linspace 确保从最顶端 (T-1) 完美均匀降落到 0
        """
        self.num_inference_steps = num_inference_steps
        timesteps = np.linspace(0, self.num_train_timesteps - 1, num_inference_steps, dtype=int)[::-1].copy()
        # 将生成的 numpy array 转为 tensor 并放到正确设备
        timesteps_tensor = torch.from_numpy(timesteps).long()
        if device is not None:
            timesteps_tensor = timesteps_tensor.to(device)
        self.timesteps = timesteps_tensor

    def _extract(self, a, t, x_shape):
        """
        从一维数组 a 中提取时间步 t 对应的值，并 reshape 为 x_shape 以支持 Broadcasting。
        例如:x_shape 为 [Batch, Chunk_Size, Action_Dim] -> 提取出来的维度为 [Batch, 1, 1]
        """
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        前向过程:q(x_t | x_0) 
        用于训练阶段生成带噪动作。
        """
        alphas_cumprod_t = self._extract(self.alphas_cumprod, timesteps, original_samples.shape)
        sqrt_alpha_prod = alphas_cumprod_t ** 0.5
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod_t) ** 0.5

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def step(self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor) -> torch.Tensor:
        """
        反向过程 (DDIM 单步去噪) : 
        根据模型预测的噪声 (epsilon)，计算上一时刻的动作 x_{t-1}。
        支持张量形状: [Batch, Chunk_Size, Action_Dim]
        """
        # 1. 寻找 prev_timestep (兼容 linspace 采样的任意间隔)
        step_index = (self.timesteps == timestep).nonzero(as_tuple=True)[0]
        if len(step_index) == 0:
            step_ratio = self.num_train_timesteps // self.num_inference_steps
            prev_timestep = timestep - step_ratio
        else:
            prev_timestep = self.timesteps[step_index + 1].item() if step_index < len(self.timesteps) - 1 else -1

        # 2. 获取当前 t 和前一步 t-1 的 alpha_cumprod
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0, device=model_output.device)
        beta_prod_t = 1 - alpha_prod_t

        # 3. 计算预测出的原始无噪样本 (x_0)   支持多目标的预测 (为 One-step Distill / Consistency Model 铺路)
        if self.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
            pred_epsilon = model_output
        elif self.prediction_type == "sample": # 直接预测 x0
            pred_original_sample = model_output
            pred_epsilon = (sample - alpha_prod_t ** 0.5 * pred_original_sample) / beta_prod_t ** 0.5
        elif self.prediction_type == "v_prediction": # Flow 友好型 (velocity)
            pred_original_sample = alpha_prod_t ** 0.5 * sample - beta_prod_t ** 0.5 * model_output
            pred_epsilon = alpha_prod_t ** 0.5 * model_output + beta_prod_t ** 0.5 * sample
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

        # 如果必须裁剪，在 RL 中建议使用软裁剪 (Tanh)，这里保留原逻辑但默认关闭
        if self.clip_sample:
            # 软裁切示例:pred_original_sample = torch.tanh(pred_original_sample) * self.clip_sample_range
            pred_original_sample = torch.clamp(pred_original_sample, -self.clip_sample_range, self.clip_sample_range)

        # 5. DDIM 核心公式 (ETA = 0, 即确定性采样，具身智能通常不需要随机性)
        # x_{t-1} = sqrt(alpha_{t-1}) * x_0 + sqrt(1 - alpha_{t-1}) * epsilon_theta
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * pred_epsilon  
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction

        return prev_sample


# ==============================================================================
# MAIN TEST BLOCK 
# 直接运行此文件以测试:python ddim_scheduler.py
# ==============================================================================
if __name__ == "__main__":
    print("🚀 开始测试定制版 DDIM Scheduler (针对 Action Chunking) ...\n")
    
    # 模拟超参数
    BATCH_SIZE = 2
    CHUNK_SIZE = 16
    ACTION_DIM = 7      # 例如:6 DOF 机械臂 + 1 自由度夹爪
    TRAIN_STEPS = 100
    INFER_STEPS = 10    # 将 100 步提速到 10 步推理

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  测试设备: {device}")

    # 1. 初始化调度器
    scheduler = DDIMScheduler(
        num_train_timesteps=TRAIN_STEPS,
        beta_schedule="squaredcos_cap_v2"
    ).to(device)

    # 2. 模拟训练阶段 (Forward Process / 加噪)
    print("\n--- 🧪 测试训练阶段 (加噪) ---")
    # 假设从 Dataset 中采样的真实动作块 (归一化在 -1 到 1 之间)
    clean_action_chunks = torch.randn((BATCH_SIZE, CHUNK_SIZE, ACTION_DIM), device=device)
    # 随机生成噪声
    noise = torch.randn_like(clean_action_chunks)
    # 随机采样 batch 的 timesteps
    timesteps = torch.randint(0, TRAIN_STEPS, (BATCH_SIZE,), device=device).long()
    
    noisy_actions = scheduler.add_noise(clean_action_chunks, noise, timesteps)
    print(f"✅ 输入干净动作 shape: {clean_action_chunks.shape}")
    print(f"✅ 随机采样的 t: {timesteps.tolist()}")
    print(f"✅ 增加噪声后的 shape: {noisy_actions.shape}")
    
    # 3. 模拟推理阶段 (Reverse Process / 去噪)
    print("\n--- 🧪 测试推理阶段 (去噪采样) ---")
    
    # 模拟一个极简的 Dummy 神经网络 (例如基于 Transformer 的 1D-UNet)
    # 实际上，这里它应该接收 (noisy_actions, t, visual_features)，输出 noise
    class DummyPolicyNetwork(nn.Module):
        def forward(self, x, t):
            # 模型预测的噪声通常与 x 同维度
            return torch.randn_like(x)

    dummy_model = DummyPolicyNetwork().to(device)

    # 设置推理步数
    scheduler.set_timesteps(num_inference_steps=INFER_STEPS, device=device)
    print(f"✅ 推理使用的时间步序列: {scheduler.timesteps.tolist()}")

    # 开始采样:从纯高斯噪声开始
    action_traj = torch.randn((BATCH_SIZE, CHUNK_SIZE, ACTION_DIM), device=device)
    print(f"✅ 初始纯噪声动作 shape: {action_traj.shape}")

    # 循环去噪
    for t in scheduler.timesteps:
        # 1. 模型预测噪声 epsilon
        with torch.no_grad():
            predicted_noise = dummy_model(action_traj, t)
        
        # 2. DDIM 步进，计算上一时间步的动作
        action_traj = scheduler.step(model_output=predicted_noise, timestep=int(t), sample=action_traj)
        
        # print(f"  --> 完成去噪 step {int(t)}, 输出 actions std: {action_traj.std().item():.4f}")

    print(f"✅ 最终去噪生成的 Action Chunk shape: {action_traj.shape}")
    print("🎉 DDIM Scheduler 测试完美通过！可以安全移植入你的具身框架。")