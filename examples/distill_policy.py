# distill_policy.py
import os
import copy
import time
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

# 引入你写好的核心组件
from models.policy import EmbodiedGenPolicy
from algos.distill import OneStepDistiller
from examples.legacy_dataset import PointCloudChunkDataset

def parse_args():
    parser = argparse.ArgumentParser(description="One-step Distillation for Flow/Diffusion Policy")
    # 环境与模型超参数
    parser.add_argument("--action_dim", type=int, default=7, help="Robot action dimension")
    parser.add_argument("--state_dim", type=int, default=7, help="Robot state dimension")
    parser.add_argument("--chunk_size", type=int, default=16, help="Action chunk length")
    parser.add_argument("--n_points", type=int, default=1024, help="Point cloud downsample size")
    parser.add_argument("--algo", type=str, default="flow", choices=["flow", "diffusion"], help="Base algorithm")
    
    # 蒸馏超参数
    parser.add_argument("--epochs", type=int, default=50, help="Distillation training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Student learning rate")
    parser.add_argument("--teacher_steps", type=int, default=10, help="Number of ODE steps teacher uses to generate pseudo-labels")
    
    # 路径与工程超参数
    parser.add_argument("--teacher_ckpt", type=str, default="", help="Path to pre-trained teacher checkpoint (optional)")
    parser.add_argument("--save_dir", type=str, default="./outputs/distill", help="Directory to save student checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # [性能核心] 是否开启伪标签预计算 (强烈建议开启)
    parser.add_argument("--precompute", action="store_true", help="Precompute Teacher ODE steps before training loop to save massive time")
    
    return parser.parse_args()


def generate_mock_trajectories(num_episodes=10, ep_len=100, n_points=2048, state_dim=7, action_dim=7):
    """ 生成一些 Mock 数据，用于在没有真实数据集时验证代码是否跑通 """
    print(f"📦 生成 Mock 数据集 (Episodes: {num_episodes})...")
    trajectories = []
    for _ in range(num_episodes):
        ep = {
            'pc': np.random.randn(ep_len, n_points, 3).astype(np.float32),
            'state': np.random.randn(ep_len, state_dim).astype(np.float32),
            'action': np.random.randn(ep_len, action_dim).astype(np.float32),
            'reward': np.random.rand(ep_len).astype(np.float32),
            'done': np.zeros(ep_len, dtype=bool)
        }
        ep['done'][-1] = True
        trajectories.append(ep)
    return trajectories


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.save_dir = os.path.join(args.save_dir, f"run_{timestamp}")
    os.makedirs(args.save_dir, exist_ok=True)
    
    # =========================================================================
    # 1. 准备数据 (真实部署时，替换为读取你的 Zarr / HDF5 / Pickle 文件)
    # =========================================================================
    raw_trajectories = generate_mock_trajectories(
        num_episodes=20, 
        state_dim=args.state_dim, 
        action_dim=args.action_dim
    )
    
    dataset = PointCloudChunkDataset(
        trajectories=raw_trajectories, 
        chunk_size=args.chunk_size, 
        n_points=args.n_points
    )
    print(f"✅ 数据准备完成: 共 {len(dataset)} 个 Transitions")

    # =========================================================================
    # 2. 初始化 Teacher 和 Student 策略网络
    # =========================================================================
    print(f"🧠 初始化 Teacher Policy ({args.algo.upper()})...")
    teacher_policy = EmbodiedGenPolicy(
        action_dim=args.action_dim, 
        chunk_size=args.chunk_size,
        use_state=True, 
        state_dim=args.state_dim,
        encoder_type="pointnext", 
        backbone_type="transformer",
        algo_type=args.algo
    ).to(args.device)

    # 尝试加载教师权重 (如果有)
    if args.teacher_ckpt and os.path.exists(args.teacher_ckpt):
        teacher_policy.load_state_dict(torch.load(args.teacher_ckpt, map_location=args.device))
        print(f"   📥 成功加载 Teacher 权重: {args.teacher_ckpt}")
    else:
        print("   ⚠️ 未提供有效的 Teacher 权重，将使用随机初始化的 Teacher 进行流程测试。")
    
    # 锁定 Teacher，设为评估模式
    teacher_policy.eval()
    for param in teacher_policy.parameters():
        param.requires_grad = False

    print("👶 初始化 Student Policy (从 Teacher 深拷贝以加速收敛)...")
    # 蒸馏的最佳实践：用 Teacher 的权重初始化 Student
    student_policy = copy.deepcopy(teacher_policy)
    student_policy.train()
    for param in student_policy.parameters():
        param.requires_grad = True # 解锁 Student

    # =========================================================================
    # 3. 初始化蒸馏器 (Distiller)
    # =========================================================================
    distiller = OneStepDistiller(
        teacher_model=teacher_policy,
        student_model=student_policy,
        device=args.device,
        lr=args.lr,
        teacher_steps=args.teacher_steps,
        use_ema=True
    )
    
    # =========================================================================
    # 4. [性能优化] Precompute Targets (提前计算 Teacher 的多步输出)
    # =========================================================================
    # 由于 Teacher 生成 Action Chunk 需要解 ODE/DDIM (如 10 步)，如果放在每个 Epoch 的 DataLoader 里，
    # 会极大拖慢训练速度。如果显存/内存允许，最好的做法是提前生成一次伪标签存下来。
    if args.precompute:
        print("\n⚡ 开启高速模式：正在离线预计算 Teacher 的多步目标生成 (Pseudo-labels)...")
        # 遍历数据集（使用顺序遍历 DataLoader 进行模拟）
        # 在真实的大规模数据集中，你可能需要单独存入硬盘
        precompute_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        
        precomputed_dataset = [] # [修复]: 将预计算结果展平存入列表，避免 shuffle 导致的数据错位
        
        for batch in tqdm(precompute_loader, desc="Precomputing Target Actions"):
            obs = batch['pc'].to(args.device)
            state = batch['state'].to(args.device)
            B = obs.shape[0]
            
            # 提前采样固定的噪声
            z_noise = torch.randn((B, args.chunk_size, args.action_dim), device=args.device)
            # 调用 Teacher 多步生成
            with torch.no_grad():
                target_action = distiller.teacher_generate(obs, state, z_noise)
            
            # [修复]: 拆解 Batch 为单个样本 (Sample) 存入列表，转移至 CPU 节省显存
            for i in range(B):
                precomputed_dataset.append({
                    'pc': obs[i].cpu(),
                    'state': state[i].cpu(),
                    'noise': z_noise[i].cpu(),
                    'target': target_action[i].cpu()
                })
        
        # 预计算模式下，使用携带 noise 和 target 的新 DataLoader，彻底安全地开启 shuffle
        dataloader = DataLoader(precomputed_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        print(f"✅ 高速模式准备完毕: 生成了 {len(precomputed_dataset)} 个包含伪标签的样本。")
    else:
        # 普通模式下，使用原始 Dataset 实时生成
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        print(f"🐢 普通模式准备完毕: 训练期间将实时执行 ODE，速度较慢。")

    # =========================================================================
    # 5. 蒸馏主训练循环 (Distillation Loop)
    # =========================================================================
    print(f"\n🔥 开始单步蒸馏训练 (Total Epochs: {args.epochs})...")
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        start_time = time.time()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for batch in pbar:
            obs = batch['pc'].to(args.device)
            state = batch['state'].to(args.device)
            
            # 检查是否使用了预计算的高速数据格式
            if args.precompute:
                pre_z = batch['noise'].to(args.device)
                pre_target = batch['target'].to(args.device)
                
                # 高速更新: 直接用拟合好的目标计算速度场匹配
                loss = distiller.update_step(obs, state, precomputed_noise=pre_z, precomputed_target=pre_target)
            else:
                # 在线/龟速更新: 现场跑 10 步 ODE (通常用于在强化学习 Rollout 中的在线蒸馏)
                loss = distiller.update_step(obs, state)
                
            epoch_loss += loss
            pbar.set_postfix({"Loss": f"{loss:.4f}"})
            
        epoch_loss /= len(dataloader)
        elapsed = time.time() - start_time
        print(f"Epoch {epoch:03d} | Avg Loss: {epoch_loss:.6f} | Time: {elapsed:.2f}s")
        
        # 定期保存权重 (保存 EMA 版本的 Student 性能更稳定)
        if epoch % 10 == 0 or epoch == args.epochs:
            ckpt_dir = os.path.join(args.save_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"student_distilled_ep{epoch}.pth")
            
            torch.save(distiller.ema_student.state_dict(), ckpt_path)
            print(f"   💾 Saved EMA Student Checkpoint to {ckpt_path}")

    print("\n🎉 单步蒸馏结束！Student 现已具备极致响应速度，可以部署到真机环境中了。")
    print("👉 部署提示：在推理时，直接调用 student_policy.sample(..., num_steps=1) 即可。")

if __name__ == '__main__':
    main()