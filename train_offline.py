# train_offline.py

import os
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import hydra
import copy
from datetime import datetime 
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

# =========================================================================
# 导入你提供的核心模块
# =========================================================================
from data.dataset import PointCloudChunkDataset
from models.policy import EmbodiedGenPolicy
from models.encoders.pointnext import PointNeXtEncoder
from models.critics.q_v_network import VNetwork, TwinQNetwork
from algos.idql import IDQL
from utils.experiment import evaluate_base, log_metrics, write_json
from utils.online_eval import seed_all, preserve_rng
from utils.debug_logger import DebugLogger
from utils.normalizer import MinMaxNormalizer

# =========================================================================
# 桥接层 (Bridge Wrappers)
# 作用：解决 IDQL 原版算法要求 Tensor，而我们的观测是 PointCloud + State 字典的问题
# =========================================================================

from algos.embodied_idql import (CriticFeatureExtractor, IDQL_VNet_Wrapper,
    IDQL_QNet_Wrapper, Policy_IDQL_Wrapper, EmbodiedIDQL)

def verify_overfitting_actions(cfg, policy, batch, normalizer, epoch, device):
    """
    直接传入从主循环截获的 batch,通过网络推理,并与真实的 Action 对比。
    """
    policy.eval()
    
    # 我们只取 Batch 中的第 0 个样本进行详细对比
    pc_t = batch['pc'][0:1].to(device)            # shape: [1, N, 3(或6)]
    state_t = batch['state'][0:1].to(device)      # shape: [1, State_Dim]
    gt_action = batch['action_chunk'][0:1].cpu().numpy() # 归一化状态下的真实动作 [1, Chunk, A_Dim]
    
    # 通过网络生成预测动作 (Diffusion/Flow 需要采样多步)
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16) if device.type == 'cuda' else torch.no_grad():
            pred_action = policy.sample(
                obs=pc_t, 
                state=state_t, 
                num_steps=cfg.model.get("num_inference_steps", 10)
            )
    pred_action = pred_action.cpu().float().numpy()

    # 反归一化，还原到真实的物理物理范围
    if normalizer is not None:
        # normalizer 处理时通常只关注最后一维，切片去掉 Batch 维度
        gt_action_real = normalizer.unnormalize(gt_action[0], 'action')
        pred_action_real = normalizer.unnormalize(pred_action[0], 'action')
    else:
        gt_action_real = gt_action[0]
        pred_action_real = pred_action[0]

    # 将结果写入文本文件
    debug_dir = os.path.join(cfg.save_dir, "debug_logs")
    os.makedirs(debug_dir, exist_ok=True)
    out_file = os.path.join(debug_dir, f"action_compare_ep{epoch}.txt")
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(f"========== Epoch {epoch} 动作过拟合对比 ==========\n")
        f.write("说明: 比较网络推理出的 Action 与 数据集中的真实 Action\n\n")
        
        # 遍历 Chunk 中的每一步
        for i in range(gt_action_real.shape[0]):
            f.write(f"--- Chunk Step {i} ---\n")
            
            # 使用 np.round 保留 4 位小数，使对齐更好看
            gt_str = np.array2string(gt_action_real[i], precision=4, suppress_small=True, separator=', ')
            pred_str = np.array2string(pred_action_real[i], precision=4, suppress_small=True, separator=', ')
            
            # 计算这一步的绝对误差
            error = np.mean(np.abs(gt_action_real[i] - pred_action_real[i]))
            
            f.write(f"真实 (GT)  : {gt_str}\n")
            f.write(f"预测 (Pred): {pred_str}\n")
            f.write(f"MAE 误差   : {error:.6f}\n\n")

    print(f"\n✅ 动作直观对比已生成，请在左侧文件树查看: {out_file}")
    
    # 恢复训练模式
    policy.train()

# =========================================================================
# Main 训练循环
# =========================================================================
@hydra.main(version_base=None, config_path="configs", config_name="train_offline")
def main(cfg: DictConfig):
    seed_all(cfg.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{cfg.run_name}_{timestamp}"
    # 临时解除 Hydra 配置的只读限制，覆写内部路径
    OmegaConf.set_struct(cfg, False)
    cfg.run_name = run_name
    cfg.save_dir = hydra.utils.to_absolute_path(cfg.get("output") or os.path.join(cfg.save_dir, run_name))
    if Path(cfg.save_dir).exists():
        raise FileExistsError(f"Output already exists: {cfg.save_dir}")
    OmegaConf.set_struct(cfg, True)
    debug_logger = DebugLogger(cfg.save_dir)

    # 1. 初始化实验记录 (Wandb 会自动使用新的带时间戳的 cfg.run_name)
    if cfg.wandb.enable:
        import wandb
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.run_name, 
            config=OmegaConf.to_container(cfg, resolve=True)
        )
    
    device = torch.device(cfg.device)
    os.makedirs(cfg.save_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(cfg.save_dir, "config.yaml"), resolve=True)

    # 2. 准备数据流 (Dataset & DataLoader)
    # # 实际项目中，替换 generate_mock_trajectories 为你的本地数据读取逻辑
    # print("📦 正在加载离线数据...")
    # from distill_policy import generate_mock_trajectories # 借用你写好的 mock
    # raw_trajectories = generate_mock_trajectories(num_episodes=50, state_dim=cfg.model.state_dim, action_dim=cfg.model.action_dim)
    
    # dataset = PointCloudChunkDataset(
    #     trajectories=raw_trajectories, 
    #     chunk_size=cfg.model.chunk_size, 
    #     n_points=cfg.dataset.n_points
    # )

    print("📦 正在加载 ManiSkill 真机/仿真离线数据...")
    
    # 导入读取函数
    from data.maniskill_dataset import load_maniskill_h5 
    
    # 从 Hydra 配置中读取路径和限制参数
    data_path = hydra.utils.to_absolute_path(cfg.dataset.data_path)
    max_episodes = cfg.dataset.get("max_episodes", None)
    
    # 动态加载数据
    raw_trajectories = load_maniskill_h5(
        data_path, 
        max_episodes=max_episodes,
        workspace_bounds=cfg.dataset.get("workspace_bounds", None),
        n_points=cfg.dataset.n_points
    )

    print("📊 正在统计归一化参数 (Normalization Stats)...")
    all_actions = np.concatenate([ep['action'] for ep in raw_trajectories], axis=0)
    all_states = np.concatenate([ep['state'] for ep in raw_trajectories], axis=0)

    normalizer = MinMaxNormalizer()
    normalizer.fit({
        'action': all_actions,
        'state': all_states
    })
    
    ckpt_dir = os.path.join(cfg.save_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    # 保存到这次实验的文件夹下，确保跟 Checkpoint 绑定！
    stats_path = os.path.join(ckpt_dir, "dataset_stats.json")
    normalizer.save(stats_path)
    print(f"✅ 归一化参数已保存至 {stats_path}")

    
    # 这里的 Dataset 内部会把你那 16384 个点，随机降采样到 cfg.dataset.n_points (比如 1024 个点)，防止显存 OOM
    dataset = PointCloudChunkDataset(
        trajectories=raw_trajectories, 
        chunk_size=cfg.model.chunk_size, 
        n_points=cfg.dataset.n_points,
        normalizer=normalizer,
        workspace_bounds=cfg.dataset.get("workspace_bounds", None)
    )

    dataloader = DataLoader(
        dataset, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        drop_last=True,
        num_workers=cfg.num_workers,          # 开启多进程数据加载 (可根据你的CPU核心数调整)
        pin_memory=True,        # 开启锁页内存，加速 CPU Tensor 向 GPU 的拷贝
        persistent_workers=cfg.num_workers > 0 # 防止每个 epoch 重新创建 worker 导致的延迟
    )
    if len(dataloader) == 0:
        raise ValueError("Dataset smaller than batch_size")
    print(f"✅ 数据加载完成! Total batches per epoch: {len(dataloader)}")

    # 3. 初始化网络模型
    print("🧠 正在初始化策略与价值网络...")
    
    # Actor (Flow/Diffusion Policy)
    base_policy = EmbodiedGenPolicy(
        in_channels=cfg.model.get("in_channels", 3),
        action_dim=cfg.model.action_dim,
        chunk_size=cfg.model.chunk_size,
        use_state=cfg.model.use_state,
        state_dim=cfg.model.state_dim,
        encoder_type=cfg.model.encoder_type,
        backbone_type=cfg.model.backbone_type,
        cond_dim=cfg.model.cond_dim,
        algo_type=cfg.model.algo_type
    ).to(device)
    wrapped_actor = Policy_IDQL_Wrapper(base_policy)

    # 初始化 EMA 策略模型
    print("🧠 正在初始化 EMA 策略模型...")
    ema_policy = copy.deepcopy(base_policy).to(device)
    ema_policy.eval() # EMA 模型不参与梯度传播，永远在 eval 模式
    for param in ema_policy.parameters():
        param.requires_grad = False
    
    ema_decay = 0.999 # Diffusion/Flow 常用指数衰减率 (建议 0.999 或 0.9999)

    # Critics (V & Q Networks)
    # [🔥 修复双重优化器冲突] 给 V 网络和 Q 网络分配各自独立的 PointCloud 编码器。
    # 这样防止 v_opt 和 q_opt 在 backward 时互相覆盖/破坏同一个 encoder 的动量与梯度。
    v_encoder = CriticFeatureExtractor(cfg).to(device)
    q_encoder = CriticFeatureExtractor(cfg).to(device)
    
    base_v_net = VNetwork(state_dim=cfg.model.cond_dim).to(device)
    base_q_net = TwinQNetwork(state_dim=cfg.model.cond_dim, action_dim=cfg.model.action_dim, chunk_size=cfg.model.chunk_size).to(device)
    
    wrapped_v_net = IDQL_VNet_Wrapper(v_encoder, base_v_net)
    wrapped_q_net = IDQL_QNet_Wrapper(q_encoder, base_q_net)

    # 4. 初始化 IDQL 算法引擎
    agent = EmbodiedIDQL(
        actor=wrapped_actor,
        q_network=wrapped_q_net,
        v_network=wrapped_v_net,
        device=device,
        tau=cfg.algo.tau,          # 统一使用 cfg.algo
        discount=cfg.algo.discount,
        beta=cfg.algo.beta,
        tau_target=cfg.algo.tau_target,
        actor_lr=cfg.algo.actor_lr,
        critic_lr=cfg.algo.critic_lr,
        use_bc_only=cfg.algo.get("use_bc_only", False)
    )

    # 5. 开始训练 (Training Loop)
    print(f"🔥 开始 IDQL 离线训练 (Total Epochs: {cfg.epochs})")
    
    best_score = None
    for epoch in range(1, cfg.epochs + 1):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{cfg.epochs}", leave=False)
        epoch_metrics = {}

        for step, batch in enumerate(pbar): 
            # 数据迁移至 Device 并拆解
            obs_dict = {
                'pc': batch['pc'].to(device),
                'state': batch['state'].to(device)
            }
            next_obs_dict = {
                'pc': batch['next_pc'].to(device),
                'state': batch['next_state'].to(device)
            }
            action_chunk = batch['action_chunk'].to(device)
            debug_logger.log_io(epoch, step, obs_dict, action_chunk)
            
            # Note: RL 算法中 rewards 和 dones 需要 [Batch, 1] 形状才能与 V 值正常广播
            rewards = batch['reward'].to(device).unsqueeze(-1)
            dones = batch['done'].to(device).unsqueeze(-1)

            # --- A. 更新 Critic (Q & V) ---
            critic_info = agent.update_critic(obs_dict, action_chunk, rewards, next_obs_dict, dones)
            
            # --- B. 更新 Actor (Flow / Diffusion with Reject Sampling) ---
            actor_info = agent.update_actor(obs_dict, action_chunk, adv=critic_info.pop('adv_for_actor'))

            # 每步软更新 EMA 权重
            with torch.no_grad():
                # 1. 更新网络权重参数
                for ema_param, param in zip(ema_policy.parameters(), base_policy.parameters()):
                    # ema_weight = decay * ema_weight + (1 - decay) * current_weight
                    ema_param.data.mul_(ema_decay).add_(param.data, alpha=1.0 - ema_decay)
                # 2. 同步网络缓冲区 (核心：同步 BatchNorm 的 running_mean 和 running_var)
                for ema_buffer, buffer in zip(ema_policy.buffers(), base_policy.buffers()):
                    # 由于 buffers 中通常存的是统计量或常数，推荐直接 hard copy
                    ema_buffer.data.copy_(buffer.data)
            
            # 合并日志
            step_metrics = {**critic_info, **actor_info}
            
            # 更新累加器以计算 Epoch 均值
            for k, v in step_metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0) + v
                
            pbar.set_postfix({"v_loss": f"{step_metrics['loss/v']:.3f}", "act_loss": f"{step_metrics['loss/actor']:.3f}"})

        # 计算 Epoch 平均指标并上传 WandB
        for k in epoch_metrics:
            epoch_metrics[k] /= len(dataloader)
        debug_logger.log_metrics(epoch, epoch_metrics)
            
        if cfg.wandb.enable:
            wandb.log(epoch_metrics, step=epoch)

        print(f"Epoch {epoch:03d} | Actor Loss: {epoch_metrics['loss/actor']:.4f} | Q Value: {epoch_metrics['q_value']:.4f} | Accept Ratio: {epoch_metrics['metrics/accept_ratio']*100:.1f}%")

        result = {}
        if epoch % cfg.eval.every == 0 or epoch == cfg.epochs:
            result = evaluate_base(cfg, ema_policy, normalizer, cfg.save_dir, epoch)
        state = dict(model_state_dict=base_policy.state_dict(),
                     ema_model_state_dict=ema_policy.state_dict(), normalizer=normalizer.stats,
                     config=OmegaConf.to_container(cfg, resolve=True), epoch=epoch, metrics=result)
        if result:
            score = (result['Eval/Success_Rate'], result['Eval/Mean_Reward'])
            if best_score is None or score > best_score:
                best_score = score
                torch.save(state, os.path.join(ckpt_dir, 'best.pth'))
                write_json(Path(cfg.save_dir)/'selection.json', dict(epoch=epoch, metrics=result,
                    checkpoint=str(Path(ckpt_dir)/'best.pth'), weight_key='ema_model_state_dict'))
        if epoch % cfg.save_epoch == 0 or epoch == cfg.epochs:
            torch.save(state, os.path.join(ckpt_dir, f'idql_policy_ep{epoch}.pth'))
            torch.save(state, os.path.join(ckpt_dir, 'last.pth'))
            if cfg.get('debug_actions', False):
                with preserve_rng():
                    verify_overfitting_actions(cfg, ema_policy, batch, normalizer, epoch, device)
        log_metrics(cfg.save_dir, epoch, {**epoch_metrics, **result}, stage='offline')

    if cfg.wandb.enable:
        wandb.finish()
    print("🎉 离线预训练彻底完成！")

if __name__ == "__main__":
    main()
