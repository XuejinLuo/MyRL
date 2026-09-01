# train_offline.py

import os
import torch
import torch.nn as nn
import numpy as np
import hydra
from datetime import datetime 
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

# =========================================================================
# 导入你提供的核心模块
# =========================================================================
from data.dataset import PointCloudChunkDataset
from models.policy import EmbodiedGenPolicy
from models.encoders.pointnext import PointNeXtEncoder
from models.critics.q_v_network import VNetwork, TwinQNetwork
from algos.idql import IDQL
from utils.eval_utils import evaluate_and_record_video

# =========================================================================
# 桥接层 (Bridge Wrappers)
# 作用：解决 IDQL 原版算法要求 Tensor，而我们的观测是 PointCloud + State 字典的问题
# =========================================================================

class CriticFeatureExtractor(nn.Module):
    """ 为 Q / V 网络共享一个 3D 点云特征提取器 """
    def __init__(self, cfg):
        super().__init__()
        self.encoder = PointNeXtEncoder(
            in_channels=3,
            output_dim=cfg.model.cond_dim,
            use_state=cfg.model.use_state,
            state_dim=cfg.model.state_dim
        )
    def forward(self, obs_dict):
        # 强制将键名对齐为 PointNeXtEncoder 需要的 'point_cloud' 和 'state'
        pn_dict = {
            'point_cloud': obs_dict['pc'],
            'state': obs_dict['state']
        }
        return self.encoder(pn_dict)

class IDQL_VNet_Wrapper(nn.Module):
    def __init__(self, encoder, v_net):
        super().__init__()
        self.encoder = encoder
        self.v_net = v_net
    def forward(self, obs_dict):
        feat = self.encoder(obs_dict)
        return self.v_net(feat)

class IDQL_QNet_Wrapper(nn.Module):
    def __init__(self, encoder, q_net):
        super().__init__()
        self.encoder = encoder
        self.q_net = q_net
    def forward(self, obs_dict, actions):
        feat = self.encoder(obs_dict)
        return self.q_net(feat, actions)

class Policy_IDQL_Wrapper(nn.Module):
    """ 包装 EmbodiedGenPolicy 使其兼容 IDQL 的单参数调用 """
    def __init__(self, policy):
        super().__init__()
        self.policy = policy
    def compute_loss(self, obs_dict, actions):
        # 解包字典，传入 policy
        return self.policy.compute_loss(
            obs=obs_dict['pc'],
            actions=actions,
            state=obs_dict['state']
        )
    def parameters(self):
        return self.policy.parameters()

# =========================================================================
# 算法继承 (Subclassing IDQL)
# 作用：完美兼容字典观测的 keep_mask 过滤，不改动你原来 algos/idql.py 一行代码
# =========================================================================
class EmbodiedIDQL(IDQL):
    def update_critic(self, obs_dict, actions, rewards, next_obs_dict, dones):
        """ 改造：接收 dict，计算 TD Loss """
        with torch.no_grad():
            # [🔥 修复 Target Q 状态错位] IQL 中 target_q 用于评估当前 (s,a) 的价值来拟合 V，必须传入当前的 obs_dict！
            target_q1, target_q2 = self.q_target(obs_dict, actions) 
            target_q = torch.minimum(target_q1, target_q2)
            next_v = self.v_net(next_obs_dict)
            
        # 1. 更新 V
        v = self.v_net(obs_dict)
        adv = target_q - v
        v_loss = self.expectile_loss(adv, self.tau).mean()
        
        self.v_opt.zero_grad()
        v_loss.backward()
        self.v_opt.step()
        
        # 2. 更新 Q
        q_target_value = rewards + self.discount * (1.0 - dones) * next_v.detach()
        q1, q2 = self.q_net(obs_dict, actions)
        
        q1_loss = torch.nn.functional.mse_loss(q1, q_target_value)
        q2_loss = torch.nn.functional.mse_loss(q2, q_target_value)
        q_loss = q1_loss + q2_loss
        
        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()
        
        # 3. Soft update
        for param, target_param in zip(self.q_net.parameters(), self.q_target.parameters()):
            target_param.data.copy_(self.tau_target * param.data + (1 - self.tau_target) * target_param.data)
            
        return {
            "loss/v": v_loss.item(), 
            "loss/q": q_loss.item(), 
            "q_value": q1.mean().item(),
            "adv_for_actor": adv.detach()
        }

    def update_actor(self, obs_dict, actions, adv=None):
        """ 改造：在对样本进行 Reject Sampling 过滤时，正确切割字典 """
        with torch.no_grad():
            if adv is None:
                v = self.v_net(obs_dict)
                q1, q2 = self.q_target(obs_dict, actions)
                q = torch.minimum(q1, q2)
                adv = q - v
            adv_stable = adv - adv.max() 
            weights = torch.exp(self.beta * adv_stable)
            accept_prob = (weights / weights.max()).squeeze(-1)
            
            random_u = torch.rand_like(accept_prob)
            keep_mask = random_u < accept_prob
            if keep_mask.sum() == 0:
                keep_mask[torch.argmax(accept_prob)] = True

        # [核心] 使用 mask 过滤字典中的张量
        filtered_obs = {k: v_tensor[keep_mask] for k, v_tensor in obs_dict.items()}
        filtered_actions = actions[keep_mask]
        
        actor_loss = self.actor.compute_loss(filtered_obs, filtered_actions)
        
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        
        return {
            "loss/actor": actor_loss.item(), 
            "metrics/accept_ratio": keep_mask.float().mean().item(),
            "metrics/adv_mean": adv.mean().item()
        }

# =========================================================================
# Main 训练循环
# =========================================================================
@hydra.main(version_base=None, config_path="configs", config_name="train_offline")
def main(cfg: DictConfig):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{cfg.run_name}_{timestamp}"
    # 临时解除 Hydra 配置的只读限制，覆写内部路径
    OmegaConf.set_struct(cfg, False)
    cfg.run_name = run_name
    cfg.save_dir = os.path.join(cfg.save_dir, run_name)
    OmegaConf.set_struct(cfg, True)

    # 1. 初始化实验记录 (Wandb 会自动使用新的带时间戳的 cfg.run_name)
    if cfg.wandb.enable:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.run_name, 
            config=OmegaConf.to_container(cfg, resolve=True)
        )
    
    device = torch.device(cfg.device)
    os.makedirs(cfg.save_dir, exist_ok=True)

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
    data_path = cfg.dataset.data_path
    max_episodes = cfg.dataset.get("max_episodes", None)
    
    # 动态加载数据
    raw_trajectories = load_maniskill_h5(
        data_path, 
        max_episodes=max_episodes,
        workspace_bounds=cfg.dataset.get("workspace_bounds", None) 
    )
    
    # 这里的 Dataset 内部会把你那 16384 个点，随机降采样到 cfg.dataset.n_points (比如 1024 个点)，防止显存 OOM
    dataset = PointCloudChunkDataset(
        trajectories=raw_trajectories, 
        chunk_size=cfg.model.chunk_size, 
        n_points=cfg.dataset.n_points 
    )

    dataloader = DataLoader(
        dataset, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        drop_last=True,
        num_workers=4,          # 开启多进程数据加载 (可根据你的CPU核心数调整)
        pin_memory=True,        # 开启锁页内存，加速 CPU Tensor 向 GPU 的拷贝
        persistent_workers=True # 防止每个 epoch 重新创建 worker 导致的延迟
    )
    print(f"✅ 数据加载完成! Total batches per epoch: {len(dataloader)}")

    # 3. 初始化网络模型
    print("🧠 正在初始化策略与价值网络...")
    
    # Actor (Flow/Diffusion Policy)
    base_policy = EmbodiedGenPolicy(
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
        critic_lr=cfg.algo.critic_lr
    )

    # 5. 开始训练 (Training Loop)
    print(f"🔥 开始 IDQL 离线训练 (Total Epochs: {cfg.epochs})")
    
    for epoch in range(1, cfg.epochs + 1):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{cfg.epochs}", leave=False)
        epoch_metrics = {}

        for batch in pbar:
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
            
            # Note: RL 算法中 rewards 和 dones 需要 [Batch, 1] 形状才能与 V 值正常广播
            rewards = batch['reward'].to(device).unsqueeze(-1)
            dones = batch['done'].to(device).unsqueeze(-1)

            # --- A. 更新 Critic (Q & V) ---
            critic_info = agent.update_critic(obs_dict, action_chunk, rewards, next_obs_dict, dones)
            
            # --- B. 更新 Actor (Flow / Diffusion with Reject Sampling) ---
            actor_info = agent.update_actor(obs_dict, action_chunk, adv=critic_info.pop('adv_for_actor'))
            
            # 合并日志
            step_metrics = {**critic_info, **actor_info}
            
            # 更新累加器以计算 Epoch 均值
            for k, v in step_metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0) + v
                
            pbar.set_postfix({"v_loss": f"{step_metrics['loss/v']:.3f}", "act_loss": f"{step_metrics['loss/actor']:.3f}"})

        # 计算 Epoch 平均指标并上传 WandB
        for k in epoch_metrics:
            epoch_metrics[k] /= len(dataloader)
            
        if cfg.wandb.enable:
            wandb.log(epoch_metrics, step=epoch)

        print(f"Epoch {epoch:03d} | Actor Loss: {epoch_metrics['loss/actor']:.4f} | Q Value: {epoch_metrics['q_value']:.4f} | Accept Ratio: {epoch_metrics['metrics/accept_ratio']*100:.1f}%")

        # 定期保存权重 (Save Checkpoint)
        if epoch % 2 == 0 or epoch == cfg.epochs:
            ckpt_path = os.path.join(cfg.save_dir, f"idql_policy_ep{epoch}.pth")
            # 仅保存 EmbodiedGenPolicy 的权重，方便后续推理和 Distill 直接加载
            torch.save(base_policy.state_dict(), ckpt_path)
            print(f"   💾 Saved Checkpoint to {ckpt_path}")
            evaluate_and_record_video(cfg, base_policy, epoch, device)

    if cfg.wandb.enable:
        wandb.finish()
    print("🎉 离线预训练彻底完成！")

if __name__ == "__main__":
    main()