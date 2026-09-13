# algos/idql.py
"""IDQL 更新逻辑；策略和 Q/V 网络由训练入口传入。"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class IDQL:
    """ 
    Implicit Diffusion Q-Learning (基于 3D Point Cloud 与 Action Chunk) 
    支持通过 Reject Sampling (拒绝采样) 训练 Flow/Diffusion Actor。
    """
    def __init__(
        self,
        actor: nn.Module,
        q_network: nn.Module,
        v_network: nn.Module,
        device="cuda",
        tau=0.7,             # Expectile 回归系数 (IQL 核心参数)
        discount=0.99,
        beta=3.0,            # Reject Sampling 温度系数
        actor_lr=1e-4,
        critic_lr=3e-4,
        tau_target=0.005,     # Target 网络软更新系数
        use_bc_only=False
    ):
        self.actor = actor.to(device)
        self.q_net = q_network.to(device)
        self.v_net = v_network.to(device)
        self.q_target = copy.deepcopy(self.q_net).to(device)
        
        self.device = device
        self.tau = tau
        self.discount = discount
        self.beta = beta
        self.tau_target = tau_target
        self.use_bc_only = use_bc_only
        
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.q_opt = torch.optim.Adam(self.q_net.parameters(), lr=critic_lr)
        self.v_opt = torch.optim.Adam(self.v_net.parameters(), lr=critic_lr)

    def expectile_loss(self, diff, expectile):
        """ Asymmetric L2 Loss (IQL 用于拟合 V 网络) """
        weight = torch.where(diff > 0, expectile, (1 - expectile))
        return weight * (diff ** 2)

    def update_critic(self, states, actions, rewards, next_states, dones):
        """ 离线阶段：更新 Q 和 V 网络 """
        with torch.no_grad():
            target_q1, target_q2 = self.q_target(states, actions)
            target_q = torch.minimum(target_q1, target_q2)
            next_v = self.v_net(next_states)
            
        # 1. 更新 V 网络 (Expectile Regression)
        v = self.v_net(states)
        adv = target_q - v
        v_loss = self.expectile_loss(adv, self.tau).mean()
        
        self.v_opt.zero_grad()
        v_loss.backward()
        self.v_opt.step()
        
        # 2. 更新 Q 网络 (TD Learning)
        q_target_value = rewards + self.discount * (1.0 - dones) * next_v.detach()
        q1, q2 = self.q_net(states, actions)
        
        q1_loss = F.mse_loss(q1, q_target_value)
        q2_loss = F.mse_loss(q2, q_target_value)
        q_loss = q1_loss + q2_loss
        
        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()
        
        # 3. Soft update Target Q Network
        for param, target_param in zip(self.q_net.parameters(), self.q_target.parameters()):
            target_param.data.copy_(self.tau_target * param.data + (1 - self.tau_target) * target_param.data)
            
        return {"v_loss": v_loss.item(), "q_loss": q_loss.item()}

    def update_actor(self, states, actions):
        """ 提取阶段：使用 Reject Sampling 训练 Flow/Diffusion 策略 """
        with torch.no_grad():
            v = self.v_net(states)
            q1, q2 = self.q_target(states, actions)
            q = torch.minimum(q1, q2)
            
            # 计算 Advantage
            adv = q - v
            
            # [🔥核心修复] 减去 advantage 的最大值，防止 torch.exp 指数爆炸变为 NaN
            adv_stable = adv - adv.max() 
            
            # Reject Sampling 核心概率计算: exp(beta * adv) 归一化到 [0, 1]
            weights = torch.exp(self.beta * adv_stable)
            max_weight = weights.max()
            accept_prob = (weights / max_weight).squeeze(-1) # [B]
            
            # 生成均匀分布 U(0,1) 进行过滤
            random_u = torch.rand_like(accept_prob)
            keep_mask = random_u < accept_prob
            
            # 安全校验: 确保至少保留一个样本，避免 Loss 变成 NaN
            if keep_mask.sum() == 0:
                keep_mask[torch.argmax(accept_prob)] = True

        # 仅对被保留的高质量样本计算 Diffusion/Flow Loss
        filtered_states = states[keep_mask]
        filtered_actions = actions[keep_mask]
        
        actor_loss = self.actor.compute_loss(filtered_states, filtered_actions)
        
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        
        return {
            "actor_loss": actor_loss.item(), 
            "accept_ratio": keep_mask.float().mean().item(),
            "adv_mean": adv.mean().item()
        }
