"""Dictionary-observation adapters shared by offline entry points."""
import torch
from torch import nn
from algos.idql import IDQL
from models.encoders.pointnext import PointNeXtEncoder

class CriticFeatureExtractor(nn.Module):
    """ 为 Q / V 网络共享一个 3D 点云特征提取器 """
    def __init__(self, cfg):
        super().__init__()
        self.encoder = PointNeXtEncoder(
            in_channels=cfg.model.get("in_channels", 3),
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
            
            if self.use_bc_only:
                # 纯 BC 模式：所有样本强制设为 True，跳过过滤
                keep_mask = torch.ones(actions.shape[0], dtype=torch.bool, device=actions.device)
            else:
                # IDQL 模式：计算优势权重并进行拒绝采样 (Reject Sampling)
                adv_stable = adv - adv.max() 
                weights = torch.exp(self.beta * adv_stable)
                accept_prob = (weights / weights.max()).squeeze(-1)
                
                random_u = torch.rand_like(accept_prob)
                keep_mask = random_u < accept_prob
                # 安全校验：防止全部被拒绝导致 Loss 为 NaN
                if keep_mask.sum() == 0:
                    keep_mask[torch.argmax(accept_prob)] = True

        # 使用 mask 过滤字典中的张量
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

