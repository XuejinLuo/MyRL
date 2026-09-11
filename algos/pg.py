# algos/pg.py
"""PPO update for FlowPPOPolicy. gamma/lambda are per chunk decision."""
import torch
from torch import nn
import torch.nn.functional as F


class FlowPolicyGradient:
    def __init__(self, policy, critic, actor_lr=1e-6, critic_lr=3e-4,
                 gamma=0.99, gae_lambda=0.95, clip_ratio=0.1,
                 entropy_coef=0.0, v_loss_coef=0.5, max_grad_norm=0.5,
                 device='cpu', target_kl=0.02, anchor_coef=1.0):
        self.policy, self.critic = policy.to(device), critic.to(device)
        self.optimizer_policy = torch.optim.AdamW(
            [p for p in policy.parameters() if p.requires_grad],
            lr=actor_lr, weight_decay=0.0)
        self.optimizer_critic = torch.optim.AdamW(
            [p for p in critic.parameters() if p.requires_grad],
            lr=critic_lr, weight_decay=0.0)
        self.gamma, self.gae_lambda = gamma, gae_lambda
        self.clip_ratio, self.v_loss_coef = clip_ratio, v_loss_coef
        self.max_grad_norm, self.device = max_grad_norm, device
        self.target_kl, self.anchor_coef = target_kl, anchor_coef
        if entropy_coef != 0:
            raise ValueError('Fixed exploration std has constant entropy; use entropy_coef=0')

    @torch.no_grad()
    def compute_gae(self, rewards, values, dones, next_value):
        # For timeout transitions rollout has already added gamma * V(final_obs).
        # dones cuts the trace at BOTH termination and timeout; no reset leakage.
        adv = torch.zeros_like(rewards)
        carry = torch.zeros_like(next_value)
        for t in reversed(range(len(rewards))):
            nv = next_value if t == len(rewards)-1 else values[t+1]
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * nv * mask - values[t]
            carry = delta + self.gamma * self.gae_lambda * mask * carry
            adv[t] = carry
        return adv, adv + values

    def update_step(
        self, states, actions, old_log_probs, returns, advantages,
        z, old_mean, ref_mean, actor_enabled=True
    ):
        # eval 模式固定 BN/Dropout 行为，不影响梯度计算。
        self.policy.eval()
        self.critic.eval()

        old_log_probs = old_log_probs.detach()
        old_mean = old_mean.detach()
        ref_mean = ref_mean.detach()
        returns = returns.detach()
        advantages = advantages.detach()

        # When the actor is disabled, avoid evaluating the ODE altogether.
        # The caller aggregates KL only where actor_checked == 1.
        kl = torch.zeros((), device=returns.device)
        entropy = torch.zeros_like(kl)
        log_ratio_abs_max = 0.0
        stop_nonfinite = stop_kl = stop_ratio = False
        if actor_enabled:
            logp, entropy, mean = self.policy.evaluate_actions(states, actions, z)
            with torch.no_grad():
                kl = (
                    (mean[:, :self.policy.exec_steps] - old_mean[:, :self.policy.exec_steps]).square()
                    / (2.0 * self.policy.std**2)
                ).sum(dim=(-1, -2)).mean()
            log_ratio = logp - old_log_probs
            finite = bool(torch.isfinite(log_ratio).all() & torch.isfinite(kl))
            log_ratio_abs_max = log_ratio.detach().abs().max().item()
            stop_nonfinite = not finite
            stop_kl = finite and kl.item() > self.target_kl
            stop_ratio = finite and log_ratio_abs_max > 20.0
        stop_actor = stop_nonfinite or stop_kl or stop_ratio

        actor_loss = torch.zeros((), device=returns.device)
        anchor_loss = torch.zeros_like(actor_loss)
        clipfrac = torch.zeros_like(actor_loss)

        actor_updated = 0.0
        actor_grad_norm = 0.0

        self.optimizer_policy.zero_grad(set_to_none=True)

        if actor_enabled and not stop_actor:
            ratio = log_ratio.exp()

            surrogate = torch.minimum(
                ratio * advantages,
                ratio.clamp(
                    1.0 - self.clip_ratio,
                    1.0 + self.clip_ratio,
                ) * advantages,
            )
            actor_loss = -surrogate.mean()

            # 与固定离线参考策略保持接近，约束整个 action chunk。
            anchor_loss = F.mse_loss(mean, ref_mean)
            policy_loss = actor_loss + self.anchor_coef * anchor_loss

            if not bool(torch.isfinite(policy_loss)):
                raise FloatingPointError("Nonfinite actor loss")

            policy_loss.backward()

            grad_norm = nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                self.max_grad_norm,
                error_if_nonfinite=True,
            )
            # 返回的是裁剪之前的梯度范数。
            actor_grad_norm = grad_norm.item()

            self.optimizer_policy.step()
            actor_updated = 1.0

            with torch.no_grad():
                clipfrac = (
                    (ratio - 1.0).abs() > self.clip_ratio
                ).float().mean()

        # Actor 停止后，Critic 仍继续更新。
        values = self.critic(states).squeeze(-1)
        critic_loss = F.mse_loss(values, returns)

        if not bool(torch.isfinite(critic_loss)):
            raise FloatingPointError("Nonfinite critic loss")

        self.optimizer_critic.zero_grad(set_to_none=True)
        (self.v_loss_coef * critic_loss).backward()

        critic_grad_norm = nn.utils.clip_grad_norm_(
            self.critic.parameters(),
            self.max_grad_norm,
            error_if_nonfinite=True,
        ).item()

        self.optimizer_critic.step()

        return {
            # 保留已有字段，兼容当前 train_online.py。
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy.detach().mean().item(),
            "anchor_loss": anchor_loss.item(),
            "conditional_kl": kl.item(),
            "clipfrac": clipfrac.item(),
            "stop_actor": float(stop_actor),
            "total_loss": (
                actor_loss.item()
                + self.anchor_coef * anchor_loss.item()
                + self.v_loss_coef * critic_loss.item()
            ),

            # 新增诊断字段。
            "actor_checked": float(actor_enabled),
            "actor_updated": actor_updated,
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_norm": critic_grad_norm,
            "log_ratio_abs_max": log_ratio_abs_max,
            "stop_nonfinite": float(stop_nonfinite),
            "stop_kl": float(stop_kl),
            "stop_ratio": float(stop_ratio),
        }
