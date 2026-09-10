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

    def update_step(self, states, actions, old_log_probs, returns, advantages,
                    z, old_mean, ref_mean, actor_enabled=True):
        # eval() does not disable autograd; fixes BN/Dropout mode across rollout/update.
        self.policy.eval()
        self.critic.eval()
        if actor_enabled:
            logp, entropy, mean = self.policy.evaluate_actions(states, actions, z)
        else:
            with torch.no_grad():
                logp, entropy, mean = self.policy.evaluate_actions(states, actions, z)
        k, std = self.policy.exec_steps, self.policy.std
        # Exact CONDITIONAL Gaussian KL at stored s,z; not marginal Flow KL.
        kl = ((mean[:, :k] - old_mean[:, :k]).square() / (2 * std**2)).sum((-1, -2)).mean()
        log_ratio = logp - old_log_probs.detach()
        finite = torch.isfinite(log_ratio).all() & torch.isfinite(kl)
        # Stop rather than silently clamp broken ratios. Guard before exp().
        stop = (not bool(finite)) or bool(kl > self.target_kl) or bool(log_ratio.abs().max() > 20)
        actor_loss = torch.zeros((), device=returns.device)
        anchor_loss = torch.zeros_like(actor_loss)
        clipfrac = torch.zeros_like(actor_loss)
        self.optimizer_policy.zero_grad(set_to_none=True)
        if actor_enabled and not stop:
            ratio = log_ratio.exp()
            surrogate = torch.minimum(ratio * advantages,
                ratio.clamp(1-self.clip_ratio, 1+self.clip_ratio) * advantages)
            actor_loss = -surrogate.mean()  # keep BOTH signs of advantage
            # Fixed offline reference, same z; full horizon preserves unused suffix too.
            anchor_loss = F.mse_loss(mean, ref_mean.detach())
            loss = actor_loss + self.anchor_coef * anchor_loss
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm,
                                     error_if_nonfinite=True)
            self.optimizer_policy.step()
            clipfrac = ((ratio-1).abs() > self.clip_ratio).float().mean()
        values = self.critic(states).squeeze(-1)
        critic_loss = F.mse_loss(values, returns.detach())
        if not torch.isfinite(critic_loss):
            raise FloatingPointError('Nonfinite critic loss')
        self.optimizer_critic.zero_grad(set_to_none=True)
        (self.v_loss_coef * critic_loss).backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm,
                                 error_if_nonfinite=True)
        self.optimizer_critic.step()
        return dict(actor_loss=actor_loss.item(), critic_loss=critic_loss.item(),
                    entropy=entropy.mean().item(), anchor_loss=anchor_loss.item(),
                    conditional_kl=kl.item(), clipfrac=clipfrac.item(),
                    stop_actor=float(stop and actor_enabled),
                    total_loss=(actor_loss + self.anchor_coef * anchor_loss
                                + self.v_loss_coef * critic_loss).item())
