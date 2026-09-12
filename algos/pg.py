# algos/pg.py
"""Conservative advantage-weighted regression, not PPO.

Gamma/lambda are per chunk decision. Flow MSE is not a log likelihood or KL.
The historical class name is retained for existing imports.
"""
import torch
from torch import nn
import torch.nn.functional as F


class FlowPolicyGradient:
    def __init__(self, policy, critic, actor_lr=3e-7, critic_lr=3e-4,
                 gamma=0.99, gae_lambda=0.95, v_loss_coef=0.5,
                 max_grad_norm=0.5, device='cpu', anchor_coef=1.0,
                 adv_temperature=1.0, max_weight=5.0):
        if adv_temperature <= 0 or max_weight < 1 or anchor_coef < 0:
            raise ValueError('Invalid regression weights')
        self.policy, self.critic = policy.to(device), critic.to(device)
        self.actor_params = [p for p in policy.parameters() if p.requires_grad]
        self.optimizer_policy = torch.optim.AdamW(self.actor_params, lr=actor_lr, weight_decay=0)
        self.optimizer_critic = torch.optim.AdamW(critic.parameters(), lr=critic_lr, weight_decay=0)
        self.gamma, self.gae_lambda = gamma, gae_lambda
        self.v_loss_coef, self.max_grad_norm = v_loss_coef, max_grad_norm
        self.anchor_coef = anchor_coef
        self.adv_temperature, self.max_weight = adv_temperature, max_weight

    @torch.no_grad()
    def compute_gae(self, rewards, values, dones, next_values, terminated):
        """next_values[t] belongs to the observation BEFORE any episode reset.

        Termination disables bootstrap; time limits bootstrap but stop the GAE
        recursion. The final rollout boundary bootstraps its next observation.
        """
        if not all(x.shape == rewards.shape for x in (values, dones, next_values, terminated)):
            raise ValueError('All GAE tensors must have matching [T, ...] shapes')
        advantages = torch.zeros_like(rewards)
        carry = torch.zeros_like(rewards[0])
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] * (1 - terminated[t]) - values[t]
            carry = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * carry
            advantages[t] = carry
        return advantages, advantages + values

    @torch.no_grad()
    def advantage_weights(self, advantages):
        """Raw sign gates samples; scale without subtracting the rollout mean.

        Computed once per rollout, with a capped exponent. No forced sample is
        selected when all advantages are nonpositive.
        """
        if not torch.isfinite(advantages).all():
            raise ValueError('Nonfinite advantages')
        scale = advantages.std(unbiased=False).clamp_min(1e-6)
        exponent = (advantages / (scale * self.adv_temperature)).clamp(
            min=0, max=float(torch.log(torch.tensor(self.max_weight))))
        return torch.where(advantages > 0, exponent.exp(), torch.zeros_like(advantages))

    def update_critic(self, features, returns):
        self.critic.eval()  # eval mode permits gradients, avoids mode-dependent targets
        values = self.critic(features).squeeze(-1)
        loss = F.mse_loss(values, returns)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite critic loss')
        self.optimizer_critic.zero_grad(set_to_none=True)
        (self.v_loss_coef * loss).backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm, error_if_nonfinite=True)
        self.optimizer_critic.step()
        return {'loss/critic': loss.item()}

    def update_actor(self, features, actions, weights, lengths):
        self.policy.eval()  # no dropout or BatchNorm drift during finetuning
        self.optimizer_policy.zero_grad(set_to_none=True)
        if weights.sum() <= 0:
            return {'loss/actor': 0., 'awr/flow_mse': 0., 'awr/anchor_mse': 0.,
                    'awr/actor_updates': 0.}
        mse, anchor = self.policy.regression_losses(features, actions, lengths)
        regression = (weights * mse).sum() / weights.sum()
        anchor_loss = anchor.mean()
        loss = regression + self.anchor_coef * anchor_loss
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite actor loss')
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor_params, self.max_grad_norm, error_if_nonfinite=True)
        self.optimizer_policy.step()
        return {'loss/actor': loss.item(), 'awr/flow_mse': regression.item(),
                'awr/anchor_mse': anchor_loss.item(), 'awr/actor_updates': 1.}
