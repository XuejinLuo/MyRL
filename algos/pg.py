"""RL-100-style per-generation-step clipped PPO for Flow policies.

Each ratio sums event dimensions (chunk coordinates), not denoising steps.
Default is the full latent chunk because attention couples all coordinates.
"""
import torch
from torch import nn
import torch.nn.functional as F


class PPORatioDiverged(FloatingPointError):
    """Finite policy drift exceeded the numerical guard."""


@torch.no_grad()
def compute_gae(rewards, values, next_values, terminated, dones, lengths,
                gamma=0.99, gae_lambda=0.95):
    if not all(x.shape == rewards.shape for x in (values, next_values, terminated, dones, lengths)):
        raise ValueError('GAE inputs must have identical shapes')
    if len(rewards) == 0 or torch.any(lengths < 1):
        raise ValueError('Empty rollout or invalid chunk lengths')
    adv = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        discount = gamma ** lengths[t]
        delta = rewards[t] + discount * (1 - terminated[t]) * next_values[t] - values[t]
        # gamma is per primitive step; lambda is per decision, as in RL-100.
        carry = delta + discount * gae_lambda * (1 - dones[t]) * carry
        adv[t] = carry
    return adv, adv + values


def sum_event_logprob(logprob, prefix_steps=None):
    if prefix_steps is not None:
        logprob = logprob[:, :prefix_steps]
    return logprob.flatten(1).sum(1)


def clipped_objective(new_logprob, old_logprob, advantages, clip_ratio):
    log_ratio = new_logprob - old_logprob
    if not torch.isfinite(log_ratio).all():
        raise FloatingPointError('Nonfinite PPO log ratio')
    # Stop the update on numerical overflow rather than silently change the loss.
    if log_ratio.abs().max() > 60:
        raise PPORatioDiverged('PPO ratio diverged; stop actor updates for this rollout')
    ratio = log_ratio.exp()
    loss = -torch.minimum(ratio * advantages,
                         ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * advantages).mean()
    kl = (ratio - 1 - log_ratio).mean()
    clipfrac = ((ratio - 1).abs() > clip_ratio).float().mean()
    return loss, kl, clipfrac, ratio


class FlowPPO:
    def __init__(self, policy, critic, actor_lr=3e-6, critic_lr=3e-4,
                 clip_ratio=0.2, value_clip=0.2, max_grad_norm=0.5,
                 target_kl=0.02, prefix_steps=None):
        self.policy, self.critic = policy, critic
        self.actor_params = list(policy.policy.backbone.parameters())
        self.actor_optimizer = torch.optim.Adam(self.actor_params, lr=actor_lr, eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr, eps=1e-5)
        self.clip_ratio, self.value_clip = clip_ratio, value_clip
        self.max_grad_norm, self.target_kl = max_grad_norm, target_kl
        self.prefix_steps = prefix_steps

    def update_critic(self, features, returns, old_values=None):
        self.critic.eval()
        values = self.critic(features).squeeze(-1)
        loss = (values - returns.detach()).square()
        if self.value_clip is not None and old_values is not None:
            clipped = old_values + (values - old_values).clamp(-self.value_clip, self.value_clip)
            loss = torch.maximum(loss, (clipped - returns.detach()).square())
        loss = 0.5 * loss.mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite value loss')
        self.critic_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm, error_if_nonfinite=True)
        self.critic_optimizer.step()
        return loss.item()

    @torch.no_grad()
    def verify_rollout(self, batch, batch_size=64):
        max_error = 0.0
        for idx in torch.arange(len(batch['features']), device=batch['features'].device).split(batch_size):
            for step in range(self.policy.sampler.num_steps):
                now = sum_event_logprob(self.policy.evaluate_transition(batch['features'][idx],
                    batch['chains'][idx], step), self.prefix_steps)
                old = sum_event_logprob(batch['logprobs'][idx, step], self.prefix_steps)
                error = (now - old).abs().max().item()
                if not torch.isfinite(now).all():
                    raise FloatingPointError('Nonfinite replay log probability')
                max_error = max(max_error, error)
        if max_error > 0.02:
            raise RuntimeError(f'Old/current logprob mismatch before update: {max_error}. Check sampler/mode/precision.')
        return max_error

    def update(self, batch, advantages, returns, batch_size=128, epochs=10,
               actor_enabled=True, verify=True):
        metrics = {'ppo/actor_updates': 0, 'ppo/early_stop': 0,
                   'ppo/ratio_guard_stop': 0,
                   'ppo/replay_logprob_error': self.verify_rollout(batch, batch_size) if verify else 0.}
        advantages = advantages.detach()
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-5)
        records = []
        value_losses = []
        stop_actor = not actor_enabled
        for _ in range(epochs):
            for idx in torch.randperm(len(returns), device=returns.device).split(batch_size):
                # Warmup fits V without clipping it to random initial predictions.
                old_values = batch['values'][idx] if actor_enabled else None
                value_losses.append(self.update_critic(batch['features'][idx], returns[idx], old_values))
                if stop_actor:
                    continue
                for step in range(self.policy.sampler.num_steps):
                    new = sum_event_logprob(self.policy.evaluate_transition(batch['features'][idx],
                        batch['chains'][idx], step), self.prefix_steps)
                    old = sum_event_logprob(batch['logprobs'][idx, step], self.prefix_steps).detach()
                    try:
                        loss, kl, cf, ratio = clipped_objective(new, old, advantages[idx], self.clip_ratio)
                    except PPORatioDiverged:
                        # Stop all remaining actor steps for this rollout.
                        # NaN/Inf errors remain fatal and are not swallowed.
                        self.actor_optimizer.zero_grad(set_to_none=True)
                        stop_actor = True
                        metrics['ppo/early_stop'] = 1
                        metrics['ppo/ratio_guard_stop'] = 1
                        break
                    records.append((loss.item(), kl.item(), cf.item(), ratio.mean().item()))
                    if self.target_kl is not None and kl.item() > 1.5 * self.target_kl:
                        stop_actor = True
                        metrics['ppo/early_stop'] = 1
                        break
                    self.actor_optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.actor_params, self.max_grad_norm, error_if_nonfinite=True)
                    self.actor_optimizer.step()
                    metrics['ppo/actor_updates'] += 1
        metrics['loss/critic'] = sum(value_losses) / len(value_losses)
        for j, key in enumerate(('loss/actor', 'ppo/approx_kl', 'ppo/clip_fraction', 'ppo/ratio_mean')):
            metrics[key] = sum(r[j] for r in records) / len(records) if records else 0.
        return metrics
