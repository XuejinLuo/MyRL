"""Inference-only best-of-K extraction from a frozen Flow policy and twin Q."""
import torch
from torch import nn


class QSelectionPolicy(nn.Module):
    def __init__(self, actor, q, normalizer, candidates):
        super().__init__()
        if isinstance(candidates, bool) or not isinstance(candidates, int) or candidates < 1:
            raise ValueError('candidates must be a positive integer')
        self.actor, self.q = actor, q
        self.normalizer, self.candidates = normalizer, candidates
        self.eval_mode = actor.eval_mode
        self.register_buffer('action_low', None)
        self.register_buffer('action_high', None)
        self.eval()

    def set_action_bounds(self, low, high):
        # Bounds of the *physical* chunk action space, before primitive execution.
        device = next(self.actor.parameters()).device
        self.action_low = torch.as_tensor(low, device=device)
        self.action_high = torch.as_tensor(high, device=device)

    @torch.no_grad()
    def sample(self, obs, num_steps=None):
        self.eval()
        if 'pc' in obs and 'object_features' not in obs:
            features = self.actor.encode(obs['pc'], obs['state'])
        else:
            features = self.actor.encode(obs)
        if self.candidates == 1:
            # Exactly the old sampling/RNG/clipping path; never call the critic.
            return self.actor.sample(features, num_steps=num_steps)
        if self.action_low is None:
            raise RuntimeError('Set environment action bounds before Q selection')
        batch, k = len(features), self.candidates
        actions = self.actor.sample(features.repeat_interleave(k, dim=0), num_steps=num_steps)
        if not torch.isfinite(actions).all():
            raise FloatingPointError('Nonfinite candidate action')
        # Mirror unnormalize + ChunkActionWrapper clipping, then use the same
        # normalized representation as critic training. Keep the original chunk
        # for execution: this must not introduce a different clipping policy.
        physical = self.normalizer.unnormalize(actions, 'action')
        physical = torch.maximum(torch.minimum(physical, self.action_high), self.action_low)
        scored_actions = self.normalizer.normalize(physical, 'action')
        # Q owns its observation encoder. Encode once, not K times per point cloud.
        q_features = self.q.encoder(obs).repeat_interleave(k, dim=0)
        q1, q2 = self.q.q_net(q_features, scored_actions[:, :self.q.prefix].contiguous())
        scores = torch.minimum(q1, q2).reshape(batch, k)
        if not torch.isfinite(scores).all():
            raise FloatingPointError('Nonfinite candidate Q score')
        selected = scores.argmax(dim=1)  # stable first-candidate tie break
        chunks = actions.reshape(batch, k, *actions.shape[1:])
        return chunks[torch.arange(batch, device=actions.device), selected]
