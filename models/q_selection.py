"""Inference-only best-of-K extraction from a frozen Flow policy and twin Q."""
import torch
from torch import nn

from models.action_mapping import ActionDiagnostics, ATOL, RTOL, map_actions


class QSelectionPolicy(nn.Module):
    def __init__(self, actor, q, normalizer, candidates, action_mode='legacy', exec_steps=None):
        super().__init__()
        if isinstance(candidates, bool) or not isinstance(candidates, int) or candidates < 1:
            raise ValueError('candidates must be a positive integer')
        if action_mode not in ('legacy', 'consistent'):
            raise ValueError(f'Unknown action mode: {action_mode}')
        self.actor, self.q = actor, q
        self.normalizer, self.candidates = normalizer, candidates
        self.action_mode = action_mode
        self.prefix = exec_steps if exec_steps is not None else getattr(q, 'prefix', None)
        if self.prefix is not None and (isinstance(self.prefix, bool) or
                                       not isinstance(self.prefix, int) or self.prefix < 1):
            raise ValueError('exec_steps must be a positive integer')
        if candidates > 1 and self.prefix != q.prefix:
            raise ValueError('Q scoring prefix must equal exec_steps')
        self.eval_mode = actor.eval_mode
        self.register_buffer('action_low', None)
        self.register_buffer('action_high', None)
        self._totals = {name: ActionDiagnostics() for name in ('candidates', 'selected')}
        self.reset_episode_diagnostics()
        self.eval()

    def reset_episode_diagnostics(self):
        self._episode = {name: ActionDiagnostics() for name in ('candidates', 'selected')}
        self._pending = None

    def action_diagnostics(self, episode=False):
        if self.prefix is None:
            return {}
        groups = self._episode if episode else self._totals
        return {key: value for name, stats in groups.items() for key, value in stats.report(name).items()}

    def set_action_bounds(self, low, high):
        # Bounds of the *physical* chunk action space, before primitive execution.
        device = next(self.actor.parameters()).device
        self.action_low = torch.as_tensor(low, device=device)
        self.action_high = torch.as_tensor(high, device=device)

    def record_execution(self, info):
        """Runner hook: trim even the last chunk to actual_steps (early termination)."""
        if self._pending is None:
            return
        values, selected, score_physical = self._pending
        steps = int(info['actual_steps'])
        if not 1 <= steps <= self.prefix:
            raise ValueError('Invalid actual_steps for action diagnostics')
        # The evaluator runs one environment; sample() itself also supports batches.
        if len(selected) != 1:
            raise ValueError('Execution diagnostics require one environment')
        observed = torch.as_tensor(info['executed_actions'], device=score_physical.device,
                                   dtype=score_physical.dtype)
        expected = score_physical[selected[0], :steps]
        if observed.shape != expected.shape or not torch.isfinite(observed).all():
            raise ValueError('Invalid executed_actions for action diagnostics')
        if self.action_mode == 'consistent' and not torch.allclose(observed, expected, atol=ATOL, rtol=RTOL):
            raise RuntimeError('Q representation does not match observed physical execution')
        for scope in ('candidates', 'selected'):
            prefix_values = {key: value[:, :steps] for key, value in values.items()}
            if scope == 'selected':
                prefix_values = {key: value[selected] for key, value in prefix_values.items()}
                prefix_values['execution_to_score'] = (observed - expected).abs()[None]
            for groups in (self._totals, self._episode):
                groups[scope].update(prefix_values)
        self._pending = None

    @torch.no_grad()
    def sample(self, obs, num_steps=None):
        self.eval()
        if 'pc' in obs and 'object_features' not in obs:
            features = self.actor.encode(obs['pc'], obs['state'])
        else:
            features = self.actor.encode(obs)
        batch, k = len(features), self.candidates
        # Do not repeat or add random draws for K=1; keep the historical RNG path.
        actions = self.actor.sample(features if k == 1 else features.repeat_interleave(k, dim=0),
                                    num_steps=num_steps)
        if self.action_low is None:
            if k == 1 and self.action_mode == 'legacy':
                return actions
            raise RuntimeError('Set environment action bounds before Q selection')
        returned, scored, physical, diagnostics = map_actions(
            actions, self.normalizer, self.action_low, self.action_high, self.action_mode)
        if self.prefix is not None and self.prefix > actions.shape[1]:
            raise ValueError('Execution prefix exceeds action chunk')
        if k == 1:
            selected = torch.arange(batch, device=actions.device)
        else:
            # Q owns its observation encoder. Encode once, not K times per cloud.
            q_features = self.q.encoder(obs).repeat_interleave(k, dim=0)
            q1, q2 = self.q.q_net(q_features, scored[:, :self.prefix].contiguous())
            scores = torch.minimum(q1, q2).reshape(batch, k)
            if not torch.isfinite(scores).all():
                raise FloatingPointError('Nonfinite candidate Q score')
            selected = torch.arange(batch, device=actions.device) * k + scores.argmax(dim=1)
        if self.prefix is not None:
            self._pending = (diagnostics, selected, physical)
        return returned[selected]
