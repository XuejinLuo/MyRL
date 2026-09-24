"""Evaluation-only action mapping; the shared training normalizer is unchanged."""
import torch

ATOL, RTOL = 2e-6, 1e-5


def map_actions(raw, normalizer, low, high, mode):
    """Return execution representation, Q representation, and coordinate diagnostics.

    The effective physical interval is unnormalize([-1, 1]) intersect env bounds.
    In particular its upper endpoint is max + eps, not max. Constant dimensions
    therefore retain the existing normalizer's eps-wide invertible interval.
    """
    if mode not in ('legacy', 'consistent'):
        raise ValueError(f'Unknown action mode: {mode}')
    if not torch.isfinite(raw).all():
        raise FloatingPointError('Nonfinite candidate action')
    low, high = low.to(raw), high.to(raw)
    if not (torch.isfinite(low).all() and torch.isfinite(high).all()) or (low > high).any():
        raise ValueError('Require finite ordered environment action bounds')
    clip = lambda x: torch.maximum(torch.minimum(x, high), low)
    raw_physical = normalizer.unnormalize(raw, 'action')
    raw_execution = clip(raw_physical)
    if mode == 'consistent':
        if 'action' not in normalizer.stats:
            raise ValueError('Consistent mode requires action normalizer statistics')
        stats = normalizer.stats['action']
        minimum = torch.as_tensor(stats['min'], device=raw.device, dtype=raw.dtype)
        maximum = torch.as_tensor(stats['max'], device=raw.device, dtype=raw.dtype)
        if (not torch.isfinite(minimum).all() or not torch.isfinite(maximum).all()
                or (minimum > maximum).any() or not 0 < normalizer.eps < float('inf')):
            raise ValueError('Require finite ordered normalizer statistics and positive eps')
        norm_low = normalizer.unnormalize(-torch.ones_like(raw), 'action')
        norm_high = normalizer.unnormalize(torch.ones_like(raw), 'action')
        effective_low = torch.maximum(norm_low, low)
        effective_high = torch.minimum(norm_high, high)
        if (effective_low > effective_high).any():
            raise ValueError('Empty intersection of normalizer and environment action bounds')
        before_env = normalizer.unnormalize(raw.clamp(-1, 1), 'action')
        physical = torch.maximum(torch.minimum(before_env, effective_high), effective_low)
        scored = normalizer.normalize(physical, 'action')
        returned = scored  # Execute exactly the representation presented to Q.
    else:
        before_env, physical = raw_physical, raw_execution
        scored = normalizer.normalize(physical, 'action')
        returned = raw
    score_physical = normalizer.unnormalize(scored, 'action')
    execution = clip(normalizer.unnormalize(returned, 'action'))
    if not (torch.isfinite(scored).all() and torch.isfinite(score_physical).all()
            and torch.isfinite(execution).all()):
        raise FloatingPointError('Nonfinite mapped action')
    if mode == 'consistent' and not torch.allclose(score_physical, execution, atol=ATOL, rtol=RTOL):
        raise RuntimeError('Q representation does not match physical execution')
    diagnostics = dict(
        raw_outside=(raw.abs() > 1),
        env_clipped=(before_env < low) | (before_env > high),
        raw_env_clipped=(raw_physical < low) | (raw_physical > high),
        raw_to_score=(raw_execution - score_physical).abs(),
        execution_to_score=(execution - score_physical).abs(),
    )
    return returned, scored, score_physical, diagnostics


class ActionDiagnostics:
    """Coordinate-weighted counters; update only the actually executed prefix."""
    def __init__(self):
        self.decisions = self.chunks = self.coordinates = 0
        self.counts = dict.fromkeys(('raw_outside', 'env_clipped', 'raw_env_clipped'), 0)
        self.maxima = dict.fromkeys(('raw_to_score', 'execution_to_score'), 0.)

    def update(self, values):
        self.decisions += 1
        self.chunks += values['raw_outside'].shape[0]
        self.coordinates += values['raw_outside'].numel()
        for key in self.counts:
            self.counts[key] += int(values[key].sum().item())
        for key in self.maxima:
            self.maxima[key] = max(self.maxima[key], float(values[key].max().item()))

    def report(self, scope):
        prefix = f'Action/{scope}/'
        result = dict(decisions=self.decisions, chunks=self.chunks, coordinates=self.coordinates)
        for key, count in self.counts.items():
            result[key + '_coordinates'] = count
            result[key + '_fraction'] = count / self.coordinates if self.coordinates else 0.
        result.update({key + '_max_abs': value for key, value in self.maxima.items()})
        return {prefix + key: value for key, value in result.items()}
