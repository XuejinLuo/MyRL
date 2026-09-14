"""CPS Gaussian transitions adapted to MyRL's noise->data OT-CFM convention.

Reference: RL-100 FlowMatchSchedulerExtended at 64264d952c1fda9d5096c090ddaa7177757a77ad.
This is a local implementation, not a drop-in upstream scheduler.
"""
import math
import torch


class FlowCPSScheduler:
    def __init__(self, num_steps=10, noise_level=0.7, min_std=0.0067,
                 sigma_min=1e-5):
        if num_steps < 1 or not 0 < noise_level <= 1 or min_std <= 0:
            raise ValueError('Require steps >= 1, 0 < noise_level <= 1, min_std > 0')
        if not 0 <= sigma_min < 1:
            raise ValueError('sigma_min must be in [0, 1)')
        self.num_steps = int(num_steps)
        self.noise_level, self.min_std, self.sigma_min = noise_level, min_std, sigma_min

    def transition(self, backbone, features, x, step):
        if not 0 <= step < self.num_steps:
            raise IndexError(step)
        t, tn = step / self.num_steps, (step + 1) / self.num_steps
        v = backbone(x, torch.full((len(x),), t, device=x.device, dtype=x.dtype), features)
        # Original path: x(t) = t * data + a(t) * noise, a=1-(1-sigma_min)t.
        # Solve its two linear equations using v=data-(1-sigma_min)*noise.
        a, an = 1 - (1 - self.sigma_min) * t, 1 - (1 - self.sigma_min) * tn
        data = (1 - self.sigma_min) * x + a * v
        noise = x - t * v
        raw_std = an * math.sin(self.noise_level * math.pi / 2)
        mean = tn * data + math.sqrt(max(an * an - raw_std * raw_std, 0.0)) * noise
        std = max(raw_std, self.min_std)
        return mean, torch.full_like(mean, std)

    @staticmethod
    def log_prob(next_x, mean, std):
        # FP32 elementwise Gaussian density of the STORED latent transition.
        return -0.5 * ((next_x.detach() - mean) / std).square() - std.log() - 0.5 * math.log(2 * math.pi)

    @torch.no_grad()
    def rollout(self, backbone, features, action_shape):
        x = torch.randn((len(features), *action_shape), device=features.device, dtype=features.dtype)
        chain, logprobs = [x], []
        for step in range(self.num_steps):
            mean, std = self.transition(backbone, features, x, step)
            # Includes final step: reporting a nonzero Gaussian std for a
            # deterministic final transition would invalidate its likelihood.
            x = mean + std * torch.randn_like(mean)
            logprobs.append(self.log_prob(x, mean, std))
            chain.append(x)
        return torch.stack(chain, 1), torch.stack(logprobs, 1)
