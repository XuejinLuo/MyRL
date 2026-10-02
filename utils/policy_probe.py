"""Bounded before/after PPO probes. No RNG consumption or optimizer mutation.

Conditional KL is evaluated at stored latent states, not a final-action KL.
Action drift propagates identical recovered Gaussian innovations through the
whole updated chain. Reported controls are pre-environment-clipping outputs,
not robot displacement. Gradient norms are a fixed small-sample diagnostic.
"""
import math
import torch
from algos.pg import clipped_objective, normalize_advantages


class PolicyUpdateProbe:
    def __init__(self, trainer, batch, advantages, exec_steps, samples=32):
        self.trainer, self.exec_steps = trainer, exec_steps
        n = min(samples, len(advantages))
        # Cover the batch (time-major, lanes interleaved), not only its first lane.
        idx = torch.linspace(0, len(advantages) - 1, n, device=advantages.device).long()
        self.features = batch['features'][idx].detach()
        self.chain = batch['chains'][idx].detach()
        self.old_logprobs = batch['logprobs'][idx].detach()
        adv = normalize_advantages(advantages)[idx]
        self.means, self.stds, self.noises = [], [], []
        self.metrics = {'Probe/Samples': n}
        actor = trainer.policy
        for step in range(actor.sampler.num_steps):
            mean, std = actor.sampler.transition(actor.policy.backbone, self.features,
                                                 self.chain[:, step], step)
            logp = trainer.event_logprob(actor.sampler.log_prob(self.chain[:, step + 1], mean, std), step)
            old = trainer.event_logprob(self.old_logprobs[:, step], step)
            loss, _, _, _ = clipped_objective(logp, old, adv, trainer.clip_ratio)
            grads = torch.autograd.grad(loss / actor.sampler.num_steps,
                                        trainer.actor_params, allow_unused=True)
            norm2 = sum(g.detach().double().square().sum().item() for g in grads if g is not None)
            self.metrics[f'Probe/Gradient_Norm_Step_{step}'] = math.sqrt(norm2)
            self.means.append(mean.detach())
            self.stds.append(std.detach())
            self.noises.append((self.chain[:, step + 1] - mean.detach()) / std.detach())
        # Reconstruct both sides the same way to avoid counting FP32 roundoff
        # from recovering innovations as a policy change.
        self.before = self._replay()

    @torch.no_grad()
    def _replay(self):
        actor = self.trainer.policy
        x = self.chain[:, 0]
        for step, noise in enumerate(self.noises):
            mean, std = actor.sampler.transition(actor.policy.backbone, self.features, x, step)
            x = mean + std * noise
        return x

    @torch.no_grad()
    def finish(self, normalizer):
        actor = self.trainer.policy
        for step, old_mean in enumerate(self.means):
            mean, std = actor.sampler.transition(actor.policy.backbone, self.features,
                                                 self.chain[:, step], step)
            old_std = self.stds[step].double()
            mean, std, old_mean = mean.double(), std.double(), old_mean.double()
            kl = ((std / old_std).log() + old_std.square() / (2 * std.square()) - .5
                  + (old_mean - mean).square() / (2 * std.square())).clamp_min(0)
            self.metrics[f'Probe/Conditional_KL_Step_{step}'] = self.trainer.event_logprob(kl, step).mean().item()
        after = self._replay()
        delta = after - self.before
        for label, values in [('Executed', delta[:, :self.exec_steps]),
                              ('Unused', delta[:, self.exec_steps:])]:
            if values.numel():
                self.metrics[f'Probe/{label}_Normalized_RMS'] = values.square().mean().sqrt().item()
        control_delta = (normalizer.unnormalize(after, 'action') -
                         normalizer.unnormalize(self.before, 'action'))[:, :self.exec_steps]
        # Keep channels separate: translation, rotation and gripper use different units.
        for j in range(control_delta.shape[-1]):
            self.metrics[f'Probe/Control_RMS_Dim_{j}'] = control_delta[..., j].square().mean().sqrt().item()
        if not all(math.isfinite(v) for v in self.metrics.values()):
            raise FloatingPointError('Nonfinite policy probe result')
        return self.metrics
