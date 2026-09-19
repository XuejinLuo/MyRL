"""On-policy Flow adapter with replayable Gaussian generation chains."""
import torch
from torch import nn
from algos.diffusion_utils.flow_cps import FlowCPSScheduler


class FlowPPOPolicy(nn.Module):
    def __init__(self, policy, num_steps=10, noise_level=0.7, min_std=0.0067,
                 eval_mode='cps'):
        super().__init__()
        if policy.algo_type != 'flow':
            raise ValueError('Flow PPO requires a Flow checkpoint; diffusion is not supported')
        if eval_mode not in ('cps', 'ode'):
            raise ValueError('eval_mode must be cps or ode')
        self.policy, self.eval_mode = policy, eval_mode
        self.policy.encoder.requires_grad_(False)
        self.sampler = FlowCPSScheduler(num_steps, noise_level, min_std, policy.scheduler.sigma_min)
        self.eval()

    @torch.no_grad()
    def encode(self, pc, state=None):
        self.eval()
        return self.policy._get_condition(pc, state).detach()

    @torch.no_grad()
    def collect(self, features):
        self.eval()
        chain, logprobs = self.sampler.rollout(self.policy.backbone, features,
            (self.policy.chunk_size, self.policy.action_dim))
        return chain[:, -1], chain, logprobs

    @torch.no_grad()
    def sample(self, features, num_steps=None):
        self.eval()
        if self.eval_mode == 'ode':
            return self.policy.scheduler.sample(self.policy.backbone, features,
                (self.policy.chunk_size, self.policy.action_dim),
                num_steps=num_steps or self.sampler.num_steps, solver='euler')
        if num_steps is not None and num_steps != self.sampler.num_steps:
            raise ValueError('CPS evaluation must use the trained number of generation steps')
        return self.collect(features)[0]

    def evaluate_transition(self, features, chain, step):
        self.eval()  # eval disables stochastic layers but permits gradients
        mean, std = self.sampler.transition(self.policy.backbone, features.detach(),
                                            chain[:, step].detach(), step)
        return self.sampler.log_prob(chain[:, step + 1], mean, std)
