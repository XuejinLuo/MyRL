"""Flow ODE mean + conditional Gaussian exploration for augmented-action PPO.

The scored action is (z, u[:exec_steps]); z ~ N(0,I) is parameter independent.
This is NOT the marginal likelihood of the original deterministic Flow policy.
Environment clipping is a deterministic map of the stored, unclipped u.
"""
import torch
from torch import nn
from torch.distributions import Normal


class FlowPPOPolicy(nn.Module):
    def __init__(self, policy, num_steps=10, exec_steps=2, std=0.02):
        super().__init__()
        if policy.algo_type != "flow":
            raise ValueError("This PPO wrapper supports Flow only")
        if num_steps < 1 or not 1 <= exec_steps <= policy.chunk_size or std <= 0:
            raise ValueError("Invalid steps or exploration standard deviation")
        self.policy = policy
        self.num_steps = int(num_steps)
        self.exec_steps = int(exec_steps)
        self.std = float(std)

    def mean(self, obs, z):
        # Encoder is frozen and stays in eval mode. ODE backbone retains gradients.
        with torch.no_grad():
            cond = self.policy._get_condition(obs['pc'], obs['state'])
        x = z.detach()
        dt = 1.0 / self.num_steps
        for i in range(self.num_steps):
            t = torch.full((x.shape[0],), i * dt, device=x.device, dtype=x.dtype)
            x = x + dt * self.policy.backbone(x, t, cond)
        return x

    def distribution(self, mean):
        return Normal(mean[:, :self.exec_steps], self.std)

    @torch.no_grad()
    def sample_with_log_prob(self, obs):
        z = torch.randn(obs['pc'].shape[0], self.policy.chunk_size,
                        self.policy.action_dim, device=obs['pc'].device)
        mean = self.mean(obs, z)
        dist = self.distribution(mean)
        action = mean.clone()
        action[:, :self.exec_steps] = dist.sample()
        logp = dist.log_prob(action[:, :self.exec_steps]).sum((-1, -2))
        return action, z, logp, mean

    def evaluate_actions(self, obs, actions, z):
        mean = self.mean(obs, z)
        dist = self.distribution(mean)
        logp = dist.log_prob(actions[:, :self.exec_steps].detach()).sum((-1, -2))
        entropy = dist.entropy().sum((-1, -2))
        return logp, entropy, mean
