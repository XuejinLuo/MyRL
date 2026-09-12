"""Online regression adapter; offline sampling/checkpoint format stays compatible."""
import copy
import torch
from torch import nn


class FlowRegressionPolicy(nn.Module):
    def __init__(self, policy):
        super().__init__()
        if policy.algo_type != 'flow':
            raise ValueError('This online regression implementation requires model.algo_type=flow')
        self.policy = policy
        self.policy.encoder.requires_grad_(False)
        self.reference_backbone = copy.deepcopy(policy.backbone).requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def encode(self, pc, state):
        self.policy.encoder.eval()
        return self.policy._get_condition(pc, state).detach()

    @torch.no_grad()
    def sample(self, features, num_steps=10):
        self.eval()
        return self.policy.scheduler.sample(
            self.policy.backbone, features,
            (self.policy.chunk_size, self.policy.action_dim), num_steps=num_steps, solver='euler')

    def regression_losses(self, features, actions, lengths):
        """Regress only executed prefix; preserve teacher field on whole chunk.

        Same condition, interpolation point and time for student and teacher.
        Fresh noise/time every update. This penalty is not a KL constraint.
        """
        self.eval()
        actions, features = actions.detach(), features.detach()
        if torch.any(lengths < 1) or torch.any(lengths > actions.shape[1]):
            raise ValueError('Invalid executed chunk lengths')
        noise = torch.randn_like(actions)
        t = torch.rand(actions.shape[0], device=actions.device) * (1 - 2e-5) + 1e-5
        sigma = self.policy.scheduler.sigma_min
        xt = (1 - (1 - sigma) * t[:, None, None]) * noise + t[:, None, None] * actions
        target = actions - (1 - sigma) * noise
        pred = self.policy.backbone(xt, t, features)
        with torch.no_grad():
            reference = self.reference_backbone(xt, t, features)
        mask = torch.arange(actions.shape[1], device=actions.device)[None, :] < lengths[:, None]
        per_step_mse = (pred - target).square().mean(-1)
        mse = (per_step_mse * mask).sum(-1) / lengths
        anchor = (pred - reference).square().mean(dim=(-1, -2))
        return mse, anchor
