"""Run from project root: python -m unittest discover -s tests -p test_ppo_regressions.py -v

Requires PyTorch only; no ManiSkill, torch_cluster, checkpoint or GPU.
"""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


def load(relative):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FlowPPOPolicy = load('algos/flow_ppo_policy.py').FlowPPOPolicy
FlowPolicyGradient = load('algos/pg.py').FlowPolicyGradient
checks = load('utils/ppo_checks.py')


class Field(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor([0.1, -0.2]))

    def forward(self, x, t, cond):
        return 0.1 * x + self.bias + 0.01 * cond[:, None, :1]


class Base(nn.Module):
    algo_type = 'flow'
    chunk_size = 3
    action_dim = 2

    def __init__(self):
        super().__init__()
        self.encoder = nn.Identity()
        self.backbone = Field()

    def _get_condition(self, pc, state):
        return self.encoder(state)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, obs):
        return self.linear(obs['state'])


class PPORegressionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.actor = FlowPPOPolicy(Base(), num_steps=3, exec_steps=2, std=0.02)
        self.critic = Critic()
        self.trainer = FlowPolicyGradient(self.actor, self.critic, actor_lr=3e-7)
        self.obs = {'pc': torch.zeros(4, 5, 3), 'state': torch.ones(4, 1)}
        self.obs['cond'] = self.actor.encode_condition(self.obs)
        with torch.no_grad():
            action, z, lp, mean = self.actor.sample_with_log_prob(self.obs)
        self.data = dict(self.obs, actions=action, z=z, old_logp=lp, old_mean=mean)

    def update(self, enabled=True, old_mean=None):
        return self.trainer.update_step(
            self.obs, self.data['actions'], self.data['old_logp'],
            returns=torch.full((4,), 3.0), advantages=torch.ones(4),
            z=self.data['z'], old_mean=self.data['old_mean'] if old_mean is None else old_mean,
            ref_mean=self.data['old_mean'], actor_enabled=enabled)

    def test_initial_replay_and_immutable_behavior_data(self):
        snapshots = {k: v.clone() for k, v in self.data.items()}
        result = checks.replay_diagnostics(self.actor, self.data, 2, check=True)
        self.assertLess(result['logp_max_error'], 1e-3)
        self.assertLess(result['kl_max'], 1e-5)
        for key, value in snapshots.items():
            self.assertTrue(torch.equal(value, self.data[key]), key)

    def test_replay_rejects_corrupt_log_probability(self):
        bad = dict(self.data, old_logp=self.data['old_logp'] + 0.1)
        with self.assertRaisesRegex(RuntimeError, 'BEFORE any optimizer step'):
            checks.replay_diagnostics(self.actor, bad, 2, check=True)

    def test_cached_condition_does_not_reencode(self):
        with patch.object(self.actor, 'encode_condition', side_effect=AssertionError('reencoded')):
            self.actor.mean(self.obs, self.data['z'])

    def test_warmup_updates_only_critic_and_skips_actor_forward(self):
        before_actor = self.actor.policy.backbone.bias.detach().clone()
        before_critic = self.critic.linear.bias.detach().clone()
        with patch.object(self.actor, 'evaluate_actions', side_effect=AssertionError('actor forward')):
            result = self.update(enabled=False)
        self.assertEqual(result['actor_checked'], 0)
        self.assertEqual(result['actor_updated'], 0)
        self.assertTrue(torch.equal(before_actor, self.actor.policy.backbone.bias))
        self.assertFalse(torch.equal(before_critic, self.critic.linear.bias))

    def test_kl_stop_preserves_actor_but_updates_critic(self):
        before_actor = self.actor.policy.backbone.bias.detach().clone()
        before_critic = self.critic.linear.bias.detach().clone()
        result = self.update(old_mean=self.data['old_mean'] + 0.1)
        self.assertEqual(result['stop_kl'], 1)
        self.assertEqual(result['actor_updated'], 0)
        self.assertTrue(torch.equal(before_actor, self.actor.policy.backbone.bias))
        self.assertFalse(torch.equal(before_critic, self.critic.linear.bias))

    def test_valid_update_changes_actor_parameters(self):
        before = self.actor.policy.backbone.bias.detach().clone()
        result = self.update()
        self.assertEqual(result['actor_updated'], 1)
        self.assertGreater(result['actor_grad_norm'], 0)
        self.assertFalse(torch.equal(before, self.actor.policy.backbone.bias))

    def test_timeout_bootstrap_does_not_leak_next_episode(self):
        # First transition is a timeout with terminal V=10; second belongs
        # to an entirely different episode and must not affect the first.
        gamma = self.trainer.gamma
        rewards = torch.tensor([[1.0 + gamma * 10.0], [1000.0]])
        values = torch.tensor([[2.0], [500.0]])
        dones = torch.tensor([[1.0], [1.0]])
        adv, ret = self.trainer.compute_gae(rewards, values, dones, torch.tensor([9999.0]))
        self.assertAlmostEqual(adv[0].item(), 1.0 + gamma * 10.0 - 2.0, places=5)
        self.assertAlmostEqual(ret[0].item(), 1.0 + gamma * 10.0, places=5)
        self.assertAlmostEqual(ret[1].item(), 1000.0, places=5)


if __name__ == '__main__':
    unittest.main()
