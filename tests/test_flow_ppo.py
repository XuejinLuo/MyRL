import copy
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
import gymnasium as gym
from omegaconf import OmegaConf
from algos.diffusion_utils.flow_matching import OTFlowMatching
from algos.diffusion_utils.flow_cps import FlowCPSScheduler
from models.online_policy import FlowPPOPolicy
from algos.pg import FlowPPO, compute_gae, clipped_objective, sum_event_logprob
from envs.chunk_wrapper import ChunkActionWrapper

torch.set_num_threads(1)


class Field(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(0.15))
    def forward(self, x, t, cond):
        return self.w * x + cond[:, :1, None] * 0.1


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.algo_type, self.chunk_size, self.action_dim = 'flow', 3, 1
        self.encoder = nn.Linear(1, 2)
        self.backbone = Field()
        self.scheduler = OTFlowMatching()
    def _get_condition(self, pc, state):
        return self.encoder(state)


def actor():
    return FlowPPOPolicy(TinyPolicy(), num_steps=3)


def test_gaussian_replay_and_final_step_variance():
    torch.manual_seed(10)
    pi = actor()
    features = torch.randn(4096, 2)
    actions, chain, old = pi.collect(features)
    for step in range(3):
        now = pi.evaluate_transition(features, chain, step)
        torch.testing.assert_close(now, old[:, step])
        mean, std = pi.sampler.transition(pi.policy.backbone, features, chain[:, step], step)
        z = (chain[:, step + 1] - mean) / std
        assert abs(z.mean().item()) < 0.05
        assert abs(z.std().item() - 1) < 0.05
        analytic = torch.distributions.Normal(mean, std).log_prob(chain[:, step + 1])
        torch.testing.assert_close(now, analytic)
    assert torch.equal(actions, chain[:, -1])


def test_noise_free_mean_reduces_to_original_euler():
    sampler = FlowCPSScheduler(num_steps=4, noise_level=1e-8)
    x, cond = torch.randn(5, 3, 1), torch.randn(5, 2)
    field = Field()
    for step in range(4):
        mean, _ = sampler.transition(field, cond, x, step)
        expected = x + field(x, torch.full((5,), step/4), cond)/4
        torch.testing.assert_close(mean, expected)


def test_negative_and_positive_advantages_have_opposite_gradient():
    logp = torch.zeros(2, requires_grad=True)
    loss, _, _, _ = clipped_objective(logp, torch.zeros(2), torch.tensor([1., -1.]), .2)
    loss.backward()
    assert logp.grad[0] < 0  # gradient descent increases positive-advantage likelihood
    assert logp.grad[1] > 0  # decreases negative-advantage likelihood


def test_gae_time_limit_termination_and_rollout_boundary():
    adv, ret = compute_gae(torch.tensor([1., 2., 3.]), torch.zeros(3),
        torch.tensor([10., 20., 30.]), torch.tensor([0., 1., 0.]),
        torch.tensor([1., 1., 0.]), torch.tensor([2., 1., 3.]), gamma=.5, gae_lambda=.9)
    torch.testing.assert_close(ret, torch.tensor([3.5, 2., 6.75]))
    # Earlier decisions recurse only within their own episode.
    adv, _ = compute_gae(torch.ones(2), torch.zeros(2), torch.zeros(2),
        torch.zeros(2), torch.zeros(2), torch.tensor([2., 1.]), gamma=.5, gae_lambda=.8)
    torch.testing.assert_close(adv, torch.tensor([1.2, 1.]))


def test_ppo_changes_actor_and_freezes_encoder():
    pi = actor()
    features = torch.randn(16, 2)
    _, chain, old = pi.collect(features)
    critic = nn.Linear(2, 1)
    ppo = FlowPPO(pi, critic, actor_lr=1e-4, target_kl=None)
    batch = dict(features=features, chains=chain, logprobs=old,
                 values=critic(features).squeeze(-1).detach())
    before = pi.policy.backbone.w.detach().clone()
    encoder = copy.deepcopy(pi.policy.encoder.state_dict())
    metrics = ppo.update(batch, chain[:, -1].mean((1, 2)), torch.randn(16), batch_size=8, epochs=2)
    # PPO v2 accumulates generation-step gradients before one minibatch update.
    assert metrics['ppo/actor_updates'] == 4
    assert pi.policy.backbone.w != before
    assert all(p.grad is None for p in pi.policy.encoder.parameters())
    for k, v in encoder.items():
        torch.testing.assert_close(v, pi.policy.encoder.state_dict()[k])


def test_kl_guard_stops_actor():
    pi = actor(); features = torch.randn(8, 2)
    _, chain, old = pi.collect(features)
    critic = nn.Linear(2, 1)
    ppo = FlowPPO(pi, critic, target_kl=1e-4)
    batch = dict(features=features, chains=chain, logprobs=old - 2,
                 values=torch.zeros(8))
    metrics = ppo.update(batch, torch.randn(8), torch.randn(8), verify=False)
    assert metrics['ppo/early_stop'] == 1 and metrics['ppo/actor_updates'] == 0


class ToyEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            'point_cloud': gym.spaces.Box(-1, 1, (4, 3), dtype=np.float32),
            'state': gym.spaces.Box(-1, 1, (1,), dtype=np.float32)})
        self.closed = False
    def obs(self):
        return {'point_cloud': np.zeros((4, 3), np.float32),
                'state': np.array([self.t / 5], np.float32)}
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed); self.t=0
        return self.obs(), {}
    def step(self, action):
        self.t += 1
        return self.obs(), float(1 - action[0]**2), False, self.t >= 3, {'success': self.t >= 3}
    def close(self):
        self.closed = True


def test_chunk_records_primitive_rewards_and_early_stop():
    env = ChunkActionWrapper(ToyEnv(), chunk_size=3, exec_steps=2)
    env.reset()
    _, r, _, _, info = env.step(np.array([[5.], [0.], [.2]], np.float32))
    np.testing.assert_allclose(info['primitive_rewards'], [0., 1.])
    assert r == 1 and info['actual_steps'] == 2
    _, _, _, truncated, info = env.step(np.zeros((3, 1), np.float32))
    assert truncated and info['actual_steps'] == 1 and env.global_step == 3


def test_training_checkpoint_eval_resume(tmp_path, monkeypatch):
    from workflows import online as train_online
    cfg = OmegaConf.create({
        'model': dict(algo_type='flow', chunk_size=3, action_dim=1, state_dim=1,
                      cond_dim=2, in_channels=3, num_inference_steps=3),
        'env': dict(exec_steps=2, num_points=4, workspace_bounds=[[-1,-1,-1],[1,1,1]]),
        'algo': dict(steps_per_epoch=4, update_epochs=1, critic_warmup_epochs=0,
            actor_lr=3e-6, critic_lr=3e-4, gamma=.99, gae_lambda=.95, clip_ratio=.2,
            value_clip=.2, max_grad_norm=.5, target_kl=.02, ratio_scope='full',
            noise_level=.7, min_std=.0067, reward_scale=1., verify_logprobs=True,
            pretrained_ckpt=str(tmp_path/'offline.pth'), stats_path=None),
        'eval': dict(every=1, sampler='cps', seeds=[1,2]),
        'epochs':1, 'batch_size':2, 'save_epoch':1, 'seed':1, 'device':'cpu',
        'resume':None, 'eval_only':False, 'save_dir':str(tmp_path/'runs'),
        'run_name':'test', 'quiet':True, 'wandb':dict(enable=False)})
    torch.save(TinyPolicy().state_dict(), cfg.algo.pretrained_ckpt)
    (tmp_path/'dataset_stats.json').write_text(json.dumps({k: {'min':[-1.], 'max':[1.]} for k in ('state','action')}))
    monkeypatch.setattr(train_online, 'build_base', lambda cfg, device: TinyPolicy().to(device))
    factory = lambda: ChunkActionWrapper(ToyEnv(), chunk_size=3, exec_steps=2)
    result = train_online.run(cfg, factory)
    from pathlib import Path
    ckpt = Path(result['run_dir'])/'checkpoints'/'epoch_0001.pth'
    saved = torch.load(ckpt, weights_only=True)
    assert saved['format']=='myrl_flow_ppo_v1' and saved['total_env_steps']==6
    assert 'actor_optimizer' in saved and 'normalizer' in saved
    cfg.resume=str(ckpt);cfg.epochs=2
    resumed=train_online.run(cfg, factory)
    assert resumed['total_env_steps']==12
    cfg.eval_only=True
    assert set(train_online.run(cfg, factory))=={'cps','ode'}


def test_real_backbone_encoder_forward_and_ppo_step():
    from models.policy import EmbodiedGenPolicy
    torch.manual_seed(2)
    base = EmbodiedGenPolicy(action_dim=2, chunk_size=3, state_dim=2, in_channels=3, cond_dim=8)
    pi = FlowPPOPolicy(base, num_steps=2)
    features = pi.encode(torch.randn(2, 256, 3), torch.randn(2, 2))
    _, chain, old = pi.collect(features)
    critic = nn.Linear(8, 1)
    ppo = FlowPPO(pi, critic, target_kl=None)
    batch = dict(features=features, chains=chain, logprobs=old, values=torch.zeros(2))
    result = ppo.update(batch, torch.tensor([-1., 1.]), torch.tensor([1., 2.]), batch_size=2, epochs=1)
    assert result['ppo/actor_updates'] == 1


def test_offline_ema_and_online_weight_selection():
    from models.checkpoint import policy_weights
    raw = {'w': torch.tensor(1.)}
    ema = {'w': torch.tensor(2.)}
    cp = {'model_state_dict': raw, 'ema_model_state_dict': ema}
    assert policy_weights(cp) is ema
    assert policy_weights(cp, 'model_state_dict') is raw
    assert policy_weights({'model_state_dict': raw}) is raw
    assert policy_weights(raw) is raw
