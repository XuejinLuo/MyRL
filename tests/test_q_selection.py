"""Critic-only training and policy extraction with real CPU optimizers."""
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from data.episodes import SCHEMA, digest, save_episode, write_json
from envs.chunk_wrapper import ChunkActionWrapper
from evaluation.runner import evaluate_policy, seed_all
from models.factory import observation_encoder, observation_tensorizer
from models.online_policy import FlowPPOPolicy
from models.q_selection import QSelectionPolicy
from tests.test_flow_ppo import TinyPolicy, ToyEnv
from tests.test_stages import config, primitive_episode, CriticFeatures
from utils.normalizer import MinMaxNormalizer
from workflows.offline_round import PrefixQ


class BufferedActor(TinyPolicy):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(1, 2), nn.BatchNorm1d(2))


def norm():
    normalizer = MinMaxNormalizer()
    normalizer.stats = {k: {'min': [-1.], 'max': [1.]} for k in ('action', 'state')}
    return normalizer


@pytest.mark.parametrize('sampler', ['cps', 'ode'])
def test_one_candidate_preserves_original_actions_rng_and_episode_results(tmp_path, sampler):
    cfg = config('offline', tmp_path)
    actor = FlowPPOPolicy(BufferedActor(), num_steps=3, eval_mode=sampler)
    normalizer = norm()
    # Invalid critic on purpose: K=1 must not touch it.
    selected = QSelectionPolicy(actor, nn.Identity(), normalizer, 1)
    obs = {'pc': torch.zeros(1, 4, 3), 'state': torch.zeros(1, 1)}
    seed_all(20)
    expected = actor.sample(actor.encode(obs['pc'], obs['state']), 3)
    rng = torch.get_rng_state().clone()
    seed_all(20)
    assert torch.equal(selected.sample(obs, 3), expected)
    assert torch.equal(torch.get_rng_state(), rng)
    factory = lambda: ChunkActionWrapper(ToyEnv(), 3, 2)
    baseline = evaluate_policy(factory, actor, observation_encoder(cfg, actor, normalizer, 'cpu'),
                               normalizer, [4000, 4001], 3, output_dir=tmp_path/'baseline')
    actual = evaluate_policy(factory, selected, observation_tensorizer(cfg, normalizer, 'cpu'),
                             normalizer, [4000, 4001], 3, output_dir=tmp_path/'selected')
    assert baseline == actual
    assert (tmp_path/'baseline/episodes.csv').read_bytes() == (tmp_path/'selected/episodes.csv').read_bytes()


class CandidateActor(nn.Module):
    eval_mode = 'cps'
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(()))
    def encode(self, pc, state):
        return state
    def sample(self, features, num_steps=None):
        # Last two actions would change the ranking if mistakenly scored.
        return torch.tensor([[[9.], [-9.], [-9.]], [[.5], [9.], [9.]],
                             [[9.], [-9.], [-9.]], [[.5], [9.], [9.]]])[:len(features)]


class ScoringEncoder(nn.Module):
    def forward(self, obs):
        # Independent critic input must be raw observations, encoded once per state.
        assert 'pc' in obs and len(obs['state']) == 2
        return obs['state']


class RecordingTwinQ(nn.Module):
    def forward(self, features, actions):
        self.seen = actions.clone()
        assert actions.shape == (4, 1, 1)
        a = actions[:, 0]
        # Q1 alone selects saturated candidate; min(Q1,Q2) selects .5.
        return a, -a


def test_selection_uses_twin_min_prefix_physical_clipping_and_correct_batch():
    actor, normalizer = CandidateActor(), norm()
    q = PrefixQ(ScoringEncoder(), RecordingTwinQ(), 1)
    policy = QSelectionPolicy(actor, q, normalizer, 2)
    policy.set_action_bounds(np.full((3, 1), -.8, np.float32), np.full((3, 1), .8, np.float32))
    obs = {'pc': torch.zeros(2, 4, 3), 'state': torch.zeros(2, 1)}
    chosen = policy.sample(obs)
    assert chosen[:, 0, 0].tolist() == [.5, .5]
    assert chosen[:, 1, 0].tolist() == [9., 9.]  # retain full original chunk
    expected = normalizer.normalize(torch.tensor([[[.8]], [[.5000007]], [[.8]], [[.5000007]]]), 'action')
    torch.testing.assert_close(q.q_net.seen, expected)


@pytest.mark.parametrize('bad', [0, -1, True, 1.5])
def test_invalid_candidate_count(bad):
    with pytest.raises(ValueError, match='positive integer'):
        QSelectionPolicy(CandidateActor(), nn.Identity(), norm(), bad)


def test_nonfinite_q_is_rejected():
    class BadQ(RecordingTwinQ):
        def forward(self, features, actions):
            q1, q2 = super().forward(features, actions)
            return q1 * float('nan'), q2
    policy = QSelectionPolicy(CandidateActor(), PrefixQ(ScoringEncoder(), BadQ(), 1), norm(), 2)
    policy.set_action_bounds(np.full((3, 1), -1.), np.full((3, 1), 1.))
    with pytest.raises(FloatingPointError, match='Q score'):
        policy.sample({'pc': torch.zeros(2, 4, 3), 'state': torch.zeros(2, 1)})


def test_critic_training_checkpoint_reload_and_matched_comparison(tmp_path, monkeypatch):
    from workflows import critic, offline_round
    from evaluation import q_selection
    cfg = config('offline', tmp_path)
    # Reuse the tiny validated model/env specification with the new stage controls.
    cfg.stage = 'critic'
    cfg.output = str(tmp_path/'critic')
    cfg.epochs, cfg.save_epoch = 2, 1
    OmegaConf.update(cfg, 'initial_ckpt', str(tmp_path/'actor.pth'), force_add=True)
    OmegaConf.update(cfg, 'weight_key', 'auto', force_add=True)
    ep = primitive_episode()
    ep_path = tmp_path/'episode.npz'
    save_episode(ep_path, ep)
    OmegaConf.update(cfg, 'manifest', str(tmp_path/'manifest.json'), force_add=True)
    write_json(cfg.manifest, dict(schema=SCHEMA, reward_mode='success',
        env=OmegaConf.to_container(cfg.env, resolve=True), sources=[dict(name='rollouts',
            episodes=[dict(path=ep_path.name, sha256=digest(ep_path), seed=11000)])]))
    source_actor = BufferedActor()
    original = copy.deepcopy(source_actor.state_dict())
    source_config = OmegaConf.to_container(cfg, resolve=True)
    # Use a non-default source sampler to check protocol retention.
    source_config['noise_level'] = .4
    torch.save(dict(model_state_dict=original, config=source_config, normalizer=norm().stats), cfg.initial_ckpt)
    source_digest = digest(cfg.initial_ckpt)
    monkeypatch.setattr(critic, 'build_base', lambda cfg, device: BufferedActor().to(device))
    monkeypatch.setattr(q_selection, 'build_base', lambda cfg, device: BufferedActor().to(device))
    monkeypatch.setattr(offline_round, 'CriticFeatureExtractor', CriticFeatures)
    built = {}
    real_build = critic.build_agent
    def build(cfg, base, device):
        agent = real_build(cfg, base, device)
        built.update(agent=agent, q=copy.deepcopy(agent.q_net.state_dict()),
                     v=copy.deepcopy(agent.v_net.state_dict()))
        def forbidden(*args, **kwargs):
            raise AssertionError('Actor update must never run')
        agent.update_actor = forbidden
        return agent
    monkeypatch.setattr(critic, 'build_agent', build)
    # No simulator is available during critic training; any accidental import fails.
    monkeypatch.setitem(sys.modules, 'envs.factory', None)
    path = critic.run(cfg)
    cp = torch.load(path, weights_only=True)
    assert cp['format'] == critic.FORMAT and cp['actor_unchanged']
    assert cp['sampler']['noise_level'] == .4
    assert digest(cfg.initial_ckpt) == source_digest
    assert all(torch.equal(original[k], v) for k, v in cp['model_state_dict'].items())
    for name in ('q', 'v'):
        assert any(not torch.equal(v, cp[name+'_state_dict'][k]) for k, v in built[name].items())
    assert any('encoder' in k for k in cp['q_state_dict'])
    restored = offline_round.build_q(cfg)
    restored.load_state_dict(cp['q_state_dict'], strict=True)
    restored.eval()
    built['agent'].q_net.eval()
    obs = {'state': torch.zeros(2, 1)}
    actions = torch.zeros(2, 3, 1)
    for a, b in zip(restored(obs, actions), built['agent'].q_net(obs, actions)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    log = [json.loads(line) for line in (Path(cfg.output)/'metrics.jsonl').read_text().splitlines()]
    assert all(row['Train/Actor_Updates'] == 0 for row in log)
    with pytest.raises(FileExistsError):
        critic.run(cfg)
    settings = OmegaConf.create(dict(device='cpu', q_selection=dict(checkpoint=str(path),
        output=str(tmp_path/'comparison'), candidates=[1, 8], sampler='cps', seed_start=4000, episodes=2)))
    def factory(cfg):
        return ChunkActionWrapper(ToyEnv(), cfg.model.chunk_size, cfg.env.exec_steps)
    monkeypatch.setitem(sys.modules, 'envs.factory', SimpleNamespace(make_env=factory))
    results = q_selection.run(settings)
    assert [r['candidates'] for r in results] == [1, 8]
    summary = json.loads((tmp_path/'comparison/summary.json').read_text())
    assert summary['seeds'] == [4000, 4001]
    assert len(summary['paired']) == 1
    assert summary['protocol']['actor_sha256'] == source_digest
    for k in (1, 8):
        report = json.loads((tmp_path/f'comparison/candidates_{k}/summary.json').read_text())
        assert report['seeds'] == summary['seeds']
        assert report['metadata']['candidates'] == k
    settings.q_selection.output = str(tmp_path/'invalid')
    for seed in (2000, 10000, 11000):
        settings.q_selection.seed_start = seed
        with pytest.raises(ValueError, match='overlap'):
            q_selection.run(settings)
    assert not (tmp_path/'invalid').exists()


def test_new_entrypoint_configs_compose():
    with initialize_config_dir(version_base=None, config_dir=str(Path(__file__).resolve().parents[1]/'configs')):
        for name in ('train_critic', 'evaluate_q_selection'):
            cfg = compose(config_name=name, overrides=['+experiment=oc_budget'])
            assert cfg.env.env_id == 'StackCube-v1'
            assert cfg.env.observation.mode == 'global_object_budget'
            OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
