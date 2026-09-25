"""Balanced replay, update-budget invariance and real CPU iterative training."""
import copy
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from data.dataset import TrajectoryDataset
from data.episodes import SCHEMA, digest, save_episode, write_json
from data.iterative_sampling import SourceDataset, FixedBatches, SamplingMetrics, validate_update_config
from tests.test_flow_ppo import TinyPolicy, ToyEnv
from tests.test_stages import config, CriticFeatures
from tests.test_q_selection import norm
from envs.chunk_wrapper import ChunkActionWrapper


def episode(length, success, marker=0.):
    return dict(pc=np.full((length+1, 4, 3), marker, np.float32),
        state=np.linspace(-.5, .5, length+1, dtype=np.float32)[:, None],
        action=np.full((length, 1), marker, np.float32),
        reward=np.array([0.]*(length-1)+[float(success)], np.float32),
        success=np.array([False]*(length-1)+[success]),
        terminated=np.array([False]*(length-1)+[success]),
        truncated=np.array([False]*(length-1)+[not success]))


def specification(episodes):
    return dict(sources=[dict(name='demonstrations', episodes=[{}]),
                         dict(name='round_000', episodes=[{} for _ in episodes[1:]])])


def fixed_config(tmp_path, mode='demo_success'):
    cfg = config('iterative', tmp_path)
    cfg.collect, cfg.rounds = False, 1
    cfg.batch_size = 4
    cfg.updates_per_round = 6
    cfg.critic_warmup_updates = 2
    cfg.eval_every_updates = 2
    cfg.save_every_updates = 3
    cfg.log_every_updates = 2
    cfg.actor_sampling.mode = mode
    cfg.ema_decay = .5
    return cfg


def test_exact_quotas_uniform_episode_then_time_and_independent_rng(tmp_path):
    cfg = fixed_config(tmp_path)
    eps = [episode(4, True), episode(2, True, .1), episode(100, True, .2), episode(300, False, .3)]
    data = SourceDataset(TrajectoryDataset(eps, cfg, norm()), specification(eps), ['demonstrations'])
    counts = np.zeros(len(eps), dtype=int)
    global_before = torch.get_rng_state().clone()
    sampler = FixedBatches(data, 8, 1000, 11, 'demo_success', .5)
    for indices in sampler:
        groups = [data.group_ids[data.dataset.indices[i][0]] for i in indices]
        assert groups.count(0) == groups.count(1) == 4
        for i in indices:
            counts[data.dataset.indices[i][0]] += 1
    assert counts[0] == 4000 and counts[3] == 0
    assert abs(counts[1]-counts[2]) < 250  # 2-step and 100-step episodes get equal mass.
    assert torch.equal(global_before, torch.get_rng_state())
    critic = list(FixedBatches(data, 4, 100, 31, 'mixed'))
    list(FixedBatches(data, 4, 20, 31, 'demo_success'))
    assert critic == list(FixedBatches(data, 4, 100, 31, 'mixed'))
    assert any(data.group_ids[data.dataset.indices[i][0]] == 2 for b in critic for i in b)


def test_source_alignment_after_rejection_and_actual_success_flags(tmp_path):
    cfg = fixed_config(tmp_path)
    cfg.dataset.max_rejected_fraction = 1.
    eps = [episode(3, True), episode(3, True, 2.), episode(3, False, .2), episode(4, True, .3)]
    spec = specification(eps)
    spec['sources'][1]['episodes'][1]['success'] = True  # Do not trust stale manifest annotations.
    data = SourceDataset(TrajectoryDataset(eps, cfg, norm()), spec, ['demonstrations'])
    assert data.dataset.original_episode_indices == [0, 2, 3]
    assert data.group_ids == [0, 2, 1]
    assert data.report['excluded_episode_indices'] == [1]
    assert data.report['groups']['rollout_failure']['transitions'] == 3
    assert data.report['sources']['source_001']['success_episodes'] == 1


def test_missing_groups_are_errors_not_silent_fallback(tmp_path):
    cfg = fixed_config(tmp_path)
    eps = [episode(3, True), episode(3, False)]
    data = SourceDataset(TrajectoryDataset(eps, cfg, norm()), specification(eps), ['demonstrations'])
    with pytest.raises(ValueError, match='successful rollout'):
        FixedBatches(data, 4, 2, 42, 'demo_success')
    assert len(list(FixedBatches(data, 4, 2, 42, 'mixed'))) == 2
    with pytest.raises(ValueError, match='missing'):
        SourceDataset(data.dataset, specification(eps), ['unknown'])


@pytest.mark.parametrize('key,value', [('updates_per_round', 0), ('updates_per_round', True),
    ('critic_warmup_updates', 6), ('eval_every_updates', 0), ('log_every_updates', -1),
    ('actor_sampling.demo_fraction', .3), ('actor_sampling.mode', 'unknown')])
def test_invalid_fixed_configuration(tmp_path, key, value):
    cfg = fixed_config(tmp_path)
    OmegaConf.update(cfg, key, value)
    with pytest.raises(ValueError):
        validate_update_config(cfg)


def test_source_metrics_account_for_rejection_and_sample_weighted_loss():
    stats = SamplingMetrics(2)
    batch = dict(sampling_source=torch.tensor([0, 1, 1, 1]), sampling_group=torch.tensor([0, 1, 1, 2]))
    details = dict(keep_mask=torch.tensor([True, True, False, True]),
                   advantages=torch.tensor([1., 2., -3., 4.]), losses=torch.tensor([2., 6., 10.]))
    stats.add('Actor', batch, details)
    stats.add('Critic', batch)
    result = stats.report()
    assert result['Actor/rollout_success/sampled'] == 2
    assert result['Actor/rollout_success/kept'] == 1
    assert result['Actor/rollout_success/keep_fraction'] == .5
    assert result['Actor/rollout_success/loss_mean'] == 6.
    assert result['Actor/source_001/loss_mean'] == 8.
    assert result['Actor/source_001/advantage_mean'] == 1.
    assert result['Critic/rollout_failure/sampled'] == 1


class LossTinyPolicy(TinyPolicy):
    def compute_loss(self, obs, actions, state, reduction='mean'):
        return self.scheduler.compute_loss(self.backbone, actions, self._get_condition(obs, state), reduction=reduction)


@pytest.mark.parametrize('bc', [False, True])
def test_actor_details_reuse_forward_preserve_loss_gradient_and_rng(tmp_path, monkeypatch, bc):
    from workflows import offline_round
    monkeypatch.setattr(offline_round, 'CriticFeatureExtractor', CriticFeatures)
    cfg = fixed_config(tmp_path)
    cfg.algo.use_bc_only = bc
    initial = LossTinyPolicy()
    a = offline_round.build_agent(cfg, copy.deepcopy(initial), torch.device('cpu'))
    b = offline_round.build_agent(cfg, copy.deepcopy(initial), torch.device('cpu'))
    obs = dict(pc=torch.zeros(4, 4, 3), state=torch.ones(4, 1))
    actions, advantage = torch.zeros(4, 3, 1), torch.tensor([[1.], [.9], [.7], [-2.]])
    torch.manual_seed(7)
    expected = a.update_actor(obs, actions, adv=advantage)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(7)
    actual, details = b.update_actor(obs, actions, adv=advantage, return_details=True)
    assert torch.equal(rng, torch.get_rng_state())
    assert expected['loss/actor'] == pytest.approx(actual['loss/actor'], rel=1e-6, abs=1e-7)
    assert details['losses'].mean().item() == actual['loss/actor']
    for before, after in zip(a.actor.parameters(), b.actor.parameters()):
        torch.testing.assert_close(before, after, atol=1e-7, rtol=1e-6)
    if bc:
        assert details['keep_mask'].all()
    else:
        assert not details['keep_mask'].all()


def prepare_inputs(cfg, tmp_path, extra_failures=0):
    eps = [episode(3, True), episode(3, True, .2), episode(30, False, .3)]
    eps += [episode(5+i, False, .4+i*.01) for i in range(extra_failures)]
    spec = specification(eps)
    for source, indices in zip(spec['sources'], [[0], list(range(1, len(eps)))]):
        source['episodes'] = []
        for i in indices:
            path = tmp_path/f'episode_{i}.npz'
            save_episode(path, eps[i])
            source['episodes'].append(dict(path=path.name, sha256=digest(path), seed=11000+i))
    spec.update(schema=SCHEMA, reward_mode='success', env=OmegaConf.to_container(cfg.env, resolve=True))
    cfg.initial_ckpt, cfg.stats_path, cfg.manifest = [str(tmp_path/n) for n in ('actor.pth', 'stats.json', 'manifest.json')]
    write_json(cfg.manifest, spec)
    norm().save(cfg.stats_path)
    base = LossTinyPolicy()
    torch.save(dict(model_state_dict=base.state_dict(), normalizer=norm().stats,
                    config=OmegaConf.to_container(cfg, resolve=True)), cfg.initial_ckpt)
    return base


@pytest.mark.parametrize('mode,extra_failures,updates', [('mixed', 0, 6), ('demo_success', 0, 6),
    ('demo_success', 5, 6), ('demo_success', 0, 7)])
def test_replay_only_actual_training_exact_budget_artifacts_and_resume(tmp_path, monkeypatch, mode, extra_failures, updates):
    from workflows import iterative, offline_round
    from evaluation import compare
    cfg = fixed_config(tmp_path, mode)
    cfg.output = str(tmp_path/'new_run')
    cfg.updates_per_round = updates
    cfg.epochs = 1
    cfg.critic_warmup_epochs = 999  # Ignored by update-budget mode.
    base = prepare_inputs(cfg, tmp_path, extra_failures)
    original = copy.deepcopy(base.state_dict())
    input_paths = [Path(cfg.initial_ckpt), Path(cfg.manifest), Path(cfg.stats_path), *tmp_path.glob('*.npz')]
    hashes = {p: digest(p) for p in input_paths}
    monkeypatch.setattr(iterative, 'build_base', lambda cfg, device: LossTinyPolicy().to(device))
    monkeypatch.setattr(compare, 'build_base', lambda cfg, device: LossTinyPolicy().to(device))
    monkeypatch.setattr(offline_round, 'CriticFeatureExtractor', CriticFeatures)
    def factory(cfg, primitive_wrapper=None, video=None):
        assert primitive_wrapper is None, 'Replay-only mode must not collect new trajectories'
        return ChunkActionWrapper(ToyEnv(), 3, 2)
    monkeypatch.setitem(sys.modules, 'envs.factory', SimpleNamespace(make_env=factory))
    agents = []
    real_build_agent = offline_round.build_agent
    def build_agent(cfg, base, device):
        agent = real_build_agent(cfg, base, device)
        agents.append((agent, copy.deepcopy(agent.q_net.state_dict()), copy.deepcopy(agent.v_net.state_dict())))
        return agent
    monkeypatch.setattr(offline_round, 'build_agent', build_agent)
    iterative.run(cfg)
    agent, old_q, old_v = agents[0]
    assert any(not torch.equal(v, agent.q_net.state_dict()[k]) for k, v in old_q.items())
    assert any(not torch.equal(v, agent.v_net.state_dict()[k]) for k, v in old_v.items())
    root = Path(cfg.output)
    assert all(digest(p) == hashes[p] for p in hashes)
    assert not list(root.rglob('*.npz'))
    rows = [json.loads(line) for line in (root/'metrics.jsonl').read_text().splitlines()]
    assert [r['update_step'] for r in rows] == ([0, 2, 3, 4, 6] if updates == 6 else [0, 2, 3, 4, 6, 7])
    assert all(r['epoch'] is None and r['budget_unit'] == 'updates' for r in rows)
    assert sum(r.get('Train/Critic_Updates', 0) for r in rows) == updates
    assert sum(r.get('Train/Actor_Updates', 0) for r in rows) == updates-2
    last = rows[-1]
    assert last['Train/Actor_Updates_Total'] == updates-2
    assert last['Cumulative/Critic/rollout_failure/sampled'] > 0
    if mode == 'demo_success':
        assert last['Cumulative/Actor/demo/sampled'] == last['Cumulative/Actor/rollout_success/sampled'] == (updates-2)*2
        assert last['Cumulative/Actor/rollout_failure/sampled'] == 0
    assert len(list(csv.DictReader((root/'metrics.csv').open()))) == len(rows)
    report = json.loads((root/'round_000/sampling.json').read_text())
    assert report['groups']['rollout_failure']['episodes'] == 1+extra_failures
    selection = json.loads((root/'selection.json').read_text())
    assert selection['criterion'] == ['Eval/Success_Rate']
    # Toy env always succeeds. A reward tie-break must not replace the incumbent.
    best = torch.load(root/'checkpoints/best.pth', weights_only=True)
    assert best['update_step'] == 0
    assert all(torch.equal(v, best['model_state_dict'][k]) for k, v in original.items())
    final = torch.load(root/f'round_000/checkpoints/step_{updates:07d}.pth', weights_only=True)
    assert any(not torch.equal(v, final['model_state_dict'][k]) for k, v in original.items())
    for r in rows:
        if r['evaluation']:
            report = json.loads((Path(r['evaluation'])/'summary.json').read_text())
            assert report['metadata']['update_step'] == r['update_step']
            assert Path(report['metadata']['checkpoint']).exists()
    with pytest.raises(FileExistsError):
        iterative.run(cfg)
    before = (root/'metrics.jsonl').read_bytes()
    cfg.resume = True
    iterative.run(cfg)
    assert (root/'metrics.jsonl').read_bytes() == before
    assert all(digest(p) == hashes[p] for p in hashes)
    cfg.comparison.checkpoints = dict(incumbent=cfg.initial_ckpt, trained=str(root/f'round_000/checkpoints/step_{updates:07d}.pth'))
    cfg.comparison.samplers = ['cps']
    cfg.comparison.output = str(tmp_path/'comparison')
    assert len(compare.run(cfg)) == 2


def test_balanced_preset_and_matched_control_compose():
    with initialize_config_dir(version_base=None, config_dir=str(Path(__file__).resolve().parents[1]/'configs')):
        for extra in ([], ['actor_sampling.mode=mixed', 'output=outputs/control']):
            cfg = compose(config_name='train_iterative', overrides=[
                '+experiment=oc_budget', '+iterative_protocol=balanced_replay', *extra])
            validate_update_config(cfg)
            assert cfg.updates_per_round == 20000 and cfg.critic_warmup_updates == 10000
            assert not cfg.collect and cfg.rounds == 1 and cfg.batch_size == 64
            assert cfg.algo.use_bc_only is False
            assert list(cfg.eval.seeds) == list(range(2000, 2050))
            OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
