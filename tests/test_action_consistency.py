"""Regression for saturated Q representations versus physically executed actions."""
import copy
import csv

import numpy as np
import pytest
import torch

from models.action_mapping import map_actions
from envs.chunk_wrapper import ChunkActionWrapper
from evaluation.runner import evaluate_policy, seed_all
from models.factory import observation_encoder, observation_tensorizer
from models.online_policy import FlowPPOPolicy
from models.q_selection import QSelectionPolicy
from tests.test_flow_ppo import ToyEnv
from tests.test_q_selection import (BufferedActor, CandidateActor, RecordingTwinQ,
                                    ScoringEncoder, norm)
from tests.test_stages import config
from workflows.offline_round import PrefixQ
from torch import nn


class PreferFirst(nn.Module):
    def forward(self, features, actions):
        self.seen = actions.clone()
        return actions[:, 0], actions[:, 0]


def test_reproduce_legacy_saturation_mismatch_before_fix():
    normalizer = norm()
    # Wide environment bounds do not hide the normalizer saturation problem.
    q = PrefixQ(ScoringEncoder(), PreferFirst(), 1)
    policy = QSelectionPolicy(CandidateActor(), q, normalizer, 2)
    policy.set_action_bounds(np.full((3, 1), -2.), np.full((3, 1), 2.))
    chosen = policy.sample({'pc': torch.zeros(2, 4, 3), 'state': torch.zeros(2, 1)})
    scored_physical = normalizer.unnormalize(q.q_net.seen[[0, 2]], 'action')
    executed = normalizer.unnormalize(chosen[:, :1], 'action').clamp(-2, 2)
    assert torch.max(torch.abs(executed - scored_physical)) > .09
    assert chosen[:, 0, 0].tolist() == [9., 9.]


@pytest.mark.parametrize('eps', [1e-6, .025])
@pytest.mark.parametrize('degenerate', [False, True])
def test_consistent_mapping_asymmetric_narrow_bounds_and_eps(eps, degenerate):
    normalizer = norm()
    normalizer.eps = eps
    normalizer.stats['action'] = {'min': [-.7, .2], 'max': [1.3, .2 if degenerate else .9]}
    raw = torch.tensor([[[-9., -1.05], [1.05, 9.], [0., .1]],
                        [[1.1, -9.], [-1.1, 1.1], [.7, -.2]]])
    low, high = torch.tensor([-.4, -.2]), torch.tensor([.6, 2.])
    returned, scored, score_physical, diagnostics = map_actions(raw, normalizer, low, high, 'consistent')
    np_executed = np.clip(normalizer.unnormalize(returned.numpy(), 'action'), low.numpy(), high.numpy())
    np.testing.assert_allclose(np_executed, score_physical.numpy(), atol=2e-6, rtol=1e-5)
    expected = normalizer.unnormalize(raw.clamp(-1, 1), 'action')
    expected = torch.maximum(torch.minimum(expected, high), low)
    torch.testing.assert_close(score_physical, expected, atol=2e-6, rtol=1e-5)
    assert torch.equal(returned, scored)
    assert (returned.abs() <= 1).all()
    assert diagnostics['raw_outside'].sum() == 8
    assert diagnostics['env_clipped'][..., 0].sum() >= 4
    if degenerate:
        assert score_physical[0, 0, 1].item() == pytest.approx(.2)
        assert score_physical[0, 1, 1].item() == pytest.approx(.2 + eps)


@pytest.mark.parametrize('mode', ['legacy', 'consistent'])
@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_actions_rejected(mode, bad):
    with pytest.raises(FloatingPointError, match='candidate action'):
        map_actions(torch.tensor([[[bad]]]), norm(), torch.tensor(-2.), torch.tensor(2.), mode)


def test_empty_intersection_and_invalid_statistics_are_explicit():
    raw = torch.zeros(1, 3, 1)
    with pytest.raises(ValueError, match='Empty intersection'):
        map_actions(raw, norm(), torch.tensor(2.), torch.tensor(3.), 'consistent')
    with pytest.raises(ValueError, match='finite ordered environment'):
        map_actions(raw, norm(), torch.tensor(float('nan')), torch.tensor(3.), 'consistent')
    normalizer = norm()
    normalizer.stats['action']['min'] = [2.]
    with pytest.raises(ValueError, match='normalizer statistics'):
        map_actions(raw, normalizer, torch.tensor(-2.), torch.tensor(3.), 'consistent')
    normalizer.stats = {}
    with pytest.raises(ValueError, match='requires action'):
        map_actions(raw, normalizer, torch.tensor(-2.), torch.tensor(3.), 'consistent')


def test_consistent_twin_min_prefix_and_full_chunk_batch_selection():
    actor, normalizer = CandidateActor(), norm()
    q = PrefixQ(ScoringEncoder(), RecordingTwinQ(), 1)
    policy = QSelectionPolicy(actor, q, normalizer, 2, action_mode='consistent')
    policy.set_action_bounds(np.full((3, 1), -.8, np.float32), np.full((3, 1), .8, np.float32))
    obs = {'pc': torch.zeros(2, 4, 3), 'state': torch.zeros(2, 1)}
    chosen = policy.sample(obs)
    assert chosen.shape == (2, 3, 1)
    executed = normalizer.unnormalize(chosen, 'action').clamp(-.8, .8)
    torch.testing.assert_close(executed[:, :1], normalizer.unnormalize(q.q_net.seen[[1, 3]], 'action'))
    torch.testing.assert_close(executed[:, 1:], torch.full((2, 2, 1), .8))
    assert not torch.equal(chosen[:, 1:], torch.full((2, 2, 1), 9.))


@pytest.mark.parametrize('mode', ['legacy', 'consistent'])
def test_diagnostics_only_include_executed_steps_and_selected_candidate(mode):
    class FixedActor(CandidateActor):
        def sample(self, features, num_steps=None):
            return torch.tensor([[[1.05], [-9.], [9.]], [[.5], [9.], [9.]]])[:len(features)]
    class AnyEncoder(nn.Module):
        def forward(self, obs):
            return obs['state']
    policy = QSelectionPolicy(FixedActor(), PrefixQ(AnyEncoder(), PreferFirst(), 2),
                              norm(), 2, action_mode=mode)
    policy.set_action_bounds(np.full((3, 1), -2.), np.full((3, 1), 2.))
    chosen = policy.sample({'pc': torch.zeros(1, 4, 3), 'state': torch.zeros(1, 1)})
    # Terminates after just one primitive step, despite configured exec_steps=2.
    executed = policy.normalizer.unnormalize(chosen[0, :1].numpy(), 'action')
    policy.record_execution(dict(actual_steps=1, executed_actions=executed))
    report = policy.action_diagnostics()
    assert report['Action/candidates/coordinates'] == 2
    assert report['Action/candidates/chunks'] == 2
    assert report['Action/candidates/raw_outside_fraction'] == .5
    assert report['Action/selected/coordinates'] == 1
    assert report['Action/selected/raw_outside_fraction'] == 1.
    assert report['Action/candidates/raw_to_score_max_abs'] > .04
    assert (report['Action/selected/execution_to_score_max_abs'] < 2e-6) == (mode == 'consistent')
    policy.reset_episode_diagnostics()
    assert policy.action_diagnostics(episode=True)['Action/candidates/coordinates'] == 0
    assert policy.action_diagnostics()['Action/candidates/coordinates'] == 2


@pytest.mark.parametrize('mode', ['legacy', 'consistent'])
@pytest.mark.parametrize('sampler', ['cps', 'ode'])
def test_single_diagnostic_path_retains_rng_and_never_calls_q(tmp_path, mode, sampler):
    cfg = config('offline', tmp_path)
    actor = FlowPPOPolicy(BufferedActor(), num_steps=3, eval_mode=sampler).eval()
    normalizer = norm()
    selected = QSelectionPolicy(actor, nn.Identity(), normalizer, 1, action_mode=mode, exec_steps=2)
    low, high = np.full((3, 1), -1., np.float32), np.full((3, 1), 1., np.float32)
    selected.set_action_bounds(low, high)
    obs = {'pc': torch.zeros(1, 4, 3), 'state': torch.zeros(1, 1)}
    seed_all(20)
    raw = actor.sample(actor.encode(obs['pc'], obs['state']), 3)
    rng = torch.get_rng_state().clone()
    seed_all(20)
    actual = selected.sample(obs, 3)
    assert torch.equal(torch.get_rng_state(), rng)
    expected = map_actions(raw, normalizer, torch.tensor(low), torch.tensor(high), mode)[0]
    assert torch.equal(actual, expected)
    if mode == 'legacy':
        assert torch.equal(actual, raw)
        factory = lambda: ChunkActionWrapper(ToyEnv(), 3, 2)
        baseline = evaluate_policy(factory, actor, observation_encoder(cfg, actor, normalizer, 'cpu'),
                                   normalizer, [4000, 4001], 3, output_dir=tmp_path/'baseline')
        actual_metrics = evaluate_policy(factory, selected, observation_tensorizer(cfg, normalizer, 'cpu'),
                                        normalizer, [4000, 4001], 3, output_dir=tmp_path/'selected')
        assert baseline == {k: v for k, v in actual_metrics.items() if k.startswith('Eval/')}
        before = list(csv.DictReader((tmp_path/'baseline/episodes.csv').open()))
        after = list(csv.DictReader((tmp_path/'selected/episodes.csv').open()))
        assert before == [{k: row[k] for k in before[0]} for row in after]


def test_realistic_full_chunk_and_execution_prefix_correspondence():
    class FullActor(CandidateActor):
        def sample(self, features, num_steps=None):
            values = torch.linspace(-1.5, 1.5, 8 * 16 * 7).reshape(8, 16, 7)
            return values.repeat(len(features) // 8, 1, 1)
    class Encoder(nn.Module):
        def forward(self, obs):
            return obs['state']
    class Q(nn.Module):
        def forward(self, features, actions):
            self.seen = actions.clone()
            assert actions.shape == (16, 2, 7)
            value = actions.sum((1, 2))[:, None]
            return value, value
    normalizer = norm()
    normalizer.stats['action'] = {'min': [-.4]*7, 'max': [.7]*7}
    q = PrefixQ(Encoder(), Q(), 2)
    policy = QSelectionPolicy(FullActor(), q, normalizer, 8, action_mode='consistent')
    policy.set_action_bounds(np.full((16, 7), -1.), np.full((16, 7), 1.))
    chosen = policy.sample({'pc': torch.zeros(2, 4, 3), 'state': torch.zeros(2, 1)})
    assert chosen.shape == (2, 16, 7)
    executed = normalizer.unnormalize(chosen[:, :2], 'action').clamp(-1, 1)
    torch.testing.assert_close(executed, normalizer.unnormalize(q.q_net.seen[[7, 15]], 'action'))


def test_paired_success_uses_named_baseline_and_seed_not_row_order():
    from evaluation.q_selection import paired_success
    def rows(values):
        return {seed: {'success': str(value)} for seed, value in values}
    episodes = dict(
        legacy_single=rows([(13, 0), (10, 0), (12, 0), (11, 1)]),
        consistent_single=rows([(10, 1), (11, 0), (12, 1), (13, 0)]),
        consistent_q8=rows([(13, 1), (12, 0), (11, 0), (10, 0)]))
    report = paired_success(episodes, [10, 11, 12, 13], 'consistent_single', 'consistent_q8')
    assert report == dict(baseline_label='consistent_single', selected_label='consistent_q8',
                         baseline_fail_selected_success=1, baseline_success_selected_fail=2,
                         success_delta=-.25)


def test_eval_keeps_critic_batchnorm_buffers_fixed_and_checks_observed_execution():
    class BufferedEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm1d(1)
        def forward(self, obs):
            return self.bn(obs['state'])
    q = PrefixQ(BufferedEncoder(), PreferFirst(), 1)
    policy = QSelectionPolicy(CandidateActor(), q, norm(), 2, action_mode='consistent')
    policy.set_action_bounds(np.full((3, 1), -2.), np.full((3, 1), 2.))
    before = copy.deepcopy(q.state_dict())
    policy.sample({'pc': torch.zeros(1, 4, 3), 'state': torch.zeros(1, 1)})
    assert all(torch.equal(value, q.state_dict()[key]) for key, value in before.items())
    with pytest.raises(RuntimeError, match='observed physical execution'):
        policy.record_execution(dict(actual_steps=1, executed_actions=np.zeros((1, 1))))
