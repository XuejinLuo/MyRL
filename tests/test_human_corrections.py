"""Takeover boundaries, label audit, and actual fixed-budget corrective training."""
import copy
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir

from data.corrections import eligible_starts, prepare
from data.dataset import TrajectoryDataset
from data.episodes import digest, load_episode, load_sources, save_episode, write_json
from data.iterative_sampling import SourceDataset, FixedBatches, validate_update_config
from envs.chunk_wrapper import ChunkActionWrapper
from tests.test_iterative_sampling import fixed_config, episode, prepare_inputs, LossTinyPolicy
from tests.test_stages import CriticFeatures
from tests.test_q_selection import norm
from tests.test_flow_ppo import ToyEnv
from utils.normalizer import MinMaxNormalizer
from workflows.collection import PrimitiveRecorder
from workflows.human_collection import FailureHints, TakeoverController


def snapshot(obs=None, info=None, contacts=True, **values):
    row = dict(flags=dict(success=False, is_cubeA_grasped=False, is_cubeA_on_cubeB=False),
               tcp_pose_world=[0., 0., .1, 1., 0., 0., 0.], gripper_width_m=.001,
               cubeA_left_force_world_N=[0., 0., 0.], cubeA_right_force_world_N=[0., 0., 0.],
               cubeB_left_force_world_N=[0., 0., 0.], cubeB_right_force_world_N=[0., 0., 0.])
    row.update(values)
    return row


class SevenActionEnv(gym.Env):
    action_space = gym.spaces.Box(-1., 1., (7,), dtype=np.float32)
    observation_space = gym.spaces.Dict({})
    def reset(self, **kwargs):
        self.actions = []
        return dict(point_cloud=np.zeros((4, 3), np.float32), state=np.zeros(1, np.float32)), {}
    def step(self, action):
        self.actions.append(np.array(action, copy=True))
        done = len(self.actions) >= 8
        return dict(point_cloud=np.zeros((4, 3), np.float32), state=np.array([len(self.actions)/10], np.float32)), 7., False, done, dict(success=done)


def seven_normalizer():
    n = MinMaxNormalizer()
    n.stats = dict(action=dict(min=[-1.]*7, max=[1.]*7))
    return n


def test_pause_and_takeover_discard_pending_prefix_and_preserve_labels():
    env = PrimitiveRecorder(SevenActionEnv())
    calls = []
    def sample(obs):
        calls.append(1)
        return np.full((4, 7), .3*len(calls), np.float32)
    c = TakeoverController(env, sample, seven_normalizer(), 2, snapshot, 20)
    c.reset(12000)
    assert not c.advance() and c.steps == 0
    c.switch('policy')
    c.advance()
    c.switch('human')
    for _ in range(20):
        assert not c.advance()  # Wall-clock waiting is not recorded.
    c.advance(c.human_command(2, .1, gripper=-1))
    c.advance(c.human_command(gripper=1))
    c.switch('policy')
    c.advance()
    assert len(calls) == 2  # stale .3 prefix was thrown away; resumed policy replans.
    np.testing.assert_allclose(env.env.actions[-1], .6, atol=1e-6)
    assert c.segments == [dict(id=0, start=1, stop=3)]
    assert [e['source'] for e in c.events] == ['policy', 'human', 'human', 'policy']
    ep = c.episode()
    assert ep['truncated'][-1] and not ep['terminated'].any()
    assert not ep['reward'].any()  # Dense reward 7 never becomes training reward.
    assert len(ep['state']) == len(ep['action'])+1


def test_uninterrupted_policy_matches_existing_chunk_execution():
    chunk = np.arange(28, dtype=np.float32).reshape(4, 7)/30-.5
    a, b = PrimitiveRecorder(SevenActionEnv()), PrimitiveRecorder(SevenActionEnv())
    c = TakeoverController(a, lambda obs: chunk, seven_normalizer(), 2, snapshot, 20)
    c.reset(12000)
    c.switch('policy')
    wrapped = ChunkActionWrapper(b, 4, 2)
    wrapped.reset(seed=12000)
    for _ in range(4):
        c.advance()
        c.advance()
        wrapped.step(seven_normalizer().unnormalize(chunk, 'action'))
    for k, value in c.episode().items():
        np.testing.assert_allclose(value, b.episode()[k], atol=1e-7)
    assert c.done and not c.advance()


def test_hints_debounce_missing_telemetry_and_intentional_release():
    detector = FailureHints(20)
    before, after = snapshot(), snapshot()
    action = np.array([0., 0., 0., 0., 0., 0., -1.])
    for _ in range(14):
        assert 'empty_closed' not in detector.update(before, after, action)
    assert 'empty_closed' in detector.update(before, after, action)
    after['flags']['is_cubeA_on_cubeB'] = True
    assert not detector.update(before, after, action)
    after['flags']['is_cubeA_on_cubeB'] = False
    after['flags']['is_cubeA_grasped'] = None
    after['cubeA_left_force_world_N'] = None
    for _ in range(30):
        assert not detector.update(before, after, action)
    # Persistent one-finger contact is a hint, independent of narrow gripper width.
    after = snapshot(gripper_width_m=.08, cubeA_right_force_world_N=[30., 0., 0.])
    for _ in range(10):
        hints = detector.update(before, after, np.zeros(7))
    assert 'one_finger_contact' in hints


def test_full_chunks_do_not_cross_boundaries_or_pad_human_tail():
    segments = [dict(start=2, stop=8, decision='accept'), dict(start=9, stop=11, decision='accept'),
                dict(start=12, stop=18, decision='reject'), dict(start=18, stop=22, decision='accept')]
    mask = eligible_starts(25, segments, 4, True)
    assert np.flatnonzero(mask).tolist() == [2, 3, 4, 18]
    assert not eligible_starts(25, segments, 4, False).any()
    with pytest.raises(ValueError, match='pending'):
        eligible_starts(5, [dict(start=0, stop=5, decision='pending')], 3, True)
    with pytest.raises(ValueError, match='overlapping'):
        eligible_starts(5, [dict(start=0, stop=3, decision='accept'), dict(start=2, stop=4, decision='accept')], 3, True)


def raw_session(cfg, tmp_path, success=True):
    session = tmp_path/'session'
    session.mkdir()
    ep = episode(10, success, .1)
    save_episode(session/'seed_12000.npz', ep)
    segments = [dict(id=0, start=2, stop=7), dict(id=1, start=8, stop=10)]
    meta = dict(seed=12000, segments=segments, final_success=success)
    write_json(session/'seed_12000.json', meta)
    item = dict(path='seed_12000.npz', sha256=digest(session/'seed_12000.npz'),
                metadata_sha256=digest(session/'seed_12000.json'), decision='keep',
                segments=[dict(s, decision='accept') for s in segments])
    write_json(session/'review.json', dict(schema='myrl_human_review_v1', episodes=[item]))
    write_json(session/'session.json', dict(schema='myrl_human_session_v1',
        config=OmegaConf.to_container(cfg, resolve=True), normalizer=norm().stats))
    return session


def test_cleaning_import_preserves_real_transitions_and_exact_actor_quota(tmp_path):
    cfg = fixed_config(tmp_path, 'demo_success_correction')
    prepare_inputs(cfg, tmp_path)
    session = raw_session(cfg, tmp_path)
    before = {p: digest(p) for p in tmp_path.glob('*.npz')}
    output = prepare(cfg.manifest, session, cfg.stats_path, tmp_path/'cleaned')
    eps, spec = load_sources(output)
    assert np.flatnonzero(eps[-1]['actor_eligible']).tolist() == [2, 3, 4]
    raw = load_episode(session/'seed_12000.npz')
    for key, value in raw.items():
        np.testing.assert_array_equal(eps[-1][key], value)
    data = SourceDataset(TrajectoryDataset(eps, cfg, norm()), spec, ['demonstrations'])
    for batch in FixedBatches(data, 4, 50, 42, 'demo_success_correction', .5, .25):
        groups = [int(data[i]['sampling_group']) for i in batch]
        assert groups.count(0) == 2 and groups.count(1) == 1 and groups.count(3) == 1
        for i in batch:
            e, t = data.dataset.indices[i]
            if groups[batch.index(i)] == 3:
                assert t in (2, 3, 4)
    assert all(digest(p) == sha for p, sha in before.items())
    # A different output name must not import the same source twice.
    with pytest.raises(ValueError, match='Duplicate raw'):
        prepare(output, session, cfg.stats_path, tmp_path/'duplicate')
    cfg.actor_sampling.mode = 'mixed'
    with pytest.raises(ValueError, match='Correction data requires'):
        TrajectoryDataset(eps, cfg, norm())


@pytest.mark.parametrize('bad', ['pending', 'bounds', 'held_out', 'boundary', 'changed_raw', 'failed', 'normalizer'])
def test_import_rejects_unsafe_labels_before_output_creation(tmp_path, bad):
    cfg = fixed_config(tmp_path, 'demo_success_correction')
    prepare_inputs(cfg, tmp_path)
    session = raw_session(cfg, tmp_path, success=bad != 'failed')
    review_path = session/'review.json'
    review = json.loads(review_path.read_text())
    item = review['episodes'][0]
    if bad == 'pending':
        item['segments'][0]['decision'] = 'pending'
    elif bad == 'boundary':
        item['segments'][0]['start'] = 0
    elif bad == 'held_out':
        p = session/'session.json'
        data = json.loads(p.read_text())
        data['config']['eval']['seeds'].append(12000)
        write_json(p, data)
    elif bad == 'normalizer':
        write_json(cfg.stats_path, {})
    elif bad in ('bounds', 'changed_raw'):
        p = session/item['path']
        ep = load_episode(p)
        p.unlink()
        ep['action'][0] = 2.
        save_episode(p, ep)
        if bad == 'bounds':
            item['sha256'] = digest(p)
    write_json(review_path, review)
    with pytest.raises(ValueError):
        prepare(cfg.manifest, session, cfg.stats_path, tmp_path/'bad')
    assert not (tmp_path/'bad').exists()


def test_actual_iterative_training_corrective_supervision_and_no_failed_actor(tmp_path, monkeypatch):
    from workflows import iterative, offline_round
    cfg = fixed_config(tmp_path, 'demo_success_correction')
    prepare_inputs(cfg, tmp_path)
    session = raw_session(cfg, tmp_path)
    cfg.manifest = str(prepare(cfg.manifest, session, cfg.stats_path, tmp_path/'cleaned'))
    cfg.output = str(tmp_path/'trained')
    monkeypatch.setattr(iterative, 'build_base', lambda cfg, device: LossTinyPolicy().to(device))
    monkeypatch.setattr(offline_round, 'CriticFeatureExtractor', CriticFeatures)
    monkeypatch.setitem(sys.modules, 'envs.factory', SimpleNamespace(
        make_env=lambda cfg, **kwargs: ChunkActionWrapper(ToyEnv(), 3, 2)))
    iterative.run(cfg)
    rows = [json.loads(s) for s in (Path(cfg.output)/'metrics.jsonl').read_text().splitlines()]
    last = rows[-1]
    assert last['Cumulative/Actor/correction/sampled'] == 4
    assert last['Cumulative/Actor/correction/kept'] == 4
    assert last['Cumulative/Actor/rollout_failure/sampled'] == 0
    assert last['Cumulative/Critic/correction/sampled'] > 0
    assert last['Cumulative/Critic/rollout_failure/sampled'] > 0
    assert json.loads((Path(cfg.output)/'selection.json').read_text())['update_step'] == 0
    checkpoint = Path(cfg.output)/'checkpoints/best.pth'
    assert torch.load(checkpoint, weights_only=True)['correction_training_seeds'] == [12000]
    from evaluation import compare
    cfg.comparison.checkpoints = dict(after=str(checkpoint))
    cfg.comparison.seed_start = 12000
    cfg.comparison.output = str(tmp_path/'invalid_comparison')
    with pytest.raises(ValueError, match='human correction training seeds'):
        compare.run(cfg)
    cfg.comparison.seed_start = 13000
    cfg.eval.seeds = [12000]
    cfg.output = str(tmp_path/'leaked')
    with pytest.raises(ValueError, match='Human correction seeds overlap'):
        iterative.run(cfg)


def test_reviewed_human_samples_survive_low_q_advantage(tmp_path, monkeypatch):
    from workflows import offline_round
    monkeypatch.setattr(offline_round, 'CriticFeatureExtractor', CriticFeatures)
    cfg = fixed_config(tmp_path)
    agent = offline_round.build_agent(cfg, LossTinyPolicy(), torch.device('cpu'))
    _, details = agent.update_actor(dict(pc=torch.zeros(4, 4, 3), state=torch.zeros(4, 1)),
        torch.zeros(4, 3, 1), adv=torch.tensor([[0.], [-1e6], [-1e6], [-1e6]]),
        return_details=True, force_keep=torch.tensor([False, False, False, True]))
    assert details['keep_mask'].tolist() == [True, False, False, True]


def test_human_preset_composes_and_validates():
    with initialize_config_dir(version_base=None, config_dir=str(Path(__file__).resolve().parents[1]/'configs')):
        cfg = compose(config_name='train_iterative', overrides=['+experiment=oc_budget', '+iterative_protocol=human_corrections'])
        validate_update_config(cfg)
        assert cfg.actor_sampling.mode == 'demo_success_correction'
        assert cfg.actor_sampling.correction_fraction == .25
        assert not cfg.collect
        assert cfg.env.control_mode == 'pd_ee_delta_pose'
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
