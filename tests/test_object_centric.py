"""Representation, storage, normalization and real model tests without SAPIEN."""
import copy
from pathlib import Path
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from data.object_centric import (build_object_observation, OBJECT_FIELDS,
                                 validate_object_arrays, validate_object_config)
from data.observations import (normalize_observation, batch_observation, observation_mode)
from data.demonstrations import load_demonstrations, export_demonstrations
from data.episodes import load_sources, validate_episode, transition
from data.dataset import TrajectoryDataset
from envs.maniskill_bridge import ManiSkillToRL100Wrapper
from envs.object_centric_wrapper import ObjectCentricObservationWrapper
from models.encoders.object_centric import ObjectCentricEncoder
from models.factory import build_base, observation_encoder
from models.online_policy import FlowPPOPolicy
from utils.normalizer import MinMaxNormalizer
from tests.test_point_sampling import cloud, make_h5, RawEnv, BOUNDS, config as global_config

CONFIG = dict(mode='object_centric', context_points=32, objects=[
    dict(name='cubeA', role='manipulated', h5_id=18, max_points=16),
    dict(name='cubeB', role='target', h5_id=19, max_points=16)])


def build(a=80, b=3, background=40, use_color=True):
    xyz, seg = cloud(a, b, background)
    return build_object_observation(xyz, np.ones_like(xyz)*.5, seg, BOUNDS, CONFIG,
                                    use_color, rng=np.random.default_rng(7))


def object_config(path):
    cfg = global_config(path)
    cfg.env.observation = CONFIG
    return cfg


@pytest.mark.parametrize('a,b,background', [(80, 3, 40), (1, 0, 2), (0, 0, 0)])
def test_real_points_no_repeats_background_exclusion_and_missing(a, b, background):
    obs = build(a, b, background)
    validate_object_arrays(obs, CONFIG, 6)
    assert obs['object_point_mask'].sum(-1).tolist() == [min(a, 16), min(b, 16)]
    assert obs['context_point_mask'].sum() == min(background, 32)
    xyz, seg = cloud(a, b, background)
    for slot, (sid, count) in enumerate([(18, a), (19, b)]):
        mask = obs['object_point_mask'][slot]
        local = obs['object_points'][slot, mask, :3]
        assert len(np.unique(local, axis=0)) == min(count, 16)
        if count:
            np.testing.assert_allclose(obs['object_centers'][slot], xyz[seg == sid].mean(0))
            reconstructed = local + obs['object_centers'][slot]
            assert np.all(np.min(np.linalg.norm(reconstructed[:, None] - xyz[seg == sid], axis=-1), axis=1) < 1e-6)
    context = obs['context_points'][obs['context_point_mask'], :3]
    if len(context):
        assert np.all(np.min(np.linalg.norm(context[:, None] - xyz[seg == 0], axis=-1), axis=1) == 0)


def test_filtering_rgb_alignment_local_coordinates_and_rng_stable_centers():
    xyz, seg = cloud(80, 3, 40)
    rgb = np.repeat(np.arange(len(xyz), dtype=np.float32)[:, None], 3, axis=1)
    xyz[0, 0], xyz[1, 0], rgb[2, 0] = np.nan, 10, np.nan
    obs = build_object_observation(xyz, rgb, seg, BOUNDS, CONFIG, rng=np.random.default_rng(1))
    again = build_object_observation(xyz, rgb, seg, BOUNDS, CONFIG, rng=np.random.default_rng(2))
    np.testing.assert_array_equal(obs['object_centers'], again['object_centers'])
    for slot in range(2):
        points = obs['object_points'][slot, obs['object_point_mask'][slot]]
        indices = points[:, 3].astype(int)
        assert not {0, 1, 2} & set(indices)
        np.testing.assert_allclose(points[:, :3] + obs['object_centers'][slot], xyz[indices], atol=1e-7)


@pytest.mark.parametrize('change', [dict(objects=[]), dict(context_points=0),
    dict(objects=[dict(name='cubeA', h5_id=18, role='wrong', max_points=16)]),
    dict(objects=[dict(name='cubeA', h5_id=18, role='target', max_points=-1)])])
def test_reject_bad_config(change):
    with pytest.raises(ValueError):
        validate_object_config(dict(CONFIG, **change))


def test_reject_missing_segmentation_and_corrupt_arrays():
    with pytest.raises(ValueError, match='requires pointcloud segmentation'):
        build_object_observation(np.zeros((1, 3)), np.zeros((1, 3)), None, BOUNDS, CONFIG)
    for key in OBJECT_FIELDS:
        obs = build()
        obs[key] = obs[key].astype(float)
        obs[key].flat[0] = np.nan
        with pytest.raises(ValueError):
            validate_object_arrays(obs)
    obs = build()
    obs['object_valid'][1] = False
    with pytest.raises(ValueError, match='must match'):
        validate_object_arrays(obs)


def test_h5_live_parity_remapping_recorder_and_v2_roundtrip(tmp_path):
    from workflows.collection import PrimitiveRecorder
    path = tmp_path/'source.h5'
    raw_obs = make_h5(path)
    cfg = object_config(path)
    np.random.seed(13)
    ep, = load_demonstrations(cfg)
    raw = RawEnv(raw_obs)
    wrapper = ObjectCentricObservationWrapper(ManiSkillToRL100Wrapper(raw, objects=CONFIG['objects']),
                                              CONFIG, BOUNDS)
    np.random.seed(13)
    first, _ = wrapper.reset()
    second, *_ = wrapper.step(np.zeros(1))
    assert wrapper.observation_space.contains(first)
    for key in (*OBJECT_FIELDS, 'state'):
        np.testing.assert_array_equal(first[key], ep[key][0])
        np.testing.assert_array_equal(second[key], ep[key][1])
    raw.obs['pointcloud']['segmentation'] = np.where(raw_obs['pointcloud']['segmentation'] == 71,
        101, np.where(raw_obs['pointcloud']['segmentation'] == 92, 202, 0))
    from types import SimpleNamespace
    raw.segmentation_id_map = {101: SimpleNamespace(name='cubeA'), 202: SimpleNamespace(name='cubeB')}
    np.random.seed(13)
    remapped, _ = wrapper.reset()
    for key in first:
        np.testing.assert_array_equal(remapped[key], first[key])
    raw.step = lambda action: (raw.obs, 1., True, False, {'success': True})
    recorder = PrimitiveRecorder(wrapper)
    recorder.reset()
    recorder.step(np.zeros(7))
    recorded = recorder.episode()
    validate_episode(recorded)
    assert 'pc' not in recorded
    export_demonstrations([ep], cfg, tmp_path/'export')
    restored, spec = load_sources(tmp_path/'export/manifest.json')
    assert spec['schema'] == 'myrl_primitive_v2'
    for key in ep:
        np.testing.assert_array_equal(restored[0][key], ep[key])
    row = transition(ep, 0, 4, 2, .9)
    np.testing.assert_array_equal(row['next_object_points'], ep['object_points'][1])


def tensor_obs(obs, state_dim=16):
    return {**{k: torch.from_numpy(v)[None] for k, v in obs.items()},
            'state': torch.zeros(1, state_dim)}


@pytest.mark.parametrize('counts', [(80, 3, 40), (0, 0, 0)])
def test_mask_invariance_permutation_finite_backward_and_256_condition(counts):
    obs = tensor_obs(build(*counts))
    encoder = ObjectCentricEncoder().eval()
    expected = encoder(obs)
    assert expected.shape == (1, 256) and torch.isfinite(expected).all()
    corrupted = {k: v.clone() for k, v in obs.items()}
    corrupted['object_points'][~obs['object_point_mask']] = torch.nan
    corrupted['context_points'][~obs['context_point_mask']] = 1e6
    torch.testing.assert_close(encoder(corrupted), expected)
    permuted = {k: v.clone() for k, v in obs.items()}
    order = torch.randperm(obs['object_points'].shape[2])
    permuted['object_points'] = permuted['object_points'][:, :, order]
    permuted['object_point_mask'] = permuted['object_point_mask'][:, :, order]
    torch.testing.assert_close(encoder(permuted), expected, rtol=1e-5, atol=1e-6)
    encoder(corrupted).square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in encoder.parameters() if p.grad is not None)


def test_center_ablation_ignores_cloud_extent_and_context():
    obs = tensor_obs(build())
    encoder = ObjectCentricEncoder(variant='centers').eval()
    expected = encoder(obs)
    for key in ('object_points', 'context_points', 'object_extents'):
        obs[key] = torch.randn_like(obs[key])*100
    torch.testing.assert_close(encoder(obs), expected)


@pytest.mark.parametrize('state_skip', [False, True])
def test_dataset_rollout_normalization_actor_critic_and_bc_gradient(tmp_path, state_skip):
    from algos.embodied_idql import CriticFeatureExtractor, Policy_IDQL_Wrapper
    path = tmp_path/'source.h5'
    make_h5(path)
    cfg = object_config(path)
    if state_skip:
        OmegaConf.update(cfg, 'env.observation.state_skip', True, force_add=True)
    ep, = load_demonstrations(cfg)
    original = {k: v.copy() for k, v in ep.items()}
    norm = MinMaxNormalizer()
    norm.fit({k: ep[k] for k in ('state', 'action')})
    ds = TrajectoryDataset([ep], cfg, norm)
    row = ds[0]
    obs = {k: v[None] for k, v in batch_observation(row).items()}
    assert obs['object_roles'].dtype == torch.int64
    assert obs['object_point_mask'].dtype == torch.bool
    base = build_base(cfg, 'cpu')
    assert isinstance(base.encoder, ObjectCentricEncoder)
    critics = [CriticFeatureExtractor(cfg) for _ in range(2)]
    for critic in critics:
        assert isinstance(critic.encoder, ObjectCentricEncoder)
        assert (critic.encoder.state_projection is not None) == state_skip
        assert critic(obs).shape == (1, 256)
    assert not set(map(id, critics[0].parameters())) & set(map(id, critics[1].parameters()))
    assert not set(map(id, base.parameters())) & set(map(id, critics[0].parameters()))
    base.eval()
    expected = base.encode(obs).detach()
    actor = FlowPPOPolicy(base, num_steps=2)
    encode = observation_encoder(cfg, actor, norm, 'cpu')
    physical = {k: ep[k][0] for k in (*OBJECT_FIELDS, 'state')}
    torch.testing.assert_close(encode(physical), expected)
    base.requires_grad_(True)
    optimizer = torch.optim.Adam(base.parameters(), lr=1e-4)
    # The existing DiT output and conditioning layers are zero-initialized.
    for _ in range(4):
        optimizer.zero_grad()
        loss = Policy_IDQL_Wrapper(base).compute_loss(obs, row['action_chunk'][None])
        loss.backward()
        optimizer.step()
        assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in base.encoder.parameters())
    if state_skip:
        assert base.encoder.state_projection.weight[:, -cfg.model.state_dim:].abs().sum() > 0
    for v in row.values():
        if v.dtype == torch.bool:
            v.logical_not_()
        else:
            v.add_(1)
    for k in ep:
        np.testing.assert_array_equal(ep[k], original[k])


def test_normalization_leaves_missing_geometry_and_padding_zero():
    obs = dict(**build(2, 0, 3), state=np.zeros(16, dtype=np.float32))
    normalized = normalize_observation(obs, MinMaxNormalizer(), [[0, 0, 0], [2, 2, 2]])
    assert not normalized['object_centers'][1].any()
    assert not normalized['context_points'][~obs['context_point_mask']].any()
    np.testing.assert_array_equal(normalized['object_points'], obs['object_points'])


def test_legacy_routing_and_explicit_comparison_opt_in():
    from evaluation.compare import comparison_protocol
    assert observation_mode({'sampling': {'mode': 'random'}}) == 'global_random'
    assert observation_mode({'sampling': {'mode': 'object_budget'}}) == 'global_object_budget'
    a = dict(env=dict(env_id='StackCube-v1', exec_steps=2, sampling={'mode': 'random'}),
             model=dict(action_dim=7, cond_dim=256, in_channels=6, encoder_type='pointnext'))
    b = copy.deepcopy(a)
    b['env']['observation'] = CONFIG
    assert comparison_protocol(a, {}, .7, .0067) != comparison_protocol(b, {}, .7, .0067)
    assert comparison_protocol(a, {}, .7, .0067, True) == comparison_protocol(b, {}, .7, .0067, True)
    b['env']['exec_steps'] = 3
    assert comparison_protocol(a, {}, .7, .0067, True) != comparison_protocol(b, {}, .7, .0067, True)


def test_diagnostic_uses_production_loader(tmp_path):
    from tools.diagnostics.check_object_centric import inspect_objects
    path = tmp_path/'source.h5'
    make_h5(path)
    result = inspect_objects(object_config(path), max_episodes=1)
    assert result['frames'] == 2
    assert result['metrics']['Object/cubeB/points_mean'] == 16
    assert result['metrics']['Object/cubeB/missing_rate'] == 0
