"""Relational geometry, H5/live parity, routing and checkpoint regressions."""
import copy
import io
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from data.object_features import build_relational_features
from data.observations import (normalize_observation, batch_observation,
                               validate_observation, validate_observation_config)
from data.demonstrations import load_demonstrations, export_demonstrations
from data.episodes import load_sources, validate_episode
from data.dataset import TrajectoryDataset
from envs.maniskill_bridge import ManiSkillToRL100Wrapper
from envs.pointcloud_wrapper import PointCloudObservationWrapper
from models.factory import build_base, observation_encoder
from utils.normalizer import MinMaxNormalizer
from tests.test_point_sampling import SAMPLING, BOUNDS, RawEnv, make_h5, config


def enabled_config(path):
    cfg = config(path)
    OmegaConf.update(cfg, 'env.observation.relational_features', True, force_add=True)
    OmegaConf.update(cfg, 'model.object_feature_dim', 23, force_add=True)
    OmegaConf.update(cfg, 'model.object_feature_hidden_dim', 64, force_add=True)
    return cfg


def geometry(xyz, seg, tcp=(0, 0, 0), **kwargs):
    return build_relational_features(xyz, seg, SAMPLING['objects'], tcp,
                                    [[-5]*3, [5]*3], **kwargs)


def test_geometry_filtering_permutation_and_missing_objects():
    xyz = np.array([[.5, 0, 0], [1.5, 0, 0], [2, 0, 0], [np.nan, 0, 0], [9, 0, 0]])
    seg = np.array([18, 18, 19, 18, 19])
    f = geometry(xyz, seg)
    np.testing.assert_array_equal(f, [1,0,0, 2,0,0, 1,0,0, 0,0,0, 1,0,0, 2,0,0, 1,0,0, 1,1])
    perm = np.array([4, 2, 0, 3, 1])
    np.testing.assert_array_equal(geometry(xyz[perm], seg[perm]), f)
    missing = geometry(xyz[seg != 18], seg[seg != 18], tcp=[.3, .1, .2])
    assert np.isfinite(missing).all()
    np.testing.assert_array_equal(missing[np.r_[0:3, 6:9, 12:15, 18:22]], 0.)
    np.testing.assert_array_equal(geometry(np.empty((0, 3)), []), np.zeros(23))
    rgb = np.zeros_like(xyz)
    rgb[1] = np.nan
    filtered = geometry(xyz, seg, rgb=rgb, use_color=True)
    assert filtered[0] == .5 and filtered[6] == 0


def test_normalization_centers_only_visible_centroids_without_mutating_input():
    f = geometry([[1,0,0]], [19])
    obs = dict(pc=np.ones((256, 3)), state=np.zeros(16), object_features=f)
    norm = normalize_observation(obs, MinMaxNormalizer(), np.array([[0,0,0], [2,2,2]]))
    np.testing.assert_array_equal(norm['object_features'][:3], [0,0,0])
    np.testing.assert_array_equal(norm['object_features'][3:6], [0,-1,-1])
    np.testing.assert_array_equal(norm['object_features'][6:], f[6:])
    np.testing.assert_array_equal(obs['object_features'], f)


def test_offline_live_parity_sampling_independence_per_frame_and_dataset(tmp_path):
    path = tmp_path/'source.h5'
    obs = make_h5(path)
    cfg = enabled_config(path)
    np.random.seed(7)
    ep, = load_demonstrations(cfg)
    raw = RawEnv(obs)
    wrapped = PointCloudObservationWrapper(ManiSkillToRL100Wrapper(raw, sampling=cfg.env.sampling),
        workspace_bounds=BOUNDS, use_color=True, sampling=cfg.env.sampling, relational_features=True)
    np.random.seed(7)
    first, _ = wrapped.reset()
    second, *_ = wrapped.step(np.zeros(1))
    for i, frame in enumerate([first, second]):
        np.testing.assert_array_equal(frame['object_features'], ep['object_features'][i])
        np.testing.assert_array_equal(frame['point_cloud'], ep['pc'][i])
        assert wrapped.observation_space.contains(frame)
    np.random.seed(89)
    other, _ = wrapped.reset()
    assert not np.array_equal(first['point_cloud'], other['point_cloud'])
    np.testing.assert_array_equal(first['object_features'], other['object_features'])
    # Re-resolve changing numeric scene IDs and recompute after motion on step.
    from types import SimpleNamespace
    raw.obs['pointcloud']['segmentation'] = np.where(obs['pointcloud']['segmentation'] == 71,
        101, np.where(obs['pointcloud']['segmentation'] == 92, 202, 0))
    raw.segmentation_id_map = {101: SimpleNamespace(name='cubeA'), 202: SimpleNamespace(name='cubeB')}
    remapped, _ = wrapped.reset()
    np.testing.assert_array_equal(first['object_features'], remapped['object_features'])
    raw.obs['extra']['tcp_pose'][0] = .2
    moved, *_ = wrapped.step(np.zeros(1))
    assert moved['object_features'][12] == pytest.approx(first['object_features'][12] - .2)
    assert moved['object_features'][15] == pytest.approx(first['object_features'][15] - .2)
    export_demonstrations([ep], cfg, tmp_path/'export')
    restored, _ = load_sources(tmp_path/'export/manifest.json')
    np.testing.assert_array_equal(restored[0]['object_features'], ep['object_features'])
    norm = MinMaxNormalizer()
    norm.fit(dict(state=ep['state'], action=ep['action']))
    # Distinguish current/final features to catch T+1 off-by-one routing.
    ep['object_features'][1, 12] += .1
    row = TrajectoryDataset([ep], cfg, norm)[0]
    torch.testing.assert_close(batch_observation(row)['object_features'], torch.tensor(ep['object_features'][0]))
    torch.testing.assert_close(batch_observation(row, 'next_')['object_features'], torch.tensor(ep['object_features'][1]))
    broken = {**ep, 'object_features': ep['object_features'][:1]}
    with pytest.raises(ValueError):
        validate_episode(broken)


def tensor_obs(cfg):
    return dict(pc=torch.rand(2, 256, cfg.model.in_channels),
                state=torch.rand(2, cfg.model.state_dim), object_features=torch.rand(2, 23))


def test_identity_rng_baseline_checkpoint_and_strict_mismatch(tmp_path):
    cfg = enabled_config(tmp_path/'unused')
    cfg.env.observation.relational_features = False
    torch.manual_seed(42)
    baseline = build_base(cfg, 'cpu').eval()
    baseline_rng = torch.get_rng_state().clone()
    cfg.env.observation.relational_features = True
    torch.manual_seed(42)
    relational = build_base(cfg, 'cpu').eval()
    assert torch.equal(baseline_rng, torch.get_rng_state())
    for key, value in baseline.state_dict().items():
        torch.testing.assert_close(relational.state_dict()[key], value, rtol=0, atol=0)
    obs = tensor_obs(cfg)
    with torch.no_grad():
        torch.testing.assert_close(relational.encode(obs), baseline.encode(obs), rtol=0, atol=0)
    for enabled, model in [(False, baseline), (True, relational)]:
        cfg.env.observation.relational_features = enabled
        saved = io.BytesIO()
        torch.save(dict(config=OmegaConf.to_container(cfg, resolve=True), model_state_dict=model.state_dict()), saved)
        saved.seek(0)
        cp = torch.load(saved, weights_only=True)
        restored_cfg = OmegaConf.create(cp['config'])
        build_base(restored_cfg, 'cpu').load_state_dict(cp['model_state_dict'], strict=True)
        restored_cfg.env.observation.relational_features = not enabled
        with pytest.raises(RuntimeError):
            build_base(restored_cfg, 'cpu').load_state_dict(cp['model_state_dict'], strict=True)
    del cfg.env.observation.relational_features
    del cfg.model.object_feature_dim
    del cfg.model.object_feature_hidden_dim
    legacy = build_base(cfg, 'cpu')
    legacy.load_state_dict(baseline.state_dict(), strict=True)
    assert set(legacy.state_dict()) == set(baseline.state_dict())


def test_actor_critic_routing_and_gradient_after_fusion_update(tmp_path):
    from algos.embodied_idql import Policy_IDQL_Wrapper, CriticFeatureExtractor
    cfg = enabled_config(tmp_path/'unused')
    base = build_base(cfg, 'cpu').eval()
    obs = tensor_obs(cfg)
    adapter = Policy_IDQL_Wrapper(base)
    optimizer = torch.optim.SGD(base.encoder.parameters(), lr=.1)
    for step in range(2):
        optimizer.zero_grad()
        loss = base.encode(obs).square().mean()
        loss.backward()
        fusion_grad = base.encoder.relational_fusion.weight.grad
        assert torch.isfinite(fusion_grad).all() and fusion_grad[:, cfg.model.cond_dim:].abs().sum() > 0
        grads = [p.grad for p in base.encoder.object_feature_encoder.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        # Identity initialization zeros the branch gradient only on the first backward.
        if step:
            assert all(g.abs().sum() > 0 for g in grads)
        optimizer.step()
    # Exercise the actual Flow adapter as well; its zero-initialized head must
    # learn before a nonzero gradient can reach any encoder.
    optimizer = torch.optim.SGD(base.parameters(), lr=.05)
    for _ in range(5):
        optimizer.zero_grad()
        loss = adapter.compute_loss(obs, torch.randn(2, cfg.model.chunk_size, cfg.model.action_dim))
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
    assert base.encoder.object_feature_encoder[0].weight.grad.abs().sum() > 0
    q, v = CriticFeatureExtractor(cfg).eval(), CriticFeatureExtractor(cfg).eval()
    assert q.encoder.object_feature_encoder[0].weight is not v.encoder.object_feature_encoder[0].weight
    assert q(obs).shape == v(obs).shape == (2, cfg.model.cond_dim)
    cfg.env.num_points = 256
    live = dict(point_cloud=obs['pc'][0].numpy(), state=obs['state'][0].numpy(),
                object_features=obs['object_features'][0].numpy())
    live['object_features'][-2:] = 1
    norm = MinMaxNormalizer()
    encode = observation_encoder(cfg, base, norm, 'cpu')
    actual = encode(live)
    normalized = normalize_observation(dict(pc=live['point_cloud'], state=live['state'],
        object_features=live['object_features']), norm, np.asarray(cfg.env.workspace_bounds))
    expected = base.encode({k: torch.tensor(v)[None] for k, v in normalized.items()})
    torch.testing.assert_close(actual, expected)


def test_experiment_single_factor_and_comparison_guards():
    from evaluation.compare import comparison_protocol
    with initialize_config_dir(version_base=None, config_dir=str(Path(__file__).resolve().parents[1]/'configs')):
        a = compose(config_name='train_offline', overrides=['+experiment=oc_budget'])
        b = compose(config_name='train_offline', overrides=['+experiment=oc_budget_relational'])
    validate_observation_config(b.env)
    assert a.env.observation.mode == b.env.observation.mode == 'global_object_budget'
    assert not a.env.observation.relational_features and b.env.observation.relational_features
    a, b = [OmegaConf.to_container(c, resolve=True) for c in (a,b)]
    protocol = lambda c, allow: comparison_protocol(c, {}, .7, .0067, allow)
    assert protocol(a, False) != protocol(b, False)
    assert protocol(a, True) == protocol(b, True)
    for section, field, value in [('env', 'control_mode', 'other'), ('env', 'exec_steps', 1),
            ('env', 'workspace_bounds', [[0,0,0], [1,1,1]]), ('model', 'chunk_size', 8),
            ('model', 'action_dim', 8), ('model', 'num_inference_steps', 20)]:
        changed = copy.deepcopy(b)
        changed[section][field] = value
        assert protocol(a, True) != protocol(changed, True)


@pytest.mark.parametrize('mode,flag', [('object_centric', True), ('global_random', True),
                                       ('global_object_budget', 'true')])
def test_invalid_config_fails_early(tmp_path, mode, flag):
    cfg = enabled_config(tmp_path/'unused')
    cfg.env.observation.mode = mode
    cfg.env.observation.relational_features = flag
    with pytest.raises(ValueError, match='relational_features'):
        validate_observation_config(cfg.env)
