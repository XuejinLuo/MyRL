"""Object retention and H5/live parity without a ManiSkill installation."""
from pathlib import Path
from types import SimpleNamespace
import gymnasium as gym
import h5py
import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from data.pointcloud import preprocess_points, resolve_target_ids, validate_sampling
from data.demonstrations import load_demonstrations, export_demonstrations
from envs.maniskill_bridge import ManiSkillToRL100Wrapper
from envs.pointcloud_wrapper import PointCloudObservationWrapper

BOUNDS = [[-1, -1, -1], [1, 1, 1]]
SAMPLING = dict(mode='object_budget', objects=[
    dict(name='cubeA', h5_id=18, num_points=256),
    dict(name='cubeB', h5_id=19, num_points=256)])


def cloud(a=1000, b=20, background=3000):
    n = a+b+background
    xyz = np.zeros((n, 3), dtype=np.float32)
    xyz[:, 0] = np.linspace(-.9, .9, n)
    seg = np.repeat([18, 19, 0], [a, b, background])
    return xyz, seg


def sample(xyz, seg, n=1024, **kw):
    return preprocess_points(xyz, None, BOUNDS, n, False, sampling=SAMPLING,
        segmentation=seg, rng=np.random.default_rng(42), return_indices=True, **kw)


def test_sparse_target_keeps_every_point_and_no_source_is_duplicated():
    xyz, seg = cloud()
    points, idx = sample(xyz, seg)
    assert points.shape == (1024, 3) and points.dtype == np.float32
    assert len(np.unique(idx)) == 1024
    assert np.sum(seg[idx] == 18) >= 256
    assert np.sum(seg[idx] == 19) == 20
    assert set(np.flatnonzero(seg == 19)) <= set(idx)
    # Combined output is shuffled, not emitted as object-ordered blocks.
    assert not np.all(seg[idx[:256]] == 18)


@pytest.mark.parametrize('a,b,background', [(800, 800, 2000), (1000, 0, 3000), (0, 0, 3000), (2, 1, 5)])
def test_missing_objects_and_undersized_clouds(a, b, background):
    xyz, seg = cloud(a, b, background)
    points, idx = sample(xyz, seg)
    assert len(points) == 1024 and np.isfinite(points).all()
    unique = np.unique(idx)
    assert np.sum(seg[unique] == 18) >= min(a, 256)
    assert np.sum(seg[unique] == 19) >= min(b, 256)
    assert len(unique) == min(len(xyz), 1024)


def test_filtering_keeps_segmentation_rgb_and_source_indices_aligned():
    xyz, seg = cloud()
    rgb = np.zeros_like(xyz)
    xyz[0, 0] = np.nan
    xyz[1, 0] = 10
    rgb[2, 0] = np.nan
    rgb[:, 1] = np.arange(len(xyz))
    points, idx = preprocess_points(xyz, rgb, BOUNDS, 1024, True,
        sampling=SAMPLING, segmentation=seg[:, None],
        rng=np.random.default_rng(1), return_indices=True)
    assert not {0, 1, 2} & set(idx)
    np.testing.assert_array_equal(points[:, 4], idx)
    np.testing.assert_array_equal(points[:, :3], xyz[idx])
    assert np.sum(seg[idx] == 19) == 20


def test_empty_cloud_and_missing_segmentation_are_distinct():
    points, idx = sample(np.empty((0, 3)), np.empty(0))
    assert not points.any() and np.all(idx == -1)
    with pytest.raises(ValueError, match='requires pointcloud segmentation'):
        preprocess_points(np.empty((0, 3)), None, BOUNDS, 1024, False, sampling=SAMPLING)


def test_random_mode_preserves_historical_sampling_and_needs_no_segmentation():
    xyz, _ = cloud()
    np.random.seed(37)
    expected = xyz[np.random.choice(len(xyz), 1024, replace=False)]
    np.random.seed(37)
    actual = preprocess_points(xyz, None, BOUNDS, 1024, False)
    np.testing.assert_array_equal(actual, expected)
    np.random.seed(37)
    explicit = preprocess_points(xyz, None, BOUNDS, 1024, False,
        sampling=dict(SAMPLING, mode='random'))
    np.testing.assert_array_equal(explicit, expected)


@pytest.mark.parametrize('change', [
    {'mode': 'unknown'}, {'objects': []},
    {'objects': [dict(name='cubeA', h5_id=18, num_points=2048)]},
    {'objects': [dict(name='cubeA', h5_id=18, num_points=0)]},
    {'objects': [dict(name='cubeA', h5_id=18, num_points=1), dict(name='cubeB', h5_id=18, num_points=1)]},
])
def test_invalid_sampling_config_fails(change):
    with pytest.raises(ValueError):
        validate_sampling(dict(SAMPLING, **change), 1024)


class RawEnv(gym.Env):
    def __init__(self, obs):
        self.obs = obs
        self.observation_space = gym.spaces.Dict({})
        self.action_space = gym.spaces.Box(-1., 1., (1,))
        self.segmentation_id_map = {71: SimpleNamespace(name='cubeA'), 92: SimpleNamespace(name='cubeB')}
    def reset(self, seed=None, options=None):
        return self.obs, {}
    def step(self, action):
        return self.obs, 0., False, False, {}


def make_h5(path, segmentation=True):
    xyz, seg = cloud()
    xyzw = np.column_stack([xyz, np.ones(len(xyz), dtype=np.float32)])
    xyzw[0, 3] = 0
    rgb = np.tile(np.array([[12, 34, 56]], dtype=np.uint8), (len(xyz), 1))
    with h5py.File(path, 'w') as f:
        g = f.create_group('traj_0')
        g['obs/pointcloud/xyzw'] = np.stack([xyzw, xyzw])
        g['obs/pointcloud/rgb'] = np.stack([rgb, rgb])
        if segmentation:
            g['obs/pointcloud/segmentation'] = np.stack([seg[:, None], seg[:, None]])
        g['obs/agent/qpos'] = np.zeros((2, 9), dtype=np.float32)
        g['obs/extra/tcp_pose'] = np.zeros((2, 7), dtype=np.float32)
        g['actions'] = np.zeros((1, 7), dtype=np.float32)
        g['success'], g['terminated'], g['truncated'] = [True], [True], [False]
    live_seg = np.where(seg == 18, 71, np.where(seg == 19, 92, 0))
    return dict(pointcloud=dict(xyzw=xyzw, rgb=rgb, segmentation=live_seg[:, None]),
        agent=dict(qpos=np.zeros(9)), extra=dict(tcp_pose=np.zeros(7)))


def config(path):
    with initialize_config_dir(version_base=None, config_dir=str(Path(__file__).resolve().parents[1]/'configs')):
        cfg = compose(config_name='train_offline')
    cfg.dataset.data_path = str(path)
    cfg.env.workspace_bounds = BOUNDS
    cfg.env.sampling = OmegaConf.create(SAMPLING)
    return cfg


def test_h5_live_reset_and_step_use_identical_sampling_with_different_ids(tmp_path):
    path = tmp_path/'source.h5'
    obs = make_h5(path)
    cfg = config(path)
    np.random.seed(7)
    ep, = load_demonstrations(cfg)
    raw = RawEnv(obs)
    wrapped = PointCloudObservationWrapper(ManiSkillToRL100Wrapper(raw, sampling=cfg.env.sampling),
        workspace_bounds=BOUNDS, use_color=True, sampling=cfg.env.sampling)
    np.random.seed(7)
    first, _ = wrapped.reset()
    second, *_ = wrapped.step(np.zeros(1))
    np.testing.assert_array_equal(first['point_cloud'], ep['pc'][0])
    np.testing.assert_array_equal(second['point_cloud'], ep['pc'][1])
    np.testing.assert_array_equal(first['state'], ep['state'][0])
    # Reset/reconfiguration changes the IDs, but must not change the selected geometry.
    raw.obs['pointcloud']['segmentation'] = np.where(obs['pointcloud']['segmentation'] == 71,
        101, np.where(obs['pointcloud']['segmentation'] == 92, 202, 0))
    raw.segmentation_id_map = {101: SimpleNamespace(name='cubeA'), 202: SimpleNamespace(name='cubeB')}
    np.random.seed(7)
    remapped, _ = wrapped.reset()
    np.testing.assert_array_equal(remapped['point_cloud'], ep['pc'][0])
    export_demonstrations([ep], cfg, tmp_path/'exported')
    from data.episodes import load_sources
    exported, spec = load_sources(tmp_path/'exported/manifest.json')
    assert spec['env']['sampling']['mode'] == 'object_budget'
    np.testing.assert_array_equal(exported[0]['pc'], ep['pc'])


def test_h5_and_live_missing_segmentation_fail_loudly(tmp_path):
    path = tmp_path/'source.h5'
    obs = make_h5(path, segmentation=False)
    cfg = config(path)
    with pytest.raises(ValueError, match='requires H5'):
        load_demonstrations(cfg)
    del obs['pointcloud']['segmentation']
    wrapper = ManiSkillToRL100Wrapper(RawEnv(obs), sampling=cfg.env.sampling)
    with pytest.raises(ValueError, match='requires live'):
        wrapper.reset()
    with pytest.raises(ValueError, match='Expected one'):
        resolve_target_ids(SAMPLING['objects'], {1: SimpleNamespace(name='other')})


def test_diagnostic_compares_production_samplers(tmp_path, capsys):
    from tools.diagnostics.check_point_sampling import inspect_sampling
    path = tmp_path/'source.h5'
    make_h5(path)
    counts = inspect_sampling(config(path), max_episodes=1, frame_step=1)
    assert counts['before']['cubeB'] == [20, 20]
    assert counts['object_budget']['cubeB'] == [20, 20]
    assert max(counts['random']['cubeB']) < 20
    assert 'visible-before but absent-after=0/2' in capsys.readouterr().out
