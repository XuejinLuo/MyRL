"""Geometric coverage, RGB alignment and the real H5/live FPS input contract."""
import builtins
import numpy as np
import pytest
from data.pointcloud import preprocess_points, require_fps
from tests.test_point_sampling import BOUNDS, RawEnv, config, make_h5
from data.demonstrations import load_demonstrations, export_demonstrations
from data.episodes import load_sources
from envs.maniskill_bridge import ManiSkillToRL100Wrapper
from envs.pointcloud_wrapper import PointCloudObservationWrapper


def sample(xyz, n, rgb=None, **kwargs):
    return preprocess_points(xyz, rgb, BOUNDS, n, rgb is not None,
                             sampling={'mode': 'fps'}, return_indices=True, **kwargs)


def test_spatially_separate_sparse_target_survives_dense_background():
    # One isolated surface sample surrounded by thousands of background points.
    # This is a coverage regression, not a claim about all target geometries.
    rng = np.random.default_rng(42)
    xyz = np.vstack([rng.uniform(-.1, .1, (4000, 3)), [.8, .8, .8]]).astype('f4')
    _, idx = sample(xyz, 32)
    assert 4000 in idx
    assert len(np.unique(idx)) == 32


def test_fps_uses_xyz_only_and_preserves_rgb_filter_alignment():
    xyz = np.random.default_rng(1).uniform(-.9, .9, (300, 3)).astype('f4')
    xyz[0, 0], xyz[1, 0] = np.nan, 5
    rgb = np.repeat(np.arange(300, dtype='f4')[:, None], 3, axis=1)
    rgb[2] = np.nan
    pc, idx = sample(xyz, 64, rgb)
    assert not {0, 1, 2} & set(idx)
    np.testing.assert_array_equal(pc[:, :3], xyz[idx])
    np.testing.assert_array_equal(pc[:, 3:], rgb[idx])
    other_rgb = -rgb * 100
    _, other_idx = sample(xyz, 64, other_rgb)
    np.testing.assert_array_equal(idx, other_idx)


@pytest.mark.parametrize('n', [1, 2, 3, 8, 16])
def test_duplicate_positions_and_padding(n):
    unique = np.array([[-.9, 0, 0], [0, 0, 0], [.9, 0, 0]], dtype='f4')
    xyz = np.repeat(unique, [15, 1, 2], axis=0)
    pc, idx = sample(xyz, n)
    assert pc.shape == (n, 3) and pc.dtype == np.float32
    assert len(np.unique(pc, axis=0)) == min(n, 3)
    np.testing.assert_array_equal(pc, xyz[idx])


def test_empty_and_single_position_cloud():
    pc, idx = sample(np.empty((0, 3)), 32)
    assert not pc.any() and np.all(idx == -1)
    pc, idx = sample(np.full((100, 3), .2), 32)
    np.testing.assert_allclose(pc, .2)
    assert np.all(idx == 0)


def test_fps_is_deterministic_without_consuming_training_rng():
    xyz = np.random.default_rng(2).uniform(-.9, .9, (300, 3))
    np.random.seed(3)
    expected = np.random.random(5)
    np.random.seed(3)
    a, _ = sample(xyz, 32)
    np.testing.assert_array_equal(np.random.random(5), expected)
    np.random.seed(999)
    b, _ = sample(xyz, 32)
    np.testing.assert_array_equal(a, b)


def test_missing_backend_has_actionable_error(monkeypatch):
    real_import = builtins.__import__
    def without_fps(name, *args, **kwargs):
        if name == 'fpsample':
            raise ImportError('not installed')
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', without_fps)
    with pytest.raises(ImportError, match='requirements-pointcloud.txt'):
        require_fps()


def test_h5_live_and_export_agree_without_segmentation(tmp_path):
    path = tmp_path / 'source.h5'
    obs = make_h5(path, segmentation=False)
    del obs['pointcloud']['segmentation']
    cfg = config(path)
    cfg.env.sampling.mode = 'fps'  # Stored object IDs must be ignored.
    ep, = load_demonstrations(cfg)
    raw = RawEnv(obs)
    del raw.segmentation_id_map
    live = PointCloudObservationWrapper(
        ManiSkillToRL100Wrapper(raw, sampling=cfg.env.sampling),
        workspace_bounds=BOUNDS, use_color=True, sampling=cfg.env.sampling)
    first, _ = live.reset()
    second, *_ = live.step(np.zeros(1))
    np.testing.assert_array_equal(first['point_cloud'], ep['pc'][0])
    np.testing.assert_array_equal(second['point_cloud'], ep['pc'][1])
    export_demonstrations([ep], cfg, tmp_path / 'exported')
    loaded, spec = load_sources(tmp_path / 'exported/manifest.json')
    assert spec['env']['sampling']['mode'] == 'fps'
    np.testing.assert_array_equal(loaded[0]['pc'], ep['pc'])


def test_comparison_works_with_default_fps_config(tmp_path, capsys):
    from tools.diagnostics.check_point_sampling import inspect_sampling
    path = tmp_path / 'source.h5'
    make_h5(path)
    cfg = config(path)
    cfg.env.sampling.mode = 'fps'
    counts = inspect_sampling(cfg, max_episodes=1, frame_step=1)
    assert set(counts) == {'before', 'random', 'fps', 'object_budget'}
    assert len(counts['fps']['cubeB']) == 2
    assert 'fps' in capsys.readouterr().out
