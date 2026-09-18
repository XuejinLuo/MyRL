"""Compare random/object budgets on the same H5 frames using production code."""
import argparse
from pathlib import Path
import h5py
import numpy as np
from hydra import compose, initialize_config_dir
from data.pointcloud import preprocess_points, validate_sampling


def inspect_sampling(cfg, max_episodes=100, frame_step=5, seed=42):
    if max_episodes < 1 or frame_step < 1:
        raise ValueError('max_episodes and frame_step must be positive')
    objects = validate_sampling(cfg.env.sampling, cfg.env.num_points)
    if not objects:
        raise ValueError('Configure env.sampling.mode=object_budget for this comparison')
    modes = ('before', 'random', 'object_budget')
    counts = {mode: {o['name']: [] for o in objects} for mode in modes}
    # Independent RNGs keep baseline draws unaffected by the alternative sampler.
    rngs = {mode: np.random.default_rng(seed) for mode in modes[1:]}
    total, episodes = [], 0
    with h5py.File(Path(cfg.dataset.data_path).expanduser(), 'r') as f:
        for name in list(f.keys())[:max_episodes]:
            cloud = f[name]['obs/pointcloud']
            if 'segmentation' not in cloud:
                raise ValueError(f'{name}: missing segmentation')
            episodes += 1
            # Match the training loader's terminal/horizon clipping.
            stop = min(len(f[name]['actions']), int(cfg.env.max_episode_steps))
            terminals = np.flatnonzero(f[name]['terminated'][:].reshape(-1))
            if len(terminals):
                stop = min(stop, int(terminals[0])+1)
            for t in range(0, stop+1, frame_step):
                xyzw = np.asarray(cloud['xyzw'][t]).reshape(-1, 4)
                valid = xyzw[:, 3] > 0
                xyz = xyzw[valid, :3]
                rgb = (np.asarray(cloud['rgb'][t]).reshape(-1, 3)[valid]/255.
                       if 'rgb' in cloud else None)
                seg = np.asarray(cloud['segmentation'][t]).reshape(-1)[valid]
                # Same finite/workspace mask as preprocessing, without sampling.
                mask = np.isfinite(xyz).all(axis=-1)
                if cfg.env.use_color and rgb is not None:
                    mask &= np.isfinite(rgb).all(axis=-1)
                lo, hi = np.asarray(cfg.env.workspace_bounds)
                mask &= ((xyz >= lo) & (xyz <= hi)).all(axis=-1)
                before_idx = np.flatnonzero(mask)
                total.append(len(before_idx))
                for obj in objects:
                    counts['before'][obj['name']].append(int(np.sum(seg[before_idx] == obj['h5_id'])))
                for mode in modes[1:]:
                    sampling = cfg.env.sampling if mode == 'object_budget' else {'mode': 'random'}
                    _, idx = preprocess_points(xyz, rgb, cfg.env.workspace_bounds,
                        cfg.env.num_points, cfg.env.use_color, sampling=sampling,
                        segmentation=seg, rng=rngs[mode], return_indices=True)
                    # Count distinct source points, never inflated padding repeats.
                    idx = np.unique(idx[idx >= 0])
                    for obj in objects:
                        counts[mode][obj['name']].append(int(np.sum(seg[idx] == obj['h5_id'])))
    if not total:
        raise ValueError('No frames to inspect')
    print(f'episodes={episodes} frames={len(total)} sample_size={cfg.env.num_points} seed={seed}')
    print('Counts are unique source points; before means after valid/finite/workspace filtering.')
    print(f'Cropped source points: mean={np.mean(total):.2f}, min={min(total)}')
    print('object    mode             mean  median   p05   zero%    <10%')
    for obj in objects:
        name = obj['name']
        print(f"Target {name}: H5 ID={obj['h5_id']}, reserved budget={obj['num_points']}")
        for mode in modes:
            a = np.asarray(counts[mode][name])
            print(f'{name:9s} {mode:14s} {a.mean():7.2f} {np.median(a):7.1f} '
                  f'{np.percentile(a, 5):5.1f} {100*np.mean(a == 0):7.2f} {100*np.mean(a < 10):7.2f}')
        before = np.asarray(counts['before'][name])
        for mode in modes[1:]:
            after = np.asarray(counts[mode][name])
            lost = np.sum((before > 0) & (after == 0))
            print(f'  {mode}: visible-before but absent-after={lost}/{len(before)} frames')
        required = np.minimum(before, obj['num_points'])
        if np.any(np.asarray(counts['object_budget'][name]) < required):
            raise AssertionError(f'{name}: reserved unique-point budget was not retained')
        if not before.any():
            print(f'  WARNING: {name} never visible. Verify H5 segmentation ID, crop and camera.')
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', help='Defaults to the shared task config H5 path')
    parser.add_argument('--max-episodes', type=int, default=100)
    parser.add_argument('--frame-step', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    with initialize_config_dir(version_base=None, config_dir=str(Path(__file__).resolve().parents[2]/'configs')):
        cfg = compose(config_name='config')
    if args.data_path:
        cfg.dataset.data_path = args.data_path
    inspect_sampling(cfg, args.max_episodes, args.frame_step, args.seed)


if __name__ == '__main__':
    main()
