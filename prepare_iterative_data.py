"""Convert original ManiSkill H5 to the explicit iterative primitive schema."""
import argparse
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
from data.iterative_store import SCHEMA, digest, save_episode, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--h5', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--stats', required=True)
    p.add_argument('--max-episodes', type=int, default=None)
    p.add_argument('overrides', nargs='*')
    args = p.parse_args()
    with initialize_config_dir(version_base=None, config_dir=str(Path(__file__).parent.resolve()/'configs')):
        cfg = compose(config_name='train_iterative', overrides=args.overrides)
    import h5py
    from data.maniskill_dataset import load_maniskill_h5
    np.random.seed(cfg.seed)
    # Validate labels BEFORE invoking the historical demonstration loader.
    with h5py.File(args.h5, 'r') as f:
        for key in list(f)[:args.max_episodes]:
            g = f[key]
            if 'success' not in g or 'terminated' not in g or 'truncated' not in g:
                raise ValueError(f'{key}: requires success/terminated/truncated; do not infer success from episode end')
            if 'tcp_pose' not in g['obs'].get('extra', {}) and 'tcp_pose' not in g['obs']['agent']:
                raise ValueError(f'{key}: tcp_pose missing')
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    raw = load_maniskill_h5(args.h5, max_episodes=args.max_episodes, workspace_bounds=cfg.env.workspace_bounds, n_points=cfg.env.num_points)
    items = []
    with h5py.File(args.h5, 'r') as f:
        for index, (key, ep) in enumerate(zip(f.keys(), raw)):
            g = f[key]
            success = g['success'][:].reshape(-1).astype(bool)
            terminated = g['terminated'][:].reshape(-1).astype(bool)
            raw_truncated = g['truncated'][:].reshape(-1).astype(bool)

            t = len(ep['action'])
            if t == 0:
                raise ValueError(f'{key}: empty trajectory')

            for name, values in (
                ('success', success),
                ('terminated', terminated),
                ('truncated', raw_truncated),
            ):
                if len(values) != t:
                    raise ValueError(f'{key}: {name} length differs from actions')

            if len(ep['pc']) != t + 1 or len(ep['state']) != t + 1:
                raise ValueError(f'{key}: expected T+1 observations')

            # Align imported demonstrations with the current environment horizon.
            horizon = int(cfg.env.max_episode_steps)
            if horizon < 1:
                raise ValueError('max_episode_steps must be positive')

            stop = min(t, horizon)

            # Keep the first genuine terminal transition and discard later actions.
            terminal_indices = np.flatnonzero(terminated)
            if terminal_indices.size:
                stop = min(stop, int(terminal_indices[0]) + 1)

            ep['pc'] = ep['pc'][:stop + 1]
            ep['state'] = ep['state'][:stop + 1]
            ep['action'] = ep['action'][:stop]

            ep['success'] = success[:stop].copy()
            ep['reward'] = ep['success'].astype(np.float32)
            ep['terminated'] = terminated[:stop].copy()

            # Historical timeout flags may remain true during continued recording.
            # Rebuild truncation for the current horizon or recording boundary.
            ep['truncated'] = np.zeros(stop, dtype=bool)
            if not ep['terminated'][-1]:
                ep['truncated'][-1] = True

            if index < 5:
                print(
                    f'{key}: actions {t} -> {stop}, '
                    f'terminated={bool(ep["terminated"][-1])}, '
                    f'truncated={bool(ep["truncated"][-1])}, '
                    f'success_any={bool(ep["success"].any())}',
                    flush=True,
                )
            if not cfg.env.use_color:
                ep['pc'] = ep['pc'][..., :3]
            file = out/f'episode_{index:06d}.npz'
            save_episode(file, ep)
            items.append(dict(path=file.name, sha256=digest(file)))
    import shutil
    shutil.copyfile(args.stats, out/'dataset_stats.json')
    write_json(out/'manifest.json', dict(schema=SCHEMA, reward_mode='success',
        env=OmegaConf.to_container(cfg.env, resolve=True),
        sources=[dict(name='demonstrations', origin=str(Path(args.h5).resolve()), episodes=items)]))
    print(out/'manifest.json')


if __name__ == '__main__':
    main()
