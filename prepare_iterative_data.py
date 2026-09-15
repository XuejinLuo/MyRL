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
    p.add_argument('overrides', nargs='*')
    args = p.parse_args()
    with initialize_config_dir(version_base=None, config_dir=str(Path(__file__).parent.resolve()/'configs')):
        cfg = compose(config_name='train_iterative', overrides=args.overrides)
    import h5py
    from data.maniskill_dataset import load_maniskill_h5
    np.random.seed(cfg.seed)
    # Validate labels BEFORE invoking the historical demonstration loader.
    with h5py.File(args.h5, 'r') as f:
        for key in f:
            g = f[key]
            if 'success' not in g or 'terminated' not in g or 'truncated' not in g:
                raise ValueError(f'{key}: requires success/terminated/truncated; do not infer success from episode end')
            if 'tcp_pose' not in g['obs'].get('extra', {}) and 'tcp_pose' not in g['obs']['agent']:
                raise ValueError(f'{key}: tcp_pose missing')
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    raw = load_maniskill_h5(args.h5, workspace_bounds=cfg.env.workspace_bounds, n_points=cfg.env.num_points)
    items = []
    with h5py.File(args.h5, 'r') as f:
        for index, (key, ep) in enumerate(zip(f.keys(), raw)):
            g = f[key]
            ep['success'] = g['success'][:].reshape(-1).astype(bool)
            ep['reward'] = ep['success'].astype(np.float32)
            ep['terminated'] = g['terminated'][:].reshape(-1).astype(bool)
            ep['truncated'] = g['truncated'][:].reshape(-1).astype(bool)
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
