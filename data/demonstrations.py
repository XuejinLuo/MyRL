"""Strict H5 ingestion into the same primitive schema used by collection."""
from pathlib import Path
import h5py
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from data.episodes import SCHEMA, digest, save_episode, validate_episode
from data.pointcloud import preprocess_points
from utils.experiment import write_json


def load_demonstrations(cfg):
    path = Path(cfg.dataset.data_path).expanduser().resolve()
    episodes = []
    with h5py.File(path, 'r') as source:
        keys = list(source.keys())
        limit = cfg.dataset.max_episodes
        if limit is not None:
            if limit < 1:
                raise ValueError('dataset.max_episodes must be positive or null')
            keys = keys[:limit]
        for key in tqdm(keys, desc='Loading demonstrations'):
            g = source[key]
            for label in ('success', 'terminated', 'truncated'):
                if label not in g:
                    raise ValueError(f'{key}: missing {label}; success cannot be inferred from episode end')
            obs = g['obs']
            tcp = obs.get('extra', {}).get('tcp_pose')
            if tcp is None:
                tcp = obs['agent'].get('tcp_pose')
            if tcp is None:
                raise ValueError(f'{key}: missing tcp_pose')
            actions = g['actions'][:].astype(np.float32)
            t = len(actions)
            if not t:
                raise ValueError(f'{key}: empty trajectory')
            labels = {k: g[k][:].reshape(-1) for k in ('success','terminated','truncated')}
            if any(len(a) != t or not np.isin(a, [0, 1]).all() for a in labels.values()):
                raise ValueError(f'{key}: invalid transition labels')
            if any(len(a) != t+1 for a in (obs['agent']['qpos'], tcp, obs['pointcloud']['xyzw'])):
                raise ValueError(f'{key}: expected T+1 observations')
            # Historical demonstrations may continue after timeout. Preserve the
            # established conversion: first true terminal, current horizon, or EOF.
            stop = min(t, int(cfg.env.max_episode_steps))
            terminals = np.flatnonzero(labels['terminated'])
            if len(terminals):
                stop = min(stop, int(terminals[0])+1)
            cloud = obs['pointcloud']
            # Read once per trajectory instead of repeated HDF5 lookups/decompression per frame.
            xyzw_frames = cloud['xyzw'][:stop+1]
            rgb_frames = cloud['rgb'][:stop+1] if 'rgb' in cloud else None
            frames = []
            for i, xyzw in enumerate(xyzw_frames):
                valid = xyzw[..., 3] > 0
                rgb = rgb_frames[i][valid] / 255.0 if rgb_frames is not None else None
                frames.append(preprocess_points(xyzw[valid, :3], rgb,
                    cfg.env.workspace_bounds, cfg.env.num_points, cfg.env.use_color))
            ep = dict(pc=np.stack(frames),
                state=np.concatenate([obs['agent']['qpos'][:stop+1], tcp[:stop+1]], axis=-1).astype(np.float32),
                action=actions[:stop], success=labels['success'][:stop].astype(bool),
                terminated=labels['terminated'][:stop].astype(bool), truncated=np.zeros(stop, dtype=bool))
            ep['reward'] = ep['success'].astype(np.float32)
            ep['truncated'][-1] = not ep['terminated'][-1]
            validate_episode(ep)
            if ep['state'].shape[1] != cfg.model.state_dim or ep['action'].shape[1] != cfg.model.action_dim:
                raise ValueError(f'{key}: state/action dimensions differ from task profile')
            episodes.append(ep)
    if not episodes:
        raise ValueError('No demonstrations found')
    print(f'Loaded {len(episodes)} demonstrations from {path}', flush=True)
    return episodes


def export_demonstrations(episodes, cfg, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    items = []
    for index, ep in enumerate(tqdm(episodes, desc='Exporting demonstrations')):
        path = directory/f'episode_{index:06d}.npz'
        save_episode(path, ep)
        items.append(dict(path=path.name, sha256=digest(path)))
    write_json(directory/'manifest.json', dict(schema=SCHEMA, reward_mode='success',
        env=OmegaConf.to_container(cfg.env, resolve=True), sources=[dict(name='demonstrations',
        origin=str(Path(cfg.dataset.data_path).expanduser().resolve()), episodes=items)]))
