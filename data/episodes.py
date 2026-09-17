"""Versioned primitive trajectories; physical actions and uncentered observations."""
import hashlib
import json
from pathlib import Path
import numpy as np

SCHEMA = 'myrl_primitive_v1'
FIELDS = ('pc', 'state', 'action', 'reward', 'success', 'terminated', 'truncated')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


from utils.experiment import write_json


def validate_episode(ep):
    t = len(ep['action'])
    if t < 1:
        raise ValueError('Empty episode')
    for key in FIELDS:
        a = np.asarray(ep[key])
        if len(a) != t + (key in ('pc', 'state')) or not np.isfinite(a).all():
            raise ValueError(f'Invalid {key}: expected T+1 observations, T transitions')
    if ep['pc'].ndim != 3 or ep['state'].ndim != 2 or ep['action'].ndim != 2:
        raise ValueError('Invalid observation/action dimensions')
    for key in ('reward', 'success', 'terminated', 'truncated'):
        if ep[key].shape != (t,):
            raise ValueError(f'{key} must be one dimensional')
    for key in ('success', 'terminated', 'truncated'):
        if not np.isin(ep[key], [0, 1]).all():
            raise ValueError(f'{key} must be boolean')
    if not np.array_equal(ep['reward'], ep['success'].astype(np.float32)):
        raise ValueError('Sparse success rewards must match success flags')
    ends = ep['terminated'].astype(bool) | ep['truncated'].astype(bool)
    if ends[:-1].any() or not ends[-1]:
        raise ValueError('Episode must end exactly once at its final transition')


def save_episode(path, ep):
    validate_episode(ep)
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **{k: ep[k] for k in FIELDS})
    tmp.replace(path)


def load_episode(path):
    with np.load(path, allow_pickle=False) as f:
        ep = {k: f[k] for k in FIELDS}
    validate_episode(ep)
    return ep


def load_sources(manifest):
    manifest = Path(manifest).resolve()
    spec = json.loads(manifest.read_text())
    if spec['schema'] != SCHEMA or spec['reward_mode'] != 'success':
        raise ValueError('Unsupported dataset schema/reward mode')
    episodes, seen = [], set()
    for source in spec['sources']:
        for item in source['episodes']:
            path = (manifest.parent / item['path']).resolve()
            actual = digest(path)
            if actual != item['sha256'] or actual in seen:
                raise ValueError(f'Changed or duplicate episode: {path}')
            seen.add(actual)
            episodes.append(load_episode(path))
    if not episodes:
        raise ValueError('No episodes in dataset')
    return episodes, spec


def transition(ep, start, chunk_size, exec_steps, gamma):
    """Prefix Bellman target; timeout bootstraps from the real final observation."""
    stop = min(start + exec_steps, len(ep['action']))
    length = stop - start
    chunk = ep['action'][start:start + chunk_size]
    if len(chunk) < chunk_size:
        chunk = np.concatenate([chunk, np.repeat(chunk[-1:], chunk_size-len(chunk), axis=0)])
    return dict(pc=ep['pc'][start], state=ep['state'][start], action_chunk=chunk,
                next_pc=ep['pc'][stop], next_state=ep['state'][stop],
                reward=np.float32(np.dot(gamma ** np.arange(length), ep['reward'][start:stop])),
                done=np.float32(ep['terminated'][stop-1]), discount=np.float32(gamma ** length))
