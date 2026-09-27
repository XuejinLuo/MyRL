"""Failure mining records and verified replay, without simulator state teleportation."""
import json
from pathlib import Path

import numpy as np

from data.episodes import digest


def state_vector(state):
    """Stable schema + numeric state; includes actors and articulation velocities."""
    schema, values = [], []
    def visit(value, path):
        if isinstance(value, dict):
            for key in sorted(value):
                visit(value[key], path+[str(key)])
        else:
            if hasattr(value, 'detach'):
                value = value.detach().cpu().numpy()
            value = np.asarray(value, dtype=np.float64)
            if not np.isfinite(value).all():
                raise ValueError('Nonfinite simulator state')
            schema.append(dict(path=path, shape=list(value.shape)))
            values.append(value.reshape(-1))
    visit(state, [])
    if not values:
        raise ValueError('Empty simulator state')
    return schema, np.concatenate(values)


def check_replay_state(state, schema, expected, step):
    actual_schema, actual = state_vector(state)
    if actual_schema != schema or actual.shape != expected.shape or not np.allclose(actual, expected, atol=1e-5, rtol=0.):
        raise ValueError(f'Failure replay diverged at step {step}; episode excluded. '
                         'Use the same simulator/assets/device, or screen again; no state was teleported.')


def takeover_step(events, steps, horizon, reserve=180, rewind=20, override=None):
    if steps < 1 or not 1 <= reserve <= horizon or rewind < 0:
        raise ValueError('Invalid failure recovery budget')
    limit = min(steps-1, horizon-reserve)
    hints = [e['step']+1 for e in events if e['hints']]
    candidate = max(0, hints[0]-rewind) if hints else limit
    if override is not None:
        if not 0 <= override <= limit:
            raise ValueError(f'takeover-step must be in [0, {limit}] for this failure')
        candidate = override
    return min(candidate, limit)


def session_file(session, relative):
    session = Path(session).resolve()
    path = (session/relative).resolve()
    if not path.is_relative_to(session):
        raise ValueError('File path leaves session directory')
    return path


def load_queue(path, checkpoint_sha, config, normalizer, sampler, excluded):
    path = Path(path).resolve()
    queue = json.loads(path.read_text())
    if queue.get('schema') != 'myrl_failure_queue_v1':
        raise ValueError('Unsupported failure queue')
    if queue['checkpoint_sha256'] != checkpoint_sha or queue['normalizer'] != normalizer or queue['sampler'] != sampler:
        raise ValueError('Failure queue checkpoint/normalizer/sampler mismatch')
    for key in ('env', 'model'):
        if queue['config'][key] != config[key]:
            raise ValueError(f'Failure queue {key} mismatch')
    if set(queue['seeds']) & set(excluded):
        raise ValueError('Screening seeds overlap evaluation/excluded seeds')
    seen = set()
    for item in queue['failures']:
        seed = item['seed']
        if seed in seen or seed not in queue['seeds'] or item['success_any']:
            raise ValueError('Invalid/duplicate failed seed')
        seen.add(seed)
        trace = session_file(path.parent, item['path'])
        if digest(trace) != item['sha256']:
            raise ValueError('Failure trace changed since screening')
        with np.load(trace, allow_pickle=False) as data:
            actions, states = data['actions'], data['states']
            n = len(actions)
            if actions.shape != (n, 7) or states.ndim != 2 or len(states) != n+1 or not np.isfinite(actions).all() or not np.isfinite(states).all():
                raise ValueError('Invalid failure replay arrays')
            if not 0 <= item['takeover_step'] < n:
                raise ValueError('Invalid failure takeover step')
    return queue
