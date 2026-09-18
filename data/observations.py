"""Representation routing and normalization shared by data and rollouts."""
import numpy as np
from data.object_centric import OBJECT_FIELDS, validate_object_arrays, validate_object_config
from data.pointcloud import validate_sampling


def observation_mode(env):
    mode = (env.get('observation') or {}).get('mode')
    if mode is None:  # Preserve historical checkpoint configuration and RNG behavior.
        return 'global_' + (env.get('sampling') or {}).get('mode', 'random')
    if mode not in ('global_random', 'global_object_budget', 'object_centric'):
        raise ValueError(f'Unknown observation mode: {mode}')
    return mode


def sampling_config(env):
    mode = observation_mode(env)
    if mode == 'object_centric':
        return None
    return dict(mode=mode.removeprefix('global_'),
                objects=list((env.get('sampling') or {}).get('objects', [])))


def validate_observation_config(env):
    if observation_mode(env) == 'object_centric':
        return validate_object_config(env['observation'])
    return validate_sampling(sampling_config(env), env['num_points'])


def observation_fields(obs):
    return (*OBJECT_FIELDS, 'state') if 'object_points' in obs else ('pc', 'state')


def episode_observation(obs):
    if 'object_points' in obs:
        return {k: obs[k] for k in (*OBJECT_FIELDS, 'state')}
    return dict(pc=obs['point_cloud'], state=obs['state'])


def normalize_observation(obs, normalizer, bounds):
    # Always copy, including masks/roles, so dataset views cannot mutate episodes.
    result = {k: np.array(v, copy=True, dtype=(np.bool_ if 'mask' in k or k == 'object_valid'
                  else np.int64 if k == 'object_roles' else np.float32)) for k, v in obs.items()}
    result['state'] = normalizer.normalize(result['state'], 'state')
    if 'object_points' not in result:
        result['pc'] = normalizer.center_point_cloud(result['pc'], bounds)
        return result
    center = np.asarray(bounds, dtype=np.float32).mean(0)
    result['object_centers'] = np.where(result['object_valid'][..., None],
                                       result['object_centers'] - center, 0.)
    ctx = result['context_points']
    ctx[..., :3] = np.where(result['context_point_mask'][..., None], ctx[..., :3] - center, 0.)
    # Local object XYZ and metric extents require no further centering.
    return result


def validate_observation(obs, cfg):
    if observation_mode(cfg.env) == 'object_centric':
        if 'pc' in obs or 'point_cloud' in obs:
            raise ValueError('Object-centric mode cannot contain a global cloud')
        validate_object_arrays(obs, cfg.env.observation, cfg.model.in_channels)
    elif 'object_points' in obs or np.shape(obs['pc']) != (cfg.env.num_points, cfg.model.in_channels):
        raise ValueError('Point cloud shape does not match config')
    if np.shape(obs['state']) != (cfg.model.state_dim,):
        raise ValueError('State dimension mismatch')
    if any(not np.isfinite(v).all() for v in obs.values()):
        raise ValueError('Nonfinite observation')


def batch_observation(batch, prefix=''):
    fields = (*OBJECT_FIELDS, 'state') if prefix + 'object_points' in batch else ('pc', 'state')
    return {k: batch[prefix + k] for k in fields}
