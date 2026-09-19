"""Segmentation to independent, masked object clouds (no simulator dependency).

XYZ is in metres. Object points are local to the mean of ALL visible, filtered
points; centers and context remain in the incoming robot/world frame. Centers
are visible-surface centroids, not privileged simulator object poses.
"""
import numbers
import numpy as np

ROLES = {'manipulated': 0, 'target': 1, 'tool': 2, 'obstacle': 3, 'other': 4}
OBJECT_FIELDS = ('object_points', 'object_point_mask', 'object_centers',
                 'object_extents', 'object_valid', 'object_roles',
                 'context_points', 'context_point_mask')


def validate_object_config(config):
    objects = list(config.get('objects', []))
    if not objects:
        raise ValueError('object_centric requires configured objects')
    names, ids = set(), set()
    for obj in objects:
        name, sid, budget = obj.get('name'), obj.get('h5_id'), obj.get('max_points')
        if not isinstance(name, str) or not name or name in names:
            raise ValueError('Object names must be nonempty and unique')
        if (not isinstance(sid, numbers.Integral) or isinstance(sid, bool)
                or sid <= 0 or sid in ids):
            raise ValueError('Object h5_id values must be unique positive integers')
        if not isinstance(budget, numbers.Integral) or isinstance(budget, bool) or budget < 1:
            raise ValueError('Object max_points must be a positive integer')
        if obj.get('role') not in ROLES:
            raise ValueError(f'Object role must be one of {tuple(ROLES)}')
        names.add(name)
        ids.add(sid)
    count = config.get('context_points', 512)
    if not isinstance(count, numbers.Integral) or isinstance(count, bool) or count < 1:
        raise ValueError('context_points must be a positive integer')
    if config.get('encoder_variant', 'points') not in ('points', 'centers'):
        raise ValueError('encoder_variant must be points or centers')
    return objects


def object_shapes(config, channels):
    objects = validate_object_config(config)
    k, m = len(objects), max(o['max_points'] for o in objects)
    n = config.get('context_points', 512)
    return dict(object_points=(k, m, channels), object_point_mask=(k, m),
                object_centers=(k, 3), object_extents=(k, 3), object_valid=(k,),
                object_roles=(k,), context_points=(n, channels), context_point_mask=(n,))


def build_object_observation(xyz, rgb, segmentation, bounds, config, use_color=True,
                             *, target_ids=None, rng=None):
    objects = validate_object_config(config)
    rng = np.random if rng is None else rng
    xyz = np.asarray(xyz, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError('xyz must have shape (N, 3)')
    if segmentation is None:
        raise ValueError('object_centric requires pointcloud segmentation; no fallback')
    seg = np.asarray(segmentation).reshape(-1)
    if len(seg) != len(xyz) or not np.isfinite(seg).all() or not np.equal(seg, np.floor(seg)).all():
        raise ValueError('segmentation must contain one finite integer ID per point')
    if use_color:
        if rgb is None:
            raise ValueError('object_centric with use_color requires RGB')
        rgb = np.asarray(rgb, dtype=np.float32)
        if rgb.shape != xyz.shape:
            raise ValueError('rgb must match xyz shape')
        points = np.concatenate([xyz, rgb], axis=-1)
    else:
        points = xyz.copy()
    valid = np.isfinite(points).all(-1)
    if bounds is not None:
        bounds = np.asarray(bounds, dtype=np.float32)
        if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or (bounds[1] <= bounds[0]).any():
            raise ValueError('Invalid workspace bounds')
        valid &= ((xyz >= bounds[0]) & (xyz <= bounds[1])).all(-1)
    points, seg = points[valid], seg[valid]
    ids = target_ids if target_ids is not None else {o['name']: o['h5_id'] for o in objects}
    if any(o['name'] not in ids for o in objects):
        raise ValueError('Missing object target ID')
    selected_ids = [ids[o['name']] for o in objects]
    if len(set(selected_ids)) != len(objects):
        raise ValueError('Object target IDs must be distinct')
    out = {key: np.zeros(shape, dtype=(bool if 'mask' in key or key == 'object_valid'
                                      else np.int64 if key == 'object_roles' else np.float32))
           for key, shape in object_shapes(config, points.shape[-1]).items()}

    def sample(source, budget):
        # No duplication: every occupied row corresponds to a real source point.
        return source if len(source) <= budget else source[rng.choice(len(source), budget, replace=False)]

    for slot, obj in enumerate(objects):
        source = points[seg == ids[obj['name']]]
        out['object_roles'][slot] = ROLES[obj['role']]
        if not len(source):
            continue
        center = source[:, :3].mean(0)
        out['object_centers'][slot] = center
        out['object_extents'][slot] = np.ptp(source[:, :3], axis=0)
        chosen = sample(source, obj['max_points']).copy()
        chosen[:, :3] -= center
        n = len(chosen)
        out['object_points'][slot, :n] = chosen
        out['object_point_mask'][slot, :n] = True
        out['object_valid'][slot] = True
    # Explicitly exclude both object masks, including unsampled object points.
    context = sample(points[~np.isin(seg, selected_ids)], len(out['context_points']))
    out['context_points'][:len(context)] = context
    out['context_point_mask'][:len(context)] = True
    return out


def validate_object_arrays(obs, config=None, channels=None):
    """Validate unbatched or T+1 batched physical observations before use."""
    missing = set(OBJECT_FIELDS) - set(obs)
    if missing:
        raise ValueError(f'Missing object observation fields: {sorted(missing)}')
    p = np.asarray(obs['object_points'])
    if p.ndim not in (3, 4) or p.shape[-1] not in (3, 6):
        raise ValueError('Invalid object_points dimensions')
    prefix, (k, m, c) = p.shape[:-3], p.shape[-3:]
    ctx = np.asarray(obs['context_points'])
    if ctx.ndim != len(prefix) + 2:
        raise ValueError('Invalid context_points dimensions')
    n = ctx.shape[-2]
    shapes = dict(object_points=(k, m, c), object_point_mask=(k, m),
                  object_centers=(k, 3), object_extents=(k, 3), object_valid=(k,),
                  object_roles=(k,), context_points=(n, c), context_point_mask=(n,))
    if min(k, m, n) < 1:
        raise ValueError('Object observation dimensions must be positive')
    expected = object_shapes(config, channels or c) if config is not None else shapes
    for key, shape in expected.items():
        a = np.asarray(obs[key])
        if a.shape != prefix + shape or not np.isfinite(a).all():
            raise ValueError(f'Invalid object observation field: {key}')
    for key in ('object_point_mask', 'context_point_mask', 'object_valid'):
        if not np.isin(obs[key], [0, 1]).all():
            raise ValueError(f'{key} must be boolean')
    mask = np.asarray(obs['object_point_mask'], dtype=bool)
    valid = np.asarray(obs['object_valid'], dtype=bool)
    if not np.array_equal(mask.any(-1), valid):
        raise ValueError('object_valid must match object_point_mask')
    if (np.asarray(obs['object_extents']) < 0).any():
        raise ValueError('Object extents cannot be negative')
    roles = np.asarray(obs['object_roles'])
    if not np.isin(roles, list(ROLES.values())).all():
        raise ValueError('Invalid object_roles')
    if config is not None:
        objects = config['objects']
        if not np.equal(roles, [ROLES[o['role']] for o in objects]).all():
            raise ValueError('Object slot roles differ from config')
        if (mask.sum(-1) > np.array([o['max_points'] for o in objects])).any():
            raise ValueError('Object point count exceeds configured budget')
    for points_key, mask_key in [('object_points', 'object_point_mask'),
                                 ('context_points', 'context_point_mask')]:
        if np.any(np.asarray(obs[points_key])[~np.asarray(obs[mask_key], dtype=bool)] != 0):
            raise ValueError('Padding must be zero')
    if np.any(np.asarray(obs['object_centers'])[~valid] != 0) or np.any(np.asarray(obs['object_extents'])[~valid] != 0):
        raise ValueError('Missing object geometry must be zero')
