"""Shared spatial filtering and sampling for H5 and live observations."""
import numbers
import numpy as np


def require_fps():
    """Load the CPU backend explicitly; never silently substitute random sampling."""
    try:
        from fpsample import bucket_fps_kdline_sampling
    except ImportError as exc:
        raise ImportError(
            'FPS sampling requires fpsample. Run: '
            'python -m pip install -r requirements-pointcloud.txt'
        ) from exc
    return bucket_fps_kdline_sampling


def fps_indices(xyz, num_points):
    """Geometry-only FPS on the entire cropped cloud, with a fixed start.

    Deduplicate coordinates before calling the backend so coincident points
    cannot consume the spatial budget. If too few distinct positions remain,
    retain every position before deterministic repeat padding. RGB is gathered
    by these same indices by the caller; it is never a distance coordinate.
    """
    _, first = np.unique(xyz, axis=0, return_index=True)
    first.sort()
    if len(first) > num_points:
        sample = require_fps()
        # QuickFPS requires 2**h <= number of candidates, even for tiny clouds.
        height = min(5, len(first).bit_length() - 1)
        local = sample(np.ascontiguousarray(xyz[first], dtype=np.float32),
                       num_points, h=height, start_idx=0)
        selected = first[np.asarray(local, dtype=np.int64)]
    else:
        selected = np.resize(first, num_points)
    return selected


def validate_sampling(sampling, num_points):
    """Missing settings mean the historical uniform-random protocol."""
    if not isinstance(num_points, numbers.Integral) or num_points < 1:
        raise ValueError('num_points must be a positive integer')
    sampling = sampling or {}
    mode = sampling.get('mode', 'random')
    if mode not in ('random', 'object_budget', 'fps'):
        raise ValueError(f'Unknown point sampling mode: {mode}')
    if mode in ('random', 'fps'):
        return []
    objects = list(sampling.get('objects', []))
    if not objects:
        raise ValueError('object_budget sampling requires objects')
    names, ids, total = set(), set(), 0
    for obj in objects:
        name, seg_id, budget = obj.get('name'), obj.get('h5_id'), obj.get('num_points')
        if not isinstance(name, str) or not name or name in names:
            raise ValueError('Sampling object names must be nonempty and unique')
        if (not isinstance(seg_id, numbers.Integral) or isinstance(seg_id, bool)
                or seg_id <= 0 or seg_id in ids):
            raise ValueError('Sampling h5_id values must be unique positive integers')
        if (not isinstance(budget, numbers.Integral) or isinstance(budget, bool)
                or budget < 1):
            raise ValueError('Object num_points budgets must be positive integers')
        names.add(name)
        ids.add(seg_id)
        total += budget
    if total > num_points:
        raise ValueError('Object point budgets exceed env.num_points')
    return objects


def resolve_target_ids(objects, segmentation_id_map):
    """Resolve names on the current scene, never reuse H5 IDs in simulation."""
    result = {}
    for obj in objects:
        name = obj['name']
        matches = [int(key) for key, actor in segmentation_id_map.items()
                   if getattr(actor, 'name', None) == name]
        if len(matches) != 1:
            raise ValueError(f'Expected one segmentation ID for {name!r}, found {matches}')
        result[name] = matches[0]
    if len(set(result.values())) != len(result):
        raise ValueError('Sampling targets must have distinct segmentation IDs')
    return result


def preprocess_points(xyz, rgb, bounds, num_points, use_color, *,
                      sampling=None, segmentation=None, target_ids=None,
                      rng=None, return_indices=False):
    """Crop once, then apply random, geometry FPS, or object-budget sampling.

    Object budgets reserve scarce visible source points before random fill.
    FPS uses XYZ only, has no segmentation dependency, and repeats only when
    fewer than num_points distinct positions survive filtering.
    Returned diagnostic indices address the input cloud; -1 denotes empty fill.
    """
    objects = validate_sampling(sampling, num_points)
    mode = (sampling or {}).get('mode', 'random')
    rng = np.random if rng is None else rng
    xyz = np.asarray(xyz, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError('xyz must have shape (N, 3)')
    if use_color:
        rgb = np.zeros_like(xyz) if rgb is None else np.asarray(rgb, dtype=np.float32)
        if rgb.shape != xyz.shape:
            raise ValueError('rgb must match xyz shape')
        points = np.concatenate([xyz, rgb], axis=-1)
    else:
        points = xyz
    seg = None
    if objects:
        if segmentation is None:
            raise ValueError('object_budget requires pointcloud segmentation; no random fallback')
        seg = np.asarray(segmentation).reshape(-1)
        if len(seg) != len(xyz) or not np.isfinite(seg).all():
            raise ValueError('segmentation must contain one finite ID per input point')
        ids = target_ids if target_ids is not None else {o['name']: o['h5_id'] for o in objects}
        if any(o['name'] not in ids for o in objects):
            raise ValueError('Missing sampling target ID')
        if len({ids[o['name']] for o in objects}) != len(objects):
            raise ValueError('Sampling targets must have distinct segmentation IDs')
    mask = np.isfinite(points).all(axis=-1)
    if bounds is not None:
        bounds = np.asarray(bounds, dtype=np.float32)
        mask &= ((xyz >= bounds[0]) & (xyz <= bounds[1])).all(axis=-1)
    source_indices = np.flatnonzero(mask)
    points = points[mask]
    if not len(points):
        sampled = np.zeros((num_points, 6 if use_color else 3), dtype=np.float32)
        indices = np.full(num_points, -1, dtype=np.int64)
    else:
        if mode == 'fps':
            selected = fps_indices(points[:, :3], num_points)
        elif not objects:
            # Preserve the old RNG behavior for existing checkpoints/baselines.
            selected = rng.choice(len(points), num_points, replace=len(points) < num_points)
        else:
            seg = seg[mask]
            available = np.ones(len(points), dtype=bool)
            reserved = []
            for obj in objects:
                candidates = np.flatnonzero(seg == ids[obj['name']])
                count = min(len(candidates), int(obj['num_points']))
                chosen = (candidates if count == len(candidates)
                          else rng.choice(candidates, count, replace=False))
                reserved.append(chosen)
                available[chosen] = False
            selected = np.concatenate(reserved)
            remaining = np.flatnonzero(available)
            count = min(num_points - len(selected), len(remaining))
            fill = rng.choice(remaining, count, replace=False)
            selected = np.concatenate([selected, fill])
            if len(selected) < num_points:
                selected = np.concatenate([selected, rng.choice(
                    len(points), num_points-len(selected), replace=True)])
            # Ball-query truncation is input-order-sensitive; avoid object blocks.
            rng.shuffle(selected)
        sampled = np.ascontiguousarray(points[selected])
        indices = source_indices[selected]
    return (sampled, indices) if return_indices else sampled
