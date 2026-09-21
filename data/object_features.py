"""Visible-surface geometry for the two-object global relational ablation.

Raw centers are world-frame, just like stored global clouds. Observation
normalization centers only those six coordinates; all other values are metric.
Relations touching an invisible object are zero, never fabricated from padding.
"""
import numpy as np

OBJECT_FEATURE_DIM = 23


def build_relational_features(xyz, segmentation, objects, tcp_position, bounds,
                              *, target_ids=None, rgb=None, use_color=False):
    if len(objects) != 2:
        raise ValueError('Relational features require exactly two ordered objects')
    xyz = np.asarray(xyz, dtype=np.float32)
    tcp = np.asarray(tcp_position, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError('xyz must have shape (N, 3)')
    if tcp.shape != (3,) or not np.isfinite(tcp).all():
        raise ValueError('tcp_position must be a finite 3-vector')
    if segmentation is None:
        raise ValueError('Relational features require segmentation')
    seg = np.asarray(segmentation).reshape(-1)
    if len(seg) != len(xyz) or not np.isfinite(seg).all():
        raise ValueError('segmentation must contain one finite ID per point')
    ids = target_ids if target_ids is not None else {o['name']: o['h5_id'] for o in objects}
    if any(o['name'] not in ids for o in objects):
        raise ValueError('Missing relational target ID')
    if len({ids[o['name']] for o in objects}) != 2:
        raise ValueError('Relational targets must have distinct segmentation IDs')
    # Match preprocess_points, including finite RGB filtering when color is used.
    mask = np.isfinite(xyz).all(-1)
    if use_color and rgb is not None:
        rgb = np.asarray(rgb, dtype=np.float32)
        if rgb.shape != xyz.shape:
            raise ValueError('rgb must match xyz shape')
        mask &= np.isfinite(rgb).all(-1)
    if bounds is not None:
        bounds = np.asarray(bounds, dtype=np.float32)
        mask &= ((xyz >= bounds[0]) & (xyz <= bounds[1])).all(-1)
    centers = np.zeros((2, 3), dtype=np.float32)
    extents = np.zeros_like(centers)
    valid = np.zeros(2, dtype=np.float32)
    for i, obj in enumerate(objects):
        points = xyz[mask & (seg == ids[obj['name']])]
        if len(points):
            # Float64 accumulation avoids order-dependent float32 centroid noise.
            centers[i] = points.mean(0, dtype=np.float64)
            extents[i] = points.max(0) - points.min(0)
            valid[i] = 1
    tcp_rel = np.where(valid[:, None].astype(bool), centers - tcp, 0.)
    pair = centers[1] - centers[0] if valid.all() else np.zeros(3)
    return np.concatenate([centers.ravel(), extents.ravel(), tcp_rel.ravel(), pair, valid]).astype(np.float32)


def validate_relational_features(features):
    features = np.asarray(features)
    if features.shape[-1:] != (OBJECT_FEATURE_DIM,) or not np.isfinite(features).all():
        raise ValueError('object_features must be finite with last dimension 23')
    if not np.isin(features[..., -2:], [0., 1.]).all():
        raise ValueError('Relational visibility flags must be 0 or 1')
