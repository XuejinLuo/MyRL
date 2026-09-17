"""Identical spatial filtering and sampling for H5 and live observations."""
import numpy as np


def preprocess_points(xyz, rgb, bounds, num_points, use_color):
    xyz = np.asarray(xyz, dtype=np.float32)
    if use_color:
        rgb = np.zeros_like(xyz) if rgb is None else np.asarray(rgb, dtype=np.float32)
        points = np.concatenate([xyz, rgb], axis=-1)
    else:
        points = xyz
    mask = np.isfinite(points).all(axis=-1)
    if bounds is not None:
        bounds = np.asarray(bounds, dtype=np.float32)
        mask &= ((xyz >= bounds[0]) & (xyz <= bounds[1])).all(axis=-1)
    points = points[mask]
    if not len(points):
        return np.zeros((num_points, 6 if use_color else 3), dtype=np.float32)
    indices = np.random.choice(len(points), num_points, replace=len(points) < num_points)
    return np.ascontiguousarray(points[indices])
