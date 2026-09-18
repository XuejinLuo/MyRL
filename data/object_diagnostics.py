"""Observability summaries; counts always exclude padding."""
import numpy as np


def summarize_objects(episodes, config):
    result = {}
    for slot, obj in enumerate(config['objects']):
        counts = np.concatenate([ep['object_point_mask'][:, slot].sum(-1) for ep in episodes])
        centers = np.concatenate([ep['object_centers'][:, slot] for ep in episodes])
        valid = counts > 0
        prefix = f"Object/{obj['name']}"
        result.update({prefix + '/points_mean': float(counts.mean()),
            prefix + '/points_median': float(np.median(counts)),
            prefix + '/points_p05': float(np.percentile(counts, 5)),
            prefix + '/valid_rate': float(valid.mean()),
            prefix + '/missing_rate': float((~valid).mean()),
            prefix + '/occupancy': float(counts.mean() / obj['max_points']),
            prefix + '/padding_ratio': float(1 - counts.mean() / obj['max_points'])})
        result[prefix + '/center_variance_xyz'] = (
            centers[valid].var(0).tolist() if valid.any() else None)
    context = np.concatenate([ep['context_point_mask'].sum(-1) for ep in episodes])
    result['Context/points_mean'] = float(context.mean())
    result['Context/points_p05'] = float(np.percentile(context, 5))
    result['Context/padding_ratio'] = float(1 - context.mean() / config.get('context_points', 512))
    return result


def batch_object_metrics(obs, config):
    metrics = {}
    for slot, obj in enumerate(config['objects']):
        counts = obs['object_point_mask'][:, slot].float().sum(-1)
        metrics[f"Object/{obj['name']}_points"] = counts.mean().item()
        metrics[f"Object/{obj['name']}_valid_rate"] = (counts > 0).float().mean().item()
        metrics[f"Object/{obj['name']}_padding_ratio"] = 1 - counts.mean().item() / obj['max_points']
    metrics['Context/points'] = obs['context_point_mask'].float().sum(-1).mean().item()
    return metrics
