"""Weights-only readable checkpoints and explicit online sampler metadata."""
import torch
import copy
import numpy as np
from models.online_policy import FlowPPOPolicy

FORMAT = 'myrl_flow_ppo_v1'


def checkpoint_primitives(value):
    """Normalize numeric metadata without copying/changing tensors or optimizer keys."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        # Preserve OrderedDict and its state_dict version metadata.
        result = copy.copy(value)
        result.update({k: checkpoint_primitives(v) for k, v in value.items()})
        return result
    if isinstance(value, list):
        return [checkpoint_primitives(v) for v in value]
    if isinstance(value, tuple):
        return tuple(checkpoint_primitives(v) for v in value)
    return value


def load_checkpoint(path, map_location='cpu'):
    """Read legacy NumPy numeric scalars while retaining weights-only restrictions.

    PR #18 could save numpy.float64 in Train/Reward. Support both NumPy module
    spellings and pre/post-1.25 dtype classes. Never allow globals discovered
    from the file, ndarray reconstruction, or an unrestricted pickle fallback.
    """
    from importlib import import_module
    module = 'numpy._core.multiarray' if hasattr(np, '_core') else 'numpy.core.multiarray'
    scalar = import_module(module).scalar
    numeric = (np.float16, np.float32, np.float64, np.int8, np.int16,
               np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64, np.bool_)
    allowed = [(scalar, 'numpy.core.multiarray.scalar'),
               (scalar, 'numpy._core.multiarray.scalar'), np.dtype]
    allowed.extend({type(np.dtype(dtype)) for dtype in numeric})
    with torch.serialization.safe_globals(allowed):
        state = torch.load(path, map_location=map_location, weights_only=True)
    return checkpoint_primitives(state)


def policy_weights(checkpoint, weight_key='auto'):
    if weight_key == 'auto':
        # Prefer EMA in legacy offline checkpoints; new snapshots already store the evaluated weights.
        for key in ('ema_model_state_dict', 'model_state_dict'):
            if key in checkpoint:
                return checkpoint[key]
        return checkpoint
    if weight_key == 'raw':
        return checkpoint
    if weight_key not in checkpoint:
        raise KeyError(f'Checkpoint has no weight key {weight_key}')
    return checkpoint[weight_key]


def make_actor(base, cfg):
    return FlowPPOPolicy(base, num_steps=cfg.model.num_inference_steps,
        noise_level=cfg.algo.noise_level, min_std=cfg.algo.min_std,
        eval_mode=cfg.eval.sampler)


def payload(base, cfg, normalizer, epoch, metrics):
    from omegaconf import OmegaConf
    return {'format': FORMAT, 'model_state_dict': base.state_dict(),
            'config': OmegaConf.to_container(cfg, resolve=True),
            'normalizer': normalizer.stats, 'epoch': epoch, 'metrics': metrics}
