"""Weights-only readable checkpoints and explicit online sampler metadata."""
import torch
from models.online_policy import FlowPPOPolicy

FORMAT = 'myrl_flow_ppo_v1'


def policy_weights(checkpoint, weight_key='auto'):
    if weight_key == 'auto':
        # Match the uploaded train_online.py: prefer EMA for offline checkpoints.
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
