"""Policy construction and normalized observation encoding shared by all stages."""
import numpy as np
import torch
from data.observations import (episode_observation, normalize_observation, validate_observation)
from models.encoders.factory import encoder_options


def build_base(cfg, device):
    from models.policy import EmbodiedGenPolicy
    keys = ('in_channels', 'action_dim', 'chunk_size', 'use_state', 'state_dim',
            'encoder_type', 'backbone_type', 'cond_dim', 'algo_type')
    options = {k: cfg.model[k] for k in keys}
    options.update(encoder_options(cfg))
    return EmbodiedGenPolicy(**options).to(device)


def observation_encoder(cfg, actor, normalizer, device):
    bounds = np.asarray(cfg.env.workspace_bounds)
    def encode(obs):
        obs = episode_observation(obs)
        validate_observation(obs, cfg)
        normalized = normalize_observation(obs, normalizer, bounds)
        tensors = {k: torch.as_tensor(v, device=device)[None] for k, v in normalized.items()}
        if 'pc' in tensors and 'object_features' not in tensors:
            # Retain compatibility with the historical two-argument adapter.
            return actor.encode(tensors['pc'], tensors['state'])
        return actor.encode(tensors)
    return encode
