"""Policy construction and normalized observation encoding shared by all stages."""
import numpy as np
import torch


def build_base(cfg, device):
    from models.policy import EmbodiedGenPolicy
    keys = ('in_channels', 'action_dim', 'chunk_size', 'use_state', 'state_dim',
            'encoder_type', 'backbone_type', 'cond_dim', 'algo_type')
    return EmbodiedGenPolicy(**{k: cfg.model[k] for k in keys}).to(device)


def observation_encoder(cfg, actor, normalizer, device):
    bounds = np.asarray(cfg.env.workspace_bounds)
    def encode(obs):
        pc = np.asarray(obs['point_cloud'], dtype=np.float32)
        state = np.asarray(obs['state'], dtype=np.float32)
        if pc.shape != (cfg.env.num_points, cfg.model.in_channels) or state.shape != (cfg.model.state_dim,):
            raise ValueError(f'Observation shape mismatch: pc={pc.shape}, state={state.shape}')
        if not np.isfinite(pc).all() or not np.isfinite(state).all():
            raise ValueError('Nonfinite observation')
        pc = normalizer.center_point_cloud(pc, bounds)
        state = normalizer.normalize(state, 'state')
        return actor.encode(torch.as_tensor(pc, device=device)[None],
                            torch.as_tensor(state, device=device)[None])
    return encode
