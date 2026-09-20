"""Shared configuration validation; fail before loading large datasets."""
import numpy as np
from data.observations import validate_observation_config, observation_mode
from models.encoders.factory import encoder_options


def validate_common(cfg):
    validate_observation_config(cfg.env)
    encoder_options(cfg)  # Reject invalid state-skip experiments before H5 ingestion.
    if observation_mode(cfg.env) == "object_centric" and cfg.model.cond_dim % 4:
        raise ValueError("Object-centric cond_dim must be divisible by 4")
    if cfg.model.algo_type != 'flow':
        raise ValueError('The three-stage CPS/PPO workflow requires model.algo_type=flow')
    if not 1 <= cfg.env.exec_steps <= cfg.model.chunk_size:
        raise ValueError('Require 1 <= env.exec_steps <= model.chunk_size')
    for key in ('epochs', 'batch_size', 'save_epoch'):
        if cfg[key] < 1:
            raise ValueError(f'{key} must be positive')
    if cfg.batch_size < 2 or cfg.num_workers < 0:
        raise ValueError('Require batch_size >= 2 and num_workers >= 0')
    if cfg.eval.every < 1 or cfg.eval.sampler not in ('cps', 'ode'):
        raise ValueError('Invalid evaluation settings')
    if not cfg.eval.seeds or len(set(cfg.eval.seeds)) != len(cfg.eval.seeds):
        raise ValueError('Evaluation seeds must be nonempty and unique')
    if cfg.video.every < 0 or cfg.video.episodes < 0:
        raise ValueError('Video counts cannot be negative')
    if cfg.video.every and cfg.video.every % cfg.eval.every:
        raise ValueError('video.every must be a multiple of eval.every (or zero)')
    if cfg.model.in_channels != (6 if cfg.env.use_color else 3):
        raise ValueError('model.in_channels must match env.use_color')
    bounds = np.asarray(cfg.env.workspace_bounds)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or (bounds[1] <= bounds[0]).any():
        raise ValueError('Invalid env.workspace_bounds')
    if cfg.env.max_episode_steps < 1:
        raise ValueError('env.max_episode_steps must be positive')
    if not cfg.experiment or '/' in cfg.experiment or cfg.experiment in ('.', '..'):
        raise ValueError('experiment must be a simple run name')
    comparison = cfg.comparison
    test_seeds = set(range(comparison.seed_start, comparison.seed_start + comparison.episodes))
    if comparison.episodes < 1 or set(cfg.eval.seeds) & test_seeds:
        raise ValueError('Comparison seeds must be nonempty and disjoint from validation seeds')
    if cfg.stage in ('offline', 'iterative'):
        if not 0 < cfg.algo.discount <= 1 or not 0 < cfg.algo.tau < 1:
            raise ValueError('Invalid IDQL discount or expectile')
        if not 0 <= cfg.ema_decay < 1 or not 0 <= cfg.critic_warmup_epochs < cfg.epochs:
            raise ValueError('Invalid EMA decay or critic warmup')
