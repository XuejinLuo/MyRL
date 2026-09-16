"""Small, shared artifact contract for offline / iterative / online runs."""
import json
from pathlib import Path
from omegaconf import OmegaConf


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def log_metrics(directory, epoch, metrics, **context):
    path = Path(directory) / 'metrics.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(dict(epoch=epoch, **context, **metrics), allow_nan=False) + '\n')


def evaluation_paths(cfg, directory, epoch, tag='validation', sampler=None):
    mode = sampler or cfg.eval.sampler
    destination = Path(directory) / 'eval' / f'{tag}_ep{epoch:04d}_{mode}'
    video = cfg.get('video', {})
    every = int(video.get('every', 0))
    record = every > 0 and (epoch % every == 0 or epoch == cfg.epochs)
    return destination, dict(directory=str(destination / 'videos'),
                            episodes=int(video.get('episodes', 2))) if record else None


def evaluate_base(cfg, base, normalizer, directory, epoch, tag='validation', seeds=None):
    """Temporarily adapt an offline model without freezing its training parameters."""
    import numpy as np
    import torch
    from models.online_policy import FlowPPOPolicy
    from utils.online_eval import evaluate_policy
    from utils.online_env import make_env_ManiSkill
    modes = [(m, m.training) for m in base.modules()]
    grads = [(p, p.requires_grad) for p in base.parameters()]
    device = next(base.parameters()).device
    destination, video = evaluation_paths(cfg, directory, epoch, tag)
    try:
        actor = FlowPPOPolicy(base, num_steps=cfg.model.num_inference_steps,
                              noise_level=cfg.get('noise_level', .7),
                              min_std=cfg.get('min_std', .0067), eval_mode=cfg.eval.sampler)
        def encode(obs):
            pc = normalizer.center_point_cloud(np.asarray(obs['point_cloud'], dtype=np.float32),
                                                np.asarray(cfg.env.workspace_bounds))
            state = normalizer.normalize(np.asarray(obs['state'], dtype=np.float32), 'state')
            return actor.encode(torch.as_tensor(pc, device=device)[None],
                                torch.as_tensor(state, device=device)[None])
        return evaluate_policy(lambda: make_env_ManiSkill(cfg, video=video), actor, encode,
            normalizer, list(cfg.eval.seeds if seeds is None else seeds),
            cfg.model.num_inference_steps, output_dir=destination,
            metadata=dict(split=tag, epoch=epoch, sampler=cfg.eval.sampler,
                          env=OmegaConf.to_container(cfg.env, resolve=True)))
    finally:
        for p, enabled in grads:
            p.requires_grad_(enabled)
        for module, training in modes:
            module.training = training
