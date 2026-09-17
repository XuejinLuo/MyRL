"""Offline initialization using the same primitive data and IDQL loop as iterative."""
from pathlib import Path
import numpy as np
import torch
import hydra
from omegaconf import OmegaConf
from data.demonstrations import load_demonstrations, export_demonstrations
from models.factory import build_base
from utils.config import validate_common
from utils.normalizer import MinMaxNormalizer
from utils.experiment import evaluate_base, tracking
from evaluation.runner import seed_all
from workflows.offline_round import train_round


def run(cfg):
    validate_common(cfg)
    if not 0 <= cfg.critic_warmup_epochs < cfg.epochs or not 0 <= cfg.ema_decay < 1:
        raise ValueError('Invalid critic warmup or EMA decay')
    seed_all(cfg.seed)
    output = Path(hydra.utils.to_absolute_path(cfg.output)).resolve()
    if output.exists():
        raise FileExistsError(f'{output}: choose a new experiment name')
    episodes = load_demonstrations(cfg)
    normalizer = MinMaxNormalizer()
    normalizer.fit({key: np.concatenate([ep[key] for ep in episodes]) for key in ('action', 'state')})
    (output/'checkpoints').mkdir(parents=True)
    normalizer.save(str(output/'checkpoints'/'dataset_stats.json'))
    export_demonstrations(episodes, cfg, output/'data')
    OmegaConf.save(cfg, output/'config.yaml', resolve=True)
    base = build_base(cfg, torch.device(cfg.device))
    frozen = OmegaConf.to_container(cfg, resolve=True)

    def save(path, epoch, metrics):
        torch.save(dict(model_state_dict=base.state_dict(), normalizer=normalizer.stats,
            config=frozen, epoch=epoch, metrics=metrics,
            dataset_manifest=str(output/'data'/'manifest.json')), path)

    def evaluate(epoch):
        return evaluate_base(cfg, base, normalizer, output, epoch)

    with tracking(cfg) as wandb_run:
        train_round(cfg, base, normalizer, episodes, output, evaluate, save,
                    log_callback=(lambda metrics, epoch: wandb_run.log(metrics, step=epoch)) if wandb_run else None)
    print(f'Offline policy: {output / "checkpoints" / "best.pth"}', flush=True)
