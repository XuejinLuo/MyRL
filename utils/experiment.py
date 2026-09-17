"""Small, shared artifact contract for offline / iterative / online runs."""
import json
from contextlib import contextmanager
from pathlib import Path
from omegaconf import OmegaConf


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def log_metrics(directory, epoch, metrics, **context):
    """One flat JSONL/CSV schema; unavailable algorithm metrics stay empty."""
    import csv
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    row = dict(schema='myrl_metrics_v2', stage=context.pop('stage', 'unknown'),
               round=context.pop('round', None), epoch=int(epoch),
               sampler=context.pop('sampler', None),
               checkpoint=context.pop('checkpoint', None),
               evaluation=context.pop('evaluation', None), **context, **metrics)
    path = directory/'metrics.jsonl'
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, allow_nan=False) + '\n')
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    fields = list(dict.fromkeys(k for item in rows for k in item))
    with (directory/'metrics.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def selection_score(metrics):
    return metrics['Eval/Success_Rate'], metrics['Eval/Mean_Reward']


def write_selection(directory, epoch, metrics, checkpoint, **context):
    write_json(Path(directory)/'selection.json', dict(schema='myrl_selection_v2',
        epoch=epoch, metrics=metrics, checkpoint=str(Path(checkpoint).resolve()),
        weight_key='model_state_dict', criterion=['Eval/Success_Rate', 'Eval/Mean_Reward'],
        **context))


def evaluation_paths(cfg, directory, epoch, tag='validation', sampler=None):
    mode = sampler or cfg.eval.sampler
    destination = Path(directory) / 'eval' / f'{tag}_ep{epoch:04d}_{mode}'
    video = cfg.get('video', {})
    every = int(video.get('every', 0))
    record = every > 0 and (epoch % every == 0 or epoch == cfg.epochs)
    return destination, dict(directory=str(destination / 'videos'),
                            episodes=int(video.get('episodes', 5))) if record else None


def evaluate_base(cfg, base, normalizer, directory, epoch, tag='validation', seeds=None, checkpoint=None):
    """Temporarily adapt an offline model without freezing its training parameters."""
    from models.online_policy import FlowPPOPolicy
    from evaluation.runner import evaluate_policy
    from envs.factory import make_env
    modes = [(m, m.training) for m in base.modules()]
    grads = [(p, p.requires_grad) for p in base.parameters()]
    device = next(base.parameters()).device
    destination, video = evaluation_paths(cfg, directory, epoch, tag)
    try:
        actor = FlowPPOPolicy(base, num_steps=cfg.model.num_inference_steps,
                              noise_level=cfg.get('noise_level', .7),
                              min_std=cfg.get('min_std', .0067), eval_mode=cfg.eval.sampler)
        from models.factory import observation_encoder
        encode = observation_encoder(cfg, actor, normalizer, device)
        return evaluate_policy(lambda: make_env(cfg, video=video), actor, encode,
            normalizer, list(cfg.eval.seeds if seeds is None else seeds),
            cfg.model.num_inference_steps, output_dir=destination,
            metadata=dict(stage=cfg.get('stage', 'evaluation'),
                          round=Path(directory).name if Path(directory).name.startswith('round_') else None,
                          split=tag, epoch=epoch, sampler=cfg.eval.sampler,
                          checkpoint=str(checkpoint or Path(directory)/'checkpoints'/f'epoch_{epoch:04d}.pth'),
                          env=OmegaConf.to_container(cfg.env, resolve=True)))
    finally:
        for p, enabled in grads:
            p.requires_grad_(enabled)
        for module, training in modes:
            module.training = training


@contextmanager
def tracking(cfg):
    run = None
    try:
        if cfg.wandb.enable:
            import wandb
            run = wandb.init(project=cfg.wandb.project, entity=cfg.wandb.get('entity'),
                             name=cfg.run_name, config=OmegaConf.to_container(cfg, resolve=True))
        yield run
    finally:
        if run:
            run.finish()
