"""Fit IQL critics on an immutable manifest without updating/collecting Actor."""
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from omegaconf import OmegaConf

from data.dataset import TrajectoryDataset
from data.episodes import digest, load_sources
from data.observations import batch_observation
from evaluation.runner import seed_all
from models.checkpoint import policy_weights
from models.factory import build_base
from utils.config import validate_common
from utils.experiment import log_metrics, tracking, write_json
from utils.normalizer import MinMaxNormalizer
from workflows.offline_round import build_agent, build_loader

FORMAT = 'myrl_frozen_actor_critic_v1'


def assert_actor_unchanged(base, original):
    if any(p.requires_grad or p.grad is not None for p in base.parameters()):
        raise RuntimeError('Critic-only training must keep Actor frozen')
    for key, value in base.state_dict().items():
        if not torch.equal(value.detach().cpu(), original[key].cpu()):
            raise RuntimeError(f'Critic-only training changed Actor: {key}')


def excluded_seeds(actor_config, spec):
    """Remember source Actor validation/collection and all manifest rollout seeds."""
    seeds = set(actor_config.get('eval', {}).get('seeds', []))
    for collection in (actor_config, actor_config.get('stages', {}).get('iterative', {})):
        if all(k in collection for k in ('collect_seed_start', 'rounds', 'episodes_per_round')):
            seeds.update(range(collection['collect_seed_start'], collection['collect_seed_start'] +
                               collection['rounds'] * collection['episodes_per_round']))
    for source in spec['sources']:
        seeds.update(int(item['seed']) for item in source['episodes'] if item.get('seed') is not None)
    return sorted(seeds)


def run(cfg):
    validate_common(cfg)
    if not 0 < cfg.algo.discount <= 1 or not 0 < cfg.algo.tau < 1:
        raise ValueError('Invalid critic discount/expectile')
    if not 0 < cfg.algo.tau_target <= 1 or cfg.algo.critic_lr <= 0:
        raise ValueError('Invalid critic learning rate/target update')
    output = Path(cfg.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f'{output}: choose a new output directory')
    source = Path(cfg.initial_ckpt).expanduser().resolve()
    manifest = Path(cfg.manifest).expanduser().resolve()
    cp = torch.load(source, map_location='cpu', weights_only=True)
    if 'config' not in cp or 'normalizer' not in cp:
        raise ValueError('Actor checkpoint requires embedded config and normalizer')
    frozen = OmegaConf.to_container(cfg, resolve=True)
    for section in ('model', 'env'):
        if cp['config'][section] != frozen[section]:
            raise ValueError(f'Actor checkpoint {section} configuration differs')
    normalizer = MinMaxNormalizer()
    normalizer.stats = cp['normalizer']
    for key, dim in [('action', cfg.model.action_dim), ('state', cfg.model.state_dim)]:
        lo, hi = [np.asarray(normalizer.stats[key][k]) for k in ('min', 'max')]
        if lo.shape != (dim,) or hi.shape != (dim,) or not np.isfinite([lo, hi]).all() or (hi < lo).any():
            raise ValueError(f'Invalid normalization: {key}')
    episodes, spec = load_sources(manifest)
    if spec['env'] != frozen['env']:
        raise ValueError('Dataset environment/preprocessing differs from current config')
    dataset = TrajectoryDataset(episodes, cfg, normalizer)
    if len(dataset) < cfg.batch_size:
        raise ValueError('Dataset smaller than batch_size; lower batch_size')
    device = torch.device(cfg.device)
    seed_all(cfg.seed)
    base = build_base(cfg, device)
    original = policy_weights(cp, cfg.weight_key)
    base.load_state_dict(original, strict=True)
    base.eval().requires_grad_(False)
    agent = build_agent(cfg, base, device)
    loader = build_loader(dataset, cfg, device)
    provenance = dict(actor_checkpoint=str(source), actor_sha256=digest(source),
                      weight_key=str(cfg.weight_key), manifest=str(manifest),
                      manifest_sha256=digest(manifest),
                      excluded_eval_seeds=excluded_seeds(cp['config'], spec),
                      episodes=len(dataset.episodes), transitions=len(dataset))
    # Use the source Actor's sampling protocol even if training defaults differ.
    actor_algo = cp['config'].get('algo', {})
    sampler = dict(noise_level=cp['config'].get('noise_level', actor_algo.get('noise_level', .7)),
                   min_std=cp['config'].get('min_std', actor_algo.get('min_std', .0067)))
    (output/'checkpoints').mkdir(parents=True)
    OmegaConf.save(cfg, output/'config.yaml', resolve=True)
    write_json(output/'provenance.json', provenance)
    with tracking(cfg) as run_log:
        for epoch in range(1, cfg.epochs + 1):
            base.eval()
            agent.q_net.train()
            agent.v_net.train()
            agent.q_target.eval()
            started, sums = perf_counter(), {}
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                metrics = agent.update_batch_critic(batch_observation(batch), batch['action_chunk'],
                    batch['reward'][:, None], batch_observation(batch, 'next_'),
                    batch['done'][:, None], batch['discount'][:, None])
                metrics.pop('adv_for_actor')
                if not all(np.isfinite(value) for value in metrics.values()):
                    raise FloatingPointError('Nonfinite critic training metric')
                with torch.no_grad():
                    for target, current in zip(agent.q_target.buffers(), agent.q_net.buffers()):
                        target.copy_(current)
                for key, value in metrics.items():
                    sums[key] = sums.get(key, 0.) + value
            metrics = {key: value/len(loader) for key, value in sums.items()}
            metrics.update({'Train/Batches': len(loader), 'Train/Actor_Updates': 0,
                            'Train/Samples': len(loader)*cfg.batch_size,
                            'Time/Train_Seconds': perf_counter()-started})
            assert_actor_unchanged(base, original)
            if epoch % cfg.save_epoch == 0 or epoch == cfg.epochs:
                payload = dict(format=FORMAT, config=frozen, normalizer=normalizer.stats,
                    model_state_dict=base.state_dict(), q_state_dict=agent.q_net.state_dict(),
                    q_target_state_dict=agent.q_target.state_dict(), v_state_dict=agent.v_net.state_dict(),
                    q_optimizer=agent.q_opt.state_dict(), v_optimizer=agent.v_opt.state_dict(),
                    epoch=epoch, metrics=metrics, provenance=provenance, sampler=sampler,
                    selection_rule='argmax_min_twin_q', actor_unchanged=True)
                torch.save(payload, output/'checkpoints'/f'epoch_{epoch:04d}.pth')
                torch.save(payload, output/'checkpoints'/'last.pth')
            log_metrics(output, epoch, metrics, stage='critic')
            if run_log:
                run_log.log(metrics, step=epoch)
            print(dict(epoch=epoch, **metrics), flush=True)
    print(f'Critic checkpoint: {output / "checkpoints/last.pth"}', flush=True)
    return output/'checkpoints'/'last.pth'
