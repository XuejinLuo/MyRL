"""Opt-in iterative training with a dataset-size-independent update budget."""
import copy
from time import perf_counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import TrajectoryDataset
from data.iterative_sampling import (SourceDataset, FixedBatches, SamplingMetrics,
                                     validate_update_config)
from data.observations import batch_observation
from utils.experiment import log_metrics, write_json, write_selection, evaluation_paths


def loader(dataset, cfg, updates, seed, mode):
    sampler = FixedBatches(dataset, cfg.batch_size, updates, seed, mode, cfg.actor_sampling.demo_fraction)
    kwargs = dict(batch_sampler=sampler, num_workers=cfg.num_workers,
                  pin_memory=torch.device(cfg.device).type == 'cuda',
                  generator=torch.Generator().manual_seed(seed))
    if cfg.num_workers:
        kwargs.update(multiprocessing_context='spawn', persistent_workers=True)
    return DataLoader(dataset, **kwargs)


@torch.no_grad()
def actor_advantage(agent, obs, actions):
    # Both fixed-budget arms use the same post-critic-update inference rule.
    # No extra BatchNorm updates or gradients from the Actor's sampling stream.
    modes = [(m, m.training) for net in (agent.q_target, agent.v_net) for m in net.modules()]
    try:
        agent.q_target.eval()
        agent.v_net.eval()
        q1, q2 = agent.q_target(obs, actions)
        return torch.minimum(q1, q2) - agent.v_net(obs)
    finally:
        for module, mode in modes:
            module.training = mode


def train_updates(cfg, base, normalizer, episodes, directory, evaluate, save,
                  baseline, log_callback, source_spec, sampling_seed):
    from workflows.offline_round import build_agent
    validate_update_config(cfg)
    if source_spec is None:
        raise ValueError('Fixed updates require the source manifest specification')
    directory, device = Path(directory), torch.device(cfg.device)
    total, warmup = cfg.updates_per_round, cfg.critic_warmup_updates
    dataset = SourceDataset(TrajectoryDataset(episodes, cfg, normalizer), source_spec,
                            list(cfg.actor_sampling.demo_sources))
    critic_loader = loader(dataset, cfg, total, sampling_seed + 1001, 'mixed')
    mode = cfg.actor_sampling.mode
    actor_loader = (loader(dataset, cfg, total-warmup, sampling_seed + 2001, mode)
                    if mode == 'demo_success' else None)
    write_json(directory/'sampling.json', dict(**dataset.report, budget_unit='updates',
        critic_updates=total, actor_updates=total-warmup, critic_warmup_updates=warmup,
        batch_size=cfg.batch_size, actor_mode=mode, demo_fraction=cfg.actor_sampling.demo_fraction,
        demo_sources=list(cfg.actor_sampling.demo_sources),
        critic_sampling='all kept transitions, uniform with replacement',
        actor_sampling=('same batch as critic' if mode == 'mixed' else
                        'fixed demo/success-rollout quota; uniform episode then uniform time'),
        success_definition='any primitive step reports success',
        critic_seed=sampling_seed+1001, actor_seed=sampling_seed+2001,
        selection_criterion=['Eval/Success_Rate'],
        loss_definition='mean of retained per-sample Flow losses from the actual training forward pass'))
    base.requires_grad_(True)
    freeze = bool(cfg.get('freeze_actor_encoder', False))
    if freeze:
        base.encoder.requires_grad_(False)
        for parameter in base.encoder.parameters():
            parameter.grad = None
    agent = build_agent(cfg, base, device)
    ema = copy.deepcopy(base).eval().requires_grad_(False)
    actor_iterator = iter(actor_loader) if actor_loader is not None else None
    best = baseline['Eval/Success_Rate'] if baseline else -float('inf')
    selected = None
    cumulative = SamplingMetrics(len(source_spec['sources']))
    interval = SamplingMetrics(len(source_spec['sources']))
    sums, critic_count, actor_count, actor_total = {}, 0, 0, 0
    started = perf_counter()
    for step, raw_batch in enumerate(critic_loader, 1):
        base.train()
        if freeze:
            base.encoder.eval()
        agent.q_net.train()
        agent.v_net.train()
        agent.q_target.eval()
        batch = {k: v.to(device) for k, v in raw_batch.items()}
        metrics = agent.update_batch_critic(batch_observation(batch), batch['action_chunk'],
            batch['reward'][:, None], batch_observation(batch, 'next_'),
            batch['done'][:, None], batch['discount'][:, None])
        metrics.pop('adv_for_actor')
        for stats in (interval, cumulative):
            stats.add('Critic', batch)
        critic_count += 1
        if step > warmup:
            actor_batch = ({k: v.to(device) for k, v in next(actor_iterator).items()}
                           if actor_iterator is not None else batch)
            obs, actions = batch_observation(actor_batch), actor_batch['action_chunk']
            adv = actor_advantage(agent, obs, actions)
            actor_metrics, details = agent.update_actor(obs, actions, adv=adv, return_details=True)
            metrics.update(actor_metrics)
            for stats in (interval, cumulative):
                stats.add('Actor', actor_batch, details)
            actor_count += 1
            actor_total += 1
        if not all(np.isfinite(value) for value in metrics.values()):
            raise FloatingPointError('Nonfinite fixed-update training metric')
        with torch.no_grad():
            for a, b in zip(ema.parameters(), base.parameters()):
                a.lerp_(b, 1-cfg.ema_decay)
            for a, b in zip(ema.buffers(), base.buffers()):
                a.copy_(b)
            for a, b in zip(agent.q_target.buffers(), agent.q_net.buffers()):
                a.copy_(b)
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.) + value
        evaluate_now = step > warmup and (step % cfg.eval_every_updates == 0 or step == total)
        save_now = step > warmup and (step % cfg.save_every_updates == 0 or step == total)
        log_now = step % cfg.log_every_updates == 0 or step == warmup or step == total
        if not (evaluate_now or save_now or log_now):
            continue
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        seconds = perf_counter() - started
        row = {k: v/(actor_count if k == 'loss/actor' or k.startswith('metrics/') else critic_count)
               for k, v in sums.items()}
        row.update({'Train/Updates': step, 'Train/Critic_Updates': critic_count,
                    'Train/Actor_Updates': actor_count, 'Train/Critic_Updates_Total': step,
                    'Train/Actor_Updates_Total': actor_total, 'Train/Critic_Samples': critic_count*cfg.batch_size,
                    'Train/Actor_Samples': actor_count*cfg.batch_size, 'Time/Train_Seconds': seconds,
                    **interval.report()})
        row.update({'Cumulative/'+k: v for k, v in cumulative.report().items()})
        result = {}
        candidate = directory/'checkpoints'/f'step_{step:07d}.pth'
        if evaluate_now or save_now:
            raw = copy.deepcopy(base.state_dict())
            try:
                base.load_state_dict(ema.state_dict())
                if evaluate_now:
                    result = evaluate(epoch=step)
                    row.update(result)
                save(candidate, step, result)
                save(directory/'checkpoints'/'last.pth', step, result)
                # Dense environment return is not a tie breaker in this experiment.
                if result and result['Eval/Success_Rate'] > best:
                    best, selected = result['Eval/Success_Rate'], str(candidate)
                    save(directory/'checkpoints'/'best.pth', step, result)
                    write_selection(directory, None, result, directory/'checkpoints'/'best.pth',
                        stage=cfg.stage, update_step=step, budget_unit='updates', criterion=['Eval/Success_Rate'])
            finally:
                base.load_state_dict(raw)
        context = dict(stage=cfg.stage, sampler=cfg.eval.sampler, round=directory.name,
                       update_step=step, budget_unit='updates',
                       checkpoint=str(candidate) if evaluate_now or save_now else None,
                       evaluation=str(evaluation_paths(cfg, directory, step)[0]) if result else None)
        for destination in (directory, directory.parent):
            log_metrics(destination, None, row, **context)
        if log_callback:
            log_callback(row, step)
        print(dict(update_step=step, **row), flush=True)
        interval = SamplingMetrics(len(source_spec['sources']))
        sums, critic_count, actor_count = {}, 0, 0
        started = perf_counter()
    return selected
