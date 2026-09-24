"""One IDQL round, reusing the existing dictionary adapters."""
import copy
from time import perf_counter
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from models.critics.q_v_network import VNetwork, TwinQNetwork
from algos.embodied_idql import (CriticFeatureExtractor, IDQL_VNet_Wrapper,
    IDQL_QNet_Wrapper, Policy_IDQL_Wrapper, EmbodiedIDQL)
from data.dataset import TrajectoryDataset
from data.observations import batch_observation
from utils.experiment import log_metrics, selection_score, write_selection, evaluation_paths


class PrefixQ(IDQL_QNet_Wrapper):
    def __init__(self, encoder, q_net, prefix):
        super().__init__(encoder, q_net)
        self.prefix = prefix

    def forward(self, obs, action):
        return super().forward(obs, action[:, :self.prefix].contiguous())


class RoundIDQL(EmbodiedIDQL):
    def update_batch_critic(self, obs, actions, rewards, nxt, dones, discounts):
        # Existing implementation accepts a broadcastable per-sample discount.
        previous = self.discount
        self.discount = discounts
        try:
            return self.update_critic(obs, actions, rewards, nxt, dones)
        finally:
            self.discount = previous


def build_q(cfg):
    return PrefixQ(CriticFeatureExtractor(cfg), TwinQNetwork(
        state_dim=cfg.model.cond_dim, action_dim=cfg.model.action_dim,
        chunk_size=cfg.env.exec_steps), cfg.env.exec_steps)


def build_agent(cfg, base, device):
    v = IDQL_VNet_Wrapper(CriticFeatureExtractor(cfg), VNetwork(cfg.model.cond_dim))
    return RoundIDQL(Policy_IDQL_Wrapper(base), build_q(cfg), v, device=device,
        **{k: cfg.algo[k] for k in ('tau', 'discount', 'beta', 'tau_target', 'actor_lr', 'critic_lr', 'use_bc_only')})


def build_loader(dataset, cfg, device):
    # Never fork a parent that may already own CUDA/SAPIEN/Vulkan state.
    # Trajectories are resident in RAM: spawn copies them to each worker.
    options = dict(batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                   num_workers=cfg.num_workers, pin_memory=device.type == 'cuda')
    if cfg.num_workers > 0:
        options.update(multiprocessing_context='spawn', persistent_workers=True)
    return DataLoader(dataset, **options)


def train_round(cfg, base, normalizer, episodes, directory, evaluate, save, baseline=None, log_callback=None,
                source_spec=None, sampling_seed=None):
    if cfg.get('updates_per_round') is not None:
        from workflows.iterative_updates import train_updates
        return train_updates(cfg, base, normalizer, episodes, directory, evaluate, save,
                             baseline, log_callback, source_spec,
                             cfg.seed if sampling_seed is None else sampling_seed)
    device = torch.device(cfg.device)
    dataset = TrajectoryDataset(episodes, cfg, normalizer)
    if len(dataset) < cfg.batch_size:
        raise ValueError('Dataset smaller than batch_size; lower batch_size')
    loader = build_loader(dataset, cfg, device)
    # FlowPPOPolicy evaluation adapter freezes the encoder at construction.
    # Re-enable it explicitly for offline updates.
    base.requires_grad_(True)
    freeze_actor_encoder = (
        cfg.get("stage") == "iterative"
        and bool(cfg.get("freeze_actor_encoder", False))
    )
    if freeze_actor_encoder:
        base.encoder.requires_grad_(False)
        base.encoder.eval()
        for parameter in base.encoder.parameters():
            parameter.grad = None

    agent = build_agent(cfg, base, device)
    ema = copy.deepcopy(base).eval().requires_grad_(False)
    best_score = selection_score(baseline) if baseline else None
    selected = None
    directory = Path(directory)
    for epoch in range(1, cfg.epochs + 1):
        base.train()
        if freeze_actor_encoder:
            base.encoder.eval()
        agent.q_net.train()
        agent.v_net.train()
        agent.q_target.eval()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        epoch_started = perf_counter()
        sums = {}
        for batch in loader:
            batch = {k: v.to(device, non_blocking=device.type == 'cuda') for k, v in batch.items()}
            obs = batch_observation(batch)
            nxt = batch_observation(batch, 'next_')
            # Q depends on executed prefix only. Actor retains the full prediction horizon.
            metrics = agent.update_batch_critic(obs, batch['action_chunk'], batch['reward'][:, None],
                    nxt, batch['done'][:, None], batch['discount'][:, None])
            adv = metrics.pop('adv_for_actor')
            if 'object_points' in obs:
                from data.object_diagnostics import batch_object_metrics
                metrics.update(batch_object_metrics(obs, cfg.env.observation))
            if epoch > cfg.critic_warmup_epochs:
                metrics.update(agent.update_actor(obs, batch['action_chunk'], adv=adv))
            else:
                metrics.update({'loss/actor': 0., 'metrics/accept_ratio': 0.})
            with torch.no_grad():
                for a, b in zip(ema.parameters(), base.parameters()):
                    a.lerp_(b, 1-cfg.ema_decay)
                for a, b in zip(ema.buffers(), base.buffers()):
                    a.copy_(b)
                for a, b in zip(agent.q_target.buffers(), agent.q_net.buffers()):
                    a.copy_(b)
            for k, val in metrics.items():
                sums[k] = sums.get(k, 0.) + val
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        train_seconds = perf_counter() - epoch_started
        row = dict(epoch=epoch, **{k: v/len(loader) for k, v in sums.items()})
        row.update({'Time/Train_Seconds': train_seconds, 'Train/Batches': len(loader),
                    'Train/Samples': len(loader)*cfg.batch_size,
                    'Train/Samples_Per_Second': len(loader)*cfg.batch_size/max(train_seconds, 1e-9)})
        result = {}
        evaluate_now = epoch % cfg.eval.every == 0 or epoch == cfg.epochs
        save_now = epoch % cfg.save_epoch == 0 or epoch == cfg.epochs
        candidate = directory/'checkpoints'/f'epoch_{epoch:04d}.pth'
        evaluation_seconds = artifact_seconds = 0.0
        if evaluate_now or save_now:
            artifact_started = perf_counter()
            raw = copy.deepcopy(base.state_dict())
            try:
                base.load_state_dict(ema.state_dict())
                if evaluate_now:
                    evaluation_started = perf_counter()
                    result = evaluate(epoch=epoch)
                    evaluation_seconds = perf_counter() - evaluation_started
                    row.update(result)
                if evaluate_now or save_now:
                    save(candidate, epoch, result)
                    save(directory/'checkpoints'/'last.pth', epoch, result)
                if result and (best_score is None or selection_score(result) > best_score):
                    best_score, selected = selection_score(result), str(candidate)
                    save(directory/'checkpoints'/'best.pth', epoch, result)
                    write_selection(directory, epoch, result, directory/'checkpoints'/'best.pth',
                                    stage=cfg.stage)
            finally:
                base.load_state_dict(raw)
            del raw
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            artifact_seconds = max(0.0, perf_counter() - artifact_started - evaluation_seconds)
        row.update({'Time/Eval_Seconds': evaluation_seconds,
                    'Time/Artifact_Seconds': artifact_seconds,
                    'Time/Epoch_Seconds': perf_counter() - epoch_started})
        context = dict(stage=cfg.stage, sampler=cfg.eval.sampler,
            round=directory.name if cfg.stage == 'iterative' else None,
            checkpoint=str(candidate) if evaluate_now or save_now else None,
            evaluation=str(evaluation_paths(cfg, directory, epoch)[0]) if result else None)
        metrics = {k: v for k, v in row.items() if k != 'epoch'}
        log_metrics(directory, epoch, metrics, **context)
        if cfg.stage == 'iterative':
            log_metrics(directory.parent, epoch, metrics, **context)
        if log_callback:
            log_callback(metrics, epoch)
        print(row, flush=True)
    return selected
