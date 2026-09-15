"""One IDQL round, reusing the existing dictionary adapters."""
import copy
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from models.critics.q_v_network import VNetwork, TwinQNetwork
from algos.embodied_idql import (CriticFeatureExtractor, IDQL_VNet_Wrapper,
    IDQL_QNet_Wrapper, Policy_IDQL_Wrapper, EmbodiedIDQL)
from data.iterative_dataset import IterativeDataset
from data.iterative_store import write_json


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


def train_round(cfg, base, normalizer, episodes, directory, evaluate, save, baseline):
    device = torch.device(cfg.device)
    dataset = IterativeDataset(episodes, cfg, normalizer)
    if len(dataset) < cfg.batch_size:
        raise ValueError('Dataset smaller than batch_size; lower batch_size')
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True,
                        num_workers=cfg.num_workers, drop_last=True)
    # FlowPPOPolicy evaluation adapter freezes the encoder at construction.
    # Re-enable it explicitly for offline updates.
    base.requires_grad_(True)
    v = IDQL_VNet_Wrapper(CriticFeatureExtractor(cfg), VNetwork(cfg.model.cond_dim))
    q = PrefixQ(CriticFeatureExtractor(cfg), TwinQNetwork(state_dim=cfg.model.cond_dim,
                action_dim=cfg.model.action_dim, chunk_size=cfg.env.exec_steps), cfg.env.exec_steps)
    agent = RoundIDQL(Policy_IDQL_Wrapper(base), q, v, device=device,
        **{k: cfg.algo[k] for k in ('tau', 'discount', 'beta', 'tau_target', 'actor_lr', 'critic_lr', 'use_bc_only')})
    ema = copy.deepcopy(base).eval().requires_grad_(False)
    best_rate = baseline['Eval/Success_Rate']
    selected = None
    directory = Path(directory)
    for epoch in range(1, cfg.epochs + 1):
        sums = {}
        for batch in loader:
            base.train()
            agent.q_net.train()
            agent.v_net.train()
            # Target statistics stay fixed; synchronize buffers after each update.
            agent.q_target.eval()
            batch = {k: v.to(device) for k, v in batch.items()}
            obs = dict(pc=batch['pc'], state=batch['state'])
            nxt = dict(pc=batch['next_pc'], state=batch['next_state'])
            # Q depends on executed prefix only. Actor retains the full prediction horizon.
            metrics = agent.update_batch_critic(obs, batch['action_chunk'], batch['reward'][:, None],
                    nxt, batch['done'][:, None], batch['discount'][:, None])
            adv = metrics.pop('adv_for_actor')
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
        row = dict(epoch=epoch, **{k: v/len(loader) for k, v in sums.items()})
        if epoch % cfg.eval.every == 0 or epoch == cfg.epochs:
            raw = copy.deepcopy(base.state_dict())
            base.load_state_dict(ema.state_dict())
            result = evaluate()
            row.update(result)
            candidate = directory/f'offline_ep{epoch}.pth'
            save(candidate, epoch, result)
            if result['Eval/Success_Rate'] > best_rate:
                best_rate, selected = result['Eval/Success_Rate'], str(candidate)
            base.load_state_dict(raw)
        write_json(directory/f'metrics_ep{epoch}.json', row)
        print(row, flush=True)
    return selected
