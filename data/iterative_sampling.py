"""Source-aware fixed-budget sampling; no trajectory arrays or rewards are edited."""
import math

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

GROUPS = ('demo', 'rollout_success', 'rollout_failure')


def validate_update_config(cfg):
    total = cfg.get('updates_per_round')
    mode = cfg.get('actor_sampling', {}).get('mode', 'mixed')
    if mode not in ('mixed', 'demo_success'):
        raise ValueError('actor_sampling.mode must be mixed or demo_success')
    if total is None:
        if mode != 'mixed':
            raise ValueError('demo_success requires updates_per_round')
        return
    if cfg.stage != 'iterative':
        raise ValueError('Fixed update budgets are only supported for iterative')
    for name in ('updates_per_round', 'eval_every_updates', 'save_every_updates', 'log_every_updates'):
        value = cfg.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    warmup = cfg.get('critic_warmup_updates')
    if isinstance(warmup, bool) or not isinstance(warmup, int) or not 0 <= warmup < total:
        raise ValueError('Require 0 <= critic_warmup_updates < updates_per_round')
    sampling = cfg.actor_sampling
    fraction = sampling.demo_fraction
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 < fraction < 1:
        raise ValueError('demo_fraction must be strictly between 0 and 1')
    quota = cfg.batch_size * fraction
    if not math.isclose(quota, round(quota), abs_tol=1e-8) or not 0 < round(quota) < cfg.batch_size:
        raise ValueError('batch_size * demo_fraction must be an integer with both groups nonempty')
    if not sampling.demo_sources or any(not isinstance(x, str) or not x for x in sampling.demo_sources):
        raise ValueError('Explicit nonempty demo_sources names are required')


class SourceDataset(Dataset):
    """Tag the kept episodes, preserving manifest alignment after bounds rejection."""
    def __init__(self, dataset, spec, demo_sources):
        self.dataset = dataset
        sources = spec['sources']
        names = [s.get('name') for s in sources]
        if any(not isinstance(n, str) or not n for n in names):
            raise ValueError('Fixed-update sampling requires named manifest sources')
        if not set(demo_sources).issubset(names):
            raise ValueError('Configured demo_sources are missing from the manifest')
        original_sources = [i for i, s in enumerate(sources) for _ in s['episodes']]
        if len(original_sources) != dataset.original_episode_count:
            raise ValueError('Manifest/episode alignment differs')
        self.source_ids = [original_sources[i] for i in dataset.original_episode_indices]
        self.group_ids = [0 if names[source] in demo_sources else (1 if ep['success'].any() else 2)
                          for source, ep in zip(self.source_ids, dataset.episodes)]
        self.offsets = np.cumsum([0] + [len(ep['action']) for ep in dataset.episodes])
        self.report = dict(
            groups={name: dict(episodes=0, transitions=0) for name in GROUPS},
            sources={f'source_{i:03d}': dict(name=name, role='demo' if name in demo_sources else 'rollout',
                      episodes=0, transitions=0, success_episodes=0) for i, name in enumerate(names)},
            input_episodes=dataset.original_episode_count,
            excluded_episode_indices=sorted(set(range(dataset.original_episode_count)) -
                                             set(dataset.original_episode_indices)))
        for ep, source, group in zip(dataset.episodes, self.source_ids, self.group_ids):
            for summary in (self.report['groups'][GROUPS[group]], self.report['sources'][f'source_{source:03d}']):
                summary['episodes'] += 1
                summary['transitions'] += len(ep['action'])
            self.report['sources'][f'source_{source:03d}']['success_episodes'] += int(ep['success'].any())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row = self.dataset[index]
        episode, _ = self.dataset.indices[index]
        return dict(row, sampling_source=torch.tensor(self.source_ids[episode]),
                    sampling_group=torch.tensor(self.group_ids[episode]))


class FixedBatches(Sampler):
    """Local RNG; exact source quotas, then uniform episode and uniform time.

    Critic/mixed sampling is uniform over all transitions with replacement.
    Actor RNG is independent, so filtering Actor pools cannot change critic indices.
    """
    def __init__(self, dataset, batch_size, updates, seed, mode='mixed', demo_fraction=.5):
        self.dataset, self.batch_size, self.updates = dataset, batch_size, updates
        self.seed, self.mode = seed, mode
        if mode not in ('mixed', 'demo_success'):
            raise ValueError('Unknown sampling mode')
        self.quota = round(batch_size * demo_fraction)
        self.pools = [np.flatnonzero(np.asarray(dataset.group_ids) == i) for i in (0, 1)]
        if mode == 'demo_success' and any(len(pool) == 0 for pool in self.pools):
            raise ValueError('demo_success requires kept demonstrations and successful rollout episodes; '
                             'no silent fallback to failures')

    def __len__(self):
        return self.updates

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        for _ in range(self.updates):
            if self.mode == 'mixed':
                yield rng.integers(len(self.dataset), size=self.batch_size).tolist()
                continue
            batch = []
            for pool, count in zip(self.pools, (self.quota, self.batch_size-self.quota)):
                episodes = rng.choice(pool, size=count, replace=True)
                for episode in episodes:
                    batch.append(int(rng.integers(self.dataset.offsets[episode], self.dataset.offsets[episode+1])))
            rng.shuffle(batch)
            yield batch


class SamplingMetrics:
    """Interval/cumulative counts; per-source losses reuse the exact training pass."""
    def __init__(self, source_count):
        self.labels = list(GROUPS) + [f'source_{i:03d}' for i in range(source_count)]
        self.values = {role: {label: dict(sampled=0, kept=0, advantage_sum=0., loss_sum=0.)
                             for label in self.labels} for role in ('Actor', 'Critic')}

    def add(self, role, batch, details=None):
        groups = batch['sampling_group'].detach().cpu()
        sources = batch['sampling_source'].detach().cpu()
        keep = details['keep_mask'].detach().cpu() if details else torch.ones(len(groups), dtype=torch.bool)
        advantages = details['advantages'].detach().cpu() if details else None
        losses = details['losses'].detach().cpu() if details else None
        for label in self.labels:
            mask = (groups == GROUPS.index(label)) if label in GROUPS else (sources == int(label[7:]))
            row = self.values[role][label]
            row['sampled'] += int(mask.sum())
            row['kept'] += int((mask & keep).sum())
            if details:
                row['advantage_sum'] += float(advantages[mask].sum())
                row['loss_sum'] += float(losses[mask[keep]].sum())

    def report(self):
        result = {}
        for role, groups in self.values.items():
            for label, row in groups.items():
                prefix = f'{role}/{label}/'
                result[prefix+'sampled'] = row['sampled']
                if role == 'Actor':
                    result[prefix+'kept'] = row['kept']
                    result[prefix+'keep_fraction'] = row['kept']/row['sampled'] if row['sampled'] else None
                    result[prefix+'advantage_mean'] = row['advantage_sum']/row['sampled'] if row['sampled'] else None
                    result[prefix+'loss_mean'] = row['loss_sum']/row['kept'] if row['kept'] else None
        return result
