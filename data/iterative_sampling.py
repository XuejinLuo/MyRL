"""Source-aware fixed-budget sampling; no trajectory arrays or rewards are edited."""
import math
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

GROUPS = ('demo', 'rollout_success', 'rollout_failure', 'correction')


def validate_update_config(cfg):
    total = cfg.get('updates_per_round')
    mode = cfg.get('actor_sampling', {}).get('mode', 'mixed')
    if mode not in ('mixed', 'demo_success', 'demo_success_correction'):
        raise ValueError('Unknown actor_sampling.mode')
    if total is None:
        if mode != 'mixed':
            raise ValueError('Source-balanced Actor sampling requires updates_per_round')
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
    if sampling.get('correction_sampling', 'episode') not in ('episode', 'uniform_start'):
        raise ValueError('Unknown correction_sampling')
    if sampling.get('correction_label_mode', 'full_chunk') not in ('full_chunk', 'masked'):
        raise ValueError('Unknown correction_label_mode')
    if sampling.get('correction_label_mode', 'full_chunk') == 'masked' and cfg.model.algo_type != 'flow':
        raise ValueError('Masked correction supervision requires model.algo_type=flow')
    early = cfg.get('eval_actor_updates', [])
    if any(type(n) is not int or n < 1 or n > total-warmup for n in early) or len(set(early)) != len(early):
        raise ValueError('eval_actor_updates must contain unique positive Actor update counts within budget')
    fraction = sampling.demo_fraction
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 < fraction < 1:
        raise ValueError('demo_fraction must be strictly between 0 and 1')
    quota = cfg.batch_size * fraction
    if not math.isclose(quota, round(quota), abs_tol=1e-8) or not 0 < round(quota) < cfg.batch_size:
        raise ValueError('batch_size * demo_fraction must be an integer with both groups nonempty')
    if not sampling.demo_sources or any(not isinstance(x, str) or not x for x in sampling.demo_sources):
        raise ValueError('Explicit nonempty demo_sources names are required')
    if mode == 'demo_success_correction':
        correction = sampling.get('correction_fraction', .25)
        if isinstance(correction, bool) or not isinstance(correction, (int, float)) or not 0 < correction < 1-fraction:
            raise ValueError('Require positive demo, success and correction fractions')
        if not math.isclose(cfg.batch_size*correction, round(cfg.batch_size*correction), abs_tol=1e-8) or round(cfg.batch_size*correction) < 1:
            raise ValueError('batch_size * correction_fraction must be a positive integer')


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
        self.group_ids = [3 if 'actor_eligible' in ep else
                          0 if names[source] in demo_sources else (1 if ep['success'].any() else 2)
                          for source, ep in zip(self.source_ids, dataset.episodes)]
        self.offsets = np.cumsum([0] + [len(ep['action']) for ep in dataset.episodes])
        self.correction_starts = {}
        self.correction_episodes = {}
        original_items = [item for source in sources for item in source['episodes']]
        for i, ep in enumerate(dataset.episodes):
            if 'actor_eligible' in ep:
                if sources[self.source_ids[i]].get('role') != 'human_correction':
                    raise ValueError('Correction episode requires human_correction source role')
                if sources[self.source_ids[i]].get('chunk_size') != dataset.chunk_size:
                    raise ValueError('Correction chunk_size changed; prepare data again')
                source = sources[self.source_ids[i]]
                label_mode = source.get('actor_label_mode', 'full_chunk')
                schema = source.get('actor_label_schema')
                if schema not in (None, 'myrl_actor_labels_v2'):
                    raise ValueError('Unsupported actor label schema')
                if label_mode != dataset.correction_label_mode:
                    raise ValueError('Correction label mode differs from config; select matching data or re-export')
                if (schema is not None or label_mode == 'masked') and 'actor_valid_length' not in ep:
                    raise ValueError('Versioned correction labels require actor_valid_length')
                starts = np.flatnonzero(ep['actor_eligible'])
                lengths = (ep['actor_valid_length'][starts] if 'actor_valid_length' in ep
                           else np.full(len(starts), dataset.chunk_size))
                if (np.any(lengths > dataset.chunk_size) or np.any(lengths < source.get('min_valid_length', 1))
                        or (label_mode == 'full_chunk' and np.any(lengths != dataset.chunk_size))):
                    raise ValueError('Correction valid lengths disagree with label mode/chunk_size')
                if np.any(starts + lengths > len(ep['action'])):
                    raise ValueError('Correction chunk crosses episode end')
                item = original_items[dataset.original_episode_indices[i]]
                self.correction_episodes[str(i)] = dict(seed=item.get('seed'), source_id=self.source_ids[i],
                    actor_starts=len(starts), label_mode=label_mode,
                    full_chunk_starts=int((lengths == dataset.chunk_size).sum()),
                    partial_chunk_starts=int((lengths < dataset.chunk_size).sum()))
                if len(starts):
                    self.correction_starts[i] = starts + self.offsets[i]
            elif sources[self.source_ids[i]].get('role') == 'human_correction':
                raise ValueError('Human correction episode is missing actor_eligible')
        self.correction_pool = (np.concatenate(list(self.correction_starts.values()))
                                if self.correction_starts else np.empty(0, dtype=np.int64))
        self.report = dict(
            groups={name: dict(episodes=0, transitions=0) for name in GROUPS},
            sources={f'source_{i:03d}': dict(name=name, role=sources[i].get('role', 'demo' if name in demo_sources else 'rollout'),
                      episodes=0, transitions=0, success_episodes=0) for i, name in enumerate(names)},
            input_episodes=dataset.original_episode_count,
            excluded_episode_indices=sorted(set(range(dataset.original_episode_count)) -
                                             set(dataset.original_episode_indices)))
        self.report['correction_actor_starts'] = sum(map(len, self.correction_starts.values()))
        self.report['correction_actor_episodes'] = len(self.correction_starts)
        self.report['correction_episodes'] = self.correction_episodes
        for ep, source, group in zip(dataset.episodes, self.source_ids, self.group_ids):
            for summary in (self.report['groups'][GROUPS[group]], self.report['sources'][f'source_{source:03d}']):
                summary['episodes'] += 1
                summary['transitions'] += len(ep['action'])
            self.report['sources'][f'source_{source:03d}']['success_episodes'] += int(ep['success'].any())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row = self.dataset[index]
        episode, start = self.dataset.indices[index]
        return dict(row, sampling_source=torch.tensor(self.source_ids[episode]),
                    sampling_group=torch.tensor(self.group_ids[episode]),
                    sampling_episode=torch.tensor(episode), sampling_start=torch.tensor(start))


class FixedBatches(Sampler):
    """Local RNG; exact quotas, unchanged demo/rollout episode/time sampling.

    Critic/mixed sampling is uniform over all transitions with replacement.
    Actor RNG is independent, so filtering Actor pools cannot change critic indices.
    Correction sampling explicitly selects legacy episode or uniform-start mass.
    """
    def __init__(self, dataset, batch_size, updates, seed, mode='mixed', demo_fraction=.5,
                 correction_fraction=.25, correction_sampling='episode'):
        self.dataset, self.batch_size, self.updates = dataset, batch_size, updates
        self.seed, self.mode = seed, mode
        if correction_sampling not in ('episode', 'uniform_start'):
            raise ValueError('Unknown correction_sampling')
        self.correction_sampling = correction_sampling
        if mode not in ('mixed', 'demo_success', 'demo_success_correction'):
            raise ValueError('Unknown sampling mode')
        self.quota = round(batch_size * demo_fraction)
        self.pools = [np.flatnonzero(np.asarray(dataset.group_ids) == i) for i in (0, 1)]
        if mode != 'mixed' and any(len(pool) == 0 for pool in self.pools):
            raise ValueError('demo_success requires kept demonstrations and successful rollout episodes; '
                             'no silent fallback to failures')
        self.counts = [self.quota, batch_size-self.quota]
        if mode == 'demo_success_correction':
            correction_quota = round(batch_size*correction_fraction)
            self.counts = [self.quota, batch_size-self.quota-correction_quota, correction_quota]
            self.pools.append(np.asarray(list(dataset.correction_starts), dtype=int))
            if not len(self.pools[-1]) or min(self.counts) < 1:
                raise ValueError('Need approved correction starts and nonempty quotas')

    def __len__(self):
        return self.updates

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        for _ in range(self.updates):
            if self.mode == 'mixed':
                yield rng.integers(len(self.dataset), size=self.batch_size).tolist()
                continue
            batch = []
            for group, (pool, count) in enumerate(zip(self.pools, self.counts)):
                if group == 2 and self.correction_sampling == 'uniform_start':
                    batch.extend(rng.choice(self.dataset.correction_pool, size=count, replace=True).tolist())
                    continue
                episodes = rng.choice(pool, size=count, replace=True)
                for episode in episodes:
                    batch.append(int(rng.choice(self.dataset.correction_starts[episode])) if group == 2 else
                                 int(rng.integers(self.dataset.offsets[episode], self.dataset.offsets[episode+1])))
            rng.shuffle(batch)
            yield batch


class SamplingMetrics:
    """Interval/cumulative counts; per-source losses reuse the exact training pass."""
    def __init__(self, source_count):
        self.labels = list(GROUPS) + [f'source_{i:03d}' for i in range(source_count)]
        self.values = {role: {label: dict(sampled=0, kept=0, advantage_sum=0., loss_sum=0.)
                             for label in self.labels} for role in ('Actor', 'Critic')}
        self.advantage_counts = Counter()
        self.correction_visits = {}
        self.valid_lengths = Counter()

    def add(self, role, batch, details=None):
        groups = batch['sampling_group'].detach().cpu()
        sources = batch['sampling_source'].detach().cpu()
        keep = details['keep_mask'].detach().cpu() if details else torch.ones(len(groups), dtype=torch.bool)
        advantages = details['advantages'].detach().cpu() if details else None
        losses = details['losses'].detach().cpu() if details else None
        advantage_valid = (details.get('advantage_valid', torch.ones_like(keep)).detach().cpu()
                           if details else None)
        if role == 'Actor' and 'sampling_episode' in batch:
            correction = batch['sampling_group'] == 3
            for episode, start, length in zip(batch['sampling_episode'][correction].cpu().tolist(),
                    batch['sampling_start'][correction].cpu().tolist(),
                    batch['actor_mask'][correction].sum(1).cpu().tolist()
                    if 'actor_mask' in batch else []):
                self.correction_visits.setdefault(str(episode), Counter())[start] += 1
                self.valid_lengths[int(length)] += 1
        for label in self.labels:
            mask = (groups == GROUPS.index(label)) if label in GROUPS else (sources == int(label[7:]))
            row = self.values[role][label]
            row['sampled'] += int(mask.sum())
            row['kept'] += int((mask & keep).sum())
            if details:
                row['advantage_sum'] += float(advantages[mask & advantage_valid].sum())
                self.advantage_counts[label] += int((mask & advantage_valid).sum())
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
                    count = self.advantage_counts[label]
                    result[prefix+'advantage_samples'] = count
                    result[prefix+'advantage_mean'] = row['advantage_sum']/count if count else None
                    result[prefix+'loss_mean'] = row['loss_sum']/row['kept'] if row['kept'] else None
        return result

    def correction_report(self, episodes):
        rows = {}
        for episode, metadata in episodes.items():
            visits = self.correction_visits.get(episode, Counter())
            rows[episode] = dict(**metadata, sampled=sum(visits.values()),
                unique_starts=len(visits), max_start_repeats=max(visits.values(), default=0),
                start_counts={str(k): v for k, v in sorted(visits.items())})
        total = sum(self.valid_lengths.values())
        return dict(episodes=rows, valid_length_histogram={str(k): v for k, v in sorted(self.valid_lengths.items())},
                    valid_length_mean=sum(k*v for k, v in self.valid_lengths.items())/total if total else None)
