import numpy as np
import torch
from torch.utils.data import Dataset
from data.episodes import transition, validate_episode
from data.observations import (observation_fields, normalize_observation,
                               validate_observation, batch_observation)


class TrajectoryDataset(Dataset):
    def __init__(self, episodes, cfg, normalizer):
        # Resolve config once in the main process.
        self.chunk_size = int(cfg.model.chunk_size)
        self.exec_steps = int(cfg.env.exec_steps)
        self.discount = float(cfg.algo.discount)
        self.workspace_bounds = np.asarray(
            [list(row) for row in cfg.env.workspace_bounds],
            dtype=np.float32,
        ).copy()

        if (
            self.workspace_bounds.shape != (2, 3)
            or not np.isfinite(self.workspace_bounds).all()
            or np.any(self.workspace_bounds[1] <= self.workspace_bounds[0])
        ):
            raise ValueError('Invalid workspace_bounds')
        self.normalizer = normalizer

        lo = np.asarray(normalizer.stats['action']['min'])
        hi = np.asarray(normalizer.stats['action']['max'])

        self.original_episode_count = len(episodes)
        self.original_episode_indices = []
        kept = []
        rejected = []

        for index, ep in enumerate(episodes):
            validate_episode(ep)
            if 'actor_eligible' in ep and (
                cfg.get('stage') != 'iterative' or cfg.get('updates_per_round') is None
                or cfg.get('actor_sampling', {}).get('mode') != 'demo_success_correction'
            ):
                raise ValueError('Correction data requires fixed-budget demo_success_correction sampling')
            validate_observation({k: ep[k][0] for k in observation_fields(ep)}, cfg)
            if 'object_points' in ep:
                from data.object_centric import validate_object_arrays
                validate_object_arrays(ep, cfg.env.observation, cfg.model.in_channels)

            if (
                ep['state'].shape[1] != cfg.model.state_dim
                or ep['action'].shape[1] != cfg.model.action_dim
            ):
                raise ValueError('State/action dimension mismatch')

            actions = ep['action']
            if not np.isfinite(actions).all():
                raise ValueError(f'Episode {index}: nonfinite actions')

            outside = (actions < lo - 1e-5) | (actions > hi + 1e-5)

            if outside.any():
                excess = np.maximum(
                    np.maximum(lo - actions, actions - hi), 0.0
                )
                rejected.append((index, float(excess.max())))
            else:
                kept.append(ep)
                self.original_episode_indices.append(index)

        # Explicit policy for out-of-range episodes with frozen normalization.
        if (
            len(rejected) > cfg.dataset.get('max_rejected_episodes', 3)
            or len(rejected) / max(len(episodes), 1) > cfg.dataset.get('max_rejected_fraction', 0.01)
        ):
            raise ValueError(
                f'Too many out-of-bounds episodes: '
                f'{len(rejected)}/{len(episodes)}. '
                'Inspect dataset/normalizer compatibility.'
            )

        for index, excess in rejected:
            print(
                f'[TrajectoryDataset] Excluding episode index={index}, '
                f'max_action_excess={excess:.8g}',
                flush=True,
            )

        self.episodes = kept
        self.indices = [
            (e, t)
            for e, ep in enumerate(self.episodes)
            for t in range(len(ep['action']))
        ]

        if not self.indices:
            raise ValueError('No valid training transitions remain')

        print(
            f'[TrajectoryDataset] Using {len(kept)}/{len(episodes)} episodes, '
            f'{len(self.indices)} transitions',
            flush=True,
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        e, t = self.indices[index]
        norm = self.normalizer

        row = transition(
            self.episodes[e],
            t,
            self.chunk_size,
            self.exec_steps,
            self.discount,
        )

        for prefix in ('', 'next_'):
            obs = normalize_observation(batch_observation(row, prefix), norm, self.workspace_bounds)
            row.update({prefix + k: v for k, v in obs.items()})

        row['action_chunk'] = norm.normalize(
            row['action_chunk'], 'action'
        )

        return {
            k: torch.as_tensor(
                np.array(v, copy=True), dtype=(torch.bool if 'mask' in k or k.endswith('object_valid')
                    else torch.int64 if k.endswith('object_roles') else torch.float32)
            )
            for k, v in row.items()
        }
