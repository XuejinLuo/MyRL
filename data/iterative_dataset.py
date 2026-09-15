import numpy as np
import torch
from torch.utils.data import Dataset
from data.iterative_store import transition


class IterativeDataset(Dataset):
    def __init__(self, episodes, cfg, normalizer):
        self.episodes, self.cfg, self.normalizer = episodes, cfg, normalizer
        self.indices = [(e, t) for e, ep in enumerate(episodes) for t in range(len(ep['action']))]
        for ep in episodes:
            if ep['pc'].shape[1:] != (cfg.env.num_points, cfg.model.in_channels):
                raise ValueError('Point cloud shape does not match config')
            if ep['state'].shape[1] != cfg.model.state_dim or ep['action'].shape[1] != cfg.model.action_dim:
                raise ValueError('State/action dimension mismatch')
            # Frozen normalization must not silently change executed action labels.
            lo = np.asarray(normalizer.stats['action']['min'])
            hi = np.asarray(normalizer.stats['action']['max'])
            if ((ep['action'] < lo-1e-5) | (ep['action'] > hi+1e-5)).any():
                raise ValueError('Actions outside frozen training bounds; inspect collection scaling')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        e, t = self.indices[index]
        cfg, norm = self.cfg, self.normalizer
        row = transition(self.episodes[e], t, cfg.model.chunk_size, cfg.env.exec_steps, cfg.algo.discount)
        for k in ('pc', 'next_pc'):
            row[k] = norm.center_point_cloud(row[k], np.asarray(cfg.env.workspace_bounds))
        for k in ('state', 'next_state'):
            row[k] = norm.normalize(row[k], 'state')
        row['action_chunk'] = norm.normalize(row['action_chunk'], 'action')
        return {k: torch.as_tensor(np.array(v, copy=True), dtype=torch.float32) for k, v in row.items()}
