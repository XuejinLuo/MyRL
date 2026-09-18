"""Small H5 BC sanity check, without simulator or evaluation/checkpoint selection.

Fits a single batch under fixed flow noise/time. A falling loss demonstrates
optimization/data plumbing only, not task success or generalization.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from torch.utils.data import DataLoader
from data.demonstrations import load_demonstrations
from data.dataset import TrajectoryDataset
from data.observations import batch_observation
from models.factory import build_base
from utils.normalizer import MinMaxNormalizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path')
    parser.add_argument('--max-episodes', type=int, default=2)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if min(args.max_episodes, args.steps, args.batch_size) < 1:
        parser.error('Counts must be positive')
    with initialize_config_dir(version_base=None, config_dir=str(Path(__file__).resolve().parents[2]/'configs')):
        cfg = compose(config_name='train_offline')
    cfg.dataset.max_episodes = args.max_episodes
    cfg.env.observation.mode = 'object_centric'
    if args.data_path:
        cfg.dataset.data_path = args.data_path
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    episodes = load_demonstrations(cfg)
    norm = MinMaxNormalizer()
    norm.fit({k: np.concatenate([ep[k] for ep in episodes]) for k in ('state', 'action')})
    ds = TrajectoryDataset(episodes, cfg, norm)
    batch = next(iter(DataLoader(ds, batch_size=min(args.batch_size, len(ds)), shuffle=True)))
    batch = {k: v.to(args.device) for k, v in batch.items()}
    obs = batch_observation(batch)
    policy = build_base(cfg, args.device).train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    losses = []
    for step in range(args.steps + 1):
        torch.manual_seed(cfg.seed + 1)
        loss = policy.compute_loss(obs, batch['action_chunk'])
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite BC loss')
        losses.append(loss.item())
        if step % 10 == 0 or step == args.steps:
            print(f'step={step} fixed_batch_flow_loss={loss.item():.6f}', flush=True)
        if step < args.steps:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    if losses[-1] >= losses[0]:
        raise RuntimeError('BC sanity loss did not decrease; inspect inputs/optimization')


if __name__ == '__main__':
    main()
