"""One checkpoint evaluator for all three stages (CPS and ODE kept separate)."""
import argparse
import csv
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', nargs='+', required=True)
    p.add_argument('--labels', nargs='+')
    p.add_argument('--output', required=True)
    p.add_argument('--seed-start', type=int, default=3000)
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--samplers', nargs='+', choices=['cps', 'ode'], default=['cps', 'ode'])
    p.add_argument('--video-episodes', type=int, default=5)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if args.episodes < 1 or args.video_episodes < 0:
        p.error('episodes must be positive; video-episodes must be nonnegative')
    labels = args.labels or [f'checkpoint_{i}' for i in range(len(args.checkpoint))]
    if len(labels) != len(args.checkpoint) or len(set(labels)) != len(labels):
        p.error('Supply one unique label per checkpoint')
    if any(Path(label).name != label or label in ('.', '..') for label in labels):
        p.error('Labels must be simple directory names')
    import torch
    from omegaconf import OmegaConf
    from models.policy import EmbodiedGenPolicy
    from utils.normalizer import MinMaxNormalizer
    from utils.online_checkpoint import policy_weights
    from utils.experiment import evaluate_base, write_json
    from data.iterative_store import digest
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    rows, protocol = [], None
    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    for label, path in zip(labels, args.checkpoint):
        path = Path(path).resolve()
        cp = torch.load(path, map_location=args.device, weights_only=True)
        if 'config' not in cp or 'normalizer' not in cp:
            raise ValueError('Use checkpoints from the updated training entry points (config + normalizer required)')
        cfg = OmegaConf.create(cp['config'])
        cfg.device = args.device
        cfg.noise_level = cfg.get('noise_level', cfg.algo.get('noise_level', .7))
        cfg.min_std = cfg.get('min_std', cfg.algo.get('min_std', .0067))
        current = dict(env=cp['config']['env'], model=cp['config']['model'],
                       normalizer=cp['normalizer'], noise_level=cfg.noise_level, min_std=cfg.min_std)
        if protocol is not None and protocol != current:
            raise ValueError('Cannot compare incompatible environment/model/normalizer/sampler settings')
        protocol = current
        if set(seeds) & set(cfg.eval.seeds):
            raise ValueError('Final test seeds overlap checkpoint-selection seeds')
        if 'collect_seed_start' in cfg:
            collection = set(range(cfg.collect_seed_start, cfg.collect_seed_start + cfg.rounds*cfg.episodes_per_round))
            if collection & set(seeds):
                raise ValueError('Final test seeds overlap iterative collection seeds')
        cfg.video = dict(every=1 if args.video_episodes else 0, episodes=args.video_episodes)
        base = EmbodiedGenPolicy(**{k: cfg.model[k] for k in (
            'in_channels', 'action_dim', 'chunk_size', 'use_state', 'state_dim',
            'encoder_type', 'backbone_type', 'cond_dim', 'algo_type')}).to(args.device)
        base.load_state_dict(policy_weights(cp), strict=True)
        normalizer = MinMaxNormalizer()
        normalizer.stats = cp['normalizer']
        for sampler in args.samplers:
            cfg.eval.sampler = sampler
            result = evaluate_base(cfg, base, normalizer, out/label, 0, tag='test', seeds=seeds)
            rows.append(dict(stage=label, sampler=sampler, checkpoint=str(path),
                             checkpoint_sha256=digest(path), **result))
        write_json(out/'summary.json', dict(protocol=protocol, test_seeds=seeds, results=rows))
        with (out/'summary.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
