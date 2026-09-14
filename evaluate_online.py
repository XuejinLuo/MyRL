"""Evaluate an online checkpoint using its saved architecture, statistics and sampler."""
import argparse
import json
import torch
from omegaconf import OmegaConf
from utils.online_checkpoint import FORMAT
from train_online import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default=None)
    parser.add_argument('--seeds', nargs='+', type=int, default=None)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint.get('format') != FORMAT:
        raise ValueError('Use train_online.py eval_only=true for an offline checkpoint')
    cfg = OmegaConf.create(checkpoint['config'])
    cfg.algo.pretrained_ckpt = args.checkpoint
    cfg.algo.pretrained_weight_key = 'model_state_dict'
    cfg.resume = None
    cfg.eval_only = True
    cfg.wandb.enable = False
    if args.device:
        cfg.device = args.device
    if args.seeds:
        cfg.eval.seeds = args.seeds
    run(cfg)


if __name__ == '__main__':
    main()
