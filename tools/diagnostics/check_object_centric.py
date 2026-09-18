"""Inspect the exact training preprocessing on a small number of real H5 episodes."""
import argparse
import json
from pathlib import Path
import numpy as np
from hydra import compose, initialize_config_dir
from data.demonstrations import load_demonstrations
from data.object_diagnostics import summarize_objects


def inspect_objects(cfg, max_episodes=10, seed=42):
    if max_episodes < 1:
        raise ValueError('max_episodes must be positive')
    cfg.dataset.max_episodes = max_episodes
    cfg.env.observation.mode = 'object_centric'
    np.random.seed(seed)
    episodes = load_demonstrations(cfg)
    result = dict(episodes=len(episodes), frames=sum(len(ep['state']) for ep in episodes),
                  seed=seed, metrics=summarize_objects(episodes, cfg.env.observation))
    print(json.dumps(result, indent=2, allow_nan=False))
    for obj in cfg.env.observation.objects:
        if result['metrics'][f"Object/{obj['name']}/valid_rate"] == 0:
            print(f"WARNING: {obj['name']} never visible. Verify H5 ID={obj['h5_id']}, workspace and camera.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path')
    parser.add_argument('--max-episodes', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    with initialize_config_dir(version_base=None, config_dir=str(Path(__file__).resolve().parents[2]/'configs')):
        cfg = compose(config_name='config')
    if args.data_path:
        cfg.dataset.data_path = args.data_path
    inspect_objects(cfg, args.max_episodes, args.seed)


if __name__ == '__main__':
    main()
