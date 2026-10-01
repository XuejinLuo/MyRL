"""Isolated CPU benchmark for a GPU-simulation training process.

SAPIEN/PhysX backend initialization is process-wide. Do not create the historical
CPU benchmark in the process already running GPU physics.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import torch
from omegaconf import OmegaConf


def isolated_evaluate(cfg, weights, normalizer_stats, mode, destination, video, metadata):
    with tempfile.TemporaryDirectory(prefix='myrl-eval-') as directory:
        request, result = Path(directory) / 'request.pth', Path(directory) / 'result.json'
        torch.save(dict(config=OmegaConf.to_container(cfg, resolve=True),
            weights={k: v.detach().cpu() for k, v in weights.items()}, normalizer=normalizer_stats,
            mode=mode, destination=str(destination), video=video, metadata=metadata), request)
        subprocess.run([sys.executable, '-m', 'evaluation.online_worker', str(request), str(result)],
                       cwd=Path(__file__).resolve().parents[1], check=True)
        return json.loads(result.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('request'); parser.add_argument('result')
    args = parser.parse_args()
    request = torch.load(args.request, map_location='cpu', weights_only=True)
    from models.factory import build_base, observation_encoder
    from models.checkpoint import make_actor
    from utils.normalizer import MinMaxNormalizer
    from envs.factory import make_env
    from evaluation.runner import evaluate_policy
    cfg = OmegaConf.create(request['config'])
    base = build_base(cfg, cfg.device)
    base.load_state_dict(request['weights'], strict=True)
    actor = make_actor(base, cfg)
    actor.eval_mode = request['mode']
    normalizer = MinMaxNormalizer(); normalizer.stats = request['normalizer']
    result = evaluate_policy(lambda: make_env(cfg, video=request['video']), actor,
        observation_encoder(cfg, actor, normalizer, cfg.device), normalizer,
        list(cfg.eval.seeds), cfg.model.num_inference_steps,
        output_dir=request['destination'], metadata=request['metadata'])
    Path(args.result).write_text(json.dumps(result))


if __name__ == '__main__':
    main()
