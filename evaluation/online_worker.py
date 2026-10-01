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
    retries = int(cfg.get('online_training', {}).get('eval_worker_retries', 1))
    if retries < 0:
        raise ValueError('eval_worker_retries must be nonnegative')
    with tempfile.TemporaryDirectory(prefix='myrl-eval-') as directory:
        request, result = Path(directory) / 'request.pth', Path(directory) / 'result.json'
        torch.save(dict(config=OmegaConf.to_container(cfg, resolve=True),
            weights={k: v.detach().cpu() for k, v in weights.items()}, normalizer=normalizer_stats,
            mode=mode, destination=str(destination), video=video, metadata=metadata), request)
        for attempt in range(retries + 1):
            result.unlink(missing_ok=True)
            try:
                subprocess.run([sys.executable, '-m', 'evaluation.online_worker', str(request), str(result)],
                               cwd=Path(__file__).resolve().parents[1], check=True)
            except subprocess.CalledProcessError as exc:
                from utils.experiment import write_json
                write_json(Path(destination) / 'worker_failure.json', dict(
                    attempt=attempt + 1, max_attempts=retries + 1,
                    returncode=exc.returncode, python=sys.executable,
                    metadata=metadata, recovered=False))
                if attempt == retries:
                    raise
                print(f'Evaluation worker failed ({exc.returncode}); retrying in a fresh process '
                      f'({attempt + 2}/{retries + 1}).', flush=True)
                continue
            metrics = json.loads(result.read_text())
            if attempt:
                write_json(Path(destination) / 'worker_failure.json', dict(
                    attempt=attempt + 1, max_attempts=retries + 1,
                    python=sys.executable, metadata=metadata, recovered=True))
            return metrics


def check_imports():
    """Exercise the evaluation import chain without loading a policy or simulator."""
    from importlib import import_module
    from importlib.metadata import version
    print(f'Python: {sys.version}\nExecutable: {sys.executable}', flush=True)
    for module, package in [('wcwidth', 'wcwidth'), ('prompt_toolkit', 'prompt_toolkit'),
                            ('IPython', 'ipython'), ('mani_skill', 'mani_skill'),
                            ('envs.factory', None)]:
        print(f'Importing {module} ...', flush=True)
        loaded = import_module(module)
        print(f'OK: {getattr(loaded, "__file__", None)}'
              + (f' (version {version(package)})' if package else ''), flush=True)
    print('Evaluation imports passed.', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('request', nargs='?'); parser.add_argument('result', nargs='?')
    parser.add_argument('--check-imports', action='store_true')
    args = parser.parse_args()
    if args.check_imports:
        check_imports()
        return
    if args.request is None or args.result is None:
        parser.error('request and result are required unless --check-imports is used')
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
