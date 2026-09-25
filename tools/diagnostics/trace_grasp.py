"""Reproduce StackCube evaluation with per-control-step actions and simulator state."""
import argparse
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import subprocess

import torch
from omegaconf import OmegaConf

from data.episodes import digest
from evaluation.grasp_trace import GraspTrace
from evaluation.runner import evaluate_policy
from models.checkpoint import policy_weights
from models.factory import build_base, observation_encoder
from models.online_policy import FlowPPOPolicy
from utils.experiment import write_json
from utils.normalizer import MinMaxNormalizer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='outputs/StackCube-v1/oc_budget/iterative/checkpoints/best.pth')
    parser.add_argument('--seeds', type=int, nargs='+', default=[6009, 6014, 6019])
    parser.add_argument('--sampler', choices=['cps', 'ode'], default='cps')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', default='outputs/StackCube-v1/oc_budget/grasp_diagnostics_best')
    parser.add_argument('--no-video', action='store_true')
    return parser.parse_args()


def run(args):
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Seeds must be unique')
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f'{output}: choose a new --output directory')
    cp = torch.load(checkpoint, map_location=args.device, weights_only=True)
    cfg = OmegaConf.create(cp['config'])
    if cfg.env.env_id != 'StackCube-v1' or cfg.env.control_mode != 'pd_ee_delta_pose':
        raise ValueError('This diagnostic supports StackCube-v1 / pd_ee_delta_pose only')
    cfg.device, cfg.eval.sampler = args.device, args.sampler
    cfg.noise_level = cfg.get('noise_level', cfg.algo.get('noise_level', .7))
    cfg.min_std = cfg.get('min_std', cfg.algo.get('min_std', .0067))
    normalizer = MinMaxNormalizer()
    normalizer.stats = cp['normalizer']
    base = build_base(cfg, args.device)
    base.load_state_dict(policy_weights(cp), strict=True)
    actor = FlowPPOPolicy(base, num_steps=cfg.model.num_inference_steps,
                         noise_level=cfg.noise_level, min_std=cfg.min_std, eval_mode=args.sampler)
    encode = observation_encoder(cfg, actor, normalizer, args.device)
    from envs.factory import make_env
    holder = []
    def wrap(env):
        trace = GraspTrace(env, output, normalizer, cfg.env.exec_steps)
        holder.append(trace)
        return trace
    def factory():
        video = None if args.no_video else dict(directory=str(output/'videos'), episodes=len(args.seeds))
        return make_env(cfg, primitive_wrapper=wrap, video=video)
    output.mkdir(parents=True)
    OmegaConf.save(cfg, output/'config.yaml', resolve=True)
    packages = {}
    for name in ('mani_skill', 'sapien', 'torch', 'gymnasium'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
            cwd=Path(__file__).resolve().parents[2], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    metadata = dict(schema='myrl_grasp_trace_v1', checkpoint=str(checkpoint),
                    checkpoint_sha256=digest(checkpoint), seeds=args.seeds, sampler=args.sampler,
                    device=args.device, versions=packages, git_commit=revision,
                    weight_key='ema_model_state_dict' if 'ema_model_state_dict' in cp else 'model_state_dict',
                    diagnostic_only=True, video_requested=not args.no_video)
    write_json(output/'provenance.json', metadata)
    try:
        metrics = evaluate_policy(factory, actor, encode, normalizer, args.seeds,
            cfg.model.num_inference_steps, output_dir=output, metadata=metadata,
            action_callback=lambda action: holder[0].record_action(action))
        video_errors = {str(p.relative_to(output)): p.read_text() for p in output.rglob('video_error.txt')}
        write_json(output/'run_status.json', dict(complete=True, video_errors=video_errors))
    except Exception as exc:
        write_json(output/'run_status.json', dict(complete=False, error=f'{type(exc).__name__}: {exc}'))
        raise
    print(f'Diagnostics: {output}', flush=True)
    print(metrics, flush=True)
    return metrics


if __name__ == '__main__':
    run(parse_args())
