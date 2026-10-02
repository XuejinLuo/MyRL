"""Compare one checkpoint in CPU benchmark / CPU training / GPU training paths.

Run each backend in a separate interpreter (PhysX initialization is global).
GPU uses one lane to isolate backend differences; this does not certify 32-lane
partial resets. Same seeds are checked against actual initial privileged states,
not assumed to imply identical scenes or physics trajectories.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

MODES = ('cpu_benchmark', 'cpu_training', 'gpu_training')


class SingleGPUView:
    """Expose the existing GPU adapter's single lane to the audit loop."""
    def __init__(self, env):
        if env.num_envs != 1:
            raise ValueError('Audit requires one GPU lane')
        self.env = env

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def reset(self, seed):
        obs, infos = self.env.reset(seed=seed)
        return obs[0], infos[0]

    def step(self, actions):
        obs, rewards, terms, truncs, infos = self.env.step(actions[None])
        return obs[0], float(rewards[0]), bool(terms[0]), bool(truncs[0]), infos[0]

    def close(self):
        self.env.close()


def audit_episodes(env, actor, encode, normalizer, seeds, horizon, initial_state):
    import numpy as np
    import torch
    from evaluation.runner import seed_all
    rows = []
    for seed in seeds:
        seed_all(seed)
        obs, _ = env.reset(seed=seed)
        state = np.asarray(initial_state(env), dtype=float).reshape(-1).tolist()
        pc = np.asarray(obs['point_cloud'])
        if not np.isfinite(pc).all():
            raise FloatingPointError('Nonfinite audit observation')
        observation = dict(shape=list(pc.shape), mean=pc.mean(axis=0).tolist(),
                           std=pc.std(axis=0).tolist())
        steps, success, total_reward, first_success = 0, False, 0., None
        first_actions = None
        while True:
            with torch.no_grad():
                actions = actor.sample(encode(obs))[0].cpu().numpy()
            if not np.isfinite(actions).all():
                raise FloatingPointError('Nonfinite audit action')
            obs, reward, term, trunc, info = env.step(normalizer.unnormalize(actions, 'action'))
            actual = int(info['actual_steps'])
            if actual < 1 or steps + actual > horizon:
                raise RuntimeError('Audit environment exceeded its primitive-step horizon')
            if first_actions is None:
                first_actions = actions[:actual].tolist()
            steps += actual
            success |= bool(info.get('success_any', info.get('success', False)))
            if success and first_success is None:
                first_success = steps  # chunk-end upper bound, not exact primitive index
            total_reward += float(reward)
            if term or trunc:
                break
            if steps >= horizon:
                raise RuntimeError('Audit environment did not report its time limit')
        rows.append(dict(seed=seed, success=int(success), primitive_steps=steps,
            first_success_chunk_end=first_success, episode_return=total_reward,
            initial_state=state, initial_pointcloud=observation,
            first_normalized_actions=first_actions))
        print(f'Audit seed={seed} success={int(success)} steps={steps}', flush=True)
    return rows


def summarize(reports):
    """Match by seed. Report discrepancies, never declare physical equivalence."""
    reference = reports['cpu_benchmark']['episodes']
    comparisons = {}
    for mode in MODES[1:]:
        other = reports[mode]['episodes']
        if [r['seed'] for r in reference] != [r['seed'] for r in other]:
            raise ValueError('Audit worker seeds differ')
        differences = []
        for a, b in zip(reference, other):
            if len(a['initial_state']) != len(b['initial_state']):
                raise ValueError('Audit initial-state dimensions differ')
            differences.append(max(abs(x-y) for x, y in zip(a['initial_state'], b['initial_state'])))
        comparisons[mode] = dict(
            success_rate=sum(r['success'] for r in other) / len(other),
            success_disagreements=sum(a['success'] != b['success'] for a, b in zip(reference, other)),
            max_initial_state_abs_difference=max(differences),
            initial_states_over_1e_5=sum(d > 1e-5 for d in differences))
    return dict(schema='myrl_backend_audit_v1',
        checkpoint=reports['cpu_benchmark']['checkpoint'],
        checkpoint_sha256=reports['cpu_benchmark']['checkpoint_sha256'],
        episodes=len(reference), seeds=[r['seed'] for r in reference], gpu_num_envs=1,
        cpu_benchmark_success_rate=sum(r['success'] for r in reference) / len(reference),
        comparisons=comparisons,
        interpretation='Diagnostic only, not target acceptance. Rewards and episode lengths differ '
                       'because training ends on first success. Same seeds need not mean equal physics.')


def worker(args):
    import torch
    from omegaconf import OmegaConf
    from models.checkpoint import load_checkpoint, policy_weights, make_actor
    from models.factory import build_base, observation_encoder
    from utils.normalizer import MinMaxNormalizer
    from envs.factory import make_env
    from envs.online_task import task_protocol, wrap_training_env
    from envs.online_vector import GPUOnlineEnv
    from envs.stackcube_training import stackcube_state
    from evaluation.runner import seed_all
    from utils.experiment import write_json
    from data.episodes import digest
    cp = load_checkpoint(args.checkpoint, map_location='cpu')
    cfg = OmegaConf.create(cp['config']); cfg.device = args.device
    protocol = task_protocol(cfg)
    if protocol['critic_input_format'] != 'privileged_stackcube_v1':
        raise ValueError('Backend audit requires a train_online_rl checkpoint')
    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    forbidden = set(cfg.eval.seeds) | set(cp.get('correction_training_seeds', []))
    collection = cfg.get('stages', {}).get('iterative', {})
    if 'collect_seed_start' in collection:
        forbidden.update(range(collection.collect_seed_start,
            collection.collect_seed_start + collection.rounds * collection.episodes_per_round))
    if forbidden.intersection(seeds):
        raise ValueError('Audit seeds overlap checkpoint selection/collection seeds')
    seed_all(seeds[0])
    base = build_base(cfg, torch.device(args.device))
    base.load_state_dict(policy_weights(cp), strict=True)
    actor = make_actor(base, cfg); actor.eval_mode = 'cps'
    normalizer = MinMaxNormalizer(); normalizer.stats = cp['normalizer']
    encode = observation_encoder(cfg, actor, normalizer, args.device)
    if args.worker == 'gpu_training':
        cfg.online_rollout.num_envs = 1
        env = SingleGPUView(GPUOnlineEnv(cfg, protocol))
    elif args.worker == 'cpu_training':
        env = wrap_training_env(make_env(cfg, reward_mode=protocol['environment_reward_mode']), protocol)
    else:
        env = make_env(cfg)
    try:
        rows = audit_episodes(env, actor, encode, normalizer, seeds, protocol['horizon'],
                              lambda e: stackcube_state(e)[0][0].detach().cpu().numpy())
        from importlib.metadata import version
        write_json(Path(args.output) / f'{args.worker}.json', dict(
            mode=args.worker, checkpoint=str(Path(args.checkpoint).resolve()),
            checkpoint_sha256=digest(args.checkpoint), maniskill_version=version('mani_skill'),
            torch_version=str(torch.__version__), protocol=protocol, episodes=rows))
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--seed-start', type=int, default=50000)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--worker', choices=MODES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.episodes < 1 or args.seed_start < 0:
        parser.error('episodes must be positive and seed-start nonnegative')
    if args.worker:
        worker(args)
        return
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    for mode in MODES:
        subprocess.run([sys.executable, '-m', 'tools.diagnostics.compare_online_backends',
            '--checkpoint', str(Path(args.checkpoint).resolve()), '--output', str(out),
            '--episodes', str(args.episodes), '--seed-start', str(args.seed_start),
            '--device', args.device, '--worker', mode], check=True,
            cwd=Path(__file__).resolve().parents[2])
    reports = {mode: json.loads((out / f'{mode}.json').read_text()) for mode in MODES}
    if len({r['checkpoint_sha256'] for r in reports.values()}) != 1:
        raise ValueError('Checkpoint changed during audit')
    report = summarize(reports)
    (out / 'summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
