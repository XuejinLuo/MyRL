"""Reproducible StackCube pipeline; each subprocess can also be run separately."""
import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def build_commands(args):
    out = Path(args.output).resolve()
    h5 = str(Path(args.h5).expanduser().resolve())
    val = 'eval.seeds=[' + ','.join(map(str, range(2000, 2000+args.val_episodes))) + ']'
    test = 'test_seeds=[' + ','.join(map(str, range(3000, 3000+args.test_episodes))) + ']'
    shared = ['env=stackcube', f'device={args.device}', val, 'eval.sampler=cps',
              f'video.every={args.video_every}', 'video.episodes=5', 'hydra.job.chdir=false']
    offline, iterative, online = [out/x for x in ('offline', 'iterative', 'online')]
    initial = offline/'checkpoints'/'best.pth'
    stats = offline/'checkpoints'/'dataset_stats.json'
    manifest = out/'data'/'manifest.json'
    return [
        ('offline', ['train_offline.py', *shared, f'output={offline}', f'dataset.data_path={h5}',
            f'dataset.max_episodes={args.demos}', f'epochs={args.offline_epochs}',
            'eval.every=10', 'save_epoch=10', 'wandb.enable=false',
            'algo.use_bc_only=false']),
        ('prepare', ['prepare_iterative_data.py', '--h5', h5, '--output', str(out/'data'),
            '--stats', str(stats), '--max-episodes', str(args.demos), 'env=stackcube']),
        ('iterative', ['train_iterative.py', *shared, f'output={iterative}',
            f'initial_ckpt={initial}', f'stats_path={stats}', f'manifest={manifest}',
            f'rounds={args.rounds}', f'episodes_per_round={args.collect_episodes}',
            f'epochs={args.iterative_epochs}', 'final_test=false', test,
            'algo.use_bc_only=false']),
        ('online', ['train_online.py', *shared, f'output={online}',
            f'algo.pretrained_ckpt={iterative / "checkpoints" / "best.pth"}',
            f'epochs={args.online_epochs}', 'wandb.enable=false']),
        ('test', ['evaluate_checkpoint.py', '--checkpoint', str(initial),
            str(iterative/'checkpoints'/'best.pth'), str(online/'checkpoints'/'best.pth'),
            '--labels', 'offline', 'iterative', 'online', '--output', str(out/'comparison'),
            '--episodes', str(args.test_episodes), '--device', args.device])]


def preflight(args):
    """Check the data identity and required schema before spending GPU time."""
    import h5py
    h5 = Path(args.h5).expanduser().resolve()
    sidecar = h5.with_suffix('.json')
    meta = json.loads(sidecar.read_text())
    env_info = meta['env_info']
    if env_info['env_id'] != 'StackCube-v1':
        raise ValueError('Dataset must be StackCube-v1, not PullCubeTool')
    if env_info['env_kwargs'].get('control_mode') != 'pd_ee_delta_pose':
        raise ValueError('Replay demonstrations into pd_ee_delta_pose first')
    reserved = set(range(2000, 2000+args.val_episodes)) | set(range(3000, 3000+args.test_episodes))
    reserved |= set(range(10000, 10000+args.rounds*args.collect_episodes))
    with h5py.File(h5) as f:
        keys = list(f)[:args.demos]
        if not keys:
            raise ValueError('Empty demonstration file')
        selected_ids = {int(k.split('_')[-1]) for k in keys}
        for ep in meta.get('episodes', []):
            if ep.get('episode_id') in selected_ids:
                seed = ep.get('episode_seed', ep.get('reset_kwargs', {}).get('seed'))
                if isinstance(seed, int) and seed in reserved:
                    raise ValueError(f'Demonstration seed {seed} overlaps reserved seeds')
        for key in keys:
            g = f[key]
            for name in ('success', 'terminated', 'truncated', 'actions', 'obs/pointcloud/xyzw',
                         'obs/pointcloud/rgb', 'obs/agent/qpos'):
                if name not in g:
                    raise ValueError(f'{key}: missing {name}; replay raw demonstrations first')
            if 'obs/extra/tcp_pose' not in g and 'obs/agent/tcp_pose' not in g:
                raise ValueError(f'{key}: tcp_pose missing')
            if g['actions'].shape[-1] != 7 or g['obs/agent/qpos'].shape[-1] != 9:
                raise ValueError('Expected Panda 7D actions and 9D qpos')
            if len(g['obs/pointcloud/xyzw']) != len(g['actions']) + 1:
                raise ValueError(f'{key}: expected T+1 observations')
    print(f'Preflight: {len(keys)} StackCube demonstrations; control/observation schema OK')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--h5', default='/home/luo/.maniskill/demos/StackCube-v1/motionplanning/trajectory.pointcloud.pd_ee_delta_pose.physx_cpu.h5')
    p.add_argument('--output', default='outputs/stackcube/run01')
    p.add_argument('--device', default='cuda')
    p.add_argument('--demos', type=int, default=1000)
    p.add_argument('--offline-epochs', type=int, default=200)
    p.add_argument('--iterative-epochs', type=int, default=30)
    p.add_argument('--online-epochs', type=int, default=100)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--collect-episodes', type=int, default=100)
    p.add_argument('--val-episodes', type=int, default=100)
    p.add_argument('--test-episodes', type=int, default=100)
    p.add_argument('--video-every', type=int, default=10)
    p.add_argument('--stage', choices=['all','offline','prepare','iterative','online','test'], default='all')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    for key in ('demos','offline_epochs','iterative_epochs','online_epochs','rounds',
                'collect_episodes','val_episodes','test_episodes'):
        if getattr(args, key) < 1:
            p.error(f'{key} must be positive')
    if args.val_episodes > 1000 or args.test_episodes > 7000 or args.iterative_epochs <= 3:
        p.error('Require val<=1000, test<=7000 and iterative-epochs>3 (critic warmup)')
    commands = build_commands(args)
    if args.dry_run:
        for stage, command in commands:
            if args.stage in ('all', stage):
                print(shlex.join([sys.executable, *command]))
        return
    preflight(args)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    spec = dict(vars(args))
    spec.pop('stage'); spec.pop('dry_run')
    spec['h5'] = str(Path(args.h5).expanduser().resolve())
    spec['output'] = str(output)
    plan = output/'pipeline.json'
    if plan.exists() and json.loads(plan.read_text()) != spec:
        raise ValueError('Pipeline configuration changed; choose a new output directory')
    plan.write_text(json.dumps(spec, indent=2))
    logs = output/'logs'
    logs.mkdir(exist_ok=True)
    for stage, command in commands:
        if args.stage not in ('all', stage):
            continue
        marker = output/f'{stage}.done.json'
        if marker.exists():
            print(f'Skipping completed stage: {stage}', flush=True)
            continue
        # An interrupted iterative stage supports completed-round continuation.
        if stage == 'iterative' and (output/'iterative'/'state.json').exists():
            command.append('resume=true')
        print(shlex.join([sys.executable, '-u', *command]), flush=True)
        with (logs/f'{stage}.log').open('a') as log:
            process = subprocess.Popen([sys.executable, '-u', *command], cwd=ROOT,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line); log.flush()
            code = process.wait()
        if code:
            raise SystemExit(f'{stage} failed (exit {code}); inspect {logs/stage}.log')
        marker.write_text(json.dumps(dict(command=command)))


if __name__ == '__main__':
    main()
