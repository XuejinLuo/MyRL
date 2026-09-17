"""Download and replay official ManiSkill demos; write the resolved H5 path."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', default='StackCube-v1')
    p.add_argument('--control-mode', default='pd_ee_delta_pose')
    p.add_argument('--output', default='demos')
    p.add_argument('--count', type=int, default=100)
    p.add_argument('--raw', help='Existing raw motion-planning H5; skips download')
    args = p.parse_args()
    if args.count < 1:
        p.error('count must be positive')
    out = Path(args.output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.raw:
        raw = Path(args.raw).expanduser().resolve()
    else:
        subprocess.run([sys.executable, '-m', 'mani_skill.utils.download_demo',
                        args.task, '-o', str(out)], check=True)
        candidates = []
        for path in out.rglob('*.h5'):
            if 'motionplanning' not in path.parts or not path.with_suffix('.json').exists():
                continue
            meta = json.loads(path.with_suffix('.json').read_text())
            kwargs = meta.get('env_info', {}).get('env_kwargs', {})
            if (meta.get('env_info', {}).get('env_id') == args.task
                    and kwargs.get('obs_mode', 'none') == 'none'):
                candidates.append(path)
        if len(candidates) != 1:
            raise ValueError(f'Expected one raw motion-planning H5, found {candidates}. Specify --raw PATH.')
        raw = candidates[0]
    meta = json.loads(raw.with_suffix('.json').read_text())
    if meta['env_info']['env_id'] != args.task:
        raise ValueError(f'Raw data is not {args.task}')
    # Action replay verifies the converted controller. Do not force env states while converting controls.
    subprocess.run([sys.executable, '-m', 'mani_skill.trajectory.replay_trajectory',
        '--traj-path', str(raw), '--obs-mode', 'pointcloud',
        '--target-control-mode', args.control_mode, '--sim-backend', 'physx_cpu',
        '--save-traj', '--count', str(args.count), '--num-procs', '1'], check=True)
    files = list(raw.parent.glob(f'*.pointcloud.{args.control_mode}.physx_cpu.h5'))
    if len(files) != 1:
        raise ValueError(f'Cannot resolve converted file unambiguously: {files}')
    import h5py
    with h5py.File(files[0]) as f:
        if not len(f):
            raise ValueError('Replay produced no successful trajectories')
        print(f'Replay retained {len(f)} trajectories (requested {args.count})')
    (out/'h5_path.txt').write_text(str(files[0]) + '\n')
    print(f'Training data: {files[0]}')


if __name__ == '__main__':
    main()
