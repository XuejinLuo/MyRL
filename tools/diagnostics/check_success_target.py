"""Report an independently evaluated checkpoint's measured success target."""
import argparse
import json
import math
from pathlib import Path


def target_report(summary, stage='rl', sampler='cps', target=.95, min_episodes=1000):
    if not 0 < target <= 1 or min_episodes < 1:
        raise ValueError('Require 0 < target <= 1 and min_episodes >= 1')
    rows = [r for r in summary['results'] if r['stage'] == stage and r['sampler'] == sampler]
    if len(rows) != 1:
        raise ValueError('Expected exactly one matching stage/sampler result')
    row = rows[0]
    n, k = int(row['Eval/Episodes']), int(row['Eval/Success_Count'])
    if not 0 <= k <= n or n < 1:
        raise ValueError('Invalid success/episode counts')
    p, z = k / n, 1.959963984540054
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return dict(stage=stage, sampler=sampler, checkpoint=row['checkpoint'],
        successes=k, episodes=n, success_rate=p, ci95=[center-half, center+half],
        target=target, minimum_episodes=min_episodes,
        measured_target_met=n >= min_episodes and p >= target,
        ci95_lower_reaches_target=n >= min_episodes and center-half >= target,
        interpretation='Measured success on this test set; does not guarantee population success.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary', type=Path)
    parser.add_argument('--stage', default='rl')
    parser.add_argument('--sampler', default='cps')
    parser.add_argument('--target', type=float, default=.95)
    parser.add_argument('--min-episodes', type=int, default=1000)
    args = parser.parse_args()
    report = target_report(json.loads(args.summary.read_text()), args.stage, args.sampler, args.target, args.min_episodes)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if report['measured_target_met'] else 1)


if __name__ == '__main__':
    main()
