"""Review a mouse/keyboard collection session and append it to a training manifest."""
import argparse
import json
from pathlib import Path

from data.corrections import prepare
from utils.experiment import write_json


def review_session(session):
    path = Path(session)/'review.json'
    review = json.loads(path.read_text())
    print('Watch the session videos before accepting. k=keep, c=critic only, d=drop.')
    for item in review['episodes']:
        metadata = json.loads((Path(session)/item['path']).with_suffix('.json').read_text())
        print(f"\n{item['path']} | final_success={metadata['final_success']} | "
              f"end={metadata['end_reason']} | video={metadata.get('video')}")
        print('Suspected types:', metadata.get('failure_hints', []))
        choices = {'k': 'keep', 'c': 'critic_only', 'd': 'drop'}
        answer = input(f"Episode [{item['decision']}] k/c/d (Enter keeps decision): ").strip().lower()
        if answer:
            if answer not in choices:
                raise ValueError('Expected k/c/d; no review changes written for this episode')
            item['decision'] = choices[answer]
        if item['decision'] == 'keep':
            for segment in item['segments']:
                print(f"Human segment {segment['id']}: steps [{segment['start']}, {segment['stop']})")
                answer = input(f"[{segment['decision']}] a=accept / r=reject / Enter=unchanged: ").strip().lower()
                if answer:
                    if answer not in ('a', 'r'):
                        raise ValueError('Expected a/r')
                    segment['decision'] = 'accept' if answer == 'a' else 'reject'
        write_json(path, review)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True)
    parser.add_argument('--base-manifest', required=True)
    parser.add_argument('--stats', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--review', action='store_true', help='Interactive terminal review before export')
    args = parser.parse_args()
    if args.review:
        review_session(args.session)
    print(prepare(args.base_manifest, args.session, args.stats, args.output))


if __name__ == '__main__':
    main()
