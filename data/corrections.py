"""Audited human labels on intact primitive episodes (no reward relabeling)."""
import json
from pathlib import Path

import numpy as np

from data.episodes import SCHEMA, LEGACY_SCHEMA, digest, load_episode, save_episode
from utils.experiment import write_json


def held_out_seeds(config):
    comparison = config.get('comparison', {})
    start, count = comparison.get('seed_start', 0), comparison.get('episodes', 0)
    return set(config.get('eval', {}).get('seeds', [])) | set(range(start, start+count))


def eligible_starts(length, segments, chunk_size, final_success):
    """Only full approved human chunks; never pad or bridge a control switch."""
    if chunk_size < 1:
        raise ValueError('Invalid chunk_size')
    result = np.zeros(length, dtype=bool)
    previous_end = 0
    for segment in segments:
        start, stop = segment['start'], segment['stop']
        if type(start) is not int or type(stop) is not int or not previous_end <= start < stop <= length:
            raise ValueError('Invalid/overlapping human segments')
        previous_end = stop
        decision = segment['decision']
        if decision not in ('pending', 'accept', 'reject'):
            raise ValueError('Unknown segment review decision')
        if decision == 'pending':
            raise ValueError('Human segment still pending review')
        if decision == 'accept' and final_success:
            result[start:max(start, stop-chunk_size+1)] = True
    return result


def prepare(base_manifest, session, stats_path, output):
    """Validate everything before exporting new immutable episodes/manifest."""
    base_manifest, session = Path(base_manifest).resolve(), Path(session).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    spec = json.loads(base_manifest.read_text())
    provenance = json.loads((session/'session.json').read_text())
    review = json.loads((session/'review.json').read_text())
    config = provenance['config']
    if provenance.get('schema') != 'myrl_human_session_v1' or review.get('schema') != 'myrl_human_review_v1':
        raise ValueError('Unsupported correction session/review')
    if spec['schema'] not in (SCHEMA, LEGACY_SCHEMA) or spec['reward_mode'] != 'success':
        raise ValueError('Unsupported base manifest')
    if spec['env'] != config['env']:
        raise ValueError('Session/base manifest environment or observation preprocessing differs')
    if json.loads(Path(stats_path).read_text()) != provenance['normalizer']:
        raise ValueError('Use the frozen normalizer from the collection checkpoint')
    forbidden = held_out_seeds(config) | set(provenance.get('excluded_seeds', []))
    bounds = provenance['normalizer']['action']
    lo, hi = np.asarray(bounds['min']), np.asarray(bounds['max'])
    seen, seen_raw = set(), set()
    for src in spec['sources']:
        for item in src['episodes']:
            path = (base_manifest.parent/item['path']).resolve()
            sha = digest(path)
            if sha != item['sha256'] or sha in seen:
                raise ValueError(f'Changed/duplicate base episode: {path}')
            seen.add(sha)
            if item.get('raw_sha256'):
                seen_raw.add(item['raw_sha256'])
            item['path'] = str(path)
    cleaned, report = [], []
    if not review['episodes']:
        raise ValueError('No collected episodes')
    recorded_seeds = set()
    for item in review['episodes']:
        path = (session/item['path']).resolve()
        if not path.is_relative_to(session):
            raise ValueError('Episode path leaves session directory')
        metadata_path = path.with_suffix('.json')
        if digest(path) != item['sha256'] or digest(metadata_path) != item['metadata_sha256']:
            raise ValueError('Raw episode/metadata changed after collection')
        metadata = json.loads(metadata_path.read_text())
        seed = metadata['seed']
        if seed in forbidden or seed in recorded_seeds:
            raise ValueError(f'Held-out or duplicate correction seed: {seed}')
        recorded_seeds.add(seed)
        raw_segments = metadata['segments']
        if len(item['segments']) != len(raw_segments) or any(
            any(a[k] != b[k] for k in ('id', 'start', 'stop'))
            for a, b in zip(item['segments'], raw_segments)
        ):
            raise ValueError('Review may change decisions, never human segment boundaries')
        disposition = item['decision']
        if disposition not in ('keep', 'critic_only', 'drop'):
            raise ValueError(f'{path.name}: episode still pending review')
        row = dict(seed=seed, path=str(path), decision=disposition, actor_starts=0)
        report.append(row)
        if disposition == 'drop':
            continue
        ep = load_episode(path)
        if 'actor_eligible' in ep:
            raise ValueError('Expected unprocessed raw session')
        if ep['action'].shape[1:] != lo.shape or ((ep['action'] < lo-1e-5) | (ep['action'] > hi+1e-5)).any():
            raise ValueError(f'{path.name}: action outside frozen bounds; do not silently clip labels')
        from data.observations import observation_fields, validate_observation
        from omegaconf import OmegaConf
        cfg = OmegaConf.create(config)
        # Shapes of all rows match; validate finite arrays via load_episode above.
        validate_observation({k: ep[k][0] for k in observation_fields(ep)}, cfg)
        success = bool(ep['success'][-1])
        segments = item['segments'] if disposition == 'keep' else [dict(s, decision='reject') for s in item['segments']]
        mask = eligible_starts(len(ep['action']), segments, config['model']['chunk_size'], success)
        ep['actor_eligible'] = mask
        row.update(steps=len(ep['action']), final_success=success, actor_starts=int(mask.sum()),
                   reason=None if success else 'failed recovery: critic only, even if human labels accepted')
        row['segments'] = [dict(id=s['id'], length=s['stop']-s['start'], decision=s['decision'],
            actor_starts=int(mask[s['start']:s['stop']].sum()),
            reason=('failed_recovery' if not success else 'critic_only' if disposition == 'critic_only'
                    else 'rejected' if s['decision'] != 'accept' else 'shorter_than_chunk'
                    if s['stop']-s['start'] < config['model']['chunk_size'] else 'accepted'))
            for s in item['segments']]
        if item['sha256'] in seen or item['sha256'] in seen_raw:
            raise ValueError('Duplicate raw correction trajectory')
        seen.add(item['sha256'])
        seen_raw.add(item['sha256'])
        cleaned.append((ep, item, metadata))
    if not sum(r['actor_starts'] for r in report):
        raise ValueError('No approved full-length successful correction chunks; inspect review/segment lengths')
    name = 'human_corrections_' + session.name
    if name in [s['name'] for s in spec['sources']]:
        raise ValueError('Session source already imported')
    output.mkdir(parents=True)
    source = dict(name=name, role='human_correction', chunk_size=config['model']['chunk_size'],
                  session=str(session), session_sha256=digest(session/'session.json'),
                  review_sha256=digest(session/'review.json'), episodes=[])
    for i, (ep, item, meta) in enumerate(cleaned):
        path = output/f'correction_{i:06d}.npz'
        save_episode(path, ep)
        source['episodes'].append(dict(path=str(path), sha256=digest(path), seed=meta['seed'],
            raw_sha256=item['sha256'], steps=len(ep['action']), success=bool(ep['success'].any()),
            actor_starts=int(ep['actor_eligible'].sum())))
    spec['sources'].append(source)
    write_json(output/'manifest.json', spec)
    write_json(output/'cleaning_report.json', dict(episodes=report,
        actor_starts=sum(r['actor_starts'] for r in report),
        policy='intact transitions for critic; approved, final-success human chunks for actor',
        base_manifest_sha256=digest(base_manifest), normalizer_sha256=digest(stats_path)))
    write_json(output/'review.json', review)
    write_json(output/'session.json', provenance)
    return output/'manifest.json'
