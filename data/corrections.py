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
    return actor_valid_lengths(length, segments, chunk_size, final_success) > 0


def actor_valid_lengths(length, segments, chunk_size, final_success,
                        label_mode='full_chunk', min_valid_length=1):
    """Versioned Actor labels; the intact primitive trajectory stays unchanged."""
    if chunk_size < 1:
        raise ValueError('Invalid chunk_size')
    if label_mode not in ('full_chunk', 'masked'):
        raise ValueError('Unknown correction label_mode')
    if type(min_valid_length) is not int or not 1 <= min_valid_length <= chunk_size:
        raise ValueError('min_valid_length must be in [1, chunk_size]')
    result = np.zeros(length, dtype=np.int64)
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
            lengths = np.minimum(chunk_size, stop - np.arange(start, stop))
            minimum = chunk_size if label_mode == 'full_chunk' else min_valid_length
            result[start:stop] = np.where(lengths >= minimum, lengths, 0)
    return result


def metadata_file(session, item):
    # Legacy flat sessions remain readable without moving or re-hashing files.
    from workflows.failure_queue import session_file
    return session_file(session, item.get('metadata', str(Path(item['path']).with_suffix('.json'))))


def prepare(base_manifest, session, stats_path, output, label_mode='full_chunk', min_valid_length=1):
    """Validate everything before exporting new immutable episodes/manifest."""
    base_manifest, session = Path(base_manifest).resolve(), Path(session).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    spec = json.loads(base_manifest.read_text())
    provenance = json.loads((session/'session.json').read_text())
    review = json.loads((session/'review.json').read_text())
    config = provenance['config']
    chunk_size = config['model']['chunk_size']
    actor_valid_lengths(0, [], chunk_size, False, label_mode, min_valid_length)
    if provenance.get('schema') != 'myrl_human_session_v1' or review.get('schema') != 'myrl_human_review_v1':
        raise ValueError('Unsupported correction session/review')
    if spec['schema'] not in (SCHEMA, LEGACY_SCHEMA) or spec['reward_mode'] != 'success':
        raise ValueError('Unsupported base manifest')
    if spec['env'] != config['env']:
        raise ValueError('Session/base manifest environment or observation preprocessing differs')
    if json.loads(Path(stats_path).read_text()) != provenance['normalizer']:
        raise ValueError('Use the frozen normalizer from the collection checkpoint')
    forbidden = held_out_seeds(config) | set(provenance.get('excluded_seeds', []))
    if set(provenance.get('screening_seeds', [])) & forbidden:
        raise ValueError('Screening seeds overlap validation/test seeds')
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
        metadata_path = metadata_file(session, item)
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
        row = dict(seed=seed, path=str(path), decision=disposition, actor_starts=0,
                   accepted_human_steps=0, full_chunk_starts=0, partial_chunk_starts=0,
                   excluded_starts=0, valid_length_histogram={}, reason='dropped' if disposition == 'drop' else None)
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
        lengths = actor_valid_lengths(len(ep['action']), segments, chunk_size, success,
                                      label_mode, min_valid_length)
        mask = lengths > 0
        ep['actor_eligible'] = mask
        ep['actor_valid_length'] = lengths
        accepted_steps = sum(s['stop']-s['start'] for s in segments if s['decision'] == 'accept') if success else 0
        values, counts = np.unique(lengths[mask], return_counts=True)
        row.update(steps=len(ep['action']), final_success=success, actor_starts=int(mask.sum()),
                   accepted_human_steps=accepted_steps,
                   full_chunk_starts=int((lengths == chunk_size).sum()),
                   partial_chunk_starts=int(((lengths > 0) & (lengths < chunk_size)).sum()),
                   excluded_starts=accepted_steps-int(mask.sum()),
                   valid_length_histogram={str(v): int(c) for v, c in zip(values, counts)},
                   reason=None if success else 'failed recovery: critic only, even if human labels accepted')
        row['segments'] = [dict(id=s['id'], length=s['stop']-s['start'], decision=s['decision'],
            actor_starts=int(mask[s['start']:s['stop']].sum()),
            excluded_tail_starts=(s['stop']-s['start']-int(mask[s['start']:s['stop']].sum())
                if success and disposition == 'keep' and s['decision'] == 'accept' else 0),
            tail_rule='full_chunk_required' if label_mode == 'full_chunk' else 'min_valid_length',
            reason=('failed_recovery' if not success else 'critic_only' if disposition == 'critic_only'
                    else 'rejected' if s['decision'] != 'accept' else 'shorter_than_minimum'
                    if s['stop']-s['start'] < (chunk_size if label_mode == 'full_chunk' else min_valid_length)
                    else 'accepted'))
            for s in item['segments']]
        if item['sha256'] in seen or item['sha256'] in seen_raw:
            raise ValueError('Duplicate raw correction trajectory')
        seen.add(item['sha256'])
        seen_raw.add(item['sha256'])
        cleaned.append((ep, item, metadata))
    if not sum(r['actor_starts'] for r in report):
        raise ValueError('No approved successful correction labels; inspect review/segment lengths')
    name = 'human_corrections_' + session.name
    if name in [s['name'] for s in spec['sources']]:
        raise ValueError('Session source already imported')
    output.mkdir(parents=True)
    source = dict(name=name, role='human_correction', chunk_size=config['model']['chunk_size'],
                  actor_label_schema='myrl_actor_labels_v2', actor_label_mode=label_mode,
                  min_valid_length=chunk_size if label_mode == 'full_chunk' else min_valid_length,
                  screening_seeds=provenance.get('screening_seeds', []),
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
        actor_episodes=sum(r['actor_starts'] > 0 for r in report),
        actor_label_schema='myrl_actor_labels_v2', actor_label_mode=label_mode,
        min_valid_length=source['min_valid_length'],
        **{key: sum(r[key] for r in report) for key in
           ('accepted_human_steps', 'full_chunk_starts', 'partial_chunk_starts', 'excluded_starts')},
        policy='intact transitions for critic; approved, final-success human chunks for actor',
        base_manifest_sha256=digest(base_manifest), normalizer_sha256=digest(stats_path)))
    write_json(output/'review.json', review)
    write_json(output/'session.json', provenance)
    return output/'manifest.json'
