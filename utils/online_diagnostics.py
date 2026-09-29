"""Deterministic diagnostics; no sampling, model calls, or training-target mutation."""
import json
from pathlib import Path
import numpy as np


def distribution(values):
    x = np.asarray(values, dtype=float)
    if not x.size:
        return dict(count=0, mean=None, std=None, p05=None, p50=None, p95=None,
                    positive_fraction=None)
    return dict(count=int(x.size), mean=float(x.mean()), std=float(x.std()),
                p05=float(np.quantile(x, .05)), p50=float(np.quantile(x, .5)),
                p95=float(np.quantile(x, .95)), positive_fraction=float((x > 0).mean()))


def value_metrics(predicted, targets):
    # Float64 diagnostic arithmetic only; inputs and PPO targets remain untouched.
    p, y = predicted.detach().double(), targets.detach().double()
    variance = y.var(unbiased=False).item()
    valid = variance > 1e-8
    return dict(MSE=(p-y).square().mean().item(), Prediction_Mean=p.mean().item(),
        Prediction_Variance=p.var(unbiased=False).item(), Target_Mean=y.mean().item(),
        Target_Variance=variance, EV_Valid=valid,
        Explained_Variance=1-(y-p).var(unbiased=False).item()/variance if valid else None)


def append_json(path, row):
    with Path(path).open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, allow_nan=False) + '\n')


class EpisodeDiagnostics:
    """Keep only the current episode + completed episodes awaiting this rollout's GAE.

    Completed summaries always persist. Optional primitive traces are capped by
    episode count AND per-episode step count. Outcome labels never enter learning.
    """
    def __init__(self, directory, gamma, options=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        options = options or {}
        self.enabled = options.get('enabled', True)
        self.trace_max_episodes = int(options.get('trace_max_episodes', 20)) if self.enabled else 0
        self.trace_max_steps = int(options.get('trace_max_steps_per_episode', 300))
        self.phase_enabled = self.enabled and options.get('task_phase_stats', True)
        self.gamma, self.next_id, self.current, self.completed = gamma, 0, None, []

    def record(self, transitions, value, discounted_reward, epoch):
        if self.current is None:
            self.current = dict(episode_id=self.next_id, start_epoch=epoch, length=0,
                raw_return=0., training_return=0., raw_discounted_return=0.,
                training_discounted_return=0., success=False, first_success_step=None,
                phases={}, flags_true_steps={}, flags_observed_steps={}, decisions=[], trace=[],
                last_phase=None, phase_run=0)
            self.next_id += 1
        episode = self.current
        decision = dict(step=episode['length'], length=len(transitions), value=float(value),
            discounted_reward=float(discounted_reward), phase=transitions[0]['phase_before'],
            epoch=epoch)
        episode['decisions'].append(decision)
        for event in transitions:
            t = episode['length']
            episode['length'] += 1
            for source, dest in (('raw_reward', 'raw'), ('training_reward', 'training')):
                episode[f'{dest}_return'] += event[source]
                episode[f'{dest}_discounted_return'] += self.gamma**t * event[source]
            episode['success'] |= event['success']
            if event['success'] and episode['first_success_step'] is None:
                episode['first_success_step'] = episode['length']
            if self.phase_enabled:
                phase = event['phase']
                episode['phase_run'] = episode['phase_run'] + 1 if phase == episode['last_phase'] else 1
                episode['last_phase'] = phase
                stats = episode['phases'].setdefault(phase, dict(steps=0, longest_run=0,
                    raw_return=0., training_return=0., raw_discounted_return=0., training_discounted_return=0.))
                stats['steps'] += 1
                stats['longest_run'] = max(stats['longest_run'], episode['phase_run'])
                for source, dest in (('raw_reward', 'raw'), ('training_reward', 'training')):
                    stats[f'{dest}_return'] += event[source]
                    stats[f'{dest}_discounted_return'] += self.gamma**t * event[source]
                for flag, val in event['flags'].items():
                    episode['flags_true_steps'][flag] = episode['flags_true_steps'].get(flag, 0) + int(val is True)
                    episode['flags_observed_steps'][flag] = episode['flags_observed_steps'].get(flag, 0) + int(val is not None)
            if episode['episode_id'] < self.trace_max_episodes and len(episode['trace']) < self.trace_max_steps:
                episode['trace'].append(dict(event))
        last = transitions[-1]
        if last['terminated'] or last['truncated']:
            episode.update(end_epoch=epoch, termination_reason=last['termination_reason'],
                raw_terminated=last['raw_terminated'], raw_truncated=last['raw_truncated'],
                outcome='success' if episode['success'] else last['termination_reason'], censored=False)
            self.completed.append(episode)
            self.current = None
        return decision

    @staticmethod
    def attach_advantages(decisions, advantages, normalized, actor_enabled):
        for row, raw, norm in zip(decisions, advantages.detach().cpu().tolist(), normalized.detach().cpu().tolist()):
            row.update(advantage=raw, normalized_advantage=norm, actor_enabled=bool(actor_enabled))

    def _write_episode(self, episode):
        decisions, trace = episode.pop('decisions'), episode.pop('trace')
        episode.pop('last_phase'); episode.pop('phase_run')
        # Complete Monte Carlo return-to-go under the sampled, possibly changing
        # policy; never substituted for the frozen GAE regression target.
        if not episode['censored']:
            actual = 0.
            for decision in reversed(decisions):
                actual = decision['discounted_reward'] + self.gamma**decision['length'] * actual
                decision['actual_discounted_return_to_go'] = actual
            errors = [d['value'] - d['actual_discounted_return_to_go'] for d in decisions]
            episode['MC/Decision_MSE'] = float(np.mean(np.square(errors)))
            episode['MC/Start_Value'] = decisions[0]['value']
            episode['MC/Start_Target'] = decisions[0]['actual_discounted_return_to_go']
        else:
            episode['MC/Decision_MSE'] = None
            episode['MC/Start_Value'] = None
            episode['MC/Start_Target'] = None
        groups = {'all': decisions}
        if self.phase_enabled:
            groups.update({phase: [d for d in decisions if d['phase'] == phase]
                           for phase in sorted({d['phase'] for d in decisions})})
        episode['advantages'] = {phase: {key: distribution([d[key] for d in rows if key in d])
            for key in ('advantage', 'normalized_advantage')} for phase, rows in groups.items()}
        episode['actor_enabled_decisions'] = sum(d.get('actor_enabled', False) for d in decisions)
        # A normalized advantage is the offered PPO signal; KL guards/clipping
        # can prevent or reduce its realized contribution.
        if trace:
            append_json(self.directory/'traces.jsonl', dict(episode_id=episode['episode_id'],
                censored=episode['censored'], trace_truncated=len(trace) < episode['length'],
                transitions=trace, decisions=decisions[:self.trace_max_steps]))
        append_json(self.directory/'episodes.jsonl', episode)
        return episode

    def flush(self, epoch, final=False):
        if final and self.current is not None:
            self.current.update(end_epoch=epoch, outcome='censored', termination_reason='run_boundary',
                                censored=True, raw_terminated=None, raw_truncated=None)
            self.completed.append(self.current)
            self.current = None
        advantage_groups = {}
        for outcome in ('success', 'failure', 'timeout', 'censored'):
            decisions = [d for e in self.completed if e['outcome'] == outcome for d in e['decisions']]
            by_phase = {'all': decisions}
            if self.phase_enabled:
                by_phase.update({phase: [d for d in decisions if d['phase'] == phase]
                                 for phase in sorted({d['phase'] for d in decisions})})
            advantage_groups[outcome] = {phase: {
                key: distribution([d[key] for d in subset if key in d])
                for key in ('advantage', 'normalized_advantage')}
                for phase, subset in by_phase.items()}
        rows = [self._write_episode(e) for e in self.completed]
        self.completed.clear()
        groups = {}
        for outcome in ('success', 'failure', 'timeout', 'censored'):
            subset = [e for e in rows if e['outcome'] == outcome]
            groups[outcome] = {key: distribution([e[key] for e in subset]) for key in (
                'length', 'raw_return', 'training_return', 'raw_discounted_return', 'training_discounted_return')}
        append_json(self.directory/'summary.jsonl', dict(epoch=epoch, final=final,
            completed_count=sum(not e['censored'] for e in rows),
            ongoing_episode_id=self.current['episode_id'] if self.current else None, outcomes=groups,
            advantages_by_outcome_and_phase=advantage_groups))
        return {f'Episode/{outcome}_Count': groups[outcome]['length']['count'] for outcome in groups}
