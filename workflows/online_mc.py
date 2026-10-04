"""Fixed episode quotas under one frozen behavior policy per PPO update."""
import time
import numpy as np
import torch
from tqdm import tqdm
from algos.pg import compute_mc_returns
from workflows.online_rollout import OnlineCollector


def return_protocol(cfg):
    estimator = cfg['algo'].get('return_estimator', 'gae')
    if estimator not in ('gae', 'mc'):
        raise ValueError('return_estimator must be gae or mc')
    baseline = cfg['algo'].get('mc_baseline', 'value')
    if baseline not in ('value', 'none'):
        raise ValueError('mc_baseline must be value or none')
    if estimator != 'mc' and baseline != 'value':
        raise ValueError('mc_baseline=none requires return_estimator=mc')
    protocol = dict(version=1, estimator=estimator)
    if estimator == 'mc':
        rollout = cfg.get('online_rollout', {})
        quota = rollout.get('episodes_per_batch', 64)
        n = rollout.get('num_envs', 1)
        if type(quota) is not int or quota < n or quota % n:
            raise ValueError('episodes_per_batch must be a positive multiple of num_envs')
        if cfg.get('online_task', {}).get('timeout_semantics') != 'finite_horizon':
            raise ValueError('MC requires finite_horizon timeout semantics')
        protocol.update(episodes_per_batch=quota, collection='complete_episode_waves',
                        mc_baseline=baseline)
    return protocol


def check_return_resume(checkpoint, cfg):
    # Pre-MC checkpoints were all GAE. Never reinterpret their optimizer/Critic
    # state as an MC run; use Actor initialization to change the estimator.
    saved = checkpoint.get('return_protocol')
    if saved is None:
        saved = return_protocol(checkpoint['config'])
    # PR #22 MC checkpoints predate this option and used G - V. Copy before
    # canonicalizing so validation does not mutate loaded checkpoint metadata.
    saved = dict(saved)
    if saved.get('estimator') == 'mc':
        saved.setdefault('mc_baseline', 'value')
    if saved != return_protocol(cfg):
        raise ValueError('Checkpoint return protocol mismatch; initialize a new run '
                         'from Actor weights instead of resume')


class CompleteEpisodeCollector(OnlineCollector):
    """One episode per lane per wave; a fixed number of waves per batch.

    Short/successful episodes cannot crowd out long/failed episodes. Finished
    lanes wait without resetting. No incomplete episode crosses an update.
    The env-step budget is a soft boundary checked AFTER a full batch, so the
    final batch may overshoot by < episodes_per_batch * horizon valid steps.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        returns_protocol = return_protocol(self.cfg)
        self.quota = returns_protocol['episodes_per_batch']
        self.baseline = returns_protocol['mc_baseline']
        if self.quota % self.n:
            raise ValueError('Episode quota must be a multiple of actual environment count')

    @torch.no_grad()
    def collect(self, epoch, remaining_steps=None):
        if remaining_steps is not None and remaining_steps <= 0:
            raise ValueError('No environment-step budget remains')
        if self.elapsed.any():
            raise RuntimeError('MC batches must start at episode reset boundaries')
        started = time.perf_counter()
        cfg, device = self.cfg, self.features.device
        self.actor.eval(); self.critic.eval()
        rows, decisions, starts = [], [], []
        counts = dict(raw=0., training=0., successes=0, steps=0, clipped=0, actions=0)
        with tqdm(total=self.quota, desc=f'Epoch {epoch} complete episodes', disable=cfg.quiet) as progress:
            for _ in range(self.quota // self.n):
                active = np.ones(self.n, dtype=bool)
                lane_rows, lane_decisions = [[] for _ in active], [[] for _ in active]
                successes = np.zeros(self.n, dtype=bool)
                # Finite horizons bound the loop even if an adapter fails to end.
                for _ in range(self.protocol['horizon']):
                    ids = np.flatnonzero(active)
                    index = torch.as_tensor(ids, device=device)
                    features = self.features[index]
                    value_input = self.value_input(self.features, self.infos)[index]
                    values = self.critic(value_input).squeeze(-1)
                    action, chain, logprob = self.actor.collect(features)
                    raw = action.cpu().numpy()
                    if not np.isfinite(raw).all():
                        raise FloatingPointError('Nonfinite generated action')
                    actions = np.zeros((self.n, cfg.model.chunk_size, cfg.model.action_dim), dtype=raw.dtype)
                    actions[ids] = self.normalizer.unnormalize(raw, 'action')
                    nxt, _, terms, truncs, infos = self.env.step(actions, active=active)
                    done = np.asarray(terms) | np.asarray(truncs)
                    for j, i in enumerate(ids):
                        info = infos[i]
                        length = int(info['actual_steps'])
                        primitive = np.asarray(info['primitive_rewards'], dtype=np.float64)
                        if primitive.shape != (length,) or not 1 <= length <= cfg.env.exec_steps:
                            raise ValueError('Invalid primitive rewards/length')
                        reward = float(np.dot(cfg.algo.gamma ** np.arange(length), primitive)) * cfg.algo.reward_scale
                        row = dict(features=features[j], chains=chain[j], logprobs=logprob[j],
                            values=values[j], rewards=reward, dones=float(done[i]), lengths=length,
                            terminated=float(terms[i]), critic_features=value_input[j])
                        lane_rows[i].append({k: torch.as_tensor(v).detach().cpu() for k, v in row.items()})
                        lane_decisions[i].append(self.diagnostics[i].record(
                            info['online_transitions'], values[j].item(), reward, epoch))
                        counts['raw'] += sum(e['raw_reward'] for e in info['online_transitions'])
                        counts['training'] += sum(e['training_reward'] for e in info['online_transitions'])
                        counts['clipped'] += int((np.abs(raw[j, :length]) > 1.1).sum())
                        counts['actions'] += raw[j, :length].size
                        counts['steps'] += length
                        self.elapsed[i] += length
                        if self.elapsed[i] > self.protocol['horizon']:
                            raise RuntimeError('Environment exceeded finite MC horizon')
                        successes[i] |= bool(info.get('success_any', info.get('success', False)))
                        self.infos[i] = info
                        if done[i]:
                            active[i] = False
                            progress.update(1)
                    if not active.any():
                        break
                    # No terminal value calls or reset observations in MC targets.
                    live = np.flatnonzero(active)
                    self.features[torch.as_tensor(live, device=device)] = self.encode([nxt[i] for i in live])
                if active.any():
                    raise RuntimeError('Environment failed to end within finite MC horizon')
                for episode, episode_decisions in zip(lane_rows, lane_decisions):
                    starts.append(len(rows))
                    rows.extend(episode)
                    decisions.extend(episode_decisions)
                counts['successes'] += int(successes.sum())
                # Every lane is terminal. Reset only now, never resample fast lanes.
                observations, self.infos = self.env.reset_done(np.ones(self.n, dtype=bool), nxt, self.infos)
                self.elapsed[:] = 0
                self.features = self.encode(observations)
        batch = {key: torch.stack([row[key] for row in rows]) for key in rows[0]}
        if not all(torch.isfinite(v).all() for v in batch.values()):
            raise FloatingPointError('Nonfinite MC rollout')
        advantages, returns = compute_mc_returns(batch['rewards'], batch['values'],
            batch['dones'], batch['lengths'], cfg.algo.gamma, baseline=self.baseline)
        # The reverse scalar recurrence runs on host rollout storage, avoiding
        # thousands of tiny CUDA launches. PPO tensors move to the device once.
        batch = {key: value.to(device) for key, value in batch.items()}
        advantages, returns = advantages.to(device), returns.to(device)
        self.decisions = decisions  # episode-major, exactly matching batch order
        batch['episode_starts'] = torch.zeros(len(rows), dtype=torch.bool, device=device)
        batch['episode_starts'][starts] = True
        elapsed = time.perf_counter() - started
        metrics = {'Env/Reward': counts['raw'], 'Env/Raw_Reward': counts['raw'],
            'Train/Reward': counts['training'], 'Env/Episodes': self.quota,
            'Env/Success_Count': counts['successes'], 'Env/Success_Rate': counts['successes'] / self.quota,
            'Env/Steps': counts['steps'], 'Env/Num_Envs': self.n,
            'Perf/Rollout_Seconds': elapsed, 'Perf/Env_Steps_Per_Second': counts['steps'] / max(elapsed, 1e-9),
            'Action/Normalization_Clip_Fraction': counts['clipped'] / max(counts['actions'], 1),
            'Return/Complete_Episodes': self.quota, 'Return/Decisions': len(rows),
            'Adv/Uses_Learned_Baseline': int(self.baseline == 'value'),
            'Return/Budget_Overshoot': max(0, counts['steps'] - remaining_steps) if remaining_steps is not None else 0}
        return batch, advantages, returns, metrics
