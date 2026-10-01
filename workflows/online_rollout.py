"""On-policy [time, environment] rollouts; flatten ONLY after independent GAE."""
import os
import time
import numpy as np
import torch
from tqdm import tqdm
from algos.pg import compute_gae
from envs.online_task import critic_features, boundary_masks
from utils.online_diagnostics import EpisodeDiagnostics


class OnlineCollector:
    def __init__(self, env, actor, critic, encode_many, normalizer, cfg, protocol, directory, seed):
        self.env, self.actor, self.critic = env, actor, critic
        self.encode, self.normalizer, self.cfg, self.protocol = encode_many, normalizer, cfg, protocol
        self.n = env.num_envs
        self.elapsed = np.zeros(self.n, dtype=np.int64)
        self.episode_success = np.zeros(self.n, dtype=bool)
        self.diagnostics = [EpisodeDiagnostics(directory if self.n == 1 else os.path.join(directory, f'env_{i:03d}'),
            cfg.algo.gamma, cfg.get('online_diagnostics')) for i in range(self.n)]
        observations, self.infos = env.reset(seed=seed)
        self.features = self.encode(observations)

    def value_input(self, features, infos):
        state = None
        if self.protocol['critic_input_format'] == 'privileged_stackcube_v1':
            state = np.stack([info['privileged_state'] for info in infos])
        return critic_features(features, self.elapsed, self.protocol, state)

    @torch.no_grad()
    def collect(self, epoch, remaining_steps=None):
        started = time.perf_counter()
        cfg, device = self.cfg, self.features.device
        data, decisions = {}, []
        counts = dict(raw=0., training=0., episodes=0, successes=0, steps=0, clipped=0, actions=0)
        self.actor.eval(); self.critic.eval()
        for _ in tqdm(range(cfg.algo.steps_per_epoch), desc=f'Epoch {epoch} rollout', disable=cfg.quiet):
            action, chain, logprob = self.actor.collect(self.features)
            value_input = self.value_input(self.features, self.infos)
            values = self.critic(value_input).squeeze(-1)
            raw = action.cpu().numpy()
            if not np.isfinite(raw).all():
                raise FloatingPointError('Nonfinite generated action')
            nxt, _, terms, truncs, infos = self.env.step(self.normalizer.unnormalize(raw, 'action'))
            done = terms | truncs
            lengths = np.array([int(info['actual_steps']) for info in infos])
            rewards, masks = [], []
            for i, info in enumerate(infos):
                primitive = np.asarray(info['primitive_rewards'], dtype=np.float64)
                if primitive.shape != (lengths[i],) or not 1 <= lengths[i] <= cfg.env.exec_steps:
                    raise ValueError('Invalid primitive rewards/length')
                reward = float(np.dot(cfg.algo.gamma ** np.arange(lengths[i]), primitive)) * cfg.algo.reward_scale
                rewards.append(reward)
                masks.append(boundary_masks(bool(terms[i]), bool(truncs[i]), self.protocol))
                decisions.append(self.diagnostics[i].record(info['online_transitions'], values[i].item(), reward, epoch))
                counts['raw'] += sum(e['raw_reward'] for e in info['online_transitions'])
                counts['training'] += sum(e['training_reward'] for e in info['online_transitions'])
                counts['clipped'] += int((np.abs(raw[i, :lengths[i]]) > 1.1).sum())
                counts['actions'] += raw[i, :lengths[i]].size
                self.episode_success[i] |= bool(info.get('success_any', info.get('success', False)))
                if done[i]:
                    counts['episodes'] += 1
                    counts['successes'] += int(self.episode_success[i])
                    self.episode_success[i] = False
            self.elapsed += lengths
            next_features = self.encode(nxt)  # terminal observation, never reset observation
            next_input = self.value_input(next_features, infos)
            bootstrap, trace = np.array(masks).T
            next_values = self.critic(next_input).squeeze(-1) * torch.as_tensor(bootstrap, device=device)
            row = dict(features=self.features, chains=chain, logprobs=logprob,
                values=values, next_values=next_values, rewards=rewards,
                terminated=terms, dones=done, lengths=lengths, critic_features=value_input,
                bootstrap_mask=bootstrap, trace_mask=trace)
            for key, value in row.items():
                tensor = value if torch.is_tensor(value) else torch.as_tensor(np.asarray(value), dtype=torch.float32)
                data.setdefault(key, []).append(tensor.detach().cpu())
            counts['steps'] += int(lengths.sum())
            self.elapsed[done] = 0
            if done.any():
                nxt, infos = self.env.reset_done(done, nxt, infos)
                # Re-encode only reset lanes; continuing lanes keep the same sampled observation.
                ids = np.flatnonzero(done)
                next_features[torch.as_tensor(ids, device=device)] = self.encode([nxt[i] for i in ids])
            self.features, self.infos = next_features, infos
            if remaining_steps is not None and counts['steps'] >= remaining_steps:
                break
        batch = {key: torch.stack(rows).to(device) for key, rows in data.items()}
        if not all(torch.isfinite(value).all() for value in batch.values()):
            raise FloatingPointError('Nonfinite rollout')
        # [T, N] avoids linking advantages across unrelated environments.
        advantages, returns = compute_gae(batch['rewards'], batch['values'], batch['next_values'],
            batch['terminated'], batch['dones'], batch['lengths'], cfg.algo.gamma, cfg.algo.gae_lambda,
            bootstrap_mask=batch['bootstrap_mask'], trace_mask=batch['trace_mask'])
        batch = {key: value.flatten(0, 1) for key, value in batch.items()}
        self.decisions = decisions  # same time-major order as flatten(0, 1)
        metrics = {'Env/Reward': counts['raw'], 'Env/Raw_Reward': counts['raw'],
            'Train/Reward': counts['training'], 'Env/Episodes': counts['episodes'],
            'Env/Success_Count': counts['successes'],
            'Env/Success_Rate': counts['successes'] / counts['episodes'] if counts['episodes'] else None,
            'Env/Steps': counts['steps'], 'Env/Num_Envs': self.n,
            'Perf/Rollout_Seconds': time.perf_counter() - started,
            'Perf/Env_Steps_Per_Second': counts['steps'] / max(time.perf_counter() - started, 1e-9),
            'Action/Normalization_Clip_Fraction': counts['clipped'] / max(counts['actions'], 1)}
        return batch, advantages.flatten(), returns.flatten(), metrics

    def attach_advantages(self, advantages, normalized, actor_enabled):
        EpisodeDiagnostics.attach_advantages(self.decisions, advantages, normalized, actor_enabled)

    def flush(self, epoch, final=False):
        result = {}
        for diagnostics in self.diagnostics:
            for key, value in diagnostics.flush(epoch, final).items():
                result[key] = result.get(key, 0) + value
        return result
