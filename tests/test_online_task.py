"""CPU acceptance tests for primitive rewards, time, GAE, diagnostics and resume."""
import copy
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from algos.pg import compute_gae, normalize_advantages
from envs.chunk_wrapper import ChunkActionWrapper
from envs.online_task import (OnlineTaskWrapper, boundary_masks, check_resume_protocol,
                             critic_features, task_protocol)
from utils.online_diagnostics import EpisodeDiagnostics, distribution, value_metrics
from tests.test_flow_ppo import TinyPolicy, ToyEnv


def config(tmp_path):
    return OmegaConf.create(dict(
        model=dict(algo_type='flow', chunk_size=3, action_dim=1, state_dim=1,
                   cond_dim=2, in_channels=3, num_inference_steps=3),
        env=dict(exec_steps=2, num_points=4, max_episode_steps=3,
                 workspace_bounds=[[-1, -1, -1], [1, 1, 1]]),
        algo=dict(steps_per_epoch=3, update_epochs=1, critic_warmup_epochs=1,
            critic_warmup_update_epochs=2, actor_lr=1e-7, critic_lr=3e-4,
            gamma=.99, gae_lambda=.95, clip_ratio=.2, value_clip=.2,
            max_grad_norm=.5, target_kl=.01, ratio_scope='full', noise_level=.7,
            min_std=.0067, reward_scale=1., verify_logprobs=True,
            pretrained_ckpt=str(tmp_path/'actor.pth'), stats_path=None),
        online_task=dict(reward_mode='success_once', success_reward=1., end_on_success=True,
            timeout_semantics='finite_horizon', critic_use_remaining_time=True,
            selection='success_only'),
        online_diagnostics=dict(enabled=True, trace_max_episodes=2,
            trace_max_steps_per_episode=3, task_phase_stats=True),
        eval=dict(every=1, sampler='cps', seeds=[1, 2]),
        epochs=2, batch_size=2, save_epoch=1, seed=1, device='cpu', resume=None,
        eval_only=False, save_dir=str(tmp_path/'runs'), run_name='test', quiet=True,
        wandb=dict(enable=False)))


class DenseNeverSuccess(gym.Env):
    def __init__(self, success_at=None, raw_timeout=None, fail_at=None):
        self.action_space = gym.spaces.Box(-1., 1., (1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(0., 100., (1,), dtype=np.float32)
        self.success_at, self.raw_timeout, self.fail_at = success_at, raw_timeout, fail_at
        self.t = 0

    def reset(self, **kwargs):
        self.t = 0
        return np.array([0.], np.float32), {}

    def step(self, action):
        self.t += 1
        return (np.array([self.t], np.float32), .6, self.t == self.fail_at,
                self.t == self.raw_timeout,
                dict(success=self.t == self.success_at, is_grasped=True))


def wrapped(tmp_path, **kwargs):
    protocol = task_protocol(config(tmp_path))
    primitive = OnlineTaskWrapper(DenseNeverSuccess(**kwargs), protocol)
    return ChunkActionWrapper(primitive, chunk_size=3, exec_steps=2), protocol


def test_dense_stalling_zero_training_return_and_exact_horizon(tmp_path):
    env, protocol = wrapped(tmp_path)
    env.reset()
    _, reward1, done, trunc, info = env.step(np.zeros((3, 1)))
    assert reward1 == 0 and not (done or trunc) and info['actual_steps'] == 2
    obs, reward2, done, trunc, info = env.step(np.zeros((3, 1)))
    assert reward2 == 0 and trunc and not done and info['actual_steps'] == 1
    assert obs[0] == 3 and env.global_step == 3
    event = info['online_transitions'][0]
    assert event['raw_reward'] == .6 and not event['raw_truncated']
    assert event['termination_reason'] == 'timeout'
    assert boundary_masks(done, trunc, protocol) == (0., 0.)
    with pytest.raises(RuntimeError, match='Reset required'):
        env.step(np.zeros((3, 1)))


@pytest.mark.parametrize('success_at,timeout', [(1, None), (1, 1), (2, None), (3, 3)])
def test_success_once_alignment_and_simultaneous_timeout(tmp_path, success_at, timeout):
    env, _ = wrapped(tmp_path, success_at=success_at, raw_timeout=timeout)
    env.reset()
    total, events = 0., []
    while True:
        _, reward, term, trunc, info = env.step(np.zeros((3, 1)))
        events.extend(info['online_transitions']); total += reward
        if term or trunc:
            break
    assert total == 1 and len(events) == success_at and env.global_step == success_at
    assert sum(e['training_reward'] for e in events) == 1
    assert events[-1]['termination_reason'] == 'success'
    assert events[-1]['raw_truncated'] == (timeout is not None)
    assert info['actual_steps'] == (2 if success_at == 2 else 1)


@pytest.mark.parametrize('gamma', [.5, .99, 1.])
def test_chunk_discount_and_three_boundary_semantics(tmp_path, gamma):
    cfg = config(tmp_path); cfg.algo.gamma = gamma
    finite = task_protocol(cfg)
    legacy = {**finite, 'timeout_semantics': 'continuing_bootstrap'}
    # The last state is 3, reset state is 0: bootstrap must use 3.
    for protocol, expected in [(finite, 0.), (legacy, gamma**1 * 3)]:
        boot, trace = boundary_masks(False, True, protocol)
        _, ret = compute_gae(torch.tensor([0.]), torch.tensor([2.]), torch.tensor([3.]),
            torch.tensor([0.]), torch.tensor([1.]), torch.tensor([1.]), gamma, .95,
            bootstrap_mask=torch.tensor([boot]), trace_mask=torch.tensor([trace]))
        assert ret.item() == pytest.approx(expected)
    # Two primitive rewards followed by a one-step success; lambda=1 equals MC.
    rewards = torch.tensor([.2 + gamma * .3, 1.])
    _, ret = compute_gae(rewards, torch.zeros(2), torch.zeros(2),
        torch.tensor([0., 1.]), torch.tensor([0., 1.]), torch.tensor([2., 1.]), gamma, 1.)
    assert ret[0].item() == pytest.approx(.2 + gamma * .3 + gamma**2)
    # Only rollout ended: keep bootstrap and don't manufacture an episode boundary.
    boot, trace = boundary_masks(False, False, finite)
    assert (boot, trace) == (1., 1.)
    _, ret = compute_gae(torch.tensor([.2 + gamma*.3]), torch.zeros(1), torch.tensor([4.]),
        torch.zeros(1), torch.zeros(1), torch.tensor([2.]), gamma, .95,
        bootstrap_mask=torch.tensor([boot]), trace_mask=torch.tensor([trace]))
    assert ret.item() == pytest.approx(.2 + gamma * .3 + gamma**2 * 4)
    assert boundary_masks(True, False, legacy) == (0., 0.)


def test_critic_time_does_not_change_actor_features(tmp_path):
    env, protocol = wrapped(tmp_path)
    original = torch.tensor([[4., 5.]])
    env.reset()
    assert critic_features(original, env.global_step, protocol)[0, -1] == 1
    env.step(np.zeros((3, 1)))
    assert critic_features(original, env.global_step, protocol)[0, -1] == pytest.approx(1/3)
    env.step(np.zeros((3, 1)))
    assert critic_features(original, env.global_step, protocol)[0, -1] == 0
    env.reset()
    assert critic_features(original, env.global_step, protocol)[0, -1] == 1
    torch.testing.assert_close(original, torch.tensor([[4., 5.]]))
    assert critic_features(original, 0, {**protocol, 'critic_use_remaining_time': False}) is original


def test_diagnostics_cross_rollouts_complete_returns_and_censoring(tmp_path):
    env, protocol = wrapped(tmp_path, success_at=3)
    diag = EpisodeDiagnostics(tmp_path/'diag', .5, dict(trace_max_episodes=1, trace_max_steps_per_episode=2))
    env.reset()
    _, _, _, _, info = env.step(np.zeros((3, 1)))
    first = diag.record(info['online_transitions'], .1, 0., 1)
    diag.attach_advantages([first], torch.tensor([-.1]), torch.tensor([0.]), False)
    assert diag.flush(1)['Episode/success_Count'] == 0
    assert diag.current['episode_id'] == 0
    _, _, _, _, info = env.step(np.zeros((3, 1)))
    last = diag.record(info['online_transitions'], .2, 1., 2)
    diag.attach_advantages([last], torch.tensor([.8]), torch.tensor([1.]), True)
    assert diag.flush(2)['Episode/success_Count'] == 1
    rows = [json.loads(x) for x in (tmp_path/'diag/episodes.jsonl').read_text().splitlines()]
    assert len(rows) == 1 and rows[0]['length'] == 3
    assert rows[0]['training_return'] == 1 and rows[0]['training_discounted_return'] == .25
    assert rows[0]['raw_discounted_return'] == pytest.approx(.6 * 1.75)
    assert rows[0]['MC/Start_Target'] == .25
    assert rows[0]['advantages']['all']['advantage']['count'] == 2
    assert rows[0]['first_success_step'] == 3
    assert rows[0]['phases']['grasped']['longest_run'] == 2
    assert json.loads((tmp_path/'diag/traces.jsonl').read_text())['trace_truncated']
    env.reset()
    _, _, _, _, info = env.step(np.zeros((3, 1)))
    diag.record(info['online_transitions'], .1, 0., 3)
    result = diag.flush(3, final=True)
    assert result['Episode/censored_Count'] == 1 and result['Episode/timeout_Count'] == 0
    censored = json.loads((tmp_path/'diag/episodes.jsonl').read_text().splitlines()[-1])
    assert censored['MC/Decision_MSE'] is None and censored['outcome'] == 'censored'
    assert distribution([])['mean'] is None


def test_value_metrics_use_frozen_target_and_zero_variance_null():
    target = torch.tensor([0., 1.]); original = target.clone()
    before = value_metrics(torch.zeros(2), target)
    after = value_metrics(target.clone(), target)
    assert before['MSE'] == .5 and after['MSE'] == 0
    assert before['Target_Variance'] == after['Target_Variance'] == .25
    torch.testing.assert_close(target, original)
    stats = value_metrics(torch.tensor([4., -2.]), torch.zeros(2))
    assert stats['Explained_Variance'] is None and not stats['EV_Valid']


@pytest.mark.parametrize('key,value', [('reward_mode', 'env'), ('timeout_semantics', 'continuing_bootstrap'),
    ('critic_use_remaining_time', False), ('success_reward', 2.), ('gamma', .9), ('version', 2)])
def test_resume_rejects_protocol_changes(tmp_path, key, value):
    protocol = task_protocol(config(tmp_path))
    check_resume_protocol({'online_protocol': protocol}, protocol)
    with pytest.raises(ValueError, match='NEW Critic'):
        check_resume_protocol({'online_protocol': protocol}, {**protocol, key: value})


def test_unversioned_legacy_checkpoint_only_resumes_legacy(tmp_path):
    cfg = config(tmp_path); del cfg['online_task']
    cp = dict(config=OmegaConf.to_container(cfg))
    check_resume_protocol(cp, task_protocol(cfg))
    with pytest.raises(ValueError, match='protocol mismatch'):
        check_resume_protocol(cp, task_protocol(config(tmp_path)))


def test_full_training_time_inputs_diagnostics_rng_and_resume(tmp_path, monkeypatch):
    from workflows import online
    cfg = config(tmp_path)
    torch.save(TinyPolicy().state_dict(), cfg.algo.pretrained_ckpt)
    (tmp_path/'dataset_stats.json').write_text(json.dumps({k: dict(min=[-1.], max=[1.]) for k in ('state', 'action')}))
    monkeypatch.setattr(online, 'build_base', lambda cfg, device: TinyPolicy().to(device))
    factory = lambda: ChunkActionWrapper(ToyEnv(), chunk_size=3, exec_steps=2)
    captured = []
    original_update = online.FlowPPO.update
    def capture(self, batch, advantages, returns, *args, **kwargs):
        captured.append(copy.deepcopy(batch))
        return original_update(self, batch, advantages, returns, *args, **kwargs)
    monkeypatch.setattr(online.FlowPPO, 'update', capture)
    first = Path(online.run(cfg, factory)['run_dir'])
    cp = torch.load(first/'checkpoints/last.pth', weights_only=True)
    initial = torch.load(first/'checkpoints/initial_policy.pth', weights_only=True)
    best = torch.load(first/'checkpoints/best.pth', weights_only=True)
    assert cp['critic']['v_net.net.0.weight'].shape[1] == 3
    assert initial['model_state_dict'].keys() == cp['model_state_dict'].keys()
    assert all(initial['model_state_dict'][k].shape == cp['model_state_dict'][k].shape for k in cp['model_state_dict'])
    assert best['epoch'] == 0  # equal success: retain initial fallback across every update
    assert initial['online_protocol']['reward_mode'] == 'success_once'
    assert 'critic' not in initial  # policy-only initialization/evaluation snapshot
    rows = [json.loads(x) for x in (first/'metrics.jsonl').read_text().splitlines()]
    assert rows[1]['ppo/actor_updates'] == 0 and rows[2]['ppo/actor_updates'] == 2
    assert rows[1]['Value/Before/Target_Mean'] == rows[1]['Value/After/Target_Mean']
    # Actual time tracks one-step terminal chunk and continues across rollout boundary.
    torch.testing.assert_close(captured[0]['critic_features'][:, -1], torch.tensor([1., 1/3, 1.]))
    torch.testing.assert_close(captured[1]['critic_features'][:, -1], torch.tensor([1/3, 1., 1/3]))
    assert captured[0]['features'].shape[1] == 2
    assert captured[0]['bootstrap_mask'].tolist() == [1., 0., 1.]
    assert captured[0]['lengths'].tolist() == [2., 1., 2.]
    # Disable diagnostics: exact same sampled chains, actor and critic parameters.
    cfg.online_diagnostics.enabled = False
    cfg.online_diagnostics.value_before_after = False
    second = Path(online.run(cfg, factory)['run_dir'])
    cp2 = torch.load(second/'checkpoints/last.pth', weights_only=True)
    for section in ('model_state_dict', 'critic'):
        for k, v in cp[section].items():
            torch.testing.assert_close(v, cp2[section][k], rtol=0, atol=0)
    for a, b in zip(captured[:2], captured[2:]):
        torch.testing.assert_close(a['chains'], b['chains'], rtol=0, atol=0)
    cfg.resume = str(first/'checkpoints/last.pth'); cfg.epochs = 3
    resumed = Path(online.run(cfg, factory)['run_dir'])
    cp3 = torch.load(resumed/'checkpoints/last.pth', weights_only=True)
    resumed0 = torch.load(resumed/'checkpoints/epoch_0002.pth', weights_only=True)
    for name in ('critic', 'model_state_dict'):
        for key in cp[name]:
            torch.testing.assert_close(cp[name][key], resumed0[name][key], rtol=0, atol=0)
    for name in ('actor_optimizer', 'critic_optimizer'):
        for param, state in cp[name]['state'].items():
            for key, value in state.items():
                torch.testing.assert_close(value, resumed0[name]['state'][param][key], rtol=0, atol=0)
    assert cp3['epoch'] == 3 and cp3['total_env_steps'] == cp['total_env_steps'] + 5
    assert cp3['selection_best']['epoch'] == 0
    assert cp3['actor_optimizer']['state'] and cp3['critic_optimizer']['state']
    assert cp3['provenance']['initial_actor'] == cp['provenance']['initial_actor']
    cfg.resume = None; cfg.algo.pretrained_ckpt = str(first/'checkpoints/last.pth'); cfg.epochs = 1
    cfg.online_task.reward_mode = 'env'
    fresh = Path(online.run(cfg, factory)['run_dir'])
    fresh0 = torch.load(fresh/'checkpoints/epoch_0000.pth', weights_only=True)
    assert not fresh0['actor_optimizer']['state'] and not fresh0['critic_optimizer']['state']
    # Unified benchmark still sees raw rewards (not success_once rewards).
    summary = json.loads((first/'eval/validation_ep0000_cps/summary.json').read_text())
    assert summary['metadata']['benchmark_protocol']['adapter'] == 'none'


def test_selection_rules_are_online_local(tmp_path):
    from workflows.online import online_selection_score
    from utils.experiment import selection_score
    protocol = task_protocol(config(tmp_path))
    early = {'Eval/Success_Rate': .7, 'Eval/Mean_Reward': 10}
    late = {'Eval/Success_Rate': .7, 'Eval/Mean_Reward': 100}
    assert online_selection_score(early, protocol) == online_selection_score(late, protocol)
    assert selection_score(late) > selection_score(early)
