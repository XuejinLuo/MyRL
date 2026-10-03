"""Complete episode returns, vector draining, budget and real PPO/resume tests."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from algos.pg import compute_mc_returns
from envs.chunk_wrapper import ChunkActionWrapper
from envs.online_task import task_protocol, OnlineTaskWrapper
from envs.online_vector import GPUOnlineEnv, SingleOnlineEnv
from tests.test_flow_ppo import TinyPolicy, ToyEnv, actor
from tests.test_online_task import config, DenseNeverSuccess
from tests.test_online_rl import FakeBatchedStack, profile
from utils.normalizer import MinMaxNormalizer
from workflows.online_mc import CompleteEpisodeCollector, return_protocol, check_return_resume


@pytest.mark.parametrize('gamma', [.5, .99, 1.])
def test_mc_discount_boundaries_and_baseline_independence(gamma):
    rewards = torch.tensor([.2 + gamma*.3, 1., -.4, -.2])
    values = torch.tensor([100., -100., 20., 30.])
    dones = torch.tensor([0., 1., 0., 1.])  # success, then timeout
    lengths = torch.tensor([2., 1., 2., 1.])
    adv, ret = compute_mc_returns(rewards, values, dones, lengths, gamma)
    torch.testing.assert_close(ret, torch.tensor([.2+gamma*.3+gamma**2, 1., -.4-gamma**2*.2, -.2]))
    torch.testing.assert_close(adv, ret-values)
    _, other = compute_mc_returns(rewards, values * -3, dones, lengths, gamma)
    torch.testing.assert_close(ret, other, rtol=0, atol=0)


@pytest.mark.parametrize('dones,lengths', [([0., 0.], [1., 1.]), ([.5, 1.], [1., 1.]),
                                         ([0., 1.], [0., 1.]), ([0., 1.], [1.5, 1.])])
def test_mc_rejects_partial_or_invalid_episodes(dones, lengths):
    with pytest.raises(ValueError):
        compute_mc_returns(torch.zeros(2), torch.zeros(2), torch.tensor(dones), torch.tensor(lengths))


def test_profile_and_resume_protocol(tmp_path):
    from workflows.online import validate
    from utils.config import validate_common
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1]/'configs'), version_base=None):
        cfg = compose(config_name='train_online_rl_mc')
    validate_common(cfg); validate(cfg)
    assert cfg.env == profile().env and cfg.model == profile().model
    assert cfg.algo.ratio_scope == 'full' and cfg.algo.actor_lr == 1e-7
    assert cfg.algo.return_estimator == 'mc' and cfg.online_rollout.backend == 'cpu'
    assert return_protocol(cfg)['episodes_per_batch'] == 64
    old = config(tmp_path)  # genuinely legacy checkpoint: no estimator field
    checkpoint = dict(config=OmegaConf.to_container(old))
    check_return_resume(checkpoint, old)
    with pytest.raises(ValueError, match='return protocol mismatch'):
        check_return_resume(checkpoint, cfg)
    checkpoint = dict(return_protocol=return_protocol(cfg))
    check_return_resume(checkpoint, cfg)
    cfg.online_rollout.episodes_per_batch = 32
    with pytest.raises(ValueError, match='return protocol mismatch'):
        check_return_resume(checkpoint, cfg)
    cfg.online_rollout.backend = 'gpu'; cfg.online_rollout.num_envs = 3
    with pytest.raises(ValueError, match='multiple'):
        validate(cfg)
    cfg.online_rollout.num_envs = 2
    cfg.online_task.timeout_semantics = 'continuing_bootstrap'
    with pytest.raises(ValueError, match='finite_horizon'):
        validate(cfg)


@pytest.mark.parametrize('success_at', [None, 1, 3])
def test_cpu_complete_quota_ignores_decision_cutoff_and_drains_budget(tmp_path, success_at):
    cfg = config(tmp_path); cfg.algo.return_estimator = 'mc'; cfg.algo.gamma = 1.
    cfg.algo.steps_per_epoch = 1
    cfg.online_rollout = dict(backend='cpu', num_envs=1, episodes_per_batch=3)
    protocol = task_protocol(cfg)
    env = SingleOnlineEnv(ChunkActionWrapper(OnlineTaskWrapper(
        DenseNeverSuccess(success_at=success_at), protocol), 3, 2))
    pi = actor(); original = copy.deepcopy(pi.state_dict())
    critic = torch.nn.Linear(3, 1)
    torch.nn.init.zeros_(critic.weight); torch.nn.init.constant_(critic.bias, 11.)
    encode = lambda obs: torch.tensor([[float(o[0]), 0.] for o in obs])
    collector = CompleteEpisodeCollector(env, pi, critic, encode, MinMaxNormalizer(),
                                         cfg, protocol, tmp_path/'diag', 0)
    batch, adv, returns, metrics = collector.collect(1, remaining_steps=1)
    length = success_at or 3
    assert metrics['Env/Steps'] == 3 * length and metrics['Env/Episodes'] == 3
    assert metrics['Return/Budget_Overshoot'] == 3 * length - 1
    assert batch['dones'].sum() == 3 and batch['episode_starts'].sum() == 3
    torch.testing.assert_close(returns, torch.full_like(returns, float(success_at is not None)))
    torch.testing.assert_close(adv, returns-11.)
    assert not collector.elapsed.any()
    collector.attach_advantages(adv, adv, True)
    assert collector.flush(1, final=True)['Episode/censored_Count'] == 0
    episodes = [json.loads(line) for line in (tmp_path/'diag/episodes.jsonl').read_text().splitlines()]
    assert len(episodes) == 3 and all(e['start_epoch'] == e['end_epoch'] == 1 for e in episodes)
    for key, value in original.items():
        torch.testing.assert_close(value, pi.state_dict()[key], rtol=0, atol=0)
    batch2, _, _, _ = collector.collect(2)
    assert batch2['features'][0, 0] == 0  # no old partial episode survives


def test_gpu_complete_waves_keep_failures_and_exclude_parked_lanes(tmp_path):
    cfg = profile(); cfg.quiet = True; cfg.algo.return_estimator = 'mc'
    cfg.online_rollout.num_envs = 2; cfg.online_rollout.episodes_per_batch = 4
    cfg.env.max_episode_steps = 3
    protocol = task_protocol(cfg); raw = FakeBatchedStack()
    env = GPUOnlineEnv(cfg, protocol, raw_env=raw)
    class Actor:
        sizes = []
        def eval(self):
            pass
        def collect(self, features):
            self.sizes.append(len(features))
            action = torch.zeros(len(features), 16, 7)
            return action, action[:, None], action[:, None]
    pi = Actor()
    critic = torch.nn.Linear(54, 1)
    torch.nn.init.zeros_(critic.weight); torch.nn.init.constant_(critic.bias, 11.)
    encode = lambda obs: torch.tensor(np.stack([o['state'][:2] for o in obs]))
    collector = CompleteEpisodeCollector(env, pi, critic, encode, MinMaxNormalizer(),
                                         cfg, protocol, tmp_path/'diag', 42)
    batch, adv, returns, metrics = collector.collect(1, remaining_steps=1)
    assert pi.sizes == [2, 1, 2, 1]
    assert batch['lengths'].tolist() == [1, 2, 1, 1, 2, 1]
    assert batch['dones'].tolist() == [1., 0., 1., 1., 0., 1.]
    assert batch['episode_starts'].tolist() == [True, True, False, True, True, False]
    assert metrics['Env/Steps'] == 8 and metrics['Env/Episodes'] == 4
    assert metrics['Env/Success_Count'] == 2  # each wave includes the slow failure
    assert raw.resets == [[0, 1], [0, 1], [0, 1]]  # no early/partial reset
    torch.testing.assert_close(returns[:3], returns[3:])
    assert returns[0] > 0 and returns[1] < 0 and returns[2] < 0
    torch.testing.assert_close(adv, returns-11.)
    collector.attach_advantages(adv, adv, True)
    stats = collector.flush(1, final=True)
    assert stats['Episode/success_Count'] == 2 and stats['Episode/timeout_Count'] == 2
    assert stats['Episode/censored_Count'] == 0


def test_gpu_rejects_parking_an_unfinished_lane():
    cfg = profile(); cfg.online_rollout.num_envs = 2
    env = GPUOnlineEnv(cfg, task_protocol(cfg), raw_env=FakeBatchedStack())
    env.reset()
    with pytest.raises(ValueError, match='Only finished'):
        env.step(np.zeros((2, 16, 7)), active=[True, False])


def test_mc_real_ppo_checkpoint_metrics_and_resume(tmp_path, monkeypatch):
    from workflows import online
    cfg = config(tmp_path); cfg.algo.return_estimator = 'mc'
    cfg.online_rollout = dict(backend='cpu', num_envs=1, episodes_per_batch=2)
    cfg.online_training = dict(total_env_steps=7, eval_every_env_steps=100,
                               keep_epoch_checkpoints=False)
    cfg.output = str(tmp_path/'run'); cfg.epochs = 100
    torch.save(TinyPolicy().state_dict(), cfg.algo.pretrained_ckpt)
    (tmp_path/'dataset_stats.json').write_text(json.dumps({k: dict(min=[-1.], max=[1.]) for k in ('state', 'action')}))
    monkeypatch.setattr(online, 'build_base', lambda cfg, device: TinyPolicy().to(device))
    factory = lambda: ChunkActionWrapper(ToyEnv(), chunk_size=3, exec_steps=2)
    result = online.run(cfg, factory)
    assert result['total_env_steps'] == 12  # two batches of two full 3-step episodes
    checkpoint = torch.load(Path(cfg.output)/'checkpoints/last.pth', weights_only=True)
    assert checkpoint['epoch'] == 2 and checkpoint['actor_optimizer']['state']
    assert checkpoint['return_protocol'] == return_protocol(cfg)
    assert checkpoint['selection_best']['epoch'] == 0
    rows = [json.loads(line) for line in (Path(cfg.output)/'metrics.jsonl').read_text().splitlines()]
    for row in rows[1:]:
        assert row['Value/Target_Is_MC'] == 1 and row['Episode/censored_Count'] == 0
        assert row['Value/MC/Before/Target_Mean'] == row['Value/MC/After/Target_Mean']
        # ToyEnv first succeeds at step 3, coincident with timeout.
        assert row['Value/MC_Start/Before/Target_Mean'] == pytest.approx(cfg.algo.gamma**2)
    cfg.resume = str(Path(cfg.output)/'checkpoints/last.pth')
    cfg.output = str(tmp_path/'resumed'); cfg.online_training.total_env_steps = 13
    result = online.run(cfg, factory)
    assert result['total_env_steps'] == 18
    resumed = torch.load(Path(cfg.output)/'checkpoints/last.pth', weights_only=True)
    assert resumed['epoch'] == 3 and resumed['selection_best']['epoch'] == 0
    cfg.algo.return_estimator = 'gae'; cfg.output = str(tmp_path/'rejected')
    with pytest.raises(ValueError, match='return protocol mismatch'):
        online.run(cfg, factory)
    assert not Path(cfg.output).exists()
