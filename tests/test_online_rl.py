"""CPU contracts for the GPU adapter, potential rewards and sustained RL loop."""
import json
from pathlib import Path
from types import SimpleNamespace
import gymnasium as gym
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from algos.pg import compute_gae
from envs.online_task import task_protocol, critic_features, check_resume_protocol, OnlineTaskWrapper
from envs.online_vector import GPUOnlineEnv, SingleOnlineEnv
from envs.stackcube_training import shaped_reward
from envs.chunk_wrapper import ChunkActionWrapper
from tests.test_online_task import config, DenseNeverSuccess
from tests.test_flow_ppo import TinyPolicy, ToyEnv, actor


def profile():
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / 'configs'), version_base=None):
        return compose(config_name='train_online_rl')


def test_profile_actor_protocol_matches_existing_budget():
    from workflows.online import validate
    from utils.config import validate_common
    cfg = profile()
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / 'configs'), version_base=None):
        old = compose(config_name='train_online', overrides=['+experiment=oc_budget'])
    assert cfg.env == old.env and cfg.model == old.model
    validate_common(cfg); validate(cfg)
    p = task_protocol(cfg)
    assert p['critic_input_dim'] == 54 and cfg.algo.gamma == 1
    changed = dict(p, potential_scale=.5)
    with pytest.raises(ValueError, match='potential_scale'):
        check_resume_protocol(dict(online_protocol=p), changed)


@pytest.mark.parametrize('success', [False, True])
@pytest.mark.parametrize('gamma', [1., .9])
def test_potential_telescopes_through_chunk_and_terminal(success, gamma):
    p = dict(success_reward=1., potential_scale=.7, gamma=gamma)
    potentials = [.2, .6, .6, .8, .9]
    rewards = [shaped_reward(success and t == 3, potentials[t], potentials[t+1], t == 3, p) for t in range(4)]
    total = sum(gamma**t * r for t, r in enumerate(rewards))
    assert total == pytest.approx(gamma**3 * success - .7 * potentials[0])
    if gamma == 1:
        assert rewards[1] == 0  # holding the same intermediate state pays nothing
    # Chunk boundaries do not change the primitive-step objective.
    assert rewards[0] + gamma * rewards[1] + gamma**2 * (rewards[2] + gamma * rewards[3]) == pytest.approx(total)


@pytest.mark.parametrize('success_at', [None, 1, 3])
def test_cpu_potential_wrapper_terminal_and_reset(tmp_path, monkeypatch, success_at):
    import envs.stackcube_training as task
    cfg = config(tmp_path)
    cfg.env.env_id = 'StackCube-v1'; cfg.model.state_dim = 16
    cfg.algo.gamma = 1
    cfg.online_task.reward_mode = 'success_potential'
    cfg.online_task.critic_input = 'privileged_stackcube'
    p = task_protocol(cfg)
    def state(env):
        return torch.zeros(1, 53), torch.tensor([.2 + env.unwrapped.t * .1]), {}
    monkeypatch.setattr(task, 'stackcube_state', state)
    env = ChunkActionWrapper(OnlineTaskWrapper(DenseNeverSuccess(success_at=success_at), p), 3, 2)
    for _ in range(2):
        _, info = env.reset(); total = 0
        assert info['privileged_state'].shape == (53,)
        while True:
            _, reward, term, trunc, _ = env.step(np.zeros((3, 1)))
            total += reward
            if term or trunc:
                break
        assert total == pytest.approx(float(success_at is not None) - .2)


def test_batched_gae_never_crosses_lanes():
    rewards = torch.tensor([[0., 0.], [1., 0.], [0., 1.]])
    dones = torch.tensor([[0., 0.], [1., 0.], [0., 1.]])
    zeros = torch.zeros_like(rewards)
    adv, _ = compute_gae(rewards, zeros, zeros, dones, dones, torch.ones_like(rewards), 1., 1.)
    torch.testing.assert_close(adv, torch.tensor([[1., 1.], [1., 1.], [0., 1.]]))


def test_privileged_features_are_separate_and_time_is_per_lane():
    cfg = profile(); p = task_protocol(cfg)
    features = torch.randn(2, 256); saved = features.clone()
    out = critic_features(features, np.array([0, 150]), p, np.zeros((2, 53)))
    assert out.shape == (2, 54)
    torch.testing.assert_close(out[:, -1], torch.tensor([1., .5]))
    torch.testing.assert_close(features, saved)
    with pytest.raises(ValueError, match='Missing'):
        critic_features(features, [0, 1], p)


class FakeBatchedStack(gym.Env):
    """CPU implementation of only the public ManiSkill interfaces we consume."""
    num_envs = 2
    device = torch.device('cpu')
    def __init__(self):
        self.observation_space = gym.spaces.Dict({})
        self.action_space = gym.spaces.Box(-1, 1, (2, 7), dtype=np.float32)
        self.single_action_space = gym.spaces.Box(-1, 1, (7,), dtype=np.float32)
        self.cube_half_size = torch.tensor([.02] * 3)
        self.t = torch.zeros(2, dtype=torch.int64)
        self.cubeA = self.cube([18, 118], .0)
        self.cubeB = self.cube([19, 119], .1)
        robot = SimpleNamespace(get_qpos=lambda: self.t[:, None].expand(2, 9).float() / 10,
                                get_qvel=lambda: torch.zeros(2, 9))
        self.agent = SimpleNamespace(robot=robot, tcp=SimpleNamespace(pose=self.cubeA.pose))
        self.resets = []
    def cube(self, ids, x):
        pose = torch.tensor([[x, 0., .02, 1., 0., 0., 0.]]).repeat(2, 1)
        return SimpleNamespace(pose=SimpleNamespace(p=pose[:, :3], raw_pose=pose),
            per_scene_id=torch.tensor(ids), linear_velocity=torch.zeros(2, 3), angular_velocity=torch.zeros(2, 3))
    def evaluate(self):
        success = (self.t >= 1) & torch.tensor([True, False])
        return dict(success=success, is_cubeA_on_cubeB=success,
                    is_cubeA_static=torch.ones(2, dtype=torch.bool), is_cubeA_grasped=~success)
    def obs(self):
        return dict(pointcloud=dict(
            xyzw=torch.tensor([[[0., 0., .02, 1.], [.1, 0., .02, 1.]]]).repeat(2, 1, 1),
            rgb=torch.ones(2, 2, 3) * 255,
            segmentation=torch.tensor([[[18], [19]], [[118], [119]]])),
            agent=dict(qpos=self.agent.robot.get_qpos()), extra=dict(tcp_pose=self.agent.tcp.pose.raw_pose))
    def reset(self, seed=None, options=None):
        ids = torch.arange(2) if options is None else options['env_idx']
        self.resets.append(ids.tolist()); self.t[ids] = 0
        return self.obs(), {}
    def step(self, actions):
        self.t += 1
        return self.obs(), torch.ones(2) * .6, torch.zeros(2, dtype=torch.bool), self.t >= 3, self.evaluate()


def test_gpu_adapter_partial_finish_padding_ids_and_reset():
    cfg = profile(); cfg.online_rollout.num_envs = 2; cfg.env.max_episode_steps = 3
    p = task_protocol(cfg); raw = FakeBatchedStack()
    env = GPUOnlineEnv(cfg, p, raw_env=raw)
    observations, infos = env.reset(seed=42)
    assert all(set(obs) == {'point_cloud', 'state'} for obs in observations)
    # The bridge must use the lane's actual IDs (118/119 in lane 1), not IDs 18/19.
    seen_ids = []
    original = env.bridge.observation
    def capture(obs, target_ids=None):
        seen_ids.append(dict(target_ids))
        return original(obs, target_ids=target_ids)
    env.bridge.observation = capture
    actions = np.zeros((2, 16, 7), dtype=np.float32)
    nxt, rewards, terms, truncs, infos = env.step(actions)
    assert [i['actual_steps'] for i in infos] == [1, 2]
    assert terms.tolist() == [True, False] and not truncs.any()
    assert len(infos[0]['online_transitions']) == 1  # padded simulation step excluded
    assert nxt[0]['state'][0] == pytest.approx(.1)  # actual terminal, not padded .2
    assert {'cubeA': 118, 'cubeB': 119} in seen_ids
    terminal = nxt[0]['state'].copy()
    with pytest.raises(RuntimeError, match='reset_done'):
        env.step(actions)
    nxt, infos = env.reset_done(terms | truncs, nxt, infos)
    assert raw.resets[-1] == [0] and nxt[0]['state'][0] == 0
    np.testing.assert_array_equal(terminal, np.array([.1] * 9 + [0, 0, .02, 1, 0, 0, 0], np.float32))
    nxt, rewards, terms, truncs, infos = env.step(actions)
    assert [i['actual_steps'] for i in infos] == [1, 1]
    assert terms[0] and truncs[1]
    assert infos[1]['online_transitions'][-1]['termination_reason'] == 'timeout'
    assert infos[1]['online_transitions'][-1]['training_reward'] < 0  # remove terminal potential
    env.close()


def test_budget_final_evaluation_checkpoint_and_resume(tmp_path, monkeypatch):
    from workflows import online
    cfg = config(tmp_path); cfg.epochs = 100
    cfg.online_training = dict(total_env_steps=7, eval_every_env_steps=1000,
                               target_success_rate=.95, keep_epoch_checkpoints=False)
    cfg.output = str(tmp_path/'run')
    torch.save(TinyPolicy().state_dict(), cfg.algo.pretrained_ckpt)
    (tmp_path/'dataset_stats.json').write_text(json.dumps({k: dict(min=[-1.], max=[1.]) for k in ('state', 'action')}))
    monkeypatch.setattr(online, 'build_base', lambda cfg, device: TinyPolicy().to(device))
    factory = lambda: ChunkActionWrapper(ToyEnv(), chunk_size=3, exec_steps=2)
    result = online.run(cfg, factory)
    assert 7 <= result['total_env_steps'] <= 8
    run = Path(cfg.output)
    cp = torch.load(run/'checkpoints/last.pth', weights_only=True)
    assert cp['epoch'] == 2 and cp['actor_optimizer']['state']
    assert not list((run/'checkpoints').glob('epoch_*.pth'))
    goal = json.loads((run/'goal_status.json').read_text())
    assert goal['final'] and goal['independent_test_required']
    assert (run/'eval/validation_ep0002_cps/summary.json').exists()
    cfg.resume = str(run/'checkpoints/last.pth')
    cfg.output = str(tmp_path/'resumed'); cfg.online_training.total_env_steps = 13
    result2 = online.run(cfg, factory)
    cp2 = torch.load(Path(cfg.output)/'checkpoints/last.pth', weights_only=True)
    assert 13 <= result2['total_env_steps'] <= 14
    assert cp2['epoch'] == 3 and cp2['selection_best']['epoch'] == 0


def test_two_lane_collector_alignment_and_terminal_bootstrap(tmp_path):
    from workflows.online_rollout import OnlineCollector
    from utils.normalizer import MinMaxNormalizer
    cfg = config(tmp_path); cfg.algo.gamma = 1.; cfg.algo.gae_lambda = 1.
    cfg.algo.steps_per_epoch = 2
    p = task_protocol(cfg)
    class Pool:
        num_envs = 2
        def __init__(self):
            self.envs = [SingleOnlineEnv(ChunkActionWrapper(OnlineTaskWrapper(
                DenseNeverSuccess(success_at=s), p), 3, 2)) for s in (1, 3)]
        def reset(self, seed=None):
            rows = [e.reset(seed=seed) for e in self.envs]
            return [r[0][0] for r in rows], [r[1][0] for r in rows]
        def step(self, actions):
            rows = [e.step(actions[i:i+1]) for i, e in enumerate(self.envs)]
            return ([r[0][0] for r in rows], np.array([r[1][0] for r in rows]),
                    np.array([r[2][0] for r in rows]), np.array([r[3][0] for r in rows]),
                    [r[4][0] for r in rows])
        def reset_done(self, done, obs, infos):
            for i in np.flatnonzero(done):
                o, f = self.envs[i].reset_done([True], [obs[i]], [infos[i]])
                obs[i], infos[i] = o[0], f[0]
            return obs, infos
    critic = torch.nn.Linear(3, 1)
    torch.nn.init.zeros_(critic.weight); torch.nn.init.zeros_(critic.bias)
    encode = lambda obs: torch.tensor([[float(o[0]), 0.] for o in obs])
    collector = OnlineCollector(Pool(), actor(), critic, encode, MinMaxNormalizer(), cfg, p, tmp_path/'diag', 0)
    batch, adv, returns, metrics = collector.collect(1)
    assert batch['lengths'].tolist() == [1, 2, 1, 1]
    assert batch['bootstrap_mask'].tolist() == [0, 1, 0, 0]
    assert batch['features'][:, 0].tolist() == [0., 0., 0., 2.]
    torch.testing.assert_close(returns, torch.ones(4))
    assert metrics['Env/Steps'] == 5 and metrics['Env/Success_Count'] == 3
    collector.attach_advantages(adv, adv, True)
    assert collector.flush(1, final=True)['Episode/success_Count'] == 3


def test_target_report_uses_counts_and_distinguishes_confidence():
    from tools.diagnostics.check_success_target import target_report
    summary = dict(results=[dict(stage='rl', sampler='cps', checkpoint='best.pth',
                                **{'Eval/Episodes': 1000, 'Eval/Success_Count': 950})])
    report = target_report(summary)
    assert report['measured_target_met'] and not report['ci95_lower_reaches_target']
    summary['results'][0]['Eval/Success_Count'] = 940
    assert not target_report(summary)['measured_target_met']
    with pytest.raises(ValueError, match='exactly one'):
        target_report(summary, stage='missing')


def test_isolated_evaluation_request_and_error_propagation(monkeypatch):
    import subprocess
    from evaluation import online_worker
    cfg = profile()
    called = []
    def process(command, cwd, check):
        assert command[1:3] == ['-m', 'evaluation.online_worker'] and check
        request = torch.load(command[-2], weights_only=True)
        assert request['weights']['test'].device.type == 'cpu'
        assert request['mode'] == 'cps' and request['metadata']['benchmark_protocol']['adapter'] == 'none'
        called.append(command)
        Path(command[-1]).write_text(json.dumps({'Eval/Success_Rate': .5}))
    monkeypatch.setattr(online_worker.subprocess, 'run', process)
    result = online_worker.isolated_evaluate(cfg, {'test': torch.ones(1)}, {}, 'cps',
        'test', None, dict(benchmark_protocol=dict(adapter='none')))
    assert result['Eval/Success_Rate'] == .5 and len(called) == 1
    def failed(*args, **kwargs):
        raise subprocess.CalledProcessError(1, 'worker')
    monkeypatch.setattr(online_worker.subprocess, 'run', failed)
    with pytest.raises(subprocess.CalledProcessError):
        online_worker.isolated_evaluate(cfg, {}, {}, 'cps', 'test', None, {})
