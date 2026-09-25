import csv
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from envs.chunk_wrapper import ChunkActionWrapper
from evaluation.grasp_trace import GraspTrace, relative_position
from evaluation.runner import evaluate_policy
from models.factory import observation_encoder
from models.online_policy import FlowPPOPolicy
from tests.test_flow_ppo import TinyPolicy, ToyEnv
from tests.test_q_selection import norm
from tests.test_stages import config


def position(x=0.):
    return SimpleNamespace(p=np.array([[x, 0., .02]]), q=np.array([[1., 0., 0., 0.]]))


class TraceEnv(ToyEnv):
    control_freq = 20
    def __init__(self, contact_error=False):
        super().__init__()
        self.cubeA, self.cubeB = SimpleNamespace(pose=position(.1)), SimpleNamespace(pose=position(.2))
        self.tcp = SimpleNamespace(pose=position())
        self.robot = SimpleNamespace(pose=position(), get_qpos=lambda: np.full((1, 9), self.t*.01),
            get_qvel=lambda: np.zeros((1, 9)), links_map={'panda_leftfinger': 'left', 'panda_rightfinger': 'right'})
        self.agent = SimpleNamespace(uid='panda', tcp=self.tcp, robot=self.robot,
            controller=SimpleNamespace(controllers={}, configs={}))
        def forces(link, obj):
            if contact_error:
                raise AttributeError('contact API unavailable')
            torch.rand(1)  # Diagnostics must not change policy RNG even if a query consumes it.
            return np.array([[1. if link == 'left' else 2., 0., 0.]])
        self.scene = SimpleNamespace(get_pairwise_contact_forces=forces)
    def reset(self, **kwargs):
        self.tcp.pose = position()
        return super().reset(**kwargs)
    def render(self):
        np.random.rand()  # Video capture must also preserve observation/policy RNG.
        return np.full((16, 16, 3), self.t, dtype=np.uint8)
    def step(self, action):
        result = super().step(action)
        self.tcp.pose.p[0, 0] += float(action[0])*.01
        obs, reward, terminated, truncated, info = result
        info.update(is_cubeA_grasped=self.t == 2)
        return obs, reward, terminated, truncated, info


def read_rows(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def test_trace_exact_prefix_clipping_alignment_and_partial_last_chunk(tmp_path):
    normalizer = norm()
    normalizer.stats['action'] = dict(min=[-2.], max=[2.])
    raw = TraceEnv()
    trace = GraspTrace(raw, tmp_path, normalizer, 2)
    env = ChunkActionWrapper(trace, 3, 2)
    env.reset(seed=6009)
    actions = np.array([[5.], [-5.], [.1]], np.float32)
    trace.record_action(actions)
    env.step(normalizer.unnormalize(actions, 'action'))
    trace.record_action(actions)
    _, _, _, truncated, _ = env.step(normalizer.unnormalize(actions, 'action'))
    assert truncated
    env.close()
    rows = read_rows(tmp_path/'seed_6009/steps.jsonl')
    assert [r['chunk_offset'] for r in rows] == [0, 1, 0]
    assert [r['decision_index'] for r in rows] == [0, 0, 1]
    assert [r['executed_action'][0] for r in rows] == [1., -1., 1.]
    assert rows[0]['normalized_action'] == [5.]
    assert rows[0]['controller_action_before_clip'][0] > 2.
    assert rows[0]['before']['qpos'] == [0.]*9
    assert rows[0]['after']['qpos'] == [.01]*9
    assert rows[0]['after'] == rows[1]['before']
    assert rows[1]['after']['flags']['is_cubeA_grasped'] is True
    assert rows[0]['tcp_actual_delta_world_m'][0] == pytest.approx(.01)
    assert rows[-1]['video_frame_after'] == 3 and rows[-1]['time_after_s'] == .15
    assert rows[0]['after']['cubeA_left_force_world_N'] == [1., 0., 0.]
    assert len(list(csv.DictReader((tmp_path/'seed_6009/steps.csv').open()))) == 3
    status = json.loads((tmp_path/'seed_6009/diagnostic_status.json').read_text())
    assert status['complete'] and status['decisions'] == 2
    assert 'arm_target_pose' in status['optional_errors']
    assert raw.closed


@pytest.mark.parametrize('sampler', ['cps', 'ode'])
def test_instrumentation_preserves_full_evaluation_and_rng(tmp_path, sampler):
    cfg = config('offline', tmp_path)
    actor = FlowPPOPolicy(TinyPolicy(), num_steps=3, eval_mode=sampler)
    normalizer = norm()
    encode = observation_encoder(cfg, actor, normalizer, 'cpu')
    plain = lambda: ChunkActionWrapper(TraceEnv(), 3, 2)
    expected = evaluate_policy(plain, actor, encode, normalizer, [6009, 6014], 3, output_dir=tmp_path/'plain')
    holder = []
    def factory():
        t = GraspTrace(TraceEnv(), tmp_path/'traced', normalizer, 2)
        holder.append(t)
        return ChunkActionWrapper(t, 3, 2)
    rng = torch.get_rng_state().clone()
    actual = evaluate_policy(factory, actor, encode, normalizer, [6009, 6014], 3,
        output_dir=tmp_path/'traced', action_callback=lambda a: holder[0].record_action(a))
    assert torch.equal(rng, torch.get_rng_state())
    assert actual == expected
    assert (tmp_path/'plain/episodes.csv').read_bytes() == (tmp_path/'traced/episodes.csv').read_bytes()
    for seed in [6009, 6014]:
        rows = read_rows(tmp_path/f'traced/seed_{seed}/steps.jsonl')
        assert len(rows) == 3 and rows[0]['primitive_step'] == 0
        assert rows[0]['before']['qpos'] == [0.]*9


def test_missing_optional_force_is_null_with_reason(tmp_path):
    normalizer = norm()
    t = GraspTrace(TraceEnv(contact_error=True), tmp_path, normalizer, 2)
    env = ChunkActionWrapper(t, 3, 2)
    env.reset(seed=1)
    t.record_action(np.zeros((3, 1), np.float32))
    env.step(normalizer.unnormalize(np.zeros((3, 1), np.float32), 'action'))
    env.close()
    rows = read_rows(tmp_path/'seed_1/steps.jsonl')
    assert rows[0]['after']['cubeA_left_force_world_N'] is None
    status = json.loads((tmp_path/'seed_1/diagnostic_status.json').read_text())
    assert not status['complete']
    assert 'contact API unavailable' in status['optional_errors']['cubeA_left_force_world_N']['message']


def test_tcp_relative_coordinates_use_orientation():
    tcp = [1., 2., 3., np.sqrt(.5), 0., 0., np.sqrt(.5)]
    np.testing.assert_allclose(relative_position(tcp, [1., 3., 3., 1., 0., 0., 0.]), [1., 0., 0.], atol=1e-7)


@pytest.mark.parametrize('video_enabled', [False, True])
def test_cli_run_loads_checkpoint_and_writes_complete_artifacts(tmp_path, monkeypatch, video_enabled):
    from tools.diagnostics import trace_grasp
    cfg = config('iterative', tmp_path)
    cfg.env.env_id = 'StackCube-v1'
    cfg.env.exec_steps = 2
    cp = tmp_path/'actor.pth'
    torch.save(dict(model_state_dict=TinyPolicy().state_dict(), normalizer=norm().stats,
                    config=OmegaConf.to_container(cfg, resolve=True)), cp)
    monkeypatch.setattr(trace_grasp, 'build_base', lambda cfg, device: TinyPolicy())
    writers = []
    if video_enabled:
        import imageio.v2 as imageio
        class Writer:
            def __init__(self, path, fps):
                self.path, self.fps, self.frames = path, fps, []
                writers.append(self)
            def append_data(self, frame):
                self.frames.append(frame.copy())
            def close(self):
                pass
        monkeypatch.setattr(imageio, 'get_writer', Writer)
    def make_env(cfg, primitive_wrapper, video=None):
        env = TraceEnv()
        if video:
            from evaluation.video import EvalVideo
            env = EvalVideo(env, **video)
        return ChunkActionWrapper(primitive_wrapper(env), 3, 2)
    monkeypatch.setitem(sys.modules, 'envs.factory', SimpleNamespace(make_env=make_env))
    args = SimpleNamespace(checkpoint=str(cp), output=str(tmp_path/'out'), device='cpu',
                           seeds=[6009, 6014, 6019], sampler='cps', no_video=not video_enabled)
    result = trace_grasp.run(args)
    assert result['Eval/Episodes'] == 3
    assert json.loads((tmp_path/'out/run_status.json').read_text())['complete']
    assert (tmp_path/'out/seed_6019/steps.csv').is_file()
    if video_enabled:
        assert len(writers) == 3
        for writer, seed in zip(writers, args.seeds):
            assert f'seed{seed}.mp4' in writer.path and writer.fps == 20
            assert [int(frame[0, 0, 0]) for frame in writer.frames] == [0, 1, 2, 3]
    with pytest.raises(FileExistsError):
        trace_grasp.run(args)
