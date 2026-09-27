"""Headless screening -> physically replayed policy prefix -> reviewed recovery."""
import json
import sys
from types import SimpleNamespace, ModuleType

import numpy as np
import pytest
from omegaconf import OmegaConf

from data.episodes import digest
from envs.chunk_wrapper import ChunkActionWrapper
from tests.test_human_corrections import SevenActionEnv, seven_normalizer, snapshot
from tools.collection.human_takeover import CollectorUI
from tools.collection.failure_screen import FailureScreen
from workflows.failure_queue import state_vector, check_replay_state, takeover_step, load_queue


class MiningEnv(SevenActionEnv):
    control_freq = 20
    agent = SimpleNamespace(uid='panda_wristcam', controller=SimpleNamespace(controllers={
        'arm': SimpleNamespace(config=SimpleNamespace(use_delta=True, use_target=False,
            normalize_action=True, frame='root_translation:root_aligned_body_rotation'))}))
    drift = False

    def reset(self, seed=None, **kwargs):
        self.seed = seed
        self.q = np.full(3, .01 if self.drift else 0.)
        return super().reset(**kwargs)

    def step(self, action):
        obs, reward, term, trunc, info = super().step(action)
        self.q += action[:3]
        # One seed succeeds transiently: success_any must exclude it from mining.
        info['success'] = (self.seed % 2 == 1 and len(self.actions) == 4) or action[0] > .8
        return obs, reward, term, trunc, info

    def get_state_dict(self):
        return {'robot': {'q': self.q.copy()}, 'time': np.array([len(self.actions)])}


@pytest.fixture
def screen_setup(tmp_path, monkeypatch):
    factory = ModuleType('envs.factory')
    factory.make_env = lambda cfg, primitive_wrapper: ChunkActionWrapper(primitive_wrapper(MiningEnv()), 4, 2)
    monkeypatch.setitem(sys.modules, 'envs.factory', factory)
    class Telemetry:
        def __init__(self, *args): self.errors = {}
        def snapshot(self, obs, info, contacts=True):
            return snapshot(cubeA_pose_world=[0,0,0,1,0,0,0])
    monkeypatch.setattr('tools.collection.human_takeover.GraspTrace', Telemetry)
    args = SimpleNamespace(episodes=2, seed_start=15000, no_video=True, recovery_reserve=4,
        rewind_steps=2, takeover_step=None, auto_pause=True)
    cfg = OmegaConf.create(dict(env=dict(exec_steps=2, max_episode_steps=8),
                                model=dict(num_inference_steps=1, chunk_size=4)))
    import torch
    actor = SimpleNamespace(sample=lambda *a: torch.full((1,4,7), .1))
    output = tmp_path/'screen'
    output.mkdir()
    ui = FailureScreen(args, cfg, actor, lambda obs: obs, seven_normalizer(), output)
    provenance = dict(config=OmegaConf.to_container(cfg), checkpoint_sha256='checkpoint',
                      normalizer=seven_normalizer().stats, sampler='cps', seeds=[15000,15001])
    try:
        queue = ui.scan(provenance)
    finally:
        ui.close_resources()
    return args, cfg, actor, output, queue


def test_mining_skips_success_any_and_records_only_failed_actions_states(screen_setup):
    args, cfg, actor, output, queue = screen_setup
    assert queue['complete'] and len(queue['results']) == 2
    assert queue['results'][1]['success_any'] and not queue['results'][1]['success_final']
    assert [e['seed'] for e in queue['failures']] == [15000]
    assert queue['failures'][0]['takeover_step'] == 4
    assert not list(output.rglob('*.mp4')) and not (output/'episodes').exists()
    assert len(list((output/'failures').iterdir())) == 1
    loaded = load_queue(output/'failure_queue.json', 'checkpoint', queue['config'],
                        queue['normalizer'], 'cps', [])
    assert loaded == queue
    with pytest.raises(ValueError, match='checkpoint'):
        load_queue(output/'failure_queue.json', 'other', queue['config'], queue['normalizer'], 'cps', [])
    with pytest.raises(ValueError, match='excluded'):
        load_queue(output/'failure_queue.json', 'checkpoint', queue['config'], queue['normalizer'], 'cps', [15001])


def playback(screen_setup, tmp_path):
    args, cfg, actor, output, queue = screen_setup
    args.episodes = 1
    args.failure_entries = queue['failures']
    args.failure_queue = str(output/'failure_queue.json')
    # Replay must execute the screened actions, without sampling fresh stochastic actions.
    def never_sample(*args): pytest.fail('Resampling instead of replay')
    actor.sample = never_sample
    ui = CollectorUI.__new__(CollectorUI)
    session = tmp_path/'human'
    session.mkdir()
    ui.init_session(args, cfg, actor, lambda obs: obs, seven_normalizer(), session)
    ui.draw = lambda record=False: None
    ui.refresh_status = lambda: None
    return ui


def test_verified_replay_keeps_policy_prefix_then_records_human_and_organized_files(screen_setup, tmp_path):
    ui = playback(screen_setup, tmp_path)
    try:
        ui.next_episode()
        assert ui.control.steps == 4 and ui.control.mode == 'human' and not ui.running
        assert not ui.control.segments
        assert all(e['source'] == 'policy' and e['replayed_policy'] for e in ui.control.events)
        ui.advance(ui.control.human_command(0,.9))
        ui.finish()
        review = json.loads((ui.output/'review.json').read_text())['episodes'][0]
        assert review['path'] == 'episodes/seed_15000.npz'
        assert review['metadata'] == 'metadata/seed_15000.json'
        assert review['segments'] == [dict(id=0,start=4,stop=5,decision='pending')]
        meta = json.loads((ui.output/review['metadata']).read_text())
        assert meta['failure_screen']['takeover_step'] == 4 and meta['final_success']
        assert (ui.output/'traces'/'seed_15000.events.jsonl').exists()
    finally:
        ui.close_resources()


def test_replay_divergence_rejects_episode_instead_of_teleporting(screen_setup, tmp_path, monkeypatch):
    ui = playback(screen_setup, tmp_path)
    monkeypatch.setattr(MiningEnv, 'drift', True)
    try:
        with pytest.raises(ValueError, match='diverged at step 0'):
            ui.next_episode()
        assert ui.control.steps == 0
        assert not list(ui.output.rglob('*.npz'))
    finally:
        ui.close_resources()


def test_failure_trace_tamper_and_outside_paths_rejected(screen_setup):
    _, _, _, output, queue = screen_setup
    def load():
        return load_queue(output/'failure_queue.json', 'checkpoint', queue['config'], queue['normalizer'], 'cps', [])
    trace = output/queue['failures'][0]['path']
    trace.write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        load()
    queue['failures'][0]['path'] = '../outside.npz'
    (output/'failure_queue.json').write_text(json.dumps(queue))
    with pytest.raises(ValueError, match='leaves'):
        load()


def test_recovery_reserve_rewinds_late_hints_and_early_terminations():
    assert takeover_step([dict(step=80,hints=['empty_closed'])], 300, 300) == 61
    assert takeover_step([dict(step=280,hints=['contact_stall'])], 300, 300) == 120
    assert takeover_step([], 300, 300) == 120
    assert takeover_step([], 2, 300) == 1
    assert takeover_step([], 1, 300) == 0
    assert takeover_step([], 300, 300, override=40) == 40
    with pytest.raises(ValueError, match='takeover-step'):
        takeover_step([], 300, 300, override=200)


def test_state_check_includes_velocity_and_schema():
    state = {'robot': {'qpos': np.zeros(2), 'qvel': np.zeros(2)}}
    schema, values = state_vector(state)
    state['robot']['qvel'][0] = .01
    with pytest.raises(ValueError, match='diverged'):
        check_replay_state(state, schema, values, 1)
