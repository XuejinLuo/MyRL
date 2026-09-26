"""Planner adapter math, primitive boundaries and isolation of viewer overlays."""
from types import SimpleNamespace
import numpy as np
import pytest

from workflows.pose_tracking import (pose_error, quaternion_product,
                                     bounded_controller_action, WaypointTracker)
from tools.collection.sapien_takeover import clean_render_scene, copy_state, same_state, SapienCollector


def rotation(axis, angle):
    return np.r_[np.cos(angle/2), np.asarray(axis)*np.sin(angle/2)]


def rotate(q, v):
    return quaternion_product(quaternion_product(q, np.r_[0, v]), q*np.array([1, -1, -1, -1]))[1:]


@pytest.mark.parametrize('rotation_sign', [-1., 1.])
def test_controller_inverse_obeys_installed_rotation_sign_and_rotated_root(rotation_sign):
    root = rotation([0, 0, 1], .8)
    current = np.r_[.1, .2, .3, rotation([1, 0, 0], 1.2)]
    def predict(action):
        local_q = np.array([1., 0, 0, 0])
        for axis, angle in zip(np.eye(3), action[3:]*rotation_sign*.1):
            local_q = quaternion_product(local_q, rotation(axis, angle))
        world_q = quaternion_product(quaternion_product(root, local_q), root*np.array([1,-1,-1,-1]))
        return np.r_[current[:3]+rotate(root, action[:3]*.1), quaternion_product(world_q, current[3:])]
    target = np.r_[current[:3]+[.04, -.02, .01], quaternion_product(rotation([0, 1, 0], .2), current[3:])]
    action = bounded_controller_action(predict, current, target, np.full(6, -1.), np.ones(6))
    expected = pose_error(target, current)
    expected[:3] *= .005/np.linalg.norm(expected[:3])
    expected[3:] *= .03/np.linalg.norm(expected[3:])
    np.testing.assert_allclose(pose_error(predict(action), current), expected, atol=2e-6)
    assert np.linalg.norm(action[3:]) <= 1.


def test_quaternion_double_cover_and_invalid_inputs():
    a = np.r_[np.zeros(3), rotation([0,0,1], np.pi-.001)]
    b = a.copy()
    b[3:] *= -1
    np.testing.assert_allclose(pose_error(a,b), 0., atol=1e-9)
    with pytest.raises(ValueError):
        pose_error(np.zeros(7), a)
    with pytest.raises(ValueError):
        pose_error(np.full(7,np.nan), a)


def test_unreachable_bounds_are_respected_and_invalid_controller_fails():
    current = np.array([0., 0, 0, 1, 0, 0, 0])
    def predict(action):
        q = rotation([0,0,1], action[5]*.1)
        q = quaternion_product(rotation([0,1,0], action[4]*.1), q)
        q = quaternion_product(rotation([1,0,0], action[3]*.1), q)
        return np.r_[action[:3]*.1,q]
    target = current.copy()
    target[0] = .1
    low, high = np.full(6, -.01), np.full(6, .01)
    action = bounded_controller_action(predict,current,target,low,high)
    assert (action <= high+1e-8).all() and (action >= low-1e-8).all()
    with pytest.raises(ValueError, match='invert'):
        bounded_controller_action(lambda a: current, current, target, low, high)
    with pytest.raises(ValueError, match='Zero'):
        bounded_controller_action(lambda a: target, current, target, low, high)


def test_waypoint_tracker_completes_stops_on_stall_and_enforces_step_budget():
    current = np.array([0., 0, 0, 1, 0, 0, 0])
    target = current.copy()
    target[0] = .1
    track = WaypointTracker([current, target], stall_steps=2)
    assert track.action(current, lambda p: p) is not None
    assert track.action(current, lambda p: p) is not None
    assert track.action(current, lambda p: p) is None and track.status == 'stalled'
    track = WaypointTracker([current, target], max_steps=1)
    assert track.action(current, lambda p: p) is not None
    moved = current.copy()
    moved[0] = .01
    assert track.action(moved, lambda p: p) is None and track.status == 'step_budget'
    track = WaypointTracker([target])
    assert track.action(target, lambda p: pytest.fail('Already at target')) is None
    assert track.status == 'reached'


def test_viewer_ghosts_decorations_and_opacity_are_removed_before_sensor_reads():
    class ControlWindow:
        show_camera_linesets = True
        show_joint_axes = True
        show_origin_frame = True
    control = ControlWindow()
    viewer = SimpleNamespace(closed=False, selected_entity_visibility=.2, plugins=[control])
    ghosts = [1,2]
    transform = SimpleNamespace(clear_ghost_objects=ghosts.clear)
    with pytest.raises(RuntimeError):
        with clean_render_scene(viewer, transform):
            assert not ghosts and viewer.selected_entity_visibility == 1.
            assert not control.show_camera_linesets and not control.show_joint_axes and not control.show_origin_frame
            with clean_render_scene(viewer, transform):
                assert not control.show_origin_frame
            raise RuntimeError('sensor capture failed')
    assert control.show_camera_linesets and control.show_joint_axes and control.show_origin_frame
    assert not ghosts  # restoration cannot pollute next observation


def test_external_state_edits_are_detected_without_aliasing():
    state = {'actors': {'cube': np.array([1.,2,3])}, 'articulations': {'robot': np.zeros(9)}}
    saved = copy_state(state)
    assert same_state(saved,state)
    state['articulations']['robot'][0] = .01
    assert not same_state(saved,state)
    assert saved['articulations']['robot'][0] == 0
    ui = SapienCollector.__new__(SapienCollector)
    ui.physical_state = saved
    ui.env = SimpleNamespace(unwrapped=SimpleNamespace(get_state_dict=lambda: state))
    with pytest.raises(RuntimeError, match='without an action'):
        ui._assert_unchanged()


class Pose:
    def __init__(self, p=(0,0,0)):
        self.p, self.q = np.asarray(p), np.array([1.,0,0,0])
    def __mul__(self, other):
        return Pose(self.p+other.p)


def planner_ui(result):
    ui = SapienCollector.__new__(SapienCollector)
    ui.active, ui.tracker, ui.gripper_steps = True, None, 0
    ui.control = SimpleNamespace(mode='human', steps=7)
    ui._assert_unchanged = lambda: None
    ui.hand_entity = object()
    ui.viewer = SimpleNamespace(selected_entity=ui.hand_entity, closed=False, plugins=[], selected_entity_visibility=1.)
    ui.transform = SimpleNamespace(_gizmo_pose=Pose([.1,0,0]), clear_ghost_objects=lambda: None)
    ui.hand_to_tcp = Pose([0,0,.1])
    ui.arm = SimpleNamespace(ee_pose=Pose(), active_joint_indices=np.arange(7), ee_link=SimpleNamespace(index=0))
    calls = []
    def plan(target, dry_run):
        calls.append((target,dry_run))
        return result
    ui.planner = SimpleNamespace(move_to_pose_with_screw=plan,
        follow_path=lambda _: pytest.fail('Must not execute joint-pos path'))
    class Pin:
        def compute_forward_kinematics(self,qpos): self.p = qpos[:3].copy()
        def get_link_pose(self,index): return Pose(self.p)
    ui.pin = Pin()
    robot = SimpleNamespace(get_qpos=lambda: np.zeros((1,9)), pose=SimpleNamespace(sp=Pose()))
    ui.env = SimpleNamespace(unwrapped=SimpleNamespace(agent=SimpleNamespace(robot=robot), control_freq=20))
    ui._message = lambda message: setattr(ui,'notice',message)
    return ui, calls


def test_official_planner_is_dry_run_and_hand_offset_is_applied(tmp_path):
    result = dict(status='Success', position=np.zeros((3,7)))
    ui, calls = planner_ui(result)
    ui.path = tmp_path/'seed_1.npz'
    ui.execute_target()
    assert calls[0][1] is True
    np.testing.assert_allclose(calls[0][0].p,[.1,0,.1])
    assert isinstance(ui.tracker, WaypointTracker)
    assert len(ui.tracker.waypoints) == 4
    assert ui.path.with_suffix('.targets.jsonl').exists()
    ui.cancel_motion()
    assert ui.tracker is None and ui.gripper_steps == 0


@pytest.mark.parametrize('result', [-1, dict(status='Failure'), dict(status='Success',position=np.zeros((150,7)))])
def test_planning_failure_never_executes_or_records_an_action(tmp_path,result):
    ui, calls = planner_ui(result)
    ui.path = tmp_path/'seed_1.npz'
    ui.execute_target()
    assert len(calls) == 1 and ui.tracker is None
    assert not ui.path.with_suffix('.targets.jsonl').exists()


def test_planner_exception_keeps_live_episode_available_for_another_target(tmp_path):
    ui, _ = planner_ui(None)
    ui.path = tmp_path/'seed_1.npz'
    def bad_plan(*args, **kwargs):
        raise RuntimeError('no IK solution')
    ui.planner.move_to_pose_with_screw = bad_plan
    ui.execute_target()
    assert ui.active and ui.tracker is None
    assert 'Planner error' in ui.notice
    assert not ui.path.with_suffix('.targets.jsonl').exists()


def test_finish_saves_review_but_keeps_viewer_environment_until_next_episode(tmp_path):
    from tools.collection.human_takeover import CollectorUI
    from tests.test_iterative_sampling import episode
    import json
    ui = CollectorUI.__new__(CollectorUI)
    closes = []
    ui.env = SimpleNamespace(close=lambda: closes.append('env'), unwrapped=SimpleNamespace(
        agent=SimpleNamespace(uid='panda_wristcam', controller=SimpleNamespace(configs={}))))
    ep = episode(4,True)
    ui.control = SimpleNamespace(steps=4, done=True, episode=lambda: ep,
        segments=[dict(id=0,start=0,stop=4)], low=np.full(7,-1.), high=np.ones(7), events=[])
    ui.active, ui.running = True, True
    ui.path, ui.output, ui.seed = tmp_path/'seed_1.npz',tmp_path,1
    ui.telemetry = SimpleNamespace(errors={})
    ui.video_path, ui.video_error = None, None
    ui.writer = SimpleNamespace(close=lambda: closes.append('writer'))
    ui.events_handle = SimpleNamespace(close=lambda: closes.append('events'))
    ui.review = dict(schema='myrl_human_review_v1',episodes=[])
    ui.refresh_status = lambda: None
    ui.finish()
    assert not ui.active and closes == ['events','writer']
    assert ui.env is not None
    assert json.loads((tmp_path/'review.json').read_text())['episodes'][0]['segments'][0]['decision'] == 'pending'
    ui.close_resources()
    assert closes == ['events','writer','env'] and ui.env is None
