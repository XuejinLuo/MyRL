"""Official SAPIEN TransformWindow + ManiSkill planner, on the live takeover episode.

The planner is dry-run only. FK waypoints are followed by the unchanged EE delta
controller, so recorded actions remain compatible with the learned policy.
"""
from collections import deque
from contextlib import contextmanager
import json
import time
import traceback

import numpy as np
import torch

from evaluation.grasp_trace import pose, array
from evaluation.runner import preserve_rng
from tools.collection.human_takeover import CollectorUI
from utils.experiment import write_json
from workflows.pose_tracking import WaypointTracker, bounded_controller_action, pose_error


def copy_state(value):
    if isinstance(value, dict):
        return {k: copy_state(v) for k, v in value.items()}
    return array(value)


def same_state(a, b):
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same_state(a[k], b[k]) for k in a)
    return np.shape(a) == np.shape(b) and np.allclose(a, b, rtol=0., atol=1e-7)


@contextmanager
def clean_render_scene(viewer, transform):
    """Ghost links copy segmentation IDs: remove them before any sensor capture."""
    decorations = []
    if viewer is not None and not viewer.closed:
        viewer.selected_entity_visibility = 1.
        transform.clear_ghost_objects()
        for plugin in viewer.plugins:
            if type(plugin).__name__ == 'ControlWindow':
                for name in ('show_camera_linesets', 'show_joint_axes', 'show_origin_frame'):
                    decorations.append((plugin, name, getattr(plugin, name)))
                    setattr(plugin, name, False)  # Property setters remove render nodes.
    # Do not recreate ghosts here: only the operator render loop may do that.
    try:
        yield
    finally:
        for plugin, name, visible in decorations:
            setattr(plugin, name, visible)  # Nodes are recreated only at the next viewer render.


def make_panel(owner):
    from sapien import internal_renderer as R
    from sapien.utils.viewer.plugin import Plugin

    class Panel(Plugin):
        def get_ui_windows(self):
            if not hasattr(self, 'window'):
                self.window = R.UIWindow().Label('MyRL takeover').Pos(420, 10).Size(450, 330)
                self.window.append(R.UIDisplayText().Bind(lambda: owner.status_text))
                for label, command in [('Run policy [P]', 'policy'), ('Pause/cancel [Space]', 'pause'),
                    ('Take over / select hand [H]', 'human'), ('Execute dragged target [N]', 'execute'),
                    ('Toggle gripper [G]', 'gripper'), ('Finish/save [F]', 'finish'),
                    ('Save + next seed [C]', 'next'), ('Save + quit [Q]', 'quit')]:
                    self.window.append(R.UIButton().Label(label).Callback(
                        lambda _, cmd=command: owner.commands.append(cmd)))
            return [self.window]

        def after_render(self):
            # SAPIEN's stock Pause checkbox otherwise blocks render() internally,
            # preventing our takeover keys from being processed. Route it to us.
            if self.viewer.paused:
                owner.commands.append('pause')
                self.viewer.paused = False
    return Panel()


class SapienCollector(CollectorUI):
    def __init__(self, args, cfg, actor, encode, normalizer, output):
        self.init_session(args, cfg, actor, encode, normalizer, output)
        self.viewer, self.transform = None, None
        self.commands = deque()
        self.tracker, self.gripper_steps = None, 0
        self.quitting, self.faulted = False, False
        self.status_text, self.last_print = '', None
        self.planner, self.physical_state = None, None

    def _init_viewer(self):
        from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
        raw = self.env.unwrapped
        self.viewer = raw.render_human()
        self.transform = next((p for p in self.viewer.plugins if type(p).__name__ == 'TransformWindow'), None)
        if self.transform is None:
            raise RuntimeError('Installed SAPIEN has no TransformWindow; use --ui buttons or a compatible SAPIEN')
        # Keep the official viewport, drag gizmo and camera controls. Remove
        # editors that can change joints, visibility or physical properties.
        self.viewer.plugins[:] = [p for p in self.viewer.plugins
                                  if type(p).__name__ in ('ControlWindow', 'TransformWindow')]
        self.transform.teleport = lambda _: self._message('Teleport disabled: drag the hand, then press N.')
        self.transform.ui_window = None  # Rebuild Teleport callback after replacing it.
        panel = make_panel(self)
        panel.init(self.viewer)
        self.viewer.plugins.append(panel)
        # Build Transform's move-group UI before select_entity invokes its callback.
        self.transform.get_ui_windows()
        self.viewer.selected_entity_visibility = 1.
        self.arm = raw.agent.controller.controllers['arm']
        self.hand = raw.agent.robot.links_map['panda_hand']
        self.hand_entity = self.hand._objs[0].entity
        # Calibrate actual hand -> controlled TCP offset, no hard-coded 10 cm.
        self.hand_to_tcp = self.hand.pose.sp.inv() * self.arm.ee_pose.sp
        self.planner = PandaArmMotionPlanningSolver(raw, debug=False, vis=False,
            base_pose=raw.agent.robot.pose, visualize_target_grasp_pose=False,
            print_env_info=False, joint_acc_limits=.5, joint_vel_limits=.5)
        self.pin = raw.agent.robot.create_pinocchio_model()
        self.select_hand()

    def select_hand(self):
        if self.viewer is None or self.viewer.closed:
            return
        self.viewer.select_entity(None)
        self.viewer.select_entity(self.hand_entity)
        self.transform.gizmo_matrix = self.hand.pose.sp.to_transformation_matrix()
        self.transform.follow = False

    def _message(self, text):
        self.notice = text
        self.refresh_status()

    def draw(self, record=False):
        # Initial reset observations were generated before the viewer exists.
        with clean_render_scene(self.viewer, self.transform):
            panel = self.render_panel()
            self.record_panel(panel, record)
        self.physical_state = copy_state(self.env.unwrapped.get_state_dict())
        if self.viewer is None:
            self._init_viewer()
        self.refresh_status()

    def _assert_unchanged(self):
        if not same_state(self.physical_state, self.env.unwrapped.get_state_dict()):
            raise RuntimeError('Viewer changed simulator state without an action; this episode is excluded. '
                               'Use N/G, never teleport or external joint/state editors.')

    def advance(self, action=None):
        self._assert_unchanged()
        with clean_render_scene(self.viewer, self.transform):
            super().advance(action)

    def cancel_motion(self):
        self.tracker, self.gripper_steps = None, 0

    def pause(self):
        self.cancel_motion()
        super().pause()

    def takeover(self):
        self.cancel_motion()
        super().takeover()
        if self.active:
            self.select_hand()
            self._message('Drag the ghost hand position/rotation, then N. G toggles gripper. P returns to policy.')

    def run_policy(self):
        self.cancel_motion()
        super().run_policy()

    def execute_target(self):
        if not self.active or self.control.mode != 'human':
            self._message('Press H to take over before dragging/executing a target.')
            return
        self._assert_unchanged()
        if self.viewer.selected_entity != self.hand_entity:
            self._message('Select the panda hand with H; dragging other scene objects is not supported.')
            return
        self.cancel_motion()
        target = self.transform._gizmo_pose * self.hand_to_tcp
        target_vector = np.r_[target.p, target.q]
        if np.linalg.norm(pose_error(target_vector, pose(self.arm.ee_pose))[:3]) > .4:
            self._message('Target too far away (>40 cm); choose a closer recovery subgoal.')
            return
        # Official planner must never call follow_path/open_gripper: those send
        # pd_joint_pos commands and bypass our primitive recorder/control protocol.
        try:
            with preserve_rng(), clean_render_scene(self.viewer, self.transform):
                result = self.planner.move_to_pose_with_screw(target, dry_run=True)
        except Exception as exc:
            self._assert_unchanged()
            self._message(f'Planner error, no execution: {exc}. Adjust target and retry N.')
            return
        self._assert_unchanged()
        if not isinstance(result, dict) or result.get('status') != 'Success':
            self._message('Motion planning failed; no physics steps were executed. Adjust target and retry N.')
            return
        positions = np.asarray(result['position'])
        indices = array(self.arm.active_joint_indices).astype(int).reshape(-1)
        if positions.ndim != 2 or positions.shape[1] != len(indices) or not 0 < len(positions) < 150 or not np.isfinite(positions).all():
            self._message('Invalid/long motion plan; no execution. Choose a closer subgoal.')
            return
        raw = self.env.unwrapped
        qpos = array(raw.agent.robot.get_qpos()).reshape(-1)
        waypoints = []
        for joints in positions:
            qpos[indices] = joints
            self.pin.compute_forward_kinematics(qpos)
            tcp = raw.agent.robot.pose.sp * self.pin.get_link_pose(self.arm.ee_link.index)
            waypoints.append(np.r_[tcp.p, tcp.q])
        waypoints.append(target_vector)
        self.tracker = WaypointTracker(waypoints, max_steps=150,
                                      stall_steps=max(1, int(raw.control_freq)))
        request = dict(step=self.control.steps, target_world=target_vector.tolist(),
                       planner='PandaArmMotionPlanningSolver.move_to_pose_with_screw(dry_run=True)',
                       waypoints=len(waypoints), execution='closed-loop pd_ee_delta_pose')
        with self.path.with_suffix('.targets.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(request)+'\n')
        self._message('Following planned TCP path. Space cancels at the next primitive boundary.')

    def convert_target(self, target):
        arm = self.arm
        current_base = arm.ee_pose_at_base
        current_world = pose(arm.ee_pose)
        def predict(action):
            tensor = torch.as_tensor(np.asarray(action), dtype=torch.float32, device=arm.device)[None]
            with torch.no_grad():
                delta = arm._preprocess_action(tensor)
                predicted = arm.root_link.pose * arm.compute_target_pose(current_base, delta)
            return pose(predicted)
        action = bounded_controller_action(predict, current_world, target,
                                           self.control.low[:6], self.control.high[:6])
        return np.r_[action, self.control.gripper].astype(np.float32)

    def toggle_gripper(self):
        if not self.active or self.control.mode != 'human':
            self._message('Press H before controlling the gripper.')
            return
        self.cancel_motion()
        self.gripper_target = -1. if self.control.gripper >= 0 else 1.
        self.gripper_steps = 10
        self._message('Opening gripper' if self.gripper_target > 0 else 'Closing gripper')

    def motion_step(self):
        if self.tracker is not None:
            tracker = self.tracker
            action = tracker.action(pose(self.arm.ee_pose), self.convert_target)
            if action is None:
                self.tracker = None
                self._message(f'Path {tracker.status}. Inspect grasp; drag next subgoal or use G/P.')
            else:
                self.advance(action)
        elif self.gripper_steps:
            self.gripper_steps -= 1
            self.advance(self.control.human_command(gripper=self.gripper_target))
        elif self.running:
            self.advance()

    def refresh_status(self):
        if not hasattr(self, 'control'):
            return
        c = self.control
        self.status_text = (f'Seed {self.seed}, step {c.steps}/{self.cfg.env.max_episode_steps}, {c.mode}\n'
            f'Success: {bool(c.info.get("success", False))}; hints: {", ".join(c.hints) or "none"}\n'
            f'{self.notice}')
        if getattr(self, 'video_error', None):
            self.status_text += '\nVIDEO ERROR: '+self.video_error
        if self.notice != self.last_print:
            print(self.status_text, flush=True)
            self.last_print = self.notice

    def finish(self):
        self.cancel_motion()
        if self.active:
            self._assert_unchanged()
        super().finish()
        self._message('Saved for review. C starts next seed; Q quits.')

    def close_resources(self, close_env=True):
        if close_env:
            if self.planner is not None:
                self.planner.close()
            self.planner, self.viewer, self.transform = None, None, None
        super().close_resources(close_env)

    def callback_error(self, kind, error, tb):
        self.running, self.active, self.faulted = False, False, True
        self.cancel_motion()
        traceback.print_exception(kind, error, tb)
        write_json(self.output/'error.json', dict(error=f'{kind.__name__}: {error}', seed=getattr(self, 'seed', None)))
        self._message(f'ERROR: {error}. Current episode excluded; Q exits. Previously saved episodes remain.')

    def loop(self):
        last_step = 0.
        keymap = {'p': 'policy', 'h': 'human', 'n': 'execute', 'g': 'gripper',
                  'space': 'pause', 'f': 'finish', 'c': 'next', 'q': 'quit'}
        callbacks = dict(policy=self.run_policy, human=self.takeover, execute=self.execute_target,
                         gripper=self.toggle_gripper, pause=self.pause, finish=self.finish)
        while not self.quitting:
            if self.viewer is None or self.viewer.closed:
                if self.active:
                    self.safe(self.finish)
                break
            self.transform.enabled = self.active and self.control.mode == 'human' and self.tracker is None
            if self.transform.enabled:
                self.transform.update_ghost_objects()
            else:
                self.transform.clear_ghost_objects()
            with preserve_rng():
                self.env.unwrapped.render_human()
            if self.viewer.closed:
                continue
            for key, command in keymap.items():
                if self.viewer.window.key_press(key):
                    self.commands.append(command)
            while self.commands:
                command = self.commands.popleft()
                if command == 'quit':
                    self.safe(self.finish)
                    self.quitting = True
                    break
                if self.faulted:
                    continue
                if command == 'next':
                    self.safe(self.finish)
                    if self.index+1 >= self.args.episodes:
                        self.quitting = True
                        break
                    self.safe(self.next_episode)
                else:
                    self.safe(callbacks[command])
            if self.quitting:
                break
            if self.active and time.monotonic()-last_step >= max(.01, self.args.policy_delay_ms/1000):
                self.safe(self.motion_step)
                last_step = time.monotonic()
            time.sleep(.005)
