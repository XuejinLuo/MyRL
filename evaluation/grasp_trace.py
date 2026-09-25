"""Read-only StackCube/Panda traces at the primitive control-step boundary."""
import csv
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from evaluation.runner import preserve_rng
from utils.experiment import write_json


def array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    result = np.asarray(value).copy()
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite diagnostic value')
    return result


def vector(value):
    return array(value).reshape(-1).tolist()


def pose(value):
    return vector(value.p) + vector(value.q)


def scalar(value):
    return array(value).item()


def flatten(value, prefix='', result=None):
    result = {} if result is None else result
    if isinstance(value, dict):
        for key, child in value.items():
            flatten(child, f'{prefix}.{key}' if prefix else key, result)
    elif isinstance(value, list):
        for i, child in enumerate(value):
            flatten(child, f'{prefix}.{i}', result)
    else:
        result[prefix] = value
    return result


def relative_position(tcp, obj):
    # Both poses are world-frame [x,y,z,qw,qx,qy,qz].
    w, x, y, z = np.asarray(tcp[3:]) / np.linalg.norm(tcp[3:])
    rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                         [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                         [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    return (rotation.T @ (np.asarray(obj[:3])-tcp[:3])).tolist()


class GraspTrace(gym.Wrapper):
    """Place inside ChunkActionWrapper, after point-cloud preprocessing/video.

    Required geometry failures abort. Optional/version-specific telemetry is null
    with explicit errors; it is never replaced by a false no-contact reading.
    """
    def __init__(self, env, output, normalizer, exec_steps):
        super().__init__(env)
        self.output, self.normalizer = Path(output), normalizer
        self.exec_steps = exec_steps
        self.directory = None
        self.handles = []
        self.errors = {}

    def optional(self, key, getter):
        try:
            return getter()
        except Exception as exc:
            previous = self.errors.get(key, {})
            self.errors[key] = dict(message=f'{type(exc).__name__}: {exc}',
                                    count=previous.get('count', 0)+1)
            return None

    def snapshot(self, obs, info, contacts=True):
        raw = self.unwrapped
        agent, robot = raw.agent, raw.agent.robot
        row = dict(tcp_pose_world=pose(agent.tcp.pose), robot_pose_world=pose(robot.pose),
                   cubeA_pose_world=pose(raw.cubeA.pose), cubeB_pose_world=pose(raw.cubeB.pose),
                   qpos=vector(robot.get_qpos()), qvel=vector(robot.get_qvel()),
                   observation_state=vector(obs['state']))
        if len(row['qpos']) != 9 or not str(agent.uid).startswith('panda'):
            raise ValueError('Grasp trace currently requires the 9-joint Panda robot')
        row['finger_qpos'] = row['qpos'][-2:]
        row['gripper_width_m'] = sum(row['finger_qpos'])
        row['cubeA_in_tcp_m'] = relative_position(row['tcp_pose_world'], row['cubeA_pose_world'])
        row['cubeA_height_change_m'] = row['cubeA_pose_world'][2]-self.initial_height if hasattr(self, 'initial_height') else 0.
        row['flags'] = {key: bool(scalar(info[key])) if key in info else None for key in
                        ('success', 'is_cubeA_grasped', 'is_cubeA_on_cubeB', 'is_cubeA_static')}
        if contacts and row['flags']['is_cubeA_grasped'] is None:
            def grasp():
                method = getattr(agent, 'is_grasping', None) or getattr(agent, 'check_grasp')
                return bool(scalar(method(raw.cubeA)))
            row['flags']['is_cubeA_grasped'] = self.optional('is_cubeA_grasped', grasp)
        for obj_name in ('cubeA', 'cubeB'):
            obj = getattr(raw, obj_name)
            for finger, link_name in [('left', 'panda_leftfinger'), ('right', 'panda_rightfinger')]:
                key = f'{obj_name}_{finger}_force_world_N'
                row[key] = self.optional(key, lambda obj=obj, link_name=link_name:
                    vector(raw.scene.get_pairwise_contact_forces(robot.links_map[link_name], obj))) if contacts else None
        controllers = getattr(getattr(agent, 'controller', None), 'controllers', {})
        row['controller_targets'] = {}
        for name in ('arm', 'gripper'):
            for attr in ('_target_qpos', '_target_pose'):
                key = f'{name}{attr}'
                def target(name=name, attr=attr):
                    value = getattr(controllers[name], attr)
                    return pose(value) if attr == '_target_pose' else vector(value)
                row['controller_targets'][key] = self.optional(key, target)
        return row

    def reset(self, **kwargs):
        self.finish()
        obs, info = self.env.reset(**kwargs)
        self.seed = int(kwargs['seed'])
        self.directory = self.output/f'seed_{self.seed}'
        self.directory.mkdir(parents=True, exist_ok=False)
        self.errors, self.rows = {}, []
        self.step_index, self.decision_index, self.offset = 0, -1, 0
        self.complete = False
        self.pending = None
        self.initial_height = float(vector(self.unwrapped.cubeA.pose.p)[2])
        self.freq = float(self.unwrapped.control_freq)
        if self.freq <= 0:
            raise ValueError('Invalid control frequency')
        with preserve_rng():
            self.before = self.snapshot(obs, info, contacts=False)
        self.actions_file = (self.directory/'actions.jsonl').open('w', encoding='utf-8')
        self.steps_file = (self.directory/'steps.jsonl').open('w', encoding='utf-8')
        self.handles = [self.actions_file, self.steps_file]
        self.complete = False
        write_json(self.directory/'initial_state.json', self.before)
        write_json(self.directory/'metadata.json', dict(seed=self.seed, control_freq=self.freq,
            robot_uid=self.unwrapped.agent.uid, exec_steps=self.exec_steps,
            action_low=vector(self.action_space.low), action_high=vector(self.action_space.high),
            pose_order='x,y,z,qw,qx,qy,qz', pose_frame='world', position_unit='metres',
            action_unit='controller action space, NOT metres/radians; see embedded controller configuration',
            controller_configs=repr(getattr(self.unwrapped.agent.controller, 'configs', None)),
            controller_components={name: repr(getattr(ctrl, 'config', None)) for name, ctrl in
                getattr(self.unwrapped.agent.controller, 'controllers', {}).items()},
            target_pose_frame='controller internal frame; do not assume world frame',
            contact_sampling='after each control step; not peak/integral over physics substeps',
            video_alignment='reset is frame 0; step k maps frame k before to frame k+1 after'))
        return obs, info

    def record_action(self, normalized_chunk):
        if self.pending is not None and self.offset != self.exec_steps:
            raise RuntimeError('Previous action prefix was not fully executed')
        self.decision_index += 1
        self.offset = 0
        self.pending = array(normalized_chunk)
        self.command = self.normalizer.unnormalize(self.pending, 'action')
        payload = dict(seed=self.seed, decision_index=self.decision_index,
                       primitive_step_start=self.step_index, time_s=self.step_index/self.freq,
                       normalized_chunk=self.pending.tolist(), controller_chunk=self.command.tolist())
        self.actions_file.write(json.dumps(payload, allow_nan=False)+'\n')
        self.actions_file.flush()

    def step(self, action):
        if self.pending is None or self.offset >= self.exec_steps:
            raise RuntimeError('Missing policy action callback before primitive step')
        executed = array(action)
        # Prove that the logged prefix is what the unchanged chunk wrapper passes.
        expected = np.clip(self.command[self.offset], self.action_space.low, self.action_space.high)
        if not np.allclose(executed, expected, rtol=0., atol=1e-7):
            raise RuntimeError('Recorded action and executed prefix differ')
        obs, reward, terminated, truncated, info = self.env.step(action)
        with preserve_rng():
            after = self.snapshot(obs, info)
        row = dict(seed=self.seed, primitive_step=self.step_index, decision_index=self.decision_index,
                   chunk_offset=self.offset, time_before_s=self.step_index/self.freq,
                   time_after_s=(self.step_index+1)/self.freq,
                   video_frame_before=self.step_index, video_frame_after=self.step_index+1,
                   normalized_action=self.pending[self.offset].tolist(),
                   controller_action_before_clip=self.command[self.offset].tolist(),
                   executed_action=executed.tolist(), reward=float(reward),
                   terminated=bool(terminated), truncated=bool(truncated), before=self.before, after=after,
                   tcp_actual_delta_world_m=(np.array(after['tcp_pose_world'][:3])-self.before['tcp_pose_world'][:3]).tolist())
        self.steps_file.write(json.dumps(row, allow_nan=False)+'\n')
        self.steps_file.flush()
        self.rows.append(flatten(row))
        self.before = after
        self.offset += 1
        self.step_index += 1
        self.complete = bool(terminated or truncated)
        return obs, reward, terminated, truncated, info

    def finish(self):
        for handle in self.handles:
            handle.close()
        self.handles = []
        if self.directory is not None:
            fields = list(dict.fromkeys(k for row in self.rows for k in row))
            with (self.directory/'steps.csv').open('w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(self.rows)
            write_json(self.directory/'diagnostic_status.json', dict(complete=self.complete,
                primitive_steps=self.step_index, decisions=self.decision_index+1, optional_errors=self.errors))
            self.directory = None

    def close(self):
        try:
            self.finish()
        finally:
            self.env.close()
