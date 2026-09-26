"""GUI-independent, primitive-step human takeover and conservative failure hints."""
from collections import deque

import numpy as np
import torch


class FailureHints:
    """Simulator-only observer. Hints are neither reward labels nor policy inputs."""
    def __init__(self, control_freq):
        if control_freq <= 0:
            raise ValueError('Invalid control frequency')
        self.freq = control_freq
        self.counts = {}
        self.was_grasped = False

    def update(self, before, after, action):
        flags = after['flags']
        grasped = flags.get('is_cubeA_grasped')
        on_target = flags.get('is_cubeA_on_cubeB')
        solved = flags.get('success') is True
        def force(name):
            value = after.get(name+'_force_world_N')
            return None if value is None else float(np.linalg.norm(value))
        left, right = force('cubeA_left'), force('cubeA_right')
        other = [force('cubeB_left'), force('cubeB_right')]
        moved = np.linalg.norm(np.asarray(after['tcp_pose_world'][:3])-before['tcp_pose_world'][:3])
        contact = left is not None and right is not None
        self.was_grasped |= grasped is True
        conditions = {
            'empty_closed': (grasped is False and on_target is False and action[-1] < -.5
                             and after['gripper_width_m'] < .008, .75),
            'one_finger_contact': (contact and grasped is False and on_target is False
                                  and ((left > 2) != (right > 2)), .5),
            'cubeB_interference': (on_target is False and any(x is not None and x > 2 for x in other), .3),
            'contact_stall': (contact and max(left, right) > 2 and np.linalg.norm(action[:3]) > .05
                              and moved < .001 and on_target is False, 1.),
            'grasp_lost': (self.was_grasped and grasped is False and on_target is False, .4),
        }
        hints = []
        for key, (condition, seconds) in conditions.items():
            self.counts[key] = self.counts.get(key, 0)+1 if condition and not solved else 0
            if self.counts[key] >= max(1, int(np.ceil(seconds*self.freq))):
                hints.append(key)
        return hints


class TakeoverController:
    """Execute policy prefixes one primitive at a time; pause never calls step.

    This is the non-ensembled ChunkActionWrapper execution rule, exposed at each
    primitive boundary so an operator can discard its unexecuted prefix on takeover.
    """
    def __init__(self, recorder, sample_chunk, normalizer, exec_steps, snapshot, control_freq):
        self.env, self.sample_chunk, self.normalizer = recorder, sample_chunk, normalizer
        self.exec_steps, self.snapshot = exec_steps, snapshot
        self.detector = FailureHints(control_freq)
        self.queue = deque()
        self.low = np.maximum(recorder.action_space.low, normalizer.stats['action']['min'])
        self.high = np.minimum(recorder.action_space.high, normalizer.stats['action']['max'])
        if self.low.shape != (7,) or np.any(self.low >= self.high) or np.any(self.low[:6] > 0) or np.any(self.high[:6] < 0):
            raise ValueError('Human controls require seven actions and zero arm deltas inside frozen bounds')
        self.mode, self.segment, self.next_segment = 'paused', None, 0
        self.steps, self.events, self.segments = 0, [], []
        self.done, self.gripper = False, float(self.high[-1])
        self.hints = []

    def reset(self, seed):
        if self.steps:
            raise RuntimeError('Create a new controller per episode')
        self.obs, self.info = self.env.reset(seed=seed)
        self.before = self.snapshot(self.obs, self.info, contacts=False)
        return self.obs

    def switch(self, mode):
        if mode not in ('paused', 'policy', 'human'):
            raise ValueError('Unknown control mode')
        if mode != self.mode:
            self.queue.clear()
            self.segment = None
            self.mode = mode

    def advance(self, human_action=None):
        if self.done or self.mode == 'paused':
            return False
        if self.mode == 'policy':
            if human_action is not None:
                raise ValueError('Switch to human before issuing human actions')
            if not self.queue:
                with torch.no_grad():
                    chunk = np.asarray(self.sample_chunk(self.obs))
                if chunk.ndim != 2 or chunk.shape[1] != 7 or len(chunk) < self.exec_steps or not np.isfinite(chunk).all():
                    raise ValueError('Invalid policy action chunk')
                # Match iterative collection's frozen-range clipping.
                physical = self.normalizer.unnormalize(np.clip(chunk, -1., 1.), 'action')
                self.queue.extend(physical[:self.exec_steps])
            requested = np.asarray(self.queue.popleft(), dtype=np.float32)
        else:
            if human_action is None:
                return False  # Waiting for a human is not a zero-action transition.
            requested = np.asarray(human_action, dtype=np.float32)
            if requested.shape != (7,) or not np.isfinite(requested).all():
                raise ValueError('Invalid human action')
            if self.segment is None:
                self.segment = dict(id=self.next_segment, start=self.steps, stop=self.steps)
                self.next_segment += 1
                self.segments.append(self.segment)
        action = np.clip(requested, self.low, self.high).astype(np.float32)
        self.obs, _, terminated, truncated, self.info = self.env.step(action)
        after = self.snapshot(self.obs, self.info)
        self.hints = self.detector.update(self.before, after, action)
        self.events.append(dict(step=self.steps, source=self.mode,
            segment_id=self.segment['id'] if self.mode == 'human' else None,
            requested_action=requested.tolist(), executed_action=action.tolist(),
            clipped=bool(np.any(requested != action)), hints=self.hints, before=self.before, after=after))
        self.steps += 1
        if self.mode == 'human':
            self.segment['stop'] = self.steps
        self.before, self.gripper = after, float(action[-1])
        self.done = bool(terminated or truncated)
        return True

    def human_command(self, axis=None, amount=0., gripper=None):
        action = np.zeros(7, dtype=np.float32)
        action[-1] = self.gripper if gripper is None else gripper
        if axis is not None:
            action[axis] = amount
        return action

    def episode(self):
        ep = self.env.episode()
        if not self.steps:
            raise ValueError('Empty collection episode')
        if not (ep['terminated'][-1] or ep['truncated'][-1]):
            ep['truncated'][-1] = True  # Operator stop: bootstrap at real final observation.
        return ep
