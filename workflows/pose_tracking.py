"""Track planner TCP waypoints through the live EE controller, without joint teleportation."""
from collections import deque

import numpy as np


def quaternion_product(a, b):
    w, x, y, z = a
    v, i, j, k = b
    return np.array([w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
                     w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v])


def pose_error(target, current):
    """World translation and shortest root/world-aligned rotation vector, wxyz poses."""
    target, current = np.asarray(target, float), np.asarray(current, float)
    if target.shape != (7,) or current.shape != (7,) or not np.isfinite([target, current]).all():
        raise ValueError('Expected finite xyz,wxyz poses')
    a, b = target[3:], current[3:]
    if min(np.linalg.norm(a), np.linalg.norm(b)) < 1e-8:
        raise ValueError('Invalid quaternion')
    a, b = a/np.linalg.norm(a), b/np.linalg.norm(b)
    relative = quaternion_product(a, b*np.array([1, -1, -1, -1]))
    if relative[0] < 0:
        relative = -relative
    sine = np.linalg.norm(relative[1:])
    rotation = relative[1:] * (2*np.arctan2(sine, relative[0])/sine if sine > 1e-10 else 2.)
    return np.r_[target[:3]-current[:3], rotation]


def bounded_controller_action(predict, current, target, low, high, position_step=.005, rotation_step=.03):
    """Invert the installed controller's pure target-pose mapping numerically.

    Probing compute_target_pose/_preprocess_action does not step physics or set
    targets. This accounts for installed ManiSkill rotation sign/representation
    instead of assuming that its rotation action is an axis-angle divided by .1.
    """
    low, high = np.asarray(low), np.asarray(high)
    desired = pose_error(target, current)
    for sl, cap in ((slice(0, 3), position_step), (slice(3, 6), rotation_step)):
        desired[sl] *= min(1., cap/max(np.linalg.norm(desired[sl]), 1e-12))
    zero = np.zeros(6)
    offset = pose_error(predict(zero), current)
    if np.linalg.norm(offset) > 1e-5:
        raise ValueError('Zero controller action does not hold current TCP; unsupported semantics')
    epsilon = 1e-3
    jacobian = np.column_stack([(pose_error(predict(axis*epsilon), current)-offset)/epsilon
                                for axis in np.eye(6)])
    if not np.isfinite(jacobian).all() or np.linalg.cond(jacobian) > 1e5:
        raise ValueError('Cannot invert controller action-to-pose mapping')
    action = np.zeros(6)
    for _ in range(3):
        residual = desired-pose_error(predict(action), current)
        action += np.linalg.solve(jacobian, residual)
        action = np.clip(action, low, high)
        action[3:] /= max(1., np.linalg.norm(action[3:]))
    return action.astype(np.float32)


class WaypointTracker:
    """Closed-loop finite path execution; every call returns at most one action."""
    def __init__(self, waypoints, max_steps=150, stall_steps=20):
        self.waypoints = deque(np.asarray(p, float).copy() for p in waypoints)
        if not self.waypoints or max_steps < 1 or stall_steps < 1:
            raise ValueError('Empty path or invalid tracking budget')
        for p in self.waypoints:
            pose_error(p, p)
        self.max_steps, self.stall_steps = max_steps, stall_steps
        self.steps, self.stalled, self.previous = 0, 0, None
        self.status = 'tracking'

    def action(self, current, convert):
        while self.waypoints:
            error = pose_error(self.waypoints[0], current)
            if np.linalg.norm(error[:3]) < .002 and np.linalg.norm(error[3:]) < .015:
                self.waypoints.popleft()
            else:
                break
        if not self.waypoints:
            self.status = 'reached'
            return None
        if self.steps >= self.max_steps:
            self.status = 'step_budget'
            return None
        if self.previous is not None:
            delta = pose_error(current, self.previous)
            self.stalled = self.stalled+1 if np.linalg.norm(delta[:3]) < .0001 and np.linalg.norm(delta[3:]) < .001 else 0
            if self.stalled >= self.stall_steps:
                self.status = 'stalled'
                return None
        self.previous = np.array(current, copy=True)
        self.steps += 1
        return convert(self.waypoints[0])
