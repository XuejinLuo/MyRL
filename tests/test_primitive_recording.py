"""Wrapper integration on a fake primitive environment; no simulator required."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
import numpy as np


class Box:
    def __init__(self, low, high, dtype=np.float32):
        self.low, self.high, self.dtype = np.array(low), np.array(high), dtype


class Wrapper:
    def __init__(self, env):
        self.env, self.action_space = env, env.action_space


class FakeEnv:
    action_space = Box([-1.], [1.])
    def reset(self, **kwargs):
        self.t = 0
        return self.obs(), {}
    def obs(self):
        return dict(point_cloud=np.full((2, 3), self.t, dtype=np.float32),
                    state=np.array([self.t], dtype=np.float32))
    def step(self, action):
        self.t += 1
        return self.obs(), 17., False, self.t == 3, {'success': self.t == 2}


def module(path):
    spec = importlib.util.spec_from_file_location('isolated_test_module', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class RecorderTests(unittest.TestCase):
    def test_primitive_alignment_clipping_and_early_end(self):
        root = Path(__file__).resolve().parents[1]
        fake = types.SimpleNamespace(Wrapper=Wrapper, Env=FakeEnv, spaces=types.SimpleNamespace(Box=Box))
        with patch.dict(sys.modules, gymnasium=fake):
            recorder_cls = module(root/'workflows/collection.py').PrimitiveRecorder
            chunk_cls = module(root/'envs/chunk_wrapper.py').ChunkActionWrapper
        recorder = recorder_cls(FakeEnv())
        env = chunk_cls(recorder, chunk_size=4, exec_steps=2)
        env.reset()
        env.step(np.full((4, 1), 5.))
        _, _, _, truncated, info = env.step(np.full((4, 1), -5.))
        ep = recorder.episode()
        self.assertTrue(truncated)
        self.assertEqual(info['actual_steps'], 1)
        np.testing.assert_array_equal(ep['action'][:, 0], [1., 1., -1.])
        np.testing.assert_array_equal(ep['state'][:, 0], [0., 1., 2., 3.])
        np.testing.assert_array_equal(ep['reward'], [0., 1., 0.])
        np.testing.assert_array_equal(ep['truncated'], [False, False, True])
        env.reset()
        self.assertEqual(len(recorder.rows['action']), 0)


if __name__ == '__main__':
    unittest.main()
