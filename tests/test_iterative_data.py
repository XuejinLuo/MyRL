from pathlib import Path
import tempfile
import unittest
import numpy as np
from data.episodes import (validate_episode, transition, save_episode,
    load_episode, load_sources, digest, write_json, SCHEMA)


def episode():
    return dict(pc=np.arange(24, dtype=np.float32).reshape(4, 2, 3),
        state=np.arange(8, dtype=np.float32).reshape(4, 2),
        action=np.array([[.1], [.2], [.3]], dtype=np.float32),
        reward=np.array([0., 1., 0.], dtype=np.float32),
        success=np.array([False, True, False]),
        terminated=np.zeros(3, dtype=bool), truncated=np.array([False, False, True]))


class DataTests(unittest.TestCase):
    def test_prefix_discount_and_observation(self):
        ep = episode()
        row = transition(ep, 0, 4, 2, .9)
        self.assertAlmostEqual(float(row['reward']), .9)
        self.assertAlmostEqual(float(row['discount']), .81)
        np.testing.assert_array_equal(row['next_state'], ep['state'][2])

    def test_timeout_bootstraps_final_observation(self):
        ep = episode()
        row = transition(ep, 2, 4, 2, .9)
        self.assertEqual(row['done'], 0)
        self.assertAlmostEqual(float(row['discount']), .9)
        np.testing.assert_array_equal(row['next_pc'], ep['pc'][3])
        np.testing.assert_array_equal(row['action_chunk'], np.full((4, 1), .3, dtype=np.float32))

    def test_terminal_does_not_bootstrap(self):
        ep = episode()
        ep['terminated'][-1] = True
        self.assertEqual(transition(ep, 2, 4, 2, .9)['done'], 1)

    def test_reject_missing_final_observation_and_cross_episode(self):
        ep = episode()
        ep['pc'] = ep['pc'][:-1]
        with self.assertRaises(ValueError):
            validate_episode(ep)
        ep = episode()
        ep['terminated'][0] = True
        with self.assertRaises(ValueError):
            validate_episode(ep)

    def test_roundtrip_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'ep.npz'
            save_episode(path, episode())
            restored = load_episode(path)
            for k, v in episode().items():
                np.testing.assert_array_equal(restored[k], v)
            with self.assertRaises(FileExistsError):
                save_episode(path, episode())

    def test_manifest_rejects_duplicates_and_changed_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            save_episode(p/'ep.npz', episode())
            item = dict(path='ep.npz', sha256=digest(p/'ep.npz'))
            spec = dict(schema=SCHEMA, reward_mode='success', sources=[dict(episodes=[item])])
            write_json(p/'manifest.json', spec)
            self.assertEqual(len(load_sources(p/'manifest.json')[0]), 1)
            spec['sources'].append(dict(episodes=[item]))
            write_json(p/'manifest.json', spec)
            with self.assertRaises(ValueError):
                load_sources(p/'manifest.json')
            spec['sources'].pop()
            item['sha256'] = 'incorrect'
            write_json(p/'manifest.json', spec)
            with self.assertRaises(ValueError):
                load_sources(p/'manifest.json')

    def test_failure_remains_failure(self):
        ep = episode()
        ep['reward'][:] = 0
        ep['success'][:] = False
        validate_episode(ep)
        self.assertFalse(ep['success'].any())
        self.assertEqual(transition(ep, 2, 4, 2, .9)['reward'], 0)


if __name__ == '__main__':
    unittest.main()
