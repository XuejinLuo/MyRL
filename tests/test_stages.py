"""Integration tests exercise stage handoff with real optimizers and a tiny CPU environment."""
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from data.demonstrations import load_demonstrations
from data.episodes import load_sources
from envs.chunk_wrapper import ChunkActionWrapper
from tests.test_flow_ppo import TinyPolicy, ToyEnv


def config(stage, tmp_path):
    with initialize_config_dir(version_base=None, config_dir=str(Path(__file__).resolve().parents[1]/'configs')):
        cfg = compose(config_name='train_'+stage)
    cfg.device = 'cpu'
    cfg.paths.root = str(tmp_path/'run')
    cfg.num_workers = 0
    cfg.env.num_points = 4
    cfg.env.use_color = False
    cfg.env.max_episode_steps = 3
    cfg.model.in_channels = 3
    cfg.model.state_dim = 1
    cfg.model.action_dim = 1
    cfg.model.cond_dim = 2
    cfg.model.chunk_size = 3
    cfg.model.num_inference_steps = 3
    cfg.eval.every = 1
    cfg.eval.seeds = [2000, 2001]
    cfg.video.every = 0
    cfg.video.episodes = 0
    cfg.comparison.episodes = 2
    for key in ('offline', 'iterative', 'online'):
        cfg.stages[key].epochs = 2
        cfg.stages[key].batch_size = 2
        cfg.stages[key].save_every = 1
    cfg.stages.offline.critic_warmup_epochs = 0
    cfg.stages.iterative.critic_warmup_epochs = 0
    cfg.stages.iterative.rounds = 1
    cfg.stages.iterative.episodes_per_round = 2
    if stage == 'online':
        cfg.algo.steps_per_epoch = 4
        cfg.algo.critic_warmup_epochs = 0
        cfg.quiet = True
    return cfg


class TrainableTinyPolicy(TinyPolicy):
    def compute_loss(self, obs, actions, state):
        cond = self._get_condition(obs, state)
        prediction = self.backbone(torch.zeros_like(actions), torch.zeros(len(state)), cond)
        return (prediction-actions).square().mean()


class CriticFeatures(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layer = torch.nn.Linear(cfg.model.state_dim, cfg.model.cond_dim)
    def forward(self, obs):
        return self.layer(obs['state'])


def primitive_episode():
    return dict(pc=np.zeros((4, 4, 3), dtype=np.float32),
        state=np.arange(4, dtype=np.float32)[:, None]/5,
        action=np.array([[-1.], [0.], [1.]], dtype=np.float32),
        reward=np.array([0., 0., 1.], dtype=np.float32),
        success=np.array([False, False, True]),
        terminated=np.zeros(3, dtype=bool), truncated=np.array([False, False, True]))


def test_three_stages_handoff_metrics_checkpoints_and_round_resume(tmp_path, monkeypatch):
    from workflows import offline, iterative, online, offline_round
    from evaluation import compare
    for module in (offline, iterative, online, compare):
        monkeypatch.setattr(module, 'build_base', lambda cfg, device: TrainableTinyPolicy().to(device))
    monkeypatch.setattr(offline_round, 'CriticFeatureExtractor', CriticFeatures)
    monkeypatch.setattr(offline, 'load_demonstrations', lambda cfg: [primitive_episode()])

    def make_env(cfg, primitive_wrapper=None, video=None):
        env = ToyEnv()
        if primitive_wrapper:
            env = primitive_wrapper(env)
        return ChunkActionWrapper(env, cfg.model.chunk_size, cfg.env.exec_steps)
    monkeypatch.setitem(sys.modules, 'envs.factory', SimpleNamespace(make_env=make_env))
    configs = {stage: config(stage, tmp_path) for stage in ('offline', 'iterative', 'online')}
    offline.run(configs['offline'])
    iterative.run(configs['iterative'])
    online.run(configs['online'])
    for stage, cfg in configs.items():
        output = Path(cfg.output)
        for name in ('config.yaml', 'metrics.jsonl', 'metrics.csv', 'selection.json',
                     'checkpoints/best.pth', 'checkpoints/last.pth', 'checkpoints/dataset_stats.json'):
            assert (output/name).exists(), (stage, name)
        rows = [json.loads(line) for line in (output/'metrics.jsonl').read_text().splitlines()]
        assert len(list(csv.DictReader((output/'metrics.csv').open()))) == len(rows)
        assert len({(row['round'], row['epoch']) for row in rows}) == len(rows)
        for row in rows:
            assert row['schema'] == 'myrl_metrics_v2' and row['stage'] == stage
            if row['evaluation']:
                summary = json.loads((Path(row['evaluation'])/'summary.json').read_text())
                assert summary['seeds'] == [2000, 2001]
                assert summary['metadata']['checkpoint'] == row['checkpoint']
                cp = torch.load(row['checkpoint'], weights_only=True)
                assert cp['metrics']['Eval/Success_Rate'] == row['Eval/Success_Rate']
        selection = json.loads((output/'selection.json').read_text())
        cp = torch.load(selection['checkpoint'], weights_only=True)
        assert cp['metrics'] == selection['metrics']
    episodes, _ = load_sources(Path(configs['iterative'].output)/'round_000'/'manifest.json')
    assert len(episodes) == 3
    cfg = configs['iterative']
    before = (Path(cfg.output)/'metrics.jsonl').read_bytes()
    cfg.stages.iterative.resume = True
    iterative.run(cfg)
    assert (Path(cfg.output)/'metrics.jsonl').read_bytes() == before
    rows = compare.run(configs['online'])
    assert len(rows) == 6 and {r['sampler'] for r in rows} == {'cps', 'ode'}


def write_h5(path, missing_success=False):
    with h5py.File(path, 'w') as f:
        g = f.create_group('traj_0')
        xyzw = np.ones((5, 4, 4), dtype=np.float32)
        xyzw[..., :3] = .1
        xyzw[:, 0, 3] = 0  # Invalid points must never enter either preprocessing path.
        g['obs/pointcloud/xyzw'] = xyzw
        g['obs/agent/qpos'] = np.arange(5, dtype=np.float32)[:, None]
        g['obs/extra/tcp_pose'] = np.zeros((5, 1), dtype=np.float32)
        g['actions'] = np.ones((4, 1), dtype=np.float32)
        if not missing_success:
            g['success'] = np.array([0, 1, 0, 0], dtype=bool)
        g['terminated'] = np.array([0, 1, 0, 0], dtype=bool)
        g['truncated'] = np.zeros(4, dtype=bool)


def test_h5_ingestion_uses_real_terminal_and_t_plus_one_observation(tmp_path):
    cfg = config('offline', tmp_path)
    path = tmp_path/'demo.h5'
    write_h5(path)
    cfg.dataset.data_path = str(path)
    cfg.model.state_dim = 2
    ep, = load_demonstrations(cfg)
    assert ep['action'].shape == (2, 1) and ep['state'].shape == (3, 2)
    assert ep['terminated'][-1] and not ep['truncated'][-1]
    assert ep['reward'].tolist() == [0., 1.]
    np.testing.assert_allclose(ep['pc'], .1)


def test_h5_missing_success_is_rejected(tmp_path):
    cfg = config('offline', tmp_path)
    path = tmp_path/'demo.h5'
    write_h5(path, missing_success=True)
    cfg.dataset.data_path = str(path)
    with pytest.raises(ValueError, match='missing success'):
        load_demonstrations(cfg)


def test_common_validation_rejects_inconsistent_video_and_test_seeds(tmp_path):
    from utils.config import validate_common
    cfg = config('offline', tmp_path)
    cfg.eval.every, cfg.video.every = 5, 7
    with pytest.raises(ValueError, match='multiple'):
        validate_common(cfg)
    cfg.video.every = 0
    cfg.comparison.seed_start = 2000
    with pytest.raises(ValueError, match='disjoint'):
        validate_common(cfg)
