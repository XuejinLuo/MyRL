"""Regressions for comparable metrics, RNG isolation and pipeline configuration."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import gymnasium as gym
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from evaluation.runner import evaluate_policy, seed_all
from utils.normalizer import MinMaxNormalizer
from envs.chunk_wrapper import ChunkActionWrapper


class Policy(torch.nn.Module):
    eval_mode = 'cps'
    def sample(self, features, num_steps):
        return torch.randn(1, 2, 1)


class Env(gym.Env):
    render_mode = 'rgb_array'
    control_freq = 20
    def __init__(self):
        self.action_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)
    def reset(self, seed=None, options=None):
        self.t = 0
        return np.zeros(1), {}
    def step(self, action):
        self.t += 1
        return np.zeros(1), float(action[0]), False, self.t == 4, {'success': self.t == 1}
    def render(self):
        np.random.rand()
        torch.rand(1)
        return np.zeros((16, 16, 3), dtype=np.uint8)


def test_metrics_record_transient_success_and_restore_rng(tmp_path):
    norm = MinMaxNormalizer()
    norm.fit({'action': np.array([[-1.], [1.]])})
    actor = Policy().train()
    seed_all(72)
    before = torch.get_rng_state().clone()
    result = evaluate_policy(lambda: ChunkActionWrapper(Env(), 2, 2), actor,
        lambda obs: obs, norm, [100, 101], 2, output_dir=tmp_path)
    assert actor.training and torch.equal(before, torch.get_rng_state())
    assert result['Eval/Success_Rate'] == 1
    assert result['Eval/Final_Success_Rate'] == 0
    assert result['Eval/Success_Count'] == 2
    rows = list(csv.DictReader((tmp_path/'episodes.csv').open()))
    assert [r['seed'] for r in rows] == ['100', '101']
    assert all(r['primitive_steps'] == '4' for r in rows)
    assert json.loads((tmp_path/'summary.json').read_text())['seeds'] == [100, 101]


def test_video_does_not_change_evaluation(tmp_path, monkeypatch):
    from evaluation.video import EvalVideo
    import imageio.v2 as imageio
    frames = []
    class Writer:
        def append_data(self, frame): frames.append(frame)
        def close(self): pass
    monkeypatch.setattr(imageio, 'get_writer', lambda *a, **kw: Writer())
    norm = MinMaxNormalizer()
    norm.fit({'action': np.array([[-1.], [1.]])})
    def run(video):
        def factory():
            env = Env()
            if video: env = EvalVideo(env, tmp_path, episodes=1)
            return ChunkActionWrapper(env, 2, 2)
        return evaluate_policy(factory, Policy(), lambda x: x, norm, [123, 124], 2)
    assert run(False) == run(True)
    assert len(frames) == 5


def test_all_tasks_share_protocol_and_source_paths():
    from utils.config import validate_common
    with initialize_config_dir(version_base=None, config_dir=str(Path(__file__).resolve().parents[1]/'configs')):
        for task, env in [('stackcube', 'StackCube-v1'), ('pullcubetool', 'PullCubeTool-v1'), ('pickcube', 'PickCube-v1')]:
            configs = [compose(config_name='train_'+stage, overrides=['task@_global_='+task])
                       for stage in ('offline', 'iterative', 'online')]
            for cfg in configs:
                validate_common(cfg)
                assert cfg.env.env_id == env
                assert env in cfg.output and env in cfg.dataset.data_path
                assert cfg.dataset.max_episodes == 1000 and cfg.video.episodes == 5
            assert all(c.eval == configs[0].eval for c in configs)
            assert configs[1].initial_ckpt == configs[0].output+'/checkpoints/best.pth'
            assert configs[1].manifest == configs[0].output+'/data/manifest.json'
            assert configs[2].algo.pretrained_ckpt == configs[1].output+'/checkpoints/best.pth'
            assert configs[0].algo.use_bc_only and not configs[1].algo.use_bc_only


def test_evaluate_base_restores_training_flags_on_failure(tmp_path, monkeypatch):
    import sys
    from utils.experiment import evaluate_base
    from algos.diffusion_utils.flow_matching import OTFlowMatching
    class Base(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(1, 1)
            self.algo_type = 'flow'
            self.scheduler = OTFlowMatching()
    base = Base().train()
    def fail(*args, **kwargs): raise RuntimeError('simulator unavailable')
    monkeypatch.setitem(sys.modules, 'envs.factory', SimpleNamespace(make_env=fail))
    cfg = OmegaConf.create(dict(model=dict(num_inference_steps=2), env=dict(workspace_bounds=[[-1,-1,-1],[1,1,1]]), epochs=1,
        eval=dict(sampler='cps', seeds=[1]), video=dict(every=0)))
    import pytest
    with pytest.raises(RuntimeError, match='simulator unavailable'):
        evaluate_base(cfg, base, None, tmp_path, 1)
    assert base.training and base.encoder.training
    assert all(p.requires_grad for p in base.parameters())
