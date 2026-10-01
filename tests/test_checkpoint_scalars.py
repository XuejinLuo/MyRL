"""Regression for PR #18's NumPy reward metric breaking weights-only resume."""
import json
import pickle
import zipfile
from collections import OrderedDict
from pathlib import Path
import numpy as np
import pytest
import torch
from models.checkpoint import load_checkpoint, checkpoint_primitives
from workflows.online import atomic_checkpoint


def test_old_numpy_checkpoint_load_and_plain_resave(tmp_path):
    weights = OrderedDict(weight=torch.tensor([1., 2.]))
    weights._metadata = OrderedDict([('', {'version': 1})])
    state = dict(model_state_dict=weights, epoch=12, total_env_steps=1000001,
        metrics={'Train/Reward': np.float64(12.25), 'count': np.int64(2), 'flag': np.bool_(True)},
        selection_best={'metrics': {'Eval/Success_Rate': np.float32(.75)}},
        actor_optimizer={'state': {0: {'step': torch.tensor(7.), 'exp_avg': torch.tensor([.1])}},
                         'param_groups': [{'lr': 1e-7, 'params': [0]}]})
    source = tmp_path/'old.pth'; torch.save(state, source)
    with pytest.raises(pickle.UnpicklingError, match='Unsupported global'):
        torch.load(source, weights_only=True)
    previous = torch.serialization.get_safe_globals()
    loaded = load_checkpoint(source)
    assert torch.serialization.get_safe_globals() == previous
    assert type(loaded['metrics']['Train/Reward']) is float
    assert type(loaded['metrics']['count']) is int and type(loaded['metrics']['flag']) is bool
    assert loaded['total_env_steps'] == 1000001
    torch.testing.assert_close(loaded['model_state_dict']['weight'], weights['weight'])
    assert loaded['model_state_dict']._metadata == weights._metadata
    torch.testing.assert_close(loaded['actor_optimizer']['state'][0]['exp_avg'], torch.tensor([.1]))
    destination = tmp_path/'new.pth'
    atomic_checkpoint(loaded, str(destination))
    plain = torch.load(destination, weights_only=True)
    assert plain['selection_best']['metrics']['Eval/Success_Rate'] == .75
    assert plain['actor_optimizer']['param_groups'] == state['actor_optimizer']['param_groups']
    # Saving a contaminated new state is also guarded, independently of the reader.
    atomic_checkpoint(state, str(destination))
    assert torch.load(destination, weights_only=True)['metrics']['Train/Reward'] == 12.25
    assert type(state['metrics']['Train/Reward']) is np.float64  # no in-place mutation


class UnrelatedObject:
    pass


def test_unrelated_pickle_globals_still_rejected(tmp_path):
    path = tmp_path/'other.pth'
    torch.save({'metrics': np.float64(1), 'other': UnrelatedObject()}, path)
    previous = torch.serialization.get_safe_globals()
    with pytest.raises(pickle.UnpicklingError):
        load_checkpoint(path)
    assert torch.serialization.get_safe_globals() == previous


@pytest.mark.parametrize('module', [b'numpy.core.multiarray', b'numpy._core.multiarray'])
def test_numpy_scalar_module_spellings(tmp_path, module):
    source, destination = tmp_path/'source.pth', tmp_path/'alias.pth'
    torch.save({'reward': np.float64(3.5)}, source, pickle_protocol=2)
    # Protocol 2 GLOBAL contains a newline-delimited module, not a framed string.
    with zipfile.ZipFile(source) as reader, zipfile.ZipFile(destination, 'w') as writer:
        for member in reader.infolist():
            data = reader.read(member.filename)
            if member.filename.endswith('/data.pkl'):
                data = data.replace(b'cnumpy._core.multiarray\nscalar\n', b'c' + module + b'\nscalar\n')
                data = data.replace(b'cnumpy.core.multiarray\nscalar\n', b'c' + module + b'\nscalar\n')
            writer.writestr(member, data)
    assert load_checkpoint(destination) == {'reward': 3.5}


def test_shaped_reward_produces_native_metric():
    from envs.stackcube_training import shaped_reward
    p = dict(success_reward=1., potential_scale=1., gamma=1.)
    reward = shaped_reward(False, np.float32(.1), np.float32(.2), False, p)
    assert type(reward) is float
    tensor = torch.ones(1)
    normalized = checkpoint_primitives({'x': (np.float64(1), [np.int32(2), tensor])})
    assert normalized['x'][1][1] is tensor


def test_resume_existing_numpy_checkpoint_without_retraining(tmp_path, monkeypatch):
    from tests.test_online_task import config
    from tests.test_flow_ppo import TinyPolicy, ToyEnv
    from envs.chunk_wrapper import ChunkActionWrapper
    from workflows import online
    cfg = config(tmp_path); cfg.output = str(tmp_path/'first')
    torch.save(TinyPolicy().state_dict(), cfg.algo.pretrained_ckpt)
    (tmp_path/'dataset_stats.json').write_text(json.dumps({k: dict(min=[-1.], max=[1.]) for k in ('state', 'action')}))
    monkeypatch.setattr(online, 'build_base', lambda cfg, device: TinyPolicy().to(device))
    factory = lambda: ChunkActionWrapper(ToyEnv(), 3, 2)
    online.run(cfg, factory)
    path = Path(cfg.output)/'checkpoints/last.pth'
    old = torch.load(path, weights_only=True)
    old['metrics']['Train/Reward'] = np.float64(4.25)
    torch.save(old, path)  # reproduce exactly the old writer's metadata defect
    cfg.resume = str(path); cfg.epochs = 3; cfg.output = str(tmp_path/'resumed')
    result = online.run(cfg, factory)
    resumed = torch.load(Path(cfg.output)/'checkpoints/epoch_0002.pth', weights_only=True)
    final = torch.load(Path(cfg.output)/'checkpoints/last.pth', weights_only=True)
    assert result['total_env_steps'] == old['total_env_steps'] + 5
    for name in ('model_state_dict', 'critic'):
        for key in old[name]:
            torch.testing.assert_close(resumed[name][key], old[name][key], rtol=0, atol=0)
    for name in ('actor_optimizer', 'critic_optimizer'):
        for key, value in old[name]['state'].items():
            for field, tensor in value.items():
                torch.testing.assert_close(resumed[name]['state'][key][field], tensor, rtol=0, atol=0)
    assert final['epoch'] == 3 and final['selection_best']['epoch'] == old['selection_best']['epoch']
