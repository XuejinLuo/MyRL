"""Evaluation failure must not discard completed RL updates or select fake scores."""
import json
import subprocess
from pathlib import Path
import pytest
import torch
from envs.chunk_wrapper import ChunkActionWrapper
from tests.test_online_task import config
from tests.test_online_rl import profile
from tests.test_flow_ppo import TinyPolicy, ToyEnv


def test_worker_retries_fresh_process_and_keeps_same_request(tmp_path, monkeypatch):
    from evaluation import online_worker
    requests = []
    def process(command, **kwargs):
        request = torch.load(command[-2], weights_only=True)
        requests.append(request)
        if len(requests) == 1:
            raise subprocess.CalledProcessError(1, command)
        Path(command[-1]).write_text(json.dumps({'Eval/Success_Rate': .75}))
    monkeypatch.setattr(online_worker.subprocess, 'run', process)
    result = online_worker.isolated_evaluate(profile(), {'w': torch.ones(2)}, {},
        'cps', tmp_path, None, {'epoch': 201})
    assert result['Eval/Success_Rate'] == .75 and len(requests) == 2
    torch.testing.assert_close(requests[0]['weights']['w'], requests[1]['weights']['w'])
    assert requests[0]['metadata'] == requests[1]['metadata'] == {'epoch': 201}
    assert json.loads((tmp_path/'worker_failure.json').read_text())['recovered']


@pytest.mark.parametrize('budget_complete', [False, True])
def test_save_before_failed_eval_then_resume(tmp_path, monkeypatch, budget_complete):
    from workflows import online
    cfg = config(tmp_path)
    cfg.epochs = 2; cfg.algo.critic_warmup_epochs = 0
    cfg.output = str(tmp_path/'failed')
    if budget_complete:
        cfg.online_training = dict(total_env_steps=5, eval_every_env_steps=5)
    torch.save(TinyPolicy().state_dict(), cfg.algo.pretrained_ckpt)
    (tmp_path/'dataset_stats.json').write_text(json.dumps({k: dict(min=[-1.], max=[1.]) for k in ('state', 'action')}))
    monkeypatch.setattr(online, 'build_base', lambda cfg, device: TinyPolicy().to(device))
    original_evaluate = online.evaluate_policy
    calls = []
    def fail_after_baseline(*args, **kwargs):
        calls.append(kwargs['metadata']['epoch'])
        if len(calls) > 1:
            raise subprocess.CalledProcessError(1, 'evaluation_worker')
        return original_evaluate(*args, **kwargs)
    monkeypatch.setattr(online, 'evaluate_policy', fail_after_baseline)
    factory = lambda: ChunkActionWrapper(ToyEnv(), 3, 2)
    with pytest.raises(subprocess.CalledProcessError):
        online.run(cfg, factory)
    path = Path(cfg.output)/'checkpoints/last.pth'
    saved = torch.load(path, weights_only=True)
    assert saved['epoch'] == 1 and saved['total_env_steps'] == 5 and saved['evaluation_pending']
    assert saved['actor_optimizer']['state'] and saved['critic_optimizer']['state']
    assert 'Eval/Success_Rate' not in saved['metrics']  # no invented evaluation
    assert saved['selection_best']['epoch'] == 0
    monkeypatch.setattr(online, 'evaluate_policy', original_evaluate)
    cfg.resume = str(path); cfg.output = str(tmp_path/'recovered')
    result = online.run(cfg, factory)
    restored = torch.load(Path(cfg.output)/'checkpoints/epoch_0001.pth', weights_only=True)
    for section in ('model_state_dict', 'critic'):
        for key, value in saved[section].items():
            torch.testing.assert_close(restored[section][key], value, rtol=0, atol=0)
    assert not restored['evaluation_pending']
    final = torch.load(Path(cfg.output)/'checkpoints/last.pth', weights_only=True)
    assert result['total_env_steps'] == (5 if budget_complete else 10)
    assert final['epoch'] == (1 if budget_complete else 2)
    if budget_complete:
        assert json.loads((Path(cfg.output)/'goal_status.json').read_text())['final']
        # The pending final evaluation is recovered without ANY optimizer step.
        for key, fields in saved['actor_optimizer']['state'].items():
            for field, value in fields.items():
                torch.testing.assert_close(final['actor_optimizer']['state'][key][field], value)


def test_disable_worker_retry(tmp_path, monkeypatch):
    from evaluation import online_worker
    cfg = profile(); cfg.online_training.eval_worker_retries = 0
    calls = []
    def failed(*args, **kwargs):
        calls.append(1)
        raise subprocess.CalledProcessError(1, 'worker')
    monkeypatch.setattr(online_worker.subprocess, 'run', failed)
    with pytest.raises(subprocess.CalledProcessError):
        online_worker.isolated_evaluate(cfg, {}, {}, 'cps', tmp_path, None, {})
    assert len(calls) == 1
