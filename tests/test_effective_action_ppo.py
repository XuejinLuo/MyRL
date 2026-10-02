import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
from hydra import compose, initialize_config_dir
from algos.pg import FlowPPO, sum_event_logprob
from models.online_policy import FlowPPOPolicy
from tests.test_flow_ppo import TinyPolicy, actor, ToyEnv
from tests.test_online_task import config
from utils.normalizer import MinMaxNormalizer
from utils.policy_probe import PolicyUpdateProbe
from tools.diagnostics.compare_online_backends import audit_episodes, summarize, MODES, SingleGPUView


def test_final_prefix_is_exact_conditional_marginal_and_keeps_intermediate_tail():
    pi = actor()
    ppo = FlowPPO(pi, nn.Linear(2, 1), final_prefix_steps=1)
    features = torch.randn(5, 2)
    _, chain, old = pi.collect(features)
    for step in range(3):
        mean, std = pi.sampler.transition(pi.policy.backbone, features, chain[:, step], step)
        marginal = torch.distributions.Independent(
            torch.distributions.Normal(mean[:, :1], std[:, :1]), 2)
        event = ppo.event_logprob(old[:, step], step)
        if step == 2:
            torch.testing.assert_close(event, marginal.log_prob(chain[:, step+1, :1]))
        else:
            torch.testing.assert_close(event, sum_event_logprob(old[:, step]))
        changed = old[:, step].clone(); changed[:, 1:] += 100
        altered = ppo.event_logprob(changed, step)
        assert torch.equal(event, altered) == (step == 2)


def test_coupled_intermediate_tail_still_changes_executed_action():
    class Coupled(nn.Module):
        def __init__(self):
            super().__init__(); self.w = nn.Parameter(torch.tensor(.2))
        def forward(self, x, t, features):
            return self.w * x.mean(dim=1, keepdim=True).expand_as(x)
    base = TinyPolicy(); base.backbone = Coupled()
    pi = FlowPPOPolicy(base, num_steps=3)
    x = torch.zeros(2, 3, 1); changed = x.clone(); changed[:, 2] = 10
    features = torch.zeros(2, 2)
    m0, _ = pi.sampler.transition(base.backbone, features, x, 1)
    m1, _ = pi.sampler.transition(base.backbone, features, changed, 1)
    assert not torch.equal(m0[:, :1], m1[:, :1])


def test_single_step_prefix_updates_only_executed_conditional_distribution():
    class Coordinates(nn.Module):
        def __init__(self):
            super().__init__(); self.mean = nn.Parameter(torch.zeros(3, 1))
        def forward(self, x, t, features):
            return self.mean.expand_as(x)
    torch.manual_seed(4)
    base = TinyPolicy(); base.backbone = Coordinates()
    pi = FlowPPOPolicy(base, num_steps=1)
    features = torch.zeros(32, 2); _, chain, old = pi.collect(features)
    ppo = FlowPPO(pi, nn.Linear(2, 1), final_prefix_steps=1, target_kl=None)
    batch = dict(features=features, chains=chain, logprobs=old, values=torch.zeros(32))
    with torch.no_grad():
        mean, _ = pi.sampler.transition(base.backbone, features, chain[:, 0], 0)
    # Conditional advantage avoids finite-sample covariance with initial noise.
    adv = chain[:, -1, 0, 0] - mean[:, 0, 0]
    ppo.update(batch, adv, torch.zeros(32), batch_size=32, epochs=1)
    assert base.backbone.mean[0].item() > 0
    torch.testing.assert_close(base.backbone.mean[1:], torch.zeros(2, 1), rtol=0, atol=0)


@pytest.mark.parametrize('scope', ['full', 'prefix', 'final_prefix'])
def test_probe_preserves_rng_gradients_and_optimizer_and_reports_real_change(scope):
    torch.manual_seed(33)
    pi = actor(); critic = nn.Linear(2, 1)
    ppo = FlowPPO(pi, critic, actor_lr=1e-4, target_kl=None,
        prefix_steps=1 if scope == 'prefix' else None,
        final_prefix_steps=1 if scope == 'final_prefix' else None)
    features = torch.randn(8, 2)
    _, chain, old = pi.collect(features)
    batch = dict(features=features, chains=chain, logprobs=old, values=torch.zeros(8))
    adv = torch.arange(8.).float(); ret = torch.ones(8)
    pi.policy.backbone.w.grad = torch.tensor(7.)
    rng = torch.get_rng_state().clone()
    before = copy.deepcopy(pi.state_dict())
    probe = PolicyUpdateProbe(ppo, batch, adv, exec_steps=1, samples=4)
    unchanged = probe.finish(MinMaxNormalizer())
    assert unchanged['Probe/Executed_Normalized_RMS'] == 0
    assert unchanged['Probe/Conditional_KL_Step_2'] == 0
    assert torch.equal(rng, torch.get_rng_state())
    assert pi.policy.backbone.w.grad.item() == 7
    assert not ppo.actor_optimizer.state
    for k, value in before.items():
        torch.testing.assert_close(value, pi.state_dict()[k])
    ppo.update(batch, adv, ret, batch_size=4, epochs=1)
    result = probe.finish(MinMaxNormalizer())
    assert result['Probe/Executed_Normalized_RMS'] > 0
    assert result['Probe/Conditional_KL_Step_2'] > 0
    assert result['Probe/Gradient_Norm_Step_2'] > 0


def test_probe_does_not_change_training_update():
    torch.manual_seed(72)
    a = actor(); b = copy.deepcopy(a)
    va = nn.Linear(2, 1); vb = copy.deepcopy(va)
    pa = FlowPPO(a, va, final_prefix_steps=1, target_kl=None)
    pb = FlowPPO(b, vb, final_prefix_steps=1, target_kl=None)
    f = torch.randn(8, 2); _, chain, logp = a.collect(f)
    batch = dict(features=f, chains=chain, logprobs=logp, values=torch.zeros(8))
    adv, returns = torch.randn(8), torch.randn(8)
    state = torch.get_rng_state()
    pa.update(batch, adv, returns, batch_size=4, epochs=2)
    end = torch.get_rng_state()
    torch.set_rng_state(state)
    probe = PolicyUpdateProbe(pb, batch, adv, 1, 4)
    pb.update(batch, adv, returns, batch_size=4, epochs=2)
    probe.finish(MinMaxNormalizer())
    assert torch.equal(end, torch.get_rng_state())
    for x, y in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(x, y, rtol=0, atol=0)


def test_new_profile_changes_only_scope_and_keeps_old_default():
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / 'configs'), version_base=None):
        old = compose(config_name='train_online_rl')
        new = compose(config_name='train_online_rl_action')
    from workflows.online import validate
    validate(new)
    assert old.algo.ratio_scope == 'full' and new.algo.ratio_scope == 'final_prefix'
    assert old.env == new.env and old.model == new.model
    assert old.online_task == new.online_task
    assert old.algo.actor_lr == new.algo.actor_lr
    assert new.resume is None and 'online_rl03/checkpoints/best.pth' in new.algo.pretrained_ckpt


def test_new_scope_trains_resumes_and_rejects_silent_scope_change(tmp_path, monkeypatch):
    from workflows import online
    from envs.chunk_wrapper import ChunkActionWrapper
    cfg = config(tmp_path); cfg.output = str(tmp_path / 'first')
    cfg.algo.ratio_scope = 'final_prefix'
    cfg.online_diagnostics.policy_probe_every = 1
    cfg.online_diagnostics.policy_probe_samples = 2
    torch.save(TinyPolicy().state_dict(), cfg.algo.pretrained_ckpt)
    (tmp_path/'dataset_stats.json').write_text(json.dumps({k: dict(min=[-1.], max=[1.]) for k in ('state', 'action')}))
    monkeypatch.setattr(online, 'build_base', lambda cfg, device: TinyPolicy().to(device))
    factory = lambda: ChunkActionWrapper(ToyEnv(), chunk_size=3, exec_steps=2)
    online.run(cfg, factory)
    cp = torch.load(Path(cfg.output)/'checkpoints/last.pth', weights_only=True)
    assert cp['metrics']['Probe/Executed_Normalized_RMS'] > 0
    cfg.resume = str(Path(cfg.output)/'checkpoints/last.pth')
    cfg.output = str(tmp_path/'second'); cfg.epochs = 3
    online.run(cfg, factory)
    cfg.algo.ratio_scope = 'full'; cfg.output = str(tmp_path/'bad')
    with pytest.raises(ValueError, match='ratio_scope'):
        online.run(cfg, factory)


def test_backend_report_checks_initial_states_and_successes_separately():
    reports = {m: dict(checkpoint='best.pth', checkpoint_sha256='same', episodes=[
        dict(seed=50000, success=1, initial_state=[0., 1.]),
        dict(seed=50001, success=0, initial_state=[1., 0.])]) for m in MODES}
    reports['gpu_training']['episodes'][0]['initial_state'][0] = .1
    reports['gpu_training']['episodes'][1]['success'] = 1
    r = summarize(reports)['comparisons']
    assert r['cpu_training']['success_disagreements'] == 0
    assert r['gpu_training']['success_disagreements'] == 1
    assert r['gpu_training']['initial_states_over_1e_5'] == 1
    reports['gpu_training']['episodes'][1]['seed'] += 1
    with pytest.raises(ValueError, match='seeds'):
        summarize(reports)


def test_audit_counts_primitive_success_and_rejects_missing_timeout():
    class Env:
        def reset(self, seed):
            self.t = 0
            return {'point_cloud': np.zeros((2, 6))}, {}
        def step(self, actions):
            self.t += 2
            return {}, .5, False, self.t == 4, dict(actual_steps=2, success=False, success_any=self.t == 2)
    class Actor:
        def sample(self, features):
            return torch.zeros(1, 2, 1)
    env = Env()
    rows = audit_episodes(env, Actor(), lambda obs: None, MinMaxNormalizer(), [50000], 4, lambda e: [0])
    assert rows[0]['success'] == 1 and rows[0]['first_success_chunk_end'] == 2
    assert rows[0]['primitive_steps'] == 4
    with pytest.raises(RuntimeError, match='time limit'):
        audit_episodes(env, Actor(), lambda obs: None, MinMaxNormalizer(), [50000], 2, lambda e: [0])


def test_gpu_audit_view_keeps_first_success_and_actual_length():
    class Raw:
        num_envs = 1
        def reset(self, seed):
            return ['obs'], [{'seed': seed}]
        def step(self, actions):
            assert actions.shape == (1, 16, 7)
            return ['end'], [1.], [True], [False], [dict(actual_steps=1, success_any=True)]
    view = SingleGPUView(Raw())
    assert view.reset(7)[1]['seed'] == 7
    result = view.step(np.zeros((16, 7)))
    assert result[2] and result[4]['actual_steps'] == 1


def test_audit_worker_failure_never_becomes_a_success_report(tmp_path, monkeypatch):
    import subprocess
    from tools.diagnostics import compare_online_backends as audit
    out = tmp_path/'audit'
    monkeypatch.setattr(audit.sys, 'argv', ['audit', '--checkpoint', 'best.pth', '--output', str(out)])
    calls = []
    def fail(command, **kwargs):
        calls.append(command)
        assert kwargs['check'] and command[-1] == 'cpu_benchmark'
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(audit.subprocess, 'run', fail)
    with pytest.raises(subprocess.CalledProcessError):
        audit.main()
    assert len(calls) == 1 and not (out/'summary.json').exists()
