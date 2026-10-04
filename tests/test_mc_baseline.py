"""No learned baseline must decouple Actor updates from Critic predictions."""
import copy
from pathlib import Path
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from algos.pg import compute_mc_returns, normalize_advantages, FlowPPO
from envs.chunk_wrapper import ChunkActionWrapper
from envs.online_task import OnlineTaskWrapper, task_protocol
from envs.online_vector import SingleOnlineEnv
from tests.test_flow_ppo import actor
from tests.test_online_task import config, DenseNeverSuccess
from utils.normalizer import MinMaxNormalizer
from workflows.online_mc import CompleteEpisodeCollector, return_protocol, check_return_resume


@pytest.mark.parametrize('gamma', [.9, 1.])
def test_returns_and_actor_signal_are_independent_of_values(gamma):
    reward = torch.tensor([.2, .8, -.1, -.3])
    values = torch.tensor([-20., 1., 30., -4.])
    dones = torch.tensor([0., 1., 0., 1.])
    lengths = torch.tensor([2, 1, 2, 1])
    old, target = compute_mc_returns(reward, values, dones, lengths, gamma)
    signal, same_target = compute_mc_returns(reward, values, dones, lengths, gamma, baseline='none')
    changed, _ = compute_mc_returns(reward, values * 50, dones, lengths, gamma, baseline='none')
    torch.testing.assert_close(signal, target, rtol=0, atol=0)
    torch.testing.assert_close(signal, changed, rtol=0, atol=0)
    torch.testing.assert_close(same_target, target, rtol=0, atol=0)
    torch.testing.assert_close(old, target-values, rtol=0, atol=0)
    torch.testing.assert_close(normalize_advantages(signal), normalize_advantages(target))
    assert not torch.allclose(normalize_advantages(old), normalize_advantages(signal))


def test_profile_changes_only_baseline_and_output():
    from workflows.online import validate
    from utils.config import validate_common
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1]/'configs'), version_base=None):
        original = compose(config_name='train_online_rl_mc')
        cfg = compose(config_name='train_online_rl_mc_nobaseline')
    validate_common(cfg); validate(cfg)
    assert cfg.algo.mc_baseline == 'none' and cfg.algo.critic_warmup_epochs == 5
    assert cfg.stages.online.initial_ckpt.endswith('/online_rl03/checkpoints/best.pth')
    cfg.algo.mc_baseline = original.algo.mc_baseline
    cfg.output = original.output
    assert OmegaConf.to_container(cfg, resolve=True) == OmegaConf.to_container(original, resolve=True)


@pytest.mark.parametrize('invalid', ['zero', None, False])
def test_invalid_baseline_is_rejected(tmp_path, invalid):
    cfg = config(tmp_path); cfg.algo.return_estimator = 'mc'; cfg.algo.mc_baseline = invalid
    with pytest.raises(ValueError, match='mc_baseline'):
        return_protocol(cfg)


def test_resume_legacy_mc_and_cross_baseline_rejection(tmp_path):
    cfg = config(tmp_path); cfg.algo.return_estimator = 'mc'
    legacy = return_protocol(cfg); del legacy['mc_baseline']
    checkpoint = dict(return_protocol=legacy, config=OmegaConf.to_container(cfg))
    check_return_resume(checkpoint, cfg)
    assert 'mc_baseline' not in checkpoint['return_protocol']
    # A legacy config-only MC checkpoint also canonicalizes to value.
    check_return_resume(dict(config=checkpoint['config']), cfg)
    cfg.algo.mc_baseline = 'none'
    with pytest.raises(ValueError, match='return protocol mismatch'):
        check_return_resume(checkpoint, cfg)
    check_return_resume(dict(return_protocol=return_protocol(cfg)), cfg)
    cfg.algo.return_estimator = 'gae'
    with pytest.raises(ValueError, match='requires return_estimator=mc'):
        return_protocol(cfg)


def test_collect_and_ppo_actor_update_ignore_changed_critic(tmp_path):
    cfg = config(tmp_path); cfg.algo.return_estimator = 'mc'; cfg.algo.mc_baseline = 'none'
    cfg.online_rollout = dict(backend='cpu', num_envs=1, episodes_per_batch=4)
    protocol = task_protocol(cfg)
    template = actor()
    encode = lambda obs: torch.tensor([[float(o[0]), 0.] for o in obs])
    results = []
    for name, bias in [('low', -200.), ('high', 200.)]:
        pi = copy.deepcopy(template)
        critic = torch.nn.Linear(3, 1)
        torch.nn.init.zeros_(critic.weight); torch.nn.init.constant_(critic.bias, bias)
        env = SingleOnlineEnv(ChunkActionWrapper(OnlineTaskWrapper(
            DenseNeverSuccess(success_at=3), protocol), 3, 2))
        collector = CompleteEpisodeCollector(env, pi, critic, encode, MinMaxNormalizer(),
                                             cfg, protocol, tmp_path/name, 42)
        torch.manual_seed(123)
        batch, signal, returns, metrics = collector.collect(1)
        assert metrics['Adv/Uses_Learned_Baseline'] == 0
        assert torch.all(batch['values'] == bias)
        torch.testing.assert_close(signal, returns, rtol=0, atol=0)
        trainer = FlowPPO(pi, critic, actor_lr=1e-4, target_kl=None)
        torch.manual_seed(456)
        updated = trainer.update(batch, signal, returns, batch_size=4, epochs=2)
        assert updated['ppo/actor_updates'] == 4 and trainer.critic_optimizer.state
        assert pi.policy.backbone.w != template.policy.backbone.w
        collector.attach_advantages(signal, normalize_advantages(signal), True)
        assert collector.flush(1, final=True)['Episode/censored_Count'] == 0
        results.append((batch['chains'], signal, copy.deepcopy(pi.state_dict())))
        env.close()
    for a, b in zip(results[0][:2], results[1][:2]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for key in results[0][2]:
        torch.testing.assert_close(results[0][2][key], results[1][2][key], rtol=0, atol=0)
