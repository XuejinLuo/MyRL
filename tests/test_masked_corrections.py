"""Real masked Flow/Transformer gradients, audited starts, and CPU optimizer integration."""
import copy
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from omegaconf import OmegaConf

from algos.diffusion_utils.flow_matching import OTFlowMatching
from algos.embodied_idql import Policy_IDQL_Wrapper
from data.corrections import actor_valid_lengths, prepare
from data.dataset import TrajectoryDataset
from data.episodes import digest, load_episode, load_sources, validate_episode
from data.iterative_sampling import SourceDataset, FixedBatches, SamplingMetrics, validate_update_config
from models.backbones.transformer import ActionDiffusionTransformer
from models.policy import EmbodiedGenPolicy
from tests.test_human_corrections import raw_session
from tests.test_iterative_sampling import fixed_config, episode, prepare_inputs, LossTinyPolicy
from tests.test_q_selection import norm
from tests.test_stages import CriticFeatures
from tests.test_flow_ppo import ToyEnv
from envs.chunk_wrapper import ChunkActionWrapper


@pytest.mark.parametrize('length', [1, 2, 6, 10, 16, 35])
def test_lengths_cover_short_segments_and_tails_without_crossing_switches(length):
    segments = [dict(start=2, stop=2+length, decision='accept'),
                dict(start=3+length, stop=5+length, decision='reject'),
                dict(start=6+length, stop=7+length, decision='accept')]
    lengths = actor_valid_lengths(length+9, segments, 16, True, 'masked')
    assert lengths[2:2+length].tolist() == [min(16, length-i) for i in range(length)]
    assert lengths[6+length] == 1
    assert np.count_nonzero(lengths) == length+1
    assert not actor_valid_lengths(length+9, segments, 16, False, 'masked').any()
    full = actor_valid_lengths(length+9, segments, 16, True)
    assert np.count_nonzero(full) == max(0, length-15)
    stricter = actor_valid_lengths(length+9, segments, 16, True, 'masked', 2)
    assert np.count_nonzero(stricter) == max(0, length-1)


def test_reexport_and_dataset_keep_critic_transitions_and_separate_actor_storage(tmp_path):
    cfg = fixed_config(tmp_path, 'demo_success_correction')
    prepare_inputs(cfg, tmp_path)
    session = raw_session(cfg, tmp_path)
    hashes = {p: digest(p) for p in session.iterdir()}
    old = prepare(cfg.manifest, session, cfg.stats_path, tmp_path/'full')
    new = prepare(cfg.manifest, session, cfg.stats_path, tmp_path/'masked', 'masked')
    assert all(digest(p) == value for p, value in hashes.items())
    eps, spec = load_sources(new)
    assert eps[-1]['actor_valid_length'].tolist() == [0, 0, 3, 3, 3, 2, 1, 0, 2, 1]
    report = json.loads((new.parent/'cleaning_report.json').read_text())
    assert report['actor_starts'] == report['accepted_human_steps'] == 7
    assert report['full_chunk_starts'] == 3 and report['partial_chunk_starts'] == 4
    assert report['actor_episodes'] == 1 and report['actor_label_mode'] == 'masked'
    with pytest.raises(ValueError, match='label mode'):
        SourceDataset(TrajectoryDataset(eps, cfg, norm()), spec, ['demonstrations'])
    cfg.actor_sampling.correction_label_mode = 'masked'
    data = SourceDataset(TrajectoryDataset(eps, cfg, norm()), spec, ['demonstrations'])
    cfg.actor_sampling.correction_label_mode = 'full_chunk'
    old_eps, old_spec = load_sources(old)
    full = SourceDataset(TrajectoryDataset(old_eps, cfg, norm()), old_spec, ['demonstrations'])
    index = int(data.offsets[-2])+6  # one human step; next action belongs to the policy
    actual, previous = data[index], full[index]
    for key in ('action_chunk', 'reward', 'done', 'discount', 'next_pc', 'next_state'):
        torch.testing.assert_close(actual[key], previous[key])
    assert actual['actor_mask'].tolist() == [True, False, False]
    assert actual['actor_actions'].data_ptr() != actual['action_chunk'].data_ptr()
    actual['actor_actions'].fill_(100)
    torch.testing.assert_close(actual['action_chunk'], previous['action_chunk'])
    assert actual['discount'].item() == pytest.approx(cfg.algo.discount**2)  # not cut at human stop
    # Legacy cleaned exports without the v2 field/marker remain readable.
    old_eps[-1].pop('actor_valid_length')
    for key in ('actor_label_schema', 'actor_label_mode', 'min_valid_length'):
        old_spec['sources'][-1].pop(key)
    legacy = SourceDataset(TrajectoryDataset(old_eps, cfg, norm()), old_spec, ['demonstrations'])
    assert legacy.report['correction_actor_starts'] == 3
    assert list(FixedBatches(full, 8, 20, 55, 'mixed')) == list(FixedBatches(data, 8, 20, 55, 'mixed'))


@pytest.mark.parametrize('bad', ['negative', 'float', 'past_end', 'ineligible', 'zero_eligible'])
def test_invalid_npz_lengths_fail_closed(bad):
    ep = episode(5, True)
    ep['actor_valid_length'] = np.array([3, 3, 3, 2, 1])
    ep['actor_eligible'] = np.ones(5, dtype=bool)
    if bad == 'negative':
        ep['actor_valid_length'][0] = -1
    elif bad == 'float':
        ep['actor_valid_length'] = ep['actor_valid_length'].astype(float)
    elif bad == 'past_end':
        ep['actor_valid_length'][-1] = 2
    elif bad == 'ineligible':
        ep['actor_eligible'][0] = False
    else:
        ep['actor_valid_length'][0] = 0
    with pytest.raises(ValueError, match='actor_valid_length'):
        validate_episode(ep)


def test_uniform_starts_remove_single_start_episode_overweight_and_audit_actual_visits(tmp_path):
    cfg = fixed_config(tmp_path, 'demo_success_correction')
    cfg.model.chunk_size = 16
    eps = [episode(20, True), episode(30, True), episode(16, True), episode(154, True)]
    for ep in eps[2:]:
        ep['actor_eligible'] = np.arange(len(ep['action'])) <= len(ep['action'])-16
    spec = dict(sources=[dict(name='demonstrations', episodes=[{}]),
        dict(name='success', episodes=[{}]), dict(name='human', role='human_correction', chunk_size=16,
        episodes=[dict(seed=15032), dict(seed=15056)])])
    data = SourceDataset(TrajectoryDataset(eps, cfg, norm()), spec, ['demonstrations'])
    assert data.report['correction_actor_episodes'] == 2 and len(data.correction_pool) == 140
    critic_before = list(FixedBatches(data, 64, 10, 31))
    modes = {}
    for mode in ('episode', 'uniform_start'):
        counts = Counter()
        stats = SamplingMetrics(3)
        for batch in FixedBatches(data, 64, 1000, 42, 'demo_success_correction', .5, .25, mode):
            identities = [data.dataset.indices[i] for i in batch]
            assert sum(e == 0 for e, _ in identities) == 32
            assert sum(e == 1 for e, _ in identities) == 16
            counts.update((e, t) for e, t in identities if e >= 2)
            # Cheap metadata batch; count *consumed* examples, not DataLoader prefetch.
            meta = dict(sampling_group=torch.tensor([data.group_ids[e] for e, _ in identities]),
                sampling_source=torch.tensor([data.source_ids[e] for e, _ in identities]),
                sampling_episode=torch.tensor([e for e, _ in identities]),
                sampling_start=torch.tensor([t for _, t in identities]), actor_mask=torch.ones(64, 16, dtype=torch.bool))
            stats.add('Actor', meta)
        modes[mode] = counts
        report = stats.correction_report(data.correction_episodes)
        assert report['episodes']['2']['sampled'] == counts[(2, 0)]
        assert report['episodes']['2']['seed'] == 15032
        assert report['episodes']['3']['unique_starts'] == 139
        assert report['valid_length_histogram'] == {'16': 16000}
    assert 7600 < modes['episode'][(2, 0)] < 8400
    assert 70 < modes['uniform_start'][(2, 0)] < 160
    assert max(modes['uniform_start'].values()) < 180
    assert critic_before == list(FixedBatches(data, 64, 10, 31))


class StateEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 8)
    def forward(self, obs):
        return self.linear(obs['state'])


def real_small_policy():
    # Use the production policy/wrapper/loss and actual temporal attention;
    # only shrink dimensions and replace the point-cloud encoder for CPU testing.
    policy = EmbodiedGenPolicy.__new__(EmbodiedGenPolicy)
    nn.Module.__init__(policy)
    policy.algo_type, policy.chunk_size, policy.action_dim = 'flow', 16, 2
    policy.encoder = StateEncoder()
    policy.backbone = ActionDiffusionTransformer(2, 8, 16, embed_dim=16, depth=2, num_heads=2)
    # Zero-init DiT would hide temporal leakage. Activate attention and output.
    with torch.no_grad():
        policy.backbone.action_out.weight.normal_(std=.1)
        for block in policy.backbone.blocks:
            block.adaLN_modulation.linear.weight.normal_(std=.1)
            block.adaLN_modulation.linear.bias.normal_(std=.1)
    policy.scheduler = OTFlowMatching()
    return policy


def loss_and_grad(policy, actions, mask):
    policy.zero_grad(set_to_none=True)
    wrapper = Policy_IDQL_Wrapper(policy)
    torch.manual_seed(712)
    loss = wrapper.compute_loss(dict(pc=torch.zeros(len(actions), 4, 3), state=torch.ones(len(actions), 1)),
                                actions, reduction='none', action_mask=mask)
    loss.mean().backward()
    return loss.detach(), {k: p.grad.clone() for k, p in policy.named_parameters()}


def test_production_transformer_loss_and_gradients_ignore_arbitrary_invalid_targets():
    torch.manual_seed(7)
    policy = real_small_policy()
    lengths = torch.tensor([1, 2, 6, 10, 16])
    mask = torch.arange(16)[None] < lengths[:, None]
    actions = torch.randn(5, 16, 2)
    loss, grads = loss_and_grad(policy, actions, mask)
    assert grads['backbone.blocks.0.attn.in_proj_weight'].abs().sum() > 0
    for garbage in (10000., float('nan')):
        changed = torch.where(mask[..., None], actions, garbage)
        other_loss, other_grads = loss_and_grad(policy, changed, mask)
        torch.testing.assert_close(loss, other_loss, atol=0, rtol=0)
        for key in grads:
            torch.testing.assert_close(grads[key], other_grads[key], atol=0, rtol=0)
    # Show that the active temporal model really responds without masking.
    all_full, _ = loss_and_grad(policy, actions, None)
    altered_full, _ = loss_and_grad(policy, torch.where(mask[..., None], actions, 100.), None)
    assert not torch.allclose(all_full, altered_full)
    full_loss, full_grads = loss_and_grad(policy, actions, torch.ones_like(mask))
    old_loss, old_grads = loss_and_grad(policy, actions, None)
    torch.testing.assert_close(full_loss, old_loss)
    for key in full_grads:
        torch.testing.assert_close(full_grads[key], old_grads[key])
    empty = mask.clone()
    empty[0] = False
    with pytest.raises(ValueError, match='valid step'):
        loss_and_grad(policy, actions, empty)
    # Network keys, tensor shapes, and inference call remain unchanged.
    clone = real_small_policy()
    clone.load_state_dict(policy.state_dict(), strict=True)
    sample = policy.scheduler.sample(policy.backbone, torch.zeros(1, 8), (16, 2), num_steps=2)
    assert sample.shape == (1, 16, 2) and torch.isfinite(sample).all()


def test_flow_normalizes_by_valid_dimensions_before_batch_mean():
    class KnownError(nn.Module):
        def forward(self, x, t, cond):
            return torch.ones_like(x) * cond[:, :1, None]
    # sigma_min=1, zero targets => target velocity=0: exact errors 1 and 9,
    # independent of sampled noise/time and unequal valid lengths 1 vs 16.
    scheduler = OTFlowMatching(sigma_min=1.)
    mask = torch.arange(16)[None] < torch.tensor([[1], [16]])
    kwargs = dict(model=KnownError(), x1=torch.zeros(2, 16, 2), cond=torch.tensor([[1.], [3.]]), action_mask=mask)
    torch.testing.assert_close(scheduler.compute_loss(**kwargs, reduction='none'), torch.tensor([1., 9.]))
    assert scheduler.compute_loss(**kwargs).item() == 5.


def test_partial_labels_never_reach_q_or_change_nonhuman_rejection(tmp_path, monkeypatch):
    from workflows import offline_round
    from workflows.iterative_updates import actor_advantage
    monkeypatch.setattr(offline_round, 'CriticFeatureExtractor', CriticFeatures)
    cfg = fixed_config(tmp_path)
    agent = offline_round.build_agent(cfg, LossTinyPolicy(), torch.device('cpu'))
    obs = dict(pc=torch.zeros(4, 4, 3), state=torch.zeros(4, 1))
    actions = torch.zeros(4, 3, 1)
    actions[-1, 1:] = float('nan')
    mask = torch.tensor([[True]*3]*3 + [[True, False, False]])
    seen = []
    def check_q(module, args):
        assert len(args[1]) == 3 and torch.isfinite(args[1]).all()
        seen.append(1)
    hook = agent.q_target.register_forward_pre_hook(check_q)
    advantages = actor_advantage(agent, obs, actions, mask.all(1))
    assert seen and advantages[-1] == 0
    hook.remove()
    results = []
    for placeholder in (-1e9, 1e9):
        advantages = torch.tensor([[0.], [-1e6], [-1e6], [placeholder]])
        torch.manual_seed(99)
        _, details = agent.update_actor(obs, actions, adv=advantages, return_details=True,
            force_keep=torch.tensor([False, False, False, True]), action_mask=mask)
        results.append(details['keep_mask'])
        assert details['advantage_valid'].tolist() == [True, True, True, False]
        assert torch.isfinite(details['losses']).all()
    assert results[0].tolist() == results[1].tolist() == [True, False, False, True]
    # Even the all-partial case is finite and needs no Q/V forward.
    hook = agent.q_target.register_forward_pre_hook(lambda *args: pytest.fail('partial labels queried Q'))
    _, details = agent.update_actor(obs, torch.zeros_like(actions), return_details=True,
        force_keep=torch.ones(4, dtype=torch.bool), action_mask=mask[-1:].expand(4, -1))
    hook.remove()
    assert details['keep_mask'].all() and not details['advantage_valid'].any()


@pytest.mark.parametrize('label_mode', ['full_chunk', 'masked'])
def test_actual_optimizer_early_evaluation_checkpoint_and_sampling_audit(tmp_path, monkeypatch, label_mode):
    from workflows import iterative, offline_round
    cfg = fixed_config(tmp_path, 'demo_success_correction')
    cfg.actor_sampling.correction_label_mode = label_mode
    cfg.actor_sampling.correction_sampling = 'uniform_start'
    cfg.eval_actor_updates = [1, 2]
    cfg.eval_every_updates = 6
    prepare_inputs(cfg, tmp_path)
    session = raw_session(cfg, tmp_path)
    cfg.manifest = str(prepare(cfg.manifest, session, cfg.stats_path, tmp_path/'clean', label_mode))
    cfg.output = str(tmp_path/'trained')
    passed_masks = []
    class CapturingPolicy(LossTinyPolicy):
        def compute_loss(self, *args, action_mask=None, **kwargs):
            passed_masks.append(None if action_mask is None else action_mask.clone())
            return super().compute_loss(*args, action_mask=action_mask, **kwargs)
    monkeypatch.setattr(iterative, 'build_base', lambda cfg, device: CapturingPolicy().to(device))
    monkeypatch.setattr(offline_round, 'CriticFeatureExtractor', CriticFeatures)
    monkeypatch.setitem(sys.modules, 'envs.factory', SimpleNamespace(
        make_env=lambda cfg, **kwargs: ChunkActionWrapper(ToyEnv(), 3, 2)))
    iterative.run(cfg)
    root = Path(cfg.output)
    rows = [json.loads(s) for s in (root/'metrics.jsonl').read_text().splitlines()]
    assert [r['update_step'] for r in rows if r['evaluation']] == [0, 3, 4, 6]
    for row in rows:
        assert row['actor_update_step'] == max(0, row['update_step']-2)
        if row['evaluation']:
            metadata = json.loads((Path(row['evaluation'])/'summary.json').read_text())['metadata']
            assert metadata['actor_update_step'] == row['actor_update_step']
            cp = torch.load(row['checkpoint'], weights_only=True)
            assert cp['actor_update_step'] == row['actor_update_step']
    assert len(passed_masks) == 4
    if label_mode == 'masked':
        assert all(mask is not None for mask in passed_masks)
        assert any(not mask.all() for mask in passed_masks)
    else:
        assert all(mask is None for mask in passed_masks)
    report = json.loads((root/'round_000/sampling.json').read_text())
    assert report['completed_actor_updates'] == 4
    assert report['correction_actor_episodes'] == 1
    audit = report['correction_audit']
    assert sum(audit['valid_length_histogram'].values()) == 4
    assert list(audit['episodes'].values())[0]['sampled'] == 4
    selection = json.loads((root/'selection.json').read_text())
    assert selection['update_step'] == selection['actor_update_step'] == 0
    original = torch.load(cfg.initial_ckpt, weights_only=True)['model_state_dict']
    final = torch.load(root/'round_000/checkpoints/step_0000006.pth', weights_only=True)['model_state_dict']
    assert any(not torch.equal(original[k], v) for k, v in final.items())


@pytest.mark.parametrize('key,value', [('eval_actor_updates', [0]), ('eval_actor_updates', [1, 1]),
    ('eval_actor_updates', [5]), ('eval_actor_updates', [True]),
    ('actor_sampling.correction_sampling', 'unknown'), ('actor_sampling.correction_label_mode', 'unknown')])
def test_invalid_new_options(tmp_path, key, value):
    cfg = fixed_config(tmp_path, 'demo_success_correction')
    OmegaConf.update(cfg, key, value)
    with pytest.raises(ValueError):
        validate_update_config(cfg)


def test_masked_export_excludes_failed_rejected_critic_only_and_dropped_episodes(tmp_path):
    from data.episodes import save_episode, write_json
    cfg = fixed_config(tmp_path, 'demo_success_correction')
    prepare_inputs(cfg, tmp_path)
    session = raw_session(cfg, tmp_path)
    review = json.loads((session/'review.json').read_text())
    template = review['episodes'][0]
    for index, decision in enumerate(('failed', 'reject', 'critic_only', 'drop'), 1):
        name = f'seed_{12000+index}'
        ep = episode(10, decision != 'failed', .1+index*.1)
        save_episode(session/f'{name}.npz', ep)
        metadata = dict(seed=12000+index, segments=[dict(id=0, start=2, stop=7), dict(id=1, start=8, stop=10)])
        write_json(session/f'{name}.json', metadata)
        item = copy.deepcopy(template)
        item.update(path=f'{name}.npz', sha256=digest(session/f'{name}.npz'),
                    metadata_sha256=digest(session/f'{name}.json'),
                    decision=decision if decision in ('critic_only', 'drop') else 'keep')
        if decision == 'reject':
            for segment in item['segments']:
                segment['decision'] = 'reject'
        review['episodes'].append(item)
    write_json(session/'review.json', review)
    manifest = prepare(cfg.manifest, session, cfg.stats_path, tmp_path/'mixed_review', 'masked')
    episodes, spec = load_sources(manifest)
    assert len(spec['sources'][-1]['episodes']) == 4
    assert [int(ep['actor_eligible'].sum()) for ep in episodes[-4:]] == [7, 0, 0, 0]
    report = json.loads((manifest.parent/'cleaning_report.json').read_text())
    assert [row['actor_starts'] for row in report['episodes']] == [7, 0, 0, 0, 0]
    assert report['episodes'][-1]['reason'] == 'dropped'


def test_early_candidate_selection_uses_success_and_actor_coordinate(tmp_path, monkeypatch):
    from workflows import offline_round
    from workflows.iterative_updates import train_updates
    from data.episodes import write_json
    monkeypatch.setattr(offline_round, 'CriticFeatureExtractor', CriticFeatures)
    cfg = fixed_config(tmp_path, 'demo_success_correction')
    cfg.actor_sampling.correction_sampling = 'uniform_start'
    cfg.eval_actor_updates = [1, 2]
    cfg.eval_every_updates = 6
    base = prepare_inputs(cfg, tmp_path)
    session = raw_session(cfg, tmp_path)
    manifest = prepare(cfg.manifest, session, cfg.stats_path, tmp_path/'clean')
    episodes, spec = load_sources(manifest)
    run = tmp_path/'round_000'
    (run/'checkpoints').mkdir(parents=True)
    saved = []
    def evaluate(epoch):
        return {'Eval/Success_Rate': {3: .4, 4: .8, 6: .8}[epoch], 'Eval/Mean_Reward': epoch*100}
    def save(path, step, result):
        saved.append((Path(path).name, step))
    selected = train_updates(cfg, base, norm(), episodes, run, evaluate, save,
        {'Eval/Success_Rate': .2}, None, spec, 42)
    assert selected == str(run/'checkpoints/step_0000004.pth')
    selection = json.loads((run/'selection.json').read_text())
    assert selection['update_step'] == 4 and selection['actor_update_step'] == 2
    assert ('step_0000003.pth', 3) in saved
    assert ('best.pth', 6) not in saved  # greater dense reward cannot break success tie
