"""Paired-seed action-range ablation with a single frozen Actor and Q checkpoint."""
import csv
from pathlib import Path

import torch
from omegaconf import OmegaConf

from data.episodes import digest
from evaluation.runner import evaluate_policy
from models.factory import build_base, observation_tensorizer
from models.online_policy import FlowPPOPolicy
from models.q_selection import QSelectionPolicy
from models.action_mapping import ATOL, RTOL
from utils.experiment import write_json
from utils.normalizer import MinMaxNormalizer
from workflows.critic import FORMAT
from workflows.offline_round import build_q


def paired_success(episodes, seeds, baseline, selected):
    wins = losses = 0
    for seed in seeds:
        before = int(episodes[baseline][seed]['success'])
        after = int(episodes[selected][seed]['success'])
        wins += after > before
        losses += after < before
    return dict(baseline_label=baseline, selected_label=selected,
                baseline_fail_selected_success=wins,
                baseline_success_selected_fail=losses,
                success_delta=(wins-losses)/len(seeds))


def run(settings):
    options = settings.q_selection
    ablation = options.get('ablation', False)
    if not isinstance(ablation, bool):
        raise ValueError('ablation must be a boolean')
    counts = [1, 8] if ablation else list(options.candidates)
    if (not counts or 1 not in counts or len(set(counts)) != len(counts) or
            any(isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in counts)):
        raise ValueError('candidates must be unique positive integers including baseline 1')
    groups = ([('legacy_single', 1, 'legacy'), ('consistent_single', 1, 'consistent'),
               ('consistent_q8', 8, 'consistent')] if ablation else
              [(f'candidates_{k}', k, 'legacy') for k in counts])
    split = options.get('split', 'diagnostic')
    if split not in ('diagnostic', 'held_out'):
        raise ValueError('split must be diagnostic or held_out')
    if options.episodes < 1 or options.sampler not in ('cps', 'ode'):
        raise ValueError('Require positive episodes and CPS/ODE sampler')
    seeds = list(range(options.seed_start, options.seed_start + options.episodes))
    path = Path(options.checkpoint).expanduser().resolve()
    cp = torch.load(path, map_location='cpu', weights_only=True)
    if cp.get('format') != FORMAT:
        raise ValueError('Run train_critic.py first: this requires a saved Q checkpoint')
    if set(seeds) & set(cp['provenance']['excluded_eval_seeds']):
        raise ValueError('Evaluation seeds overlap Actor validation or training collection seeds')
    output = Path(options.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f'{output}: choose a new q_selection.output')
    cfg = OmegaConf.create(cp['config'])
    cfg.device = settings.device
    device = torch.device(settings.device)
    base = build_base(cfg, device)
    base.load_state_dict(cp['model_state_dict'], strict=True)
    base.eval().requires_grad_(False)
    q = build_q(cfg).to(device)
    q.load_state_dict(cp['q_state_dict'], strict=True)
    q.eval().requires_grad_(False)
    actor = FlowPPOPolicy(base, num_steps=cfg.model.num_inference_steps,
                         eval_mode=options.sampler, **cp['sampler'])
    normalizer = MinMaxNormalizer()
    normalizer.stats = cp['normalizer']
    encode = observation_tensorizer(cfg, normalizer, device)
    protocol = dict(checkpoint=str(path), checkpoint_sha256=digest(path),
                    actor_sha256=cp['provenance']['actor_sha256'], critic_epoch=cp['epoch'],
                    sampler=options.sampler, sampler_parameters=cp['sampler'],
                    num_inference_steps=cfg.model.num_inference_steps,
                    env=cp['config']['env'], model=cp['config']['model'],
                    normalizer=cp['normalizer'], normalizer_eps=normalizer.eps,
                    selection_rule=cp['selection_rule'],
                    q_weights='q_state_dict', scoring_prefix=cfg.env.exec_steps,
                    comparison='clipping_ablation' if ablation else 'legacy', split=split,
                    seeds=seeds,
                    action_mapping=dict(legacy='original execution; saturated training representation for Q',
                        consistent='unnormalize([-1,1]) intersect environment; return Q representation'),
                    diagnostics=dict(scope='actually executed prefix, trimmed to actual_steps',
                        fractions='coordinate counts / coordinates; pooled across decisions',
                        candidates='all K candidates at visited states', selected='chosen candidate only',
                        env_clipped='physical bounds applied after mode-specific normalized clipping',
                        raw_env_clipped='physical bounds applied to the original raw candidate',
                        raw_to_score='original physical execution minus inverse Q representation',
                        execution_to_score='mode physical execution minus inverse Q representation; '
                                           'selected uses observed executed_actions',
                        single='hypothetical Q representation only; critic is never called',
                        atol=ATOL, rtol=RTOL))
    from envs.factory import make_env
    output.mkdir(parents=True)
    OmegaConf.save(settings, output/'config.yaml', resolve=True)
    results, episodes = [], {}
    for label, k, mode in groups:
        policy = QSelectionPolicy(actor, q, normalizer, k, action_mode=mode, exec_steps=cfg.env.exec_steps)
        directory = output/label
        def factory():
            env = make_env(cfg)
            policy.set_action_bounds(env.action_space.low, env.action_space.high)
            return env
        metrics = evaluate_policy(factory, policy, encode, normalizer, seeds,
            cfg.model.num_inference_steps, output_dir=directory,
            metadata=dict(**protocol, label=label, candidates=k, action_mode=mode))
        results.append(dict(label=label, candidates=k, action_mode=mode, **metrics))
        with (directory/'episodes.csv').open() as stream:
            episodes[label] = {int(row['seed']): row for row in csv.DictReader(stream)}
        print(f'{label} (K={k}, {mode}): {metrics["Eval/Success_Rate"]:.1%}', flush=True)
    paired = []
    comparisons = ([('legacy_single', 'consistent_single'),
                    ('consistent_single', 'consistent_q8')] if ablation else
                   [('candidates_1', f'candidates_{k}') for k in counts if k != 1])
    for baseline, selected in comparisons:
        paired.append(paired_success(episodes, seeds, baseline, selected))
    write_json(output/'summary.json', dict(protocol=protocol, seeds=seeds, results=results,
                                         paired=paired))
    with (output/'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f'Comparison: {output / "summary.csv"}', flush=True)
    return results
