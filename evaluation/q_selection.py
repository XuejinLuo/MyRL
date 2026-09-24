"""Paired-seed evaluation of a single frozen Actor with and without Q ranking."""
import csv
from pathlib import Path

import torch
from omegaconf import OmegaConf

from data.episodes import digest
from evaluation.runner import evaluate_policy
from models.factory import build_base, observation_tensorizer
from models.online_policy import FlowPPOPolicy
from models.q_selection import QSelectionPolicy
from utils.experiment import write_json
from utils.normalizer import MinMaxNormalizer
from workflows.critic import FORMAT
from workflows.offline_round import build_q


def run(settings):
    options = settings.q_selection
    counts = list(options.candidates)
    if (not counts or 1 not in counts or len(set(counts)) != len(counts) or
            any(isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in counts)):
        raise ValueError('candidates must be unique positive integers including baseline 1')
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
                    normalizer=cp['normalizer'], selection_rule=cp['selection_rule'],
                    q_weights='q_state_dict', scoring_prefix=cfg.env.exec_steps,
                    action_mapping='legacy unnormalize + environment clipping + training normalization')
    from envs.factory import make_env
    output.mkdir(parents=True)
    OmegaConf.save(settings, output/'config.yaml', resolve=True)
    results, episodes = [], {}
    for k in counts:
        policy = QSelectionPolicy(actor, q, normalizer, k)
        directory = output/f'candidates_{k}'
        def factory():
            env = make_env(cfg)
            policy.set_action_bounds(env.action_space.low, env.action_space.high)
            return env
        metrics = evaluate_policy(factory, policy, encode, normalizer, seeds,
            cfg.model.num_inference_steps, output_dir=directory,
            metadata=dict(**protocol, candidates=k, split='held_out'))
        results.append(dict(candidates=k, **metrics))
        with (directory/'episodes.csv').open() as stream:
            episodes[k] = {int(row['seed']): row for row in csv.DictReader(stream)}
        print(f'Candidates {k}: {metrics["Eval/Success_Rate"]:.1%}', flush=True)
    paired = []
    for k in counts:
        if k == 1:
            continue
        wins = losses = 0
        for seed in seeds:
            before, after = int(episodes[1][seed]['success']), int(episodes[k][seed]['success'])
            wins += after > before
            losses += after < before
        paired.append(dict(candidates=k, baseline_fail_selected_success=wins,
                           baseline_success_selected_fail=losses,
                           success_delta=(wins-losses)/len(seeds)))
    write_json(output/'summary.json', dict(protocol=protocol, seeds=seeds, results=results,
                                         paired=paired))
    with (output/'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f'Comparison: {output / "summary.csv"}', flush=True)
    return results
