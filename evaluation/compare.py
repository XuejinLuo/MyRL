"""Matched held-out comparison of any subset of the three training stages."""
import csv
from pathlib import Path
import torch
from omegaconf import OmegaConf
from models.factory import build_base
from models.checkpoint import policy_weights
from utils.normalizer import MinMaxNormalizer
from utils.experiment import evaluate_base, write_json
from data.episodes import digest


def comparison_protocol(config, normalizer, noise_level, min_std, allow_observation_variants=False):
    """Opt-in representation ablations; control/action/seeds stay strictly matched."""
    import copy
    env, model = copy.deepcopy(config['env']), copy.deepcopy(config['model'])
    if allow_observation_variants:
        for key in ('sampling', 'observation', 'num_points', 'use_color'):
            env.pop(key, None)
        for key in ('encoder_type', 'in_channels'):
            model.pop(key, None)
    return dict(env=env, model=model, normalizer=normalizer,
                noise_level=noise_level, min_std=min_std)


def run(settings):
    options = settings.comparison
    seeds = list(range(options.seed_start, options.seed_start + options.episodes))
    if not seeds or not options.checkpoints or not options.samplers:
        raise ValueError('Comparison requires checkpoints, samplers and positive episodes')
    if len(set(options.samplers)) != len(options.samplers) or set(options.samplers) - {'cps', 'ode'}:
        raise ValueError('Comparison samplers must be unique CPS/ODE modes')
    out = Path(options.output).expanduser().resolve()
    if out.exists():
        raise FileExistsError(f'{out}: select a new comparison.output')
    # Preflight every input before producing a partially comparable report.
    inputs, protocol, observation_protocols = [], None, {}
    for label, checkpoint in options.checkpoints.items():
        if Path(label).name != label or label in ('.', '..'):
            raise ValueError('Comparison labels must be simple directory names')
        path = Path(checkpoint).expanduser().resolve()
        cp = torch.load(path, map_location='cpu', weights_only=True)
        if 'config' not in cp or 'normalizer' not in cp:
            raise ValueError(f'{path}: requires embedded config and normalizer')
        cfg = OmegaConf.create(cp['config'])
        cfg.device = settings.device
        cfg.noise_level = cfg.get('noise_level', cfg.algo.get('noise_level', .7))
        cfg.min_std = cfg.get('min_std', cfg.algo.get('min_std', .0067))
        current = comparison_protocol(cp['config'], cp['normalizer'], cfg.noise_level, cfg.min_std,
                                      options.get('allow_observation_variants', False))
        observation_protocols[label] = dict(env=cp['config']['env'], model=cp['config']['model'])
        if protocol is not None and protocol != current:
            raise ValueError('Cannot compare incompatible environment/model/normalizer/sampler settings')
        protocol = current
        if set(seeds) & set(cfg.eval.seeds):
            raise ValueError('Test seeds overlap checkpoint-selection seeds')
        collection_cfg = cfg.get('stages', {}).get('iterative', cfg)
        if 'collect_seed_start' in collection_cfg:
            collection = range(collection_cfg.collect_seed_start,
                collection_cfg.collect_seed_start + collection_cfg.rounds*collection_cfg.episodes_per_round)
            if set(collection) & set(seeds):
                raise ValueError('Test seeds overlap iterative collection seeds')
        inputs.append((label, path, cfg))
    out.mkdir(parents=True)
    OmegaConf.save(settings, out/'config.yaml', resolve=True)
    rows = []
    for label, path, cfg in inputs:
        cp = torch.load(path, map_location=settings.device, weights_only=True)
        cfg.video = dict(every=1 if settings.video.episodes else 0, episodes=settings.video.episodes)
        base = build_base(cfg, settings.device)
        base.load_state_dict(policy_weights(cp), strict=True)
        normalizer = MinMaxNormalizer()
        normalizer.stats = cp['normalizer']
        for sampler in options.samplers:
            cfg.eval.sampler = sampler
            result = evaluate_base(cfg, base, normalizer, out/label, cp.get('epoch', 0),
                                   tag='test', seeds=seeds, checkpoint=path)
            rows.append(dict(stage=label, sampler=sampler, checkpoint=str(path),
                             checkpoint_sha256=digest(path), **result))
        del base, cp
    write_json(out/'summary.json', dict(protocol=protocol, observations=observation_protocols,
                                       test_seeds=seeds, results=rows))
    with (out/'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Comparison: {out / "summary.csv"}', flush=True)
    return rows
