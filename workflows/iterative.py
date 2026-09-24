"""Collect -> append immutable sources -> offline IDQL -> validation selection."""
import json
import shutil
from pathlib import Path
import numpy as np
import torch
import hydra
from omegaconf import OmegaConf
from models.factory import build_base, observation_encoder
from utils.config import validate_common
from models.online_policy import FlowPPOPolicy
from utils.normalizer import MinMaxNormalizer
from models.checkpoint import policy_weights
from evaluation.runner import seed_all
from data.episodes import (digest, load_sources, load_episode,
                                  save_episode, write_json)
from workflows.offline_round import train_round
from utils.experiment import evaluate_base, log_metrics, write_selection, evaluation_paths, tracking


def validate(cfg):
    from data.iterative_sampling import validate_update_config
    validate_update_config(cfg)
    if not isinstance(cfg.get('collect', True), bool):
        raise ValueError('collect must be a boolean')
    if not cfg.get('collect', True) and (cfg.rounds != 1 or cfg.get('reuse_collection_dir')):
        raise ValueError('collect=false requires rounds=1 and no reuse_collection_dir; use the full manifest')
    fixed = cfg.get('updates_per_round') is not None
    for k in (('rounds', 'episodes_per_round', 'batch_size') if fixed else
              ('rounds', 'episodes_per_round', 'epochs', 'batch_size')):
        if cfg[k] < 1:
            raise ValueError(f'{k} must be positive')
    if not fixed and not 0 <= cfg.critic_warmup_epochs < cfg.epochs:
        raise ValueError('Require 0 <= critic_warmup_epochs < epochs')
    if cfg.batch_size < 2 or cfg.num_workers < 0 or (not fixed and cfg.eval.every < 1):
        raise ValueError('Invalid loader/evaluation settings')
    if cfg.model.algo_type != 'flow' or not 1 <= cfg.env.exec_steps <= cfg.model.chunk_size:
        raise ValueError('Require Flow policy and valid execution prefix')
    if cfg.collect_sampler not in ('cps', 'ode') or cfg.eval.sampler not in ('cps', 'ode'):
        raise ValueError('Unknown sampler')
    if not 0 <= cfg.ema_decay < 1 or not 0 < cfg.algo.discount <= 1:
        raise ValueError('Invalid EMA/discount')
    validation = set(cfg.eval.seeds)
    test = set(range(cfg.comparison.seed_start, cfg.comparison.seed_start + cfg.comparison.episodes))
    collection = set(range(cfg.collect_seed_start, cfg.collect_seed_start + cfg.rounds*cfg.episodes_per_round))
    if not validation or not test or validation & test or collection & (validation | test):
        raise ValueError('Collection, validation and test seeds must be nonempty/disjoint')
    if len(validation) != len(cfg.eval.seeds):
        raise ValueError('Duplicate evaluation seeds')


def run(cfg):
    validate_common(cfg)
    validate(cfg)
    with tracking(cfg) as wandb_run:
        return run_iterations(cfg, wandb_run)


def run_iterations(cfg, wandb_run=None):
    for key in ('initial_ckpt', 'stats_path', 'manifest'):
        if not cfg[key]:
            raise ValueError(f'Set {key} to an existing file')
    seed_all(cfg.seed)
    output = Path(hydra.utils.to_absolute_path(cfg.output)).resolve()
    manifest = Path(hydra.utils.to_absolute_path(cfg.manifest)).resolve()
    source = Path(hydra.utils.to_absolute_path(cfg.initial_ckpt)).resolve()
    stats = Path(hydra.utils.to_absolute_path(cfg.stats_path)).resolve()
    for input_path in (manifest, source, stats):
        if not input_path.is_file():
            raise FileNotFoundError(f'{input_path}: run train_offline.py first or configure the iterative inputs')
    frozen = OmegaConf.to_container(cfg, resolve=True)
    fixed = cfg.get('updates_per_round') is not None
    coordinate = 'step_0000000' if fixed else 'epoch_0000'
    selection_options = dict(criterion=['Eval/Success_Rate'], budget_unit='updates') if fixed else {}
    if output.exists():
        if not cfg.resume:
            raise FileExistsError(f'{output}: use resume=true for completed-round continuation')
        old = OmegaConf.to_container(OmegaConf.load(output/'config.yaml'), resolve=True)
        # Runs saved before fixed-budget support have none of these opt-in keys.
        # Fill only historical defaults; changed settings must still fail below.
        defaults = dict(collect=True, updates_per_round=None, critic_warmup_updates=10000,
                        eval_every_updates=2000, save_every_updates=2000, log_every_updates=1000,
                        actor_sampling=dict(mode='mixed', demo_fraction=.5, demo_sources=['demonstrations']))
        for key, value in defaults.items():
            old.setdefault(key, value)
        old['resume'] = frozen['resume']
        old['stages']['iterative']['resume'] = frozen['stages']['iterative']['resume']
        if old != frozen:
            raise ValueError('Resume configuration changed; create a new output directory')
        state = json.loads((output/'state.json').read_text())
    else:
        output.mkdir(parents=True)
        OmegaConf.save(cfg, output/'config.yaml', resolve=True)
        state = dict(next_round=0, checkpoint=str(source), checkpoint_sha256=digest(source),
                     manifest=str(manifest), manifest_sha256=digest(manifest), stats_sha256=digest(stats))
        write_json(output/'state.json', state)
        if fixed:
            write_json(output/'provenance.json', dict(initial_checkpoint=str(source),
                checkpoint_sha256=digest(source), manifest=str(manifest), manifest_sha256=digest(manifest),
                stats_path=str(stats), stats_sha256=digest(stats), collect=cfg.get('collect', True)))
    if digest(stats) != state['stats_sha256']:
        raise ValueError('Frozen normalizer changed')
    if digest(state['checkpoint']) != state['checkpoint_sha256'] or digest(state['manifest']) != state['manifest_sha256']:
        raise ValueError('Checkpoint or manifest changed since last completed round')
    normalizer = MinMaxNormalizer()
    normalizer.load(str(stats))
    for key, dim in [('action', cfg.model.action_dim), ('state', cfg.model.state_dim)]:
        lo, hi = [np.asarray(normalizer.stats[key][k]) for k in ('min', 'max')]
        if lo.shape != (dim,) or hi.shape != (dim,) or not np.isfinite([lo, hi]).all() or (hi < lo).any():
            raise ValueError(f'Invalid normalization: {key}')
    device = torch.device(cfg.device)
    base = build_base(cfg, device)
    actor = FlowPPOPolicy(base, num_steps=cfg.model.num_inference_steps,
            noise_level=cfg.noise_level, min_std=cfg.min_std, eval_mode=cfg.eval.sampler)

    def load(path):
        cp = torch.load(path, map_location=device, weights_only=True)
        if 'normalizer' in cp and cp['normalizer'] != normalizer.stats:
            raise ValueError('Checkpoint and frozen normalizer disagree')
        if 'config' in cp:
            for section in ('model', 'env'):
                if cp['config'][section] != frozen[section]:
                    raise ValueError(f'Checkpoint {section} configuration differs')
        base.load_state_dict(policy_weights(cp, cfg.weight_key), strict=True)
        base.eval()

    encode = observation_encoder(cfg, actor, normalizer, device)

    from envs.factory import make_env
    from workflows.collection import PrimitiveRecorder
    def evaluate(seeds=None, epoch=0):
        return evaluate_base(cfg, base, normalizer, rd, epoch, seeds=seeds)

    def save(path, epoch, metrics):
        torch.save(dict(model_state_dict=base.state_dict(), normalizer=normalizer.stats,
                        config=frozen, epoch=None if fixed else epoch, metrics=metrics,
                        **(dict(update_step=epoch, budget_unit='updates') if fixed else {}),
                        dataset_manifest=state['manifest']), path)

    for round_index in range(state['next_round'], cfg.rounds):
        load(state['checkpoint'])
        episodes, spec = load_sources(state['manifest'])
        if fixed:
            recorded = {int(item['seed']) for src in spec['sources'] for item in src['episodes']
                        if item.get('seed') is not None}
            if recorded & set(cfg.eval.seeds):
                raise ValueError('Validation seeds overlap recorded manifest collection seeds')
        if spec['env'] != frozen['env']:
            raise ValueError('Dataset environment/preprocessing differs from current config')
        rd = output/f'round_{round_index:03d}'
        rd.mkdir(exist_ok=True)
        (rd/'checkpoints').mkdir(exist_ok=True)
        # Restart an interrupted round from its incumbent; finished episodes are reused.
        # Config and incumbent hashes above prevent mixing incompatible runs.
        # A restarted round replaces its old metric rows, never duplicates them.
        for name in ('metrics.jsonl', 'metrics.csv'):
            (rd/name).unlink(missing_ok=True)
        root_log = output/'metrics.jsonl'
        if root_log.exists():
            rows = [json.loads(line) for line in root_log.read_text().splitlines()]
            root_log.write_text(''.join(json.dumps(row)+'\n' for row in rows if row.get('round') != rd.name))
        baseline = evaluate()
        save(rd/'checkpoints'/f'{coordinate}.pth', 0, baseline)
        save(rd/'checkpoints'/'best.pth', 0, baseline)
        write_selection(rd, None if fixed else 0, baseline, rd/'checkpoints'/'best.pth',
                        stage='iterative', round=rd.name, **selection_options,
                        **(dict(update_step=0) if fixed else {}))
        for destination in (rd, output):
            log_metrics(destination, None if fixed else 0, baseline, stage='iterative', round=rd.name, sampler=cfg.eval.sampler,
                        **(dict(update_step=0, budget_unit='updates') if fixed else {}),
                        checkpoint=str(rd/'checkpoints'/f'{coordinate}.pth'),
                        evaluation=str(evaluation_paths(cfg, rd, 0)[0]))
        write_json(rd/'baseline.json', baseline)
        items = []
        if cfg.get('collect', True):
            actor.eval_mode = cfg.collect_sampler
            holder = []
            def recorder(env):
                obj = PrimitiveRecorder(env)
                holder.append(obj)
                return obj
            env = make_env(cfg, primitive_wrapper=recorder)
            items = []
            try:
                for ep_index in range(cfg.episodes_per_round):
                    seed = cfg.collect_seed_start + round_index*cfg.episodes_per_round + ep_index
                    path = rd/f'episode_{ep_index:06d}.npz'
                    # 一轮对照实验：复用已有 rollout，保留独立输出。
                    reuse_dir = cfg.get("reuse_collection_dir")
                    if reuse_dir:
                        if cfg.rounds != 1:
                            raise ValueError("reuse_collection_dir requires rounds=1")

                        source_episode = (
                            Path(hydra.utils.to_absolute_path(reuse_dir)) / path.name
                        )
                        if not source_episode.is_file():
                            raise FileNotFoundError(source_episode)

                        if not path.exists():
                            shutil.copyfile(source_episode, path)
                        elif digest(path) != digest(source_episode):
                            raise ValueError(f"Reused episode differs: {path}")
                    if path.exists():
                        ep = load_episode(path)
                    else:
                        seed_all(seed)
                        obs, _ = env.reset(seed=seed)
                        done = False
                        calls = 0
                        while not done:
                            with torch.no_grad():
                                action = actor.sample(encode(obs), cfg.model.num_inference_steps)[0].cpu().numpy()
                            if not np.isfinite(action).all():
                                raise ValueError('Nonfinite policy action')
                            # Stay inside original normalization range to retain exact labels.
                            physical = normalizer.unnormalize(np.clip(action, -1., 1.), 'action')
                            obs, _, terminated, truncated, _ = env.step(physical)
                            done = bool(terminated or truncated)
                            calls += 1
                            if calls > cfg.env.max_episode_steps:
                                raise RuntimeError('Environment did not end within configured horizon')
                        ep = holder[0].episode()
                        save_episode(path, ep)
                    items.append(dict(path=str(path), sha256=digest(path), seed=seed,
                                      success=bool(ep['success'].any()), steps=len(ep['action'])))
                    print(f'Round {round_index} collected {ep_index+1}/{cfg.episodes_per_round}', flush=True)
            finally:
                env.close()
            # Rebase paths; previous manifests/data are never modified.
            parent = Path(state['manifest']).parent
            for src in spec['sources']:
                for item in src['episodes']:
                    item['path'] = str((parent/item['path']).resolve())
            spec['sources'].append(dict(name=f'round_{round_index:03d}',
                checkpoint=state['checkpoint'], checkpoint_sha256=state['checkpoint_sha256'],
                sampler=cfg.collect_sampler, episodes=items))
            new_manifest = rd/'manifest.json'
            write_json(new_manifest, spec)
            episodes, spec = load_sources(new_manifest)
        else:
            new_manifest = Path(state['manifest'])
        seed_all(cfg.seed + round_index)
        load(state['checkpoint'])
        # Candidate checkpoints point to the dataset actually used in this round.
        previous_manifest = state['manifest']
        state['manifest'] = str(new_manifest)
        selected = train_round(cfg, base, normalizer, episodes, rd, evaluate, save, baseline,
            log_callback=(lambda metrics, epoch: wandb_run.log(metrics,
                step=round_index*((cfg.updates_per_round if fixed else cfg.epochs)+1)+epoch)) if wandb_run else None,
            source_spec=spec, sampling_seed=cfg.seed + round_index)
        state['checkpoint'] = selected or str(rd/'checkpoints'/f'{coordinate}.pth')
        state.update(next_round=round_index+1, checkpoint_sha256=digest(state['checkpoint']),
                     manifest_sha256=digest(new_manifest))
        write_json(rd/'collection.json', dict(selected=state['checkpoint'], improved=selected is not None,
            prior_manifest=previous_manifest, collected=cfg.get('collect', True),
            collection_success_rate=float(np.mean([x['success'] for x in items])) if items else None))
        write_json(output/'state.json', state)
    load(state['checkpoint'])
    (output/'checkpoints').mkdir(exist_ok=True)
    selected_cp = torch.load(state['checkpoint'], map_location='cpu', weights_only=True)
    # Persist the policy used by the final round, with its original validation metrics.
    final = output/'checkpoints'/'best.pth'
    selected_cp['model_state_dict'] = base.state_dict()
    selected_cp.pop('ema_model_state_dict', None)
    selected_cp['config'] = frozen
    selected_cp['normalizer'] = normalizer.stats
    torch.save(selected_cp, final)
    shutil.copyfile(final, output/'checkpoints'/'last.pth')
    write_selection(output, selected_cp.get('epoch', -1), selected_cp.get('metrics', {}), final,
        stage='iterative', source=state['checkpoint'], source_sha256=state['checkpoint_sha256'],
        **selection_options, **(dict(update_step=selected_cp['update_step']) if fixed else {}))
    normalizer.save(str(output/'checkpoints'/'dataset_stats.json'))
    print(f'Selected policy: {final}', flush=True)
