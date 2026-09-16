"""Collect -> append immutable sources -> offline IDQL -> validation selection."""
import json
import shutil
from pathlib import Path
import numpy as np
import torch
import hydra
from omegaconf import OmegaConf
from models.policy import EmbodiedGenPolicy
from models.online_policy import FlowPPOPolicy
from utils.normalizer import MinMaxNormalizer
from utils.online_checkpoint import policy_weights
from utils.online_eval import evaluate_policy, seed_all
from data.iterative_store import (SCHEMA, digest, load_sources, load_episode,
                                  save_episode, write_json)
from workflows.offline_round import train_round
from utils.experiment import evaluate_base


def validate(cfg):
    for k in ('rounds', 'episodes_per_round', 'epochs', 'batch_size'):
        if cfg[k] < 1:
            raise ValueError(f'{k} must be positive')
    if not 0 <= cfg.critic_warmup_epochs < cfg.epochs:
        raise ValueError('Require 0 <= critic_warmup_epochs < epochs')
    if cfg.batch_size < 2 or cfg.num_workers < 0 or cfg.eval.every < 1:
        raise ValueError('Invalid loader/evaluation settings')
    if cfg.model.algo_type != 'flow' or not 1 <= cfg.env.exec_steps <= cfg.model.chunk_size:
        raise ValueError('Require Flow policy and valid execution prefix')
    if cfg.collect_sampler not in ('cps', 'ode') or cfg.eval.sampler not in ('cps', 'ode'):
        raise ValueError('Unknown sampler')
    if not 0 <= cfg.ema_decay < 1 or not 0 < cfg.algo.discount <= 1:
        raise ValueError('Invalid EMA/discount')
    validation, test = set(cfg.eval.seeds), set(cfg.test_seeds)
    collection = set(range(cfg.collect_seed_start, cfg.collect_seed_start + cfg.rounds*cfg.episodes_per_round))
    if not validation or not test or validation & test or collection & (validation | test):
        raise ValueError('Collection, validation and test seeds must be nonempty/disjoint')
    if len(validation) != len(cfg.eval.seeds) or len(test) != len(cfg.test_seeds):
        raise ValueError('Duplicate evaluation seeds')


@hydra.main(version_base=None, config_path='configs', config_name='train_iterative')
def main(cfg):
    validate(cfg)
    for key in ('initial_ckpt', 'stats_path', 'manifest'):
        if not cfg[key]:
            raise ValueError(f'Set {key} to an existing file')
    seed_all(cfg.seed)
    output = Path(hydra.utils.to_absolute_path(cfg.output)).resolve()
    manifest = Path(hydra.utils.to_absolute_path(cfg.manifest)).resolve()
    source = Path(hydra.utils.to_absolute_path(cfg.initial_ckpt)).resolve()
    stats = Path(hydra.utils.to_absolute_path(cfg.stats_path)).resolve()
    frozen = OmegaConf.to_container(cfg, resolve=True)
    if output.exists():
        if not cfg.resume:
            raise FileExistsError(f'{output}: use resume=true for completed-round continuation')
        old = json.loads((output/'config.json').read_text())
        old['resume'] = frozen['resume']
        if old != frozen:
            raise ValueError('Resume configuration changed; create a new output directory')
        state = json.loads((output/'state.json').read_text())
    else:
        output.mkdir(parents=True)
        write_json(output/'config.json', frozen)
        OmegaConf.save(cfg, output/'config.yaml', resolve=True)
        state = dict(next_round=0, checkpoint=str(source), checkpoint_sha256=digest(source),
                     manifest=str(manifest), manifest_sha256=digest(manifest), stats_sha256=digest(stats))
        write_json(output/'state.json', state)
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
    base = EmbodiedGenPolicy(**{k: cfg.model[k] for k in (
        'in_channels', 'action_dim', 'chunk_size', 'use_state', 'state_dim',
        'encoder_type', 'backbone_type', 'cond_dim', 'algo_type')}).to(device)
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

    def encode(obs):
        pc = normalizer.center_point_cloud(np.asarray(obs['point_cloud'], dtype=np.float32), np.asarray(cfg.env.workspace_bounds))
        state = normalizer.normalize(np.asarray(obs['state'], dtype=np.float32), 'state')
        return actor.encode(torch.as_tensor(pc, device=device)[None], torch.as_tensor(state, device=device)[None])

    from utils.online_env import make_env_ManiSkill
    from workflows.collection import PrimitiveRecorder
    def evaluate(seeds=None, epoch=0):
        return evaluate_base(cfg, base, normalizer, rd, epoch, seeds=seeds)

    def save(path, epoch, metrics):
        torch.save(dict(model_state_dict=base.state_dict(), normalizer=normalizer.stats,
                        config=frozen, epoch=epoch, metrics=metrics,
                        dataset_manifest=state['manifest']), path)

    for round_index in range(state['next_round'], cfg.rounds):
        load(state['checkpoint'])
        episodes, spec = load_sources(state['manifest'])
        if spec['env'] != frozen['env']:
            raise ValueError('Dataset environment/preprocessing differs from current config')
        rd = output/f'round_{round_index:03d}'
        rd.mkdir(exist_ok=True)
        (rd/'checkpoints').mkdir(exist_ok=True)
        # Restart an interrupted round from its incumbent; finished episodes are reused.
        # Config and incumbent hashes above prevent mixing incompatible runs.
        baseline = evaluate()
        write_json(rd/'baseline.json', baseline)
        actor.eval_mode = cfg.collect_sampler
        holder = []
        def recorder(env):
            obj = PrimitiveRecorder(env)
            holder.append(obj)
            return obj
        env = make_env_ManiSkill(cfg, primitive_wrapper=recorder)
        items = []
        try:
            for ep_index in range(cfg.episodes_per_round):
                seed = cfg.collect_seed_start + round_index*cfg.episodes_per_round + ep_index
                path = rd/f'episode_{ep_index:06d}.npz'
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
        episodes, _ = load_sources(new_manifest)
        seed_all(cfg.seed + round_index)
        load(state['checkpoint'])
        # Candidate checkpoints point to the dataset actually used in this round.
        previous_manifest = state['manifest']
        state['manifest'] = str(new_manifest)
        selected = train_round(cfg, base, normalizer, episodes, rd, evaluate, save, baseline)
        if selected:
            state['checkpoint'] = selected
        state.update(next_round=round_index+1, checkpoint_sha256=digest(state['checkpoint']),
                     manifest_sha256=digest(new_manifest))
        write_json(rd/'selection.json', dict(selected=state['checkpoint'], improved=selected is not None,
            prior_manifest=previous_manifest, collection_success_rate=float(np.mean([x['success'] for x in items]))))
        write_json(output/'state.json', state)
    load(state['checkpoint'])
    final = output/'selected_policy.pth'
    selected_cp = torch.load(state['checkpoint'], map_location='cpu', weights_only=True)
    save(final, selected_cp.get('epoch', -1), selected_cp.get('metrics', {}))
    write_json(output/'selection.json', dict(source=state['checkpoint'],
        source_sha256=state['checkpoint_sha256'], checkpoint=str(output/'checkpoints'/'best.pth'),
        weight_key='model_state_dict', metrics=selected_cp.get('metrics', {})))
    normalizer.save(str(output/'dataset_stats.json'))
    # Test is evaluated only after all selection; never feeds back into the loop.
    (output/'checkpoints').mkdir(exist_ok=True)
    shutil.copyfile(final, output/'checkpoints'/'best.pth')
    normalizer.save(str(output/'checkpoints'/'dataset_stats.json'))
    if cfg.final_test:
        result = evaluate_base(cfg, base, normalizer, output, cfg.epochs,
                               tag='test', seeds=cfg.test_seeds)
        write_json(output/'test.json', result)
    print(f'Selected policy: {final}', flush=True)


if __name__ == '__main__':
    main()
