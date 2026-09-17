"""ManiSkill Flow fine-tuning using RL-100-style generation-chain PPO."""
import json
import os
from datetime import datetime
import numpy as np
import torch
import hydra
from omegaconf import OmegaConf
from tqdm import tqdm
from models.factory import build_base, observation_encoder
from models.critics.q_v_network import VNetwork
from algos.pg import FlowPPO, compute_gae
from utils.normalizer import MinMaxNormalizer
from utils.experiment import evaluation_paths, log_metrics, write_selection, selection_score
from evaluation.runner import evaluate_policy, seed_all
from models.checkpoint import FORMAT, policy_weights, make_actor, payload


def validate(cfg):
    for value in (cfg.algo.steps_per_epoch, cfg.algo.update_epochs, cfg.batch_size,
                  cfg.eval.every, cfg.save_epoch, cfg.epochs, cfg.model.num_inference_steps):
        if value < 1:
            raise ValueError('All rollout, update and evaluation counts must be positive')
    if cfg.model.algo_type != 'flow':
        raise ValueError('This migration implements Flow PPO; select model=flow_3d')
    if not 1 <= cfg.env.exec_steps <= cfg.model.chunk_size:
        raise ValueError('Require 1 <= exec_steps <= chunk_size')
    if cfg.algo.ratio_scope not in ('full', 'prefix'):
        raise ValueError('ratio_scope must be full or prefix')
    if not 0 < cfg.algo.gamma <= 1 or not 0 <= cfg.algo.gae_lambda <= 1:
        raise ValueError('Invalid gamma/lambda')
    if cfg.algo.actor_lr <= 0 or cfg.algo.critic_lr <= 0 or not 0 < cfg.algo.clip_ratio < 1:
        raise ValueError('Invalid PPO learning rate/clip ratio')
    if cfg.algo.max_grad_norm <= 0 or cfg.algo.critic_warmup_epochs < 0:
        raise ValueError('Invalid gradient clipping/warmup')
    if cfg.algo.value_clip is not None and cfg.algo.value_clip <= 0:
        raise ValueError('value_clip must be positive or null')
    if cfg.algo.target_kl is not None and cfg.algo.target_kl <= 0:
        raise ValueError('target_kl must be positive or null')
    if cfg.algo.reward_scale <= 0:
        raise ValueError('reward_scale must be positive')


def run(cfg, env_factory=None):
    if cfg.get('stage'):
        from utils.config import validate_common
        validate_common(cfg)
    validate(cfg)
    custom_env_factory = env_factory is not None
    if env_factory is None:
        from envs.factory import make_env
        env_factory = lambda: make_env(cfg)
    seed_all(cfg.seed)
    device = torch.device(cfg.device)
    source = cfg.resume or cfg.algo.pretrained_ckpt
    if not source:
        raise ValueError('Set stages.online.initial_ckpt to an existing checkpoint')
    source = hydra.utils.to_absolute_path(source)
    checkpoint = torch.load(source, map_location=device, weights_only=True)
    if cfg.resume and (checkpoint.get('format') != FORMAT or not all(k in checkpoint for k in ('critic', 'actor_optimizer', 'critic_optimizer', 'total_env_steps'))):
        raise ValueError('resume requires an online epoch_*.pth or last.pth with optimizer state; best.pth is for initialization/evaluation')
    if 'config' in checkpoint:
        for section in ('model', 'env'):
            if checkpoint['config'][section] != OmegaConf.to_container(cfg[section], resolve=True):
                raise ValueError(f'Pretrained checkpoint {section} configuration differs')
    if checkpoint.get('format') == FORMAT:
        saved = checkpoint['config']
        for section, keys in {'model': list(cfg.model), 'env': list(cfg.env),
                              'algo': ['noise_level', 'min_std', 'gamma', 'gae_lambda',
                                       'ratio_scope', 'reward_scale']}.items():
            for key in keys:
                current = OmegaConf.to_container(cfg[section], resolve=True).get(key)
                if saved[section].get(key) != current:
                    raise ValueError(f'Checkpoint mismatch: {section}.{key}')
        if saved['eval']['sampler'] != cfg.eval.sampler:
            raise ValueError('Checkpoint eval sampler mismatch')
    normalizer = MinMaxNormalizer()
    if 'normalizer' in checkpoint:
        normalizer.stats = checkpoint['normalizer']
    else:
        stats_path = cfg.algo.stats_path or os.path.join(os.path.dirname(source), 'dataset_stats.json')
        normalizer.load(hydra.utils.to_absolute_path(stats_path))
    for key, dim in [('action', cfg.model.action_dim), ('state', cfg.model.state_dim)]:
        stats = normalizer.stats.get(key, {})
        lo, hi = np.asarray(stats.get('min', [])), np.asarray(stats.get('max', []))
        if lo.shape != (dim,) or hi.shape != (dim,) or not np.isfinite([lo, hi]).all() or (hi < lo).any():
            raise ValueError(f'Invalid {key} normalization statistics; expected dimension {dim}')
    base = build_base(cfg, device)
    weight_key = "model_state_dict" if checkpoint.get("format") == FORMAT else cfg.algo.get("pretrained_weight_key", "auto")
    base.load_state_dict(policy_weights(checkpoint, weight_key), strict=True)
    actor = make_actor(base, cfg)
    critic = VNetwork(state_dim=cfg.model.cond_dim).to(device)
    trainer = FlowPPO(actor, critic, actor_lr=cfg.algo.actor_lr,
        critic_lr=cfg.algo.critic_lr, clip_ratio=cfg.algo.clip_ratio,
        value_clip=cfg.algo.value_clip, max_grad_norm=cfg.algo.max_grad_norm,
        target_kl=cfg.algo.target_kl,
        prefix_steps=cfg.env.exec_steps if cfg.algo.ratio_scope == 'prefix' else None)
    start_epoch, total_steps = 0, 0
    if cfg.resume:
        critic.load_state_dict(checkpoint['critic'])
        trainer.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        trainer.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        # Resume moments, but honor the explicitly configured learning rates.
        for group in trainer.actor_optimizer.param_groups:
            group['lr'] = cfg.algo.actor_lr
        for group in trainer.critic_optimizer.param_groups:
            group['lr'] = cfg.algo.critic_lr
        start_epoch, total_steps = checkpoint['epoch'], checkpoint['total_env_steps']
        if cfg.epochs <= start_epoch and not cfg.eval_only:
            raise ValueError('epochs must exceed the resumed epoch')
    encode = observation_encoder(cfg, actor, normalizer, device)

    def evaluate(mode, epoch=0):
        previous = actor.eval_mode
        actor.eval_mode = mode
        try:
            destination, video = evaluation_paths(cfg, run_dir, epoch, sampler=mode)
            factory = env_factory if custom_env_factory else lambda: make_env(cfg, video=video)
            return evaluate_policy(factory, actor, encode, normalizer,
                list(cfg.eval.seeds), cfg.model.num_inference_steps, output_dir=destination,
                metadata=dict(stage='online', round=None, split='validation', epoch=epoch, sampler=mode,
                              checkpoint=os.path.join(run_dir, 'checkpoints', f'epoch_{epoch:04d}.pth'),
                              env=OmegaConf.to_container(cfg.env, resolve=True)))
        finally:
            actor.eval_mode = previous

    run_dir = hydra.utils.to_absolute_path(cfg.get('output') or os.path.join(
        cfg.save_dir, f'{cfg.run_name}_{datetime.now():%Y%m%d_%H%M%S_%f}'))
    if os.path.exists(run_dir):
        raise FileExistsError(f'Output already exists: {run_dir}')
    if cfg.eval_only:
        results = {mode: evaluate(mode) for mode in ('cps', 'ode')}
        print(json.dumps(results, indent=2), flush=True)
        return results
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    os.makedirs(ckpt_dir)
    normalizer.save(os.path.join(ckpt_dir, 'dataset_stats.json'))
    OmegaConf.save(cfg, os.path.join(run_dir, 'config.yaml'), resolve=True)
    wandb_run = None
    if cfg.wandb.enable:
        import wandb
        wandb_run = wandb.init(project=cfg.wandb.project, entity=cfg.wandb.get('entity'), name=os.path.basename(run_dir),
                               config=OmegaConf.to_container(cfg, resolve=True))

    def log(epoch, metrics):
        has_eval = 'Eval/Success_Rate' in metrics
        has_checkpoint = has_eval or epoch % cfg.save_epoch == 0 or epoch == cfg.epochs
        log_metrics(run_dir, epoch, metrics, stage='online', sampler=cfg.eval.sampler,
            checkpoint=os.path.join(ckpt_dir, f'epoch_{epoch:04d}.pth') if has_checkpoint else None,
            evaluation=str(evaluation_paths(cfg, run_dir, epoch)[0]) if has_eval else None)
        if wandb_run:
            wandb_run.log(metrics, step=epoch)
        print(json.dumps({'epoch': epoch, **metrics}, ensure_ascii=False), flush=True)

    best_score, best_metrics, env = None, None, None
    def save_best(epoch, result):
        nonlocal best_score, best_metrics
        score = selection_score(result)
        if best_score is None or score > best_score:
            best_score, best_metrics = score, result
            torch.save(payload(base, cfg, normalizer, epoch, result), os.path.join(ckpt_dir, 'best.pth'))
            write_selection(run_dir, epoch, result, os.path.join(ckpt_dir, 'best.pth'), stage='online')

    def save_epoch(epoch, metrics):
        state = payload(base, cfg, normalizer, epoch, metrics)
        state.update({'critic': critic.state_dict(), 'actor_optimizer': trainer.actor_optimizer.state_dict(),
            'critic_optimizer': trainer.critic_optimizer.state_dict(), 'total_env_steps': total_steps})
        torch.save(state, os.path.join(ckpt_dir, f'epoch_{epoch:04d}.pth'))
        torch.save(state, os.path.join(ckpt_dir, 'last.pth'))

    try:
        baseline = evaluate(cfg.eval.sampler, start_epoch)
        log(start_epoch, baseline)
        save_epoch(start_epoch, baseline)
        save_best(start_epoch, baseline)
        torch.save(payload(base, cfg, normalizer, start_epoch, baseline), os.path.join(ckpt_dir, 'initial_policy.pth'))
        env = env_factory()
        # Resume resets the simulator at an epoch boundary, not an exact trajectory continuation.
        obs, _ = env.reset(seed=cfg.seed + start_epoch)
        features = encode(obs)
        episode_success = False
        for epoch in range(start_epoch + 1, cfg.epochs + 1):
            data = {key: [] for key in ('features', 'chains', 'logprobs', 'rewards', 'values',
                'next_values', 'terminated', 'dones', 'lengths')}
            reward_sum, successes, episodes, env_steps, clipped_count, action_count = 0., 0, 0, 0, 0, 0
            actor.eval()
            critic.eval()
            for _ in tqdm(range(cfg.algo.steps_per_epoch), desc=f'Epoch {epoch} rollout', disable=cfg.quiet):
                with torch.no_grad():
                    action, chain, logprob = actor.collect(features)
                    value = critic(features).item()
                raw = action[0].cpu().numpy()
                if not np.isfinite(raw).all():
                    raise FloatingPointError('Nonfinite generated action')
                # Clipping is a fixed environment transform. NEVER overwrite latent
                # chain/logprobs with clipped or physically executed actions.
                nxt, reward, terminated, truncated, info = env.step(normalizer.unnormalize(raw, 'action'))
                terminated, truncated = bool(terminated), bool(truncated)
                done = terminated or truncated
                length = int(info['actual_steps'])
                rewards = np.asarray(info['primitive_rewards'], dtype=np.float64)
                if rewards.shape != (length,) or not 1 <= length <= cfg.env.exec_steps:
                    raise ValueError('Invalid primitive reward/length metadata from ChunkActionWrapper')
                discounted_reward = float(np.dot(cfg.algo.gamma ** np.arange(length), rewards)) * cfg.algo.reward_scale
                next_features = encode(nxt)  # BEFORE reset: time-limit bootstrap uses final observation
                with torch.no_grad():
                    next_value = 0. if terminated else critic(next_features).item()
                row = dict(features=features[0].cpu(), chains=chain[0].cpu(), logprobs=logprob[0].cpu(),
                    rewards=discounted_reward, values=value, next_values=next_value,
                    terminated=float(terminated), dones=float(done), lengths=length)
                for key, val in row.items():
                    data[key].append(val)
                clipped_count += int((np.abs(raw[:length]) > 1.1).sum())
                action_count += raw[:length].size
                reward_sum += float(reward)
                env_steps += length
                episode_success |= bool(info.get('success_any', info.get('success', False)))
                if done:
                    successes += int(episode_success)
                    episodes += 1
                    episode_success = False
                    obs, _ = env.reset()
                    features = encode(obs)
                else:
                    features = next_features
            batch = {key: (torch.stack(vals).to(device) if torch.is_tensor(vals[0]) else
                torch.as_tensor(vals, dtype=torch.float32, device=device)) for key, vals in data.items()}
            if not all(torch.isfinite(v).all() for v in batch.values()):
                raise FloatingPointError('Nonfinite rollout')
            adv, returns = compute_gae(batch['rewards'], batch['values'], batch['next_values'],
                batch['terminated'], batch['dones'], batch['lengths'], cfg.algo.gamma, cfg.algo.gae_lambda)
            metrics = trainer.update(batch, adv, returns, cfg.batch_size, cfg.algo.update_epochs,
                actor_enabled=epoch > cfg.algo.critic_warmup_epochs, verify=cfg.algo.verify_logprobs)
            total_steps += env_steps
            with torch.no_grad():
                predicted = critic(batch['features']).squeeze(-1)
                ev = 1 - (returns - predicted).var(unbiased=False) / returns.var(unbiased=False).clamp_min(1e-8)
            metrics.update({'Env/Reward': reward_sum, 'Env/Episodes': episodes, 'Env/Success_Count': successes,
                'Env/Success_Rate': successes / episodes if episodes else None,
                'Env/Steps': env_steps, 'Env/Total_Steps': total_steps,
                'Action/Normalization_Clip_Fraction': clipped_count / max(action_count, 1),
                'Value/Explained_Variance': ev.item(), 'Value/Return_Mean': returns.mean().item(),
                'Adv/Mean': adv.mean().item(), 'Adv/Std': adv.std(unbiased=False).item(),
                'Adv/Negative_Fraction': (adv < 0).float().mean().item()})
            if epoch % cfg.eval.every == 0 or epoch == cfg.epochs:
                result = evaluate(cfg.eval.sampler, epoch)
                metrics.update(result)
                metrics['Eval/Delta_Success'] = result['Eval/Success_Rate'] - baseline['Eval/Success_Rate']
                save_best(epoch, result)
            log(epoch, metrics)
            if epoch % cfg.save_epoch == 0 or epoch % cfg.eval.every == 0 or epoch == cfg.epochs:
                save_epoch(epoch, metrics)
        return {'run_dir': run_dir, 'best_metrics': best_metrics, 'total_env_steps': total_steps}
    finally:
        if env is not None:
            env.close()
        if wandb_run:
            wandb_run.finish()
