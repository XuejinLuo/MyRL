"""ManiSkill Flow fine-tuning using RL-100-style generation-chain PPO."""
import json
import copy
from importlib.metadata import version, PackageNotFoundError
import os
import time
from datetime import datetime
import numpy as np
import torch
import hydra
from omegaconf import OmegaConf
from models.factory import build_base, observation_encoder, batched_observation_encoder
from models.critics.q_v_network import VNetwork
from algos.pg import FlowPPO, normalize_advantages
from utils.normalizer import MinMaxNormalizer
from utils.experiment import evaluation_paths, log_metrics, write_selection, selection_score
from evaluation.runner import evaluate_policy, seed_all
from models.checkpoint import (FORMAT, policy_weights, make_actor, payload,
                               load_checkpoint, checkpoint_primitives)
from envs.online_task import task_protocol, check_resume_protocol, wrap_training_env
from utils.online_diagnostics import value_metrics
from envs.online_vector import SingleOnlineEnv, GPUOnlineEnv
from workflows.online_rollout import OnlineCollector
from workflows.online_mc import CompleteEpisodeCollector, return_protocol, check_return_resume
from utils.experiment import write_json
from data.episodes import digest


def online_selection_score(result, protocol):
    if protocol['selection'] == 'success_only':
        return (result['Eval/Success_Rate'],)
    return selection_score(result)


def atomic_checkpoint(state, path):
    """An interrupted write must not destroy the previous resumable checkpoint."""
    temporary = path + '.tmp'
    torch.save(checkpoint_primitives(state), temporary)
    os.replace(temporary, path)


def runtime_protocol(env):
    try:
        maniskill = version('mani_skill')
    except PackageNotFoundError:
        maniskill = None
    base = env.unwrapped
    return dict(maniskill_version=maniskill,
                environment_class=f'{type(base).__module__}.{type(base).__name__}',
                actual_reward_mode=getattr(base, 'reward_mode', None))


def validate(cfg):
    rollout = cfg.get('online_rollout', {})
    backend, n = rollout.get('backend', 'cpu'), rollout.get('num_envs', 1)
    if backend not in ('cpu', 'gpu') or n < 1 or (backend == 'cpu' and n != 1):
        raise ValueError('Use cpu with num_envs=1, or gpu with num_envs>=1')
    return_protocol(cfg)
    if backend == 'gpu':
        from data.observations import observation_mode, relational_enabled
        if (cfg.env.get('env_id') != 'StackCube-v1' or cfg.env.obs_mode != 'pointcloud'
                or observation_mode(cfg.env) not in ('global_random', 'global_object_budget')
                or relational_enabled(cfg.env)):
            raise ValueError('GPU online currently supports StackCube global pointcloud observations without relational features')
        if (cfg.online_task.reward_mode not in ('success_once', 'success_potential')
                or cfg.online_task.timeout_semantics != 'finite_horizon'):
            raise ValueError('GPU rollout requires success reward and finite_horizon')
    training = cfg.get('online_training', {})
    if training.get('eval_worker_retries', 1) < 0:
        raise ValueError('eval_worker_retries must be nonnegative')
    for key in ('total_env_steps', 'eval_every_env_steps'):
        if training.get(key) is not None and training[key] < 1:
            raise ValueError(f'{key} must be positive or null')
    if not 0 < training.get('target_success_rate', .95) <= 1:
        raise ValueError('target_success_rate must be in (0, 1]')
    for value in (cfg.algo.steps_per_epoch, cfg.algo.update_epochs, cfg.batch_size,
                  cfg.eval.every, cfg.save_epoch, cfg.epochs, cfg.model.num_inference_steps):
        if value < 1:
            raise ValueError('All rollout, update and evaluation counts must be positive')
    if cfg.model.algo_type != 'flow':
        raise ValueError('This migration implements Flow PPO; select model=flow_3d')
    if not 1 <= cfg.env.exec_steps <= cfg.model.chunk_size:
        raise ValueError('Require 1 <= exec_steps <= chunk_size')
    if cfg.algo.ratio_scope not in ('full', 'prefix', 'final_prefix'):
        raise ValueError('ratio_scope must be full, prefix or final_prefix')
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
    warmup_updates = cfg.algo.get('critic_warmup_update_epochs')
    if warmup_updates is not None and warmup_updates < 1:
        raise ValueError('critic_warmup_update_epochs must be positive or null')
    for key in ('trace_max_episodes', 'trace_max_steps_per_episode'):
        if cfg.get('online_diagnostics', {}).get(key, 0) < 0:
            raise ValueError(f'{key} must be nonnegative')
    diag = cfg.get('online_diagnostics', {})
    if diag.get('policy_probe_every', 0) < 0 or diag.get('policy_probe_samples', 32) < 1:
        raise ValueError('Invalid policy probe interval/sample count')
    if cfg.algo.reward_scale <= 0:
        raise ValueError('reward_scale must be positive')


def run(cfg, env_factory=None):
    if cfg.get('stage'):
        from utils.config import validate_common
        validate_common(cfg)
    validate(cfg)
    protocol = task_protocol(cfg)
    returns_protocol = return_protocol(cfg)
    rollout = cfg.get('online_rollout', {})
    training = cfg.get('online_training', {})
    budget = training.get('total_env_steps')
    eval_interval = training.get('eval_every_env_steps')
    keep_epochs = training.get('keep_epoch_checkpoints', True)
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
    checkpoint = load_checkpoint(source, map_location=device)
    if cfg.resume and (checkpoint.get('format') != FORMAT or not all(k in checkpoint for k in ('critic', 'actor_optimizer', 'critic_optimizer', 'total_env_steps'))):
        raise ValueError('resume requires an online epoch_*.pth or last.pth with optimizer state; best.pth is for initialization/evaluation')
    if cfg.resume:
        check_resume_protocol(checkpoint, protocol)
        check_return_resume(checkpoint, cfg)
    if 'config' in checkpoint:
        for section in ('model', 'env'):
            if checkpoint['config'][section] != OmegaConf.to_container(cfg[section], resolve=True):
                raise ValueError(f'Pretrained checkpoint {section} configuration differs')
    if cfg.resume and checkpoint.get('format') == FORMAT:
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
        if list(saved['eval']['seeds']) != list(cfg.eval.seeds):
            raise ValueError('Checkpoint selection seeds mismatch; initialize a new run to change validation seeds')
    normalizer = MinMaxNormalizer()
    normalizer_source = source + ':embedded'
    if 'normalizer' in checkpoint:
        normalizer.stats = checkpoint['normalizer']
    else:
        stats_path = cfg.algo.stats_path or os.path.join(os.path.dirname(source), 'dataset_stats.json')
        normalizer_source = hydra.utils.to_absolute_path(stats_path)
        normalizer.load(normalizer_source)
    for key, dim in [('action', cfg.model.action_dim), ('state', cfg.model.state_dim)]:
        stats = normalizer.stats.get(key, {})
        lo, hi = np.asarray(stats.get('min', [])), np.asarray(stats.get('max', []))
        if lo.shape != (dim,) or hi.shape != (dim,) or not np.isfinite([lo, hi]).all() or (hi < lo).any():
            raise ValueError(f'Invalid {key} normalization statistics; expected dimension {dim}')
    base = build_base(cfg, device)
    weight_key = "model_state_dict" if checkpoint.get("format") == FORMAT else cfg.algo.get("pretrained_weight_key", "auto")
    base.load_state_dict(policy_weights(checkpoint, weight_key), strict=True)
    actor = make_actor(base, cfg)
    critic = VNetwork(state_dim=protocol['critic_input_dim']).to(device)
    trainer = FlowPPO(actor, critic, actor_lr=cfg.algo.actor_lr,
        critic_lr=cfg.algo.critic_lr, clip_ratio=cfg.algo.clip_ratio,
        value_clip=cfg.algo.value_clip, max_grad_norm=cfg.algo.max_grad_norm,
        target_kl=cfg.algo.target_kl,
        prefix_steps=cfg.env.exec_steps if cfg.algo.ratio_scope == 'prefix' else None,
        final_prefix_steps=cfg.env.exec_steps if cfg.algo.ratio_scope == 'final_prefix' else None)
    start_epoch, total_steps = 0, 0
    if cfg.resume:
        try:
            critic.load_state_dict(checkpoint['critic'], strict=True)
        except RuntimeError as exc:
            raise ValueError('Critic shape does not match online_protocol; initialize a new run from Actor weights') from exc
        trainer.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        trainer.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        # Resume moments, but honor the explicitly configured learning rates.
        for group in trainer.actor_optimizer.param_groups:
            group['lr'] = cfg.algo.actor_lr
        for group in trainer.critic_optimizer.param_groups:
            group['lr'] = cfg.algo.critic_lr
        start_epoch, total_steps = checkpoint['epoch'], checkpoint['total_env_steps']
        if cfg.epochs <= start_epoch and not cfg.eval_only and not checkpoint.get('evaluation_pending', False):
            raise ValueError('epochs must exceed the resumed epoch')
    if (budget is not None and total_steps >= budget and not cfg.eval_only
            and not (cfg.resume and checkpoint.get('evaluation_pending', False))):
        raise ValueError('total_env_steps must exceed the resumed total')
    encode = observation_encoder(cfg, actor, normalizer, device)
    encode_many = (batched_observation_encoder(cfg, actor, normalizer, device)
                   if rollout.get('backend', 'cpu') == 'gpu' else
                   lambda observations: torch.cat([encode(obs) for obs in observations], dim=0))

    def evaluate(mode, epoch=0):
        previous = actor.eval_mode
        actor.eval_mode = mode
        try:
            destination, video = evaluation_paths(cfg, run_dir, epoch, sampler=mode)
            metadata = dict(stage='online', round=None, split='validation', epoch=epoch, sampler=mode,
                checkpoint=os.path.join(run_dir, 'checkpoints', f'epoch_{epoch:04d}.pth' if keep_epochs else 'last.pth'),
                env=OmegaConf.to_container(cfg.env, resolve=True), training_protocol=protocol,
                benchmark_protocol={**benchmark_protocol, 'sampler': mode})
            if rollout.get('backend', 'cpu') == 'gpu':
                from evaluation.online_worker import isolated_evaluate
                return isolated_evaluate(cfg, base.state_dict(), normalizer.stats,
                                         mode, destination, video, metadata)
            factory = env_factory if custom_env_factory else lambda: make_env(cfg, video=video)
            return evaluate_policy(factory, actor, encode, normalizer,
                list(cfg.eval.seeds), cfg.model.num_inference_steps, output_dir=destination,
                metadata=metadata)
        finally:
            actor.eval_mode = previous

    benchmark_protocol = dict(adapter='none', success_definition='any primitive step reports success',
        horizon=protocol['horizon'], sampler=cfg.eval.sampler,
        num_inference_steps=int(cfg.model.num_inference_steps),
        noise_level=float(cfg.algo.noise_level), min_std=float(cfg.algo.min_std))
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
            checkpoint=os.path.join(ckpt_dir, f'epoch_{epoch:04d}.pth' if keep_epochs else 'last.pth') if has_checkpoint else None,
            evaluation=str(evaluation_paths(cfg, run_dir, epoch)[0]) if has_eval else None)
        if wandb_run:
            wandb_run.log(metrics, step=epoch)
        print(json.dumps({'epoch': epoch, **metrics}, ensure_ascii=False), flush=True)

    best_score, best_metrics, env = None, None, None
    best_state = None
    runtime = None
    source_hash = digest(source)
    provenance = dict(source_checkpoint=source, source_sha256=source_hash,
        normalizer_source=normalizer_source, resume=bool(cfg.resume),
        initial_actor=(checkpoint.get('provenance', {}).get('initial_actor') if cfg.resume else None)
                      or dict(path=source, sha256=source_hash),
        simulator_resume='reset at epoch boundary')

    def snapshot(epoch, metrics):
        state = payload(base, cfg, normalizer, epoch, metrics)
        state.update(online_protocol=protocol, benchmark_protocol=benchmark_protocol,
                     runtime_protocol=runtime, return_protocol=returns_protocol, provenance=provenance)
        return state

    def save_best(epoch, result):
        nonlocal best_score, best_metrics, best_state
        score = online_selection_score(result, protocol)
        if best_score is None or score > best_score:
            best_score, best_metrics = score, result
            best_state = copy.deepcopy(snapshot(epoch, result))
            best_state['model_state_dict'] = {k: v.cpu() for k, v in best_state['model_state_dict'].items()}
            atomic_checkpoint(best_state, os.path.join(ckpt_dir, 'best.pth'))
            write_selection(run_dir, epoch, result, os.path.join(ckpt_dir, 'best.pth'), stage='online',
                criterion=['Eval/Success_Rate'] if protocol['selection'] == 'success_only' else None)

    def save_epoch(epoch, metrics, evaluation_pending=False):
        state = snapshot(epoch, metrics)
        state.update({'critic': critic.state_dict(), 'actor_optimizer': trainer.actor_optimizer.state_dict(),
            'critic_optimizer': trainer.critic_optimizer.state_dict(), 'total_env_steps': total_steps,
            'selection_best': best_state, 'evaluation_pending': evaluation_pending})
        if keep_epochs:
            atomic_checkpoint(state, os.path.join(ckpt_dir, f'epoch_{epoch:04d}.pth'))
        atomic_checkpoint(state, os.path.join(ckpt_dir, 'last.pth'))

    try:
        # Benchmark factory stays unadapted; only the training env receives this wrapper.
        if rollout.get('backend', 'cpu') == 'gpu':
            if custom_env_factory:
                raise ValueError('Custom single-environment factory cannot be used with GPU rollout')
            env = GPUOnlineEnv(cfg, protocol)
        else:
            raw_env = (env_factory() if custom_env_factory else
                       make_env(cfg, **({'reward_mode': protocol['environment_reward_mode']}
                                       if protocol['environment_reward_mode'] is not None else {})))
            env = SingleOnlineEnv(wrap_training_env(raw_env, protocol))
        runtime = runtime_protocol(env)
        if rollout.get('backend', 'cpu') == 'gpu':
            runtime.update(training_backend='physx_cuda', training_num_envs=env.num_envs)
        if cfg.resume and checkpoint.get('runtime_protocol') not in (None, runtime):
            raise ValueError('Runtime environment/reward mode differs from resumed checkpoint')
        write_json(os.path.join(run_dir, 'protocol.json'), dict(training=protocol,
            benchmark=benchmark_protocol, runtime=runtime, returns=returns_protocol, **provenance))
        print(json.dumps(dict(training_protocol=protocol, runtime=runtime,
                              return_protocol=returns_protocol)), flush=True)
        if cfg.resume and checkpoint.get('selection_best'):
            best_state = checkpoint['selection_best']
            best_state['model_state_dict'] = {k: v.cpu() for k, v in best_state['model_state_dict'].items()}
            best_metrics = best_state['metrics']
            best_score = online_selection_score(best_metrics, protocol)
            atomic_checkpoint(best_state, os.path.join(ckpt_dir, 'best.pth'))
            write_selection(run_dir, best_state['epoch'], best_metrics, os.path.join(ckpt_dir, 'best.pth'),
                stage='online', criterion=['Eval/Success_Rate'] if protocol['selection'] == 'success_only' else None)
        # Persist even the current resume state BEFORE launching a fallible worker.
        save_epoch(start_epoch, checkpoint.get('metrics', {}) if cfg.resume else {}, evaluation_pending=True)
        baseline = evaluate(cfg.eval.sampler, start_epoch)
        log(start_epoch, baseline)
        save_best(start_epoch, baseline)
        save_epoch(start_epoch, baseline)
        atomic_checkpoint(snapshot(start_epoch, baseline), os.path.join(ckpt_dir, 'initial_policy.pth'))
        if start_epoch >= cfg.epochs or (budget is not None and total_steps >= budget):
            # Recover a failed final evaluation without taking another optimizer step.
            target = float(training.get('target_success_rate', .95))
            write_json(os.path.join(run_dir, 'goal_status.json'), dict(
                target_success_rate=target, validation_success_rate=baseline['Eval/Success_Rate'],
                validation_target_met=baseline['Eval/Success_Rate'] >= target,
                best_validation_success_rate=best_metrics['Eval/Success_Rate'],
                total_env_steps=total_steps, final=True, independent_test_required=True))
            return {'run_dir': run_dir, 'best_metrics': best_metrics, 'total_env_steps': total_steps}
        # Resume resets the simulator at an epoch boundary, not an exact trajectory continuation.
        collector_class = CompleteEpisodeCollector if returns_protocol['estimator'] == 'mc' else OnlineCollector
        collector = collector_class(env, actor, critic, encode_many, normalizer, cfg, protocol,
                                    os.path.join(run_dir, 'diagnostics'), cfg.seed + start_epoch)
        last_eval_steps = total_steps
        for epoch in range(start_epoch + 1, cfg.epochs + 1):
            batch, adv, returns, rollout_metrics = collector.collect(epoch,
                None if budget is None else budget - total_steps)
            env_steps = rollout_metrics['Env/Steps']
            # Freeze one target for both diagnostics; after is training-set fit.
            frozen_returns = returns.detach().clone()
            before = value_metrics(batch['values'], frozen_returns)
            actor_enabled = epoch > cfg.algo.critic_warmup_epochs
            update_epochs = cfg.algo.update_epochs if actor_enabled else (
                cfg.algo.get('critic_warmup_update_epochs') or cfg.algo.update_epochs)
            normalized = normalize_advantages(adv)
            collector.attach_advantages(adv, normalized, actor_enabled)
            update_started = time.perf_counter()
            probe = None
            probe_every = cfg.get('online_diagnostics', {}).get('policy_probe_every', 0)
            if actor_enabled and probe_every and epoch % probe_every == 0:
                from utils.policy_probe import PolicyUpdateProbe
                probe = PolicyUpdateProbe(trainer, batch, adv, cfg.env.exec_steps,
                    cfg.online_diagnostics.get('policy_probe_samples', 32))
            metrics = trainer.update(batch, adv, frozen_returns, cfg.batch_size, update_epochs,
                actor_enabled=actor_enabled, verify=cfg.algo.verify_logprobs)
            if probe is not None:
                metrics.update(probe.finish(normalizer))
            metrics['Perf/Update_Seconds'] = time.perf_counter() - update_started
            total_steps += env_steps
            with torch.no_grad():
                predicted = critic(batch['critic_features']).squeeze(-1)
            after = value_metrics(predicted, frozen_returns)
            if cfg.get('online_diagnostics', {}).get('value_before_after', True):
                for label, values in (('Before', before), ('After', after)):
                    metrics.update({f'Value/{label}/{key}': val for key, val in values.items()})
            metrics['Value/Target_Is_MC'] = int(returns_protocol['estimator'] == 'mc')
            if returns_protocol['estimator'] == 'mc':
                # Before = prediction on newly collected complete episodes;
                # After = fit on this same training batch, not held-out accuracy.
                starts = batch['episode_starts']
                for label, predictions in (('Before', batch['values']), ('After', predicted)):
                    metrics.update({f'Value/MC/{label}/{key}': val for key, val in
                                    value_metrics(predictions, frozen_returns).items()})
                    metrics.update({f'Value/MC_Start/{label}/{key}': val for key, val in
                                    value_metrics(predictions[starts], frozen_returns[starts]).items()})
            metrics.update(rollout_metrics)
            metrics.update({'Env/Total_Steps': total_steps,
                'Value/Explained_Variance': after['Explained_Variance'],
                'Value/Return_Mean': frozen_returns.mean().item(),
                'Adv/Mean': adv.mean().item(), 'Adv/Std': adv.std(unbiased=False).item(),
                'Adv/Negative_Fraction': (adv < 0).float().mean().item(),
                'Adv/Normalized_Mean': normalized.mean().item(),
                'Adv/Normalized_Std': normalized.std(unbiased=False).item(),
                'Adv/Normalized_Positive_Fraction': (normalized > 0).float().mean().item(),
                'Adv/Actor_Enabled': actor_enabled})
            final = epoch == cfg.epochs or (budget is not None and total_steps >= budget)
            metrics.update(collector.flush(epoch, final=final))
            do_eval = final or (total_steps - last_eval_steps >= eval_interval if eval_interval else epoch % cfg.eval.every == 0)
            if do_eval:
                save_epoch(epoch, metrics, evaluation_pending=True)
                try:
                    result = evaluate(cfg.eval.sampler, epoch)
                except Exception:
                    print(f'Evaluation failed after epoch {epoch}; training state is saved at '
                          f'{os.path.join(ckpt_dir, "last.pth")}. Resume from this checkpoint '
                          'with a new output directory after fixing the worker error.', flush=True)
                    raise
                metrics.update(result)
                metrics['Eval/Delta_Success'] = result['Eval/Success_Rate'] - baseline['Eval/Success_Rate']
                save_best(epoch, result)
                last_eval_steps = total_steps
                target = float(training.get('target_success_rate', .95))
                write_json(os.path.join(run_dir, 'goal_status.json'), dict(
                    target_success_rate=target, validation_success_rate=result['Eval/Success_Rate'],
                    validation_target_met=result['Eval/Success_Rate'] >= target,
                    best_validation_success_rate=best_metrics['Eval/Success_Rate'],
                    total_env_steps=total_steps, final=final, independent_test_required=True))
            log(epoch, metrics)
            if epoch % cfg.save_epoch == 0 or do_eval or final:
                save_epoch(epoch, metrics)
            if final:
                break
        return {'run_dir': run_dir, 'best_metrics': best_metrics, 'total_env_steps': total_steps}
    finally:
        if env is not None:
            env.close()
        if wandb_run:
            wandb_run.finish()
