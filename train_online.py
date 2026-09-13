"""Conservative online Flow regression with paired baseline evaluation.

This is AWR-style weighted Flow Matching, NOT PPO or exact maximum likelihood.
"""
import json
import os
from datetime import datetime
import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from models.policy import EmbodiedGenPolicy
from models.online_policy import FlowRegressionPolicy
from models.critics.q_v_network import VNetwork
from algos.pg import FlowPolicyGradient
from utils.normalizer import MinMaxNormalizer
from utils.debug_logger import DebugLogger
from utils.online_env import make_env_ManiSkill
from utils.online_eval import evaluate_policy, seed_all


@hydra.main(version_base=None, config_path='configs', config_name='train_online')
def main(cfg: DictConfig):
    if min(cfg.algo.steps_per_epoch, cfg.algo.update_epochs, cfg.algo.critic_update_epochs,
           cfg.batch_size, cfg.eval.every, cfg.save_epoch, cfg.epochs) < 1:
        raise ValueError('Rollout, epoch, minibatch and evaluation counts must be positive')
    seed_all(cfg.seed)
    device = torch.device(cfg.device)
    ckpt = os.path.abspath(cfg.algo.pretrained_ckpt)
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f'Pretrained checkpoint required: {ckpt}')
    stats_path = os.path.join(os.path.dirname(ckpt), 'dataset_stats.json')
    normalizer = MinMaxNormalizer()
    normalizer.load(stats_path)  # fail early instead of silently training unnormalized
    for key, dim in [('action', cfg.model.action_dim), ('state', cfg.model.state_dim)]:
        stats = normalizer.stats.get(key, {})
        lo, hi = np.asarray(stats.get('min', [])), np.asarray(stats.get('max', []))
        if lo.shape != (dim,) or hi.shape != (dim,) or not np.isfinite([lo, hi]).all() or (hi < lo).any():
            raise ValueError(f'Invalid {key} normalization statistics for dimension {dim}')

    run_name = f'{cfg.run_name}_{datetime.now():%Y%m%d_%H%M%S}'
    save_dir = os.path.join(cfg.save_dir, run_name)
    ckpt_dir = os.path.join(save_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(save_dir, 'config.yaml'), resolve=True)
    normalizer.save(os.path.join(ckpt_dir, 'dataset_stats.json'))
    logger = DebugLogger(save_dir)
    wandb_run = None
    if cfg.wandb.enable:
        import wandb
        wandb_run = wandb.init(project=cfg.wandb.project, name=run_name,
                              config=OmegaConf.to_container(cfg, resolve=True))

    base_policy = EmbodiedGenPolicy(
        in_channels=cfg.model.in_channels, action_dim=cfg.model.action_dim,
        chunk_size=cfg.model.chunk_size, use_state=cfg.model.use_state,
        state_dim=cfg.model.state_dim, encoder_type=cfg.model.encoder_type,
        backbone_type=cfg.model.backbone_type, cond_dim=cfg.model.cond_dim,
        algo_type=cfg.model.algo_type).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=True)
    base_policy.load_state_dict(state.get('ema_model_state_dict', state.get('model_state_dict', state)))
    actor = FlowRegressionPolicy(base_policy).to(device)
    # Freeze pretrained encoder and reuse its exact cached features for both heads.
    critic = VNetwork(state_dim=cfg.model.cond_dim).to(device)
    trainer = FlowPolicyGradient(
        actor, critic, actor_lr=cfg.algo.actor_lr, critic_lr=cfg.algo.critic_lr,
        gamma=cfg.algo.gamma, gae_lambda=cfg.algo.gae_lambda,
        v_loss_coef=cfg.algo.v_loss_coef, max_grad_norm=cfg.algo.max_grad_norm,
        device=device, anchor_coef=cfg.algo.anchor_coef,
        adv_temperature=cfg.algo.adv_temperature, max_weight=cfg.algo.max_weight)
    bounds = np.asarray(cfg.env.workspace_bounds)

    def encode(obs):
        pc = normalizer.center_point_cloud(obs['point_cloud'], bounds)
        state = normalizer.normalize(obs['state'], 'state')
        return actor.encode(torch.as_tensor(pc, dtype=torch.float32, device=device)[None],
                            torch.as_tensor(state, dtype=torch.float32, device=device)[None])

    def evaluate():
        return evaluate_policy(lambda: make_env_ManiSkill(cfg), actor, encode,
                               normalizer, list(cfg.eval.seeds), cfg.model.num_inference_steps)

    def log(epoch, metrics):
        logger.log_metrics(epoch, metrics)
        if wandb_run is not None:
            wandb_run.log(metrics, step=epoch)

    best_score = None
    def save_best(epoch, metrics):
        nonlocal best_score
        score = (metrics['Eval/Success_Rate'], metrics['Eval/Mean_Reward'])
        if best_score is None or score > best_score:
            best_score = score
            # Raw policy state is directly compatible with evaluate.py.
            torch.save(base_policy.state_dict(), os.path.join(ckpt_dir, 'online_best.pth'))
            with open(os.path.join(ckpt_dir, 'best_metrics.json'), 'w') as f:
                json.dump({'epoch': epoch, **metrics}, f, indent=2)

    env = None
    try:
        baseline = evaluate()
        log(0, baseline)
        save_best(0, baseline)  # preserve offline baseline if all updates get worse
        torch.save(base_policy.state_dict(), os.path.join(ckpt_dir, 'offline_baseline.pth'))
        print(f'Offline baseline: {baseline}', flush=True)
        env = make_env_ManiSkill(cfg)
        obs, _ = env.reset(seed=cfg.seed)
        features = encode(obs)
        episode_success = False  # persists across rollout/epoch boundaries
        for epoch in range(1, cfg.epochs + 1):
            actor.eval()
            critic.eval()
            data = {key: [] for key in ('features', 'actions', 'rewards', 'values',
                                       'next_values', 'dones', 'terminated', 'lengths')}
            epoch_reward, successes, episodes, env_steps = 0., 0, 0, 0
            for step in tqdm(range(cfg.algo.steps_per_epoch), desc=f'Epoch {epoch} rollout'):
                with torch.no_grad():
                    value = critic(features).item()
                    action = actor.sample(features, cfg.model.num_inference_steps)[0].cpu().numpy()

                # 注入探索扰动，打破策略的确定性
                noise_scale = cfg.algo.get("explore_noise", 0.02)  
                if noise_scale > 0:
                    exploration_noise = np.random.normal(scale=noise_scale, size=action.shape)
                    # 因为 action 处于 [-1, 1] 的归一化空间，加上噪声后需要安全裁剪
                    action = np.clip(action + exploration_noise, -1.0, 1.0) 

                logger.log_io(epoch, step, {'features': features}, action)
                nxt, reward, terminated, truncated, info = env.step(normalizer.unnormalize(action, 'action'))
                terminated, truncated = bool(terminated), bool(truncated)
                done = terminated or truncated
                next_features = encode(nxt)  # final observation, BEFORE reset
                with torch.no_grad():
                    next_value = 0. if terminated else critic(next_features).item()
                length = int(info['actual_steps'])
                # Correct the executed prefix target to include both normalization
                # clipping and the physical Box bounds applied by the wrapper.
                target = action.copy()
                lo = np.asarray(normalizer.stats['action']['min'])
                hi = np.asarray(normalizer.stats['action']['max'])
                executed = np.asarray(info['executed_actions']).reshape(length, cfg.model.action_dim)
                target[:length] = 2 * (executed - lo) / (hi - lo + normalizer.eps) - 1
                row = dict(features=features[0].cpu().clone(), actions=target,
                           rewards=float(reward), values=value, next_values=next_value,
                           dones=float(done), terminated=float(terminated), lengths=length)
                for key, val in row.items():
                    data[key].append(val)
                epoch_reward += float(reward)
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
            batch = {k: (torch.stack(v).to(device) if k == 'features' else
                         torch.as_tensor(np.asarray(v), dtype=torch.float32, device=device))
                     for k, v in data.items()}
            batch['lengths'] = batch['lengths'].long()
            adv, returns = trainer.compute_gae(batch['rewards'], batch['values'], batch['dones'],
                                              batch['next_values'], batch['terminated'])
            weights = trainer.advantage_weights(adv)  # preserve uncentered sign
            critic_losses, actor_losses = [], []
            size = len(returns)
            for _ in range(cfg.algo.critic_update_epochs):
                for idx in torch.randperm(size, device=device).split(cfg.batch_size):
                    critic_losses.append(trainer.update_critic(batch['features'][idx], returns[idx])['loss/critic'])
            if epoch > cfg.algo.critic_warmup_epochs:
                for _ in range(cfg.algo.update_epochs):
                    for idx in torch.randperm(size, device=device).split(cfg.batch_size):
                        actor_losses.append(trainer.update_actor(batch['features'][idx], batch['actions'][idx],
                                                                  weights[idx], batch['lengths'][idx]))
            with torch.no_grad():
                predicted = critic(batch['features']).squeeze(-1)
                var = returns.var(unbiased=False)
                ev = 1 - (returns - predicted).var(unbiased=False) / (var + 1e-8)
            metrics = {'Env/Reward': epoch_reward, 'Env/Success_Count': successes,
                       'Env/Episodes_Done': episodes, 'Env/Steps': env_steps,
                       'Env/Success_Rate': successes / episodes if episodes else float('nan'),
                       'Value/Mean_V_Pred': batch['values'].mean().item(),
                       'Value/Mean_Return': returns.mean().item(), 'value/explained_var': ev.item(),
                       'loss/critic': float(np.mean(critic_losses)),
                       'awr/positive_fraction': (weights > 0).float().mean().item(),
                       'awr/actor_updates': sum(x['awr/actor_updates'] for x in actor_losses),
                       'awr/critic_warmup': float(epoch <= cfg.algo.critic_warmup_epochs)}
            for key in ('loss/actor', 'awr/flow_mse', 'awr/anchor_mse'):
                active = [x[key] for x in actor_losses if x['awr/actor_updates']]
                metrics[key] = float(np.mean(active)) if active else 0.
            if epoch % cfg.eval.every == 0 or epoch == cfg.epochs:
                result = evaluate()
                metrics.update(result)
                metrics['Eval/Delta_Success'] = result['Eval/Success_Rate'] - baseline['Eval/Success_Rate']
                save_best(epoch, result)
                print(f'Epoch {epoch} evaluation: {result}', flush=True)
            log(epoch, metrics)
            print(f"Epoch {epoch:03d} reward={epoch_reward:.1f} success={successes}/{episodes} "
                  f"actor_updates={metrics['awr/actor_updates']} flow_mse={metrics['awr/flow_mse']:.5f}")
            if epoch % cfg.save_epoch == 0 or epoch == cfg.epochs:
                torch.save(base_policy.state_dict(), os.path.join(ckpt_dir, f'online_ep{epoch}.pth'))
    finally:
        if env is not None:
            env.close()
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == '__main__':
    main()
