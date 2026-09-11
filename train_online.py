# train_online.py
import os
import copy
import random
from algos.flow_ppo_policy import FlowPPOPolicy
import torch
import torch.nn as nn
import numpy as np
import hydra
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import wandb
import gymnasium as gym
import mani_skill.envs

# =========================================================================
# 导入你的核心架构组件
# =========================================================================
from models.policy import EmbodiedGenPolicy
from models.encoders.pointnext import PointNeXtEncoder
from models.critics.q_v_network import VNetwork
from algos.pg import FlowPolicyGradient
from envs.pointcloud_wrapper import PointCloudObservationWrapper
from envs.chunk_wrapper import ChunkActionWrapper
from envs.maniskill_bridge import ManiSkillToRL100Wrapper 
from utils.eval_utils import evaluate_and_record_video
from utils.debug_logger import DebugLogger
from utils.normalizer import MinMaxNormalizer
from utils.ppo_checks import configure_ppo_numerics, replay_diagnostics

# =========================================================================
# 桥接层 (Bridge Wrappers)
# PG 更新要求 Tensor 输入，而我们的环境吐出的是 Dict。这里用 Wrapper 进行解包对齐。
# =========================================================================

class CriticFeatureExtractor(nn.Module):
    """ 为 Critic 专门提供 3D 点云特征提取 """
    def __init__(self, cfg):
        super().__init__()
        self.encoder = PointNeXtEncoder(
            in_channels=cfg.model.get("in_channels", 3),
            output_dim=cfg.model.cond_dim,
            use_state=cfg.model.use_state,
            state_dim=cfg.model.state_dim
        )
    def forward(self, obs_dict):
        pn_dict = {
            'point_cloud': obs_dict['pc'],
            'state': obs_dict['state']
        }
        return self.encoder(pn_dict)

class Critic_PG_Wrapper(nn.Module):
    """ 将 Encoder 和 VNetwork 组合，对外暴露单一的 forward 接口 """
    def __init__(self, encoder, v_net):
        super().__init__()
        self.encoder = encoder
        self.v_net = v_net
        
    def forward(self, obs_dict):
        # PG 会调用 self.critic(states)，这里 states 实际上就是 obs_dict
        feat = self.encoder(obs_dict)
        return self.v_net(feat)

def make_env_ManiSkill(cfg):
    """ 创建并包装 ManiSkill 真实仿真环境 """

    env_id = cfg.env.get("env_id", "PushCube-v1")
    obs_mode = cfg.env.get("obs_mode", "pointcloud")
    control_mode = cfg.env.get("control_mode", "pd_ee_delta_pose")
    render_mode = cfg.env.get("render_mode", "rgb_array")
    
    # 1. 实例化 ManiSkill 环境 (以最经典的抓取方块任务为例)
    env = gym.make(
        env_id,
        obs_mode=obs_mode,
        control_mode=control_mode,
        render_mode=render_mode,
        max_episode_steps=int(cfg.env.get("max_episode_steps", 300))
    )
    
    # 2. 接入 ManiSkill 数据适配器 (转换为 {'xyz', 'rgb', 'state'})
    env = ManiSkillToRL100Wrapper(env)
    
    # 3. 接入你原来写好的 PointCloud Wrapper
    bounds = cfg.env.get("workspace_bounds", [[-0.5, -0.5, 0.0], [0.5, 0.5, 0.5]])
    ws_bounds = np.array(bounds)
    
    env = PointCloudObservationWrapper(
        env=env,
        num_points=cfg.env.num_points,  # 比如 1024
        workspace_bounds=ws_bounds,
        use_color=cfg.env.use_color
    )
    
    # 4. 接入你写好的 Action Chunk Wrapper
    env = ChunkActionWrapper(
        env=env,
        chunk_size=cfg.model.chunk_size,
        exec_steps=cfg.env.exec_steps,
        exp_weight=cfg.env.exp_weight
    )
    
    return env

class RolloutBuffer:
    def __init__(self, device):
        self.device = device
        self.rows = []

    def clear(self):
        self.rows.clear()

    def add(self, **row):
        self.rows.append({k: np.array(v, copy=True) for k, v in row.items()})

    def get_tensors(self):
        return {k: torch.as_tensor(np.stack([r[k] for r in self.rows]),
                                  dtype=torch.float32, device=self.device)
                for k in self.rows[0]}


def scalar(x):
    return x.item() if hasattr(x, 'item') else x


@hydra.main(version_base=None, config_path='configs', config_name='train_online')
def main(cfg: DictConfig):
    configure_ppo_numerics()
    seed = int(cfg.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(cfg.device)
    run_name = f'{cfg.run_name}_{datetime.now():%Y%m%d_%H%M%S}'
    OmegaConf.set_struct(cfg, False)
    cfg.run_name = run_name
    cfg.save_dir = os.path.join(cfg.save_dir, run_name)
    OmegaConf.set_struct(cfg, True)
    os.makedirs(cfg.save_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(cfg.save_dir, 'config.yaml'))
    debug_logger = DebugLogger(cfg.save_dir)
    if cfg.wandb.enable:
        wandb.init(project=cfg.wandb.project, name=run_name,
                   config=OmegaConf.to_container(cfg, resolve=True))

    ckpt = cfg.algo.pretrained_ckpt
    if not ckpt or not os.path.isfile(ckpt):
        raise FileNotFoundError(f'Offline checkpoint required: {ckpt}')
    stats = os.path.join(os.path.dirname(ckpt), 'dataset_stats.json')
    if not os.path.isfile(stats):
        raise FileNotFoundError(f'Offline normalization statistics required: {stats}')
    normalizer = MinMaxNormalizer()
    normalizer.load(stats)
    ws_bounds = np.asarray(cfg.env.workspace_bounds)
    base_policy = EmbodiedGenPolicy(
        in_channels=cfg.model.get('in_channels', 3),
        action_dim=cfg.model.action_dim, chunk_size=cfg.model.chunk_size,
        use_state=cfg.model.use_state, state_dim=cfg.model.state_dim,
        encoder_type=cfg.model.encoder_type, backbone_type=cfg.model.backbone_type,
        cond_dim=cfg.model.cond_dim, algo_type=cfg.model.algo_type).to(device)
    weights = torch.load(ckpt, map_location=device)
    if 'ema_model_state_dict' in weights:
        weights = weights['ema_model_state_dict']
    elif 'model_state_dict' in weights:
        weights = weights['model_state_dict']
    base_policy.load_state_dict(weights, strict=True)
    del weights
    base_policy.encoder.requires_grad_(False)
    base_policy.eval()
    actor = FlowPPOPolicy(base_policy, num_steps=cfg.model.num_inference_steps,
                          exec_steps=cfg.env.exec_steps,
                          std=cfg.algo.exploration_std).to(device)
    actor.eval()
    reference = copy.deepcopy(actor).eval()
    reference.requires_grad_(False)

    critic_encoder = CriticFeatureExtractor(cfg).to(device)
    # Same architecture, copied parameters; no shared optimizer parameters.
    critic_encoder.encoder.load_state_dict(base_policy.encoder.state_dict())
    critic_encoder.requires_grad_(False)
    critic = Critic_PG_Wrapper(critic_encoder,
                VNetwork(state_dim=cfg.model.cond_dim).to(device)).to(device).eval()
    trainer = FlowPolicyGradient(
        actor, critic, actor_lr=cfg.algo.actor_lr, critic_lr=cfg.algo.critic_lr,
        gamma=cfg.algo.gamma, gae_lambda=cfg.algo.gae_lambda,
        clip_ratio=cfg.algo.clip_ratio, entropy_coef=cfg.algo.entropy_coef,
        v_loss_coef=cfg.algo.v_loss_coef, max_grad_norm=cfg.algo.max_grad_norm,
        device=device, target_kl=cfg.algo.target_kl, anchor_coef=cfg.algo.anchor_coef)

    def encode(obs):
        pc = normalizer.center_point_cloud(obs['point_cloud'], ws_bounds)
        state = normalizer.normalize(obs['state'], 'state')
        return {'pc': torch.as_tensor(pc, dtype=torch.float32, device=device).unsqueeze(0),
                'state': torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)}

    ckpt_dir = os.path.join(cfg.save_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    normalizer.save(os.path.join(ckpt_dir, 'dataset_stats.json'))
    # Preserve global RNG state around video evaluation.
    def video(epoch):
        np_state, py_state = np.random.get_state(), random.getstate()
        try:
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                return evaluate_and_record_video(cfg, base_policy, epoch, device,
                                          normalizer=normalizer, seed=seed)
        finally:
            np.random.set_state(np_state)
            random.setstate(py_state)
            actor.eval()

    baseline_metrics = video(0)  # same multi-episode evaluation before any update
    debug_logger.log_metrics(0, baseline_metrics)
    if cfg.wandb.enable:
        wandb.log(baseline_metrics, step=0)
    env = make_env_ManiSkill(cfg)
    obs, _ = env.reset(seed=seed)
    buffer = RolloutBuffer(device)
    ep_reward, ep_success = 0.0, False
    try:
        for epoch in range(1, cfg.epochs+1):
            buffer.clear()
            actor.eval()
            critic.eval()
            epoch_reward, finished_rewards, finished_successes = 0.0, [], []
            for step in tqdm(range(cfg.algo.steps_per_epoch), desc=f'Epoch {epoch} Rollout'):
                obs_dict = encode(obs)
                with torch.no_grad():
                    # Frozen actor/reference encoders are identical. Cache the
                    # exact rollout feature so PPO never recomputes its point grouping.
                    obs_dict['cond'] = actor.encode_condition(obs_dict)
                    value = critic(obs_dict).item()
                    action, z, old_logp, old_mean = actor.sample_with_log_prob(obs_dict)
                    ref_mean = reference.mean(obs_dict, z)
                debug_logger.log_io(epoch, step, obs_dict, action)
                raw_action = action[0].cpu().numpy()
                # STORE raw action; only clip the separate environment copy.
                real_action = normalizer.unnormalize(np.clip(raw_action, -1, 1), 'action')
                next_obs, reward, terminated, truncated, info = env.step(real_action)
                reward = float(scalar(reward))
                terminated, truncated = bool(scalar(terminated)), bool(scalar(truncated))
                done = terminated or truncated
                epoch_reward += reward
                ep_reward += reward
                ep_success |= bool(scalar(info.get('success', False)))
                gae_reward = reward
                # Timeout bootstraps from FINAL observation, before env.reset().
                if truncated and not terminated:
                    with torch.no_grad():
                        gae_reward += trainer.gamma * critic(encode(next_obs)).item()
                buffer.add(pc=obs_dict['pc'][0].cpu().numpy(),
                           state=obs_dict['state'][0].cpu().numpy(),
                           cond=obs_dict['cond'][0].cpu().numpy(),
                           actions=raw_action, z=z[0].cpu().numpy(),
                           old_logp=old_logp.item(), old_mean=old_mean[0].cpu().numpy(),
                           ref_mean=ref_mean[0].cpu().numpy(), rewards=gae_reward,
                           dones=float(done), values=value)
                obs = next_obs
                if done:
                    finished_rewards.append(ep_reward)
                    finished_successes.append(float(ep_success))
                    ep_reward, ep_success = 0.0, False
                    obs, _ = env.reset()

            data = buffer.get_tensors()
            with torch.no_grad():
                next_value = critic(encode(obs)).squeeze(-1)
            adv, returns = trainer.compute_gae(data['rewards'][:, None],
                data['values'][:, None], data['dones'][:, None], next_value)
            adv, returns = adv.flatten(), returns.flatten()
            adv = (adv-adv.mean()) / (adv.std(unbiased=False)+1e-8)
            batch_size = int(cfg.algo.minibatch_size)
            n = len(adv)
            # Preserve behavior logp, means and reference means exactly as sampled.
            # This includes the autograd path used during an actual actor update.
            initial = replay_diagnostics(
                actor, data, batch_size, check=True,
                logprob_tolerance=float(cfg.algo.logprob_tolerance),
                initial_kl_tolerance=float(cfg.algo.initial_kl_tolerance))

            records = []
            actor_enabled = epoch > cfg.algo.critic_warmup_epochs
            stop_kl_value = None
            first_kl = None
            for _ in range(cfg.algo.update_epochs):
                indices = np.random.permutation(n)
                for start in range(0, n, batch_size):
                    idx = indices[start:start + batch_size]
                    losses = trainer.update_step(
                        states={key: data[key][idx] for key in ('pc', 'state', 'cond')},
                        actions=data['actions'][idx], old_log_probs=data['old_logp'][idx],
                        returns=returns[idx], advantages=adv[idx], z=data['z'][idx],
                        old_mean=data['old_mean'][idx], ref_mean=data['ref_mean'][idx],
                        actor_enabled=actor_enabled)
                    losses['batch_n'] = len(idx)
                    if losses['actor_checked'] and first_kl is None:
                        first_kl = losses['conditional_kl']
                    if losses['stop_actor']:
                        actor_enabled = False
                        stop_kl_value = losses['conditional_kl']
                    records.append(losses)

            active = [r for r in records if r['actor_updated']]
            checked = [r for r in records if r['actor_checked']]
            count, actor_updates = len(records), len(active)
            def weighted_mean(rows, key):
                return sum(r[key] * r['batch_n'] for r in rows) / sum(r['batch_n'] for r in rows)

            metrics = {
                'critic_loss': weighted_mean(records, 'critic_loss'),
                'PPO/critic_grad_norm': weighted_mean(records, 'critic_grad_norm'),
                'PPO/actor_updates': actor_updates,
                'PPO/actor_update_fraction': actor_updates / count,
                'PPO/actor_lr': trainer.optimizer_policy.param_groups[0]['lr'],
                'PPO/old_logp_max_error': initial['logp_max_error'],
                'PPO/old_mean_max_error': initial['mean_max_error'],
                'KL/initial_mean': initial['kl_mean'],
                'KL/initial_max': initial['kl_max'],
                'Reward/EpochSum': epoch_reward,
                'Episodes/completed': len(finished_rewards),
            }
            for key in ('stop_nonfinite', 'stop_kl', 'stop_ratio'):
                metrics[f'PPO/{key}_count'] = int(sum(r[key] for r in records))
            if active:
                for source, dest in (
                    ('actor_loss', 'actor_loss_active'),
                    ('anchor_loss', 'anchor_loss_active'),
                    ('actor_grad_norm', 'actor_grad_norm_active'),
                    ('clipfrac', 'clipfrac_active')):
                    metrics[f'PPO/{dest}'] = weighted_mean(active, source)
            if checked:
                metrics['KL/checked_mean'] = weighted_mean(checked, 'conditional_kl')
                metrics['KL/checked_max'] = max(r['conditional_kl'] for r in checked)
                metrics['KL/first_before_update'] = first_kl
            if stop_kl_value is not None:
                metrics['KL/trigger_at_stop'] = stop_kl_value

            # Fixed, evenly spaced rollout subset; diagnostic only, no rollback.
            monitor_n = min(n, int(cfg.algo.kl_monitor_size))
            monitor_idx = torch.linspace(0, n - 1, monitor_n, device=device).long()
            final = replay_diagnostics(actor, {k: v[monitor_idx] for k, v in data.items()}, batch_size)
            metrics['KL/final_fixed_batch'] = final['kl_mean']
            metrics['KL/final_max_sample'] = final['kl_max']
            if finished_rewards:
                metrics['Reward/EpisodeMean'] = float(np.mean(finished_rewards))
                metrics['Success/TrainEpisodeRate'] = float(np.mean(finished_successes))

            def fmt(value):
                return 'N/A' if value is None else f'{value:.6f}'
            print(
                f'Epoch {epoch:03d} | Reward Sum: {epoch_reward:.2f} | '
                f'Actor updates: {actor_updates}/{count} | '
                f'Actor loss: {fmt(metrics.get("PPO/actor_loss_active"))} | '
                f'Actor grad(pre-clip): {fmt(metrics.get("PPO/actor_grad_norm_active"))} | '
                f'Critic: {metrics["critic_loss"]:.4f} | '
                f'Old logp error: {initial["logp_max_error"]:.3e} | '
                f'KL initial/stop/final: {initial["kl_mean"]:.3e}/'
                f'{fmt(stop_kl_value)}/{final["kl_mean"]:.6f} | '
                f'Episode reward: {fmt(metrics.get("Reward/EpisodeMean"))} | '
                f'Train success: {fmt(metrics.get("Success/TrainEpisodeRate"))} | '
                f'Stop KL/nonfinite/ratio: {metrics["PPO/stop_kl_count"]}/'
                f'{metrics["PPO/stop_nonfinite_count"]}/{metrics["PPO/stop_ratio_count"]}'
            )
            if epoch % cfg.save_epoch == 0 or epoch == cfg.epochs:
                # Compatible with existing evaluate.py; this is deployment, not resume state.
                torch.save(base_policy.state_dict(),
                           os.path.join(ckpt_dir, f'pg_finetuned_ep{epoch}.pth'))
                metrics.update(video(epoch))
            if cfg.wandb.enable:
                wandb.log(metrics, step=epoch)
            debug_logger.log_metrics(epoch, metrics)
    finally:
        env.close()
        if cfg.wandb.enable:
            wandb.finish()


if __name__ == '__main__':
    main()

