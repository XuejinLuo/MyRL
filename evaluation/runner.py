"""Matched multi-episode evaluation with isolated global random states."""
import random
from contextlib import contextmanager
import numpy as np
import torch
from tqdm import tqdm

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def preserve_rng():
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def evaluate_policy(make_env, actor, encode, normalizer, seeds, num_steps,
                    output_dir=None, metadata=None, action_callback=None):
    """Same env factory/horizon/FP32 sampler as training; no extra action noise.

    Evaluation errors propagate rather than being reported as zero success.
    """
    if not seeds:
        raise ValueError('Evaluation requires at least one seed')
    module_modes = [(m, m.training) for m in actor.modules()]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Evaluation seeds must be unique")
    rewards, successes, rows = [], [], []
    env = None
    with preserve_rng():
        pbar = tqdm(
            total=len(seeds),
            desc=f"Eval {getattr(actor, 'eval_mode', 'unknown').upper()}",
            unit="ep",
            dynamic_ncols=True,
            mininterval=0.5,
        )
        try:
            seed_all(int(seeds[0]))
            env = make_env()
            actor.eval()
            for seed in seeds:
                pbar.set_postfix(seed=int(seed), refresh=False)
                seed_all(int(seed))
                if hasattr(actor, 'reset_episode_diagnostics'):
                    actor.reset_episode_diagnostics()
                obs, _ = env.reset(seed=int(seed))
                done, truncated, success, reward_sum = False, False, False, 0.
                primitive_steps = 0
                while not (done or truncated):
                    with torch.no_grad():
                        features = encode(obs)
                        actions = actor.sample(features, num_steps=num_steps)[0].cpu().numpy()
                    if not np.isfinite(actions).all():
                        raise FloatingPointError('Nonfinite evaluation action')
                    if action_callback is not None:
                        with preserve_rng():
                            action_callback(actions.copy())
                    obs, reward, done, truncated, info = env.step(normalizer.unnormalize(actions, 'action'))
                    if hasattr(actor, 'record_execution'):
                        actor.record_execution(info)
                    success = success or bool(info.get('success_any', info.get('success', False)))
                    reward_sum += float(reward)
                    primitive_steps += int(info.get("actual_steps", 1))
                rewards.append(reward_sum)
                successes.append(float(success))
                rows.append(dict(seed=int(seed), success=int(success),
                    success_final=int(bool(info.get("success", False))),
                    episode_return=reward_sum, primitive_steps=primitive_steps))
                if hasattr(actor, 'action_diagnostics'):
                    rows[-1].update(actor.action_diagnostics(episode=True))
                pbar.set_postfix(
                    seed=int(seed),
                    success_rate=f"{np.mean(successes):.1%}",
                    mean_reward=f"{np.mean(rewards):.1f}",
                    refresh=False,
                )
                pbar.update(1)
        finally:
            pbar.close()
            if env is not None:
                env.close()
            for module, mode in module_modes:
                module.training = mode
    result = {'Eval/Success_Rate': float(np.mean(successes)),
            'Eval/Mean_Reward': float(np.mean(rewards)),
            'Eval/Episodes': len(seeds)}

    result['Eval/Success_Count'] = int(sum(successes))
    result['Eval/Final_Success_Rate'] = float(np.mean([r['success_final'] for r in rows]))
    # Wilson interval: descriptive binomial uncertainty, not a paired significance test.
    n, p, z = len(seeds), float(np.mean(successes)), 1.959963984540054
    center = (p + z*z/(2*n))/(1+z*z/n)
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/(1+z*z/n)
    result['Eval/CI95_Low'], result['Eval/CI95_High'] = float(center-half), float(center+half)
    if hasattr(actor, 'action_diagnostics'):
        result.update(actor.action_diagnostics())
    if output_dir is not None:
        import csv
        from pathlib import Path
        from utils.experiment import write_json
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory/'episodes.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        write_json(directory/'summary.json', dict(schema='myrl_eval_v1',
            success_definition='any primitive step reports success',
            seeds=[int(s) for s in seeds], metadata=metadata or {}, **result))
    return result
