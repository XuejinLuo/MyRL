"""Matched multi-episode evaluation with isolated global random states."""
import random
from contextlib import contextmanager
import numpy as np
import torch


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


def evaluate_policy(make_env, actor, encode, normalizer, seeds, num_steps):
    """Same env factory/horizon/FP32 sampler as training; no extra action noise.

    Evaluation errors propagate rather than being reported as zero success.
    """
    if not seeds:
        raise ValueError('Evaluation requires at least one seed')
    module_modes = [(m, m.training) for m in actor.modules()]
    rewards, successes = [], []
    env = None
    with preserve_rng():
        try:
            seed_all(int(seeds[0]))
            env = make_env()
            actor.eval()
            for seed in seeds:
                seed_all(int(seed))
                obs, _ = env.reset(seed=int(seed))
                done, truncated, success, reward_sum = False, False, False, 0.
                while not (done or truncated):
                    with torch.no_grad():
                        features = encode(obs)
                        actions = actor.sample(features, num_steps=num_steps)[0].cpu().numpy()
                    obs, reward, done, truncated, info = env.step(normalizer.unnormalize(actions, 'action'))
                    success = success or bool(info.get('success_any', info.get('success', False)))
                    reward_sum += float(reward)
                rewards.append(reward_sum)
                successes.append(float(success))
        finally:
            if env is not None:
                env.close()
            for module, mode in module_modes:
                module.training = mode
    return {'Eval/Success_Rate': float(np.mean(successes)),
            'Eval/Mean_Reward': float(np.mean(rewards)),
            'Eval/Episodes': len(seeds)}
