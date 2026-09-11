"""Numerical settings and immutable behavior-policy replay checks."""
import torch


def configure_ppo_numerics():
    # MHA otherwise may select different eval/no_grad and autograd paths.
    # Set this once, before rollout, training, reference inference and evaluation.
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('highest')


def replay_diagnostics(actor, data, batch_size, *, check=False,
                       logprob_tolerance=1e-3, initial_kl_tolerance=1e-5):
    """Use the actor-update autograd path, without backward or buffer writes.

    Initial diagnostics use all rollout samples. Final diagnostics may use a
    fixed subset supplied by the caller. KL is summed over executed dimensions
    then averaged over samples, with sample-count weighting across minibatches.
    """
    actor.eval()
    n = len(data['old_logp'])
    if n < 1 or batch_size < 1:
        raise ValueError('Empty replay data or invalid minibatch size')
    lp_max, mean_max, kl_sum, kl_max = 0.0, 0.0, 0.0, 0.0
    with torch.enable_grad():
        for start in range(0, n, batch_size):
            sl = slice(start, start + batch_size)
            obs = {key: data[key][sl] for key in ('pc', 'state', 'cond')
                   if key in data}
            lp, _, mean = actor.evaluate_actions(obs, data['actions'][sl], data['z'][sl])
            # Detach immediately; diagnostics never backpropagate.
            lp, mean = lp.detach(), mean.detach()
            lp_error = (lp - data['old_logp'][sl]).abs()
            delta = mean[:, :actor.exec_steps] - data['old_mean'][sl, :actor.exec_steps]
            kl = (delta.square() / (2.0 * actor.std**2)).sum((-1, -2))
            if not bool(torch.isfinite(lp_error).all() & torch.isfinite(kl).all()):
                raise FloatingPointError('Nonfinite policy replay diagnostics')
            lp_max = max(lp_max, lp_error.max().item())
            mean_max = max(mean_max, delta.abs().max().item())
            kl_sum += kl.sum().item()
            kl_max = max(kl_max, kl.max().item())
    result = {'logp_max_error': lp_max, 'mean_max_error': mean_max,
              'kl_mean': kl_sum / n, 'kl_max': kl_max}
    if check and (lp_max > logprob_tolerance or kl_max > initial_kl_tolerance):
        raise RuntimeError(
            'Behavior-policy replay mismatch BEFORE any optimizer step: '
            f'{result}. Original rollout logp/mean were preserved. '
            'Check cached features, inference/autograd kernels and precision; '
            'do not overwrite old_logp or enlarge tolerances to hide this error.'
        )
    return result
