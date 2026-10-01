"""Training-only StackCube state and potential. Never supplied to the Actor."""
import torch

PRIVILEGED_DIM = 53  # Panda qpos(9), qvel(9), TCP(7), cubes(26), flags(2)


@torch.no_grad()
def stackcube_state(env):
    base = env.unwrapped
    a, b, agent = base.cubeA, base.cubeB, base.agent
    flags = dict(base.evaluate())
    grasped = flags.get('is_cubeA_grasped', flags.get('is_grasped'))
    if grasped is None:
        grasped = agent.is_grasping(a)
    flags['is_grasped'] = grasped
    qpos, qvel = agent.robot.get_qpos(), agent.robot.get_qvel()
    state = torch.cat((qpos, qvel, agent.tcp.pose.raw_pose,
        a.pose.raw_pose, a.linear_velocity, a.angular_velocity,
        b.pose.raw_pose, b.linear_velocity, b.angular_velocity,
        grasped.float()[:, None], flags['is_cubeA_on_cubeB'].float()[:, None]), dim=-1)
    if state.shape[-1] != PRIVILEGED_DIM or not torch.isfinite(state).all():
        raise ValueError('StackCube privileged Critic requires finite Panda state (53 dimensions)')
    goal = b.pose.p.clone()
    goal[:, 2] += 2 * base.cube_half_size[2]
    reach = 1 - torch.tanh(5 * torch.linalg.vector_norm(agent.tcp.pose.p - a.pose.p, dim=-1))
    lift = ((a.pose.p[:, 2] - base.cube_half_size[2]) / .08).clamp(0, 1)
    align = 1 - torch.tanh(5 * torch.linalg.vector_norm(a.pose.p - goal, dim=-1))
    on = flags['is_cubeA_on_cubeB'].float()
    held_or_on = torch.maximum(grasped.float(), on)
    stable = 1 - torch.tanh(10 * torch.linalg.vector_norm(a.linear_velocity, dim=-1)
                          + torch.linalg.vector_norm(a.angular_velocity, dim=-1))
    potential = (.2 * reach + .2 * grasped.float() + .2 * held_or_on * lift
                 + .2 * held_or_on * align + .2 * on * (~grasped.bool()).float() * stable)
    return state.detach().clone(), potential.detach().clone(), flags


def shaped_reward(success, before, after, terminal, protocol):
    """Terminal potential is zero at BOTH success and the finite task deadline."""
    terminal_potential = 0. if terminal else after
    return float(float(success) * protocol['success_reward']
                 + protocol['potential_scale'] * (protocol['gamma'] * terminal_potential - before))
