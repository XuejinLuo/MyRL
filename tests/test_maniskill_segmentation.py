import h5py

path = (
    "/home/luo/.maniskill/demos/StackCube-v1/"
    "motionplanning/"
    "trajectory.pointcloud.pd_ee_delta_pose.physx_cpu.h5"
)

with h5py.File(path, "r") as f:
    ep = list(f.keys())[0]

    print("episode:", ep)
    print("obs keys:")
    print(list(f[ep]["obs"].keys()))

    print("\npointcloud keys:")
    print(list(f[ep]["obs"]["pointcloud"].keys()))

    for k in f[ep]["obs"]["pointcloud"].keys():
        x = f[ep]["obs"]["pointcloud"][k]
        print(k, x.shape, x.dtype)

import gymnasium as gym
import mani_skill.envs

env = gym.make(
    "StackCube-v1",
    obs_mode="pointcloud",
    control_mode="pd_ee_delta_pose",
    sim_backend="physx_cpu",
    num_envs=1,
)

env.reset(seed=0)

print("=" * 60)
print("Segmentation ID map")
print("=" * 60)

for obj_id, obj in sorted(
    env.unwrapped.segmentation_id_map.items()
):
    name = getattr(obj, "name", "UNKNOWN")
    print(f"{obj_id:4d} -> {name}")

env.close()