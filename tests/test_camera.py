import h5py

path = "/home/luo/.maniskill/demos/PickCube-v1/motionplanning/trajectory.pointcloud.pd_ee_delta_pose.physx_cpu.h5"

with h5py.File(path, "r") as f:
    key = list(f.keys())[0]
    g = f[key]

    print("trajectory:", key)
    print("xyzw shape:",
          g["obs"]["pointcloud"]["xyzw"].shape)
    print("rgb shape:",
          g["obs"]["pointcloud"]["rgb"].shape)