# data/check_maniskill_dataset.py
import h5py

# 指向你刚刚生成出的那个庞大的新文件
h5_path = "/home/luo/.maniskill/demos/PickCube-v1/motionplanning/trajectory.pointcloud.pd_joint_pos.physx_cpu.h5" 

with h5py.File(h5_path, 'r') as f:
    # 随便取第一条轨迹
    first_traj_key = list(f.keys())[0]
    traj = f[first_traj_key]
    
    print("这条轨迹包含的键值:", list(traj.keys()))
    
    # 检查点云数据
    if 'obs' in traj and 'pointcloud' in traj['obs']:
        pc_data = traj['obs']['pointcloud']
        # ManiSkill 3 点云默认存储键名可能是 xyzw 或 xyz
        if 'xyzw' in pc_data:
            print("🎉 点云(xyzw)数据维度:", pc_data['xyzw'].shape)
        elif 'xyz' in pc_data:
            print("🎉 点云(xyz)数据维度:", pc_data['xyz'].shape)
    else:
        print("⚠️ 还是没有找到 pointcloud 数据，检查生成命令！")