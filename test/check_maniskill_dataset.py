# data/check_maniskill_dataset.py
import h5py

h5_path = "/home/luo/.maniskill/demos/PickCube-v1/motionplanning/trajectory.pointcloud.pd_ee_delta_pose.physx_cpu.h5" 

with h5py.File(h5_path, 'r') as f:
    first_traj_key = list(f.keys())[0]
    traj = f[first_traj_key]
    
    if 'obs' in traj and 'pointcloud' in traj['obs']:
        pc_data = traj['obs']['pointcloud']
        
        # 检查 XYZ
        if 'xyzw' in pc_data:
            print("🎉 包含点云坐标 (xyzw)，维度:", pc_data['xyzw'].shape)
        elif 'xyz' in pc_data:
            print("🎉 包含点云坐标 (xyz)，维度:", pc_data['xyz'].shape)
            
        # 检查 RGB (这里是关键)
        if 'rgb' in pc_data:
            print("🎉 包含颜色信息 (rgb)！维度:", pc_data['rgb'].shape)
            print("   -> 数据样本展示:", pc_data['rgb'][0, 0, :]) # 看看是 0-255 还是 0-1
        else:
            print("⚠️ 糟糕，当前的点云数据中没有 rgb 字段！")