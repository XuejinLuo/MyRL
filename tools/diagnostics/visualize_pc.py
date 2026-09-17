import h5py
import numpy as np
import open3d as o3d
import time

def main():
    h5_path = "/home/luo/.maniskill/demos/PickCube-v1/motionplanning/trajectory.pointcloud.pd_ee_delta_pose.physx_cpu.h5"
    
    # 🌟 1. 定义你的工作空间裁剪范围 (参考你项目中的 offline_data.yaml)
    # 格式: [[x_min, y_min, z_min], [x_max, y_max, z_max]]
    # 注意: z_min 设为 0.0 通常是为了过滤掉桌面以下的机器基座和无关地面
    workspace_bounds = np.array([
        [-0.5, -0.5, 0.02], 
        [ 0.5,  0.5, 0.5]
    ])
    
    with h5py.File(h5_path, 'r') as f:
        # 获取第一条轨迹
        traj_key = list(f.keys())[0]
        traj = f[traj_key]
        
        # 提取第 0 帧的点云
        frame_idx = 0 
        pc_dict = traj['obs']['pointcloud']
        
        # 提取坐标
        if 'xyzw' in pc_dict:
            xyz = pc_dict['xyzw'][frame_idx, :, :3] 
        else:
            xyz = pc_dict['xyz'][frame_idx, :, :]
            
        # 提取颜色
        rgb = pc_dict['rgb'][frame_idx, :, :]
        if rgb.max() > 1.0:
            rgb = rgb.astype(np.float32) / 255.0 
            
        print(f"📦 裁剪前 -> 点云数量: {xyz.shape[0]}")
        
        # 🌟 2. 根据 bounds 生成布尔掩码并进行裁剪
        mask = (
            (xyz[:, 0] >= workspace_bounds[0, 0]) & (xyz[:, 0] <= workspace_bounds[1, 0]) &
            (xyz[:, 1] >= workspace_bounds[0, 1]) & (xyz[:, 1] <= workspace_bounds[1, 1]) &
            (xyz[:, 2] >= workspace_bounds[0, 2]) & (xyz[:, 2] <= workspace_bounds[1, 2])
        )
        
        # 应用掩码 (注意：xyz 和 rgb 必须用同一个 mask 同步裁剪！)
        xyz_cropped = xyz[mask]
        rgb_cropped = rgb[mask]
        
        print(f"✂️ 裁剪后 -> 点云数量: {xyz_cropped.shape[0]}")
        print(f"✅ 成功提取点云! 形状: XYZ {xyz_cropped.shape}, RGB {rgb_cropped.shape}")
        
        # ==========================================
        # 🌟 3. 使用 Open3D 进行 3D 渲染
        # ==========================================
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_cropped)
        pcd.colors = o3d.utility.Vector3dVector(rgb_cropped)
        
        # 创建原点坐标系 (红X, 绿Y, 蓝Z)
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        
        # 🌟 4. [福利] 在可视化画面中画出裁剪框的轮廓 (红线框)，方便调试范围是否合理
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=workspace_bounds[0], max_bound=workspace_bounds[1])
        bbox.color = (1.0, 0.0, 0.0) # 边框颜色设为红色
        
        print("🌍 正在打开 3D 视窗... (可用鼠标左键拖拽旋转，滚轮缩放)")
        # 将点云、坐标轴、红色线框一起送入渲染
        o3d.visualization.draw_geometries(
            [pcd, coordinate_frame, bbox], 
            window_name=f"Cropped Frame {frame_idx} of {traj_key}"
        )

if __name__ == "__main__":
    main()