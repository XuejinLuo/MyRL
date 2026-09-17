# view_pc.py
import numpy as np
import open3d as o3d
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="交互式查看 3D 点云")
    parser.add_argument(
        "file", 
        type=str, 
        nargs="?", # 设为可选参数
        default="/home/luo/MyRL/outputs/eval/run_20260907_212454/first_frame_pc.npy", 
        help="点云文件路径（如果不传，则使用默认写死的路径）"
    )
    args = parser.parse_args()

    try:
        # 1. 加载 numpy 数据
        pc_data = np.load(args.file)
        print(f"✅ 成功加载点云数据，形状: {pc_data.shape}")
    except Exception as e:
        print(f"❌ 加载文件失败: {e}")
        sys.exit(1)

    # 2. 转换为 Open3D 格式
    pcd = o3d.geometry.PointCloud()
    
    # 提取前 3 维作为 XYZ 坐标
    pcd.points = o3d.utility.Vector3dVector(pc_data[:, :3])

    # 如果有 RGB 特征（维度 >= 6），则上色
    if pc_data.shape[-1] >= 6:
        colors = pc_data[:, 3:6]
        # Open3D 的颜色范围需要是 [0, 1]，如果原图是 [0, 255]，这里做个防御性归一化
        if colors.max() > 1.0:
            colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
        print("🎨 检测到 RGB 颜色通道，已对点云进行上色。")
    else:
        # 如果没有颜色通道，统一涂成科技蓝，方便观察
        pcd.paint_uniform_color([0.0, 0.65, 0.92])
        print("⚪ 未检测到颜色通道，使用默认单色渲染。")

    # 3. 创建世界坐标系指示器 (红=X, 绿=Y, 蓝=Z)
    # size 控制坐标轴大小，可以根据你的 Workspace 大小微调
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])

    print("\n👉 交互操作提示：")
    print("  - 鼠标左键：拖拽旋转")
    print("  - 鼠标右键：平移画面")
    print("  - 鼠标滚轮：放大 / 缩小")
    print("  - 按键盘 'Q' 键或关闭窗口以退出程序\n")

    # 4. 弹出 3D 渲染窗口
    o3d.visualization.draw_geometries(
        [pcd, coordinate_frame], 
        window_name="Embodied AI - 1st Frame PointCloud Viewer",
        width=1024,
        height=768
    )

if __name__ == "__main__":
    main()