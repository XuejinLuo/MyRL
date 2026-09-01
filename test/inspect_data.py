import os
import torch
import numpy as np

# 引入你项目已写好的数据读取器和 Dataset
from data.maniskill_dataset import load_maniskill_h5
from data.dataset import PointCloudChunkDataset

def main():
    # 1. 你的数据路径 (从 configs/dataset/offline_data.yaml 中提取)
    h5_path = "/home/luo/.maniskill/demos/PickCube-v1/motionplanning/trajectory.pointcloud.pd_ee_delta_pose.physx_cpu.h5"
    
    if not os.path.exists(h5_path):
        print(f"⚠️ 找不到数据文件，请检查路径是否正确: {h5_path}")
        return
        
    print(f"📦 开始加载 ManiSkill 数据...")
    # 2. 读取轨迹 (限制只读前 2 个 Episode，节省内存和时间)
    trajectories = load_maniskill_h5(
        h5_path=h5_path, 
        max_episodes=2, 
        workspace_bounds=[[-0.5, -0.5, 0.0], [0.5, 0.5, 0.5]]
    )
    
    # 3. 使用你的 PointCloudChunkDataset 进行包装
    dataset = PointCloudChunkDataset(
        trajectories=trajectories, 
        chunk_size=16, 
        n_points=1024
    )
    
    print(f"✅ 数据集解析完成，前 2 个 Episode 共切分出 {len(dataset)} 个有效样本 (Transitions)。\n")
    
    # 4. 提取第 0 个样本（第一步的观测和预测）
    sample = dataset[0]
    
    print("================ 📊 [样本内部的数据结构] ================")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"🔑 {key:12s} | Shape: {str(list(value.shape)):16s} | Type: {value.dtype}")
        else:
            print(f"🔑 {key:12s} | Value: {value}")
            
    print("\n================ 🤖 [具体数值抽查] ====================")
    # 【新增】设置 numpy 打印格式：取消科学计数法，保留4位小数，加宽打印行宽，让矩阵显示更美观
    np.set_printoptions(precision=4, suppress=True, linewidth=150)
    
    print("📍 [State] 当前机器人的 State (全维度):")
    print(sample['state'].numpy())
    
    print("\n📍 [Next State] 下一步机器人的 State (全维度):")
    print(sample['next_state'].numpy())
    
    print(f"\n🎯 [Action Chunk] 未来 {sample['action_chunk'].shape[0]} 步的动作序列 (Shape: {list(sample['action_chunk'].shape)}):")
    print(sample['action_chunk'].numpy())
    
    print("\n💰 [Reward] 这一步的稀疏 Reward:", sample['reward'].item())
    
    # 5. [附加] 3D 点云可视化
    print("\n👀 正在启动 matplotlib 可视化 3D 点云 (关闭图像窗口以退出程序)...")
    try:
        import matplotlib.pyplot as plt
        
        # 提取点云的 numpy 数组，shape: (1024, 3)
        pc = sample['pc'].numpy() 
        
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 根据 Z 轴的高度来上色，更有立体感
        sc = ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=pc[:, 2], cmap='viridis', s=6)
        
        ax.set_title("ManiSkill Observation (1024 Points)")
        ax.set_xlabel("X (Forward)")
        ax.set_ylabel("Y (Left)")
        ax.set_zlabel("Z (Up)")
        
        # 限制坐标轴比例为 1:1:1 左右，防止渲染出的抓取场景被拉伸变形
        ax.set_xlim([-0.5, 0.5])
        ax.set_ylim([-0.5, 0.5])
        ax.set_zlim([0.0, 0.5])
        
        plt.colorbar(sc, ax=ax, label="Z Height", shrink=0.5)
        plt.show()
        
    except ImportError:
        print("💡 提示：你所在的 Python 环境没有安装 matplotlib。运行 `pip install matplotlib` 即可绘制出 3D 点云图。")

if __name__ == "__main__":
    main()