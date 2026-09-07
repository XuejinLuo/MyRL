# data/maniskill_dataset.py
import h5py
import numpy as np
from tqdm import tqdm

def load_maniskill_h5(h5_path, max_episodes=None, workspace_bounds=None, n_points=1024):
    """
    读取 ManiSkill 生成的点云 .h5 数据，转换为离线 RL 训练所需格式。
    """
    print(f"📦 正在从 {h5_path} 加载 ManiSkill 数据...")
    trajectories = []
    
    with h5py.File(h5_path, 'r') as f:
        # 获取所有轨迹的键名，例如 'traj_0', 'traj_1'
        keys = list(f.keys())
        if max_episodes is not None:
            keys = keys[:max_episodes]
            
        for ep_key in tqdm(keys, desc="Loading Episodes"):
            traj = f[ep_key]
            
            # 1. 提取点云 (取前三维 X, Y, Z)
            # 原始维度 [T, 16384, 4] -> 切片为 [T, 16384, 3]
            pc_dict = traj['obs']['pointcloud']
            xyzw = pc_dict['xyzw'][:]
            xyz = xyzw[..., :3].astype(np.float32)
            
            # 提取颜色并归一化到 [0, 1]，然后与 xyz 拼接
            if 'rgb' in pc_dict:
                rgb = (pc_dict['rgb'][:] / 255.0).astype(np.float32)
                xyz = np.concatenate([xyz, rgb], axis=-1)  # 此时特征维度变为 6

            # 空间裁剪
            if workspace_bounds is not None:
                bounds = np.array(workspace_bounds)
            else:
                bounds = np.array([[-0.5, -0.5, 0.0], [0.5, 0.5, 0.5]])
                
            T = xyz.shape[0]
            C = xyz.shape[2]
            fixed_xyz = np.zeros((T, n_points, C), dtype=np.float32)
            
            # 逐帧过滤掉多余的背景点，并立即降采样
            for t in range(T):
                pts = xyz[t]
                mask = (
                    (pts[:, 0] >= bounds[0, 0]) & (pts[:, 0] <= bounds[1, 0]) &
                    (pts[:, 1] >= bounds[0, 1]) & (pts[:, 1] <= bounds[1, 1]) &
                    (pts[:, 2] >= bounds[0, 2]) & (pts[:, 2] <= bounds[1, 2])
                )
                valid_pts = pts[mask]
                num_valid = valid_pts.shape[0]
                
                # 提前进行目标数量的点云采样
                if num_valid >= n_points:
                    choices = np.random.choice(num_valid, n_points, replace=False)
                    fixed_xyz[t] = valid_pts[choices]
                elif num_valid > 0:
                    choices = np.random.choice(num_valid, n_points, replace=True)
                    fixed_xyz[t] = valid_pts[choices]
                else:
                    # 如果这帧画面里没有有效的点（全被裁剪掉了），维持 zeros 不变
                    pass
            
            # 2. 提取本体状态 (Proprioception)
            qpos = traj['obs']['agent']['qpos'][:]
            state = qpos.astype(np.float32) 
            
            # 3. 提取动作
            action = traj['actions'][:].astype(np.float32)
            
            # 4. 提取奖励和结束标志 (给 IDQL 的 Critic 更新使用)
            # 真实的演示轨迹通常都是成功的，可以用 'success' 作为稀疏奖励
            if 'success' in traj:
                reward = traj['success'][:].astype(np.float32)
            else:
                reward = np.zeros(len(action), dtype=np.float32)
                reward[-1] = 1.0 # 假设最后一步成功
                
            done = np.zeros(len(action), dtype=bool)
            if 'terminated' in traj:
                done = traj['terminated'][:]
            done[-1] = True # 强行确保最后一步为 Done
            
            trajectories.append({
                'pc': fixed_xyz,
                'state': state,
                'action': action,
                'reward': reward,
                'done': done
            })

            # print(f"Raw RGB Max: {pc_dict['rgb'][:].max()}, Min: {pc_dict['rgb'][:].min()}")
    print(f"✅ 成功加载 {len(trajectories)} 条专家轨迹！")
    return trajectories