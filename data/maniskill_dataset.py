# data/maniskill_dataset.py
import h5py
import numpy as np
from tqdm import tqdm

def load_maniskill_h5(h5_path, max_episodes=None, workspace_bounds=None):
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
            cropped_xyz_list = []
            
            # 逐帧过滤掉多余的背景点
            for t in range(xyz.shape[0]):
                pts = xyz[t]
                mask = (
                    (pts[:, 0] >= bounds[0, 0]) & (pts[:, 0] <= bounds[1, 0]) &
                    (pts[:, 1] >= bounds[0, 1]) & (pts[:, 1] <= bounds[1, 1]) &
                    (pts[:, 2] >= bounds[0, 2]) & (pts[:, 2] <= bounds[1, 2])
                )
                cropped_xyz_list.append(pts[mask])
            
            # 使用 object 类型的 array 存储不定长的点云帧
            xyz_cropped = np.empty(len(cropped_xyz_list), dtype=object)
            xyz_cropped[:] = cropped_xyz_list
            
            # 2. 提取本体状态 (Proprioception)
            # 拼接 qpos 和 qvel。假设是 PickCube 任务，通常加起来是 14 维左右
            qpos = traj['obs']['agent']['qpos'][:]
            qvel = traj['obs']['agent']['qvel'][:]
            state = np.concatenate([qpos, qvel], axis=-1).astype(np.float32)
            
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
                'pc': xyz_cropped,
                'state': state,
                'action': action,
                'reward': reward,
                'done': done
            })
            
    print(f"✅ 成功加载 {len(trajectories)} 条专家轨迹！")
    return trajectories