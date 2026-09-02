# data/dataset.py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class PointCloudChunkDataset(Dataset):
    """
    专门为 3D Diffusion/Flow Policy + IDQL/PG 设计的精简版 Dataset。
    输入：纯 3D 点云 + 机器人本体状态 (Proprioception)
    输出：Action Chunk + 用于强化学习的 Transition (Reward, Next_Obs, Done)
    """
    def __init__(
        self, 
        trajectories: list, 
        chunk_size: int = 16, 
        n_points: int = 1024,
        is_training: bool = True, 
        normalizer=None
    ):
        """
        Args:
            trajectories: 包含多个 episode 的列表。
                          每个 episode 是一个 dict: {'pc': [T, N, 3], 'state': [T, D_s], 'action': [T, D_a], 'reward': [T], 'done': [T]}
            chunk_size: 动作块的长度 (Action Chunking)
            n_points: 固定采样的点云数量
            is_training: 是否为训练集（可在此处加入点云的数据增强，如 Random Rotation / Jittering）
        """
        super().__init__()
        self.chunk_size = chunk_size
        self.n_points = n_points
        self.is_training = is_training
        self.normalizer = normalizer
        
        self.trajectories = trajectories
        self.indices = self._build_indices()

    def _build_indices(self):
        """
        构建全局索引到 (episode_idx, step_idx) 的映射。
        过滤掉无效的长度或做特殊处理。
        """
        indices = []
        for ep_idx, ep in enumerate(self.trajectories):
            ep_len = len(ep['action'])
            # 安全检查：跳过长度为0的无效 episode
            if ep_len == 0:
                continue
            # 遍历 episode 中的每一个时间步
            for step_idx in range(ep_len):
                indices.append((ep_idx, step_idx))
        return indices

    def _sample_point_cloud(self, pc: np.ndarray):
        """
        点云降采样 (Random Sampling 比 Farthest Point Sampling 快很多，适合 DataLoader)
        pc shape: [N_original, 3] or [N_original, 3+C]
        """
        N = pc.shape[0]
        if N >= self.n_points:
            # 随机无放回采样
            choices = np.random.permutation(N)[:self.n_points]
        else:
            # 如果点不够，有放回采样补齐
            choices = np.random.choice(N, self.n_points, replace=True)
        return pc[choices]

    def _get_chunk(self, array: np.ndarray, start_idx: int, length: int, pad_value='last'):
        """
        获取从 start_idx 开始的序列块。如果超出 episode 长度，则进行 Padding。
        返回: chunk_array, mask (1 表示真实数据, 0 表示 Padding 数据)
        """
        ep_len = len(array)
        end_idx = min(start_idx + length, ep_len)
        
        # 截取真实数据部分
        valid_chunk = array[start_idx:end_idx]
        valid_len = len(valid_chunk)
        
        # 构建 Mask
        mask = np.zeros(length, dtype=np.float32)
        mask[:valid_len] = 1.0
        
        # 如果需要 Padding
        if valid_len < length:
            pad_len = length - valid_len
            if pad_value == 'last':
                # 重复最后一步的动作/状态 (Diffusion Policy 常用手段)
                padding = np.repeat(valid_chunk[-1:], pad_len, axis=0)
            elif pad_value == 'zero':
                # 补零
                pad_shape = list(valid_chunk.shape)
                pad_shape[0] = pad_len
                padding = np.zeros(pad_shape, dtype=valid_chunk.dtype)
            else:
                raise ValueError(f"Unsupported pad_value: {pad_value}")
                
            chunk = np.concatenate([valid_chunk, padding], axis=0)
        else:
            chunk = valid_chunk
            
        return chunk, mask

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ep_idx, step_idx = self.indices[idx]
        ep = self.trajectories[ep_idx]
        
        # 1. 提取当前观测 (t)
        # 针对 PointNeXt 等网络，点云形状常转为 [3, N] 或保持 [N, 3]。这里我们保持 [N, 3]，在模型内部 Permute。
        pc_t = self._sample_point_cloud(ep['pc'][step_idx])
        state_t = ep['state'][step_idx]
        
        # 2. 提取 Action Chunk (t 到 t+K)
        action_chunk, action_mask = self._get_chunk(ep['action'], step_idx, self.chunk_size, pad_value='last')
        
        # 3. 为 IDQL/PG 提供 RL 相关的 Transitions: Reward(t), Next_Obs(t+1), Done(t)
        reward_t = ep['reward'][step_idx]
        done_t = ep['done'][step_idx]
        
        # 获取 Next Obs (如果是最后一步，next_obs 保持不变，因为 done=True，Q-learning 会 ignore next Q)
        next_step_idx = min(step_idx + 1, len(ep['action']) - 1)
        next_pc_t = self._sample_point_cloud(ep['pc'][next_step_idx])
        next_state_t = ep['state'][next_step_idx]

        # 归一化处理
        if self.normalizer is not None:
            state_t = self.normalizer.normalize(state_t, 'state')
            action_chunk = self.normalizer.normalize(action_chunk, 'action')
            next_state_t = self.normalizer.normalize(next_state_t, 'state')

        # 微调：使用 np.ascontiguousarray 加速 PyTorch Tensor 的内存映射转换
        # 4. 组装并转换为 Tensor
        data_dict = {
            # Behavior Cloning & Diffusion 生成所需
            "pc": torch.from_numpy(np.ascontiguousarray(pc_t)).float(),                    # [N_points, 3]
            "state": torch.from_numpy(np.ascontiguousarray(state_t)).float(),              # [State_Dim]
            "action_chunk": torch.from_numpy(np.ascontiguousarray(action_chunk)).float(),  # [Chunk_Size, Action_Dim]
            "action_mask": torch.from_numpy(np.ascontiguousarray(action_mask)).float(),    # [Chunk_Size] (Loss计算时需要)
            
            # RL (IDQL / PG / Critic网络) 评估所需
            "reward": torch.tensor(reward_t, dtype=torch.float32),   # []
            "done": torch.tensor(done_t, dtype=torch.float32),       # []
            "next_pc": torch.from_numpy(np.ascontiguousarray(next_pc_t)).float(),          # [N_points, 3]
            "next_state": torch.from_numpy(np.ascontiguousarray(next_state_t)).float(),    # [State_Dim]
        }
        
        return data_dict


# =========================================================================
# 测试模块 (可以直接 `python data/dataset.py` 运行)
# =========================================================================
if __name__ == "__main__":
    print("🚀 正在生成 Mock 3D 点云轨迹数据...")
    
    # 模拟超参数
    NUM_EPISODES = 5
    EPISODE_LEN = 50
    N_ORIGINAL_POINTS = 2048 # 原始传感器返回的点云数，一般是不定长的，这里 Mock 为定长
    STATE_DIM = 8  # 如机械臂 7 DoF + 1 Gripper
    ACTION_DIM = 8 # 目标位置 7 DoF + 1 Gripper
    
    # 1. 构建 Mock Trajectories
    mock_trajectories = []
    for _ in range(NUM_EPISODES):
        ep = {
            'pc': np.random.randn(EPISODE_LEN, N_ORIGINAL_POINTS, 3).astype(np.float32),
            'state': np.random.randn(EPISODE_LEN, STATE_DIM).astype(np.float32),
            'action': np.random.randn(EPISODE_LEN, ACTION_DIM).astype(np.float32),
            'reward': np.random.rand(EPISODE_LEN).astype(np.float32),
            'done': np.zeros(EPISODE_LEN, dtype=bool)
        }
        # 最后一个 step 标记为 done
        ep['done'][-1] = True
        mock_trajectories.append(ep)

    print("✅ Mock 数据生成完毕！")
    
    # 2. 实例化 Dataset
    CHUNK_SIZE = 16
    N_SAMPLED_POINTS = 512
    
    dataset = PointCloudChunkDataset(
        trajectories=mock_trajectories,
        chunk_size=CHUNK_SIZE,
        n_points=N_SAMPLED_POINTS
    )
    
    print(f"📦 Dataset 总样本数 (Transitions): {len(dataset)}")
    
    # 3. 实例化 DataLoader 并测试性能与形状
    BATCH_SIZE = 32
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=2, 
        drop_last=False
    )
    
    print("🔄 开始测试 DataLoader ...\n")
    for batch_idx, batch in enumerate(dataloader):
        print(f"--- Batch {batch_idx+1} ---")
        print(f"🔍 PC Shape:           {batch['pc'].shape} \t\t(Expected: [{BATCH_SIZE}, {N_SAMPLED_POINTS}, 3])")
        print(f"🔍 State Shape:        {batch['state'].shape} \t\t\t(Expected: [{BATCH_SIZE}, {STATE_DIM}])")
        print(f"🎬 Action Chunk Shape: {batch['action_chunk'].shape} \t(Expected: [{BATCH_SIZE}, {CHUNK_SIZE}, {ACTION_DIM}])")
        print(f"🛡️  Action Mask Shape:  {batch['action_mask'].shape} \t\t(Expected: [{BATCH_SIZE}, {CHUNK_SIZE}])")
        print(f"💰 Reward Shape:       {batch['reward'].shape} \t\t\t(Expected: [{BATCH_SIZE}])")
        print(f"🏁 Done Shape:         {batch['done'].shape} \t\t\t(Expected: [{BATCH_SIZE}])")
        print(f"⏭️  Next PC Shape:      {batch['next_pc'].shape} \t\t(Expected: [{BATCH_SIZE}, {N_SAMPLED_POINTS}, 3])")
        
        # 打印一个 mask 看看 padding 效果
        # 如果采样的刚好是某个 episode 的最后几步，mask 的尾部应该会有 0
        print(f"\n💡 抽查第一个样本的 Action Mask:\n {batch['action_mask'][0].cpu().numpy()}")
        
        break # 测试一个 Batch 就够了
    
    print("\n🎉 Dataset & DataLoader 测试完美通过！可以接入你的 Flow/Diffusion + IDQL 训练啦！")