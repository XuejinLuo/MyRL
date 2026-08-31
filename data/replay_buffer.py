# data/replay_buffer.py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple

class ActionChunkReplayBuffer:
    """
    专为 PointCloud + Chunk Action + PG/IDQL 设计的在线 Replay Buffer。
    采用扁平化存储，但在采样时自动构建 Action Chunk，避免内存中存在大量重复动作数据。
    """
    def __init__(
        self, 
        capacity: int, 
        num_points: int, 
        pc_dim: int, 
        state_dim: int, 
        action_dim: int, 
        chunk_size: int,
        device: str = "cpu"
    ):
        self.capacity = capacity
        self.chunk_size = chunk_size
        self.device = device
        
        # 指针与容量追踪
        self.ptr = 0
        self.size = 0
        
        # State: 3D Point Cloud + Proprioception (本体感觉/低维状态)
        self.obs_pc = np.zeros((capacity, num_points, pc_dim), dtype=np.float32)
        self.obs_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_obs_pc = np.zeros((capacity, num_points, pc_dim), dtype=np.float32)
        self.next_obs_state = np.zeros((capacity, state_dim), dtype=np.float32)
        
        # Action, Reward, Done
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        # done 标志用于 Chunk 采样时判断是否越界（Episode结束）
        self.dones = np.zeros((capacity, 1), dtype=bool)

    def add(self, obs_pc, obs_state, action, reward, next_obs_pc, next_obs_state, done):
        """添加单步 transition，Chunk 会在 sample 时动态生成"""
        self.obs_pc[self.ptr] = obs_pc
        self.obs_state[self.ptr] = obs_state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs_pc[self.ptr] = next_obs_pc
        self.next_obs_state[self.ptr] = next_obs_state
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _get_chunk_action(self, start_idx: int) -> np.ndarray:
        """
        获取从 start_idx 开始的长度为 chunk_size 的 action。
        如果期间遇到 done==True (即episode结束)，则用最后一个有效动作进行 Padding。
        """
        chunk = np.zeros((self.chunk_size, self.actions.shape[-1]), dtype=np.float32)
        
        # [修复1] 增加一个 has_done 标志位，防止跨越 episode 边界
        has_done = False 
        
        for i in range(self.chunk_size):
            curr_idx = (start_idx + i) % self.capacity
            
            # [修复2] 防止读取到未来的脏数据或跨越环形缓冲区的新旧边界
            if i > 0 and curr_idx == self.ptr:
                has_done = True
                
            # 如果当前步越界或者上一步已经是 done，则 padding 最后一个有效动作
            if i > 0 and self.dones[(start_idx + i - 1) % self.capacity]:
                has_done = True # 一旦遇到 done，后续全部变为 padding 状态
                
            if has_done:
                chunk[i] = chunk[i - 1] # Action Padding
            else:
                chunk[i] = self.actions[curr_idx]
                
        return chunk

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """随机采样 Batch，返回带有 Chunk Action 的 Tensor 字典"""
        idxs = np.random.randint(0, self.size, size=batch_size)
        
        batch_obs_pc = self.obs_pc[idxs]
        batch_obs_state = self.obs_state[idxs]
        batch_next_obs_pc = self.next_obs_pc[idxs]
        batch_next_obs_state = self.next_obs_state[idxs]
        batch_rewards = self.rewards[idxs]
        batch_dones = self.dones[idxs]
        
        # 动态提取 Action Chunk
        batch_action_chunks = np.array([self._get_chunk_action(idx) for idx in idxs])
        
        # 转换为 PyTorch Tensor 并送到指定 Device
        return {
            "obs_pc": torch.FloatTensor(batch_obs_pc).to(self.device),
            "obs_state": torch.FloatTensor(batch_obs_state).to(self.device),
            "action_chunk": torch.FloatTensor(batch_action_chunks).to(self.device),
            "reward": torch.FloatTensor(batch_rewards).to(self.device),
            "next_obs_pc": torch.FloatTensor(batch_next_obs_pc).to(self.device),
            "next_obs_state": torch.FloatTensor(batch_next_obs_state).to(self.device),
            "done": torch.FloatTensor(batch_dones).to(self.device),
        }


class OfflineChunkDataset(Dataset):
    """
    用于离线预训练 (BC / IDQL) 的 PyTorch Dataset。
    如果你有存好的 HDF5 或 NPZ 文件，可以将其加载并在这里包装。
    """
    def __init__(self, data_path: str = None, chunk_size: int = 16):
        super().__init__()
        self.chunk_size = chunk_size
        
        # 这里用 dummy 数据模拟离线数据加载，实际使用时替换为 np.load(data_path)
        print(f"Loading offline dataset from {data_path}...")
        self.num_samples = 1000
        self.obs_pc = np.random.randn(self.num_samples, 1024, 3).astype(np.float32)
        self.obs_state = np.random.randn(self.num_samples, 14).astype(np.float32)
        self.actions = np.random.randn(self.num_samples, 7).astype(np.float32)
        self.dones = np.zeros((self.num_samples, 1), dtype=bool)
        
        # 随机设置几个 done=True 模拟 Episode 分界
        self.dones[99::100] = True 

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        # 提前切出 Action Chunk
        action_chunk = np.zeros((self.chunk_size, self.actions.shape[-1]), dtype=np.float32)
        
        # [修复1 同理] 需要 has_done 标志
        has_done = False
        
        for i in range(self.chunk_size):
            curr_idx = min(idx + i, self.num_samples - 1)
            
            if i > 0 and self.dones[curr_idx - 1]:
                has_done = True
                
            if has_done:
                action_chunk[i] = action_chunk[i - 1] # Padding
            else:
                action_chunk[i] = self.actions[curr_idx]
                
        return {
            "obs_pc": torch.FloatTensor(self.obs_pc[idx]),
            "obs_state": torch.FloatTensor(self.obs_state[idx]),
            "action_chunk": torch.FloatTensor(action_chunk)
        }



# ==========================================
# 🛠️ Main 函数测试模块
# ==========================================
if __name__ == "__main__":
    print("🚀 开始测试 Chunk Action Replay Buffer & Dataset 🚀\n")

    # 参数设置
    CAPACITY = 5000
    NUM_POINTS = 1024   # 点云降采样后的点数 (通常1024或2048)
    PC_DIM = 3          # 点云特征维度 (xyz)
    STATE_DIM = 14      # 机器人本体状态维度 (如关节角+末端位置)
    ACTION_DIM = 7      # 动作维度 (如6D位姿+抓手)
    CHUNK_SIZE = 16     # Diffusion Policy 生成的 Action Chunk 长度
    BATCH_SIZE = 32

    # 1. 测试在线 Replay Buffer (用于 PG / 强化微调)
    print("-" * 50)
    print("🧪 1. Testing Online ActionChunkReplayBuffer...")
    buffer = ActionChunkReplayBuffer(
        capacity=CAPACITY,
        num_points=NUM_POINTS,
        pc_dim=PC_DIM,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        chunk_size=CHUNK_SIZE,
        device="cpu"
    )

    # 模拟环境交互，塞入 100 步 Transition (设第49步和99步为 Episode 结束)
    for step in range(100):
        dummy_pc = np.random.randn(NUM_POINTS, PC_DIM)
        dummy_state = np.random.randn(STATE_DIM)
        dummy_action = np.ones(ACTION_DIM) * step  # 动作值设为 step 方便观察 Padding
        dummy_reward = np.random.rand(1)
        done = (step == 49 or step == 99)

        buffer.add(
            obs_pc=dummy_pc, obs_state=dummy_state, action=dummy_action, 
            reward=dummy_reward, next_obs_pc=dummy_pc, next_obs_state=dummy_state, done=done
        )

    # 抽样并检查维度
    batch = buffer.sample(BATCH_SIZE)
    print("✅ Buffer Sampled Successfully!")
    print(f"   -> obs_pc shape:       {batch['obs_pc'].shape}  (Expected: [{BATCH_SIZE}, {NUM_POINTS}, {PC_DIM}])")
    print(f"   -> obs_state shape:    {batch['obs_state'].shape}     (Expected: [{BATCH_SIZE}, {STATE_DIM}])")
    print(f"   -> action_chunk shape: {batch['action_chunk'].shape}  (Expected: [{BATCH_SIZE}, {CHUNK_SIZE}, {ACTION_DIM}])")
    
    # 检查其中一个 sample 是否正确 Padding 了 (边界测试)
    print(f"   💡 随机检查第一个 sample 的 action_chunk 前 5 步:\n{batch['action_chunk'][0, :5, 0].cpu().numpy()}")


    # 2. 测试离线 Dataset (用于 BC / IDQL 预训练)
    print("\n" + "-" * 50)
    print("🧪 2. Testing OfflineChunkDataset with PyTorch DataLoader...")
    dataset = OfflineChunkDataset(data_path="dummy_path.npz", chunk_size=CHUNK_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # 迭代一个 Batch
    for offline_batch in dataloader:
        print("✅ Dataloader Sampled Successfully!")
        print(f"   -> offline obs_pc shape:       {offline_batch['obs_pc'].shape}")
        print(f"   -> offline action_chunk shape: {offline_batch['action_chunk'].shape}")
        break  # 只测试一个 batch

    print("\n🎉 所有数据流测试通过，此 Buffer 可以直接对接到你的 3D Diffusion/Flow 模型中！")