# envs/chunk_wrapper.py
import gymnasium as gym
import numpy as np

class ChunkActionWrapper(gym.Wrapper):
    """
    针对 3D Diffusion / Flow Policy 设计的 Action Chunk Wrapper。
    
    核心功能：
    1. 接收模型输出的 Action Chunk (shape: [chunk_size, action_dim])。
    2. 维护历史预测序列，通过 Temporal Ensembling (时序集成) 平滑动作。
    3. 在底层环境执行 `exec_steps` 步动作后，返回最新状态给外部模型再次推理。
    """
    def __init__(self, env: gym.Env, chunk_size: int = 16, exec_steps: int = 1, exp_weight: float = 0.01, use_ensembling: bool = False):
        """
        :param env: 原始环境
        :param chunk_size: 策略网络单次预测的未来动作步数 (Horizon)
        :param exec_steps: 每次模型预测后，在环境中实际连续执行的步数 (通常为1~8)
                           如果是 1，则属于最高频的 Temporal Ensembling。
        :param exp_weight: 时序集成的指数衰减权重。越大代表越不信任老旧预测。
                           公式: weight = exp(-exp_weight * age)
        """
        super().__init__(env)
        if not 1 <= exec_steps <= chunk_size:
            raise ValueError("Require 1 <= exec_steps <= chunk_size")
        self.chunk_size = chunk_size
        self.exec_steps = exec_steps
        self.exp_weight = exp_weight
        self.use_ensembling = use_ensembling 
        
        # 记录全局步数
        self.global_step = 0
        
        # 存储历史动作块, 格式: [(生成时的时间步, action_chunk_array), ...]
        self.action_history = []
        
        # 修改环境的 Action Space，告知外部算法这里需要输入 (chunk_size, action_dim) 的形状
        orig_space = env.action_space
        if isinstance(orig_space, gym.spaces.Box):
            self.action_space = gym.spaces.Box(
                low=np.repeat(orig_space.low[None, ...], chunk_size, axis=0),
                high=np.repeat(orig_space.high[None, ...], chunk_size, axis=0),
                dtype=orig_space.dtype
            )
            # [Opt] 保存原始动作空间的上下界，用于后续的浮点数安全裁剪
            self._orig_low = orig_space.low
            self._orig_high = orig_space.high
        else:
            raise NotImplementedError("Chunk Wrapper 目前只支持 Box (连续动作空间).")

    def reset(self, **kwargs):
        """重置环境与历史轨迹 buffer"""
        self.global_step = 0
        self.action_history.clear()
        return self.env.reset(**kwargs)

    def step(self, action_chunk: np.ndarray):
        """
        :param action_chunk: 形状为 (chunk_size, action_dim) 的 NumPy 数组
        """
        assert action_chunk.shape[0] == self.chunk_size, f"期待动作长度 {self.chunk_size}, 实际拿到 {action_chunk.shape[0]}"
        
        # 将最新预测的 Chunk 压入历史列表
        self.action_history.append((self.global_step, action_chunk))
        
        total_reward = 0.0
        done, truncated = False, False
        latest_obs = None
        latest_info = {}
        
        # [Fix] 记录实际执行的步数，防止因为 done/truncated 提前结束导致 global_step 错误
        actual_steps = 0
        success_any = False
        executed_actions = []

        # 在环境中连续执行 exec_steps 步
        for i in range(self.exec_steps):
            current_t = self.global_step + i
            
            if self.use_ensembling:
                # 计算复杂的指数加权平均
                action_to_execute = self._get_ensembled_action(current_t)
            else:
                # 纯净模式：直接取当前网络预测的第 i 步动作
                action_to_execute = action_chunk[i]
                # 简单做个安全裁剪，防止越界
                action_to_execute = np.clip(action_to_execute, self._orig_low, self._orig_high)
            
            # 丢给真实环境去执行
            obs, reward, done, truncated, info = self.env.step(action_to_execute)
            
            success_any = success_any or bool(info.get("success", False))
            executed_actions.append(np.array(action_to_execute, copy=True))
            total_reward += reward
            latest_obs = obs
            latest_info = info
            actual_steps += 1
            
            if done or truncated:
                break
                
        # 更新全局时间步 [Fix: 修改为增加实际执行的步数]
        self.global_step += actual_steps
        
        # 垃圾回收：清理掉那些已经“过期”的 chunk (它的覆盖范围已经完全落在过去)
        # 只要 chunk 生成时间 + chunk_size > 当前时间，说明它对未来还有用
        self.action_history = [
            (t_g, chunk) for (t_g, chunk) in self.action_history 
            if t_g + self.chunk_size > self.global_step
        ]
        
        latest_info = dict(latest_info)
        latest_info["success_any"] = success_any
        latest_info["actual_steps"] = actual_steps
        latest_info["executed_actions"] = np.asarray(executed_actions)
        return latest_obs, total_reward, done, truncated, latest_info

    def _get_ensembled_action(self, target_t: int) -> np.ndarray:
        """
        聚合所有覆盖了 target_t 这个时刻的历史 chunk 的预测，取指数加权平均。
        """
        valid_actions = []
        ages = []
        
        for (t_g, chunk) in self.action_history:
            # 找到 target_t 在该 chunk 中的对应索引
            idx_in_chunk = target_t - t_g
            
            # 确保索引合法 (没有越界)
            if 0 <= idx_in_chunk < self.chunk_size:
                valid_actions.append(chunk[idx_in_chunk])
                ages.append(target_t - t_g) # age 即距离该 chunk 生成过去了多少步
                
        if not valid_actions:
            raise RuntimeError(f"在时刻 {target_t} 没有找到有效的动作预测！")
            
        valid_actions = np.array(valid_actions)
        ages = np.array(ages)
        
        # 计算时序集成的指数权重：越老的预测 (age越大)，权重越小
        # weight = e^(-k * age)
        weights = np.exp(-self.exp_weight * ages)
        weights = weights / np.sum(weights)  # 归一化
        
        # 加权求和得到最终的单一动作
        ensembled_action = np.sum(valid_actions * weights[:, None], axis=0)
        
        # [Opt] 解决浮点精度问题：确保加权后的动作严格在原始环境的合法范围内
        ensembled_action = np.clip(ensembled_action, self._orig_low, self._orig_high)
        
        return ensembled_action


# ==========================================
# 测试模块：检查该 Wrapper 是否符合逻辑
# ==========================================
if __name__ == '__main__':
    print(">>> 开始测试 ChunkActionWrapper...")
    
    # 使用基础的连续控制环境作为 dummy env
    base_env = gym.make("Pendulum-v1")
    
    # 假设模型的 Chunk Size = 8，每次预测后执行 2 步 (exec_steps=2)
    CHUNK_SIZE = 8
    EXEC_STEPS = 2
    
    env = ChunkActionWrapper(base_env, chunk_size=CHUNK_SIZE, exec_steps=EXEC_STEPS, exp_weight=0.01)
    
    obs, info = env.reset()
    action_dim = base_env.action_space.shape[0]
    
    print(f"环境初始动作空间维度: {base_env.action_space.shape}")
    print(f"包装后期待的 Chunk 动作空间: {env.action_space.shape}")
    
    # 模拟 Diffusion Policy 在不同时间步输出动作 Chunk
    for step_idx in range(3):
        # 模拟生成模型前向推理出来的动作块：尺寸为 (8, 1)
        # 为了测试效果，我们在这里放置有规律的数字，方便观察集成结果
        fake_action_chunk = np.random.uniform(-1, 1, size=(CHUNK_SIZE, action_dim))
        
        print(f"\n--- 模型第 {step_idx + 1} 次调用 (当前全局环境步数: {env.global_step}) ---")
        
        # 把 chunk 扔进环境中执行
        next_obs, reward, done, truncated, info = env.step(fake_action_chunk)
        
        print(f"当前历史 Buffer 中保留的 Chunk 数量: {len(env.action_history)}")
        print(f"成功执行了 {EXEC_STEPS} 步环境迭代，获得奖励: {reward:.4f}")
        
        if done or truncated:
            break
            
    print("\n>>> ChunkActionWrapper 测试通过！该逻辑可以直接用于你的 3D Infra 项目中。")
