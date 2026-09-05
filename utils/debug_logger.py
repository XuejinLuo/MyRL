import os
import torch
import numpy as np

class DebugLogger:
    def __init__(self, save_dir, max_steps_per_epoch=3):
        """
        初始化日志工具
        :param save_dir: 实验保存路径
        :param max_steps_per_epoch: 每个 Epoch 最多只记录前多少个 step/batch 的 I/O，防止文件爆炸
        """
        self.save_dir = os.path.join(save_dir, "debug_logs")
        self.max_steps_per_epoch = max_steps_per_epoch
        os.makedirs(self.save_dir, exist_ok=True)
        self.metrics_file = os.path.join(self.save_dir, "epoch_losses.txt")
        
        # 初始化清空指标文件
        with open(self.metrics_file, "w", encoding="utf-8") as f:
            f.write("========== 训练 Epoch 指标记录 ==========\n")
            
    def log_metrics(self, epoch, metrics_dict):
        """记录该 Epoch 的总体 Loss 指标"""
        with open(self.metrics_file, "a", encoding="utf-8") as f:
            metric_str = " | ".join([f"{k}: {v:.6f}" for k, v in metrics_dict.items()])
            f.write(f"Epoch {epoch:03d} | {metric_str}\n")
            
    def _format_tensor(self, tensor_data, log_first_sample_only=True):
        """内部辅助函数：将 Tensor 转换为精简版 Numpy 字符串"""
        if tensor_data is None:
            return "None"
            
        data_np = tensor_data.detach().cpu().numpy() if torch.is_tensor(tensor_data) else tensor_data
        
        orig_shape = list(data_np.shape)
        
        # [核心减负 1] 如果有 Batch 维度，强制只取 Batch 中的第 0 个样本
        # 判断标准：如果是离线训练，通常带有 batch，ndim >= 2。在线单步通常也有 [1, ...] 维度
        prefix_msg = ""
        if log_first_sample_only and data_np.ndim > 1:
            data_np = data_np[0]
            prefix_msg = f"  (仅展示 Batch[0], 原始 Shape: {orig_shape})\n"
            
        # [核心减负 2] precision=4 限制小数位数，节省横向空间
        data_str = np.array2string(
            data_np, 
            separator=', ', 
            precision=4,         # 仅保留 4 位小数
            threshold=np.inf, 
            suppress_small=True
        )
        return prefix_msg + "  " + data_str.replace('\n', '\n  ')

    def log_io(self, epoch, step, obs, action):
        """
        记录当前 Epoch 单步的输入输出
        """
        # [核心减负 3] 超过限制的 step 直接跳过，不写入硬盘！
        if step >= self.max_steps_per_epoch:
            return
            
        io_file = os.path.join(self.save_dir, f"epoch_{epoch:03d}_io_debug.txt")
        
        # 第一次写入用 w 清空，否则追加
        mode = "w" if step == 0 else "a"
        
        with open(io_file, mode, encoding="utf-8") as f:
            f.write(f"\n{'='*20} Step / Batch {step} {'='*20}\n")
            
            # ================= 记录输入 (Input) =================
            f.write("[Input]\n")
            if isinstance(obs, dict):
                # 图像和点云依然只输出 Shape
                if "pc" in obs and obs["pc"] is not None:
                    f.write(f"  - Point Cloud Shape: {list(obs['pc'].shape)}\n")
                if "rgb" in obs and obs["rgb"] is not None:
                    f.write(f"  - RGB Shape: {list(obs['rgb'].shape)}\n")
                
                # 本体状态输出 (利用新的精简函数)
                if "state" in obs and obs["state"] is not None:
                    f.write(f"  - Robot State Values:\n")
                    f.write(self._format_tensor(obs["state"]) + "\n")
            else:
                f.write(f"  - Unknown Obs Type: {type(obs)}\n")
                
            # ================= 记录输出 (Output) =================
            f.write("[Output]\n")
            f.write(f"  - Action Values:\n")
            f.write(self._format_tensor(action) + "\n")
            
            # 记录文件末尾提示
            if step == self.max_steps_per_epoch - 1:
                f.write(f"\n[⚠️ 提示: 已达到 max_steps_per_epoch={self.max_steps_per_epoch}，本 Epoch 后续 Step 不再记录。]\n")