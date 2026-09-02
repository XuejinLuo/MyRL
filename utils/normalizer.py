# utils/normalizer.py
import os
import json
import numpy as np
import torch
from typing import Dict, Union

class MinMaxNormalizer:
    """
    专为 Flow / Diffusion Policy 设计的 Min-Max 归一化器。
    将 Action 和 State 严格映射到 [-1, 1] 之间。
    """
    def __init__(self):
        # 存储格式: {'action': {'min': [...], 'max': [...]}, 'state': {...}}
        self.stats = {}
        self.eps = 1e-6  # 防止除以 0

    def fit(self, data_dict: Dict[str, np.ndarray]):
        """
        在训练前，根据整个离线数据集计算 min 和 max
        :param data_dict: {'action': np.ndarray(N, dim), 'state': np.ndarray(N, dim)}
        """
        for key, data in data_dict.items():
            if data.size == 0:
                continue
            self.stats[key] = {
                'min': np.min(data, axis=0).tolist(),
                'max': np.max(data, axis=0).tolist()
            }
            
    def normalize(self, data: Union[np.ndarray, torch.Tensor], key: str):
        """ 将数据归一化到 [-1, 1] """
        if key not in self.stats:
            return data # 如果没有计算过该键的统计信息，原样返回

        stat_min = self._to_same_type(self.stats[key]['min'], data)
        stat_max = self._to_same_type(self.stats[key]['max'], data)

        # 映射到 [0, 1]
        normalized = (data - stat_min) / (stat_max - stat_min + self.eps)
        # 映射到 [-1, 1]
        normalized = normalized * 2.0 - 1.0
        return normalized

    def unnormalize(self, data: Union[np.ndarray, torch.Tensor], key: str):
        """ 将网络输出的 [-1, 1] 数据反归一化到真实物理范围 """
        if key not in self.stats:
            return data
            
        stat_min = self._to_same_type(self.stats[key]['min'], data)
        stat_max = self._to_same_type(self.stats[key]['max'], data)

        # 防御性裁剪，防止模型刚开始训练时输出飞车导致物理引擎崩溃
        if isinstance(data, torch.Tensor):
            data = torch.clamp(data, -1.0, 1.0)
        else:
            data = np.clip(data, -1.0, 1.0)

        # 反映射
        unnormalized = (data + 1.0) / 2.0 * (stat_max - stat_min + self.eps) + stat_min
        return unnormalized

    def _to_same_type(self, stat_list, target_data):
        """ 内部辅助：自动对齐 numpy 和 torch 的数据类型与设备 """
        if isinstance(target_data, torch.Tensor):
            return torch.tensor(stat_list, device=target_data.device, dtype=target_data.dtype)
        else:
            return np.array(stat_list, dtype=target_data.dtype)

    def save(self, file_path: str):
        """ 保存统计信息到 json 文件 """
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=4)

    def load(self, file_path: str):
        """ 从 json 文件加载统计信息 """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到归一化配置文件: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            self.stats = json.load(f)