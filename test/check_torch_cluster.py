import torch
from torch_cluster import fps

# 创建一个模拟的 3D 点云 (Batch=2, Points=1000, Dim=3) 放入 GPU
xyz = torch.rand(2000, 3, device='cuda')
# 创建对应的 Batch 索引
batch = torch.cat([torch.zeros(1000), torch.ones(1000)]).long().to('cuda')

# 降采样到原点数的 25% (250个点)
idx = fps(xyz, batch, ratio=0.25)

print(f"CUDA FPS 算子运行成功！降采样后的索引维度: {idx.shape}")