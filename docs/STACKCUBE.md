# StackCube

StackCube 已纳入统一任务配置，不再使用专属一键串行入口。

在 `configs/config.yaml` 选择 `task@_global_: stackcube`，分别运行 `python train_offline.py`、`python train_iterative.py`、`python train_online.py`。已有 pointcloud H5 时无需下载或重放。

完整配置、输出协议与迁移说明见 [README](../README.md) 和 [REFACTOR](REFACTOR.md)。
