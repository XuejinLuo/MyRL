# StackCube 目标点预算对照实验

目的：检验小物体在全局随机降采样中丢失，是否限制离线 BC 的闭环成功率。
本实验使用仿真 segmentation 辅助采样；ID 不输入网络。它不是纯视觉分割方案，
也不能恢复相机未看到或已被工作空间裁剪掉的点。

## 默认行为

`configs/task/stackcube.yaml` 默认启用：

```yaml
env:
  sampling:
    mode: object_budget
    objects:
      - {name: cubeA, h5_id: 18, num_points: 256}
      - {name: cubeB, h5_id: 19, num_points: 256}
```

总输入仍为 1024 个 XYZRGB 点。先在有效点和工作空间裁剪后，分别为 A/B
无放回保留 `min(可见点数, 256)` 个点，再从所有尚未选中的点中无放回填满剩余预算。
因此 256 是预留预算，不是最终点数上限，回填可能再选到目标点。
目标不足预算时全部保留，不为了凑满单个物体预算而复制点。
只有整帧裁剪后少于 1024 点时，先保留全部源点，再重复填充到固定大小；空帧输出零。
合并后打乱顺序，避免编码器按输入索引截断邻域时偏向先拼接的物体。

- 保持网络、优化器、BC 设置、点数、工作空间、训练轮数和评估设置不变。
- 本次不新增 TCP 局部配额、FPS、多相机或动态训练重采样，先验证物体预算这一项变化。
- H5 使用显式配置的 `h5_id`。18/19 来自用户当前诊断脚本，换演示文件时必须核对。
  H5 中的裸分割 ID 本身不包含物体名称；代码无法仅凭数字证明语义对应正确。
- 实时观测使用 `segmentation_id_map` 按 `cubeA`/`cubeB` 名称解析 ID，reset 后重新解析，
  不沿用 H5 的数字。H5 和实时观测共用同一个裁剪/采样函数。
- 缺 segmentation 或实时名称无法唯一匹配时明确报错，不静默退回随机采样。
- 其他任务默认保持 `random`。没有 sampling 字段的旧 checkpoint 评估也保持原采样行为。
- 新配置随 checkpoint 和数据 manifest 一起保存；后续阶段的现有兼容性检查会拒绝混用不同预处理配置。

## 在本机运行

先激活已有 ManiSkill / PyTorch 环境，在仓库根目录运行。无需重新下载或重新录制演示。

1. 比较同一批 H5 帧的新旧采样结果：

   ```bash
   python -m tools.diagnostics.check_point_sampling --max-episodes 100 --frame-step 5
   ```

   路径、分割 ID、预算和工作空间来自当前配置。可通过 `--data-path /path/to/trajectory.h5`
   指定另一个 H5；对应 ID 在 task 配置中修改。
   工具输出“裁剪后、采样前”统计、随机采样统计、预算采样统计，
   并统计“采样前可见、采样后消失”的帧数。计数去除重复填充，反映独立源点数。
   为与训练一致，统计在第一处真实终止或 horizon 处截断；旧 `test_check_cube_points.py`
   统计完整轨迹，因此聚合数值未必完全相同。

   预期：预算采样中，每个目标至少保留 `min(裁剪后可见点数, 256)` 个独立点，
   “采样前可见、采样后消失”应为零。若采样前就为零，检查遮挡、裁剪或 ID。

2. 开始新的离线训练：

   ```bash
   python train_offline.py
   ```

   `configs/config.yaml` 的默认实验名已改为 `run03_object_budget`，避免覆盖 `run02`。
   offline 从 `dataset.data_path` 指定的原始 H5 重新生成采样输入，不读取旧 NPZ。
   新输入及其采样配置导出到：

   ```text
   outputs/StackCube-v1/run03_object_budget/offline/data/
   ```

   每帧仍在 H5 加载时采样一次，之后各轮复用；这次不同时引入动态重采样。
   如果该输出目录已存在，请修改 experiment 后重跑。

3. 如需重新训练同代码的随机采样对照组：

   ```bash
   python train_offline.py experiment=run03_random_control env.sampling.mode=random
   ```

   两组保持相同训练种子、演示数、网络、训练轮数和评估种子。只有采样策略不同。
   如扩大验证种子集，两组都应使用相同配置。

4. 独立评估时分别读取各自 checkpoint 的采样设置：

   ```bash
   python evaluate.py 'comparison.checkpoints={offline:outputs/StackCube-v1/run03_object_budget/offline/checkpoints/best.pth}' comparison.output=outputs/StackCube-v1/run03_object_budget/test_offline
   python evaluate.py 'comparison.checkpoints={offline:outputs/StackCube-v1/run03_random_control/offline/checkpoints/best.pth}' comparison.output=outputs/StackCube-v1/run03_random_control/test_offline
   ```

   默认都在 3000–3099 共 100 个测试种子上评估 CPS 和 ODE，查看各自 `summary.csv`
   和逐 episode 结果。现有比较器拒绝在一次调用中混用不同 env 配置，所以分开调用。
   也可将随机组路径替换为已有 `run02` 的 checkpoint；评估会使用旧 checkpoint
   保存的随机采样配置。用验证集选择权重，最终测试用于报告结果。

## 验证范围

新增测试覆盖稀少目标保留、缺失目标、裁剪/无效点、固定大小填充、分组打乱、
随机模式兼容，以及实际 H5 loader 与 live reset/step 包装器在不同分割 ID 下的一致性。
合成 H5 导出与 manifest 也经过往返检查。

CPU 回归命令（不包含依赖用户本机 H5 的历史人工诊断脚本）：

```bash
python -m pytest -q tests/test_point_sampling.py tests/test_stages.py tests/test_experiment.py tests/test_flow_ppo.py tests/test_iterative_data.py tests/test_primitive_recording.py
```

本次开发环境没有用户的真实 H5 和 ManiSkill/SAPIEN，未执行真实仿真或 GPU 训练。
采样保留通过测试不代表任务成功率一定提高；需要上述本机对照实验验证。
