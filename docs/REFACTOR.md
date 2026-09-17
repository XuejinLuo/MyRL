# 2026-09-17 结构与运行方式迁移

基于默认分支 `online` 的 `424ee29`。三个训练入口保持独立，参数通过 `configs/config.yaml` 和专用 task/algo/model 配置控制。

## 路径迁移

| 原位置 | 新位置 / 处理 |
| --- | --- |
| 根目录 `train_*.py` 中的实现 | `workflows/offline.py`、`iterative.py`、`online.py`；根目录保留轻量入口 |
| `run_stackcube.py` | 删除；不再串行启动全部阶段 |
| `evaluate.py`、`evaluate_online.py`、`evaluate_checkpoint.py` | 根目录仅保留配置驱动的 `evaluate.py`；实现位于 `evaluation/compare.py` |
| `utils/online_eval.py`、`utils/eval_utils.py` | 统一为 `evaluation/runner.py`，删除旧的另一套环境/评估逻辑 |
| `utils/online_env.py`、`utils/eval_video.py` | 分别迁至 `envs/factory.py`、`evaluation/video.py` |
| `utils/online_checkpoint.py` | `models/checkpoint.py` |
| `data/maniskill_dataset.py`、`prepare_iterative_data.py` | `data/demonstrations.py`；offline 自动导出本次数据，无独立转换命令 |
| `data/dataset.py`、`data/iterative_dataset.py` | 主流程只使用 `data/dataset.py` 的 `TrajectoryDataset`；旧 Dataset 仅随蒸馏演示放在 `examples/` |
| `data/iterative_store.py` | `data/episodes.py` |
| `data/replay_buffer.py` | 删除：三个训练阶段没有使用它 |
| `utils/debug_logger.py` | 删除：统一使用实验指标；旧 debug IO 文件不再默认生成 |
| `configs/env/stackcube.yaml` | 删除重复环境副本；任务差异集中在 `configs/task/` |
| `configs/dataset/offline_data.yaml`、`configs/eval.yaml` | 合并到 `configs/config.yaml` |
| `prepare_stackcube_demos.py`、`generate_prompt.py` | 移入 `tools/`；演示准备支持任务参数 |
| `distill_policy.py` | `examples/distill_policy.py`，不是主训练入口 |
| `test/` | 自动测试移入 `tests/`，人工检查移入 `tools/diagnostics/` |

旧命令行入口和模块 import 路径不再全部保留，避免再增加一层兼容文件。旧权重读取保留 EMA/raw 选择；旧结果目录不改写。

## 影响算法输入的变化

此次不只移动文件，以下变化需要在对比旧实验时考虑：

1. **统一 transition 语义。** 原 offline 的 Q 输入完整动作 chunk，但使用单步 reward/next state；现在与 iterative 共用执行前缀 Q、折扣累计 reward、真实前缀末端观察和 `gamma ** actual_steps`。
2. **区分终止与超时。** 真实 terminal 不 bootstrap，timeout 从真实最终观察 bootstrap。H5 必须提供 T+1 观察及 success/terminated/truncated 标签，不再把最后一步强行当成功或 terminal。
3. **保留既有迭代导入的 horizon 处理。** 演示截到第一个真实 terminal、当前任务 horizon 或文件结束；历史持续为真的 timeout 标记不当作每步 terminal，截断末端按 timeout 标记。
4. **统一点云预处理。** 演示与在线观测共用裁剪、随机采样和颜色拼接；排除齐次坐标 w=0 的无效点与非有限值。缺少 TCP pose 明确报错。
5. **BC 初始化成为直接 offline 的默认。** 这是相对重构前最新代码的改变：`424ee29` 的直接入口及 StackCube 总入口都为 IDQL，先前称“与旧流程一致”不准确。需要离线 IDQL 时修改 `stages.offline.use_bc_only: false`。
6. **权重与指标绑定。** 两个离线阶段保存的 `model_state_dict` 就是评估过的 EMA 策略；不再要求下游猜测 raw/EMA。旧 checkpoint 中的 `ema_model_state_dict` 仍支持读取。
7. **统一选择规则。** 成功率优先，平均 reward 打破平局；迭代候选与本轮 incumbent 使用相同验证种子比较。无改进保留 incumbent，但采集数据继续累积。
8. **在线 PPO 优化公式与默认超参数未改。** 删除了训练入口中特有的额外 ODE baseline；独立 `evaluate.py` 统一提供 CPS/ODE 对比。

新数据不覆盖旧 manifest；归一化在离线阶段拟合，后续冻结。旧数据的少量越界 episode 排除规则现在由 `dataset.max_rejected_episodes` 和 `max_rejected_fraction` 显式控制，默认仍是最多 3 条且不超过 1%。

## 对应关系

每个 metrics 行的 `checkpoint` 指向该行实际保存的 epoch 权重，`evaluation` 指向对应逐 episode 和汇总结果；非保存/非评估 epoch 的字段为 null。跨阶段以同一 seed、sampler、success 定义比较；迭代额外使用 round 标识。loss、PPO KL 等不同算法指标不强行改名为同一个指标。

迭代根目录的 `last.pth` 表示最后接受的模型；每轮 `last.pth` 是该轮最后训练的候选。在线 `last.pth` 还包含优化器，支持新输出目录下的 epoch 边界续训。

## 验证

26 项 CPU 回归/集成测试通过，覆盖 PPO 更新、超时 bootstrap、数据完整性、评估与视频 RNG 隔离、三任务配置、三阶段真实参数更新与权重衔接、指标指向、迭代已完成轮次续跑，以及独立 CPS/ODE 对比。完整 ManiSkill、GPU、真实视频编码与实际任务成功率尚未验证。
