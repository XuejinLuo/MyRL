# MyRL

点云 Flow Policy 的三个独立训练阶段：离线初始化 → 迭代离线 IDQL → 在线 PPO。

已有 iterative 数据、希望固定 Actor 检查 Q 选动作是否有效：参见
[Q 训练与单次采样 / 8 选 1 对照](docs/Q_SELECTION.md)。独立入口为
`train_critic.py` 和 `evaluate_q_selection.py`。

## 日常运行

在现有 ManiSkill / PyTorch 环境中，从仓库根目录运行。日常参数集中在 `configs/config.yaml`：

```yaml
defaults:
  - env: maniskill
  - model: flow_3d
  - task@_global_: stackcube   # 改为 pickcube 或 pullcubetool 即可切换已有任务
  - _self_

experiment: run01              # 新实验修改这里，避免覆盖已有结果
device: cuda
```

已有点云演示时，直接分别运行：

```bash
python train_offline.py
python train_iterative.py
python train_online.py
```

每条命令只运行对应阶段，不自动启动后续阶段。无须手填上一阶段带时间戳的目录，也无须拼接长命令。

- 默认任务为 StackCube，演示路径随任务和控制模式变化，使用当前用户的 `~/.maniskill/demos/`；也可直接修改 `dataset.data_path`。
- 默认最多读取 **1000 条**演示；文件不足时读取已有条数，`null` 表示全部读取。
- `stages.offline/iterative/online` 分别控制训练轮数、batch size、保存频率和输入权重。
- 三阶段使用同一份 `eval`：默认 CPS、每 10 epoch 评估、固定 2000–2009 共 10 个种子。需要更多回合时，在 `eval.seeds` 中增加种子，三个阶段一起生效。
- 录像默认每 10 epoch 以及最后一次评估保存前 **5 个 episode**；最多不超过评估回合数。`video.every: 0` 关闭训练录像。
- 默认 offline 的 actor 使用 BC 初始化，iterative 使用 IDQL 拒绝采样（重构前最新代码两者均为 IDQL；这是重构时的默认值变化）。需要直接离线 IDQL 时，将 `stages.offline.use_bc_only` 改为 `false`。
- 算法参数分别在 `configs/algo/idql.yaml` 和 `configs/algo/pg.yaml`，网络参数在 `configs/model/flow_3d.yaml`。三阶段流程目前要求 Flow 模型。

离线阶段会自动导出本次实际读取的 primitive 演示、manifest 和归一化参数，迭代阶段直接复用，不再需要单独运行转换脚本。

若跳过迭代阶段，修改 `stages.online.source_stage: offline`，再单独运行 `python train_online.py`。使用已有权重时，修改对应的 `initial_ckpt`；迭代阶段还需要匹配的 `stats_path` 与 `dataset.manifest`。

## Object-budget + Relational 实验

保留 global PointNeXt，在 object-budget 基线上增加 23 维可见物体几何关系：

```bash
python train_offline.py +experiment=oc_budget
python train_offline.py +experiment=oc_budget_relational
```

输入定义、checkpoint 兼容性和固定 100 seeds 的 CPS/ODE 对比命令见
[Relational 实验说明](docs/OBJECT_BUDGET_RELATIONAL.md)。

## StackCube Object-Centric 3D

StackCube 默认采用 **GT segmentation → 独立物体点云 → object/context/state tokens → Flow Policy**。
cubeA、cubeB 各最多 256 个真实点，背景最多 512 个点；不足部分零填充并显式 mask。
物体分支使用 masked PointNet，不再参与全局 PointNeXt/FPS；Actor、Q、V 使用同一观测定义。
默认实验名 `run04_object_centric`，从现有 H5 重新预处理并训练，无须重新采集演示。

```bash
python -m tools.diagnostics.check_object_centric --max-episodes 10
python -m tools.diagnostics.overfit_object_centric --max-episodes 2 --steps 100
python train_offline.py
```

先确认两个物体的有效点数和 missing rate，再做完整训练。第一阶段使用 ManiSkill GT 分割，
尚未接入 SAM 或真实机器人。旧基线保留：`env.observation.mode=global_random` 或
`env.observation.mode=global_object_budget`，并选择新的 `experiment`。
显式 `observation.mode` 优先于旧 `sampling.mode`。
架构、输入约定、四组消融与固定 100 seeds 对比见 [Object-Centric 运行说明](docs/OBJECT_CENTRIC.md)。
历史全局预算方法见 [点云采样实验](docs/POINT_SAMPLING.md)。

可单独测试融合输出直接拼接 robot state（保留 state token，不增加相对位置特征）：

```bash
python train_offline.py experiment=oc_points_state_skip \
  env.observation.mode=object_centric env.observation.encoder_variant=points \
  +env.observation.state_skip=true
```

默认关闭，旧 checkpoint 继续使用原结构。开启需新实验重训；评估命令、初始化和
阶段交接说明见 [state 直接通路消融](docs/OBJECT_CENTRIC.md#单项消融融合输出直接拼接-robot-state)。

## 任务切换

只需在 `configs/config.yaml` 的 defaults 中选择任务。环境 ID、默认演示路径、输出目录、各阶段输入权重路径同步变化。

已有配置：`stackcube`、`pickcube`、`pullcubetool`，均为 Panda / `pd_ee_delta_pose` 的起始配置，不代表各任务已经验证成功率。

新增任务时复制一份 `configs/task/*.yaml`，修改 `env.env_id`，按实际任务覆盖 horizon、工作空间、机器人 state/action 维度等。颜色通道变化时，`env.use_color` 与 `model.in_channels` 必须对应。共用的点云与控制设置放在 `configs/env/maniskill.yaml`。无需修改 Python 训练代码。

## 统一输出

输出位置固定为 `outputs/<环境 ID>/<experiment>/<阶段>/`。例如 `outputs/StackCube-v1/run01/offline/`。

| 文件 | 三个阶段的共同含义 |
| --- | --- |
| `config.yaml` | 本次运行解析后的完整配置快照 |
| `metrics.jsonl` | 每个 epoch 一行，含 stage、round、epoch、sampler、checkpoint、evaluation 和指标 |
| `metrics.csv` | 同一批记录的表格形式；暂未产生的指标为空 |
| `selection.json` | 入选权重、epoch、指标及选择标准 |
| `checkpoints/best.pth` | 验证集最优策略；成功率优先，相同时比较平均 reward |
| `checkpoints/last.pth` | 最近保存的策略；迭代根目录保存最终接受的策略 |
| `checkpoints/dataset_stats.json` | 与模型绑定的归一化参数 |
| `eval/validation_epXXXX_cps/summary.json` | 评估种子、sampler、环境、checkpoint 及汇总指标 |
| `eval/validation_epXXXX_cps/episodes.csv` | 每个 seed 的 success、final success、return、primitive steps |
| `eval/validation_epXXXX_cps/videos/` | 该次评估的视频 |

迭代训练的每一轮位于 `iterative/round_000/` 等目录，包含相同的评估与 checkpoint 结构；根目录 `metrics.*` 汇总所有轮次，以 `(round, epoch)` 对应。每次评估的权重都保存为 `checkpoints/epoch_XXXX.pth`，不会只留下无法复现的指标。

共同指标使用相同单位：`Eval/Success_Rate` 为 0–1，表示 episode 内任一 primitive step 成功；同时报告最终时刻成功率、成功次数、回合数、平均 reward 和 Wilson 95% 区间。`loss/*`、`ppo/*` 等算法专属指标保留原义，不能拿 PPO loss 与 IDQL loss 直接比较。

## 独立最终对比

三个阶段完成后，另行运行：

```bash
python evaluate.py
```

配置来自同一个 `config.yaml` 的 `comparison`。默认读取三个阶段的 `best.pth`，在独立的 3000–3099 共 100 个种子上分别评估 CPS 和 ODE，保存 `comparison/summary.csv`、`summary.json` 和每个模型的逐 episode 结果。不把最终测试成绩反馈给模型选择。

只评测某个阶段时，在 `comparison.checkpoints` 中只保留该阶段。评估前会检查模型、环境、归一化、sampler 参数是否兼容，并拒绝与验证/迭代采集种子重叠的测试集。重复对比请修改 `comparison.output`，已有输出不会被覆盖。最终对比录像数量由 `video.episodes` 控制，设为 0 关闭。

## 续跑与已有实验

- 离线阶段没有优化器续训接口；新训练使用新的 `experiment`。
- 迭代阶段设置 `stages.iterative.resume: true`，在原输出目录继续。完成的轮次跳过；中断轮次从该轮初始模型重训，复用已保存的完整采集 episode。指标会替换中断轮次的旧记录。除 resume 开关外，配置必须一致。
- 在线续训：设置 `stages.online.resume` 为某次 `checkpoints/last.pth` 或 `epoch_XXXX.pth`，并选择新的 `experiment`；`stages.online.epochs` 是续训后目标总 epoch。恢复优化器，但仿真从 epoch 边界 reset，不是逐 step 精确续跑。
- `best.pth` 用于初始化和评估；在线优化器续训需要 `last.pth` 或周期权重。
- 旧 checkpoint 的 EMA/raw 权重读取保留兼容；旧环境、模型或归一化不兼容时会明确报错。旧版不含配置/normalizer 的 checkpoint 不支持新的统一最终对比。

这次统一了离线与迭代的数据语义，旧离线训练曲线不能直接视作同一实现的复跑。具体变化见 [迁移说明](docs/REFACTOR.md) 与 [CHANGELOG](CHANGELOG.md)。

## 代码职责

| 位置 | 职责 |
| --- | --- |
| 根目录的 4 个 Python 文件 | 三个独立训练入口、一个独立评估入口 |
| `workflows/` | 各阶段调度；`offline_round.py` 是 offline/iterative 共用的 IDQL 训练循环 |
| `data/demonstrations.py` | H5 读取与 primitive 演示导出 |
| `data/episodes.py` | 轨迹校验、manifest、哈希及执行前缀 transition |
| `data/dataset.py` | 两个离线阶段共用的 `TrajectoryDataset` |
| `data/pointcloud.py` | 演示与实时观测共用的裁剪和采样 |
| `data/object_centric.py`、`data/observations.py` | 物体点云构建、观测路由与统一归一化 |
| `models/encoders/object_centric.py` | 带 mask 的物体/背景编码与 token 融合 |
| `evaluation/` | 固定种子评估、录像和最终对比 |
| `envs/factory.py` | 三阶段共用的 ManiSkill 环境构建 |
| `models/factory.py`、`models/checkpoint.py` | 模型/观测编码构建、checkpoint 协议 |
| `utils/` | 配置校验、归一化、实验记录 |
| `tools/` | 演示准备和人工诊断，不参与训练调度 |
| `examples/` | 历史蒸馏演示及专用旧 Dataset，不参与三个训练阶段 |
| `tests/` | CPU 回归与集成测试 |

已有 pointcloud H5 时不需要准备演示。缺少时可用 `python -m tools.prepare_demos`，通过 `--task` 选择任务；该工具保留 ManiSkill 官方下载/重放流程。

## 验证范围

使用原有环境中的 PyTorch、Hydra/OmegaConf、NumPy、h5py、Gymnasium、tqdm、imageio，以及 pytest：

```bash
python -m pytest -q tests/test_object_centric.py tests/test_point_sampling.py tests/test_stages.py tests/test_experiment.py tests/test_flow_ppo.py tests/test_iterative_data.py tests/test_primitive_recording.py
```

CPU 测试包含三个阶段的真实优化器更新、小型模拟环境、权重衔接、统一输出和迭代续跑；不等同于完整 ManiSkill GPU 训练。本次没有测量 StackCube 等任务的成功率，也没有验证真实渲染。
上述命令排除了 `tests/` 内依赖本机 H5 路径的历史人工诊断脚本。

### DataLoader 崩溃与性能诊断

默认 `num_workers: 0`，不启动数据加载子进程。数据已预载入内存；这是规避 SAPIEN/CUDA 初始化后 fork worker 导致段错误、同时避免 spawn 复制大规模数据的保守默认值，不保证它在所有机器上最快。

如有足够 CPU 内存，可在配置中尝试 `num_workers: 1` 或 `2`；代码固定采用 `spawn` 和常驻 worker，不再使用系统默认的 fork。内存中的轨迹会复制到每个 spawn worker，应比较实测吞吐后决定 worker 数。

两个离线阶段新增 `Time/Train_Seconds`（包含数据加载、传输和更新）、`Time/Eval_Seconds`、`Time/Artifact_Seconds`（权重切换/保存/恢复）与 `Train/Samples_Per_Second`。GPU 在 epoch 计时边界同步；epoch 总耗时不包含末尾日志写入。BC/IDQL、batch size、数据量和采样接受率不同时，不能只按 GPU 占用率比较速度。

已有输出不会被覆盖；崩溃后重跑需使用新的 `experiment`。当前 offline checkpoint 没有优化器状态，不支持从崩溃 epoch 精确续训；保留已保存的权重用于评估或后续阶段。
