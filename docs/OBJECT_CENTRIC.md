# Object-Centric 3D：第一阶段实现与实验

基线固定为本次开发起点 `online@602b7ee`，没有修改或合并该分支。
新代码默认 StackCube 使用 object-centric；其他 task 保留 legacy random。
目标是检验目标物体的独立表示能否提高闭环成功率，不能从代码测试推断成功率提升。

## 输入与网络

| 字段 | 单帧形状 | 约定 |
| --- | --- | --- |
| object_points | K × M × C | local XYZ（米）+ 可选 RGB [0,1]；默认 K=2、M=256、C=6 |
| object_point_mask | K × M | bool；只标记真实源点，不复制凑数 |
| object_centers | K × 3 | 采样前全部有效可见点的均值，原始坐标系 |
| object_extents | K × 3 | 采样前可见点的轴对齐包围盒尺寸 |
| object_valid | K | bool；等于该物体 mask.any() |
| object_roles | K | manipulated / target / tool / obstacle / other 的固定枚举 |
| context_points | N × C | 默认最多 512 个非目标点，原始坐标系 |
| context_point_mask | N | bool；空场景也显式 mask |
| state | state_dim | 原始 qpos + TCP pose，StackCube 为 16 维 |

不同物体可设置不同 max_points，张量 M 取最大预算；每个 slot 仍只选自己的预算。
中心与尺寸是**可见表面统计量**，受遮挡影响，不是真实质心或完整物体尺寸。
不能恢复相机没有看到或被工作空间裁剪掉的几何。

H5 和 live 均先用 xyzw.w>0 过滤，再经过共同 builder 的有限值/工作空间过滤。
RGB 与 segmentation 始终同步索引。每个物体无放回采样，不足时保留全部真实点。
context 排除全部配置目标的 segmentation，包含机器人和其他场景点。
缺少 segmentation 直接报错；配置使用颜色时缺 RGB 也报错。
H5 ID 18/19 来自当前 StackCube 数据，换 H5 必须核对语义；代码无法凭数字判断物体名称。
实时 ID 每帧通过场景物体名称解析，reset/reconfigure 后不沿用 H5 ID。

每个 object 独立经过共享 point-wise MLP → masked max + masked mean pooling，
加入全局位置、尺寸、validity、点占用比例、role；背景单独编码；再与机器人 state token
通过两层 Transformer 融合。默认输出仍为 **[B,256]**。
物体分支没有 FPS 或后续点数缩减。所有 mask=False 的点在 MLP 前和池化时均被屏蔽。
缺失物体保留携带 role/validity 的 token，几何为零；整个场景为空也不产生全遮蔽 attention。

归一化在 dataset 和 rollout 共用：object local XYZ/extent 不再平移；
有效 object_centers 与有效 context XYZ 减去工作空间中心；无效字段始终保持零。
state/action 沿用原有 normalizer。Actor、Q、V 使用同一编码器选择逻辑、独立参数。
Flow Transformer、Flow Matching、action chunk、IDQL/PPO 更新目标均保持原实现。

## 先检查，再重训

在现有 `rl100` 环境、仓库根目录运行：

```bash
python -m tools.diagnostics.check_object_centric --max-episodes 10
python -m tools.diagnostics.overfit_object_centric --max-episodes 2 --steps 100
python train_offline.py experiment=run04_object_centric
```

两个诊断工具支持 `--data-path /path/to/trajectory.h5`。统计工具检查训练实际使用的帧，
包含 final observation，沿用 terminal/horizon 截断。输出每个物体 mean/median/p05 点数、
missing/valid rate、occupancy/padding ratio、可见中心方差，以及 context 点数。
中心方差包含真实物体运动，不应当直接解释成感知噪声。
若某个物体从未可见，先核对 ID、裁剪和视角，不要直接启动完整训练。
BC 工具固定一批输入和 flow noise/time，检查可优化性；loss 降低不等同于泛化或任务成功。

完整训练日志增加 `Object/cubeA_points`、`Object/cubeA_valid_rate`、
`Object/cubeA_padding_ratio`（cubeB 同理）及 `Context/points`。
每个 epoch 统计实际训练 minibatch；若 drop_last/shuffle，不一定等于全量 H5 统计。

## 四组对照

下面四次运行使用相同演示、seed、BC/IDQL 超参数和验证种子。为减少变量混杂，先保持默认
XYZRGB、256/256/512；只改表示与独立输出目录。不要复用旧输出目录或旧 PointNeXt 权重。

```bash
python train_offline.py experiment=oc_random env.observation.mode=global_random
python train_offline.py experiment=oc_budget env.observation.mode=global_object_budget
python train_offline.py experiment=oc_centers env.observation.encoder_variant=centers
python train_offline.py experiment=oc_points
```

`centers` 仅用分割后可见点云中心、role/validity 和 robot state，忽略点形状、颜色、extent、context。
这是 **visible-center baseline**，不是报告中建议的仿真真实 object-state oracle；
当前 H5 统一输入未保证包含真实物体位姿，因此没有把估计中心包装成 oracle。
它仍能帮助判断复杂几何编码是否必要，但其失败不能排除控制学习问题。

`env.observation.mode` 是新入口；设为 `null` 或旧 checkpoint 完全没有该字段时，才读取旧
`env.sampling.mode`。global_random/global_object_budget 的采样实现和 PointNeXt 权重结构保留。
`env.num_points=1024` 仅约束 global 模式；object 模式按 objects[].max_points/context_points 分配。
模型工厂依据 observation.mode 选择 Actor/Q/V 编码器，无须同时改 model.encoder_type。

四组训练后，在独立 3000–3099 共 100 seeds 上统一比较：

```bash
python evaluate.py comparison.allow_observation_variants=true \
  '~comparison.checkpoints' \
  '+comparison.checkpoints={random:outputs/StackCube-v1/oc_random/offline/checkpoints/best.pth,budget:outputs/StackCube-v1/oc_budget/offline/checkpoints/best.pth,centers:outputs/StackCube-v1/oc_centers/offline/checkpoints/best.pth,objects:outputs/StackCube-v1/oc_points/offline/checkpoints/best.pth}' \
  comparison.output=outputs/StackCube-v1/oc_comparison video.episodes=0
```

默认同时评估 CPS/ODE，并保存逐 seed 结果和 Wilson 区间。
命令先删除默认三阶段 checkpoint 映射，再添加四组实验，避免 Hydra 将两个映射合并。
新增开关只放宽 observation/sampling、输入通道/点数及 encoder 类型的比较限制；
动作维度、chunk、执行步数、控制模式、工作空间、normalizer、采样器参数仍必须一致。
每个 checkpoint 的完整 observation 配置分别保存在报告中。
验证/测试/迭代采集种子隔离和 checkpoint selection 协议不变。

## 数据、checkpoint 和后续阶段

### 单项消融：融合输出直接拼接 robot state

可选 `env.observation.state_skip=true` 只增加一条 state 直接通路：

```text
h = LayerNorm(Transformer(tokens)[:, 0])
condition = Linear(concat(h, normalized_robot_state))
```

保留现有 state token；直接拼接的是 dataset/rollout 共用 normalizer 产生的同一份
robot state（StackCube 为 16 维 qpos + TCP pose），不增加 TCP–物体或物体–物体相对位置。
输出维度仍是 `model.cond_dim`（默认 256）。points/centers 均支持；global 模式或
`model.use_state=false` 时开启会在读取 H5 前报错。Actor/Q/V 均沿用共享工厂、独立参数。

新 Linear 初始化为 `[I, 0]`、bias=0，初始输出等于原融合输出，state 列可训练。
该层构造不消耗后续模型初始化的 CPU RNG，因此相同 seed 下已有编码器与 Flow
backbone 的初始权重不变。新投影仅增加 `cond_dim * (cond_dim + state_dim + 1)` 个参数
（默认每个编码器 69,888 个），这是本消融的额外模型容量。

此开关默认缺省/关闭，缺省与 false 均保留原 state_dict 结构。为兼容旧配置及阶段交接，
没有向默认 YAML 注入新字段；Hydra 命令用 **`+`** 添加：

```bash
python train_offline.py experiment=oc_points_state_skip \
  env.observation.mode=object_centric env.observation.encoder_variant=points \
  +env.observation.state_skip=true
```

如果要对照 centers，将 experiment 改为 `oc_centers_state_skip`，variant 改为 `centers`。
保持各自无 skip 基线的训练轮数、演示、seed、选模和评估协议相同。
开启后需使用新实验目录重新训练，不能只在旧模型评估时打开；旧模型仍按原结构评估。
checkpoint 自动保存此开关，`evaluate.py` 用 checkpoint 内嵌配置重建网络。

仅评估新增 points 模型：

```bash
python evaluate.py '~comparison.checkpoints' \
  '+comparison.checkpoints={offline:outputs/StackCube-v1/oc_points_state_skip/offline/checkpoints/best.pth}' \
  comparison.output=outputs/StackCube-v1/oc_points_state_skip/test_best_100 \
  comparison.seed_start=3000 comparison.episodes=100 video.episodes=0
```

此命令同时报告 CPS/ODE，不需要在评估启动配置重复设置 state_skip。不同表示/skip
配置的多个 checkpoint 一起比较时，仍需 `comparison.allow_observation_variants=true`。
顶层评估 `config.yaml` 是启动设置；实际网络/观测配置见汇总 `summary.json` 的
`observations`，实际选中 epoch 见训练 `selection.json`。

后续训练继续显式传入同一开关和实验名：

```bash
python train_iterative.py experiment=oc_points_state_skip +env.observation.state_skip=true
python train_online.py experiment=oc_points_state_skip +env.observation.state_skip=true
```

在线 PPO 仍按原流程冻结整个 encoder（包括新投影）；本修改不改变训练目标、采样、
归一化、动作执行或评估规则。CPU 测试仅验证通路、梯度与阶段交接，不代表任务成功率提高。

### 原始数据与阶段交接

新导出的 manifest 使用 `myrl_primitive_v2`，物体字段为独立 NPZ 数组；所有观测保留 T+1 帧。
v1 global 数据仍可读；global/object schema 不可混用。重用原始 H5 重新 preprocessing 即可，
不要把已下采样的旧 1024 点 NPZ 转成 object-centric（丢失的点无法恢复）。
采集器、执行前缀 transition、timeout final observation、保存/读取均支持结构化观测。
checkpoint 包含完整 env.observation 配置；后续阶段继续要求观测和模型配置一致。
新 encoder 不能加载旧 PointNeXt checkpoint，需重新 offline 训练。

先完成 offline 对照，再用相同 experiment/observation 配置分别运行：

```bash
python train_iterative.py experiment=oc_points
python train_online.py experiment=oc_points
```

本实现已接通 iterative/online 数据与模型接口，不自动启动这些训练。
在线 PPO 沿用冻结 encoder 和缓存 feature 的既有流程。

## 真实部署边界与验证范围

第一阶段只使用 GT segmentation，没有加载 SAM/SAM2，也没有宣称可直接部署真实机器人。
未来视觉分割需把稳定语义物体 mask 与深度对齐，将 XYZ 变换到与训练一致的坐标系，
归一化 RGB 后送入同一 builder；处理跟踪、遮挡、mask 噪声和深度误差后再验证闭环。
物体槽位取决于配置中的名字/role，不把场景数字 ID 输入策略。

CPU 回归覆盖 H5/live reset/step 等价、场景 ID 重映射、空/稀疏物体、NaN 过滤、padding
不变性、点排列不变性、有限梯度、256 维输出、Actor/Q/V、schema 往返、归一化一致性，
以及真实 object encoder/Flow backbone 的 offline → iterative → online 小环境交接。
未访问用户真实 H5、未运行 ManiSkill/SAPIEN 渲染、完整 GPU 训练或 100-seed 真实仿真评估。
这些必须由上述本机实验提供证据，才能判断成功率提升。
