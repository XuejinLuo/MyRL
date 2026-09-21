# Object-budget + relational geometry

本阶段保留 1024 点 global PointNeXt、cubeA/cubeB 各 256 点预算以及原有
Flow、BC/IDQL 超参数。新增输入 `object_features`，不改 `state_dim` 或 `cond_dim`。

## 运行

从同一份 H5、相同训练设置分别重新训练：

```bash
python train_offline.py +experiment=oc_budget
python train_offline.py +experiment=oc_budget_relational
```

若 `outputs/StackCube-v1/oc_budget` 已存在，可直接使用该实验中协议一致的 baseline
checkpoint；需要重跑时给两组配置指定新的 `paths.root`，避免覆盖旧结果。
两份 experiment 配置继承同一套训练参数，只改变实验目录和关系分支。
不要单独更改其中一组的 seed、数据量、训练轮数或 checkpoint selection。

统一评估固定 100 个测试种子（默认 3000–3099），同时输出 CPS 与 ODE：

```bash
python evaluate.py comparison.allow_observation_variants=true \
  '~comparison.checkpoints' \
  '+comparison.checkpoints={budget:outputs/StackCube-v1/oc_budget/offline/checkpoints/best.pth,relational:outputs/StackCube-v1/oc_budget_relational/offline/checkpoints/best.pth}' \
  comparison.output=outputs/StackCube-v1/budget_vs_relational \
  comparison.seed_start=3000 comparison.episodes=100 video.episodes=0
```

输出目录必须尚不存在。评估从每个 checkpoint 的内嵌 config 重建模型，
无需在 evaluate 命令中开启 relational_features。旧 checkpoint 默认不带新分支，
继续严格加载；不能给旧权重临时开启新分支。新模型必须重新训练。

`summary.csv/json` 汇总 CPS/ODE 成功率、Wilson 95% 区间；各 sampler 的
`episodes.csv` 记录 per-seed 成功、最终成功、步数和回报；评估 metadata 保存
selected epoch 和 checkpoint。control、action、workspace、Flow 推理参数仍检查一致，
两组使用同一批测试 seeds，不能与 checkpoint-selection seeds 重叠。

## 特征约定

采样前全部有效、有限、workspace 内的可见分割点用于计算 geometry；启用 RGB 时，
RGB 非有限点也按原采样规则排除。每次 reset/step 都重新计算。离线使用配置的 H5 ID，
在线按物体名字重新解析当前 scene ID。没有访问模拟器的物体真实 pose。

顺序由 `env.sampling.objects` 确定，实验中固定为 cubeA、cubeB：

| 索引 | 内容 |
| --- | --- |
| 0–2 / 3–5 | A / B 可见表面 centroid |
| 6–8 / 9–11 | A / B 可见表面 axis-aligned extent |
| 12–14 / 15–17 | center A / B 减去 TCP position |
| 18–20 | center B 减去 center A |
| 21 / 22 | A / B 可见标志，float 0/1 |

不可见物体的 center、extent 以及涉及它的关系向量均为零，valid=0。
存盘中心与点云均为世界坐标；送入网络时只对有效中心减 workspace center，
缺失中心继续保留零。extent 与相对向量保持米制，不拟合额外归一化统计。
所有观测字段按 T+1 保存，transition 同时提供 `object_features` 和 `next_object_features`。

## 编码与兼容性

PointNeXt 已融合 state，新分支不重复拼 state。
23 → 64 → 64 GELU MLP 后与原 cond_dim 特征拼接，再线性映射回原 cond_dim。
融合初始化为 `[I, 0]`，bias=0；新层初始化隔离 RNG，原 PointNeXt、Flow 和 critic
初始化不发生偏移。Actor、Q、V 使用同一工厂，各自拥有独立参数。

恒等融合意味着第一步的 object MLP 梯度为零；融合的物体权重学习后，梯度才能
进入 MLP。Flow 本身也采用零初始化输出层，因此端到端梯度需要数次更新才能传入
encoder。这不是冻结分支。关闭或缺省开关时保留旧 encoder 的参数名和结构。

## 验证与范围

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest tests/test_relational.py -q
```

测试覆盖 geometry、缺失物体、RGB/空间过滤、点顺序与采样 seed 独立性、
H5/live parity、逐帧 TCP 更新、T+1/导出/dataset routing、融合初始化与梯度、
Actor/Q/V、旧 checkpoint 和 comparison 约束。

本 PR 不包含训练得到的新权重或成功率提升结论，需要在有 demonstrations 和
ManiSkill 的环境中执行上述训练与评估。第二阶段 local object encoder 暂不引入。
可选的六类 failure taxonomy 暂未增加：当前评估只可靠记录 success/最终 success，
不足以区分 dropped、unstable、not released；需要单独补充逐 primitive-step 的阶段观测，
不根据缺失信息猜测标签。当前输入仍依赖逐帧可靠 segmentation。
