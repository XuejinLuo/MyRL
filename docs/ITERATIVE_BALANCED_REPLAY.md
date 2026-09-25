# Iterative：固定更新预算与分开的 Actor/Critic 采样

目的：复用现有数据，检验 Actor 学习哪些动作比单纯扩大数据集更重要。
从当前保留的 iterative best 开始，不重新训练 offline，不增加 rollout，不使用 Q8。
本改动是待验证的实验方案，不保证成功率提升。

## 直接运行两组受控实验

在仓库根目录、原 ManiSkill 环境执行。两个命令读取同一份 Actor、normalizer 和完整的
`iterative_expand600/round_000/manifest.json`，默认输出到两个新目录。
原文件及 manifest 中的 episode 不会修改。已有输出目录拒绝覆盖。

对照组：Actor 使用与 Critic 相同的、全数据 transition 均匀采样 batch。

```bash
python train_iterative.py +experiment=oc_budget +iterative_protocol=balanced_replay \
  actor_sampling.mode=mixed \
  output=outputs/StackCube-v1/oc_budget/iterative_replay_control
```

平衡组：Actor 每个 batch 在拒绝采样前固定为 32 个 demonstration 样本和 32 个成功
rollout 样本；在各组内部先均匀采 episode，再均匀采时间步。

```bash
python train_iterative.py +experiment=oc_budget +iterative_protocol=balanced_replay
```

第二条默认输出 `outputs/StackCube-v1/oc_budget/iterative_balanced_replay`。
两个实验是一个预算、两个采样方案；不要把本次平衡组直接与此前不同训练预算的 BC/IDQL
训练日志比较来归因。采样方案同时改变来源比例、rollout 成功筛选和 episode 长度权重，
若有效，再拆分消融，不能从这一次比较认定是哪一个因素单独起作用。

## 固定配置

| 项目 | 两组共同设置 |
|---|---|
| Actor 来源 | `outputs/StackCube-v1/oc_budget/iterative/checkpoints/best.pth` |
| normalizer | `outputs/StackCube-v1/oc_budget/iterative/checkpoints/dataset_stats.json`，不重新拟合 |
| 数据 | `outputs/StackCube-v1/oc_budget/iterative_expand600/round_000/manifest.json` |
| 收集新数据 | `collect=false`，仅支持 `rounds=1` |
| 每轮 Critic 更新 | `updates_per_round=20000` |
| 仅训练 Critic 的前缀 | `critic_warmup_updates=10000` |
| Actor 更新 | 后 10000 次，每次一次 Actor optimizer step |
| Batch size | 64；拒绝采样后 Actor 的有效 batch 会变小 |
| 学习率 | Actor 1e-5，Critic 3e-4 |
| Actor 规则 | 保留 Flow Matching 和现有优势拒绝采样；`use_bc_only=false` |
| 观测、动作 | 保留 checkpoint 匹配配置，不改点云表示、normalizer、chunk/prefix |
| Critic 数据 | 所有通过既有校验的数据，含失败 episode；transition 均匀、有放回采样 |
| 日志 | 每 1000 次更新；warmup 结束、评估/保存事件和最后一步也记录 |
| 评估/保存 | warmup 后每 2000 次更新，末次必执行 |
| 验证场景 | 2000–2049；仅用于选模，不用 4000/5000 诊断场景选模 |
| 模型选择 | 验证成功率严格提高才替换；同成功率保留更早模型，包含 step 0 基线 |

该预算是首轮实验起点，不是已优化的超参数。固定预算模式不使用 `epochs`、
`critic_warmup_epochs`、`eval.every` 或 `save_epoch` 决定更新/评估时机。
Critic 总步数**包含** warmup，Actor 总步数为二者之差。
如需改预算，两组必须同时修改上述 update 字段。
模型选择在本模式下不再使用环境累计奖励作同成功率的 tie-break；旧 epoch 模式保留原行为。

文件路径不同可覆盖 `initial_ckpt=... stats_path=... manifest=...`。manifest 必须已包含
需要使用的 demonstrations 和 rollout，`collect=false` 不会追加、复制或重复登记数据。
会拒绝已记录的采集 seed 与验证 seed 重叠；未记录 seed 的历史 demonstrations 不能据此
自动验证场景独立性。

## 采样与统计的准确含义

- `actor_sampling.demo_sources=[demonstrations]` 明确指定 manifest 中哪些 source 名称属于
  demonstration；其余 source 作为 rollout。自定义 manifest 必须明确配置正确名称。
- 成功 rollout 根据实际 episode 的 `success.any()` 判定，不依赖可能过时的 manifest 注释。
  选中成功 episode 后采样其完整有效时间范围，不表示该轨迹每个动作都最优。
- 缺少 demonstrations 或成功 rollout 时平衡模式报错，不自动补失败样本或切换算法。
- 现有越界 episode 剔除规则仍生效；source 映射在剔除后保留正确对应关系，并记录剔除索引。
- 50/50 是**拒绝采样前**的比例；Q/V 优势拒绝采样会改变最终保留比例。固定 Actor optimizer
  step 数不等于固定被接受的样本数，必须结合 kept 数量分析。
- Critic 与平衡 Actor 使用独立的本地采样 RNG；两组 Critic 的抽样索引在相同数据/种子下
  相同。不承诺不同 Actor 方案的整个训练随机数序列逐步相同。
- 两组固定预算路径都在当前 Critic 更新后，以 eval 模式的 target Q/V 重新计算 Actor
  batch 的优势，避免额外前向更新统计缓冲区。保留 `min(Q1,Q2)-V`、beta、拒绝概率和 loss。
- Q/V 仍每轮从头初始化，未继承已有 frozen-Q checkpoint；未改变稀疏 reward、terminated/
  truncated 或 bootstrap 规则。此实验不能直接回答 timeout bootstrap 是否引起价值高估。

## 结果文件

每个新输出目录包含：

- `config.yaml`、`provenance.json`：冻结配置和源 Actor/manifest/normalizer 文件哈希。
- `round_000/sampling.json`：通过校验的数据各来源 episode/transition 数、成功数、剔除索引、
  source 编号映射、采样和预算定义。
- 根目录及 `round_000/metrics.csv` / `metrics.jsonl`：实际更新计数和分来源统计。
- `round_000/checkpoints/step_XXXXXXX.pth`：按 update_step 命名的 Actor checkpoint。
- `round_000/eval/validation_stepXXXXXXX_cps/episodes.csv` 与 `summary.json`：逐场景评估。
- `selection.json`、`checkpoints/best.pth`：含 step 0 的最终选模；没有提高时保留输入 Actor。

固定预算日志及 checkpoint 的 `epoch=null`，以 `update_step` 和 `budget_unit=updates`
解释训练进度。`Train/*_Updates` 是本条日志区间次数，`*_Updates_Total` 是轮内累计次数。
`Actor/{demo,rollout_success,rollout_failure}/` 和 `Actor/source_XXX/` 分别提供类别与具体
manifest source 的 sampled、kept、keep_fraction、advantage_mean 和 loss_mean；两套分类
描述同一批样本，不能相加。Critic 同样报告各来源 sampled 数。
`Cumulative/` 字段为轮内累计统计。没有 Actor 样本或没有保留样本时，对应均值为 null。

Actor 分来源 loss 是实际训练 forward 中保留样本的逐样本 Flow MSE，按样本数加权汇总；
不增加另一次 forward 或随机采样。不同来源 loss 可辅助诊断，不是场景成功率指标。

完成轮次可用 `resume=true` 跳过已完成工作。中途恢复沿用既有设计：从该轮输入 Actor
重新开始整轮，不提供逐 update 的 optimizer/RNG 恢复；checkpoint 不保存完整 Q/V 状态。

## 最后统一比较

两组跑完后，使用此前未用于选模的 6000–6099；若这组已被用于调参，请换另一组：

```bash
python evaluate.py +experiment=oc_budget \
  '~comparison.checkpoints' \
  '+comparison.checkpoints={incumbent:outputs/StackCube-v1/oc_budget/iterative/checkpoints/best.pth,control:outputs/StackCube-v1/oc_budget/iterative_replay_control/checkpoints/best.pth,balanced:outputs/StackCube-v1/oc_budget/iterative_balanced_replay/checkpoints/best.pth}' \
  'comparison.samplers=[cps]' comparison.allow_observation_variants=false \
  comparison.seed_start=6000 comparison.episodes=100 video.episodes=0 \
  comparison.output=outputs/StackCube-v1/oc_budget/iterative_replay_comparison_seed6000
```

比较三组成功率及逐 seed 变化。先看平衡方案能否超过同预算 control 和原始 incumbent，
不要用累计环境奖励挑选更好的模型。若不改善，再根据采样统计和 Q/V 诊断决定是否继续。

上传两组训练目录与最终比较目录中的 JSON/JSONL/CSV/YAML 即可，不必传模型：

```bash
zip -r iterative_replay_results.zip \
  outputs/StackCube-v1/oc_budget/iterative_replay_control \
  outputs/StackCube-v1/oc_budget/iterative_balanced_replay \
  outputs/StackCube-v1/oc_budget/iterative_replay_comparison_seed6000 \
  -i '*.csv' '*.json' '*.jsonl' '*.yaml'
```

CPU 回归：

```bash
python -m pytest -q -rs tests/test_iterative_sampling.py tests/test_stages.py \
  tests/test_q_selection.py tests/test_action_consistency.py tests/test_experiment.py \
  tests/test_flow_ppo.py tests/test_iterative_data.py tests/test_primitive_recording.py \
  tests/test_object_centric.py tests/test_relational.py tests/test_state_skip.py tests/test_point_sampling.py
```
