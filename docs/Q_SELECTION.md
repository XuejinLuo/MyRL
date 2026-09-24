# 固定 Actor/Q：动作范围一致性三组对照

本轮只评估，直接复用已经训练好的 epoch 10 Q checkpoint，其中包含同一份 Actor、
Q、normalizer、环境/模型配置和 CPS 参数。无需重新训练、采集数据或启动 online。
修复不保证成功率提高：先排除 Q 动作表示饱和与实际执行不同的实现问题。

## 一条运行命令

在更新后的仓库根目录，使用原来的 ManiSkill 训练环境：

```bash
python evaluate_q_selection.py +experiment=oc_budget \
  q_selection.checkpoint=outputs/StackCube-v1/oc_budget/critic/checkpoints/epoch_0010.pth \
  q_selection.ablation=true q_selection.sampler=cps \
  q_selection.seed_start=4000 q_selection.episodes=100 q_selection.split=diagnostic \
  q_selection.output=outputs/StackCube-v1/oc_budget/q_selection_clipping_ablation
```

这也是新入口的默认配置。已有输出目录会被拒绝，不覆盖历史 `q_selection` 或
`q_selection_epoch5`。重跑时请指定新的 `q_selection.output`。
checkpoint 路径不同时只需改对应参数。配置覆盖中的模型/环境字段不会替换 checkpoint
保存的模型/环境；设备由 `device` 指定。三组均复用 checkpoint 的 CPS noise_level、
min_std 和 num_inference_steps。

| 输出标签 | 候选数 | 动作处理 | 目的 |
|---|---:|---|---|
| `legacy_single` | 1 | 原始动作与原执行路径 | 复查历史单候选基线 |
| `consistent_single` | 1 | 统一动作范围 | 测量仅改变动作范围的影响 |
| `consistent_q8` | 8 | 同一统一动作处理，取最高 min(Q1,Q2) | 测量 Q 选择的额外影响 |

两个单候选组均不调用 Q，也不引入额外随机采样。旧单候选的动作值、采样随机数状态和
执行结果保持原行为。相同 seed 配对的是场景，不保证 K=1 与 K=8 使用相同的采样噪声序列。
4000–4099 已用于多轮诊断，输出明确标记为 `diagnostic`，不是独立最终测试集。
代码仍拒绝与 checkpoint 记录的 Actor 验证/数据采集种子重叠的场景。

## 动作处理

旧流程已考虑环境物理边界，但把物理动作再次 normalize 后会饱和。例如环境未截断时，
1.05 和 1.10 都可能被 Q 当作 1.00，却执行不同的物理动作。

一致性模式先将归一化候选限制在 `[-1,1]`，再限制到
`unnormalize([-1,1])` 与环境物理边界的交集，最后转回 Q 的归一化表示，
同时将**这个表示**作为执行器的返回值。normalizer 使用 `max-min+eps`，所以有效物理
上端点是 `max+eps`，不能直接用训练 `max` 替代；常量维度沿用现有 eps 宽度。
非对称范围和更窄环境边界按每个维度处理，空交集、无效统计或非有限输入显式报错。
未全局修改 normalizer，训练流程及其他评估默认行为不变。

Q 使用自己的观测编码器，只评分 `env.exec_steps` 前缀（当前为 2 步），返回完整
chunk（当前为 16 步）。一致性模式在映射时校验反变换与预期执行值，并在每次环境
step 后用 `executed_actions` 验证实际执行值，容差为 atol=2e-6、rtol=1e-5。
当前工厂使用无 temporal ensembling 的 ChunkActionWrapper。

## 结果与诊断

结果根目录：`outputs/StackCube-v1/oc_budget/q_selection_clipping_ablation/`。
请上传根目录的 `summary.csv`、`summary.json`、`config.yaml`，以及三个标签目录中的
`episodes.csv` 和 `summary.json`（可直接打包整个结果目录，不必上传 checkpoint）。

根 `summary.json` 的两条 `paired` 明确记录 baseline_label 和 selected_label：

1. `legacy_single` → `consistent_single`：动作范围变化的影响。
2. `consistent_single` → `consistent_q8`：Q 选择的额外影响，优先看这一对。

每对均报告 baseline 失败而 selected 成功、反向退步数量及成功率净变化。
比较 `Eval/Success_Rate`、成功次数及逐 seed 结果，不用累计环境奖励替代成功率。
汇总 protocol 与各组 metadata 保存 checkpoint 路径/哈希、源 Actor 哈希、Q epoch、
模型/环境、normalizer、种子、采样参数及模式说明；每组结果保存 label、candidates、action_mode。

汇总及逐 episode CSV 的 `Action/candidates/` 和 `Action/selected/` 分别统计
所有候选与最终选中动作。统计只计实际执行前缀，并按 `actual_steps` 排除提前结束后的步数：

| 字段后缀 | 含义 |
|---|---|
| `decisions`, `chunks`, `coordinates` | 决策数、候选 chunk 数、实际统计的动作坐标数 |
| `raw_outside_fraction` | 原始归一化候选超出 [-1,1] 的坐标比例 |
| `env_clipped_fraction` | 本组动作路径中环境物理边界裁剪的坐标比例 |
| `raw_env_clipped_fraction` | 原始候选按旧反变换执行时，环境会裁剪的坐标比例 |
| `raw_to_score_max_abs` | 原始物理执行动作与评分表示反变换之间的最大差值 |
| `execution_to_score_max_abs` | 本组物理执行动作与评分表示反变换的最大差值；selected 使用环境实际记录 |

比例均为累计坐标数之比，不是每个 chunk 比例的均值；相应 `*_coordinates` 保留分子。
候选统计使用所选策略访问的状态，并按该次实际执行步数截取所有候选；未进行候选分支 rollout。
单候选的“评分表示”仅为诊断计算，不会调用 Q。`raw_to_score_max_abs` 在一致性模式下
也可能非零，表示修正改变了动作；应接近零的是 `execution_to_score_max_abs`。

若 consistent_q8 仍明显差于 consistent_single，不能继续把全部退化归因于动作裁剪。
下一步再单独设计同状态候选分支执行与 Q 排序诊断，本 PR 未加入该实验。

## 兼容与验证

历史两组评估可显式设置 `q_selection.ablation=false q_selection.candidates=[1,8]`，
并指定独立输出目录；两个候选数量均使用 legacy 路径。三组模式固定为 1/1/8，不扫描超参数。
`train_critic.py` 的既有训练入口保持不变，本次实验不需要运行它。

CPU 回归（不需要 ManiSkill 仿真）：

```bash
python -m pytest -q tests/test_action_consistency.py tests/test_q_selection.py \
  tests/test_stages.py tests/test_experiment.py tests/test_flow_ppo.py \
  tests/test_iterative_data.py tests/test_primitive_recording.py \
  tests/test_object_centric.py tests/test_relational.py tests/test_state_skip.py tests/test_point_sampling.py
```

CPU/toy 验证不能替代用户 checkpoint 上的 ManiSkill 成功率评估。
