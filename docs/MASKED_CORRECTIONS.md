# 人工纠正监督修复与 A/B 实验指南

本次修改针对两个已定位问题：人工组按 episode 均匀采样会过度重复短轨迹的少量 chunk；
完整 H=16 标签规则遗漏短人工段和长段尾部。代码和 CPU 测试已实现，真实 StackCube 训练收益尚待验证。
保留现有 point cloud、object-budget、网络参数结构、控制协议、300 步时限和 frozen normalizer。

## 1. 实验顺序与配置

| 实验 | `correction_sampling` | `correction_label_mode` | 输入 |
|---|---|---|---|
| 旧行为复测 | `episode` | `full_chunk` | 旧 cleaned manifest |
| A：只修正采样 | `uniform_start` | `full_chunk` | 旧 cleaned manifest |
| B：增加短段/尾部监督 | `uniform_start` | `masked` | 从同一原始 session 重新导出 |

以上键均位于 `actor_sampling` 下。`human_corrections` preset 现在默认对应 A，并增加早期验证；
普通 iterative 的默认值仍是旧 episode/full_chunk。复测历史调度时另设 `eval_actor_updates=[]`。
做新的严格对照时三组可保持相同早期验证调度；验证 RNG 与训练隔离。

A/B 均使用原 iterative best、相同统计量和原数据池。batch=64 时仍为 32 原示范 + 16 自主成功 + 16 人工纠正。
Critic 独立均匀抽全部真实 transition；Actor 抽样不改变 Critic 的索引序列。
旧 session（平铺或分类目录）、旧只有 `actor_eligible` 的 NPZ/manifest 均可读取。

以下命令在用户现有环境、仓库根目录 `/home/luo/MyRL` 执行。输出目录必须不存在；
已有结果用新名字保留。无需重新采集、重新审核或改写旧文件。

## 2. 先跑 A：复用旧完整标签

```bash
python train_iterative.py +experiment=oc_budget \
  +iterative_protocol=human_corrections \
  initial_ckpt=outputs/StackCube-v1/oc_budget/iterative/checkpoints/best.pth \
  stats_path=outputs/StackCube-v1/oc_budget/iterative/checkpoints/dataset_stats.json \
  manifest=outputs/StackCube-v1/oc_budget/human_failure_replay_01/manifest.json \
  output=outputs/StackCube-v1/oc_budget/iterative_human_A_uniform_full \
  actor_sampling.correction_sampling=uniform_start \
  actor_sampling.correction_label_mode=full_chunk \
  'eval_actor_updates=[250,500,1000,2000]' \
  seed=42
```

保持默认总更新 20000、Critic warmup 10000、Actor LR=1e-5、Critic LR=3e-4、EMA=0.999、IDQL。
若历史运行的训练 seed 不是 42，A/B 都替换为该 seed；不要在 A/B 之间更改其他因素。

启动后的 `round_000/sampling.json` 应显示 `correction_sampling=uniform_start`、
`correction_label_mode=full_chunk`。若数据与交接材料一致且未因 bounds 排除轨迹，
应有 **14 条有效 Actor 人工轨迹、912 个起点**；保留的 18 条人工轨迹不等于有效 Actor 轨迹数。
这些是旧材料的预期，需以本地报告为准。

复测旧采样可复制 A 命令，改为 `actor_sampling.correction_sampling=episode`，
另用 `output=outputs/StackCube-v1/oc_budget/iterative_human_legacy`。
若还要复测原评估节奏，改为 `'eval_actor_updates=[]'`。

## 3. 为 B 从原 session 导出 masked 标签

```bash
python -m tools.prepare_corrections \
  --session outputs/StackCube-v1/oc_budget/human_failure_session_01 \
  --base-manifest outputs/StackCube-v1/oc_budget/iterative_expand600/round_000/manifest.json \
  --stats outputs/StackCube-v1/oc_budget/iterative/checkpoints/dataset_stats.json \
  --output outputs/StackCube-v1/oc_budget/human_failure_replay_masked_01 \
  --label-mode masked --min-valid-length 1
```

这里 `--base-manifest` 必须是**尚未导入这批人工 session** 的原 manifest，不能填 A 的人工 manifest。
复用已完成的 `review.json`，不要加 `--review`，除非确实要重新审核。
新目录包含带 `actor_valid_length` 的 NPZ、独立 manifest 和 cleaning report，不覆盖原始 session 或旧标签。
源记录 `actor_label_schema=myrl_actor_labels_v2`、`actor_label_mode=masked`、最小有效长度和 chunk_size。
训练配置与导出模式不匹配会报错，避免把旧 full 标签误称为 masked 实验。

对最终成功且接受的人工段 `[start, stop)`，起点 t 的有效长度为 `min(H, stop-t)`；
默认至少 1 步，短段和每段尾部都可提供监督。`--min-valid-length` 可显式提高门槛，报告会记录排除数；
本次 A/B 请保持 1。失败恢复、reject、critic_only 不产生 Actor 标签，drop 不进入训练。

检查 `cleaning_report.json`：

- `actor_episodes`、`actor_starts`：实际可用轨迹与起点。
- `accepted_human_steps`、`full_chunk_starts`、`partial_chunk_starts`、`excluded_starts`。
- `episodes`：逐 seed 的审核结果、有效长度直方图与段排除原因。

若原数据和审核未变，B 预期 **17 条 Actor 人工轨迹、1155 个起点 = 912 完整 + 243 部分起点**。
这不是重新审计原 NPZ 得出的实测值；本开发环境没有用户的原始 session 数组。

## 4. 跑 B：只增加 masked 标签

```bash
python train_iterative.py +experiment=oc_budget \
  +iterative_protocol=human_corrections \
  initial_ckpt=outputs/StackCube-v1/oc_budget/iterative/checkpoints/best.pth \
  stats_path=outputs/StackCube-v1/oc_budget/iterative/checkpoints/dataset_stats.json \
  manifest=outputs/StackCube-v1/oc_budget/human_failure_replay_masked_01/manifest.json \
  output=outputs/StackCube-v1/oc_budget/iterative_human_B_uniform_masked \
  actor_sampling.correction_sampling=uniform_start \
  actor_sampling.correction_label_mode=masked \
  'eval_actor_updates=[250,500,1000,2000]' \
  seed=42
```

B 从**同一个原 iterative best**开始，不从 A 的 best 接着训练。

Actor 使用独立 `actor_actions` 和 `[B,H]` 的 `actor_mask`；Critic 使用原 `action_chunk`，
真实执行前缀、reward、done、discount 和 next observation 不在人工边界截断。
Flow 在插值前先移除无效目标；未知未来 token 输入为同一随机抽样中的纯 prior noise，
不包含后续 policy 动作，也不把零填充当作监督。损失先除以每样本有效步数×动作维度，再对保留样本平均。
这使 1 步样本不会因为除以完整 H 而被隐式降权。

未改变 Transformer 参数或推理调用，也不要求推理时传 mask。纯噪声未来仍可参与时间维交互，
但不携带未知标签；固定随机条件下改变无效标签不会改变有效 loss/梯度。该训练条件的闭环收益仍需 B 验证。
masked 监督目前限 Flow，配置为 diffusion 时明确拒绝。

人工样本始终 force_keep；**partial 样本不查询 Actor 筛选用 Q/V，也不进入拒绝概率的最大优势归一化**。
完整样本保留原真实动作 Q/V 路径（含完整人工 chunk 的旧逻辑），使 A 不顺带修改筛选算法。
日志 `advantage_samples` 是实际有意义的优势样本数；partial 的零占位不会计入 `advantage_mean`。
Critic 仍能在独立抽样中使用这些时刻的真实 transition。

## 5. 看哪些结果

新早期验证按 **Actor 更新次数**指定，总更新坐标仍用于文件名：

| Actor 更新 | 总更新 step | checkpoint（在 round_000/checkpoints 下） |
|---:|---:|---|
| 0 | 0 | `step_0000000.pth` |
| 250 | 10250 | `step_0010250.pth` |
| 500 | 10500 | `step_0010500.pth` |
| 1000 | 11000 | `step_0011000.pth` |
| 2000 | 12000 | `step_0012000.pth` |

之后保留原每 2000 总更新的节奏，最终仍评估 step 20000。
metrics、checkpoint、selection、评估 metadata 同时记录 `update_step` 与 `actor_update_step`。
每次早期验证都保存对应候选和评估目录；best 仅按验证成功率严格提升替换，dense return 不参与平局选择。
全部候选都未提升时，best 保留 step 0。

`round_000/sampling.json` 随日志/评估刷新，包含已消费样本的累计统计，不把 DataLoader 预取计作学习：

- `correction_actor_episodes` / `correction_actor_starts`：实际 eligible 池。
- `actual_counts`：各组/各源 Actor 和 Critic 实际抽样、保留、loss 与优势统计。
- `correction_audit.episodes`：按保留 episode ID 索引，含 seed、source、抽样数、唯一访问起点数、
  单起点最高重复次数和完整 `start_counts`；无 Actor 标签的保留 episode 也显示零。
- `correction_audit.valid_length_histogram` / `valid_length_mean`：实际人工监督长度分布。

A 中 seed 15032 的唯一完整起点应与其它**单个起点**接近同频，而不是与其它**整条轨迹**同频。
160000 次人工采样、912 个起点时，单起点期望约 175 次；实际有随机波动。
B 中 15032 有 16 个起点，不能继续按“唯一 chunk”解释它的 episode 总次数。

请返回 A/B 的 `config.yaml`、`metrics.jsonl`、`selection.json`、`state.json`、
`round_000/sampling.json` 及 B 的 `cleaning_report.json`。先判断早期走势、实际采样和 best 的 step；
若 best=0，比较两个 best 文件不能代表训练后的变化，应查看明确的候选 step。

## 6. 最后做独立配对评估

2000–2049 用于验证选择；15000–15099 **整批筛选种子**都属于训练数据选择过程，
包含筛掉的成功场景。checkpoint 继续保留并阻止这些 seed 被当成独立测试。
3000/4000/5000/6000 开始的历史场景也不能简单称为未见测试。

下面示例使用 17000–17099，运行前请确认它未参与任何历史采集、筛选、调参或选择；
若已使用，替换为自己确认未用的新范围，并同时修改输出名称。16000–16099 也不能默认未用。

```bash
python evaluate.py +experiment=oc_budget \
  '~comparison.checkpoints' \
  '+comparison.checkpoints={before:outputs/StackCube-v1/oc_budget/iterative/checkpoints/best.pth,A:outputs/StackCube-v1/oc_budget/iterative_human_A_uniform_full/checkpoints/best.pth,B:outputs/StackCube-v1/oc_budget/iterative_human_B_uniform_masked/checkpoints/best.pth}' \
  'comparison.samplers=[cps]' \
  comparison.seed_start=17000 comparison.episodes=100 \
  comparison.output=outputs/StackCube-v1/oc_budget/human_AB_seed17000 \
  video.episodes=0
```

三者观测协议相同，无需 `allow_observation_variants=true`。
看同 seed 的“原成功变失败 / 原失败变成功”，不要把人工恢复成功率当自主成功率。
如果 A/B 仍未改善，再分别测试较低人工比例、较少 Actor 更新、BC 或 encoder 冻结，每次只改一个因素。

## 7. 验证范围

本次开发验证：94 passed，1 skipped。跳过的是运行环境禁止 Unix socket 导致无法执行的
多进程 tensor IPC 测试；其余 CPU 测试通过。真实图形仿真和完整训练不包含在该数字内。

```bash
python -m pytest -q tests/test_masked_corrections.py tests/test_human_corrections.py \
  tests/test_iterative_sampling.py tests/test_iterative_data.py tests/test_stages.py \
  tests/test_flow_ppo.py tests/test_failure_queue.py tests/test_experiment.py
```

覆盖 1/2/6/10/16/>16 步、控制权边界、审核排除、旧数据读取、精确配额、Critic 索引/transition 不变、
真实 Flow/Transformer loss/梯度、真实 CPU 优化器训练、早期评估与 step0 保底、筛选种子隔离。
没有在开发环境执行真实 ManiSkill/SAPIEN 图形仿真、用户数据重导出或 20000 次完整训练；不承诺成功率提升。
