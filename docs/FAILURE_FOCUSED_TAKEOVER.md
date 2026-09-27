# 只采集需要人工纠正的失败场景

先让 best 无人值守跑完一批新种子，再从确定失败的轨迹里挑选场景，重放到较早的接管点。
你只需处理失败队列，不必从头盯着每条成功轨迹。这是 iterative 的数据采集阶段；训练时不等待人工。

## 1. 为什么之前人工动作慢、容易耗尽 300 步

旧拖拽执行每步额外限制为 5 mm / 0.03 rad，并严格追踪每个密集规划点，容易消耗多余控制步。
新版普通模式默认 20 mm / 0.1 rad，并沿连续路径有限前瞻，最终目标仍要求 2 mm / 0.015 rad 精度。
实际移动量受控制器、冻结 action normalizer、IK 和接触影响，不能保证恰好快四倍。

- 按 **V** 切换普通/精细模式；精细模式不超过 5 mm / 0.03 rad，适合接近方块时微调。
- 切速会取消正在执行的路径；目标保留，再按 N 执行。
- 可用 `--human-position-step 0.01 --human-rotation-step 0.06` 调低普通模式上限。
- `--policy-delay-ms 50` 仅影响墙钟播放速度，不能节省仿真步数。
- 暂停、拖目标、不按 N 时不消耗仿真步；执行人工动作仍然消耗。
- 不延长原任务 horizon。失败筛选会把接管点前移，给恢复预留步数。

## 2. 无人值守筛选失败

从仓库根目录，在原来的 ManiSkill 环境运行；15000–15099 作为新采集种子，不能再用于独立测试。
如果你已经把它们用于测试，请换一批未用过的种子。所有输出目录必须是新目录。

```bash
python -m tools.collection.human_takeover \
  --screen-only \
  --checkpoint outputs/StackCube-v1/oc_budget/iterative/checkpoints/best.pth \
  --seed-start 15000 --episodes 100 \
  --recovery-reserve 180 --rewind-steps 20 \
  --output outputs/StackCube-v1/oc_budget/failure_screen_01
```

不弹出 Viewer、不需要 MPlib，不保存成功轨迹的视频/点云，只打印进度并写摘要。
仍需要能运行 ManiSkill 的 GPU/渲染环境，不能在纯 CPU 无渲染环境代替真实筛选。
每条策略跑到正常结束；只将**全过程一次 success 都没有**的轨迹放入 `failure_queue.json`。
筛选沿用采集端 frozen-range clipping，不是独立评估入口，也不报告为最终 benchmark。

输出 `failures/seed_*.npz` 保存失败轨迹的实际动作及仿真状态，用于后续校验重放；这些文件不是训练 episode。
`failure_queue.json` 的 `results` 包含全部筛选结果，`failures` 包含失败 seed、提示、接管点和文件哈希。
队列逐条保存；筛选中断时，已有完整条目仍可使用（`complete=false` 表示没有跑完整批）。

默认接管点：第一次持续提示出现前 20 步；没有提示时取 horizon−180。
接管点至多为第 120 步（300 步任务），也必须早于该条实际结束，故至少预留 180 步恢复预算。
这是恢复时间预算，不保证那个时刻已经出现可见错误。失败类型只是启发式提示。
需要统一更早介入，可在**筛选命令**加 `--takeover-step 40`；仍需满足剩余预算及实际轨迹长度。

## 3. 只对失败队列进行接管

先试队列第一条：

```bash
python -m tools.collection.human_takeover \
  --ui sapien \
  --checkpoint outputs/StackCube-v1/oc_budget/iterative/checkpoints/best.pth \
  --failure-queue outputs/StackCube-v1/oc_budget/failure_screen_01/failure_queue.json \
  --queue-start 0 --episodes 1 \
  --output outputs/StackCube-v1/oc_budget/human_failure_trial_01 \
  --policy-delay-ms 50
```

程序自动 reset 该失败 seed，执行筛选时记录的原始动作，在接管点自动进入 human 模式、提示音并等待。
这个前缀重放过程不需要按 P，也不受观察速度延时影响；录制视频时仍逐步保留前缀画面。
每步核对 actor/articulation 位姿及速度；若仿真版本、资产或运行条件导致偏离，拒绝导出当前条，
不会设置机器人/方块状态来强行“对齐”。请用相同环境重新筛选。
恢复后的 P 会重新采样策略，不继续原失败轨迹的动作队列。

熟悉后用另一个输出目录采集接下来的失败（避免重复导入试采 seed）：

```bash
python -m tools.collection.human_takeover \
  --ui sapien \
  --checkpoint outputs/StackCube-v1/oc_budget/iterative/checkpoints/best.pth \
  --failure-queue outputs/StackCube-v1/oc_budget/failure_screen_01/failure_queue.json \
  --queue-start 1 --episodes 20 \
  --output outputs/StackCube-v1/oc_budget/human_failure_session_01 \
  --policy-delay-ms 50
```

`queue-start` 是失败列表的下标，不是 seed；`episodes` 是最多处理多少条失败。不足 20 条时取剩余条目。
队列为空会明确提示，没有需要接管的场景。C 保存并自动重放下一条失败，Q 保存退出。
中途退出后，从上次队列下标的下一条开始，用新输出目录继续；不要把同一条重复导入。
仍可按 P 看模型是否能继续恢复，或 F/C 放弃困难场景，不需要救下每条失败。

## 4. 平移和旋转怎么操作

1. 已自动接管时不需要再按 H；手动接管则先 H。
2. 找到官方 **Transform** 面板，点击 **Translate** 出现平移箭头，点击 **Rotate** 出现旋转环。
3. 选择 **Local / World** 决定旋转轴的参考坐标。
4. 拖动旋转环，或编辑 **Rotation** 三个角度（单位：度），**Enter** 确认输入。
5. 目标仅是预览，最后按 **N** 才执行。**G** 开合夹爪；**Space** 取消路径并暂停。

H 会重新选择手部并重置目标，不是平移/旋转切换键。Rotation 不是机器人关节角。
旋转与平移都通过原来的 7 维 EE-delta 控制器记录，手部到 TCP 的真实偏移仍参与转换。
MyRL 面板现在也显示旋转操作提示和当前速度上限。

## 5. 看视频、审核、清洗

| 路径 | 内容 |
|---|---|
| `videos/seed_*.mp4` | 只需先打开这个目录看视频，含总览和腕部/基座相机 |
| `episodes/seed_*.npz` | 真实点云、状态、执行动作和稀疏奖励 |
| `metadata/seed_*.json` | 人工段边界、接管来源、最终成功与错误信息 |
| `traces/seed_*.events.jsonl` | 每步来源、动作、接触提示，重放前缀仍标 policy |
| `traces/seed_*.targets.jsonl` | N 的目标、规划点数和执行速度上限 |
| `review.json` / `session.json` | 审核决定、文件索引和会话来源 |

同一 seed 文件名对应；视频引用采用相对路径，整个目录可以一起移动。
旧 `human_drag_session_01` 的平铺文件仍可直接用审核工具，不必手动搬移或重写哈希。

```bash
python -m tools.prepare_corrections \
  --session outputs/StackCube-v1/oc_budget/human_failure_session_01 \
  --base-manifest outputs/StackCube-v1/oc_budget/iterative_expand600/round_000/manifest.json \
  --stats outputs/StackCube-v1/oc_budget/iterative/checkpoints/dataset_stats.json \
  --output outputs/StackCube-v1/oc_budget/human_failure_replay_01 \
  --review
```

`k` 保留后逐个人工段 `a` 接受 / `r` 拒绝；`c` 只供 Critic；`d` 丢弃整条。
只有最终成功、人工批准、同一人工段内的完整 action chunk 才供 Actor；快动作导致人工段短于
chunk_size=16 时不会补零凑标签，报告会显示 `shorter_than_chunk`，没有合格 chunk 时拒绝训练导出。
可以连续执行多个恢复子目标形成一个有意义的人工段，不要为了凑长度录入空等动作。

## 6. 和旧成功数据混合训练

**需要混合，而且 human_corrections 协议已实现分来源固定比例采样。**
它不是把整个“人工参与过的成功 episode”都当专家。

| Actor 每个 batch（64） | 数量 | 标签规则 |
|---|---:|---|
| 原示范 | 32（50%） | 原 BC/IDQL 规则 |
| 自主成功 rollout | 16（25%） | 原 BC/IDQL 规则 |
| 审核通过的人工纠正 chunk | 16（25%） | 强制保留 BC 监督，避免被旧 Q 拒绝 |

Critic 保留原示范、成功/失败 rollout，以及保留的完整纠正轨迹，使用真实稀疏奖励。
人工救成功之前的失败策略动作不会变成 Actor 专家标签；采集阶段无人值守筛选的文件不自动进入训练。

```bash
python train_iterative.py +experiment=oc_budget \
  +iterative_protocol=human_corrections \
  manifest=outputs/StackCube-v1/oc_budget/human_failure_replay_01/manifest.json \
  output=outputs/StackCube-v1/oc_budget/iterative_human_failure
```

默认初始化为上面的 iterative best，保留原 2000–2049 验证种子选择 best。
如果此前已清洗旧人工 session，`--base-manifest` 应改为那份清洗后的 manifest，以累积而非丢掉旧纠正数据。
其他初始化 checkpoint 必须同时使用匹配的 frozen stats 和环境配置。
训练日志的 `Actor/correction/sampled` / `kept` 用于核对人工数据确实参与训练。

## 7. 独立验证提升

失败队列的恢复成功率有筛选偏差且有人帮助，不能与自主成功率比较。
新 checkpoint 会记录**整批筛选种子**，包括被跳过的成功 seed；独立评估拒绝与它们重叠。
使用一批之前未用于采集、筛选或模型选择的测试种子，例如尚未使用过的 16000–16099：

```bash
python evaluate.py +experiment=oc_budget \
  'comparison.checkpoints={before:outputs/StackCube-v1/oc_budget/iterative/checkpoints/best.pth,after:outputs/StackCube-v1/oc_budget/iterative_human_failure/checkpoints/best.pth}' \
  comparison.seed_start=16000 comparison.episodes=100 \
  comparison.output=outputs/StackCube-v1/oc_budget/human_failure_comparison_seed16000 \
  video.episodes=5
```

CPU 测试覆盖筛选、重放状态校验、人工/策略来源隔离、目录兼容、路径前瞻和真实小模型混合训练。
开发环境没有 ManiSkill/SAPIEN 图形仿真，尚未验证本机操作手感、真实恢复成功率或训练提升；先试采一条。
