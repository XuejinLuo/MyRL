# 鼠标/键盘人工接管与纠正数据

这是 iterative 的**采集 → 审核 → 固定数据训练**扩展。采集独立运行，不在优化器更新过程中等人操作。
现有 offline / online 默认行为不变；不需要遥操作手柄，不需要重新生成原有专家演示。
目前只支持 StackCube-v1、panda_wristcam、physx_cpu、pd_ee_delta_pose 和非 temporal-ensembling 执行。

## 先理解接管范围

- 不需要每条失败都救。优先覆盖空抓、单指顶住、邻近方块干涉、抓住后掉落等代表性状态。
- 不必刚闭爪就立刻接管：允许夹爪闭合、接触稳定，并观察模型能否自行恢复。
  当确认已空抓且持续闭爪，或反复下压/推偏没有进展时，再接管。
- 明显正在把物体推离可操作区域时可以提前接管，不必等自动提示。
- 人工接管后的成功不是自主策略成功率。最终评估必须关闭人类帮助。
- 原来诊断过的 6009 / 6014 / 6019 若用于训练，就不能继续作为独立测试样本。
  默认采集从 12000 开始；请确保你没在其他未记录的实验中将这批种子用于测试。

## 1. 启动采集

在已有 ManiSkill 环境、有图形桌面的机器上，从仓库根目录执行：

```bash
python -m tkinter
```

能出现测试窗口就说明 Tk 可用。缺 Tk 时在当前 Python/Conda 环境安装 Tk；无图形桌面的
SSH 需先配置远程桌面/显示。GUI 使用 Tk + Pillow，录像使用已有的 imageio/ffmpeg。
依赖缺失时：

```bash
python -m pip install pillow imageio imageio-ffmpeg
```

采集 10 条新场景：

```bash
python -m tools.collection.human_takeover \
  --checkpoint outputs/StackCube-v1/oc_budget/iterative/checkpoints/best.pth \
  --seed-start 12000 --episodes 10 \
  --output outputs/StackCube-v1/oc_budget/human_session_01 \
  --auto-pause
```

默认参数已指向上面的 checkpoint；不加 `--auto-pause` 则仅提示，不自动暂停。
`--policy-delay-ms 100` 控制观察时的墙钟速度；不会修改控制频率、动作执行次数或仿真步长。
可用 `--exclude-seeds 6009 6014 6019` 额外声明禁止进入训练的种子。
输出目录必须不存在，重复采集用新名称和新种子。

## 2. 界面操作（按键或鼠标点击按钮）

窗口显示外部总览、原有 base camera 与腕部相机 RGB。总览只供操作者看；
Actor 仍使用 checkpoint 中的原始点云预处理、state、相机/机器人配置和冻结 normalizer。
不会把外部总览 RGB 或诊断用物体真值坐标追加进 Actor 输入。
腕部相机随机器人运动，接管的每个物理步都重新生成真实观测，不复制接管前的点云。
启动时检查 `panda_wristcam` 和控制器坐标约定，不能用普通 Panda 静默替代。

| 按键 | 行为 |
|---|---|
| P | 模型连续执行 / 将控制权交还模型 |
| Space | 暂停，仿真和 episode 时钟均停止 |
| H | 人工接管；丢弃模型未执行的动作前缀 |
| N | 模型执行一个 primitive step；同一前缀未被接管时仍按 exec_steps 消费 |
| D / A | 机器人根坐标系 X 正 / 负方向 |
| W / S | Y 正 / 负方向 |
| E / Q | Z 正 / 负方向 |
| L / J | 绕 X 旋转正 / 负方向 |
| I / K | 绕 Y 旋转正 / 负方向 |
| U / O | 绕 Z 旋转正 / 负方向 |
| G / B | 打开 / 闭合夹爪 |
| T | 保持当前夹爪指令，推进仿真让夹爪/接触稳定 |
| F | 保存并结束当前条；再点 Next seed |

这是**按钮/键盘增量控制**，本 PR 未实现 3D 鼠标拖拽目标或运动规划接管。
移动前先按 H。按一下只执行选定的 1/5/10/20 个 primitive steps，等待操作时不会自动录入零动作。
键盘长按不是连续遥操作，请按下后松开或点击按钮。

`Controller delta` 是归一化控制器动作，不是米/弧度，也不是模型动作归一化值。
例如 Panda 常用位置范围 ±0.1 m 时，0.05 对应每步 5 mm 的**目标**位移；实际位移受 IK、接触影响。
建议从 0.02 / 0.05、单步开始，方向看外部总览确认；腕部图像方向不等于根坐标轴方向。
所有实际动作限制在 controller bounds 与冻结 action normalizer 范围的交集，原始请求、裁剪结果都会记录。
模型采集沿用 iterative 的归一化动作先裁到 [-1,1] 规则，并非无接管 benchmark。

夹爪通常需要多步才完全开合，可选择 10/20 步执行 G/B，或者用 T 等待。
这些动作有实际动力学意义，清洗时不会因为位移小就删除。不要录入大量无意义的 T。
暂停不消耗 horizon，人工执行会消耗；沿用 checkpoint 的 300 步等任务时限，不偷偷延长训练任务。
若时间不足，保存为失败/截断，再用新场景采集更早的纠正。

例：空抓后，H → G 张开 → E 抬起 → 调整 X/Y 和角度 → Q 下降 → B 闭合 → T 稳定 → E 抬升。
恢复后可继续人工堆叠，也可以 P 让模型完成。每次控制权切换都会分开标记。
不要通过直接拖动物体、设置机器人/方块 pose 来“恢复”，否则采集不到真实的纠正动作。

## 3. 失败提示怎么理解

提示使用仿真可用的抓取标记、夹爪开度、手指接触力和 TCP 位移，**不参与奖励或策略输入**。
这些阈值是起始启发式，没有经过分类准确率标定；缺失的可选读数不会当作零接触。

| 提示 | 连续条件 / 默认确认时间 | 人工核实 |
|---|---|---|
| empty_closed | 闭合指令、开度 <8 mm、未抓住、未叠放，0.75 s | 是否真的空抓，是否在主动松开重试 |
| one_finger_contact | 只有一侧对目标方块接触力 >2 N、未抓住/叠放，0.5 s | 单指顶边、推偏，还是正常短暂调整 |
| cubeB_interference | 手指对另一方块接触力 >2 N、尚未叠放，0.3 s | 是否干涉；接触不必然是失败 |
| contact_stall | 对目标有接触、平移动作范数 >0.05、每步 TCP 位移 <1 mm，1 s | 是否卡住；正常抓稳也可能低位移 |
| grasp_lost | 曾有抓取标记，之后未抓住且未叠放，0.4 s | 是掉落还是正常释放 |

不同类型可同时出现；无提示也不代表没有失败。正常堆叠/成功时抑制这些提示。
`--auto-pause` 每条轨迹对每种提示最多自动停一次，避免继续观察时反复被同一提示打断。
暂停后由人选择 H 接管、P 继续或者 F 放弃。**不会自动替你执行恢复动作。**

## 4. 保存什么、怎样审核

每个 session 包含：

- `session.json`：checkpoint 哈希、冻结 config/normalizer、软件版本、种子和 sampler。
- `seed_*.npz`：T 个真实执行动作、稀疏 success 奖励和终止标记；T+1 个真实点云/state。
- `seed_*.events.jsonl`：每步 policy/human 来源、人工段编号、请求/执行动作、接触诊断和提示。
- `seed_*.json`：人工段的 `[start, stop)`、结束原因、最终 success、可选诊断/录像错误。
- `seed_*.mp4`：总览 + 相机画面 + 步号/来源；第 0 帧是 reset，第 k+1 帧对应第 k 步后。
- `review.json`：默认所有数据待审核。未经审核不会进入训练。

F 提前结束时将最后一步标为 `truncated=True`，不会伪造成功或 terminated；Q/V 从真实最终观测 bootstrap。
发生仿真/观测异常时暂停并写 error.json，该条不导出成有效数据；已完成条目仍保留，可重开新 session。
视频错误会显式记录，采集可继续；审核前需有足够的可视证据，不要盲目接受。

先看视频、对照 events 的步号，再执行审核和导入：

```bash
python -m tools.prepare_corrections \
  --session outputs/StackCube-v1/oc_budget/human_session_01 \
  --base-manifest outputs/StackCube-v1/oc_budget/iterative_expand600/round_000/manifest.json \
  --stats outputs/StackCube-v1/oc_budget/iterative/checkpoints/dataset_stats.json \
  --output outputs/StackCube-v1/oc_budget/human_replay_01 \
  --review
```

按终端提示逐条选择：

- `k`：保留，随后每个人工段选 `a` 接受 / `r` 拒绝。
- `c`：只用于 Critic，整条不提供 Actor 专家监督。
- `d`：整条丢弃，例如异常观测、无效操作、明显采集故障。

人工段应连贯、方向正确、没有大段误操作。当前审核粒度是整个接管段；不允许手改起止索引把
模型动作“划入人工段”。要细分采集，使用暂停/交还模型后再次接管建立新的段。
审核结果可保留在 review.json，之后不加 `--review` 直接重试导入；导出必须用新目录。

清洗规则：

1. 校验原始 NPZ/元数据哈希、T/T+1、有限数值、稀疏奖励、终止一致性、观测形状。
2. 校验环境/预处理和冻结 normalizer 一致；越界不通过事后裁剪篡改训练标签。
3. 排除重复数据与 checkpoint 中已声明的验证/测试种子；训练时再次对本次评估种子检查。
4. 只有**最终任务成功 + 人工明确接受**的段可供 Actor。没救成功但数据有效可保留给 Critic。
   这是保守起点，会漏掉“抓取已恢复、后来堆叠失败”的局部有效片段，暂不自动将其标为专家。
5. Actor 只取完全在一个接受段里的完整 chunk（当前 H=16）。长度 <16 的段不提供 Actor 样本；
   尾部不足 16 步的位置不 padding，不跨回模型段/被拒绝段，也不拼接不连续状态。
6. 不删除轨迹中间的等待/坏动作后重新连边；Critic 的完整真实时间顺序、奖励、结束条件不变。

导入输出 `cleaning_report.json` 和独立 `manifest.json`，不会修改原 manifest/原始轨迹。
若没有合格的完整纠正 chunk，会报错停止，而不是悄悄退回普通 replay。

## 5. 接入 iterative 训练

```bash
python train_iterative.py +experiment=oc_budget \
  +iterative_protocol=human_corrections
```

该配置默认接入上面 `human_replay_01/manifest.json`，从原 iterative best 开始。
若路径不同，可覆盖 `manifest=...`、`initial_ckpt=...`、`stats_path=...` 和 `output=...`。

- 固定预算：20,000 次 Critic 更新，前 10,000 次 warmup，10,000 次 Actor 更新。
- batch=64：Actor 32 原始 demonstrations + 16 原有成功 rollout + 16 审核通过的纠正 chunk。
  组内先均匀选 episode，再选合格起点；三组都必须有数据，不静默从失败动作补足。
- 原示范/成功 rollout 保留已有 IDQL 筛选；**审核通过的人工纠正不被 Q/V 的低分拒绝**。
  `algo.use_bc_only=true` 可让全部三组都用 BC，其他预算不变。
- Critic 均匀采样所有保留轨迹，包括模型失败前缀、人工纠正与失败恢复；奖励仍是每步真实 success。
- `sampling.json`、metrics 中单独报告 `Actor/correction/{sampled,kept,...}`，检查纠正确实被学习。
- 纠正来源不会自动当成普通成功 rollout 或普通 demonstration；即使该局最后成功也不能泄漏错误前缀给 Actor。
- `actor_eligible` 与 chunk_size 绑定；改 chunk_size 必须重新准备数据。旧 epoch/mixed 训练拒绝此类数据，防止误用。
- 验证成功率没有严格提升时保留原 best；不会因为人为救成功的采集成功率高就替换 best。
- checkpoint 内保存纠正训练种子，`evaluate.py` 也拒绝把这些种子当作独立测试集；
  从含纠正数据的 checkpoint 继续 iterative 时保留该记录。

先采小批，确认恢复动作和导入报告，再扩到多种场景。不要只在一个种子上反复练习。
按原验证种子选择模型，最后用独立种子无人工帮助比较原 best 和新 best：

```bash
python evaluate.py +experiment=oc_budget \
  '~comparison.checkpoints' \
  '+comparison.checkpoints={before:outputs/StackCube-v1/oc_budget/iterative/checkpoints/best.pth,after:outputs/StackCube-v1/oc_budget/iterative_human_corrections/checkpoints/best.pth}' \
  'comparison.samplers=[cps]' \
  comparison.seed_start=13000 comparison.episodes=100 \
  comparison.output=outputs/StackCube-v1/oc_budget/human_comparison_seed13000 \
  video.episodes=5
```

这里的 13000–13099 必须一直留作测试，之后采集不能使用。也可沿用未被接管训练污染的既有测试集。

## 验证范围与实现参考

CPU 测试覆盖：暂停不推进、接管丢弃旧前缀、恢复推理、与原 primitive 执行规则对齐、提示持续时间、
动作分块边界、审核/哈希/越界/种子保护、真实优化器训练与纠正源采样计数。
开发环境未安装可运行的 ManiSkill/SAPIEN 图形仿真，GUI 真实腕部相机与接触操作需要在本机首轮验证。
先采 1 条检查方向、相机、录像和清洗结果，再大批量采集；不声称已验证提升幅度。

- [ManiSkill 控制器与坐标系](https://maniskill.readthedocs.io/en/latest/user_guide/concepts/controllers.html)
- [ManiSkill 官方遥操作](https://maniskill.readthedocs.io/en/latest/user_guide/data_collection/teleoperation.html)
- [Python Tkinter](https://docs.python.org/3/library/tkinter.html)
