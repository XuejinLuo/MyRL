# Online 首次成功奖励与有限时限协议

本次实现用于检验：dense reward 是否鼓励中间状态停留，而不是完成任务。
**代码测试通过不代表策略成功率提高；真实 ManiSkill、GPU 与本机 iterative 权重仍需用户验证。**
Actor 的点云、16 维 state、冻结编码器、Flow 骨干、动作裁剪及 full latent PPO ratio 均保持原协议。

## 1. 两种模式

| 配置 | 训练奖励 | 超时 bootstrap | Critic 输入 | best 平局 |
|---|---|---|---|---|
| `online_task=legacy`（默认） | 环境原奖励 × reward_scale | reset 前最终状态 V | 冻结特征 | 环境累计奖励更高者 |
| `online_task=success_once` | 未成功 0，首次成功 +1 × reward_scale 并立即结束 | 0，有限任务预算终点 | 冻结特征 + 剩余时间 | 保留较早 checkpoint |

`success_once` 在每个 primitive step 判断成功；第 1 步成功不会继续执行 chunk 的第 2 步。
成功与超时同一步出现时，奖励和 outcome 按成功记录；原始 terminated/truncated 均保留。
环境真正终止也切断 bootstrap。所有回合边界切断 GAE trace。
仅 rollout 采满时不 reset、不标记失败，仍使用实际 next state 的 V；本批 GAE 递推到此为止。
失败回合照常参与 PPO，不人为指定负优势。

有限时限 Critic 的最后一维是 `max(0, (H-t)/H)`；t 按实际执行步数推进并在 reset 时归零。
Actor 不接收剩余时间或仿真成功/阶段标志。`success_once` 要求环境显式提供 `info['success']`。
`finite_horizon` 要求启用时间条件 Critic；旧模式的外部截断语义仍可使用。

chunk 奖励为 `reward_scale * sum(gamma**i * primitive_reward[i])`，bootstrap 折扣为
`gamma**actual_steps`；lambda 按 chunk 决策。默认 gamma=0.99，因此目标是**折扣成功回报**，
偏好较早成功，并非无折扣成功概率；本 PR 未改 gamma，也未加入 shaping、BC 或停滞惩罚。

`online_task.environment_reward_mode=null` 保持安装版本的原始环境默认奖励模式；
`protocol.json` 保存实际运行环境的 `actual_reward_mode` 和 ManiSkill 版本，而不是假定官网版本。
如显式设置该项，只影响训练环境的 raw reward；统一 benchmark 仍使用原环境默认值。
评估 `summary.json` 另外记录实际 benchmark 环境 reward_mode。

## 2. 获取代码与本地 CPU 验证

PR 合并前使用功能分支；有本地未提交修改时先妥善保留，不使用强制覆盖命令。

```bash
cd /home/luo/MyRL
git fetch origin
git switch codex/online-success-reward-protocol
python -m pytest tests/test_flow_ppo.py tests/test_online_task.py -q
```

合并后可切回 `online` 并 `git pull --ff-only`。以下命令在已有 `rl100` 环境中运行。
输出目录必须不存在；重跑时换后缀，不覆盖已有实验。

## 3. 配置和初始化检查

在同一个 Bash 终端定义共同参数。明确使用原始 iterative Actor，不使用 dense warmed Critic。

```bash
cd /home/luo/MyRL
ONLINE_COMMON=(
  +experiment=oc_budget
  online_task=success_once
  stages.online.initial_ckpt=outputs/StackCube-v1/oc_budget/iterative/checkpoints/best.pth
  stages.online.stats_path=outputs/StackCube-v1/oc_budget/iterative/checkpoints/dataset_stats.json
  stages.online.resume=null
  stages.online.batch_size=128
  stages.online.save_every=1
  algo.steps_per_epoch=2048
  algo.update_epochs=1
  algo.actor_lr=1e-7
  algo.critic_lr=3e-4
  algo.gamma=0.99
  algo.ratio_scope=full
  algo.verify_logprobs=true
  eval.every=1
  eval.sampler=cps
  'eval.seeds=[2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2025,2026,2027,2028,2029,2030,2031,2032,2033,2034,2035,2036,2037,2038,2039,2040,2041,2042,2043,2044,2045,2046,2047,2048,2049]'
  seed=42
  video.every=0
  video.episodes=0
)
python train_online.py "${ONLINE_COMMON[@]}" --cfg job --resolve
```

先跑短 rollout（仍有 epoch 0 和 epoch 1 各 50 场景验证，因此仿真耗时主要在评估）：

```bash
python train_online.py "${ONLINE_COMMON[@]}" \
  output=outputs/StackCube-v1/oc_budget/online_success_smoke01 \
  stages.online.epochs=1 algo.steps_per_epoch=32 \
  algo.critic_warmup_epochs=1 algo.critic_warmup_update_epochs=1
```

检查 `protocol.json` 的 horizon=300、success_once、finite_horizon、Critic 额外一维、实际版本与 reward_mode。
初始化评估仍是原 benchmark：此前该 Actor 为 33/50；如不能复现，先查 checkpoint hash、环境版本和观测/动作/采样配置。
少量 rollout 可能没有完整成功回合，不能据此认定奖励故障。

## 4. 新 Critic 预热

以下从 iterative Actor **重新创建 Critic 和两个 optimizer**。10 轮预热、每轮 Critic 更新 10 遍是
一个明确的起始预算，不是保证充分收敛的阈值；检查新 rollout 的 Before 指标与成功样本数。

```bash
python train_online.py "${ONLINE_COMMON[@]}" \
  output=outputs/StackCube-v1/oc_budget/online_success_warmup01 \
  stages.online.epochs=10 \
  algo.critic_warmup_epochs=10 algo.critic_warmup_update_epochs=10
```

Actor 在 epoch 1–10 的更新次数应为 0，评估逐 seed 应保持初始结果。
新 `critic_warmup_update_epochs` 只在 Actor 冻结时生效；null 回退到旧 `update_epochs`。
不要把训练后高 EV 当作新数据泛化的证据。

## 5. 短程 Actor 在线训练

从**新成功奖励**预热 checkpoint 恢复。总 epoch 上限为 25，新增 epoch 11–25 共 15 轮，
每轮 2048/128=16 次 Actor 更新，最多 240 次；KL/ratio guard 可降低实际次数。

```bash
python train_online.py "${ONLINE_COMMON[@]}" \
  stages.online.resume=outputs/StackCube-v1/oc_budget/online_success_warmup01/checkpoints/last.pth \
  output=outputs/StackCube-v1/oc_budget/online_success_actor01 \
  stages.online.epochs=25 \
  algo.critic_warmup_epochs=10 algo.critic_warmup_update_epochs=10 \
  algo.update_epochs=1
```

同协议 resume 恢复 Actor、Critic、优化器 moments、epoch、累计 primitive steps 和历史 best。
学习率使用本次显式配置。仿真在 epoch 边界重新 reset，**不保证原轨迹或 RNG 精确续接**。
前一运行未完成回合标为 censored；新运行 episode_id 从 0 开始，以运行目录区分。
恢复的选择种子必须相同，防止把不同验证集的分数混在一起。

新 checkpoint 含 `online_protocol`、Critic 输入格式/维度、benchmark 协议、runtime 协议和来源 hash。
改变奖励、终止语义、时间输入、gamma、reward_scale、实际运行环境等会拒绝 resume。
要更换协议，用 `stages.online.resume=null` 加 `stages.online.initial_ckpt=...`；仅加载 Actor/normalizer，
Critic 和 optimizer 均重新创建。`best.pth`、`initial_policy.pth` 只供评估/初始化，
恢复必须用 `epoch_*.pth` 或 `last.pth`。旧未标记 online checkpoint 仅按历史 legacy 协议恢复。

新模式 best 只按成功率严格提高才替换，raw reward 不打破平局；无收益时保留最初策略。
为跨运行保留 best，epoch checkpoint 额外保存 `selection_best` Actor 快照，文件会比以前大。
旧 checkpoint 没有历史 best 内嵌快照，恢复时仍从恢复策略建立回退。

## 6. 如何读诊断

| 文件/字段 | 含义 |
|---|---|
| `metrics.jsonl` / `metrics.csv` | 每轮训练、价值、优势与验证指标 |
| `Env/Raw_Reward`（兼容字段 `Env/Reward`） | 本轮采样原始环境未折扣奖励总和 |
| `Train/Reward` | 本轮实际训练未折扣奖励总和，已含 reward_scale |
| `Value/Before/*` / `Value/After/*` | 在同一冻结 GAE/bootstrap target 上的 MSE、EV、均值/方差；Before 是新 rollout 更新前，After 是本批拟合后 |
| `Value/*/EV_Valid` | target 方差 <=1e-8 时 false，EV=null；不输出虚假的可靠 EV |
| `Adv/*` | 原始 GAE 和 PPO 实际使用公式的全批标准化优势；Actor_Enabled 区分预热 |
| `diagnostics/episodes.jsonl` | 每个完整/最终 censored 回合唯一摘要，跨 rollout 连续计数 |
| `diagnostics/summary.jsonl` | 按 success/failure/timeout/censored 的数量、均值与 p05/p50/p95；空类别 count=0、数值 null |
| `diagnostics/traces.jsonl` | 有界 primitive 事件与决策 trace，保留 raw/train reward、原始及有效边界、阶段、优势 |

回合摘要的 `raw_return`、`training_return` 是未折扣总和；
`raw_discounted_return`、`training_discounted_return` 从回合首步开始按 primitive 时间折扣。
`MC/Decision_MSE` 比较采样时 V 与完整回合实际 return-to-go，`MC/Start_*` 是回合起点校准。
这是实际采样策略轨迹的事后校准，跨轮回合可能包含不同策略版本，**不是独立 held-out EV，也不是 GAE target**。
censored 只有已观察到的部分回报，MC 字段为 null，不能当作失败或完整回报。

阶段只取环境提供的 success/is_grasped/is_cubeA_on_cubeB/is_cubeA_static；缺失为 unknown/null，
不以机器人是否移动代替任务阶段。`phases` 保存停留步数、最长连续停留和各阶段回报贡献，
flags 同时保存可用步数与为真步数。优势按决策前阶段分组；结果标签在回合结束后才加入。
标准化优势是提供给 Actor 的信号，KL guard、PPO clipping、关闭 Actor 等会改变实际更新贡献，
不能把正优势计数当作真正生效的梯度次数。

默认 trace 最多 20 回合、每回合 300 primitive steps；摘要始终写入。
可设 `online_diagnostics.enabled=false` 关闭阶段细节/trace，
`online_diagnostics.value_before_after=false` 关闭额外前后字段。
诊断不消耗随机数，CPU 测试验证开关不改变采样链或训练后参数。

重点核对：失败停留的 training_return 是否为 0；timeout 的训练 target 是否符合有限时限；
Before 是否改善；完整成功反馈有多少；新成功行为是否伴随旧成功行为丢失。
只看 reward 或 After EV 变高，不能宣布成功率提高。

## 7. 候选比较与打包

先看固定 2000–2049 验证集是否出现可信候选，再运行配对比较。
18000–18099 已看过结果，只能作为固定回归集，不称为全新最终测试集。
新最终测试集需要核对训练/筛选种子来源后再选，本指南不提前消耗。

```bash
python evaluate.py \
  '~comparison.checkpoints' \
  '+comparison.checkpoints={iterative:outputs/StackCube-v1/oc_budget/iterative/checkpoints/best.pth,success_online:outputs/StackCube-v1/oc_budget/online_success_actor01/checkpoints/best.pth}' \
  comparison.output=outputs/StackCube-v1/oc_budget/online_success_regression18000_01 \
  comparison.seed_start=18000 comparison.episodes=100 \
  'comparison.samplers=[cps]' video.episodes=0
```

比较读取各 checkpoint 的原观测/动作配置，**不安装训练 reward wrapper**；
`summary.json` 分开保存 training_protocols 与 benchmark 协议。旧 offline/iterative/online 模型仍可评估。

以下打包配置、协议、指标、诊断和评估（不包含权重/视频/大型二进制）：

```bash
python - <<'PY'
from pathlib import Path
from datetime import datetime
from zipfile import ZipFile, ZIP_DEFLATED
root = Path('outputs/StackCube-v1/oc_budget')
names = ['online_success_smoke01', 'online_success_warmup01', 'online_success_actor01',
         'online_success_regression18000_01']
archive = Path('myrl_online_success_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.zip')
with ZipFile(archive, 'x', compression=ZIP_DEFLATED) as z:
    for name in names:
        folder = root / name
        if not folder.exists():
            print('skip:', folder)
            continue
        for p in sorted(folder.rglob('*')):
            if p.is_file() and p.suffix in {'.json', '.jsonl', '.csv', '.yaml', '.txt'}:
                z.write(p, p.relative_to(root))
print(archive.resolve())
PY
```

提供包后即可分析成功/失败回报、阶段优势、Critic 前后误差和配对得失，不需要上传大模型权重。

## 8. 本次验证范围

已在 CPU（Python 3.12、PyTorch 2.6.0+cpu）执行：

```bash
python -m pytest tests/test_flow_ppo.py tests/test_online_task.py \
  tests/test_experiment.py tests/test_stages.py tests/test_primitive_recording.py -q
```

结果：**42 passed，1 skipped**。跳过的是现有 tensor IPC 测试，运行环境不允许其所需的 Unix socket。
覆盖 Flow log-prob 重放、Actor/编码器行为、primitive 奖励与边界、时间输入、同协议恢复和协议拒绝、
诊断 RNG 不变、旧三阶段交接、旧模型统一比较；另已检查 Hydra 新模式配置可解析和 `git diff --check`。
未运行真实 ManiSkill/SAPIEN、GPU 训练、用户 checkpoint 或 50/100 场景评估；这些由上面的本机步骤补验证。
