# MC 对照：Actor 不使用学习型价值基线

上一次 `online_rl_mc_cpu01` 完成 301,026 个有效环境步：初始验证 77/100，
后续最高 75/100，最终 70/100，best 仍为 epoch 0。它没有证明 MC 改动提高了成功率。
新回合起点的 Critic 预测误差也高于使用上一批回报均值的简单预测，但这还不能证明
Critic 是平台期的唯一原因。本次用一个直接对照检验 Actor 对 Critic 预测的依赖。

## 改动范围

新配置 `train_online_rl_mc_nobaseline` 继承 PR #22 的 `train_online_rl_mc`，
只修改输出目录和 `algo.mc_baseline=none`：

| 模式 | Actor 原始信号 | Critic 目标 |
| --- | --- | --- |
| `mc_baseline=value`（原默认） | `G - V_before` | 完整回合回报 `G` |
| `mc_baseline=none`（新对照） | `G` | 同一个完整回合回报 `G` |

两种信号都沿用现有整批标准化，然后进入 full-event Flow PPO、clip、KL guard
及更新探针。`none` 指不减学习型 V；标准化仍会减去批均值。
奖励仍为 success + potential shaping，所以 `G` 不是简单把每个成功决策填 1、
失败决策填 0，仍包含原有势函数回报。不添加新奖励或成功筛选。

保持原始 Actor 来源、奖励、学习率、CPS 噪声、每批 64 回合、PPO 4 轮更新和
5 批 Actor 冻结预热。预热对新信号不是必要条件，这里保留以便和上次 MC 实验比较。
Critic 继续训练和记录指标，预测不进入新模式的 Actor 优势。无需重跑上次对照。

旧 GAE 和 MC 默认行为保留。旧 MC checkpoint 缺少 `mc_baseline` 时按 `value`
解释，只允许恢复到原模式；跨 baseline 的 resume 会报错，应从 Actor 权重新建实验。

## 获取与训练

```bash
conda activate rl100
cd ~/MyRL
git fetch origin
git switch --track origin/codex/online-mc-no-value-baseline

python train_online.py --config-name train_online_rl_mc_nobaseline \
  stages.online.initial_ckpt=outputs/StackCube-v1/oc_budget/online_rl03/checkpoints/best.pth \
  online_training.total_env_steps=300000 \
  output=outputs/StackCube-v1/oc_budget/online_rl_mc_nobaseline_cpu01
```

若本地分支已经存在，改用 `git switch codex/online-mc-no-value-baseline`，然后
`git pull --ff-only origin codex/online-mc-no-value-baseline`。合并后也可以更新 `online`。

从同一个 `online_rl03` best Actor 新建实验，不从上次 MC 的 last 恢复。
CPU 单环境模拟，网络仍按配置使用 CUDA。先跑约 30 万步，包含 5 批预热。
为保留完整回合，每批结束才检查预算，默认最多超出 19,199 个有效步。
每约 5 万步及最后自动评估；目录必须尚不存在。

如之前的 Python 导入故障仍需独立缓存，在训练前继续设置：

```bash
export PYTHONPYCACHEPREFIX="$(mktemp -d /tmp/myrl-pycache-XXXXXX)"
export HYDRA_FULL_ERROR=1
```

仅在本次训练中断且预算尚未完成时恢复：

```bash
python train_online.py --config-name train_online_rl_mc_nobaseline \
  stages.online.resume=outputs/StackCube-v1/oc_budget/online_rl_mc_nobaseline_cpu01/checkpoints/last.pth \
  online_training.total_env_steps=300000 \
  output=outputs/StackCube-v1/oc_budget/online_rl_mc_nobaseline_cpu02
```

预算仍是本次实验累计值。正常完成后不用执行恢复命令，也不用默认续到 1000 万步。

## 看哪些结果

- `Adv/Uses_Learned_Baseline` 必须为 0；原 MC 配置为 1。
- 新模式的原始 `Adv/Mean` 应与 `Value/Return_Mean` 一致，`Adv/Std` 是实际回报的标准差。
- `protocol.json` 的 `returns.mc_baseline` 为 `none`，checkpoint 也保存这一协议。
- `Value/MC/*` 仍衡量 Critic；即使拟合很好，也不能代替策略成功率。
- 用 `Eval/Success_Rate`、`selection.json` 和 `goal_status.json` 判断是否超过初始 Actor。
  若 best 仍为 epoch 0，就是保留了初始策略，不能当作新 RL 提升。

这是关于学习信号的对照实验，不保证达到 95%。没有真实 ManiSkill 训练验证之前，
测试通过只证明实现行为符合预期。100 回合验证用于选择模型，不等于独立验收。

## 打包回传

训练结束后运行下面命令。它要求存在新实验目录，自动包含同前缀的恢复目录，
并在上次 `online_rl_mc_cpu01` 仍存在时一起打包，方便直接比较。不包含权重和视频。

```bash
cd ~/MyRL
python - <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from datetime import datetime

root = Path('outputs/StackCube-v1/oc_budget')
runs = sorted(p for p in root.glob('online_rl_mc_nobaseline_cpu*') if p.is_dir())
if not runs:
    raise SystemExit('未找到 online_rl_mc_nobaseline_cpu* 训练目录')
control = root / 'online_rl_mc_cpu01'
if control.is_dir():
    runs.append(control)
archive = Path(f'myrl_mc_nobaseline_results_{datetime.now():%Y%m%d_%H%M%S}.zip')
count = 0
with ZipFile(archive, 'x', ZIP_DEFLATED) as z:
    for run in runs:
        for p in sorted(run.rglob('*')):
            if (p.is_file() and 'checkpoints' not in p.relative_to(run).parts
                    and p.suffix in {'.json', '.jsonl', '.csv', '.yaml', '.yml', '.log'}):
                z.write(p, p.relative_to(root))
                count += 1
print(f'已打包 {count} 个文件：{archive.resolve()}')
print('包含运行目录：', ', '.join(p.name for p in runs))
PY
```

上传输出的 ZIP 即可，里面包括配置、协议、训练指标、选择结果、各次评估及回合诊断。

## 开发验证

```bash
python -m pytest tests/test_mc_baseline.py tests/test_online_mc.py \
  tests/test_effective_action_ppo.py tests/test_flow_ppo.py tests/test_online_rl.py \
  tests/test_online_task.py tests/test_online_eval_recovery.py tests/test_checkpoint_scalars.py -q
```

包括 Critic 预测变化时新模式信号和实际 Actor 更新保持一致、回报目标不变、
新配置只改 baseline/输出目录、旧协议兼容、跨模式恢复拒绝及完整回合/GPU lane 回归。
