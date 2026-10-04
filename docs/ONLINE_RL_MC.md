# 完整回合 Monte Carlo 回报的 Flow PPO

现有 CPU 对照训练 30 万步后，最佳验证仍是初始策略的 77/100，最终策略为 71/100。
这不足以把平台期归因于 GPU 后端；之前的高 Critic explained variance 衡量的是
GAE/自举目标拟合，不能当成实际完成任务回报的预测能力。

这次改为用完整回合的实际回报训练 Actor 和 Critic。它仍是自主环境交互的在线 RL，
不需要新增示范、人工接管或重跑 offline/iterative。95% 是实验目标，尚未实测达到。

这轮 MC 实验未超过初始 best 时，下一项可选对照见
[Actor 不使用学习型价值基线](ONLINE_RL_MC_NOBASELINE.md)。本文默认仍为 `G - V`。

## 学习目标与采样

`train_online_rl_mc` 使用 `algo.return_estimator=mc`：

1. 每批保持策略不变，采满固定 **64 个完整回合**，全部成功、失败和超时回合都保留。
2. 将一个回合内的 chunk 按时间相连，从真实终点反向计算
   `G[t] = reward[t] + gamma**actual_steps[t] * G[t+1]`。
   `reward[t]` 已包含 chunk 内的逐原始步折扣和 reward scale；回合终点的后续回报为零。
3. Actor 使用 `G[t] - V_before(s[t])`，随后沿用整批优势标准化和原 full-event Flow PPO。
   Critic 拟合固定的 `G[t]`。目标完全不使用下一状态 Critic 预测。
4. 不让未完成轨迹跨 Actor 更新，不在固定 rollout 长度处补 bootstrap，
   也不把 `gae_lambda=1` 当作完整回合的替代品。MC 模式不使用 `gae_lambda` 或
   `algo.steps_per_epoch` 控制采样；旧 GAE 配置继续使用原来的实现。

新配置沿用原 Actor、冻结视觉编码器、特权状态 Critic、奖励、CPS、full 概率比、
学习率、PPO 更新轮数和 5 批 Critic 预热。由于收集单位变成完整回合，
每批决策数和预热的原始步数会随回合长度变化；它不是严格匹配更新次数的单行参数对照。
MC 的方差也可能高于 GAE，收益需看验证结果。

GPU 模式按 wave 采样：每个环境各跑一个回合，提前结束的 lane 等待其他 lane 完成，
全部结束后才重置并开始下一 wave。每批回合数必须是环境数的正整数倍。
PhysX 在等待期间仍计算填充步，但它们不会进入训练数据、回报、成功率或有效环境步预算。
该做法可能比持续自动 reset 的 GAE 采样慢。日志吞吐量按有效步计算。

步数预算在**完整一批结束后**检查。默认 64 回合 × 300 步时，最终有效步数
最多比预算多 19,199 步；`Return/Budget_Overshoot` 记录最后一批的超出量。
评估也在完整批次结束后进行，不保证恰好在第 50,000 步发生。

## 获取代码与第一轮训练

```bash
conda activate rl100
cd ~/MyRL
git fetch origin
git switch --track origin/codex/online-complete-episode-mc

python train_online.py --config-name train_online_rl_mc \
  stages.online.initial_ckpt=outputs/StackCube-v1/oc_budget/online_rl03/checkpoints/best.pth \
  online_training.total_env_steps=300000 \
  output=outputs/StackCube-v1/oc_budget/online_rl_mc_cpu01
```

已有本地分支时用 `git switch codex/online-complete-episode-mc`，再运行
`git pull --ff-only origin codex/online-complete-episode-mc`。合并 PR 后也可更新 `online` 使用。

先采用 CPU 单环境，以便与已完成的 CPU 对照比较；网络仍使用配置中的 CUDA 设备。
这是从同一个 `online_rl03` 最佳 Actor **新建 MC 实验**，重建 Critic 和优化器。
不要从旧 GAE `last.pth` resume 到 MC；代码会拒绝混用。总预算包含 5 批 Critic 预热。
运行目录必须尚不存在。

如果你的 Python 环境仍依赖之前的独立缓存规避导入报错，在启动前继续设置：

```bash
export PYTHONPYCACHEPREFIX="$(mktemp -d /tmp/myrl-pycache-XXXXXX)"
export HYDRA_FULL_ERROR=1
```

## 中断恢复与后续预算

仅当第一轮尚未完成时，从 **MC 自己的** `last.pth` 恢复，保持同一回合配额和后端：

```bash
python train_online.py --config-name train_online_rl_mc \
  stages.online.resume=outputs/StackCube-v1/oc_budget/online_rl_mc_cpu01/checkpoints/last.pth \
  online_training.total_env_steps=300000 \
  output=outputs/StackCube-v1/oc_budget/online_rl_mc_cpu02
```

预算是该 MC 实验的累计有效步数，不是再增加 30 万。已正常完成时无需运行恢复命令。
先检查这轮结果，再决定是否把预算提高到 100 万；不是默认继续跑 1000 万。
恢复会重置模拟器，不是逐轨迹精确重现，但 checkpoint 边界没有待完成训练回合。
旧 GAE checkpoint 仍可以在原 GAE 配置下恢复。

GPU 采样也支持。若另开 GPU 实验，可使用：

```bash
python train_online.py --config-name train_online_rl_mc \
  online_rollout.backend=gpu online_rollout.num_envs=32 \
  online_rollout.episodes_per_batch=64 \
  output=outputs/StackCube-v1/oc_budget/online_rl_mc_gpu01
```

这条是可选的新实验，不是 CPU 运行的 resume，也不要求现在同时运行。

## 结果与回传

训练自动执行初始、周期和最终 benchmark 验证，并保存 `best.pth` 和 `last.pth`。
验证成功率相同则保留先前 best；所以必须查看 `selection.json` 的 epoch，
不能把 epoch 0 的 best 当作 RL 已取得提升。100 回合验证用于选择模型，不能代替独立验收。

| 字段 | 含义 |
| --- | --- |
| `Value/Target_Is_MC` | 新模式为 1，GAE 为 0 |
| `Value/MC/Before/*` | 更新前 Critic 对本批实际 return-to-go 的预测质量 |
| `Value/MC/After/*` | 更新后对同一批数据的拟合质量，不是独立泛化测试 |
| `Value/MC_Start/Before/*` | 每回合起点各取一个样本，预测与真实回合回报比较 |
| `Return/Complete_Episodes` | 本批完整回合数，默认 64 |
| `Return/Decisions` | 本批有效 chunk 决策数，随回合长度变化 |
| `Return/Budget_Overshoot` | 完成整批导致的有效步数超预算量 |
| `Episode/censored_Count` | 正常完成的 MC 批次应为 0 |

MSE 越低越好；explained variance 遇到目标方差为零会写 null。
`Before` 是更新前的新采样回合预测，不是专门保留的独立测试集。
仍以 benchmark 成功率的变化判断策略收益，不能只凭 Critic 拟合提高判断有效。

训练完打包以下日志即可，不需要上传模型权重或视频：

```bash
python - <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
root = Path('outputs/StackCube-v1/oc_budget')
run = root / 'online_rl_mc_cpu01'
if not run.is_dir():
    raise SystemExit(f'Missing directory: {run}')
with ZipFile('myrl_mc_cpu_results01.zip', 'x', ZIP_DEFLATED) as z:
    for p in sorted(run.rglob('*')):
        if p.is_file() and 'checkpoints' not in p.parts and p.suffix in {'.json', '.jsonl', '.csv', '.yaml'}:
            z.write(p, p.relative_to(root))
print('myrl_mc_cpu_results01.zip')
PY
```

这会包含 config、protocol、metrics、goal_status、selection、各次验证和完整回合诊断。
如果发生过恢复，也请打包恢复运行目录。暂不需要再次对未变的初始 best 做 1000 回合验收。

## 开发验证

```bash
python -m pytest tests/test_online_mc.py tests/test_effective_action_ppo.py \
  tests/test_flow_ppo.py tests/test_online_rl.py tests/test_online_task.py \
  tests/test_online_eval_recovery.py tests/test_checkpoint_scalars.py -q
```

覆盖逐原始步折扣、成功/超时边界、不依赖 Critic bootstrap、完整回合预算、
长短回合配额、GPU lane 等待与重置、诊断对齐、真实 PPO 更新、checkpoint 和跨目标恢复拒绝。
开发验证使用 CPU PyTorch 与模拟环境接口；未运行真实 ManiSkill CPU/CUDA 训练，
没有声称本次修改已提高实际成功率。
