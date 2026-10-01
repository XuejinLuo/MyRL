# StackCube 自主在线 RL：训练与 95% 目标验收

本方案从现有 oc_budget Actor 初始化，之后通过自主交互和 Flow PPO 更新策略。
无需新增示范、人工接管或重新运行 offline/iterative。95% 是需要实测的目标，代码修改不构成达标承诺。

## 修改内容

- Actor 仍使用原来的 RGB 点云、object-budget 采样和机器人状态，视觉编码器保持冻结，更新 Flow backbone。
- 新的 Critic 使用训练环境中的 Panda qpos/qvel、TCP pose、两方块 pose/速度、抓取/堆叠标志（53 维），加剩余时间，共 54 维。这些状态不进入 Actor，也不用于 benchmark 选动作。
- 任务为 300 个原始环境步内首次成功，成功 +1 并结束，gamma=1；超时结束且不 bootstrap。采样批次结束仍 bootstrap。
- 过程奖励为 `success + beta * (gamma * Phi(next) - Phi(current))`，beta 默认 1。Phi 包含接近、抓取、抬起、对齐、松手稳定，固定在 [0, 1]。成功、失败、超时的终止势能均为零。因此 gamma=1 时完整轨迹回报为 `success - beta * Phi(initial)`，反复停在中间状态不会不断得分。失败轨迹的 shaped return 可以为负，成功回报也不一定恰为 1，属于正常结算。
- 默认 32 个 GPU 物理环境，每环境采集 128 个 chunk 决策（默认每 chunk 执行 2 步）；Actor 批量推理。点云预处理保留原来的 CPU/NumPy 实现，整体加速幅度需要实测。
- 每环境独立维护轨迹、剩余时间、实际步数、结束标志、reset 和诊断；先在 `[T, N]` 上计算 GAE，再展平训练。chunk 中提前结束的 lane 可能随批量物理步产生填充推进，填充不计入奖励、步数或 PPO，终止观测在其实际边界保存。
- Critic 先预热 5 个 rollout，每批更新 8 遍；随后每批最多 4 遍 PPO，Actor 学习率 1e-7，KL guard 保持启用。Critic 取消 value clipping。保留原采样噪声与 full latent ratio。
- 默认总预算 100 万实际环境步，包含 Critic 预热；每约 10 万步评估，预算结束强制评估/保存。总步数可能多出最多一个 vector chunk（默认不足 64 步）。`epochs=100000` 是次级保护上限。
- 最佳模型只按验证成功率选择，平分保留较早模型；保留 `best.pth`、`last.pth` 和 `initial_policy.pth`，默认不累计 epoch 大文件。训练达到验证目标不会自动停止。
- GPU 训练的验证由独立进程运行原来的单环境 CPU benchmark，避免在同一进程混用物理后端。所有最终比较仍使用原 benchmark（无训练奖励 wrapper），所以 GPU 训练成功率与 CPU 验证成功率应分别看待。

新功能通过 `--config-name train_online_rl` 启用；原 `train_online` 配置和旧 checkpoint 协议保持兼容。
GPU 路径目前明确限定为 StackCube/Panda、global pointcloud（默认 oc_budget），不支持 object_centric/relational 变体。

## 1. 获取代码

在已有的 MyRL 目录运行：

```bash
conda activate rl100
git fetch origin
git switch --track origin/codex/online-asymmetric-rl
```

如果本地已经有这个分支，使用 `git switch codex/online-asymmetric-rl` 后 `git pull --ff-only`。
初始权重默认是：

```text
outputs/StackCube-v1/oc_budget/iterative/checkpoints/best.pth
```

也可在首次训练命令末尾增加 `stages.online.initial_ckpt=outputs/StackCube-v1/oc_budget/online_success_actor01/checkpoints/best.pth`。
新奖励和 Critic 与过去的 online checkpoint 不同：旧权重用 `initial_ckpt` 初始化 Actor，不用 `resume` 恢复旧 Critic/优化器。

## 2. 短程检查：确认你的 GPU 仿真、渲染和 Actor 更新可以运行

首次运行保留现有 ManiSkill 环境版本，避免同时升级依赖。需要可用的 NVIDIA CUDA 和 SAPIEN 渲染支持。
本仓库的 CPU 合约测试无法代替这一步真实 GPU 检查。

```bash
python train_online.py --config-name train_online_rl \
  output=outputs/StackCube-v1/oc_budget/online_rl_smoke01 \
  online_rollout.num_envs=4 \
  online_training.total_env_steps=1024 \
  online_training.eval_every_env_steps=1024 \
  algo.steps_per_epoch=32 \
  algo.critic_warmup_epochs=0 \
  algo.update_epochs=1 \
  'eval.seeds=[2000,2001]'
```

这只检查数据路径和更新流程，不评价学习效果，也不作为正式训练的 resume 来源。
输出中应有有限的 value/advantage、`ppo/actor_updates > 0`、有效的 `ppo/replay_logprob_error`，且总步数达到预算。
运行结束应生成 `checkpoints/last.pth`、`checkpoints/best.pth`、`goal_status.json`。

## 3. 正式训练：先跑 100 万步

```bash
python train_online.py --config-name train_online_rl \
  output=outputs/StackCube-v1/oc_budget/online_rl01
```

训练会先评估初始化模型，再预热 Critic，再持续更新 Actor。选择 seeds 为 2000–2099。
显存不足时，在首次正式运行时加 `online_rollout.num_envs=8 algo.steps_per_epoch=512`，
这样每批仍有约 4096 个 chunk 决策。之后续训必须沿用同样的环境数和 backend。
CPU 单环境替代路径可设 `online_rollout.backend=cpu online_rollout.num_envs=1 algo.steps_per_epoch=4096`；
同样支持新奖励和 Critic，但采样速度需要单独衡量。

所有 output 都必须是新的目录。失败重跑可改用 `...02`，不要覆盖已有训练。

## 4. 继续训练到累计 1000 万步

如果旧版报 `Unsupported global: numpy.core.multiarray.scalar`，这是 PR #18 将 NumPy
奖励标量写进 checkpoint 导致的序列化兼容问题。更新到包含修复的代码后直接重跑续训命令，
会在 `weights_only=True` 下兼容读取旧数值标量，并将后续保存的指标转换为 Python 原生类型。
无需重新训练 100 万步或覆盖原 checkpoint；模型、Critic、优化器和累计步数保持恢复。
OpenGL 的 `No OpenGL_accelerate module loaded` 信息不是这个异常的原因。

```bash
python train_online.py --config-name train_online_rl \
  stages.online.resume=outputs/StackCube-v1/oc_budget/online_rl01/checkpoints/last.pth \
  online_training.total_env_steps=10000000 \
  output=outputs/StackCube-v1/oc_budget/online_rl02
```

这是累计 1000 万步，非额外 1000 万步。恢复 Actor、Critic、优化器状态、epoch、总步数和历史最佳模型。
仿真会在边界重新 reset，不保证逐轨迹精确重放。中断后可以从最近保存的 last.pth 继续，仍须指定新的 output。
改变奖励、gamma、Critic 格式、backend、并行数或验证 seeds 时需开始新实验，不能混作同一次 resume。

100 万步是首次检查点；若成功率仍在上升，可继续长训。若持续停滞/下降或 KL guard 几乎阻止全部更新，
应使用日志定位后调整，1000 万步本身不保证达标。默认不会自动升学习率或更改探索分布。

## 5. 最终 1000 回合验收

用最终选定的 best.pth，一次性评估新 seeds 40000–40999；不要用这批 seeds 调参或选模型。
下面比较原 iterative 和新 RL，固定 CPS 采样器：

```bash
python evaluate.py \
  '~comparison.checkpoints' \
  '+comparison.checkpoints={iterative:outputs/StackCube-v1/oc_budget/iterative/checkpoints/best.pth,rl:outputs/StackCube-v1/oc_budget/online_rl02/checkpoints/best.pth}' \
  comparison.output=outputs/StackCube-v1/oc_budget/online_rl_test40000_01 \
  comparison.seed_start=40000 comparison.episodes=1000 \
  'comparison.samplers=[cps]' video.episodes=0

python -m tools.diagnostics.check_success_target \
  outputs/StackCube-v1/oc_budget/online_rl_test40000_01/summary.json
```

若只完成了第 3 步，把 checkpoint 路径中的 `online_rl02` 换成 `online_rl01`。
验收工具输出成功次数、成功率、Wilson 95% 区间和 `measured_target_met`。
至少 1000 回合且实测成功率 >=95% 时退出码为 0，否则为 1。
950/1000 表示本次样本达到 95%，并不意味着总体成功率的置信下界也达到 95%；工具分别报告两者。
此前的 seeds 18000–18099 已经用于分析，不作为本次最终独立验收。

## 日志与回传

- `metrics.jsonl/csv`：验证成功率、累计环境步数、Actor 更新数、KL、Critic 拟合、采样/更新时间与吞吐。
- `goal_status.json`：当前/最佳验证成功率和目标状态；不是最终独立测试结论。
- `protocol.json`：奖励、Critic 格式、ManiSkill 版本、训练后端/并行数、源权重 SHA256。
- `diagnostics/env_XXX/episodes.jsonl`：各环境的完整回合；CPU 单环境直接在 diagnostics 下。
- `eval/validation_*/episodes.csv`：逐场景验证结果。

打包首次正式训练日志（不包含大型 checkpoint）：

```bash
python - <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
run = Path('outputs/StackCube-v1/oc_budget/online_rl01')
if not run.is_dir():
    raise SystemExit(f'Missing run: {run}')
with ZipFile('myrl_online_rl01_logs.zip', 'w', ZIP_DEFLATED) as z:
    for p in run.rglob('*'):
        if p.is_file() and 'checkpoints' not in p.parts and p.suffix in {'.json', '.jsonl', '.csv', '.yaml'}:
            z.write(p, p.relative_to(run.parent))
print('myrl_online_rl01_logs.zip')
PY
```

## 开发验证与边界

```bash
python -m pytest tests/test_online_rl.py tests/test_online_task.py tests/test_flow_ppo.py -q
```

覆盖势函数结算、Critic/Actor 隔离、GPU 接口模拟、不同 lane 的分割 ID、chunk 内提前结束、padding 排除、
独立 GAE、预算结束强制评估、checkpoint/resume 与原行为兼容。接口模拟使用 CPU fake 环境，不声称完成了真实 CUDA/SAPIEN 训练。

参考：[ManiSkill PPO 示例](https://github.com/mani-skill/ManiSkill/blob/main/examples/baselines/ppo/examples.sh)、
[非对称 Actor–Critic](https://arxiv.org/abs/1710.06542)、
[势函数奖励塑形](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)。
