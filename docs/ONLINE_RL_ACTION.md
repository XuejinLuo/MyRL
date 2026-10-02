# 73.1% 平台期之后：执行动作前缀 PPO 对照

2026-10-02 的 1000 回合测试：iterative 为 692/1000，online_rl03 最佳策略为
731/1000。累计 1000 万步后的改进为 3.9 个百分点，尚未达到 95%。
本次引入一个独立的 PPO 改动及必要诊断，不承诺这项改动一定突破平台。

## 改了什么

新增 `train_online_rl_action` 配置，`algo.ratio_scope=final_prefix`：

- 中间去噪步仍对整个 latent chunk 计算概率比。Transformer 的注意力会耦合各坐标，
  中间的后 14 步 latent 可能影响后续生成的前 2 步，不能全部裁掉。
- 最后一个去噪步只对 `env.exec_steps` 对应的动作前缀计算概率比、clip 和 KL guard。
  当前 CPS 的条件高斯按坐标独立，给定完整前一 latent，最后输出的后 14 步不会再参与
  推理，也不会被环境执行。因此该条件分布的前缀边缘概率可精确计算。
- 对默认 StackCube 配置，前九步各用 16×7 个坐标，最后一步用 2×7 个坐标。
  这是对 per-step PPO surrogate 的改动，不声称整个最终动作分布的 likelihood 已可精确计算。
- 不改变 Actor 架构、动作 chunk、采样噪声、奖励、学习率或 Critic。原配置继续使用 `full`。
  旧的 `prefix`（所有去噪步都裁前缀）也保留，但不是本次使用的模式。
- 生成和 benchmark 采样过程不变，只在训练时减少最终未执行坐标对更新的干扰。
  可能减少无关梯度/概率比波动，实际收益需对照验证。

理论边界：这里依赖当前关闭 temporal ensembling，最终 tail 不进入环境或后续决策。
如果以后复用整个动作 chunk，必须重新检查该假设。

## 获取代码

```bash
conda activate rl100
cd ~/MyRL
git fetch origin
git switch --track origin/codex/online-effective-action-ppo
```

已有本地分支时用 `git switch codex/online-effective-action-ppo`，再 `git pull --ff-only`。
若之前通过新缓存目录才解决 Python 正则编译异常，在新终端继续设置：

```bash
export PYTHONPYCACHEPREFIX="$(mktemp -d /tmp/myrl-pycache-XXXXXX)"
export HYDRA_FULL_ERROR=1
```

## 1. 先做环境对照（每条路径 50 回合）

```bash
python -m tools.diagnostics.compare_online_backends \
  --checkpoint outputs/StackCube-v1/oc_budget/online_rl03/checkpoints/best.pth \
  --output outputs/StackCube-v1/oc_budget/backend_audit01 \
  --seed-start 50000 --episodes 50
```

同一权重和 CPS 配置，分别启动三个独立进程，运行 CPU benchmark、CPU training wrapper、
GPU training wrapper。不会在同一进程混用 PhysX 后端；失败会直接报错，不写虚构的汇总。
输出 `summary.json` 及三份逐回合 JSON，包含实际初始特权状态、初始点云统计、首个动作前缀、
成功标记及步数。记录 checkpoint SHA256，确保检查期间权重未更换。

重点读取：

- `cpu_benchmark_success_rate` 与各路径 `success_rate`：同一策略的任务完成率。
- `success_disagreements`：与 CPU benchmark 在同一 seed 下的成功结果分歧数。
- `max_initial_state_abs_difference`、`initial_states_over_1e_5`：检查相同 seed 是否真的
  产生近似一致的初始状态。不要仅凭 seed 相同就认为环境相同。

CPU training 在首次成功时结束，benchmark 可能继续到时限；回报与长度不应直接比大小。
`first_success_chunk_end` 只是首次观测到成功的 chunk 结束步数，不是精确的首次成功原始步。
GPU 审计固定为 **一个环境**，隔离后端差异；不能据此证明 32 环境部分 reset 已完全等价。
若 CPU training 与 benchmark 成功结果出现明显分歧，或 CPU/GPU 差距很大，先回传审计
结果定位，再开始长训。50 回合是诊断，不是 95% 验收。

## 2. 原 PPO 对照组：新增 100 万步

```bash
python train_online.py --config-name train_online_rl_action \
  algo.ratio_scope=full \
  stages.online.initial_ckpt=outputs/StackCube-v1/oc_budget/online_rl03/checkpoints/best.pth \
  online_training.total_env_steps=1000000 \
  output=outputs/StackCube-v1/oc_budget/online_rl_full_control01
```

## 3. 新 PPO 实验组：新增 100 万步

```bash
python train_online.py --config-name train_online_rl_action \
  stages.online.initial_ckpt=outputs/StackCube-v1/oc_budget/online_rl03/checkpoints/best.pth \
  online_training.total_env_steps=1000000 \
  output=outputs/StackCube-v1/oc_budget/online_rl_action01
```

两组都从同一个 best Actor 初始化，重建 Critic 和优化器并执行相同预热，使用相同 seed、
环境数、奖励及学习率。实验组唯一的算法差别是 `ratio_scope`。
这里的 100 万是**新实验的步数**；不是把旧 1000 万步 checkpoint resume 到 100 万。
不需要重新运行 offline/iterative。若原训练用了 8 环境，两条命令都加：
`online_rollout.num_envs=8 algo.steps_per_epoch=512`。

协议改变不能从旧 `last.pth` 直接 resume；代码会拒绝跨 scope 恢复。
本模式中断时则可正常从自己的 last 恢复，例如：

```bash
python train_online.py --config-name train_online_rl_action \
  stages.online.resume=outputs/StackCube-v1/oc_budget/online_rl_action01/checkpoints/last.pth \
  online_training.total_env_steps=1000000 \
  output=outputs/StackCube-v1/oc_budget/online_rl_action02
```

此时预算恢复为当前实验的累计步数。已完成 100 万且没有 pending 评估时，不运行这条恢复命令。
对照组恢复还须保留 `algo.ratio_scope=full`。

## 4. 看哪些新增指标

每 10 个 epoch 对固定选取的 32 个 rollout 样本做更新前后对照，所有指标写入原来的
`metrics.csv/jsonl`。不额外采集环境数据，不消费随机数、不写参数 `.grad`、不修改 Adam 状态。

| 指标 | 含义 |
| --- | --- |
| `Probe/Gradient_Norm_Step_0..9` | 更新前，每个去噪步对当前 PPO 损失的梯度范数；包含 `1/num_steps` 权重，使用同一小批探针样本 |
| `Probe/Conditional_KL_Step_0..9` | 整批 PPO 更新后，在保存的 latent 条件上计算的高斯 KL；按当前 scope 汇总坐标 |
| `Probe/Executed_Normalized_RMS` | 固定初始噪声和各步噪声，完整重新生成后的执行前缀变化 |
| `Probe/Unused_Normalized_RMS` | 同样条件下，最终未执行 tail 的变化 |
| `Probe/Control_RMS_Dim_0..6` | 反归一化并应用 normalizer 裁剪后，各动作通道的前缀变化；尚未经过环境 action-space 裁剪，不是机器人位移 |

每步 KL 不能与 `ppo/approx_kl` 的跨步平均直接混比，也不是最终动作分布的 KL。
梯度范数只代表这批探针样本，不能单独认定哪个步骤“学会了任务”。诊断开销包含在
`Perf/Update_Seconds` 中，可用 `online_diagnostics.policy_probe_every=0` 关闭。
验证成功率是否提高仍是主要指标，不因动作变化更大就自动判定实验更好。

## 5. 回传及后续验收

先回传 backend_audit01 的全部 JSON，以及两组的 `config.yaml`、`metrics.csv`、
`goal_status.json`、`selection.json`。不必上传权重或再次跑原来的 1000 回合测试。

```bash
python - <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
root = Path('outputs/StackCube-v1/oc_budget')
runs = [root/name for name in ('backend_audit01', 'online_rl_full_control01', 'online_rl_action01')]
for run in runs:
    if not run.is_dir():
        raise SystemExit(f'Missing directory: {run}')
with ZipFile('myrl_action_ppo_results01.zip', 'x', ZIP_DEFLATED) as z:
    for run in runs:
        for p in run.rglob('*'):
            if p.is_file() and 'checkpoints' not in p.parts and p.suffix in {'.json', '.jsonl', '.csv', '.yaml'}:
                z.write(p, p.relative_to(root))
print('myrl_action_ppo_results01.zip')
PY
```

先用原验证集判断趋势和修改效果。若明确改善，再增加新模式的训练预算。
40000–40999 已经用于本轮最终测试，50000–50049 用于这次诊断；下一轮选定模型后的
独立验收预留 60000–60999（须确认未用于任何训练或调参），不反复用于挑模型。

## 验证与限制

```bash
python -m pytest tests/test_effective_action_ppo.py tests/test_flow_ppo.py \
  tests/test_online_rl.py tests/test_online_task.py \
  tests/test_online_eval_recovery.py tests/test_checkpoint_scalars.py -q
```

测试覆盖最终高斯前缀边缘密度、中间 latent 耦合、未执行最终坐标的梯度排除、探针对
训练随机流及更新的无干扰、同模式恢复及跨模式拒绝、审计成功统计与初始状态差异。
开发环境使用 CPU 测试和仿真接口替身，没有运行实际 ManiSkill CUDA 训练或证明成功率提高。

背景参考：[DPPO 官方项目](https://diffusion-ppo.github.io/) 的内外层 MDP 解释；
[ManiSkill RNG 文档](https://maniskill.readthedocs.io/en/latest/user_guide/concepts/rng.html)。
`final_prefix` 是针对本仓库当前 CPS 与执行语义的实现选择，不声称复现某篇论文的完整方法。
