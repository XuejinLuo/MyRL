# 固定 Actor，用 Q 比较单次采样与 8 选 1

目标：保持现有约 66% 成功率的 Actor 不变，检查已有数据学到的 Q 能否帮助选择动作。
本实验不采集新数据，不训练 Actor，也不启动 online PPO。Q 的排序可能无效甚至降低成功率，需看配对评估结果。

## 运行两条命令

在仓库根目录，使用原来的 ManiSkill 训练环境：

```bash
python train_critic.py +experiment=oc_budget \
  initial_ckpt=outputs/StackCube-v1/oc_budget/iterative/checkpoints/best.pth \
  manifest=outputs/StackCube-v1/oc_budget/iterative_expand600/round_000/manifest.json
```

默认从头训练独立的 Q/V 编码器和价值网络，使用已有混合数据，训练 10 个 epoch，
`algo.critic_lr=3e-4`、`batch_size=64`。Actor 的参数及 BatchNorm 等缓冲区全部固定，
每个 epoch 检查是否与源 checkpoint 完全一致。已有文件不会被覆盖。
如果本地文件路径不同，只修改 `initial_ckpt` 和 `manifest`。
环境和模型配置必须与 Actor、manifest 一致；不重新拟合 normalizer。

训练完成后：

```bash
python evaluate_q_selection.py +experiment=oc_budget
```

默认加载 `outputs/StackCube-v1/oc_budget/critic/checkpoints/last.pth`，
在同一组 100 个场景（seed 4000–4099）上分别测试 `candidates=[1,8]`，均使用 CPS。
评估实际使用 checkpoint 内保存的模型、环境、normalizer、采样参数及 Actor 权重。
代码会拒绝与源 Actor 验证种子、已知采集种子重叠的评估种子。
如果这组种子已被你用于调参，请改 `q_selection.seed_start`，保留独立最终测试集。

## 看哪个结果

打开 `outputs/StackCube-v1/oc_budget/q_selection/summary.csv`：

- `candidates=1`：原 Actor 直接执行，作为当前对照；不要直接拿历史 66% 代替这行。
- `candidates=8`：同一 Actor 生成 8 个候选 chunk，用 `min(Q1,Q2)` 最高者执行。
- 比较 `Eval/Success_Rate`；同时查看成功次数和 Wilson 95% 区间。

`summary.json` 的 `paired` 还记录单次失败而 8 选 1 成功的数量、反向退步数量、成功率差值。
两组 `candidates_*/episodes.csv` 可按 seed 逐个核对。
相同 seed 匹配的是初始场景；不同候选数量会消耗不同的随机数，不保证两种策略的采样噪声逐步相同。
小幅差异不能直接认定改进。

这次先固定 10 个 epoch 和 8 个候选做一次比较。没有自动用测试成功率挑选 epoch；
`last.pth` 是训练结束的模型，不叫 `best.pth`。如需反复调整 epoch、候选数或学习率，
请另设开发评估种子，最终独立测试种子只用于最终比较。

## 实现细节

- 复用现有 IQL critic 更新：expectile V、双 Q、target Q、真实执行长度对应的折扣及超时 bootstrap。
- 新 checkpoint 同时保存 Q、target Q、V（均包含各自编码器）、Q/V 优化器、原 Actor、
  normalizer、配置、源 Actor/manifest 路径和 SHA256。当前入口不提供中断续训。
- `candidates=1` 不调用 Q，保留原采样、随机数和动作裁剪路径。
- Q 使用自己的观测编码器，观测每次只编码一次；只评分前 `env.exec_steps` 个动作，
  默认是 16 步预测中的前 2 步，完整 chunk 仍交给原执行器。
- 评分前先复现原动作反归一化（裁剪到 `[-1.1,1.1]`）和环境物理边界裁剪，
  再按训练时的 normalizer 编码。执行仍使用原候选，不额外改变动作裁剪规则。
  超过训练归一化范围的动作会在 Q 输入中饱和到 `[-1,1]`，Q 无法区分这些越界部分。
- 这是保守双 Q 的 best-of-K 基线，不是完整复现 IDQL 的加权重采样或 RL-100 全流程。
- 训练及评估输出目录已存在时会报错；新实验需显式设置新 `output`，评估对应设置
  `q_selection.checkpoint` 和 `q_selection.output`。

CPU 回归测试（不需要 ManiSkill 仿真）：

```bash
python -m pytest -q tests/test_q_selection.py tests/test_stages.py tests/test_experiment.py tests/test_flow_ppo.py
```
