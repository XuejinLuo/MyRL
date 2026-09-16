# StackCube：离线 → 迭代离线 → 在线

基于 `XuejinLuo/MyRL` 的 `online` 分支，基准提交 `0427188a83b78e25bda4a8fe1d4d74ed161230e7`。
这份改动提供训练和评测流程，不包含训练好的 StackCube 权重或实测成功率。

## 1. 安装补丁

将下载的 ZIP 解压到独立目录，在你自己的 MyRL 根目录运行：

```bash
conda activate rl100
cd ~/MyRL
git apply --check /path/to/extracted/stackcube.patch
git apply /path/to/extracted/stackcube.patch
python -m pip install imageio imageio-ffmpeg
```

`git apply --check` 不通过时，不要强行覆盖本地改动。包内 `files/` 是所有修改文件的完整版本，便于对照合并。
沿用现有可工作的 PyTorch、ManiSkill、torch_cluster 环境，不需要升级这些包。

## 2. 准备 StackCube 数据

不能复用 PullCubeTool 的权重、示范数据或 `dataset_stats.json`。
下面脚本调用 ManiSkill 官方下载与轨迹重放工具，自动定位原始 motionplanning 数据并转换成点云 + `pd_ee_delta_pose`：

```bash
python prepare_stackcube_demos.py --output demos --count 100
STACKCUBE_H5="$(cat demos/stackcube_h5_path.txt)"
```

如果已经下载了原始 H5 和同名 JSON，可以跳过下载：

```bash
python prepare_stackcube_demos.py \
  --raw ~/.maniskill/demos/StackCube-v1/motionplanning/trajectory.h5 \
  --output demos --count 100
STACKCUBE_H5="$(cat demos/stackcube_h5_path.txt)"
```

原始路径以你实际文件位置为准。转换通过动作重放验证控制器，不使用 `--allow-failure`，不把失败演示强行纳入初始数据。转换可能丢弃不能成功重放的轨迹，脚本会打印实际数量。不要对正在训练的数据重新运行转换；新数据放新目录。点云数据较大，首次建议 100 条。

官方工具的等价转换命令：

```bash
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path /path/to/trajectory.h5 \
  --obs-mode pointcloud --target-control-mode pd_ee_delta_pose \
  --sim-backend physx_cpu --save-traj --count 100 --num-procs 1
```

## 3. 从头到尾执行

```bash
python run_stackcube.py \
  --h5 "$STACKCUBE_H5" \
  --output outputs/stackcube/run01
```

默认流程：

| 阶段 | 入口 | 默认设置 | 传给下一阶段 |
|---|---|---|---|
| 离线初始化 | train_offline.py | 100 条演示、200 epochs、BC actor | 验证集最佳 EMA 权重 |
| 数据转换 | prepare_iterative_data.py | 相同演示子集、冻结归一化参数 | primitive 轨迹 manifest |
| 迭代离线 RL | train_iterative.py | 3 轮 × 新采集 100 回合 × 30 epochs | 成功率严格超过 incumbent 才替换 |
| 在线 RL | train_online.py | 100 epochs、原有 Flow PPO 设置 | 包括 epoch 0 在内的验证集最佳权重 |
| 最终对比 | evaluate_checkpoint.py | 各阶段最佳模型，CPS/ODE 各 100 回合 | comparison/summary.csv |

**BC 初始化是明确的实验选择：**在 `train_offline.py` 中设置 `algo.use_bc_only=true`，让 actor 先学习专家演示，避免初始化阶段被不成熟的 critic 筛选；critic 仍训练。随后 `train_iterative.py` 明确恢复 `algo.use_bc_only=false`，执行原有 IDQL。没有更改 IDQL/PPO 的更新公式。若需研究“从零开始 IDQL”作为对照，复制 dry-run 的离线命令并设 `algo.use_bc_only=false`，用独立输出目录。

**建议先检查离线基线。** 不希望直接跑完整流程时：

```bash
python run_stackcube.py --h5 "$STACKCUBE_H5" --output outputs/stackcube/run01 --stage offline
# 检查 offline/selection.json、offline/metrics.jsonl 和视频后继续：
python run_stackcube.py --h5 "$STACKCUBE_H5" --output outputs/stackcube/run01
```

第二条命令会跳过已完成的 offline，再继续后面阶段。可以分别使用 `--stage prepare`、`--stage iterative`、`--stage online`、`--stage test`；前置产物必须存在。
如果离线成功率仍接近零，先查演示质量、动作控制模式和录像，不应把“继续增加在线轮数”当成默认解决办法。如果已经接近 100%，也要注意后续算法提升空间太小。

查看实际底层命令而不执行：

```bash
python run_stackcube.py --h5 "$STACKCUBE_H5" --output outputs/stackcube/run01 --dry-run
```

可调参数示例（使用新目录）：

```bash
python run_stackcube.py --h5 "$STACKCUBE_H5" --output outputs/stackcube/run02 \
  --offline-epochs 100 --rounds 3 --iterative-epochs 30 --online-epochs 100 \
  --val-episodes 100 --test-episodes 100 --video-every 20
```

管线按完成阶段记录断点；迭代阶段中断可从上一个完成轮继续。离线不支持优化器级 resume。在线可以用原有 `train_online.py resume=...` 恢复到一个**新输出目录**；管线不会猜测未完成阶段的最佳恢复方式。在线 epoch checkpoint/last 包含优化器；best 是选出的推理权重，不能用它恢复优化器。

## 4. 统一评测口径

- 所有阶段：StackCube-v1、CPU 物理后端、最多 300 个 primitive steps、1024 点 RGB 点云、动作 chunk=16、执行前缀=2、推理步数=10。
- 验证集：固定种子 2000–2099，只用于模型选择。offline 每 10 epochs；iterative 和 online 每 5 epochs。末轮必评。
- 测试集：固定种子 3000–3099，在所有训练和选择结束后一次性比较三个阶段。
- 迭代采集：10000 起，默认三轮共 300 个不同种子。脚本也会检查数据 JSON 中可得的演示种子，避免与保留集重叠。
- 三阶段统一使用 CPS 选模型；最终额外报告 ODE。CPS 与 ODE 各自横向比较，不相互混用。
- 演示、normalizer、环境配置和 checkpoint 都自动衔接。进入在线阶段时验证 model/env 一致，防止同维度但不同任务的 checkpoint 混入。
- 评估临时保存并恢复 Python/NumPy/PyTorch RNG 和模型训练状态，降低评估频率对训练随机序列的影响。不同硬件/库版本仍可能有数值差异。

| 指标 | 定义 / 用途 |
|---|---|
| Eval/Success_Rate | 一回合内任一 primitive step 的 success 为真即成功；数值 0–1 |
| Eval/Final_Success_Rate | 回合结束时仍报告 success 的比例；识别短暂堆叠后失败 |
| Eval/Success_Count, Eval/Episodes | 实际成功数与总回合数 |
| Eval/Mean_Reward | 同一评估环境累计回报的均值 |
| Eval/CI95_Low, Eval/CI95_High | 成功率的 Wilson 95% 区间 |
| Env/Success_Rate | 在线训练 rollout 的成功率，只用于诊断，不能代替固定种子评估 |

验证集 best 偏高是模型选择的正常结果；最终比较看测试集。100 回合相差 1–2 个成功不应直接认定算法有效。若据测试结果继续调参，该测试集便参与了开发，最终报告应再留一组新测试种子。先用一个训练 seed 验证流程，研究结论还需要多个独立训练 seed。

现有算法的训练奖励保持原状：离线/迭代的数据训练使用 success 稀疏奖励，在线环境使用原有环境默认奖励。**跨阶段统一的是评估协议，训练 loss/reward 不作为同口径的性能指标。**

## 5. 输出文件

所有路径相对于 `outputs/stackcube/run01/`：

| 路径 | 内容 |
|---|---|
| pipeline.json | 完整管线参数，后续续跑不得悄悄更改 |
| logs/offline.log、iterative.log、online.log 等 | 各进程完整控制台输出 |
| offline/config.yaml、iterative/config.yaml、online/config.yaml | 实际训练配置 |
| 各阶段 metrics.jsonl | 一行一个 epoch，评估时含统一 Eval/* 字段；迭代额外含 round |
| 各阶段 checkpoints/best.pth | 对下一阶段/最终测试暴露的统一选中模型路径 |
| 各阶段 checkpoints/dataset_stats.json | 与模型一致的归一化参数 |
| 各阶段 selection.json | 选中模型来源和选择时指标 |
| offline/checkpoints/last.pth、online/checkpoints/last.pth | 最近一次保存的模型；offline 不含恢复训练优化器 |
| iterative/round_000/ 等 | 本轮新轨迹、manifest、训练日志、候选 checkpoint 和评估 |
| 各评估目录 summary.json、episodes.csv | 汇总和每个 seed 的成功、终态成功、回报、实际步数 |
| comparison/summary.csv、summary.json | 三个阶段 × CPS/ODE 的最终总表 |
| comparison/阶段名/eval/test_ep0000_采样器/ | 最终测试的逐回合结果与视频 |

视频默认每 10 epochs 的评估录制前两个固定种子，epoch 0 基线（如有）和末轮也录。录像发生在 chunk wrapper 内部的 primitive step，避免动作块导致跳帧。视频不挑成功案例，文件名标明 seed。路径如：

- `offline/eval/validation_ep0010_cps/videos/`
- `iterative/round_000/eval/validation_ep0010_cps/videos/`
- `online/eval/validation_ep0010_cps/videos/`

`--video-every 0` 关闭训练期录像；最终评估默认仍录两条，可单独运行 `evaluate_checkpoint.py --video-episodes 0`。缺少编码器或渲染失败会明确警告并写 `video_error.txt`，不伪造视频或成功率。

单独重新测试三个已选模型（新输出目录）：

```bash
python evaluate_checkpoint.py \
  --checkpoint outputs/stackcube/run01/offline/checkpoints/best.pth \
               outputs/stackcube/run01/iterative/checkpoints/best.pth \
               outputs/stackcube/run01/online/checkpoints/best.pth \
  --labels offline iterative online \
  --output outputs/stackcube/run01/retest \
  --seed-start 3000 --episodes 100 --samplers cps ode --video-episodes 2
```

## 6. 已验证与限制

CPU 回归测试覆盖：同种子评估、逐回合产物、短暂成功与终态成功区分、录像不改变随机采样、训练状态恢复、Hydra 三阶段配置、在线保存/加载/恢复、真实策略网络前向和 PPO 更新、原始步轨迹与迭代数据校验。
运行命令：

```bash
python -m pytest -q test/test_experiment.py test/test_flow_ppo.py \
  test/test_iterative_data.py test/test_primitive_recording.py
```

当前交付环境没有运行 ManiSkill/渲染/GPU 的完整 StackCube 训练，所以不能预报成功率或声称端到端仿真已通过。请先跑离线阶段观察初始基线。旧 `evaluate.py` 仍保留；这次跨阶段对比统一使用 `evaluate_checkpoint.py`。

参考：[ManiSkill 演示数据](https://maniskill.readthedocs.io/en/latest/user_guide/datasets/demos.html)、[轨迹转换](https://maniskill.readthedocs.io/en/latest/user_guide/datasets/replay.html)。
