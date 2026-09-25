# StackCube 抓取诊断

原 `evaluate.py` 只输出 episode 统计和可选视频；`inspect_eval_episode --save` 保存观测，
没有逐 primitive step 的完整动作/接触记录。新增命令不训练、不更改 checkpoint，不改
normalizer、CPS 参数、动作裁剪、控制频率、执行前缀或任务终止规则。

在原 ManiSkill 环境、仓库根目录执行：

```bash
python -m tools.diagnostics.trace_grasp \
  --checkpoint outputs/StackCube-v1/oc_budget/iterative/checkpoints/best.pth \
  --seeds 6009 6014 6019 \
  --sampler cps \
  --output outputs/StackCube-v1/oc_budget/grasp_diagnostics_best
```

以上都是默认值，也可以仅运行 `python -m tools.diagnostics.trace_grasp`。
默认 CUDA、保存全部三个视频；`--device cpu` 或 `--no-video` 可覆盖。
只支持当前 StackCube-v1 / Panda / pd_ee_delta_pose。输出目录存在则拒绝覆盖，重跑改
`--output`。用于失败诊断，可重用测试种子，但不能用这些场景选择模型后再当独立测试。

## 文件与时间对齐

- 根目录 `episodes.csv` / `summary.json`：原评估函数的逐 seed 成功率、回报、步数。
- `config.yaml` / `provenance.json`：冻结配置、checkpoint SHA256、软件版本、git commit。
- `run_status.json`：是否完整运行；视频错误会明确列出。
- `seed_6009/` 等目录：
  - `initial_state.json`：reset 后、执行前的状态。
  - `actions.jsonl`：每次策略输出的完整归一化动作 chunk，以及按原 normalizer
    反归一化后的 controller chunk（其中未执行的后缀不是实际执行动作）。
  - `steps.jsonl` / `steps.csv`：每个实际控制步的动作及执行前/后状态。
  - `metadata.json`：控制频率、动作边界、机器人、控制器信息、坐标和单位。
  - `diagnostic_status.json`：该 seed 是否结束、步数、可选字段读取失败原因。
- `videos/episode_000_seed6009.mp4` 等：同步视频，文件名中 seed 是匹配依据。
  视频 reset 为第 0 帧；`primitive_step=k` 对应从第 k 帧到第 k+1 帧。
  秒数按实际 control_freq 计算，不依赖固定 20 Hz。

逐步状态包括 TCP/机器人基座/红绿方块世界位姿、关节位置和速度、Panda 两夹指关节
位置与开口宽度、策略原始 observation state、红块在 TCP 坐标系的位置和相对初始高度、
任务 grasp/on-top/static/success 标志，以及双夹指分别对两块方块的世界坐标接触力。
同时尝试读取控制器内部 target pose/qpos；不同版本不支持时写 null 并记录原因。

动作同时保存：网络归一化输出、normalizer 反归一化后动作、chunk wrapper 裁剪后的
实际输入。**pd_ee_delta_pose 的动作是控制器动作空间数值，不能直接当米或弧度**；
几何位姿用米、四元数顺序为 wxyz。TCP 实际平移单独记为世界坐标米。
控制器内部 target pose 的参考系取决于安装版本，不能直接与世界坐标 TCP 相减。
Panda 开口宽度为最后两个手指关节位置之和。

contact 是控制步结束时的力读数（N），不是该控制步所有物理子步的最大力；可能漏掉
短暂接触。reset 接触字段为 null。字段不可用时为 null，绝不伪装为零接触/未抓取。
没有自动判定“卡住”“空抓”“稳定抓取”，应结合命令、实际状态和持续时间分析。
仿真真值只记录，不送入策略；不额外请求/采样点云。诊断与渲染保留全局 RNG 状态。

## 提供结果

只上传数值记录即可，已有视频无需重复上传：

```bash
zip -r grasp_diagnostics_best.zip \
  outputs/StackCube-v1/oc_budget/grasp_diagnostics_best \
  -i '*.csv' '*.json' '*.jsonl' '*.yaml' '*.txt'
```

需要连本次同步视频一起提供：

```bash
zip -r grasp_diagnostics_best_with_video.zip \
  outputs/StackCube-v1/oc_budget/grasp_diagnostics_best
```

不需要上传 .pth。首次运行后检查 `episodes.csv` 是否与此前相同 seed 的结果一致。
发生错误时目录中的部分 JSONL 和 `run_status.json` 仍有诊断价值。
