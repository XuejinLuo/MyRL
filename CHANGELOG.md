# Changelog

## 2026-09-17 — 离线 worker 段错误与额外开销修复

- 用户日志显示 SAPIEN CUDA initialization error 后 DataLoader worker SIGSEGV；推测为评估后 fork 继承运行时状态，尚无本机 SAPIEN 复现。
- 默认 num_workers=0；启用多 worker 时固定 spawn + persistent_workers，避免反复 fork。spawn 会复制内存轨迹，需留意 CPU 内存。
- CUDA 恢复 pin_memory，并使用 non_blocking 传输；普通 epoch 不再复制/切换整份 EMA 权重，移除 Dataset 的二次 NumPy 拷贝。
- H5 改为每条轨迹批量读取，而非逐帧重复访问；演示导出增加进度条。
- 增加训练/评估/权重文件耗时及样本吞吐；保持 IDQL/BC 配置、训练目标、评估协议不变。
- 更正历史说明：重构前最新 StackCube 总入口的 use_bc_only 实际为 false，并非 true。
- 验证：27 项 CPU 测试通过；1 项 spawn 测试因当前运行环境禁止 Unix socket 而跳过，未复现真实 SAPIEN/GPU 崩溃。

## 2026-09-17 — 配置驱动的独立三阶段重构

基准：`online@424ee29`。

- 根目录仅保留三个独立训练入口和统一评估入口；移除 StackCube 串行总入口，工具与历史蒸馏演示移出主流程。
- `configs/config.yaml` 统一任务、演示路径、1000 条演示上限、阶段衔接、验证种子、5 条录像和最终对比设置；新增 StackCube/PickCube/PullCubeTool 任务配置。
- offline/iterative 共用 Dataset、执行前缀 IDQL 循环和 EMA；H5 读取自动导出 manifest，合并 JSON 写入、点云预处理、模型构建、环境与评估实现。
- 统一 `metrics.jsonl/csv`、`selection.json`、`best/last/epoch_XXXX.pth`、逐 seed 评估输出与 checkpoint 对应关系；迭代续跑清除中断轮次的重复日志。
- 行为变化：offline 改用执行前缀 Bellman 目标与真实 timeout bootstrap，默认 BC actor 初始化；严格标签/T+1 检查、过滤无效点。PPO 优化公式和超参数未修改。
- 26 项 CPU 测试通过，含真实优化器的小模型三阶段集成与最终比较；未执行完整 ManiSkill/GPU 训练。运行见 README，详细迁移见 `docs/REFACTOR.md`。

## 2026-09-16 — StackCube 三阶段可复测流程

基准：`online@0427188a83b78e25bda4a8fe1d4d74ed161230e7`。

- 新增 StackCube 配置、演示准备入口和 `run_stackcube.py`，串联离线、数据转换、迭代、在线、最终测试；保留原 PullCubeTool 默认配置。
- 管线用 BC actor 初始化，再恢复迭代 IDQL，PPO 更新公式与现有超参数不变。
- 三阶段共享固定种子评估器；统一 Eval/*、metrics.jsonl、config.yaml、checkpoints/best.pth、selection.json；记录每个 seed 的结果和 Wilson 区间。
- 验证集选模型；管线关闭迭代中途 test，全部训练结束后用独立测试集对比 offline/iterative/online，CPS/ODE 分开报告。
- 新增少量固定种子的 primitive-step MP4；隔离录像/评估 RNG；训练模型 eval/梯度开关恢复。
- 离线 checkpoint 内嵌配置和 normalizer；在线初始化检查环境/模型；数据转换支持与离线相同的 max-episodes。
- 修正新评估统计的 NumPy 标量序列化，保证 weights_only=True 可读。
- 更新两项历史 PPO 测试的 actor_updates 预期以匹配现有 v2 的 minibatch 更新计数，没有更改 pg.py。
- 22 项 CPU 回归测试通过；未执行完整 ManiSkill StackCube GPU 训练。运行说明见 docs/STACKCUBE.md。
