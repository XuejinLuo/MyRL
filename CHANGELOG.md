# Changelog

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
