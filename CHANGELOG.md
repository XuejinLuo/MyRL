# Changelog

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
