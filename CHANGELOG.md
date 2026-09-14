# CHANGELOG

## 2026-09-14 — PPO 更新 v2（待实机/仿真验证）

### 基线与原因
- 基于 XuejinLuo/MyRL 的 online 分支；pg.py 原文件 Git blob：
  `5c7bfd6a63d6025f88dd6683e432399986bed92a`。
- 原实现对生成步 0→9 各执行一次 optimizer.step，后续步使用已改变的参数。
- 本次日志的 95 轮 actor 更新次数均为 10k−1，结合该实现可定位到末步停止；
  末步 std=0.0067 是敏感因素，但不能仅凭此认定所有性能问题的原因。

### 修改
- `algos/pg.py`：每个生成步仍独立计算 PPO ratio、裁剪和 KL。
  每步对 loss/生成步数反传累积梯度；全部通过后裁剪梯度并执行一次 optimizer.step。
  等价于对生成步平均 loss 求梯度，避免同时保留全部计算图。
- 任一步 KL/有限概率比超限：清空当前 minibatch 梯度，停止本轮后续 actor 更新；
  critic 继续。NaN/Inf 仍报错，不回滚此前已接受的更新。
- 新增 `ppo/update_version=2`、`stop_step`（0 起，-1=未停止）、
  `stop_kl`、`max_abs_log_ratio`、`kl_step_0...9`。
  未计算的分步 KL 为 null；stop_kl 仅 KL 停止时有效。
- `actor_updates` 改为实际 minibatch optimizer 次数，不可直接与 v1 比较。
  均值 KL 包含导致停止的检查；分步 KL 是已检查样本的均值。
- `test/test_flow_ppo.py`：调整更新次数断言，新增相同参数检查及末步拒绝后清空梯度测试。

### 本轮固定项
- 不修改采样器、奖励、归一化、编码器冻结、full ratio 或 GAE。
- actor_lr=3e-7、target_kl=0.01、update_epochs=1、batch_size=128、
  steps_per_epoch=2048、critic_warmup_epochs=5 保持原值。
- 不把整条链 log-prob 求和后计算单一 ratio；不通过删除末步/关闭 KL 来消除报警。

### 验证与实验记录
- 已做 Python 语法检查；交付环境缺少 PyTorch，未运行 pytest/ManiSkill，未证明成功率提升。
- 本地运行：`python -m pytest test/test_flow_ppo.py -q`。
- 从原离线 checkpoint 启动，resume=null，先跑 30 轮；使用固定 100 个验证种子。
- 保留旧运行目录；记录代码版本、完整配置、checkpoint、种子及环境交互步数。
- v2 每轮最多 16 次 actor optimizer 更新（2048/128），预热期为 0。
- 观察分步 KL、停止步号、成功率；停止仍频繁时单独对比 actor_lr=1e-7。
- 只有验证数据支持才扩展实验；最终用独立测试种子比较，不能承诺初始成功率门槛。

### 后续变更规则
- 每轮只验证一个主要假设；改动追加记录，不覆盖旧结论。
- 记录：原因 / 文件与参数 / 对照结果 / 保留或回退决定。
- 逐回合评估导出、BC 对照、更多 rollout、噪声调整留待后续，不属于本次修改。
