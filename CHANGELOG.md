======================================================
时间：2026-09-12 16:50
实验目标：尝试使用 PPO (train_online.py) 微调预训练好的 Flow Policy。
遇到的问题：
- Critic 网络（V_pred）收敛正常，但是 Actor 的 Success_Count 在 Epoch 2 瞬间掉到 0。
- ppo/approx_kl 极小。
根因分析：
- PPO 中的 -MSE 伪造 log_prob 并在 t=0.5 单点更新，彻底破坏了 Flow Matching 的连续 ODE 向量场。
- PPO 传统的 ratio 截断和负 Advantage 丢弃不适合生成式模型。
修改内容：
1. [models/policy.py] - `evaluate_actions`：取消固定 t=0.5，改为随机采样 t，直接返回 MSE。
2. [algos/pg.py] - `update_step`：废弃 PPO 截断，改用 AWR（优势加权回归），使用 exp(beta * adv) 软更新。
3. [configs/train_online.yaml] - 学习率下调至 1e-5，关闭 entropy_coef。
======================================================

======================================================
时间：2026-09-12 18:50
实验目标：解决 Flow Policy 在线 PPO 微调时策略崩溃 (Success Rate 掉 0) 的问题。
遇到的问题：
- 在先前的 AWR 尝试中，模型发生灾难性遗忘。V_pred 收敛正常，但 Actor 被完全破坏。
根因分析：
1. 【指数爆炸】：`exp(3.0 * adv)` 导致部分样本的梯度放大了 400 倍，瞬间冲毁了预训练的连续向量场。
2. 【生成模型的致命缺陷】：Advantage 归一化后，一半的“垃圾探索动作”变成了正优势。对于传统 PPO，负优势会拉低概率；但对于 Flow Matching，即便是乘以极小的负优势权重，模型本质上依然在进行“行为克隆”学习垃圾动作，导致偏离专家流形。
修改内容：
1. [algos/pg.py] - `update_step`：引入正优势过滤 (Positive Advantage Filtering)。通过 `mask = (advantages >= 0.0)` 彻底屏蔽差于预期的动作，仅对产生正向突破的动作进行加权克隆。同时将 beta 下调至 1.0，限制最大权重乘数为 e^2。
2. [configs/train_online.yaml] - `update_epochs` 从 6 降低至 2，防止由于单批次在线数据分布狭窄而引起的模型过拟合 (Overfitting)。
======================================================

======================================================
时间：2026-09-12 (最新修复)
实验目标：解决 Flow Policy 在线微调时发生的灾难性遗忘与 Success Rate 暴跌问题。
遇到的问题：
- 在之前引入正优势过滤后，Actor 仍然发生灾难性遗忘。表现为 `ppo/approx_kl`（实为 Flow MSE）随着 Epoch 数不降反升，且在 Epoch 2 时成功率即刻掉 0。
根因分析：
1. 【噪声过拟合】：在 `train_online.py` 的 PPO 微调数据准备阶段，代码使用了 `fixed_noise = torch.randn_like(flat_actions)`，并在整个 `update_epochs` 循环中**重复使用同一份固定噪声**去拟合动作。
2. 【向量场坍塌】：Flow Matching 的核心是学习从**任意标准高斯噪声** $x_0$ 指向目标动作 $x_1$ 的连续向量场。如果固定了噪声，模型就会死记硬背从这几个特定噪声点到动作的单一映射。当下一轮 Rollout 采样到全新噪声时，模型完全无法泛化，导致输出动作直接崩溃。
修改内容：
1. [train_online.py] - 移除了 `fixed_noise` 和冗余的 `old_log_probs_list` 预计算逻辑（顺便节省了一次全局前向传播，提升近乎一倍的训练速度）。
2. [train_online.py] - 在调用 `trainer.update_step` 时，将 `noise` 和 `old_log_probs` 参数置为 `None`，强制让 `evaluate_actions` 内部每次更新时都采样全新的随机噪声（$x_0$），真正对齐 Flow Matching 的训练范式。
运行预期：
- 模型不会再过拟合于特定的噪声，全局向量场得以保持连续和稳定。
- 动作崩溃问题解决，Reward 和 Success_Count 将会伴随着 Critic 的收敛而稳步上升。
======================================================