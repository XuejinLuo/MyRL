# 官方拖拽界面的人工接管

现在默认使用 ManiSkill 官方 `interactive_panda` 所用的 SAPIEN Viewer / TransformWindow，
并调用官方 `PandaArmMotionPlanningSolver` 规划器。你可以拖动目标机械臂的位置和姿态，按 N 才执行，
不再逐个按 XYZ / 旋转按钮。原按钮界面仍可用 `--ui buttons` 打开。

## 和直接运行官方脚本有什么区别

官方脚本创建新的环境，使用 `pd_joint_pos`，不会加载你的 best 或接管模型运行到一半的现场。
这里直接在当前策略运行的同一个环境接管，保留 panda_wristcam、相机、点云预处理和 frozen normalizer。

执行过程：

1. 模型进入问题状态后，H 暂停模型并选择 `panda_hand`。
2. 鼠标拖动 Transform 的平移/旋转手柄，预览目标机械臂；拖动不推进仿真。
3. N 调用官方 screw planner，**只规划，不调用它的关节位置执行器**。
4. 对规划的关节路径做 FK，得到 TCP 位姿路径；用原 `pd_ee_delta_pose` 闭环跟踪路径。
5. 通过原 primitive recorder 保存真实执行动作与实时点云/state，来源标为 human。

因此记录的是模型能够学习的 EE-delta 动作，无需先采 joint-pos H5 再离线转换。
路径跟踪和官方 joint-pos 执行不会完全相同，也不保证避开所有场景障碍；请用短距离子目标，观察实际接触。
运动受控制器、冻结动作范围、任务 horizon 和跟踪误差影响；规划成功不等于实际抓取成功。

## 启动

先合并本 PR，在你已能运行官方 `interactive_panda` 的同一个 Python 环境里执行。
此模式依赖 SAPIEN 与 ManiSkill motion planning（MPlib），不需要 Tk 图形窗口。
若官方命令能运行，通常已经具备这些依赖；遇到缺失依赖按 ManiSkill 官方 motion planning 安装说明处理。

```bash
python -m tools.collection.human_takeover \
  --ui sapien \
  --checkpoint outputs/StackCube-v1/oc_budget/iterative/checkpoints/best.pth \
  --seed-start 14000 --episodes 10 \
  --output outputs/StackCube-v1/oc_budget/human_drag_session_01 \
  --auto-pause \
  --policy-delay-ms 50
```

默认保存总览 + 基座/腕部 RGB 的同步视频，不需要 `--save-video`。用 `--no-video` 才关闭。
这里选 14000 开始是为了与前一版 12000 采集、文档中 13000 测试分开；也需避开你自己另行使用过的测试种子。
已有输出不覆盖；只试一条可设置 `--episodes 1`，之后采集用新目录和新种子。

## 操作

窗口有官方 Transform 拖拽工具和新增的 MyRL takeover 面板，可用鼠标按钮或快捷键：

| 操作 | 功能 |
|---|---|
| P | 运行模型／交还控制权 |
| H | 人工接管，选择并将目标重置到当前机械臂手部位置 |
| 鼠标拖拽 | 调整目标位置和姿态，只移动预览，不实际执行 |
| N | 规划并执行当前拖拽目标 |
| G | 切换开合夹爪，执行 10 个 primitive steps |
| Space | 暂停并取消剩余规划路径/夹爪动作；在下一 primitive 边界停止 |
| F | 保存当前轨迹，停留在当前场景 |
| C | 保存当前轨迹并开始下一 seed |
| Q / 关闭窗口 | 保存当前有效轨迹并退出 |

鼠标选择了其他对象时，N 会拒绝执行；再按 H 重新选择机械臂手部。
H 会重置拖拽目标，因此先 H，再拖动，再 N。P / H / Space 都会取消未执行的人工路径。
官方 Control 面板可选择相机查看腕部画面；输出视频保留三个视角。
如果使用官方 Control 的 Pause，程序会把它转为采集暂停；建议使用 MyRL 面板或 Space。
按 N 后后台一次只执行一个 primitive step，界面继续响应，随时可以暂停或交还控制权。
规划本身是同步调用；求解期间按键会在求解返回后处理，尚未推进任何仿真步。

空抓恢复示例：

- H 接管，G 打开夹爪。
- 拖目标向上退开，N 执行。
- 在上方对准方块并调整旋转，N 执行。
- 拖目标向下到合适高度，N 执行。
- G 闭合，观察是否抓稳，再拖目标抬升、N 执行。
- 可以继续人工堆叠，也可以 P 交还模型。

不要按 Teleport 改变机器人或方块真实位置。本模式禁用 Teleport，并移除能直接改关节和物理属性的编辑面板。
如果仍检测到没有执行动作就发生的仿真状态变化，当前条会被排除，避免产生不真实的专家标签。

## 跟踪停止与数据隔离

- 目标离当前 TCP 超过 40 cm、规划失败或路径达到 150 个规划点时，不执行；调整为更近的子目标。
- 每次跟踪限制最多 150 个真实控制步，达到任务终止条件会立即停止。
- TCP 持续约 1 秒几乎不移动会中止当前路径；应退开、改变目标，避免持续顶压。
- 到达判据为位置误差 <2 mm、旋转误差 <0.015 rad；这只是路径完成判据，不是任务 success。
- 运行时根据已安装控制器的纯动作→目标位姿计算，检查并转换旋转方向/缩放，不硬编码版本相关符号。
- 目标 ghost、相机线框和坐标轴可能影响渲染。每次真实环境 step、相机/点云采集前移除它们，
  同时恢复实体不透明度；只在面向操作者的 Viewer 渲染时重建。不会把目标虚影当成策略输入。
- `seed_*.targets.jsonl` 额外记录每次 N 的目标、规划方式和路径长度；原 actions/state/events/review 格式不变。
- 自动失败提示仍只是启发式；不会自动指定恢复目标。人工接管成功率不能当成无人干预模型成功率。

## 审核、导入与训练

规则沿用 [HUMAN_CORRECTIONS.md](HUMAN_CORRECTIONS.md)：最终恢复成功、人工接受、同一人工段内完整 chunk 才供 Actor。
失败前缀不因后续人工救成功而成为专家动作，完整有效轨迹仍供 Critic 使用真实 sparse success 奖励。

```bash
python -m tools.prepare_corrections \
  --session outputs/StackCube-v1/oc_budget/human_drag_session_01 \
  --base-manifest outputs/StackCube-v1/oc_budget/iterative_expand600/round_000/manifest.json \
  --stats outputs/StackCube-v1/oc_budget/iterative/checkpoints/dataset_stats.json \
  --output outputs/StackCube-v1/oc_budget/human_drag_replay_01 \
  --review
```

```bash
python train_iterative.py +experiment=oc_budget \
  +iterative_protocol=human_corrections \
  manifest=outputs/StackCube-v1/oc_budget/human_drag_replay_01/manifest.json \
  output=outputs/StackCube-v1/oc_budget/iterative_human_drag
```

若已导入过其他 session，可将 `--base-manifest` 指向之前清洗后的 manifest，以累积数据；重复导入会被拒绝。

## 验证范围

CPU 测试覆盖控制器旋转符号和旋转基坐标、动作限幅、规划 dry-run 与手部/TCP 偏移、规划失败不执行、
逐步跟踪完成/停滞/预算、界面渲染对象隔离、外部状态修改检测、保存后切换下一条。
同时回归既有人工纠正采集、审核和训练测试。开发环境没有可运行的 SAPIEN/ManiSkill 图形仿真，
尚未实机验证拖拽交互、真实控制器跟踪效果或成功率提升；请先试采一条。

官方源码参考：
- [interactive_panda.py](https://github.com/mani-skill/ManiSkill/blob/main/mani_skill/examples/teleoperation/interactive_panda.py)
- [motionplanner.py](https://github.com/mani-skill/ManiSkill/blob/main/mani_skill/examples/motionplanning/panda/motionplanner.py)
- [SAPIEN TransformWindow](https://github.com/haosulab/SAPIEN/blob/master/python/py_package/utils/viewer/transform_window.py)
