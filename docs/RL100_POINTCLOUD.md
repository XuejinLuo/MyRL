# RL-100 对照：先保住输入几何，再训练策略

## 源码依据与结论

本次对照版本：RL-100 `64264d952c1fda9d5096c090ddaa7177757a77ad`，
MyRL `online@a14558400f21f2dd5203d6cf00b5d14537a46de2`。

| 路径 | 源码实际行为 | 对本问题的意义 |
| --- | --- | --- |
| [MetaWorld wrapper](https://github.com/Lei-Kun/RL-100/blob/64264d952c1fda9d5096c090ddaa7177757a77ad/RL-100/rl_100/env/metaworld/metaworld_wrapper.py) | `get_point_cloud` 变换坐标、按工作空间裁剪，再调用 FPS | 先减少背景，再控制空间覆盖 |
| [mjpc_wrapper](https://github.com/Lei-Kun/RL-100/blob/64264d952c1fda9d5096c090ddaa7177757a77ad/RL-100/rl_100/gym_util/mjpc_wrapper.py) | PyTorch3D FPS 只使用 XYZ，按索引取回完整点特征；Adroit 也有任务专用裁剪配置 | RGB 不参与几何距离；不是均匀随机抽 1024 点 |
| [真实演示预处理](https://github.com/Lei-Kun/RL-100/blob/64264d952c1fda9d5096c090ddaa7177757a77ad/tools/teleop_off2off_data/realsense.py) | 标定坐标中的固定 bounding box + `fpsample.bucket_fps_kdtree_sampling` | 实机也先控制工作空间；该路径没有目标分割配额 |
| [3D Flow 配置](https://github.com/Lei-Kun/RL-100/blob/64264d952c1fda9d5096c090ddaa7177757a77ad/RL-100/rl_100/config/rl100_3d_flow.yaml) | `use_pc_color: False`、`encoder_type: dp3` | RL-100 与当前 MyRL 的 RGB + PointNeXt 也有网络差异，不能把结果全部归因于 FPS |

这些路径没有“每个目标至少多少点”的保证，也不是对物体完全不可见问题的解决方案。
不同任务的相机、裁剪和几何尺度不同，不应把 RL-100 实机裁剪数值复制到 ManiSkill。

MyRL 原来的 random 模式按点数比例分配输入预算。假设裁剪后有 N 个点、目标有 m 个点，
随机保留 K 个点，目标保留点数的期望为 `K*m/N`。
例如 N=25000、m=20、K=1024 时，期望仅 0.8192 点；整帧漏掉目标完全可能发生。
网络内部的 FPS 位于这个丢点步骤之后，不能把已经删除的几何恢复出来。

当前仓库已经有 `object_budget`，它直接使用仿真分割 ID 保留目标点，
适合作为验证感知瓶颈的对照，但不是无需外部感知的实机方案。
**FPS 比 random 更重视空间覆盖，但可能比 object_budget 保留更少的目标点。**
小目标贴近台面、遮挡严重或裁剪不合适时，FPS 仍会失败。

## 本次修改

- 默认 `env.sampling.mode=fps`，实验名 `run04_fps`，总点数、RGB、网络、工作空间、训练参数不变。
- H5 加载和 live reset/step 都调用 `data.pointcloud.preprocess_points`。
- 对整个有效、有限、裁剪后的 XYZ 点云执行 CPU QuickFPS；不随机预抽候选池。
  使用固定起点与 `h=5`（少于 32 个候选位置时降低树高），不消耗训练随机数；RGB 用同一索引取回。
  这是与 RL-100 相同的 FPS 原则，不声称与其 CUDA/KDTree 后端逐点一致。
- 重合 XYZ 只保留首次出现的观测（包括其 RGB）；不足 K 个不同位置时，先保留全部位置再重复补齐。
  空点云返回零点和诊断索引 -1。重复点不会凭空提供几何信息。
- 使用固定版本 `fpsample==1.0.2`。缺依赖明确报错，不静默退回 random；无需安装 PyTorch3D。
- 保留 random 和 object_budget。缺少 sampling 字段时仍使用历史 random 行为。
  新配置随 checkpoint 和 manifest 保存，避免训练和评估使用不同协议。

## 运行与对照

在原有 ManiSkill/PyTorch 环境中执行：

```bash
python -m pip install -r requirements-pointcloud.txt
python -m tools.diagnostics.check_point_sampling --max-episodes 10 --frame-step 5
python train_offline.py
```

没有适用的 wheel 时 fpsample 会从源码编译，需要 C++17 编译器；详见
[fpsample 安装说明](https://github.com/leonardodalinky/fpsample#installation)。

诊断对相同 H5 帧输出 before/random/fps/object_budget 的均值、中位数、p05、
零点帧比例、少于 10 点的帧比例及“原先可见但采样后消失”的帧数。
诊断标签来自原始 H5 segmentation，仅用于计数和 object_budget；FPS 不读取它。
默认标签是 StackCube 的 cubeA=18/cubeB=19，须用自己的 H5 核实；其他任务需配置对应标签。
训练本身无需这些标签。

对照实验只切换采样模式和输出名：

```bash
python train_offline.py experiment=run04_random env.sampling.mode=random
python train_offline.py experiment=run04_object_budget env.sampling.mode=object_budget
```

三组保持相同演示、训练轮数和种子。各自从原始 H5 重新生成输入、manifest 并训练离线初始策略。
不需要重新收集现有完整点云 H5，但不能把旧的已下采样 NPZ 再送给 FPS 当作新数据。
不要只给旧权重切换采样器后就把成功率当成新方案的训练效果。

独立评估 FPS 组（测试种子默认 3000–3099）：

```bash
python evaluate.py 'comparison.checkpoints={offline:outputs/StackCube-v1/run04_fps/offline/checkpoints/best.pth}' comparison.output=outputs/StackCube-v1/run04_fps/test_offline
```

另两组替换 experiment 路径，分开调用，以各 checkpoint 保存的输入协议评估。
先用验证集选择权重，再比较固定测试集成功率及区间。若 FPS 仍远低于 object_budget，
说明仍需检查相机可见性、工作空间或可部署的前景感知，不能据此认为增大 PPO 更新即可解决。

## 验证范围

```bash
python -m pytest -q tests/test_fps_sampling.py tests/test_point_sampling.py tests/test_stages.py tests/test_experiment.py tests/test_flow_ppo.py tests/test_iterative_data.py tests/test_primitive_recording.py
```

测试覆盖空间上分离的稀少点保留、XYZ/RGB 对齐、重复位置、空云、小点云、随机状态隔离、
缺失依赖报错、无 segmentation 的 H5/live 一致性、manifest 往返和三种采样器对照。
合成覆盖测试不代表真实任务成功率；开发环境没有用户的真实 H5 或 ManiSkill/SAPIEN，
未执行真实仿真或 GPU 训练。成功率提升需上述本机对照实验确认。

本次结果：55 passed，1 skipped（运行环境禁止多进程 tensor IPC 所需的 Unix socket）。
合成 25000 点到 1024 点的热运行中位数：random 0.96 ms、FPS 20.33 ms（7 次，含去重）。
该数字仅说明此开发 CPU 上的预处理成本，不包含相机、网络或环境步进，也不是控制周期保证。
