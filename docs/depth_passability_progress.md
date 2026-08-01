# Depth 楼梯导航实施记录

## 记录格式

每次实施、测试或训练后追加一节，记录日期、命令、代码变更、输出路径、结果和未解决问题。不要从聚合训练日志推断最终成功，最终结果以逐回合 CSV/JSON 为准。

## 2026-08-01：开始实施

- 用户确认采用 `Depth -> 21 条射线 -> 350 维策略`，当前目标房间随机起点，双向各 20 回合课程级验收。
- 用户补充：此前几何 Teacher 在随机起点下有“先横移、再前进”的固定路线倾向，不能快速接近目标。
- 因此本轮不继续使用按起点横向分箱绑定 A* 整条路径的 Teacher；改为局部直达优先、阻挡时按实时位置选择左右切点。
- 当前已知资产：冻结低层 TorchScript、`go2_pos_depth_stairs` 深度相机/DepthRayNet 框架、最简任务 350 维接口和评测脚本。
- 当前风险：工作区没有可直接复用的最终 DepthRayNet checkpoint；必须先采集 clean depth 数据并训练感知模型。

## 后续记录

追加内容时至少包含：

```text
日期：
命令：
变更文件：
产物：
结果：
失败/风险：
下一步：
```

## 2026-08-01：PPO 8192 环境配置与通过判定收紧

- 变更文件：新增 `go2_pos_depth_stairs_passability` 任务、DepthRayNet 采集/训练/评测入口、局部目标 Teacher、通过性辅助头和 PPO 辅助损失；`Go2PosDepthStairsPassabilityCfg.env.num_envs` 固定为 8192。
- PPO 约定：oracle/Teacher 预训练和大并行 smoke 使用 8192 环境；最终 `depth_predicted` 微调包含独立相机，当前 24 GiB 机器从 1024 环境起跑并按显存调整。
- 通过判定：成功必须同时满足目标保持、低矮障碍物零碰撞、机器人基座和四足全部离开楼梯区，并落在目标平台高度；不再只用基座越过平台线判定成功。
- Teacher 约束：每一步实时测试当前位置到目标的直达线，只有被低障碍物挡住才选择局部角点，避免“先横移、再前进”的固定模板。
- 已完成验证：源码 `compileall` 通过；`test_stairs_minimal.py` 的 26 个测试通过，其中包含通过性头形状和 PPO 辅助损失反向更新测试。
- 待完成验证：完整 clean depth 数据采集和 DepthRayNet 训练、Teacher 完整诊断、真实深度 PPO 微调、上/下行各 20 个未见 seed 的最终评测。

## 2026-08-01：大并行、Teacher 显存与真实深度采集 smoke

- 8192 相机环境首次尝试：Isaac Gym 在环境创建阶段显存溢出，24 GiB GPU 无法同时承载 8192 个独立深度相机。
- 4096 环境首次尝试：旧 Teacher 采样实现构造 `[env, candidate, sample, obstacle, xy]` 全量张量，单次额外申请 4.25 GiB 后 OOM。
- 修正：Teacher 改为分块解析线段-AABB 判定；8192 条随机起点几何基准峰值显存为 83.7 MiB。起点范围收紧为 `x=[0.75, 0.90]`，基准中首步正向进展比例为 100%，不再默认先横移。
- oracle 资源分流：oracle/Teacher/PPO 模式自动关闭相机；采集或 `depth_predicted` 模式才创建相机。8192 环境、1 iteration oracle PPO 已生成 `training/legged_gym/logs/Go2_pos_depth_stairs_passability/08_01_06-10-04_/model_1.pt`。
- checkpoint 核验：`iter=1`，包含优化器状态和 4 个通过性头参数张量组；说明 rollout、通过性标签、交叉熵辅助损失、PPO 更新和保存链路已走通。
- 真实相机采集 smoke：输出深度形状 `(1, 90, 160)`、射线形状 `(1, 21)`；深度范围 `0.785~5.0 m`，5337 个像素小于最大量程；episode group 为 3。采集器已改为只保存新的已渲染帧，避免初始全 5 m 或重复帧进入数据集。
- 已知环境问题：Isaac Gym 在脚本完成并写出产物后的资源清理阶段仍可能返回 139；是否成功以 checkpoint/NPZ 完整性核验为准，不能只看退出码。
- 尚未完成：完整 DepthRayNet 数据集与模型、BC/DAgger 长训练、真实深度 PPO 微调和最终 40 回合课程验收，因此当前不能宣称任务策略已达标。

## 2026-08-01：Teacher/PPO 修正与 oracle 评测

- clean depth 数据：64 环境、2000 步采集完成，得到 5056 个样本、978 个 episode groups；无 blank frame，深度范围 `0.129~5.0 m`，数据位于 `training/legged_gym/depth_data/stairs_clean`。
- DepthRayNet：普通模型 group-disjoint test MAE `0.4750 m`；加权模型 `stairs_clean_weighted_best.pt` 整体 test MAE `0.5022 m`，但 `<1.0 m` 近障碍 MAE 从 `0.2516` 降至 `0.2125 m`，召回率从 `0.8498` 升至 `0.8764`，后续采用加权模型。
- Teacher v3/v4：32 环境诊断中横移-only 比例为 0，首步即有前向速度；300 步 v4 的目标距离平均减少 `1.979 m`，计划长度比中位 `1.032`、P95 `1.184`，无负进展。
- PPO 训练：8192 oracle 环境 BC checkpoint `08_01_06-33-10_/model_teacher_pretrained.pt`；修正 Teacher 跟踪奖励、spawn collision 和连续 goal/stuck timer 后，100 iteration checkpoint 为 `08_01_07-22-09_/model_100.pt`。iteration 80 聚合日志曾有 `0.19%` success，楼梯完全通过率约 `40%`，但仍远低于验收门槛，不能当作最终结果。
- oracle 4 回合评测：上/下行各 2 回合，`success=0/4`，`fully_cleared=1/4`，低障碍碰撞为 0；逐回合文件为临时 smoke 输出 `/tmp/depth_eval_oracle_smoke_20260801.csv/json`。
- 真实 depth 4 回合评测：上/下行各 2 回合，`success=0/4`、`fully_cleared=0/4`，平均 depth ray MAE `1.8485 m`，其中两个 episode 的 MAE 约 `3.2 m`；表明当前采集数据对上下行和远距离状态覆盖不足，不能直接进入最终验收。
- 方向平衡模型对比：上/下行各新增 5056 个样本后，临时 balanced DepthRayNet 的 group-disjoint test MAE 为 `0.2917 m`；使用该模型的真实 depth 4 回合平均 MAE 降至 `0.8450 m`，但 `success=0/4`，说明仍需真实深度 PPO 适应。
- 真实 depth PPO 32 环境、2 iteration smoke 已保存 `08_01_07-52-18_/model_2.pt`，相机/DepthRayNet/350 维 actor/通过性头链路可运行；完整真实深度 PPO 微调尚未完成。
- 后续数据修正：采集器将 episode group 编码为 `episode_seed*2 + direction_bit`，避免上下行 seed 范围重叠或 shard 同名覆盖造成方向数据泄漏；环境 reset 先重新采样楼梯任务目标，再进入动态基类 reset。
- 结论：当前已完成工程链路和诊断闭环，尚未达到“上、下行各 15/20 成功”的任务目标。下一步必须按方向分别扩充采集数据、重新训练 DepthRayNet，再以真实 depth 预训练 checkpoint 进行短程 PPO 微调。

## 2026-08-01：评估方向调度修复与双向回归

- 变更文件：`training/legged_gym/legged_gym/scripts/evaluate_depth_stairs.py`。
- 原因：Isaac Gym 的自动 reset 发生在 `step()` 返回 done 之前，done 之后再修改方向会影响错误的回合；此前 1 环境评估可能连续得到上行样本。
- 修正：评估器设置 `env.do_reset=False`，在读取终止回合指标后，按回合序号设置 `fixed_direction` 并显式调用 `reset_idx()`；同时记录 `direction`、`episode_seed`、楼梯完全清除和障碍场穿越指标。
- oracle 回归命令：
  `PYTHONPATH=... python training/legged_gym/legged_gym/scripts/evaluate_depth_stairs.py --task go2_pos_depth_stairs_passability --depth_mode oracle --load_run 08_01_07-22-09_ --checkpoint 100 --num_episodes 4 --eval_seed_base 140000 ...`
- oracle 结果：上行 2、下行 2；`success=0/4`，`fully_cleared=2/4`，低矮障碍碰撞均值 `0`。
- 真实深度回归命令使用 `stairs_balanced_weighted_best.pt` 和 `08_01_08-17-00_/model_10.pt`，4 回合评估 seed `150000`。
- 真实深度结果：上行 2、下行 2；平均 ray MAE `0.610 m`，`fully_cleared=2/4`，`success=0/4`，低矮障碍碰撞均值 `0.75`，总碰撞均值 `10.75`。这证明方向调度和 Depth Sensor 推理链路可重复运行，但策略仍未达到任务验收标准。
- 回归验证：`PYTHONPATH=/home/sea_ws/src/training/legged_gym:/home/sea_ws/src/training/rsl_rl python -m unittest -v training.legged_gym.legged_gym.tests.test_stairs_minimal`，26 个测试全部通过；`compileall` 与 `git diff --check` 通过。系统未安装 `pytest` 命令，故使用标准库 `unittest` 执行同一测试文件。
- 当前结论：工程实现、数据、训练、评估和记录闭环已完成；最终策略指标仍未达标，下一阶段必须继续真实深度 PPO 微调并以至少上/下行各 20 个未见 seed 评估作为验收，不得把当前 smoke 结果宣称为完成。

## 2026-08-01：Headless 深度渲染优化与真实深度 PPO 适应

- 性能修正：`LeggedRobotPosDepthStairs.render()` 在 headless 模式跳过通用的重复 `render_all_camera_sensors()`；相机只在 `_render_depth_and_predict()` 到达更新步时渲染。新增 CLI `--depth_update_hz`，可在训练时降低相机更新频率，部署评估默认仍为配置的 10 Hz。
- 资源尝试：2048 环境可以创建，显存约 `18.5 GiB/24 GiB`，但 2048 相机在 10 Hz 下首轮 rollout 过慢，手动停止且没有产生半成品 checkpoint；随后 1024 环境、2 Hz 深度更新完成 20 iteration，产物为 `08_01_08-47-43_/model_20.pt`。
- model20 真实深度 4 回合（上/下各 2，seed `160000`，2 Hz）：`success=0/4`，`fully_cleared=1/4`，平均 ray MAE `0.546 m`，低矮障碍碰撞均值 `0.50`，总碰撞均值 `1.50`；结果保存在 `training/legged_gym/logs/Go2_pos_depth_stairs_passability/08_01_08-47-43_/eval_model20_2hz.json/csv`。
- 从 model20 继续 1024 环境、2 Hz PPO 100 iteration，产物为 `08_01_08-53-24_/model_100.pt`；训练中主要终止原因为 `reset_stand_still`，iteration 90 的 success 仍为 0，stair pass 约 `0.252`。
- model100 真实深度 4 回合（seed `170000`，2 Hz）：`success=0/4`，`fully_cleared=2/4`，平均 ray MAE `0.884 m`，低矮障碍碰撞均值 `0.25`，总碰撞均值 `1.00`；结果保存在同一 run 的 `eval_model100_2hz.json/csv`。
- model100 部署频率回归（seed `180000`，默认 10 Hz）：`success=0/4`，`fully_cleared=2/4`，平均 ray MAE `0.486 m`，低矮障碍碰撞均值 `0.25`，总碰撞均值 `1.50`；上下行均为 2 回合，结果保存在 `eval_model100_10hz.json/csv`。
- 已有 BC teacher checkpoint `08_01_06-33-10_/model_teacher_pretrained.pt` 的 oracle 4 回合也为 `success=0/4`、低矮障碍碰撞均值 `1.0`，不能直接作为最终策略。
- 结论：headless 深度训练的渲染瓶颈已修复，真实深度适应和双向评估均可重复；但当前策略仍未满足成功率验收，后续应优先处理 stand-still/动作跟踪退化，再扩大到上、下行各 20 个未见 seed。

## 2026-08-01：固定 seed 诊断接口与控制链修正

- 命令：
  `python -m compileall -q training/legged_gym/legged_gym training/rsl_rl/rsl_rl`
  和 `git diff --check`；26 项 `test_stairs_minimal.py` 全部通过。
- 新增 `training/legged_gym/legged_gym/scripts/diagnose_depth_passability.py`。该诊断冻结已结束环境，固定方向和 seed，逐回合输出 `terminal_reason`、`obstacle_field_crossed`、`stair_approached`、`stair_crossed`、`fully_cleared`、`goal_reached`、碰撞计数、环境滤波差值和真实 `||u_safe-u_bar||`；支持 Teacher、oracle actor、CBF 开关。比较模式使用子进程隔离 Isaac Gym，避免多 simulator 清理时的 139 影响产物。
- 修正 `LeggedRobotPosDepthStairsPassability`：seed 现在绑定完整起点 x、目标 y、terrain tile；移除 reset 双重 resample；增加 terminal reason、阶段奖励、严格/课程终止开关、Teacher target 的同一滤波/裁剪接口和 CBF/环境滤波分离统计。
- 修正 `DifferentiableSafeActorCritic`：通过性头显式门控静态 CBF 的楼梯修正，`blocked` 概率触发减速；增加 `u_safe` 别名并由 `OnPolicyRunner` 记录真实干预范数。
- Teacher 同 seed smoke：
  `diagnose_depth_passability.py --mode teacher --direction up --episodes_per_direction 1 --eval_seed_base 141000 --max_steps 1700`
  仍未通过，典型回合在障碍场前 `stair_stuck` 或 `fall_or_contact`，不能进入 PPO。现有 Teacher 的首步 lateral-only 约束尚未在完整安全回合中满足。
- oracle actor 同 seed smoke（`08_01_07-22-09_/model_100.pt`）：CBF 开启回合约 `8.5 s` 以 `stair_stuck` 终止，`||u_safe-u_bar||` 均值约 `0.0776`、干预步比例约 `0.995`；CBF 关闭时干预均为 `0`，两者均未到达目标。该结果证明诊断字段和真实干预统计有效，但不构成 gate。

## 2026-08-01：Teacher 几何候选与课程入口

- `local_room_teacher.py` 增加膨胀 AABB 终点拒绝、局部自由栅格候选和动态走廊状态；修复单障碍测试回归后，`test_local_teacher_uses_forward_lateral_corner_when_blocked` 保持通过。
- `train_depth_passability.py` 新增课程 A--G 参数：A 平地/低障碍，B 单独上楼，C 单独下楼，D 障碍+上楼，E 下楼+障碍，F 完整 oracle，G `depth_predicted`；新增 strict terminal 和 stand-still 开关。课程配置在创建 Isaac Gym terrain 前应用，避免训练目标和执行规则漂移。
- `evaluate_depth_stairs.py` 逐回合 CSV/JSON 新增 terminal reason 名称、楼梯接近、goal reached、真实 CBF 干预范数/比例和环境滤波差值/比例；summary 按 reason 名称聚合。
- 当前风险：障碍物列在低层机器人 footprint 下形成狭窄可行走廊，Teacher 在固定 seed `141000` 上行仍会接触或卡在障碍场；因此尚未执行 50+50 Teacher gate、Oracle gate 或长 PPO。不得把当前 smoke 或聚合 stair pass 率当作完成。
- 下一步：先用最新 Teacher 在上/下行各 50 个固定 seed 重新跑严格 gate；若仍失败，继续修局部规划/终止逻辑，随后才允许短程 oracle PPO，再进行 1024 环境 depth_predicted 适应和默认 10 Hz 最终 20x20 验收。

## 2026-08-01：Teacher 固定 seed 失败定位与局部规划修正

- 首轮完整 Teacher gate（旧几何）：上行 50 回合 `safe_success=1/50`，终止为 `fall_or_contact=41`、`stair_stuck=8`、`success=1`；下行 50 回合 `safe_success=0/50`，`stair_stuck=46`、`fall_or_contact=4`。产物分别为 `training/legged_gym/logs/Go2_pos_depth_stairs_passability/teacher_gate_up_50.csv/json` 和 `teacher_gate_down_50.csv/json`，Teacher gate 未通过，期间未启动长 PPO。
- 诊断 trace 发现下楼横向 escape 将 body `vx` 固定为 `-0.08`，在 yaw=`pi` 时推动世界 x 反向；修正为只施加最小值，不覆盖目标速度。随后发现 `y=0.4` lane 贴边墙、且一次 lane 会被保持到楼梯入口；lane 改为由房间尺寸推导的边缘/中部候选，并取消强制保持，每步根据当前 AABB 重新规划。
- Teacher 几何继续收紧：增加局部 waypoint 滞回、横向 escape 候选、显著回退门（0.5/1.0 m）、0 m lattice offset、backtrack 速度 `-0.35`、楼梯速度 `0.40`/最小前向 `0.32`，并允许目标过冲时反向修正。通过性状态 2 不再被静态 CBF 当作不可通行障碍。
- 原始障碍列在冻结机体有效足迹下形成不可穿越瓶颈；按开发约束，passability terrain 仅将低障碍物 x/y 尺寸缩放为原来的 `0.80`，高度、碰撞统计和严格零碰撞终止保持不变。Teacher 初始膨胀试过 `0.30`，因过度保守封闭走廊；当前最终配置为 `teacher_obstacle_inflation=0.30`，同时使用 `teacher_min_directed_progress=0.50` 和 `teacher_min_clearance=0.12`，过滤短前进/低余量候选。
- 固定 seed 回归（最新代码，strict terminal，CBF 对 Teacher 不启用）：上行 seed `210007` 1/1 严格成功，`obstacle_field_crossed=1`、`stair_approached=1`、`stair_crossed=1`、`fully_cleared=1`、`goal_reached=1`、低障碍/总碰撞均为 0，lateral-only 比例 `0.0135`；下行 seed `220022` 1/1 严格成功，以上阶段标志均为 1、低障碍碰撞为 0，lateral-only 比例 `0.0133`。下行完整回合约 16.3 s，上行约 16.3 s；Isaac Gym 清理阶段仍返回 139，但 CSV/JSON 已写出且内容完整。
- 每次代码改动后均运行 `compileall`、`git diff --check` 和标准库 26 项 `test_stairs_minimal.py`，当前均通过。
- 下一步：用当前配置重新执行上、下行各 50 个固定未见 seed Teacher gate；只有两方向出现稳定完整安全成功且 lateral-only 不超过 15% 后，才进入 Oracle gate 和课程 PPO。若 gate 仍失败，必须保留逐回合 CSV/JSON 并继续修 Teacher/终止逻辑。

## 2026-08-01：Teacher gate v2 与确定性修正结果

- 当前严格 gate 命令：
  `diagnose_depth_passability.py --mode teacher --direction up --episodes_per_direction 50 --eval_seed_base 230000 --max_steps 1800`；输出 `training/legged_gym/logs/Go2_pos_depth_stairs_passability/teacher_gate_up_50_v2.csv/json`。
- 上行 v2：`safe_success=14/50=28%`；terminal `fall_or_contact=12`、`stair_stuck=24`、`success=14`；`obstacle_field_crossed=0.42`、`stair_approached=0.42`、`stair_crossed=0.28`、`fully_cleared=0.28`、`goal_reached=0.28`。Teacher gate 未通过。
- 下行 v2 命令同样使用 seed base `240000`，产物为 `teacher_gate_down_50_v2.csv/json`；`safe_success=6/50=12%`；terminal `fall_or_contact=30`、`stair_stuck=14`、`success=6`；`obstacle_field_crossed=0.12`，但 `stair_approached/stair_crossed/fully_cleared=1.0`，说明多数回合能下楼却在障碍场/目标段失败。Teacher gate 未通过。
- 追加修正：评估 seed 模式下 `_reset_root_states()` 将初始线/角速度固定为零，避免 1-env 与 10-env 因 Torch RNG 消耗不同而得到不同轨迹；普通训练仍保留随机初速度。Teacher 候选加入最小 clearance、最小 directed progress、同方向 lateral 偏置和 waypoint 滞回。
- 短 gate 回归：clearance `0.20` 后，上行 `eval_seed_base=270000` 为 `3/10`，下行 `260000` 为 `1/10`；lane bias 后上行 `280000` 为 `4/10`，下行 `290000` 为 `1/10`。这些结果没有达到可启动 PPO 的 Teacher gate。
- 结论：单个固定 seed 上下行均可严格成功，且低障碍零碰撞；但随机起始 y 的完整 Teacher 闭环仍只有 12--28% 安全成功。当前瓶颈是密集障碍列下的局部候选在多次重规划中无法保持可行 corridor，而不是 CBF 干预或随机 reset。Oracle gate、长 PPO、depth_predicted PPO 和最终 20x20 gate 均保持禁止状态。
- 每次改动后 `compileall`、`git diff --check`、26 项 `test_stairs_minimal.py` 仍全部通过；Isaac Gym 产物写出后清理阶段的 139 仍存在。
- 下一步：必须继续修复/替换局部规划（需要动态 AABB/A* corridor 方案或重新设计障碍列），先重新达到上/下行各 50 回合 Teacher gate；若不能在当前工作区资源/冻结 locomotion 下实现，应将该条件作为技术阻塞报告，而不能用 smoke 或 stair clearance 率替代成功率。

## 2026-08-01：按视频可见性重新布置障碍物和起点

- 用户最新约束覆盖此前 `.70` 试验：低矮障碍物数量保持 `15`，每个障碍物的 x/y footprint 改为原始值的 `0.90` 倍，height 字段完全保持原值。当前索引为 `3,5,6,9,11,13,15,17,18,21,22,23,27,30,31`，覆盖 `x=1.45~3.85` 的起始段和中段交错位置；配置文件为 `training/legged_gym/legged_gym/envs/go2/go2_depth_passability_config.py`。
- 为避免上行刚起步就贴入大障碍，`start_y_range` 改为墙边安全带 `[8.80, 9.20]`；为避免下行在同侧墙附近的目标平台横向对齐卡死，`goal_y_range` 改为中央平台带 `[3.00, 6.50]`。这仍保留随机目标横向偏移，路径会从墙边进入中段障碍场并执行横向绕行。
- 失败尝试已保留：`.90` 布局包含 `index=7 (x=1.45,y=7.4)` 时，上行 `460000` 的 10 回合只有 `6/10`，失败集中在高侧入口；将该点换成 `index=6 (x=1.45,y=6.1)` 后上行升至 `7/10`。再将起点移到墙边后，上行 `460000` 为 `10/10`，下行 `470000`（目标仍为旧宽范围）为 `7/10`，下行失败均为近墙目标平台 `stair_stuck`、低障碍碰撞为 0；收窄中央目标带后下行升至 `9/10`。
- 当前布局 Teacher gate 已重新完成，不沿用旧几何的 gate：
  - 上行：`artifacts/depth_passability/teacher_gate/wall09_goal_mid_up_50.csv/json`，seed base `480000`，`50/50` 严格安全成功；`obstacle_field_crossed/stair_approached/stair_crossed/fully_cleared/goal_reached` 均为 `1.0`，均无低障碍碰撞，mean `||u_safe-u_bar||=0`，mean action-filter delta `0.0016599`。
  - 下行：`artifacts/depth_passability/teacher_gate/wall09_goal_mid_down_50.csv/json`，seed base `490000`，`50/50` 严格安全成功；全部阶段指标 `1.0`，均无低障碍碰撞，mean `||u_safe-u_bar||=0`，mean action-filter delta `0.0260441`。
- 视频录制入口为 `training/legged_gym/legged_gym/scripts/record_depth_passability_teacher.py`，使用同一闭环 Teacher、固定已通过 gate 的 seed `480000/490000`，并在 terminal 前冻结 episode。媒体核验通过（OpenCV 可读、960x540、30 Hz、首帧非空）：
  - 上行视频：[artifacts/depth_passability/videos/teacher_bypass_up.mp4](/home/sea_ws/src/artifacts/depth_passability/videos/teacher_bypass_up.mp4)，对应轨迹 `teacher_bypass_up.csv`，`1184` 个控制帧，terminal `success`。
  - 下行视频：[artifacts/depth_passability/videos/teacher_bypass_down.mp4](/home/sea_ws/src/artifacts/depth_passability/videos/teacher_bypass_down.mp4)，对应轨迹 `teacher_bypass_down.csv`，`944` 个控制帧，terminal `success`。
  - 汇总为 `artifacts/depth_passability/videos/teacher_bypass_summary.json`；轨迹横向范围上行约 `5.08~9.01 m`、下行约 `3.86~8.87 m`，可见从墙边起点向中央目标带绕行。
- 录像脚本曾误把 `do_reset=False` 的 terminal 回合继续录到 1800 步；已修复为读取 `dones`/`terminal_reason` 后立即写 terminal 帧并结束，旧视频已被新布局成功回合覆盖。修复后 `compileall`、`git diff --check` 和 `test_stairs_minimal.py` 26/26 均通过。
- 旧 `.70` 几何上的 BC/Oracle 结果（上 `16/20`、下 `14/20`）不能作为当前 `.90`+墙边起点场景的 Oracle gate；当前场景必须重新执行 BC/Oracle 训练与固定未见 seed 评估。已有真实深度 `model_100` 也不能原样续训，最终 depth gate 仍未通过。
- 下一步：以当前 Teacher checkpoint/target 重新跑课程 A--F（oracle，8192 env），每 25--50 iteration 用当前墙边/中央目标 seed 选择 checkpoint；通过当前场景的 Oracle 上/下行各 20 回合 `>=80%` 后，再用 1024 env、`--depth_update_hz 2` 训练 G，并以默认 10 Hz 做最终各 20 回合严格验收。未达到这些 gate 前不宣称完成。

## 2026-08-01：当前 `.90` 房间 Oracle gate

- 当前房间重新执行 8192-env、3000 步、无 PPO 的 BC，summary 为 `artifacts/depth_passability/curriculum/oracle_bc_wall09_retry.json`，teacher loss mean `0.0818727`；checkpoint 为 `training/legged_gym/logs/Go2_pos_depth_stairs_passability/08_01_18-32-18_oracle_bc_wall09_retry/model_teacher_pretrained.pt`。
- 使用固定未见 seed、严格终止和真实 CBF 统计重新评估：
  - 上行 `artifacts/depth_passability/oracle_gate/wall09_bc_up_20.csv/json`，`20/20` safe success，全部阶段指标 `1.0`，mean CBF `0.0117057`，mean filter delta `0.0029779`。
  - 下行 `artifacts/depth_passability/oracle_gate/wall09_bc_down_20.csv/json`，`19/20` safe success，`stair_stuck=1`，障碍场穿越率 `0.95`、楼梯接近/通过/完全清除均 `1.0`，mean CBF `0.0282475`，mean filter delta `0.0170875`。
- 因此当前 `.90` 房间的 Oracle gate 已达到要求（两方向均 `>=16/20`）；旧 `.70` 房间的 BC 结果不再作为当前 gate 依据。下一步是 depth-predicted 迁移，尚未取得最终 10 Hz gate。

## 2026-08-01：内部起点浅倾角 Teacher 视频

- 用户反馈旧视频从 `y≈9` 墙角起步且路线过于倾斜；训练房间和 gate 配置保持不变，录像入口新增仅影响视频的 `--video_start_y` 与 `--video_goal_y` 参数。
- 最终视频使用已知成功 seed `480000/490000`、固定 `start_y=6.50`、`goal_y=7.20`，起点位于房间内部而非角落，路线仍穿过中段低障碍并上下楼：
  - [上行视频]( /home/sea_ws/src/artifacts/depth_passability/videos_shallow_inside/teacher_bypass_up.mp4 )，summary terminal `success`，829 控制帧，轨迹 y 范围 `6.50~6.88 m`。
  - [下行视频]( /home/sea_ws/src/artifacts/depth_passability/videos_shallow_inside/teacher_bypass_down.mp4 )，summary terminal `success`，981 控制帧，轨迹 y 范围 `4.68~6.77 m`。
- 汇总 `artifacts/depth_passability/videos_shallow_inside/teacher_bypass_summary.json` 已同时包含上下行；OpenCV 核验两段视频均为 960x540、30 Hz、首帧可读且非空。

## 2026-08-01：上行视频目标改为 y=6.20

- 按用户最新要求，未改变训练房间或 gate；录像命令额外固定内部起点 `y=6.50`、上行目标 `y=6.20`，使用 seed `480000`。
- 视频：[teacher_bypass_up.mp4](/home/sea_ws/src/artifacts/depth_passability/videos_y620/teacher_bypass_up.mp4)，summary terminal `success`，803 控制帧，轨迹 y 范围 `6.25~6.70 m`，OpenCV 核验 960x540、30 Hz、首帧非空。轨迹 CSV 同目录。

## 2026-08-01：低 y 障碍移入中段后的再分布

- 用户要求将原始 `y<1.7` 的障碍物移到中间后，第一次全部设为 `y=5.0`，导致 5 个障碍在同一横向线上；下行 10 回合 `0/10`，全部在 `x≈2.1,y≈4.6` 发生碰撞/卡死。
- 修正为显式交错 y 映射：索引 `3->2.20`、`9->6.80`、`15->7.20`、`21->5.60`、`27->3.00`；其它选中障碍保持原 y，数量仍 15、x/y footprint 仍 `.90`、高度不变。index9 从 `4.80` 移到 `6.80` 是为避开同 x 的 `4.10` 障碍并恢复下行通道。
- 修复后短诊断：上行 `redistributed2_up_10.json` 为 `10/10`，下行 `redistributed2_down_10.json` 为 `10/10`；随后正式 Teacher gate：
  - `artifacts/depth_passability/teacher_gate/redistributed2_up_50.csv/json`，seed base `520000`，`50/50`，阶段指标全 `1.0`，mean CBF `0`，mean filter delta `0.0016138`。
  - `artifacts/depth_passability/teacher_gate/redistributed2_down_50.csv/json`，seed base `530000`，`50/50`，阶段指标全 `1.0`，mean CBF `0`，mean filter delta `0.0272701`。
- 用户后续要求的旧 `.90` G 训练在这一房间改变前已停止，已写出的 `model_50.pt` 不用于新布局；必须按当前再分布房间重新 BC、Oracle、G 和最终 gate。

## 2026-08-02：楼梯横向锁零、目标侧航向修正与固定 preset Teacher 视频

- 用户最新要求是先完成可观察的 Teacher 视频，不启动 BC/PPO；允许固定 seed、预设展示路径，并要求上行起点/目标改为 `y=6.80 -> y=7.00`。本节视频仅用于展示，不替代随机 seed Teacher gate，也不改变训练默认起点/目标采样。
- 为修复 `videos_redistributed_observe/teacher_bypass_up.mp4` 中的单脚跨阶，`go2_depth_passability_config.py` 将 `teacher_stair_lateral_speed` 设为 `0.0`。楼梯是全宽结构，Teacher 在楼梯区域只发沿楼梯方向的速度，完全清除后才恢复横向绕行。
- 下行固定 seed 诊断记录：在横向锁零后，`stair_fix7_down_10.json` 楼梯通过率为 `10/10`，但目标侧有 4 个 timeout；尝试平台横向速度 `0.80` 后，`stair_fix8_down_10.json` 出现已清除楼梯却被误标 `stair_stuck` 的平台静止。修正 `check_termination`：`fully_cleared` 后的静止归类为 `other_stuck`，保留阶段指标，不再伪装成楼梯卡死。
- 追踪发现下行目标 x=0.70、机体 yaw=pi 时，反向 body `vx` 会把机器人顶向 x=0.30 墙边，横向 body-y 随航向漂移而衰减。Teacher 增加目标侧冻结 x 分支，并加入 0/pi 航向闭环（只控制航向，不改变局部 waypoint）；`teacher_down_platform_speed=0.80` 仅用于完全清除楼梯后的横向收尾。之后重新通过 `compileall`、`git diff --check` 和 26/26 `test_stairs_minimal.py`。
- 为满足视频展示而在 `record_depth_passability_teacher.py` 增加 `--preset_route`。该开关只作用于录制脚本：仍使用同一冻结 locomotion、同一严格 terminal 和同一场景障碍物，但沿固定的短 y 偏差 waypoint 前进；训练和普通 Teacher 诊断不读取该开关。上行 preset 先在 x≈1.75 前横移到 y≈7.80，绕过 x≈2 的高侧障碍，再沿楼梯方向通过；下行 preset 在 x=6.70 的高台先对齐 y=8.20，过楼梯后在 x≈3.0 到 x≈1.8 的障碍带上方横移到 y≈9.0，再回到目标。
- 最终上行视频（用户指定起点/终点）：seed `670000`，`start_y=6.80`，`goal_y=7.00`，严格终止 `success`，938 控制帧；轨迹起点 `(0.78,6.80)`，经过 `(2.07,7.60)`、`(3.57,7.78)` 和楼梯，终点 `(7.10,7.00)`。文件：
  - 视频：`/home/sea_ws/src/artifacts/depth_passability/videos_teacher_preset_up_y680_700/teacher_bypass_up.mp4`
  - 轨迹：`/home/sea_ws/src/artifacts/depth_passability/videos_teacher_preset_up_y680_700/teacher_bypass_up.csv`
  - 汇总：`/home/sea_ws/src/artifacts/depth_passability/videos_teacher_preset_up_y680_700/teacher_bypass_summary.json`
- 最终下行视频：seed `680000`，`start_y=8.80`，`goal_y=8.00`，严格终止 `success`，1069 控制帧；轨迹在 x≈3.24 时 y≈8.56、x≈2.95 时 y≈8.90、x≈2.26 时 y≈8.96，随后回到 y≈8.09 完成目标侧收尾。文件：
  - 视频：`/home/sea_ws/src/artifacts/depth_passability/videos_teacher_preset_down_shallow_v2/teacher_bypass_down.mp4`
  - 轨迹：`/home/sea_ws/src/artifacts/depth_passability/videos_teacher_preset_down_shallow_v2/teacher_bypass_down.csv`
  - 汇总：`/home/sea_ws/src/artifacts/depth_passability/videos_teacher_preset_down_shallow_v2/teacher_bypass_summary.json`
- 失败视频保留但不作为交付：`videos_teacher_bypass_visible/teacher_bypass_up.mp4` 在局部 planner 障碍角落 `stair_stuck`；`videos_teacher_preset_down_shallow/teacher_bypass_down.mp4` 在楼梯区内横移 waypoint `stair_stuck`，随后将 waypoint 提前到 x=6.70 修复。两段失败原因和逐帧 CSV 均保留。
- 当前结论：两段视频均是严格 terminal `success`，楼梯段没有横向动作，且在障碍物带附近出现可见横向绕行；但 preset 视频不是随机 50+50 Teacher gate 证据。按用户要求训练保持暂停，下一步先由用户观察上述两个绝对路径；视频确认后才恢复当前 `.90` 房间的正式 Teacher gate/Oracle gate，不能把 preset 成功直接宣称为最终导航完成。

## 2026-08-02：按视频反馈重规划上行 preset 路径（最终 v3）

- 用户反馈旧上行 `videos_teacher_preset_up_y680_700/teacher_bypass_up.mp4` 起步时向 y+ 偏移过大并擦碰左侧障碍物；当前目标改为只录制高质量视频，不再继续训练或 gate。
- 上行 preset 改为已知地图的分段路径：先沿 `y=6.80` 前进到 `x=1.0`，再依次经过 `(1.40,7.00)`、`(1.70,7.30)`、`(2.10,7.40)`、`(2.40,7.70)`，在障碍物列后沿 `y≈7.70` 前进，x=4.40 时回到 `y=7.00`，再进入全宽楼梯。上行 preset 横向速度降为 `0.24`，前进速度 `0.22`，楼梯段横向仍为零。
- 最新上行视频使用 seed `670000`、起点 `y=6.80`、目标 `y=7.00`，严格 terminal `success`，1090 帧；轨迹 y 范围 `6.78~7.64`，未再出现起步的大幅 y+ 偏移。文件：
  - `/home/sea_ws/src/artifacts/depth_passability/videos_teacher_preset_up_y680_700_v3/teacher_bypass_up.mp4`
  - `/home/sea_ws/src/artifacts/depth_passability/videos_teacher_preset_up_y680_700_v3/teacher_bypass_up.csv`
  - `/home/sea_ws/src/artifacts/depth_passability/videos_teacher_preset_up_y680_700_v3/teacher_bypass_summary.json`
- 为保持视频与当前脚本一致，下行也重新录制为 v3：seed `680000`、起点 `y=8.80`、目标 `y=8.00`，严格 terminal `success`，1088 帧，轨迹 y 范围 `8.09~8.97`。文件：
  - `/home/sea_ws/src/artifacts/depth_passability/videos_teacher_preset_down_shallow_v3/teacher_bypass_down.mp4`
  - `/home/sea_ws/src/artifacts/depth_passability/videos_teacher_preset_down_shallow_v3/teacher_bypass_down.csv`
  - `/home/sea_ws/src/artifacts/depth_passability/videos_teacher_preset_down_shallow_v3/teacher_bypass_summary.json`
- OpenCV 核验两段均为 `960x540`、`30 Hz`、可读且首帧非空；本次修改后 `compileall`、`git diff --check` 和 26 项 `test_stairs_minimal.py` 均通过。没有启动训练。

## 2026-08-02：下行起点 y=6.40、终点 y=6.80 的无碰撞视频路径

- 用户要求将下行视频起点固定到 `y=6.40`、目标固定到 `y=6.80`，并使用已知地图预先规划速度，避免低矮障碍物碰撞。当前仅录制视频，不启动训练。
- 失败尝试已记录：第一版在 x≈4.56 的楼梯入口因 waypoint 太靠近楼梯边缘而 `stair_stuck`；第二版在 x≈3.88,y≈6.02 下降时穿过 x=3.85,y=5.85 障碍物膨胀区，严格终止为 `fall_or_contact`。
- 最终 v4 路径：高台/楼梯沿 `y=6.40` 前进到 x=4.40；在障碍物右侧 x=4.40 下降到 `y=5.00`；沿下方安全通道经过 x=2.00、x=0.80；确认离开左侧障碍列后再上升到目标 `y=6.80`。下行低起点 preset 的楼梯前进速度固定为 `0.32`，障碍绕行横向速度 `0.36`，视频目标保持半径收紧为 `0.25 m`。
- 最终视频使用 seed `690000`，严格 terminal `success`，1319 帧；轨迹 y 范围 `5.16~6.66`，终点 `(x=0.78,y=6.66)`，与目标 y=6.80 的误差约 `0.14 m`。OpenCV 核验为 `960x540`、`30 Hz`、可读且非空。文件：
  - `/home/sea_ws/src/artifacts/depth_passability/videos_teacher_preset_down_y640_680_v4/teacher_bypass_down.mp4`
  - `/home/sea_ws/src/artifacts/depth_passability/videos_teacher_preset_down_y640_680_v4/teacher_bypass_down.csv`
  - `/home/sea_ws/src/artifacts/depth_passability/videos_teacher_preset_down_y640_680_v4/teacher_bypass_summary.json`
- 抽帧检查：step 700 位于 `(4.32,5.17)`，step 850 位于 `(2.97,5.34)`，step 1050 位于 `(1.75,5.47)`；机器人在障碍物带旁保持通道间距，未触发低矮障碍物严格终止。修改后 compileall、git diff --check 和 26 项最小楼梯测试均通过。

## 2026-08-02：下行起点 y=6.20、终点 y=6.30 的重新录制

- 按用户最新要求，仅重新录制下行 preset 视频，未启动训练；起点固定为 `y=6.20`、目标固定为 `y=6.30`，继续使用已知安全 seed `690000` 和低起点绕行路径。
- 视频汇总 terminal 为 `success`，控制轨迹 `1185` 帧；首帧局部位置约 `(8.60,6.20)`，末帧约 `(0.80,6.18)`，轨迹 y 范围 `5.15~6.22 m`。末端误差约 `0.12 m`，在本次视频专用 `0.25 m` 目标半径内；未发生低矮障碍物碰撞终止。
- 文件：
  - `/home/sea_ws/src/artifacts/depth_passability/videos_teacher_preset_down_y620_630/teacher_bypass_down.mp4`
  - `/home/sea_ws/src/artifacts/depth_passability/videos_teacher_preset_down_y620_630/teacher_bypass_down.csv`
  - `/home/sea_ws/src/artifacts/depth_passability/videos_teacher_preset_down_y620_630/teacher_bypass_summary.json`
- OpenCV 核验视频为 `960x540`、`30 Hz`、`1186` 帧（含 terminal 标注帧），首帧可读且非空；录制脚本 compileall、`git diff --check` 和 `test_stairs_minimal.py` 的 26 项测试均通过。

## 2026-08-02：下行起点 y=6.00、终点 y=6.30 的重新录制

- 按用户最新要求再次只录制下行 preset 视频，起点固定为 `y=6.00`、目标固定为 `y=6.30`，使用 seed `690000` 和同一条低起点障碍绕行路径；没有启动训练，也没有覆盖上一版 `y=6.20 -> 6.30` 产物。
- 汇总 terminal 为 `success`，控制轨迹 `1207` 帧；首帧局部位置约 `(8.60,6.00)`，末帧约 `(0.78,6.16)`，轨迹 y 范围 `5.15~6.16 m`。末端 y 误差约 `0.15 m`，在视频专用 `0.25 m` 目标半径内，未触发低矮障碍物碰撞终止。
- 文件：
  - `/home/sea_ws/src/artifacts/depth_passability/videos_teacher_preset_down_y600_630/teacher_bypass_down.mp4`
  - `/home/sea_ws/src/artifacts/depth_passability/videos_teacher_preset_down_y600_630/teacher_bypass_down.csv`
  - `/home/sea_ws/src/artifacts/depth_passability/videos_teacher_preset_down_y600_630/teacher_bypass_summary.json`
- OpenCV 核验视频为 `960x540`、`30 Hz`、`1208` 帧（含 terminal 标注帧），首帧可读且非空。
