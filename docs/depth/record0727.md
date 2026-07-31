# 2026-07-27 SEA-Nav 双向台阶导航交接记录

## 1. 当前结论

本轮已完成最简方案的代码实现、单元/仿真 smoke test，以及盲式低层 locomotion 到 iteration 2800 的正式训练。按用户要求，2800 checkpoint 和 gate 写盘后已停止训练；没有残留训练进程，也没有可用的 2900 中间 checkpoint。

当前低层 **尚未通过最终 90% gate**，不能当成最终验收模型：

- 上行 `0.25/0.40/0.55/0.70 m/s`：`20/20, 20/20, 20/20, 20/20`。
- 下行 `0.25/0.40/0.55/0.70 m/s`：`3/20, 20/20, 20/20, 20/20`。
- 2800 gate 全部为零跌倒；失败集中在下行 `0.25 m/s` 的 17 次 `stair_stuck`。
- 没有把 timeout 计作 success；逐回合结果使用互斥终止原因。

当前没有执行正式上层 100 回合评测、上层微调和最终 demo 录制。原因是流程规定必须先通过低层 gate。

## 2. 可直接恢复的产物

### 低层 checkpoint

完整的 2800 checkpoint：

```text
/home/sea_ws/src/training/legged_gym/logs/Go2_blind_stair_loco/07_27_05-04-43_staged_gate/model_2800.pt
```

Checkpoint SHA256：

```text
98abb6cf55147f7e937d775d1b87a10f96c575e7d1e256c9be971d43320b83f6
```

2800 已导出的部署模型和严格元数据：

```text
/home/sea_ws/src/training/legged_gym/legged_gym/ctrl_model/blind_stair_loco.pt
/home/sea_ws/src/training/legged_gym/legged_gym/ctrl_model/blind_stair_loco.pt.json
```

TorchScript SHA256：

```text
5d97cfcfab19ebb78b5ed5b6b6809081dbcdc7cd0a04febfd3013876eeef5d58
```

元数据中的 `checkpoint_path` 和两个 SHA256 已与上述文件核对一致。

### Gate 结果

```text
/home/sea_ws/src/training/legged_gym/logs/Go2_blind_stair_loco/target_room_gates/gate_2800.csv
/home/sea_ws/src/training/legged_gym/logs/Go2_blind_stair_loco/target_room_gates/gate_2800.json
/home/sea_ws/src/training/legged_gym/logs/Go2_blind_stair_loco/target_room_gates/gate_history.json
```

500 到 2800 的每个 `gate_<iteration>.csv/json` 都保留在同一目录。CSV 是逐回合轨迹结果，JSON 是八组聚合结果。

### 最接近通过的历史 checkpoint

2600 gate 只有 3 次卡死，是当前总失败数最少的一版：

- 上行：`19/20, 20/20, 20/20, 20/20`。
- 下行：`18/20, 20/20, 20/20, 20/20`。
- 零跌倒，但因为 gate 要求零卡死，仍未通过。

对应 checkpoint：

```text
/home/sea_ws/src/training/legged_gym/logs/Go2_blind_stair_loco/07_27_04-51-36_staged_gate/model_2600.pt
```

## 3. 已实现的代码

### 3.1 盲式 locomotion

新增任务 `go2_blind_stair_loco`：

- Actor/Critic 固定使用 45 维本体观测：`base_ang_vel(3) + projected_gravity(3) + velocity_command(3) + joint_pos_error(12) + joint_vel(12) + last_action(12)`。
- 不使用机身线速度、地形高度、射线、相机、深度图或点云作为 actor 输入。
- 12 维输出使用 SEA 外部关节顺序 `FR, FL, RR, RL`。
- MLP 为 `[512, 256, 128] + ELU`；PPO 为 24 steps/env、5 epochs、4 minibatches、初始学习率 `1e-3`、`gamma=0.99`、`lambda=0.95`。
- 控制契约固定为 50 Hz、action scale `0.25`、`Kp=30`、`Kd=0.75`。
- 关闭只适用于上层三维动作的 range/smooth regularization。
- runner 不再强依赖 `env.rays`。
- 线速度奖励跟踪真实 `[vx, vy]` 命令；增加命令方向归一化的实际进展奖励，解决台阶边缘缺少稠密梯度的问题。
- 每次 episode reset 清零 low-level last action。

主要文件：

```text
training/rsl_rl/rsl_rl/modules/actor_critic.py
training/rsl_rl/rsl_rl/algorithms/ppo.py
training/rsl_rl/rsl_rl/runners/on_policy_runner.py
training/legged_gym/legged_gym/envs/base/legged_robot_blind_stair_loco.py
training/legged_gym/legged_gym/envs/go2/go2_blind_stair_loco_config.py
```

### 3.2 地形与训练课程

- 正式配置为 8192 environments。
- 基础地形比例为 20% 平地、40% 上行负金字塔台阶、40% 下行正金字塔台阶。
- 自定义五级台阶生成器确保踏面严格为 `0.30 m`，高度课程为 `0.02 ... 0.10 m`，避免 Isaac Gym 原生台阶整数截断造成尺寸错误。
- 初始只采样前三个高度等级，之后分段提高至等级 6-9。
- 训练 spawn/命令与目标房间 gate 对齐：上行从低中心向 `+x`，下行从高中心向 `-x`。
- iteration 1800 后采用 10% 平地、10% 上行、80% 下行训练。
- iteration 2700 后启用低速聚焦：70% 台阶命令为 `0.25 m/s`，其余 30% 保留另外三种速度。
- 8192 环境使用 `max_gpu_contact_pairs=44_000_000` 和 buffer multiplier 5，实测无 contact buffer warning。
- 新任务在 headless 时显式禁用 graphics，修复 SEA 创建仿真时的崩溃；旧任务默认行为未改。

主要文件：

```text
training/legged_gym/legged_gym/utils/terrain.py
training/legged_gym/legged_gym/utils/helpers.py
training/legged_gym/legged_gym/envs/base/base_task.py
training/legged_gym/legged_gym/scripts/train_blind_stair_loco.py
```

### 3.3 低层导出和导航接入

- 新增 `blind_stair` backend，原 SLR 三网络 backend 保留。
- TorchScript 加载时严格检查 `45 -> 12`、观测布局、实际 URDF 关节顺序、默认关节角、逐关节 Kp/Kd、频率、action scale 和模型 SHA256。
- 元数据同时记录 checkpoint 路径和 checkpoint SHA256。
- `LeggedRobotPos` 在 reset 时清零 `actions_orig`，防止上一回合动作污染。

主要文件：

```text
training/legged_gym/legged_gym/low_level.py
training/legged_gym/legged_gym/envs/base/legged_robot_pos.py
training/legged_gym/legged_gym/scripts/export_blind_stair_loco.py
```

### 3.4 最简上层任务

新增 `go2_pos_stairs_minimal`：

- 保留原上层 350 维观测、21 射线、10 帧历史和三维速度动作，旧 `Go2_pos_depth_stairs/.../model_200.pt` 可严格加载。
- 21 个射线恒填最大距离，不创建相机、不查询高度图；最简任务关闭 CBF shield。
- 房间保留 10 m x 10 m、五级 `0.08 m x 0.30 m` 台阶和 0.40 m 高台，移除低矮障碍物。
- 起终点位于中心通道，episode 方向 50% 上行、50% 下行。
- 稳态 Go2 基座在低台面约为 0.28 m，因此正确台面高度容差校准为 `0.16 m`；高低平台相差 0.40 m，错误平台仍至少超出 0.24 m，不会被误判为正确高度。

主要文件：

```text
training/legged_gym/legged_gym/envs/base/legged_robot_pos_stairs_minimal.py
training/legged_gym/legged_gym/envs/go2/go2_stairs_minimal_config.py
training/rsl_rl/rsl_rl/modules/cbf_actor_critic.py
```

### 3.5 Timeout 和互斥终止原因

终止原因编码为且只能为以下之一：

```text
success / stair_stuck / fall_or_contact / timeout / other_stuck
```

已实现：

- timeout 使用 `episode_length >= max_episode_length`。
- 失败优先级为 `fall_or_contact > stair_stuck > timeout > other_stuck > success`，因此 success 和 timeout 同步时只记录 timeout。
- success 必须满足 XY 距离 `<0.5 m`、正确台面高度、方向对应的真实跨越、连续保持 12 个控制步，且没有失败终止。
- 任一成功条件失效，`goal_hold_timer` 立即清零。
- 台阶区域连续 2 秒朝目标进展 `<0.15 m`，提前终止为 `stair_stuck`。
- 每个完成 episode 在 reset 前保存独立 `direction / terminal_reason / stair_crossed / episode_steps` 快照。
- 运行时断言 success/timeout 不相交，且每个 done 必须具有唯一非空原因。
- PPO 的 `extras["time_outs"]` 每一步刷新，避免沿用上一帧 timeout。

## 4. 验证记录

已完成并通过：

- Python 单元测试 12/12：台阶几何、45 维布局、互斥终止、timeout 边界、连续保持清零、错误高度拒绝等。
- 真实 Isaac Gym 终止回归：`stair_stuck`、错误高度不成功、到达后离开不成功、success/timeout 同步记 timeout、有效成功记 success。
- 低层 `32 env x 2 iterations` smoke test。
- 低层 `8192 env x 1 iteration` GPU/PPO 张量和显存测试。
- 上层 `32 env x 2 iterations` smoke test。
- 上层 `8192 env x 1 iteration` GPU/PPO 张量和显存测试。
- 旧上层 checkpoint 严格加载，350 维输入输出 3 维动作。
- 低层临时导出后 TorchScript `45 -> 12` 严格加载回归。
- 当前方向进展奖励与低速聚焦分别通过 `32 env x 2 iterations` smoke test。
- `compileall` 和涉及文件的 `git diff --check` 通过。

终止回归脚本：

```bash
conda run -n sea_nav python training/legged_gym/legged_gym/scripts/regress_stairs_terminal.py \
  --task go2_pos_stairs_minimal --headless
```

单元测试：

```bash
PYTHONPATH=/home/sea_ws/src/training/legged_gym:/home/sea_ws/src/training/rsl_rl \
conda run -n sea_nav python -m unittest \
  training/legged_gym/legged_gym/tests/test_stairs_minimal.py
```

## 5. 正式低层训练过程

八组顺序固定为：上行 `0.25/0.40/0.55/0.70`，然后下行 `0.25/0.40/0.55/0.70`。下表数字为每组 20 回合的成功数。

| Iteration | 上行成功数 | 下行成功数 | 总跌倒 | 总卡死 | 说明 |
|---:|---|---|---:|---:|---|
| 500-1500 | 四组均 0 | 四组均 0 | 0 | 多数回合 | 初始训练与逐项修复阶段 |
| 1600 | `0,0,0,11` | `0,0,0,0` | 0 | 146 | 首次出现上行成功 |
| 1700 | `0,0,12,20` | `0,0,0,0` | 0 | 107 | 上行中高速开始形成 |
| 1800 | `0,17,20,20` | `0,0,0,0` | 0 | 103 | 之后启用下行地形聚焦 |
| 1900 | `0,10,20,20` | `0,0,0,0` | 0 | 110 | 修正 staged spawn/方向映射 |
| 2000 | `0,12,20,20` | `0,0,0,0` | 0 | 108 | 下行开始接近台阶边缘 |
| 2100 | `0,20,19,19` | `0,0,0,0` | 0 | 102 | 上行中高速稳定 |
| 2200 | `0,19,20,20` | `0,0,0,0` | 0 | 101 | 下行仍在边缘停滞 |
| 2300 | `0,14,20,14` | `0,0,0,0` | 0 | 111 | 确认需要稠密方向进展梯度 |
| 2400 | `2,20,20,20` | `0,0,1,0` | 0 | 97 | 加入方向进展奖励；发现真实低台面高度窗偏紧 |
| 2500 | `18,20,20,20` | `0,20,20,20` | 0 | 22 | 高度窗校准后中高速双向全过 |
| 2600 | `19,20,20,20` | `18,20,20,20` | 0 | 3 | 当前最接近完整 gate |
| 2700 | `19,20,20,20` | `0,19,20,19` | 0 | 23 | 低速下行能力波动 |
| 2800 | `20,20,20,20` | `3,20,20,20` | 0 | 17 | 低速聚焦一段后，上行全过；下行低速仍不稳 |

注意：早期部分回合既未成功也未触发 gate 的 `fall/stuck`，所以总卡死数不一定等于 160。所有原始结果以 gate CSV/JSON 为准。

## 6. 下一次从哪里继续

### 6.1 最直接的续训方式

当前编排器已经支持从任意分段 checkpoint 恢复，并会在 iteration 2700 后自动添加 `--down_stair_focus --low_speed_focus`。从本次最后完整的 2800 继续到计划上限 3000：

```bash
cd /home/sea_ws/src
conda run -n sea_nav python \
  training/legged_gym/legged_gym/scripts/train_blind_stair_loco.py \
  --num_envs 8192 \
  --first_gate_iteration 500 \
  --gate_interval 100 \
  --max_iterations 3000 \
  --resume_run 07_27_05-04-43_staged_gate \
  --resume_iteration 2800
```

该命令会训练并 gate 2900；若通过立即停止，否则继续到 3000。3000 仍未通过时脚本会以非零状态结束，但 checkpoint、TorchScript 和 gate CSV/JSON 仍会正常保留。

### 6.2 推荐的稳态改进分支

2800 的下行低速仍明显波动，而 2600 只差 3 个 stuck。若 2900/3000 仍未通过，不建议盲目继续用当前自适应 PPO 大步更新。推荐：

1. 以 `model_2600.pt` 建一个分支，或比较 2800，选择低速下行 gate 更好的起点。
2. 给末段训练增加显式 `final_finetune` 开关：固定学习率约 `1e-4`，恢复权重但不要恢复旧 optimizer state，避免自适应学习率使已学会的速度组反复遗忘。
3. 继续保持 70% `0.25 m/s`，但把地形比例从当前 10/10/80 调整为约 10/30/60，防止上行低速和下行其他速度退化。
4. 每 25-50 iterations 跑一次同一 seed 的八组 gate；只有八组均 `>=18/20` 且所有组零跌倒、零 stuck 才接受。
5. 不修改 `stair_stuck` 的 2 秒/0.15 m 门槛来“提高”成功率；当前失败是真实低速下行停滞。

若从最佳 2600 手动重新导出后 gate，可使用：

```bash
cd /home/sea_ws/src
export PYTHONPATH=/home/sea_ws/src/training/legged_gym:/home/sea_ws/src/training/rsl_rl
export WANDB_MODE=disabled

conda run -n sea_nav python \
  training/legged_gym/legged_gym/scripts/export_blind_stair_loco.py \
  --checkpoint training/legged_gym/logs/Go2_blind_stair_loco/07_27_04-51-36_staged_gate/model_2600.pt \
  --output training/legged_gym/legged_gym/ctrl_model/blind_stair_loco.pt

conda run -n sea_nav python \
  training/legged_gym/legged_gym/scripts/gate_blind_stair_loco.py \
  --task go2_pos_stairs_minimal --headless \
  --output_csv /tmp/gate_2600_repeat.csv \
  --output_summary /tmp/gate_2600_repeat.json
```

执行这组命令会覆盖固定交付路径的 TorchScript；若随后决定继续使用 2800，必须重新从 2800 checkpoint 导出。

## 7. 低层通过后的任务顺序

### 第一步：评测现有上层 checkpoint

旧上层模型：

```text
/home/sea_ws/src/training/legged_gym/logs/Go2_pos_depth_stairs/07_14_06-25-22_/model_200.pt
```

正式双向各 50 回合评测：

```bash
cd /home/sea_ws/src
conda run -n sea_nav python \
  training/legged_gym/legged_gym/scripts/evaluate_stairs_minimal.py \
  --task go2_pos_stairs_minimal --headless \
  --episodes_per_direction 50 --success_threshold 0.60 \
  --output_csv training/legged_gym/logs/Go2_pos_stairs_minimal/baseline_eval.csv \
  --output_summary training/legged_gym/logs/Go2_pos_stairs_minimal/baseline_eval.json
```

必须检查 JSON 中上下行都至少 60%，并检查 CSV 中每条 success 都有 `stair_crossed=1` 且 `terminal_reason=success`。

### 第二步：仅在旧上层不达标时微调

```bash
conda run -n sea_nav python \
  training/legged_gym/legged_gym/scripts/train_stairs_minimal.py \
  --task go2_pos_stairs_minimal --headless --num_envs 8192 \
  --max_iterations 1000 --gate_interval 100 \
  --gate_episodes_per_direction 50 --gate_success_threshold 0.60
```

脚本先评测旧 checkpoint；若双向已达标则不训练。否则每 100 iterations gate，首个双向达标 checkpoint 即停止。

### 第三步：最终未见种子评测和交付

- 使用最终上层 checkpoint 再跑上/下行各 50 回合，保存 `final_eval.csv/json`。
- 统计五类互斥失败原因。
- 分别录制完整成功上行和下行视频；录制脚本会拒绝保存非 success 回合。
- 交付低层 checkpoint、TorchScript、JSON 元数据、最终上层 checkpoint、100 回合 CSV/JSON 和两段视频。

命令和完整运行说明另见：

```text
/home/sea_ws/src/docs/stairs_minimal.md
```

## 8. 中断状态与注意事项

- 当前训练进程已停止，GPU 上没有本任务残留进程。
- `model_2800.pt` 完整；不存在 `model_2900.pt`，不要尝试从 2900 恢复。
- 固定部署模型当前确实来自 2800；若手动导出历史 checkpoint，记得在最终选择后重新导出。
- 2800 gate 未通过，正式上层评测/微调尚未开始。
- `docs/depth/plan.md` 在开始本任务前已有用户改动，本轮没有回滚或覆盖该改动。
- 工作树尚未提交；新增源文件、文档和 `blind_stair_loco.pt(.json)` 都是未跟踪文件，提交前需按 `git status` 审核。

## 9. 2800 训练环境人工审查视频

使用 `model_2800.pt` 在第 6 课程行的五级 `0.08 m x 0.30 m` 训练台阶录制。每个方向分别覆盖 `0.25/0.40/0.55/0.70 m/s`，关闭观测噪声、摩擦/质量随机化和外力扰动。视频为近距离跟随视角，`960x540 @ 50 FPS`。

产物目录：

```text
/home/sea_ws/src/training/legged_gym/logs/Go2_blind_stair_loco/review_2800/
```

上行：

```text
up_01_025cms.mp4    crossed, 7.38 s
up_02_040cms.mp4    crossed, 5.14 s
up_03_055cms.mp4    crossed, 4.06 s
up_04_070cms.mp4    crossed, 3.66 s
```

下行：

```text
down_01_025cms.mp4  not crossed within 20.00 s
down_02_040cms.mp4  crossed, 5.48 s
down_03_055cms.mp4  crossed, 3.84 s
down_04_070cms.mp4  crossed, 3.34 s
```

每段视频都有同名 CSV 轨迹。`up_summary.json` 和 `down_summary.json` 记录 checkpoint 路径、iteration、SHA256、速度、帧数、跨越结果和最终位置。

低速下行在最后一级附近停止：最大定向局部位移为 `2.5733 m`，最后 2 秒进展仅 `0.0001 m`。这与 2800 gate 的下行 `0.25 m/s` 卡死一致，不是录制器提前停止。

可复用录制脚本：

```text
training/legged_gym/legged_gym/scripts/record_blind_stair_loco.py
```

## 10. 人工审查发现的设计偏差（问题定义）

### 10.1 下行方向错误：当前是倒退下台阶

人工检查 2800 视频后确认，当前下行保持机器人头部朝 `+x`，同时给机体系负向线速度，因此机器人以头部较高、尾部在前的姿态倒退下台阶。这虽然可能满足旧 gate 的世界坐标位移条件，但不符合目标行为。

期望行为是：

- 下行开始时机器人头部朝向下台阶和目标方向。
- 机器人使用正向步态，头在前、向前行进下台阶。
- 不接受通过负向 `vx` 倒退完成下行。

代码原因：

- `LeggedRobotBlindStairLoco._resample_commands()` 当前对下行设置 `vx=-speed`。
- 下行 spawn 没有把 base yaw 旋转 180 度，机器人仍面向 `+x`。
- gate 和录制器也沿用负向 `vx`，所以旧评测会把倒退下行算作成功。

继续训练前需要统一修改：

1. 下行训练 spawn 将 base yaw 设置为 `pi`，使头部朝世界 `-x` 和下台阶方向。
2. 上、下行都向低层发送机体系正向 `vx=+speed`；下行的世界位移依靠 yaw=`pi` 产生，而不是使用负 `vx`。
3. 最简导航下行 episode 的初始 yaw 也应朝向目标，或明确要求上层先原地转向再发送正向速度。为最快实现，建议直接按方向设置初始 yaw：上行 `0`、下行 `pi`。
4. gate 和录制脚本增加朝向约束：机身前向向量必须与世界目标方向同向，且跨台阶期间的机体系前向速度为正。
5. 下行 gate 不再使用负向速度命令；八组仍表示四个正向速度在上/下两种坡向上的能力。

建议的运行时验收条件：

```text
dot(body_forward_xy, world_goal_direction_xy) > cos(20 deg)
command_vx > 0
body_forward_velocity > 0
```

该问题改变了低层下行训练契约。因此，2800 的旧下行成功率和视频只能用于定位问题，不能继续作为正式 gate 结果；修正后必须重新训练/微调并重跑全部下行 gate。

### 10.2 跨越终点过近：机器人尚未完全离阶就结束

人工检查还发现，上下行视频在机器人基座刚越过最后一级时结束，后腿或足端仍可能位于台阶上，没有完整进入目标侧平地。

当前条件确实偏近：

- 训练金字塔台阶的外缘约为局部坐标 `|x|=3.0 m`。
- `record_blind_stair_loco.py` 使用定向局部坐标 `>=3.15 m`，仅比台阶边缘多 `0.15 m`。
- 目标房间 gate 上行使用 `x>=6.30 m`，基本等于台阶末端；下行使用 `x<=4.50 m`，离台阶起点仅 `0.30 m`。
- 这些条件只检查基座位置和高度，没有确认四足都已离开台阶并稳定站到同一目标平面。

新的完成定义应为“完整离开台阶并进入目标侧平地”，建议统一为：

1. 基座越过最后一级后至少继续前进 `0.8 m`。训练地形可使用定向局部坐标 `>=3.8 m`；目标房间可使用上行 `x>=7.1 m`、下行 `x<=4.0 m`。
2. 四个足端均位于目标侧平地范围，不允许后足仍在最后一级踏面上。
3. 四足目标平面高度/接触状态有效，机身姿态稳定。
4. 上述条件连续保持 12 个控制步；任一条件失效立即清零计时器。
5. `stair_stuck` 和 timeout 检测持续到完整离阶条件成立，不能在最后一级提前停止。

具体阈值在实现时应由台阶几何量计算，不应在训练环境、导航环境、gate 和录制器中分别硬编码。建议新增统一的 `stair_end_x / clearance_distance / all_feet_cleared` 配置或辅助函数。

需要新增的回归场景：

- 基座刚越过台阶边缘、后足仍在踏面上：不得 success。
- 四足已到目标侧平地但未保持满 12 步：不得 success。
- 四足完整离阶并稳定保持 12 步：可以 success。
- 到达离阶位置后退回最后一级：保持计时立即清零。
- 恰好在 timeout 步完整离阶：仍只能记 timeout。

### 10.3 后续执行顺序修订

下一次继续任务时，不应直接按第 6 节命令把旧契约从 2800 训练到 3000。新的顺序应为：

1. 先修正下行 yaw、正向速度命令和统一的完整离阶条件。
2. 更新 gate、录制器、最简导航 success/stuck 判断和相应单元测试。
3. 用 32 environments、2 iterations 验证新方向契约和终点逻辑。
4. 从 2800 checkpoint 尝试低学习率微调；若策略因旧负向下行行为难以纠正，再从较早 checkpoint 或重新训练分支。
5. 重新执行四速度上/下行 gate，并录制八段人工审查视频。
6. 只有“头在前正向下行”和“上下行四足完整离阶”都满足后，才进入上层 100 回合评测。

在这两项仿真修正完成前，2800 模型状态应标记为：

```text
工程链路可运行，但运动方向与完成边界不符合最终需求，不可验收。
```

## 11. 修复后仿真复核（未训练）

已在仿真契约中完成第 10 节两项修正，本次没有调用训练入口，也没有更新 2800 checkpoint 或 TorchScript 权重。

实现内容：

- 下行 spawn yaw 设置为 `pi`，机器人头部朝世界下台阶方向。
- 上行和下行都发送机体系正向 `vx>0`；下行不再使用负 `vx` 倒退。
- 完成条件改为：基座越过台阶外缘 `0.8 m`、四足全部越过台阶外缘、足端位于目标平面、机身朝向目标，并连续保持 12 个控制步。
- 最简导航、低层训练环境、gate 和录制器使用同一方向/离阶定义。
- `stair_stuck` 检测区域延伸到完整离阶边界，最后一级外侧停滞不能提前算成功。

使用原始 `model_2800.pt` 在 `0.08 m x 0.30 m` 五级台阶、`0.40 m/s` 正向命令下各执行一次：

```text
上行：fully_cleared，6.52 s，最终局部 x=+3.9134 m，四足完全离阶
下行：fully_cleared，7.16 s，最终局部 x=-3.9097 m，四足完全离阶
```

下行全程命令为机体系 `vx=+0.40 m/s`，最小目标朝向对齐值为 `0.9447`，满足 20 度朝向约束；这是头在前的正向下台阶，不是倒退。

合并人工审查视频：

```text
/home/sea_ws/src/training/legged_gym/logs/Go2_blind_stair_loco/review_2800_fixed_sim/model_2800_fixed_sim_review.mp4
```

分段视频、逐步轨迹和摘要：

```text
/home/sea_ws/src/training/legged_gym/logs/Go2_blind_stair_loco/review_2800_fixed_sim/up_01_040cms.mp4
/home/sea_ws/src/training/legged_gym/logs/Go2_blind_stair_loco/review_2800_fixed_sim/down_01_040cms.mp4
/home/sea_ws/src/training/legged_gym/logs/Go2_blind_stair_loco/review_2800_fixed_sim/up_01_040cms.csv
/home/sea_ws/src/training/legged_gym/logs/Go2_blind_stair_loco/review_2800_fixed_sim/down_01_040cms.csv
/home/sea_ws/src/training/legged_gym/logs/Go2_blind_stair_loco/review_2800_fixed_sim/up_summary.json
/home/sea_ws/src/training/legged_gym/logs/Go2_blind_stair_loco/review_2800_fixed_sim/down_summary.json
```

本次只是单次确定性人工复核，不替代修正后四速度、多回合 gate；是否继续训练应在人工确认该视频姿态满足预期后决定。
