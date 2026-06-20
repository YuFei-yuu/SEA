# SEA-Nav Motion-Aware 复杂动态障碍物改造记录与快速验证报告

> 日期：2026-06-20 | 环境：RTX 4090 (24GB) + Ubuntu 20.04 + Isaac Gym

---

## 一、改造目标

本次改造目标是将 `go2_pos_dynamic_complex` 从“复杂动态障碍物工程可行性验证”推进到更适合持续训练和后续研究汇报的版本，重点解决：

- 复杂动态障碍物 reset/update 过程存在 Python 循环和在线重采样，训练效率偏低。
- 动态障碍物之间的碰撞、穿模和 PhysX 接触求解不是当前研究重点，容易干扰避障算法验证。
- 原 SEA-Nav 的 LSE-CBF 主要基于瞬时 ray 距离，对动态障碍物的相对速度和 TTC 利用不足。
- 需要保留当前 `830` 维观测结构，避免再次破坏已有复杂任务 checkpoint 和训练链路。

本轮实现后的设计目标是：

- episode 初始化阶段一次性生成每个动态障碍物的轨迹。
- step 阶段只按轨迹索引刷新动态障碍物位置和速度。
- 忽略动态障碍物之间、动态障碍物与静态障碍物之间的真实碰撞和穿模。
- 机器人与动态障碍物的碰撞仍通过几何距离正常触发和检测。
- 在原静态 LSE-CBF 后增加基于 dynamic token 的 velocity-aware dynamic CBF。

---

## 二、主要修改内容

### 1. 复杂动态障碍物改为 episode 预生成轨迹

涉及文件：

- `training/legged_gym/legged_gym/envs/go2/go2_pos_config.py`
- `training/legged_gym/legged_gym/envs/base/legged_robot_pos_dynamic.py`

新增配置：

```python
trajectory_mode = "episode_precomputed"
trajectory_extra_horizon = 0.8
disable_simulation_contacts = True
```

实现方式：

- 在 reset 阶段生成：
  - `dynamic_traj_pos`
  - `dynamic_traj_vel`
  - `dynamic_traj_step`
- 轨迹缓存形状为：

```text
[num_envs, max_dynamic_obstacles, trajectory_len, 2]
```

- `linear_crossing` 和 `linear_diagonal` 改为闭合 ping-pong 轨迹。
- `circular` 和 `figure_eight` 使用预采样相位轨迹。
- step 阶段不再逐 env、逐 obstacle 调用 `_advance_linear_slot()`，而是通过张量索引直接取轨迹上的当前位置和速度。

### 2. 保留机器人-动态障碍物几何碰撞检测

本轮没有依赖 PhysX 接触判断动态障碍物碰撞，而是继续使用几何距离：

```text
dist(robot_xy, dynamic_obstacle_xy) < dynamic_collision_config.threshold
```

触发后会更新：

- `dynamic_collision_event`
- `dynamic_collision_count`
- `total_collision_count`
- `reset_buf`
- `terminate_buf`

因此即使动态障碍物之间互相穿过，机器人撞到动态障碍物仍然会被检测。

### 3. 修正 near-miss replay 触发逻辑

新增 sticky 状态：

```python
near_miss_occurred
```

原因：

- 原先只看当前 step 的 `near_miss_event`，reset 时可能已经错过危险事件。
- 改为 episode 内只要发生过 near-miss，就记录 sticky flag。

reset 时 replay 触发条件变为：

```text
near_miss_occurred 或 near_miss_count > 0
```

同时 replay 会恢复 `dynamic_traj_step`，保证机器人回放到危险片段前后，动态障碍物仍沿同一条 episode 轨迹继续运动。

### 4. 增加 velocity-aware dynamic CBF

涉及文件：

- `training/rsl_rl/rsl_rl/modules/cbf_lse_layer.py`
- `training/rsl_rl/rsl_rl/modules/cbf_actor_critic.py`

新增无参数安全层：

```python
DynamicTokenCBFLayer
```

使用 dynamic token：

```text
[rel_x, rel_y, rel_vx, rel_vy, radius, ttc, valid]
```

CBF 形式：

```text
h = ||p_rel||^2 - d_safe^2
dot(h) = 2 * p_rel^T * (v_obs - u_robot)
dot(h) + alpha * h >= 0
```

策略输出流程变为：

```text
u_bar
  -> static/ray LSE-CBF
  -> u_static_safe
  -> dynamic token CBF
  -> u_s
```

输出动作维度仍为 `[vx, vy, yaw_rate]`，yaw action 不被 dynamic CBF 修改。

### 5. 评估指标补充

涉及文件：

- `training/legged_gym/legged_gym/scripts/evaluate_dynamic.py`

新增输出：

- `active_dynamic_count`
- `min_dynamic_clearance`
- `dynamic_cbf_intervention_rate`

这些指标用于区分：

- 场景中实际激活了多少动态障碍物。
- 机器人与动态障碍物的最小几何余量。
- 动态风险区域中 safety layer 的介入强度。

---

## 三、使用的验证命令

### 1. Python 静态检查

```bash
cd /home/sea_ws/src

python -m py_compile \
  training/legged_gym/legged_gym/envs/go2/go2_pos_config.py \
  training/legged_gym/legged_gym/envs/base/legged_robot_pos_dynamic.py \
  training/rsl_rl/rsl_rl/modules/cbf_lse_layer.py \
  training/rsl_rl/rsl_rl/modules/cbf_actor_critic.py \
  training/legged_gym/legged_gym/scripts/evaluate_dynamic.py
```

### 2. dynamic CBF 小张量单测

```bash
cd /home/sea_ws/src

PYTHONPATH=/home/sea_ws/src/training/rsl_rl python - <<'PY'
import torch
from rsl_rl.modules.cbf_lse_layer import DynamicTokenCBFLayer

layer = DynamicTokenCBFLayer()
u = torch.tensor([[0.5, 0.0, 0.1], [0.2, 0.1, -0.1]])
dyn = torch.zeros(2, 4, 7)
dyn[:, 0, 0] = 0.5
dyn[:, 0, 2] = -0.4
dyn[:, 0, 4] = 0.32
dyn[:, 0, 6] = 1.0
base = torch.zeros(2, 2)
alpha = torch.ones(2, 1)
out = layer(u, dyn, base, alpha)
print(out.shape)
print(out)
PY
```

### 3. `32 env` 环境 reset/step 冒烟测试

```bash
cd /home/sea_ws/src/training/legged_gym

PYTHONPATH=/home/sea_ws/src/training/legged_gym:/home/sea_ws/src/training/rsl_rl \
OMP_NUM_THREADS=8 \
python - <<'PY'
import os, sys, time
import isaacgym
import torch
from legged_gym.envs import *
from legged_gym.utils import task_registry, get_args

args = get_args()
args.task = "go2_pos_dynamic_complex"
args.headless = True
args.num_envs = 32

env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
env_cfg.env.num_envs = 32
env_cfg.noise.add_noise = False
env_cfg.domain_rand.push_robots = False
env_cfg.domain_rand.randomize_friction = False
env_cfg.domain_rand.randomize_base_mass = False
env_cfg.asset.terminate_after_contacts_on = []

env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
obs, _ = env.reset()
actions = torch.zeros((env.num_envs, env.cfg.env.num_nav_actions), device=env.device)
obs, _, _, dones, infos = env.step(actions)

counts = env.dynamic_active_mask.sum(dim=1).detach().cpu()
print("obs_shape", tuple(obs.shape))
print("active_min", int(counts.min().item()))
print("active_max", int(counts.max().item()))
print("active_mean", float(counts.float().mean().item()))
print("traj_shape", tuple(env.dynamic_traj_pos.shape))
print("traj_step_minmax", int(env.dynamic_traj_step.min().item()), int(env.dynamic_traj_step.max().item()))
print("dones_sum", int(dones.sum().item()))
print("min_ttc_mean", float(env.min_ttc.mean().item()))
print("min_clearance_mean", float(env.min_dynamic_clearance.mean().item()))
PY
```

### 4. `32 env / 200 step` 性能验证

```bash
cd /home/sea_ws/src/training/legged_gym

PYTHONPATH=/home/sea_ws/src/training/legged_gym:/home/sea_ws/src/training/rsl_rl \
OMP_NUM_THREADS=8 \
python - <<'PY'
# 与上一个脚本相同的环境创建逻辑，连续 step 200 次并统计耗时。
PY
```

### 5. 几何碰撞触发验证

```bash
cd /home/sea_ws/src/training/legged_gym

PYTHONPATH=/home/sea_ws/src/training/legged_gym:/home/sea_ws/src/training/rsl_rl \
OMP_NUM_THREADS=8 \
python - <<'PY'
# 创建 4 env，把 env 0 的机器人 root state 移到第一个 active dynamic obstacle 中心，
# 然后调用 update_percetion() 和 check_termination()。
PY
```

### 6. 最小训练闭环

```bash
cd /home/sea_ws/src/training/legged_gym

PYTHONPATH=/home/sea_ws/src/training/legged_gym:/home/sea_ws/src/training/rsl_rl \
WANDB_MODE=disabled \
OMP_NUM_THREADS=8 \
python legged_gym/scripts/train.py \
  --task go2_pos_dynamic_complex \
  --headless \
  --num_envs 8 \
  --max_iterations 1
```

以及：

```bash
cd /home/sea_ws/src/training/legged_gym

PYTHONPATH=/home/sea_ws/src/training/legged_gym:/home/sea_ws/src/training/rsl_rl \
WANDB_MODE=disabled \
OMP_NUM_THREADS=8 \
python legged_gym/scripts/train.py \
  --task go2_pos_dynamic_complex \
  --headless \
  --num_envs 32 \
  --max_iterations 1
```

---

## 四、验证结果

### 1. 静态检查结果

`py_compile` 通过。

`git diff --check` 通过，没有发现尾随空格或补丁格式问题。

### 2. dynamic CBF 小张量单测结果

输出：

```text
torch.Size([2, 3])
tensor([[ 0.0398,  0.0000,  0.1000],
        [-0.1102,  0.1000, -0.1000]])
```

结论：

- `DynamicTokenCBFLayer` 输出维度正确。
- 输出中没有 NaN/Inf。
- yaw action 保持透传。
- 当动态 token 表示前方障碍物正在靠近时，`vx` 被明显压低，符合动态 CBF 的预期效果。

### 3. `32 env` 环境冒烟测试结果

关键输出：

```text
obs_shape (32, 830)
active_min 6
active_max 10
active_mean 7.78125
traj_shape (32, 10, 2294, 2)
traj_step_minmax 0 2
dones_sum 2
min_ttc_mean 2.4051685333251953
min_clearance_mean 2.3231585025787354
motion_counts_first_env [3, 0, 3, 1]
```

结论：

- 观测维度保持为 `830`，没有破坏当前复杂任务输入结构。
- 每个 episode 的 active dynamic obstacle 数量稳定落在 `6~10`。
- 轨迹缓存已生成，形状为 `(32, 10, 2294, 2)`。
- 首步存在少量 reset，但环境能够正常 reset/step。

### 4. `32 env / 200 step` 性能验证结果

关键输出：

```text
obs_shape (32, 830)
active_min 6
active_max 10
active_mean 7.84375
traj_step_minmax 5 201
dones_total_200 16
elapsed_200_steps 6.881
steps_per_sec 29.07
min_ttc_mean 2.9516024589538574
min_clearance_mean 2.756077766418457
```

结论：

- 200 step 内 active dynamic obstacle 数量持续保持 `6~10`。
- `dynamic_traj_step` 会随 episode 前进，reset 后的 env 会重新回到较小 step，因此 min/max 为 `5~201` 是合理现象。
- 当前更新逻辑已经不再卡在 `_bbox_points_valid()`、`_init_dynamic_slot()`、`_advance_linear_slot()` 等逐障碍 Python 逻辑上。

### 5. 几何动态碰撞验证结果

关键输出：

```text
slot 0
dynamic_collision_event_0 True
dynamic_collision_count_0 1
reset_buf_0 True
```

结论：

- 机器人和动态障碍物的几何碰撞可以正常触发。
- 动态障碍物之间允许穿过，不影响机器人碰撞检测。
- `dynamic_collision_count` 和 early reset 均正常工作。

### 6. 最小训练闭环结果

`8 env / 1 iteration` 保存模型：

```text
/home/sea_ws/src/training/legged_gym/logs/Go2_pos_dynamic_complex/06_20_02-58-09_/model_1.pt
```

`32 env / 1 iteration` 保存模型：

```text
/home/sea_ws/src/training/legged_gym/logs/Go2_pos_dynamic_complex/06_20_02-59-42_/model_1.pt
```

checkpoint 检查：

```text
06_20_02-58-09_ iter 1 keys 32
06_20_02-59-42_ iter 1 keys 32
```

结论：

- 环境、策略前向、PPO 采样、PPO 更新、checkpoint 保存链路已打通。
- `DynamicTokenCBFLayer` 是无参数安全层，因此 checkpoint 中没有新增 dynamic CBF 参数，这是预期行为。

注意：

- `train.py` 在 checkpoint 保存后仍出现 Isaac Gym 清理阶段的 `Segmentation fault (core dumped)`，返回码为 `139`。
- 由于 checkpoint 已经成功写出并可被 `torch.load()` 读取，本次判断训练闭环本身已经完成。
- 该问题更像 Isaac Gym 进程析构/清理阶段问题，而不是 Python 逻辑或 PPO 更新失败。

---

## 五、视频录制

### 使用命令

```bash
cd /home/sea_ws/src/training/legged_gym

PYTHONPATH=/home/sea_ws/src/training/legged_gym:/home/sea_ws/src/training/rsl_rl \
OMP_NUM_THREADS=8 \
xvfb-run -a python legged_gym/scripts/play_record.py \
  --task go2_pos_dynamic_complex \
  --resume \
  --load_run 06_20_02-59-42_ \
  --checkpoint 1
```

### 视频产物

#### 1. 障碍物轨迹检查短视频

该视频使用启发式导航器录制 `600` 帧，主要用于检查复杂动态障碍物是否按 episode 预生成轨迹平滑运动。

| 文件 | 路径 | 大小 | 帧数 | active dynamic obstacles |
|---|---|---:|---:|---:|
| obstacle check video | `training/legged_gym/logs/Go2_pos_dynamic_complex/exported/motion_aware_dynamic_complex_obstacle_check.mp4` | 9.4 MB | 600 | 7 |

关键终端输出：

```text
Video saved: /home/sea_ws/src/training/legged_gym/logs/Go2_pos_dynamic_complex/exported/motion_aware_dynamic_complex_obstacle_check.mp4 (9.4 MB, 600 frames)
active_dynamic_count [7]
trajectory_mode episode_precomputed
```

结论：

- 视频对应的环境已经启用 `episode_precomputed` 轨迹模式。
- 单个 episode 中动态障碍物数量为 `7`，符合复杂任务 `6~10` 的配置。

#### 2. 最新 checkpoint 策略视频

该视频使用本轮 `32 env / 1 iteration` 产出的最新 checkpoint：

```text
training/legged_gym/logs/Go2_pos_dynamic_complex/06_20_02-59-42_/model_1.pt
```

| 文件 | 路径 | 大小 | 帧数 |
|---|---|---:|---:|
| policy video | `training/legged_gym/logs/Go2_pos_dynamic_complex/exported/06_20_02-59-42_model_1.mp4` | 66.2 MB | 5689 |

关键终端输出：

```text
Loading model from: /home/sea_ws/src/training/legged_gym/logs/Go2_pos_dynamic_complex/06_20_02-59-42_/model_1.pt
Loaded policy from: /home/sea_ws/src/training/legged_gym/logs/Go2_pos_dynamic_complex/06_20_02-59-42_/model_1.pt
Video will be saved to: /home/sea_ws/src/training/legged_gym/logs/Go2_pos_dynamic_complex/exported/06_20_02-59-42_model_1.mp4
Recorder mode: policy
=== Episode 1/5 finished | success=0.00 dynamic_collision_count=0.00 ===
=== Episode 2/5 finished | success=0.00 dynamic_collision_count=0.00 ===
=== Episode 3/5 finished | success=0.00 dynamic_collision_count=0.00 ===
=== Episode 4/5 finished | success=0.00 dynamic_collision_count=0.00 ===
=== Episode 5/5 finished | success=0.00 dynamic_collision_count=0.00 ===
Video saved: /home/sea_ws/src/training/legged_gym/logs/Go2_pos_dynamic_complex/exported/06_20_02-59-42_model_1.mp4  (66.2 MB, 5689 frames)
```

结论：

- 最新复杂任务 checkpoint 可以正常加载并录制策略视频。
- 由于该模型只训练了 `1 iteration`，`5/5` episode 成功率为 `0` 属于预期现象。
- 本视频主要用于检查复杂动态障碍环境、轨迹播放和策略录像链路，不代表最终避障性能。

---

## 六、模型与代码产物

### 模型产物

| 产物 | 路径 | 大小 |
|---|---|---:|
| `8 env / 1 iter` checkpoint | `training/legged_gym/logs/Go2_pos_dynamic_complex/06_20_02-58-09_/model_1.pt` | 13 MB |
| `32 env / 1 iter` checkpoint | `training/legged_gym/logs/Go2_pos_dynamic_complex/06_20_02-59-42_/model_1.pt` | 13 MB |

### 修改文件

```text
training/legged_gym/legged_gym/envs/base/legged_robot_pos_dynamic.py
training/legged_gym/legged_gym/envs/go2/go2_pos_config.py
training/legged_gym/legged_gym/scripts/evaluate_dynamic.py
training/rsl_rl/rsl_rl/modules/cbf_actor_critic.py
training/rsl_rl/rsl_rl/modules/cbf_lse_layer.py
```

改动统计：

```text
5 files changed, 420 insertions(+), 38 deletions(-)
```

---

## 七、当前结论

本次修改完成后，`go2_pos_dynamic_complex` 已经从“在线采样/在线更新的复杂动态障碍物环境”改造成更适合训练的“episode 预生成轨迹环境”。

当前最重要的结论是：

1. 复杂动态障碍物数量已经稳定达到 `6~10`。
2. 动态障碍物 step 更新已经张量化，不再依赖逐障碍在线重采样。
3. 动态障碍物可以互相穿过，避免 PhysX 接触求解干扰。
4. 机器人与动态障碍物的几何碰撞仍能正常触发。
5. 原 `830` 维观测结构保持不变。
6. dynamic token 已进入 velocity-aware CBF 安全层。
7. 最小训练闭环已经能保存新的复杂任务 checkpoint。

---

## 八、`64 env / 500 iterations` 训练实验记录

### 1. 实验目标

在完成 motion-aware 复杂动态障碍物环境改造后，本轮实验希望回答两个问题：

- 当前 `go2_pos_dynamic_complex` 是否已经具备稳定训练出有效避障策略的条件。
- motion-aware dynamic CBF、安全奖励和预生成轨迹环境组合后，策略会学到“有效到达目标”，还是只学到“保守避障”。

### 2. 使用命令

#### 训练命令

```bash
cd /home/sea_ws/src

PYTHONPATH=/home/sea_ws/src/training/legged_gym:/home/sea_ws/src/training/rsl_rl python training/legged_gym/legged_gym/scripts/train.py   --task go2_pos_dynamic_complex   --num_envs 64   --max_iterations 500   --headless
```

#### 50 episodes 评估命令

```bash
cd /home/sea_ws/src

PYTHONPATH=/home/sea_ws/src/training/legged_gym:/home/sea_ws/src/training/rsl_rl python training/legged_gym/legged_gym/scripts/evaluate_dynamic.py   --task go2_pos_dynamic_complex   --load_run 06_20_05-14-50_   --checkpoint 500   --num_episodes 50   --headless
```

#### 10 episodes 录像命令

```bash
cd /home/sea_ws/src

PYTHONPATH=/home/sea_ws/src/training/legged_gym:/home/sea_ws/src/training/rsl_rl python training/legged_gym/legged_gym/scripts/play_record.py   --task go2_pos_dynamic_complex   --load_run 06_20_05-14-50_   --checkpoint 500   --num_episodes 10   --headless
```

### 3. 训练产物

本轮训练 run 目录：

```text
training/legged_gym/logs/Go2_pos_dynamic_complex/06_20_05-14-50_
```

关键产物：

| 产物 | 路径 | 说明 |
|---|---|---|
| 最终模型 | `training/legged_gym/logs/Go2_pos_dynamic_complex/06_20_05-14-50_/model_500.pt` | 500 iterations 训练结果 |
| 训练指标 | `training/legged_gym/logs/Go2_pos_dynamic_complex/06_20_05-14-50_/train_metrics.csv` | 每 10 iter 记录一次 |
| 训练摘要 | `training/legged_gym/logs/Go2_pos_dynamic_complex/06_20_05-14-50_/analysis/train_summary.txt` | 训练后自动整理 |
| 50 episodes 评估摘要 | `training/legged_gym/logs/Go2_pos_dynamic_complex/06_20_05-14-50_/analysis/eval_50_summary.txt` | 固定评估结果 |
| 训练曲线 | `training/legged_gym/logs/Go2_pos_dynamic_complex/06_20_05-14-50_/analysis/*.png` | reward / collision / TTC / optimization / efficiency |
| 10 episodes 录像 | `training/legged_gym/logs/Go2_pos_dynamic_complex/exported/06_20_05-14-50_model_500.mp4` | 用于观察策略行为 |

### 4. 训练过程摘要

训练日志中实际写入的最后一个统计点为 `iteration=490`，`train_metrics.csv` 共 `48` 个记录点。

`train_summary.txt` 关键内容：

```text
last_iteration: 490
last_mean_reward: 0.1141
best_success: 0.0000
best_safe_success: 0.0000
last_dynamic_collision_count: 0.0064
last_total_collision_count: 0.0064
last_near_miss_count: 0.7097
last_min_ttc: 3.6019
last_min_dynamic_clearance: 3.1702
```

训练阶段观察到的现象：

- `success` 和 `safe_success` 在整个训练过程中始终没有抬升到非零。
- `dynamic_collision_count` 长时间维持在很低水平，说明策略和安全层确实在压制动态碰撞。
- `shield_intervention_rate`、`dynamic_cbf_intervention_rate` 在训练中长期非零，说明安全层频繁介入。
- `mean_reward` 在零附近小幅波动，没有出现随着成功率增长而明显抬升的趋势。

这说明当前训练过程更像是在学“保守安全动作”，而不是学“稳定到达目标”。

### 5. 50 episodes 评估结果

`evaluate_dynamic.py` 固定评估 `50` 个 episode 后，得到：

```text
success_rate: 0.0000
safe_success_rate: 0.0000
avg_total_collision_count: 0.0200
avg_dynamic_collision_count: 0.0000
avg_body_collision_count: 0.0200
avg_near_miss_count: 0.6600
avg_min_ttc: 3.1299
avg_shield_intervention_rate: 0.4484
avg_active_dynamic_count: 6.5200
avg_min_dynamic_clearance: 2.1584
avg_dynamic_cbf_intervention_rate: 0.1945
timeout_rate: 0.0000
mean_time_to_goal: 0.0000
```

结论：

- 最终模型在 `50` 个 episode 中 `0` 次成功，到达率为 `0%`。
- 动态障碍物碰撞次数平均为 `0`，说明“动态避碰”本身并没有失效。
- 机器人本体碰撞次数很低，但仍存在少量静态或其他非动态碰撞。
- `near_miss_count` 仍然明显非零，说明机器人经常进入危险动态交互区域。
- `timeout_rate=0` 且大量 episode 持续时间极短，说明当前失败不是“走太久到不了”，而更像是“很快进入终止状态或停滞状态”。

### 6. 10 episodes 视频结果

视频文件：

```text
training/legged_gym/logs/Go2_pos_dynamic_complex/exported/06_20_05-14-50_model_500.mp4
```

文件大小：

```text
1.7 MB
```

关键终端输出：

```text
=== Episode 1/10 finished | success=0.00 dynamic_collision_count=0.00 ===
=== Episode 2/10 finished | success=0.00 dynamic_collision_count=0.00 ===
=== Episode 3/10 finished | success=0.00 dynamic_collision_count=0.00 ===
=== Episode 4/10 finished | success=0.00 dynamic_collision_count=0.00 ===
=== Episode 5/10 finished | success=0.00 dynamic_collision_count=0.00 ===
=== Episode 6/10 finished | success=0.00 dynamic_collision_count=0.00 ===
=== Episode 7/10 finished | success=0.00 dynamic_collision_count=0.00 ===
=== Episode 8/10 finished | success=0.00 dynamic_collision_count=0.00 ===
=== Episode 9/10 finished | success=0.00 dynamic_collision_count=0.00 ===
=== Episode 10/10 finished | success=0.00 dynamic_collision_count=0.00 ===
Video saved: .../06_20_05-14-50_model_500.mp4  (1.6 MB, 97 frames)
```

结论：

- `10/10` episode 全部失败。
- 整段视频只有 `97` 帧，说明大量 episode 非常短。
- 从这一现象看，当前模型没有学到有效的复杂动态导航行为，更多是在少量动作后快速进入失败或终止状态。

### 7. 本轮训练结论

本轮 `64 env / 500 iterations` 训练的主要结论不是“复杂动态避障任务已解决”，而是：

1. 当前环境、训练、评估、录像和分析链路都已经完整跑通。
2. 当前策略在复杂动态环境下能够把动态碰撞压到很低，说明 dynamic token + dynamic CBF + TTC/near-miss reward 的安全侧信号是有效的。
3. 但当前配方没有学出任务成功能力，策略没有稳定到达目标。
4. 当前失败模式更接近“过度保守 / 快速终止 / 未建立有效推进策略”，而不是“动态碰撞太多导致失败”。

因此，这次训练更适合作为“复杂动态任务基座已搭好，但当前训练配方未收敛”的实验记录，而不能作为最终性能结果。

---

## 九、下一步建议

建议后续按下面顺序推进：

1. 人工检查本次录制视频，确认动态障碍物是否沿预生成轨迹平滑运动，是否仍有异常抖动。
2. 如果视频确认环境合理，启动 `32 env / 20~100 iterations` 的短训。
3. 用 `evaluate_dynamic.py` 固定评估 `20/50 episodes`，记录 success、safe_success、dynamic collision、near-miss、min TTC、min dynamic clearance。
4. 做消融对比：
   - trajectory-only
   - trajectory + TTC/near-miss reward
   - trajectory + dynamic CBF
   - trajectory + dynamic CBF + near-miss replay
5. 如果训练仍不稳定，优先调 curriculum，而不是立刻引入 GNN、Transformer 或 Reach-Avoid value network。


---

## 九、`64 env / 500 iterations` 失败排查与修复

### 1. 问题现象

在 `06_20_05-14-50_` 的 `64 env / 500 iterations` 训练中，`success` 和 `safe_success` 始终为 `0`。同时 `shield_intervention_rate`、`dynamic_cbf_intervention_rate` 长期非零，最初判断可能是安全层过强、奖励稀疏或任务过难。

进一步查看训练曲线后，发现更基础的问题是 episode 极短：

```text
mean_episode_length 平均约 9 step
mean episode_duration 平均约 0.17 s
success = 0
safe_success = 0
timeout = 0
```

这说明失败不是“跑满时间但没有到达目标”，而是大量 episode 在刚开始就提前 reset，策略几乎拿不到有效的长轨迹样本。

### 2. 诊断结论

额外运行 reset 原因诊断后，发现 `contact_force > 50` 是主要 early reset 来源。原逻辑对所有刚体检查横向接触力：

```python
self.reset_buf |= torch.any(
    torch.norm(self.contact_forces[:, :, :2], dim=-1) > 50.0,
    dim=1,
)
```

这会把足端正常落地、摩擦或初始化冲击也当作 hard reset。诊断中曾出现：

```text
83 次 done 中 contact50 触发 83 次
其中 62 次发生在初始化阶段
目标、timeout、fall、dynamic collision 均不是主要 reset 原因
```

因此，500 iterations 训练失败的第一原因不是复杂动态避障本身，而是 reset 逻辑过严导致 episode 被切得过短。

### 3. 修复内容

#### 3.1 reset 原因日志

在 `legged_robot_pos_dynamic.py` 中新增 episode 级 reset 诊断字段：

```text
reset_goal
reset_stand_still
reset_timeout
reset_fall
reset_contact50
reset_initial_contact50
reset_spawn_collision
reset_terminate_contact
reset_dynamic_collision
```

这些字段会进入 `infos["episode"]`，从而被 `train_metrics.csv` 自动记录。

#### 3.2 hard contact reset 收窄

将 `contact50` hard reset 从“所有刚体”收窄为只检查 `termination_contact_indices`，即 base/head 等真正危险部位。

同时加入 warmup：

```python
hard_contact_warmup_steps = 10
```

初始化前几步的高接触力只记录为 `reset_initial_contact50`，不再直接 reset。

#### 3.3 安全层介入统计增强

新增 episode 平均介入量和二值介入比例：

```text
shield_intervention_mean
shield_intervention_step_rate
dynamic_cbf_intervention_mean
dynamic_cbf_intervention_step_rate
```

原来的 `shield_intervention_rate` 和 `dynamic_cbf_intervention_rate` 保留，用于兼容已有分析脚本。

#### 3.4 增加目标进展奖励

新增 dense reward：

```python
_reward_goal_progress = prev_distance - distance
```

并在 complex 默认配置中启用：

```python
goal_progress = 6.0
```

目的是让 PPO 在尚未到达目标前也能获得“是否朝目标靠近”的连续学习信号。

#### 3.5 complex 默认难度改为 Easy 起点

为了避免随机初始化策略直接进入 Hard 复杂动态场景，`go2_pos_dynamic_complex` 默认训练难度调整为：

```python
count_range = [2, 4]
speed_range = [0.15, 0.35]
enable_near_miss_replay = False
goal_reached_time = 20
```

动态障碍轨迹仍保持 episode 预生成模式，观测维度仍保持 `830`。

### 4. 验证结果

#### 4.1 静态检查

```text
python -m py_compile ... 通过
git diff --check 通过
```

#### 4.2 `16 env / 30 step` 冒烟

修复后关键输出：

```text
obs_shape (16, 830)
active_min 2
active_max 4
active_mean 3.0625
first_step_dones 0
done_total_30 0
episode_len_minmax 31 31
reset_contact50 0.0
reset_spawn_collision 0.0
reset_terminate_contact 0.0
reset_dynamic_collision 0.0
```

结论：

- 观测维度保持 `830`。
- Easy curriculum 下动态障碍数量符合 `2~4`。
- 首步不再大面积 reset。
- 30 step 内没有因为 `contact50`、spawn collision、dynamic collision 等原因提前 reset。
- 之前 `0.1~0.2 s` episode 的核心问题已修复。

#### 4.3 `8 env / 1 iteration` 训练闭环

运行后成功生成 checkpoint：

```text
training/legged_gym/logs/Go2_pos_dynamic_complex/06_20_07-17-40_/model_1.pt
```

训练启动时 reward 列表已包含：

```text
_reward_goal_progress: 6.00
```

说明新增奖励函数已正确接入 reward pipeline。

注意：Isaac Gym 退出阶段仍出现 `Segmentation fault (core dumped)`，返回码 `139`。由于 checkpoint 已保存，且前向、采样、PPO 更新和保存链路均已执行，该问题仍判断为 Isaac Gym 析构阶段问题，不是本次 Python 逻辑错误。

### 5. 下一轮训练建议

建议先不要直接恢复之前的 `06_20_05-14-50_` 结果继续训练，而是从当前修复后的 Easy complex 重新开始：

```bash
cd /home/sea_ws/src

WANDB_MODE=disabled \
PYTHONPATH=/home/sea_ws/src/training/legged_gym:/home/sea_ws/src/training/rsl_rl \
OMP_NUM_THREADS=8 \
python training/legged_gym/legged_gym/scripts/train.py \
  --task go2_pos_dynamic_complex \
  --num_envs 64 \
  --max_iterations 500 \
  --headless
```

重点观察：

```text
mean_episode_length 是否从个位数提升到数百步以上
reset_contact50 是否接近 0
reset_spawn_collision 是否接近 0
success / safe_success 是否开始非零
shield_intervention_step_rate 是否随训练下降
dynamic_cbf_intervention_step_rate 是否只在近动态障碍时上升
rew_goal_progress 是否为正并逐步稳定
```

若 Easy 阶段 success 能稳定抬升，再逐步恢复难度：

```text
Stage 1: 2-4 obstacles, 0.15-0.35 m/s, no near-miss replay
Stage 2: 3-5 obstacles, 0.20-0.45 m/s, enable TTC / near-miss reward
Stage 3: 5-8 obstacles, 0.25-0.55 m/s, enable near-miss replay
Stage 4: 6-10 obstacles, 0.35-0.65 m/s, full complex benchmark
```

---

## 六、Easy complex `64 env / 500 iterations` 训练结果

### 1. 训练设置

本轮按修复后的 Easy complex 配置重新训练：

```bash
cd /home/sea_ws/src

WANDB_MODE=disabled \
PYTHONPATH=/home/sea_ws/src/training/legged_gym:/home/sea_ws/src/training/rsl_rl \
OMP_NUM_THREADS=8 \
python training/legged_gym/legged_gym/scripts/train.py \
  --task go2_pos_dynamic_complex \
  --num_envs 64 \
  --max_iterations 500 \
  --headless
```

训练输出目录：

```text
training/legged_gym/logs/Go2_pos_dynamic_complex/06_20_07-23-45_
```

已保存 checkpoint：

```text
model_200.pt
model_300.pt
model_400.pt
model_500.pt
```

注意：训练结束后 Isaac Gym 清理阶段仍返回 `139`，但 `model_500.pt` 和 `train_metrics.csv` 已正常落盘。本次判断按“训练完成、退出析构阶段崩溃”处理。

### 2. 训练日志关键指标

`train_metrics.csv` 共记录 `48` 行，最后一条为 `iteration=490`。关键结果：

```text
last success:                       0.2917
last safe_success:                  0.2917
last mean_episode_length:         853.91
last dynamic_collision_count:       0.7083
last body_collision_count:          0.0000
last total_collision_count:         0.7083
last near_miss_count:              17.1250
last shield_intervention_step_rate: 0.9973
last dynamic_cbf_intervention_step_rate: 0.3016
last reset_goal:                    0.2917
last reset_timeout:                 0.0000
last reset_terminate_contact:       0.0000
last reset_dynamic_collision:       0.7083
```

训练过程中的最好窗口：

```text
max success:             1.0000 @ iteration 250
max safe_success:        1.0000 @ iteration 250
max mean_episode_length: 877.11 @ iteration 260
max reset_goal:          1.0000 @ iteration 250
```

最后 10 个记录窗口的平均值：

```text
success:                 0.3542
safe_success:            0.1208
mean_episode_length:   744.89
reset_goal:              0.3542
reset_dynamic_collision: 0.1479
reset_terminate_contact: 0.4635
```

结论：

- 和旧训练中 `success/safe_success` 长期为 `0` 相比，本轮已经明显学起来。
- episode length 从旧问题中的个位数/十几步提升到数百步，说明 early reset、contact 误触发和 spawn collision 问题已经基本解除。
- 训练仍然不稳定，不同窗口之间 `success` 和 `safe_success` 波动较大。
- `shield_intervention_step_rate` 仍接近 `1.0`，说明 nominal policy 大量动作仍需要安全层修正。
- 最后一条记录主要失败原因是 `reset_dynamic_collision=0.7083`，而最后 10 个窗口平均主要失败原因是 `reset_terminate_contact=0.4635`。

### 3. `model_500` 快速固定评估

固定评估命令：

```bash
cd /home/sea_ws/src/training/legged_gym

PYTHONPATH=/home/sea_ws/src/training/legged_gym:/home/sea_ws/src/training/rsl_rl \
python legged_gym/scripts/evaluate_dynamic.py \
  --task go2_pos_dynamic_complex \
  --load_run 06_20_07-23-45_ \
  --checkpoint 500 \
  --num_episodes 50 \
  --headless
```

由于单环境串行评估耗时较长，本次在完成 `40/50` 个 episode 后手动中断，已完成 episode 的快速统计如下：

```text
completed_episodes:             40
success_rate:                 0.4000
safe_success_rate:            0.2750
avg_dynamic_collision_count:  0.3250
avg_body_collision_count:     0.7000
avg_total_collision_count:    1.0250
timeout_rate:                 0.0250
avg_near_miss_count:         15.5250
avg_active_dynamic_count:     3.0750
```

结论：

- `model_500` 不是“完全不会走”的策略，固定评估中已有约 `40%` success。
- `safe_success` 仍明显低于 `success`，主要被动态碰撞和 body collision 拉低。
- 动态障碍数量平均约 `3.1`，符合当前 Easy 配置的 `2~4` 范围。
- 该 checkpoint 可以作为后续调参基线，但还不适合作为最终展示 checkpoint。

### 4. 是否符合预期

阶段性符合预期：

- 目标一：让 Easy complex 从 `success=0` 状态恢复到可学习状态，已达成。
- 目标二：修复首步/早期大面积 reset，已达成。
- 目标三：让 episode 能持续到数百步并产生有效 PPO 经验，已达成。

尚未符合最终预期：

- `success` 和 `safe_success` 还不稳定。
- `safe_success` 与 `success` 差距仍大。
- `shield_intervention_step_rate` 仍长期接近 `1.0`。
- 动态碰撞和 body collision 仍是主要失败来源。

下一步建议：

- 保留当前 Easy 难度，不急于升到 Medium。
- 先降低安全层对训练初期 nominal action 的强干预，或把 intervention loss / action penalty 调得更平滑。
- 增强“朝目标推进但保持动态余量”的 dense reward，减少只靠 CBF 硬拦截。
- 增加 checkpoint 选择逻辑，优先评估 `model_300.pt`、`model_400.pt`、`model_500.pt`，不要只看最后一个 checkpoint。
- 对 body collision 单独排查：区分是否来自静态障碍、墙体、地形边界或机器人自接触误计。

---

## 七、`model_500` 视频观察后的问题诊断与下一步优化方向

### 1. 观察到的主要问题

使用 `model_500` 录制 `10` 个 episode 后，视频中能看到两类比较典型的失败模式：

```text
1. 目标点已经在附近，但机器人绕了较大远路，不能稳定向目标收敛。
2. 动态障碍物有明显横向运动趋势时，机器人没有选择从障碍物运动方向的后方通过，反而向障碍物未来位置侧躲避，最终发生动态碰撞。
```

这两个问题不是单纯“训练不够久”，更像是当前 reward 和 safety layer 对“短路收敛”和“动态让行方向”的约束还不够明确。

### 2. 近目标绕远问题

当前相关机制：

- `goal_progress = 6.0` 已经提供距离下降奖励，但它只奖励 `prev_distance - distance`，没有显式惩罚绕圈、横向漂移或背离目标。
- `velo_dir = 4.0` 主要鼓励局部前向速度和目标方向对齐，但在复杂障碍和 CBF 频繁介入时，最终执行动作可能已经偏离 nominal action。
- `far_goal = distance > 0.5`，进入近目标区域后，部分 reward 会切换到 reach bonus，而不是继续强推“快速稳定停到目标点”。
- `goal_reached_time = 20` 要求连续保持到达条件，若机器人在目标附近仍被 CBF 或动态障碍扰动，容易出现擦边、绕圈和重新远离。
- 训练中 `shield_intervention_step_rate` 接近 `1.0`，说明策略输出经常被安全层改写；如果 nominal policy 没学会“安全地直奔目标”，视频里就会表现为明明目标近，却走出大弧线。

建议优先做三类调整：

```text
Priority A: 强化近目标收敛
- 将 goal_reached_time 从 20 降到 10~15 做一轮对照，减少近目标区域反复等待造成的绕行。
- 提高 reach_pos_target_tight 或 near-goal reach bonus，例如 10.0 -> 15.0。
- 新增 near_goal_stop_reward：distance < 0.6 时奖励低速度、低 yaw_rate、朝向目标的小动作，避免目标附近继续大幅横移。
```

```text
Priority B: 增加绕远惩罚
- 新增 negative_progress_penalty：当 distance - prev_distance > 0 时给惩罚，而不是只让 progress reward 变小。
- 新增 orbit_penalty：惩罚垂直于 goal direction 的速度分量，尤其在 distance < 1.5m 时更强。
- 新增 time_to_goal_penalty：每步小额负奖励，促使策略不要用很长绕路换取暂时安全。
```

```text
Priority C: 检查安全层是否把近目标动作改偏
- 在评估脚本中记录 goal_alignment、u_bar、u_static_safe、u_s 的夹角和范数。
- 如果近目标处 u_bar 指向目标、u_s 偏离目标，说明 CBF 约束太保守或 ray/dynamic token 风险估计过强。
- 如果 u_bar 本身就不指向目标，说明 PPO reward shaping 还不够，需要先改 reward，而不是先调 CBF。
```

建议第一轮只做轻量改动：

```text
goal_reached_time: 20 -> 12
reach_pos_target_tight: 10.0 -> 15.0
新增 negative_progress_penalty，scale 先取 -2.0
新增 orbit_penalty，scale 先取 -0.5 ~ -1.0，仅在 distance < 1.5m 时启用
```

### 3. 动态障碍让行方向错误问题

当前 dynamic token 已包含：

```text
[rel_x, rel_y, rel_vx, rel_vy, radius, ttc, valid]
```

但现有训练信号仍偏弱：

- Easy complex 中 `ttc_risk = -0.8`、`near_miss = -0.3`，相比基础动态任务的 `-2.0 / -1.0` 明显更温和。
- `ttc_risk` 只惩罚“快撞上”，没有奖励“从障碍物未来轨迹后方通过”。
- `near_miss` 主要看 clearance 和 min TTC，也不区分机器人是从障碍物前方抢行，还是从后方让行。
- dynamic CBF 当前是解析投影，主要沿 `rel_pos` 径向修正动作；它能把动作推出危险区，但不一定知道应该向哪一侧绕，尤其在横穿障碍物场景中容易产生“躲向障碍物未来位置”的问题。
- 当前 dynamic collision threshold 是 `0.65`，而 `DynamicTokenCBFLayer` 默认安全距离约为 `radius 0.32 + margin 0.20 = 0.52`。这意味着 CBF 认为尚可接受的位置，几何碰撞检测可能已经判为动态碰撞，需要优先对齐。

建议优先做四类调整：

```text
Priority A: 对齐 dynamic CBF 安全距离和碰撞阈值
- 将 DynamicTokenCBFLayer 的 safety_margin 从 0.20 提高到 0.35~0.45。
- 目标是让 d_safe >= dynamic_collision_threshold + 0.05。
- 以当前 radius=0.32、collision_threshold=0.65 估算，margin 至少应接近 0.38。
```

```text
Priority B: 增强 TTC / near-miss 惩罚，但保持 Easy 可学
- ttc_risk: -0.8 -> -1.2 或 -1.5。
- near_miss: -0.3 -> -0.5。
- ttc_threshold: 1.5 保持或略增到 1.8，给策略更早的风险信号。
- near_miss ttc_threshold: 0.8 -> 1.0。
```

```text
Priority C: 新增 crossing-side / pass-behind reward
- 对每个 top-K dynamic token 计算机器人速度相对障碍物未来轨迹的通过方向。
- 当障碍物横向穿越机器人前方时，奖励机器人选择从障碍物运动方向的后方通过。
- 对“抢到障碍物前方”并导致小 TTC 的动作给额外惩罚。
```

可实现的简化版 reward：

```text
rel_pos = obstacle_pos - robot_pos
rel_vel = obstacle_vel - robot_vel
closing = -dot(rel_pos, rel_vel)
side = cross(rel_vel, rel_pos)

若 closing > 0 且 ttc < threshold：
  惩罚机器人朝 obstacle future position 侧的横向速度
  奖励机器人选择使 predicted clearance 增大的侧向速度
```

工程上建议不要一开始写得太复杂，先做一个 `dynamic_avoid_direction_reward`：

```text
- 输入：dynamic_tokens、robot action 或 base velocity。
- 只对 ttc < 1.5 且 valid 的 token 生效。
- 奖励 predicted_min_clearance(t+0.4s) 比当前 clearance 增大。
- 惩罚 predicted_min_clearance(t+0.4s) 变小且机器人横向速度指向障碍物未来位置。
```

```text
Priority D: 让 CBF 不只“径向推开”，而是偏向更安全的通过侧
- 在 DynamicTokenCBFLayer 中加入未来位置约束：
  h_future = ||p_rel + v_rel * tau||^2 - d_safe^2
  tau 可先取 0.4 或 0.6。
- 对当前 h 和 h_future 都做约束，避免机器人选择会撞上未来障碍位置的动作。
- 或者在多个候选修正方向中选择 predicted clearance 最大的动作，而不是只选择 candidate_norm 最大的投影。
```

### 4. 推荐下一轮实验顺序

建议不要一次性把所有东西都改掉，否则不好判断到底是谁起作用。推荐按下面顺序做消融：

```text
Exp-1: near-goal 收敛修正
- goal_reached_time = 12
- reach_pos_target_tight = 15.0
- 加 negative_progress_penalty 和 orbit_penalty
- 保持 dynamic CBF 不变
观察：近目标绕远是否减少，success 是否提升。
```

```text
Exp-2: CBF 安全距离对齐
- DynamicTokenCBFLayer safety_margin = 0.38 或 0.40
- 保持 reward 不变
观察：dynamic_collision_count 是否下降，dynamic_cbf_intervention_rate 是否上升过多。
```

```text
Exp-3: TTC / near-miss 增强
- ttc_risk = -1.2
- near_miss = -0.5
- near_miss ttc_threshold = 1.0
观察：机器人是否更早减速或绕行，是否出现过度保守导致 timeout。
```

```text
Exp-4: pass-behind / future-clearance reward
- 新增 dynamic_avoid_direction_reward
- 只在 ttc < 1.5 的 top-K token 上启用
观察：横穿障碍物时，机器人是否开始选择从障碍物后方通过。
```

评价指标建议固定：

```text
success_rate
safe_success_rate
avg_dynamic_collision_count
avg_body_collision_count
avg_near_miss_count
avg_min_ttc
avg_min_dynamic_clearance
shield_intervention_step_rate
dynamic_cbf_intervention_step_rate
mean_time_to_goal
```

视频检查重点：

```text
1. 目标距离 < 1.5m 时，机器人是否还会大弧线绕行。
2. 横穿障碍物从右向左运动时，机器人是否避免继续向左抢行。
3. dynamic CBF 介入后，动作是减速/后让，还是把机器人推向障碍物未来位置。
4. 成功回合是否只是“运气好障碍少”，还是确实形成稳定让行动作。
```

### 5. 当前优先结论

下一步最值得先做的不是直接升 curriculum，而是在 Easy complex 内把两个行为补齐：

```text
1. 近目标阶段要更“贪心”地收敛，减少绕远。
2. 动态障碍阶段要更“预测式”地让行，优先从障碍物未来轨迹后方通过。
```

推荐第一批代码改动：

```text
- 增加 negative_progress_penalty。
- 增加 near-goal orbit_penalty。
- 将 DynamicTokenCBFLayer safety_margin 调到 0.38~0.40。
- 将 ttc_risk / near_miss 轻度增强。
- 增加日志：u_bar/u_static_safe/u_s 与 goal direction 的夹角，以及 future dynamic clearance。
```

完成这批后，再重新训练 `64 env / 500 iterations`，并对 `model_300/model_400/model_500` 都录制短视频，选择行为最稳定的 checkpoint，而不是默认使用最后一个模型。
