# SEA-Nav 训练与 Demo 报告

> 日期：2026-06-06 | 环境：AutoDL + RTX 4090 (24GB) + Ubuntu 20.04

---

## 一、训练命令

```bash
# 激活环境
conda activate sea_nav
cd /home/sea_ws/src

# 训练
WANDB_MODE=disabled xvfb-run -a \
  python training/legged_gym/legged_gym/scripts/train.py \
  --headless --num_envs 512 --max_iterations 50
```

### 参数说明

| 参数 | 值 | 说明 |
|---|---|---|
| `--headless` | (flag) | 无 GUI 模式，Isaac Gym 不创建 viewer |
| `--num_envs` | 512 | 并行仿真环境数（默认 2048，此处降低以保证稳定性） |
| `--max_iterations` | 50 | PPO 训练迭代数（默认 2000，此处为快速验证） |
| `--task` | `go2_pos_rough` | 任务名称（默认值），对应 Go2 四足机器人在障碍物房间中的导航 |
| `WANDB_MODE=disabled` | (env) | 禁用 wandb 日志（无需注册账号） |
| `xvfb-run -a` | (wrapper) | 提供虚拟 X 显示（Isaac Gym camera sensor 在 headless 模式下需要） |

### 关键配置（Go2PosRoughCfg）

| 配置项 | 值 | 说明 |
|---|---|---|
| 地形类型 | `hard_room` | 带随机障碍物的室内房间 |
| 地形课程 | 10 级 | 难度递增的障碍物密度 |
| Episode 长度 | 40 秒 | 每个 episode 最大时长 |
| Decimation | 4 | 每步策略执行 4 次 PD 控制 |
| 控制频率 | 50 Hz | 策略输出导航指令的频率 |
| 观测 | 550 维 | 41 rays + 12 props + 2 goal, history=10 |
| 导航动作 | 3 维 | [vx, vy, ω_yaw] 目标速度 |
| 关节动作 | 12 维 | 12 个关节的位置目标 |

---

## 二、终端输出（训练日志）

### 启动日志

```
+++ Using GPU PhysX
Physics Engine: PhysX
Physics Device: cuda:0
GPU Pipeline: enabled
Setting seed: 1
hard_room col: 0 : 10
generating curriculum terrains hard_room
 generated all curriculum terrains!
```

### 奖励函数配置

```
              _reward_ang_vel_xy: -0.05       # 角速度惩罚
          _reward_close_obst_vel:  5.00       # 靠近障碍物时的安全速度奖励
               _reward_collision: -4.00       # 碰撞惩罚
  _reward_reach_pos_target_tight: 10.00       # 到达目标奖励
                   _reward_stuck: -5.00       # 卡住/死胡同惩罚
                _reward_velo_dir:  4.00       # 朝目标方向运动奖励
```

### 训练指标（每 10 iterations）

| Iteration | Value Loss | Surrogate Loss | Mean Reward | 说明 |
|---|---|---|---|---|
| 20 | 0.7061 | -0.0018 | 3.38 | 训练初期，策略开始学习 |
| 30 | 0.7072 | -0.0028 | 7.85 | Mean reward 明显上升 |
| 40 | 1.2609 | -0.0032 | 9.01 | 策略持续改善 |

> Mean reward 从 3.38 提升到 9.01，表明策略在 50 iterations (512 envs, 48 steps/env) 后已学习到基本的导航行为。

---

## 三、训练产物

| 产物 | 路径 | 大小 |
|---|---|---|
| 模型 checkpoint | `training/legged_gym/logs/Go2_pos_rough/06_06_03-46-25_/model_50.pt` | 11 MB |

---

## 四、Demo 视频录制

### 命令

```bash
conda activate sea_nav
cd /home/sea_ws/src

# 录制 5 个 episode 的导航视频
xvfb-run -a python training/legged_gym/legged_gym/scripts/play_record.py
```

### 视频输出

```
Video saved: .../logs/Go2_pos_rough/exported/06_06_03-46-25_model_50.mp4
Recording: 6537 frames, 5 episodes, 30 FPS, 1000×1000, 91 MB
```

### 视频说明

- 相机从斜上方跟随机器人移动（offset: 4m x, -4m y, 3m z）
- 可见：机器人模型、障碍物房间、蓝色目标点标记、青色导航射线
- 每个 episode 约 40 秒（模拟时间），机器人自主导航至随机目标

---

## 五、环境软件版本

| 组件 | 版本 |
|---|---|
| Python | 3.8.20 (conda) |
| PyTorch | 2.0.1+cu118 |
| CUDA Toolkit | 11.8 (conda-forge) |
| NVIDIA Driver | 570.124.04 |
| Isaac Gym | Preview 4 |
| rsl_rl | 1.0.2 |
| legged_gym | 1.0.0 |
| NumPy | 1.24.4 |
| OpenCV | 4.13.0 |

## 六、GPU 信息

```
GPU: NVIDIA GeForce RTX 4090 (24 GB)
CUDA Capability: sm_89 (Ada Lovelace)
PhysX: GPU mode (sm_70 PTX → JIT compiled for sm_89)
```
