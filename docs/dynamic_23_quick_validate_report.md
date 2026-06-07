# SEA-Nav 动态障碍物导航 2/3 障碍物小规模快速验证报告

> 日期：2026-06-06 | 环境：AutoDL + RTX 4090 (24GB) + Ubuntu 20.04

---

## 一、验证目标

在已经完成 `go2_pos_dynamic_1` 小规模验证的基础上，继续做：

- `go2_pos_dynamic_2` 的短程微调与录像验证
- `go2_pos_dynamic_3` 的短程微调与录像验证

本次目标仍然是验证“方案可行”，不是最终达到课程项目的稳定指标。

---

## 二、验证前检查

先做了 `128 env` 的冒烟测试，确认两个任务都能正常 `reset/step`：

- `go2_pos_dynamic_2 (128, 550) 2 1`
- `go2_pos_dynamic_3 (128, 550) 3 2`

含义：

- 观测维度保持 `550`
- `dynamic_count` 分别为 `2` 和 `3`
- 首步 `done_sum` 很小，说明环境可正常运行

---

## 三、使用的命令

### 1. `go2_pos_dynamic_2` 训练

```bash
conda activate sea_nav
cd /home/sea_ws/src

WANDB_MODE=disabled xvfb-run -a \
  python training/legged_gym/legged_gym/scripts/train.py \
  --task go2_pos_dynamic_2 \
  --headless \
  --resume \
  --resume_experiment_name Go2_pos_dynamic_1 \
  --load_run 06_06_22-36-22_ \
  --checkpoint 190 \
  --num_envs 128 \
  --max_iterations 60
```

### 2. `go2_pos_dynamic_2` 录像

```bash
conda activate sea_nav
cd /home/sea_ws/src

xvfb-run -a python training/legged_gym/legged_gym/scripts/play_record.py \
  --task go2_pos_dynamic_2 \
  --resume \
  --load_run 06_06_23-15-21_ \
  --checkpoint 250
```

### 3. `go2_pos_dynamic_3` 训练

```bash
conda activate sea_nav
cd /home/sea_ws/src

WANDB_MODE=disabled xvfb-run -a \
  python training/legged_gym/legged_gym/scripts/train.py \
  --task go2_pos_dynamic_3 \
  --headless \
  --resume \
  --resume_experiment_name Go2_pos_dynamic_2 \
  --load_run 06_06_23-15-21_ \
  --checkpoint 250 \
  --num_envs 128 \
  --max_iterations 60
```

### 4. `go2_pos_dynamic_3` 录像

```bash
conda activate sea_nav
cd /home/sea_ws/src

xvfb-run -a python training/legged_gym/legged_gym/scripts/play_record.py \
  --task go2_pos_dynamic_3 \
  --resume \
  --load_run 06_06_23-18-12_ \
  --checkpoint 310
```

---

## 四、关键参数

| 项目 | `dynamic_2` | `dynamic_3` |
|---|---:|---:|
| 并行环境数 | 128 | 128 |
| 训练迭代数 | 60 | 60 |
| 动态障碍物数量 | 2 | 3 |
| 起始 checkpoint | `dynamic_1/model_190.pt` | `dynamic_2/model_250.pt` |

---

## 五、训练日志摘录

### 1. `go2_pos_dynamic_2`

关键日志：

```text
Loading model from: /home/sea_ws/src/training/legged_gym/logs/Go2_pos_dynamic_1/06_06_22-36-22_/model_190.pt
```

训练指标摘录：

| Iteration | Mean Reward | Success | Safe Success | Dynamic Collision | Body Collision | Total Collision | Timeout | Time To Goal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 210 | 36.81 | 0.3958 | 0.3750 | 0.1875 | 0.3750 | 0.5625 | 0.2708 | 11.3700 |
| 230 | 44.19 | 0.8125 | 0.1667 | 0.1875 | 0.6458 | 0.8333 | 0.0000 | 17.0096 |
| 240 | 54.05 | 0.1458 | 0.0000 | 0.0417 | 0.7708 | 0.8125 | 0.6458 | 4.8300 |

说明：

- 在 2 个动态障碍物下，策略已经能明显学会到达目标。
- 但训练期指标波动较大，尤其是 `safe_success` 和 `total_collision_count`，说明当前还没有收敛稳定。

### 2. `go2_pos_dynamic_3`

关键日志：

```text
Loading model from: /home/sea_ws/src/training/legged_gym/logs/Go2_pos_dynamic_2/06_06_23-15-21_/model_250.pt
```

训练指标摘录：

| Iteration | Mean Reward | Success | Safe Success | Dynamic Collision | Body Collision | Total Collision | Timeout | Time To Goal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 270 | 37.25 | 0.1875 | 0.1875 | 0.4583 | 0.2917 | 0.7500 | 0.2500 | 3.8125 |
| 280 | 44.29 | 0.4583 | 0.4583 | 0.2083 | 0.3333 | 0.5417 | 0.0833 | 11.0700 |
| 290 | 50.29 | 0.6875 | 0.6875 | 0.1042 | 0.2708 | 0.3750 | 0.0000 | 10.3996 |
| 300 | 49.18 | 0.5833 | 0.5833 | 0.3125 | 0.1042 | 0.4167 | 0.0000 | 10.3950 |

说明：

- 在 3 个动态障碍物下，短训也已经出现了 `0.58 ~ 0.69` 的训练期成功率。
- 相比 `dynamic_2`，动态碰撞更敏感，但训练趋势仍然说明这条路线可继续扩展。

---

## 六、视频产物

### 1. `go2_pos_dynamic_2` 微调后视频

| 文件 | 路径 | 大小 |
|---|---|---|
| finetuned video | `training/legged_gym/logs/Go2_pos_dynamic_2/exported/06_06_23-15-21_model_250.mp4` | 21 MB |

关键终端输出：

```text
=== Episode 1/5 finished | success=1.00 dynamic_collision_count=0.00 ===
=== Episode 2/5 finished | success=1.00 dynamic_collision_count=0.00 ===
=== Episode 3/5 finished | success=1.00 dynamic_collision_count=0.00 ===
=== Episode 4/5 finished | success=1.00 dynamic_collision_count=0.00 ===
=== Episode 5/5 finished | success=1.00 dynamic_collision_count=0.00 ===
Video saved: .../06_06_23-15-21_model_250.mp4  (20.3 MB, 2877 frames)
```

结论：

- 2 个动态障碍物的小规模验证效果很好。
- 录像中 `5/5` 全成功，可以作为当前阶段最强的正面证据。

### 2. `go2_pos_dynamic_3` 微调后视频

| 文件 | 路径 | 大小 |
|---|---|---|
| finetuned video | `training/legged_gym/logs/Go2_pos_dynamic_3/exported/06_06_23-18-12_model_310.mp4` | 34 MB |

关键终端输出：

```text
=== Episode 1/5 finished | success=1.00 dynamic_collision_count=0.00 ===
=== Episode 2/5 finished | success=1.00 dynamic_collision_count=0.00 ===
=== Episode 3/5 finished | success=0.00 dynamic_collision_count=1.00 ===
=== Episode 4/5 finished | success=1.00 dynamic_collision_count=0.00 ===
=== Episode 5/5 finished | success=0.00 dynamic_collision_count=1.00 ===
Video saved: .../06_06_23-18-12_model_310.mp4  (33.4 MB, 4789 frames)
```

结论：

- 3 个动态障碍物的小规模验证中，`5` 个 episode 有 `3` 个成功、`2` 个因动态碰撞失败。
- 这已经达到“方案可行、能继续优化”的阶段，但还没到最终汇报要求的稳定水平。

---

## 七、模型产物

| 模型 | 路径 | 大小 |
|---|---|---|
| `dynamic_2` 微调模型 | `training/legged_gym/logs/Go2_pos_dynamic_2/06_06_23-15-21_/model_250.pt` | 11 MB |
| `dynamic_3` 微调模型 | `training/legged_gym/logs/Go2_pos_dynamic_3/06_06_23-18-12_/model_310.pt` | 11 MB |

---

## 八、当前阶段结论

结合 `dynamic_1`、`dynamic_2`、`dynamic_3` 的小规模验证，可以得到：

1. 方案已经连续扩展到 `1 -> 2 -> 3` 个动态障碍物，并且训练链路都能跑通。
2. `dynamic_2` 的视频表现已经比较强，`5/5` 全成功，说明当前方案在双障碍物场景下已经具备较好的可用性。
3. `dynamic_3` 的视频表现为 `3/5` 成功，说明三障碍物任务已经具备可行性，但稳定性和防撞性还需要进一步训练。
4. 当前最需要的不是重做算法，而是：
- 延长 `dynamic_3` 训练时长
- 固定随机种子做更正式的 20/50 episode 评测
- 录制若干挑选过的成功案例作为最终 demo

---

## 九、建议的下一步

建议接下来直接做：

1. 把 `go2_pos_dynamic_3` 训练从当前短训延长到 `200~400` iterations。
2. 单独写一个稳定的评测脚本，不依赖当前 `evaluate_dynamic.py`，固定 20/50 episode 输出成功率和碰撞次数。
3. 在 `dynamic_3` 上挑选 `3~5` 个成功 episode，录制最终提交视频。
4. 如果导师要求更高稳定性，再考虑轻微降低障碍物速度上限或增加训练时长，而不是先改网络结构。
