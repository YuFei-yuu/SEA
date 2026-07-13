# SEA-Nav 动态障碍物导航小规模快速验证报告

> 日期：2026-06-06 | 环境：AutoDL + RTX 4090 (24GB) + Ubuntu 20.04

---

## 一、验证目标

本次验证目标不是完成最终 3 个动态障碍物任务，而是快速确认以下几点：

- 当前实现的动态障碍物环境可以正常运行并录制视频。
- 原始静态模型在 `go2_pos_dynamic_1` 上的基线表现可被观察。
- 使用 `go2_pos_rough -> go2_pos_sparse_static -> go2_pos_dynamic_1` 的分阶段微调策略可行。
- 训练日志中能看到成功率、碰撞统计和动态碰撞惩罚开始起作用。

---

## 二、使用的命令

### 1. 基线录像：静态模型直接跑 1 个动态障碍物任务

```bash
conda activate sea_nav
cd /home/sea_ws/src

xvfb-run -a python training/legged_gym/legged_gym/scripts/play_record.py \
  --task go2_pos_dynamic_1 \
  --resume \
  --resume_experiment_name Go2_pos_rough \
  --load_run 06_06_03-46-25_ \
  --checkpoint 50
```

作用：不训练，直接观察静态 baseline 在动态场景中的表现。

### 2. 稀疏静态适应训练

```bash
conda activate sea_nav
cd /home/sea_ws/src

WANDB_MODE=disabled xvfb-run -a \
  python training/legged_gym/legged_gym/scripts/train.py \
  --task go2_pos_sparse_static \
  --headless \
  --resume \
  --resume_experiment_name Go2_pos_rough \
  --load_run 06_06_03-46-25_ \
  --checkpoint 50 \
  --num_envs 128 \
  --max_iterations 60
```

作用：先让策略适应“少量静态障碍物 + 新起终点采样”。

### 3. 动态障碍物微调训练

```bash
conda activate sea_nav
cd /home/sea_ws/src

WANDB_MODE=disabled xvfb-run -a \
  python training/legged_gym/legged_gym/scripts/train.py \
  --task go2_pos_dynamic_1 \
  --headless \
  --resume \
  --resume_experiment_name Go2_pos_sparse_static \
  --load_run 06_06_22-25-47_ \
  --checkpoint 110 \
  --num_envs 128 \
  --max_iterations 80
```

作用：在 1 个动态障碍物场景下做短程微调。

### 4. 微调后录像

```bash
conda activate sea_nav
cd /home/sea_ws/src

xvfb-run -a python training/legged_gym/legged_gym/scripts/play_record.py \
  --task go2_pos_dynamic_1 \
  --resume \
  --load_run 06_06_22-36-22_ \
  --checkpoint 190
```

作用：录制微调后模型在 `go2_pos_dynamic_1` 上的表现。

---

## 三、关键参数

| 项目 | 值 |
|---|---|
| 动态任务 | `go2_pos_dynamic_1` |
| 静态预适应任务 | `go2_pos_sparse_static` |
| 并行环境数 | `128` |
| 静态适应迭代数 | `60` |
| 动态微调迭代数 | `80` |
| 动态障碍物数量 | `1` |
| 动态障碍物尺寸 | `0.45 x 0.45 x 0.8` |
| 动态障碍物速度范围 | `[0.15, 0.40] m/s` |
| 起始静态 checkpoint | `Go2_pos_rough/06_06_03-46-25_/model_50.pt` |

---

## 四、训练过程中发现并修复的问题

### 问题：`go2_pos_dynamic_1` 在 128 env 训练时出现 GPU illegal memory access

现象：

- 1 env 推理和录像正常。
- 128 env 训练在 `env.reset()` 阶段崩溃。
- 报错表面出现在 `torch.randint(...)`，但根因不是随机采样本身。

根因分析：

- 动态环境 `LeggedRobotPosDynamic` 每个 env 有 `2` 个 actor：`1` 个机器人 + `1` 个动态障碍物。
- 原始 `_reset_dofs()` 沿用了单 actor 环境的写法，把 `env_ids` 直接传给 `set_dof_state_tensor_indexed`。
- 在多 actor 场景下，这会把 `env_id` 误当成 actor 索引，导致 DOF reset 写错对象并破坏 GPU 状态。

修复：

- 在 [legged_robot_pos_dynamic.py](/home/sea_ws/src/training/legged_gym/legged_gym/envs/base/legged_robot_pos_dynamic.py) 中重写 `_reset_dofs()`。
- 使用 `self.robot_actor_indices[env_ids]` 作为真正的 actor 索引传给 `set_dof_state_tensor_indexed`。

修复后结果：

- `128 env` 的 `env.reset()` 和首步 `env.step()` 已通过冒烟测试。
- `go2_pos_dynamic_1` 多环境训练可正常进行。

---

## 五、终端输出摘录

### 1. `go2_pos_sparse_static` 训练

关键日志：

```text
Loading model from: /home/sea_ws/src/training/legged_gym/logs/Go2_pos_rough/06_06_03-46-25_/model_50.pt
```

训练指标摘录：

| Iteration | Mean Reward | Success | Safe Success | Total Collision | Timeout | Time To Goal |
|---|---:|---:|---:|---:|---:|---:|
| 80 | 43.70 | 0.1042 | 0.0208 | 0.3542 | 0.3750 | 3.2358 |
| 100 | 64.99 | 0.3125 | 0.3125 | 0.6042 | 0.5000 | 4.9062 |

说明：

- 稀疏静态场景下，短时间训练后已经出现成功 episode。
- 说明从 `go2_pos_rough` 微调到稀疏静态任务是可行的。

### 2. `go2_pos_dynamic_1` 训练

关键日志：

```text
Loading model from: /home/sea_ws/src/training/legged_gym/logs/Go2_pos_sparse_static/06_06_22-25-47_/model_110.pt
```

训练指标摘录：

| Iteration | Mean Reward | Success | Safe Success | Dynamic Collision | Body Collision | Total Collision | Timeout | Time To Goal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 140 | 48.33 | 0.5417 | 0.2500 | 0.2500 | 0.2917 | 0.5417 | 0.2083 | 13.1717 |
| 150 | 59.33 | 0.6667 | 0.6667 | 0.3125 | 0.3333 | 0.6458 | 0.0000 | 13.3104 |
| 160 | 64.97 | 0.5208 | 0.5208 | 0.1042 | 0.4583 | 0.5625 | 0.0000 | 13.7725 |
| 170 | 57.08 | 0.6458 | 0.6458 | 0.0000 | 0.3542 | 0.3542 | 0.0000 | 17.7073 |
| 180 | 56.51 | 0.2361 | 0.2222 | 0.5000 | 0.2361 | 0.7361 | 0.0556 | 4.9079 |

说明：

- 短训后已经出现明显的成功率提升，说明“动态障碍物 actor + 动态 rays 融合 + 微调”这条路线可行。
- `dynamic_collision_count` 已经在训练日志中非零出现，说明动态碰撞事件统计和惩罚已经真正生效。
- 结果仍有波动，说明当前还处于“小规模可行性验证”阶段，尚未稳定到最终交付水准。

---

## 六、视频产物

### 1. 基线视频：未微调，直接用静态模型跑动态场景

| 文件 | 路径 | 大小 |
|---|---|---|
| baseline video | `training/legged_gym/logs/Go2_pos_dynamic_1/exported/06_06_03-46-25_model_50.mp4` | 65 MB |

关键终端输出：

```text
=== Episode 1/5 finished | success=0.00 dynamic_collision_count=0.00 ===
=== Episode 2/5 finished | success=0.00 dynamic_collision_count=0.00 ===
=== Episode 3/5 finished | success=0.00 dynamic_collision_count=0.00 ===
=== Episode 4/5 finished | success=0.00 dynamic_collision_count=0.00 ===
=== Episode 5/5 finished | success=0.00 dynamic_collision_count=0.00 ===
Video saved: .../06_06_03-46-25_model_50.mp4  (64.1 MB, 10009 frames)
```

结论：

- 纯静态模型在 `go2_pos_dynamic_1` 上几乎不能完成任务。
- 这证明“原模型结构可以复用，但原权重不能直接当最终答案”。

### 2. 微调后视频：`go2_pos_dynamic_1` 短训模型

| 文件 | 路径 | 大小 |
|---|---|---|
| finetuned video | `training/legged_gym/logs/Go2_pos_dynamic_1/exported/06_06_22-36-22_model_190.mp4` | 31 MB |

关键终端输出：

```text
=== Episode 1/5 finished | success=0.00 dynamic_collision_count=0.00 ===
=== Episode 2/5 finished | success=1.00 dynamic_collision_count=0.00 ===
=== Episode 3/5 finished | success=0.00 dynamic_collision_count=0.00 ===
=== Episode 4/5 finished | success=0.00 dynamic_collision_count=0.00 ===
=== Episode 5/5 finished | success=1.00 dynamic_collision_count=0.00 ===
Video saved: .../06_06_22-36-22_model_190.mp4  (30.9 MB, 4723 frames)
```

结论：

- 5 个 episode 中出现了 `2` 次成功，到达率约 `40%`。
- 相比 baseline 的 `0/5`，已经能作为“小规模可行性验证”的直接证据。

---

## 七、模型产物

| 模型 | 路径 | 大小 |
|---|---|---|
| 稀疏静态适应模型 | `training/legged_gym/logs/Go2_pos_sparse_static/06_06_22-25-47_/model_110.pt` | 11 MB |
| 动态 1 障碍物微调模型 | `training/legged_gym/logs/Go2_pos_dynamic_1/06_06_22-36-22_/model_190.pt` | 11 MB |

---

## 八、当前结论

本次小规模快速验证得到的结论是：

1. 当前动态障碍物环境实现是可用的。
- 动态障碍物可以正常出现在仿真中，并可通过录像观察。
- 动态障碍物已经被融合进原始 ray 观测中。
- 动态碰撞事件统计和惩罚已经真正进入训练信号。

2. 原始静态模型不能直接解决动态任务。
- baseline 视频 `0/5` 成功，说明仅靠静态训练出的权重不足以应对动态障碍物。

3. 分阶段微调路线可行。
- `go2_pos_rough -> go2_pos_sparse_static -> go2_pos_dynamic_1` 这条路线已经跑通。
- `dynamic_1` 短训阶段能观察到 `0.52 ~ 0.67` 的训练期成功率峰值。
- 微调后录像中已经出现成功 episode。

4. 还没达到最终交付标准。
- 当前只是 1 个动态障碍物的小规模验证。
- 成功率与碰撞统计仍有波动，还需要延长训练、做稳定评测，并继续扩展到 `dynamic_2` 和 `dynamic_3`。

---

## 九、下一步建议

建议后续按下面顺序继续：

1. 延长 `go2_pos_dynamic_1` 训练到 `200~400` iterations，并做固定 20/50 episode 的稳定评测。
2. 在 `dynamic_1` 稳定后，再启动 `go2_pos_dynamic_2` 微调。
3. 最后再上 `go2_pos_dynamic_3`，并录制正式 demo。
4. 如果结果不稳，优先增加训练时长和评测次数，不优先改网络结构。
