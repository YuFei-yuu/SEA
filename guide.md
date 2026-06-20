# SEA-Nav 复现、训练、评估与可视化指南

> 适用环境：Ubuntu 20.04 + Python 3.8 + Isaac Gym Preview 4 + NVIDIA GPU
> 当前重点任务：`go2_pos_dynamic_complex` 多动态障碍物复杂环境安全导航

---

## 一、目录与任务速览

### 1. 项目路径

```bash
/home/sea_ws/src
├── training/legged_gym      # Isaac Gym 环境、任务配置、训练/评估/录像脚本
├── training/rsl_rl          # PPO、Actor-Critic、CBF safety layer
├── docs                     # 调研、周报、实验记录
└── guide.md                 # 本指南
```

### 2. 常用任务名

| 任务 | 说明 |
|---|---|
| `go2_pos_rough` | 原始静态导航任务 |
| `go2_pos_sparse_static` | 稀疏静态房间 |
| `go2_pos_dynamic_1/2/3` | 简单动态障碍物任务 |
| `go2_pos_dynamic_complex` | 当前主线：复杂静态结构 + 多动态障碍物 |

### 3. 推荐通用环境变量

后续命令默认在 `/home/sea_ws/src` 下执行：

```bash
cd /home/sea_ws/src
conda activate sea_nav

export PYTHONPATH=/home/sea_ws/src/training/legged_gym:/home/sea_ws/src/training/rsl_rl
export WANDB_MODE=disabled
export OMP_NUM_THREADS=8
```

---

## 二、环境配置

### 1. 基础要求

| 组件 | 推荐配置 | 备注 |
|---|---|---|
| OS | Ubuntu 20.04 LTS | Isaac Gym 主要支持 Linux |
| Python | 3.8 | Isaac Gym Preview 4 要求 `<3.9` |
| GPU | RTX 4090 或类似 NVIDIA GPU | 推荐 24GB 显存 |
| CUDA | 11.8 | RTX 4090 推荐 CUDA 11.8 |
| PyTorch | 2.0.1 + cu118 | 与 Isaac Gym/gymtorch 较稳定 |

### 2. 创建 Conda 环境

```bash
conda create -n sea_nav python=3.8 -y
conda activate sea_nav
```

国内环境可先配置镜像：

```bash
cat > ~/.condarc << 'EOF'
channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
show_channel_urls: true
EOF
```

### 3. 系统依赖

```bash
apt-get update
apt-get install -y build-essential libpython3.8-dev libgl1-mesa-glx xvfb
```

### 4. CUDA 与 PyTorch

```bash
conda install -c conda-forge cudatoolkit=11.8 -y

pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install torch==2.0.1 torchvision==0.15.2 \
  --index-url https://download.pytorch.org/whl/cu118
```

验证：

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### 5. Isaac Gym Preview 4

```bash
cd /tmp
tar xzf /home/sea_ws/src/IsaacGym_Preview_4_Package.tar.gz
cd isaacgym/python
pip install -e .
```

若 NumPy 版本较新，修复 `np.float`：

```bash
sed -i 's/dtype=np\.float,/dtype=np.float64,/' /tmp/isaacgym/python/isaacgym/torch_utils.py
```

### 6. 安装本地包

```bash
cd /home/sea_ws/src/training/rsl_rl
pip install -e .

cd /home/sea_ws/src/training/legged_gym
pip install -e .
```

### 7. 其他 Python 依赖

```bash
pip install wandb matplotlib opencv-python scipy pyyaml pillow imageio ninja
```

当前可视化脚本只强依赖标准库和 `matplotlib`，不要求安装 `pandas`。

---

## 三、当前代码中的关键修复

这些修复当前仓库已经包含，后续复用时只需确认没有被回退。

### 1. TorchScript 模型加载到 GPU

`legged_robot_pos.py` 中 `torch.jit.load(...)` 需要使用 `map_location=self.device`，避免 JIT 模型在 CPU、Isaac Gym 张量在 GPU 时冲突。

### 2. Headless 渲染同步

`base_task.py` 的 `render()` 中需要先执行：

```python
self.gym.step_graphics(self.sim)
self.gym.render_all_camera_sensors(self.sim)
```

否则 headless 录像时 camera sensor 可能停留在初始帧。

### 3. `train.py` 配置打印简化

训练入口已将 `config = print_config()` 简化为：

```python
config = {}
```

避免部分环境下 `class_to_dict()` 打印配置时崩溃。

---

## 四、训练方式

### 1. 最小冒烟训练

用于确认环境、PPO、checkpoint 保存链路正常：

```bash
cd /home/sea_ws/src
export PYTHONPATH=/home/sea_ws/src/training/legged_gym:/home/sea_ws/src/training/rsl_rl
export WANDB_MODE=disabled
export OMP_NUM_THREADS=8

python training/legged_gym/legged_gym/scripts/train.py \
  --task go2_pos_dynamic_complex \
  --num_envs 8 \
  --max_iterations 1 \
  --headless
```

### 2. 当前推荐训练命令

用于 Easy complex 阶段的主要实验：

```bash
cd /home/sea_ws/src
export PYTHONPATH=/home/sea_ws/src/training/legged_gym:/home/sea_ws/src/training/rsl_rl
export WANDB_MODE=disabled
export OMP_NUM_THREADS=8

python training/legged_gym/legged_gym/scripts/train.py \
  --task go2_pos_dynamic_complex \
  --num_envs 64 \
  --max_iterations 500 \
  --headless
```

训练产物会保存到：

```text
training/legged_gym/logs/Go2_pos_dynamic_complex/<MM_DD_HH-MM-SS_>/
```

常见文件：

| 文件 | 说明 |
|---|---|
| `model_*.pt` | checkpoint |
| `train_metrics.csv` | 训练曲线数据 |
| `analysis/` | 训练分析图表和报告 |

### 3. 训练后自动生成可视化报告

如果希望训练结束后自动画图：

```bash
python training/legged_gym/legged_gym/scripts/train.py \
  --task go2_pos_dynamic_complex \
  --num_envs 64 \
  --max_iterations 500 \
  --headless \
  --analyze_after_train
```

如果还希望训练后自动评估多个 checkpoint：

```bash
python training/legged_gym/legged_gym/scripts/train.py \
  --task go2_pos_dynamic_complex \
  --num_envs 64 \
  --max_iterations 500 \
  --headless \
  --analyze_after_train \
  --eval_checkpoints_after_train 300 400 500
```

说明：

- `--analyze_after_train` 会调用 `analyze_dynamic_training.py`。
- `--eval_checkpoints_after_train 300 400 500` 会调用 `evaluate_checkpoints.py`。
- 自动 checkpoint 评估默认每个 checkpoint 跑 `50 episodes`。
- Isaac Gym 退出阶段可能出现 `139` 或 `-11`，只要 checkpoint、CSV、图表已经落盘，通常按“训练/评估完成，退出清理阶段崩溃”处理。

---

## 五、训练指标与可视化报告

### 1. 训练过程记录的数据

`train_metrics.csv` 现在会记录：

| 类别 | 典型字段 |
|---|---|
| RL 基础指标 | `timesteps`, `fps`, `mean_reward`, `mean_episode_length`, `mean_action_std` |
| PPO 优化指标 | `value_loss`, `surrogate_loss`, `regularization_loss`, `smooth_loss`, `intervention_loss` |
| 成功率 | `success`, `safe_success`, `timeout`, `time_to_goal` |
| 安全事件 | `dynamic_collision_count`, `body_collision_count`, `total_collision_count`, `near_miss_count` |
| 动态风险 | `min_ttc`, `min_dynamic_clearance`, `active_dynamic_count` |
| CBF 介入 | `shield_intervention_rate`, `shield_intervention_step_rate`, `dynamic_cbf_intervention_rate` |
| reset 原因 | `reset_goal`, `reset_timeout`, `reset_contact50`, `reset_dynamic_collision` 等 |
| reward 分项 | 所有 `rew_*` 字段 |

### 2. 对已有 run 生成训练报告

不需要重新训练，已有 `train_metrics.csv` 即可画图：

```bash
cd /home/sea_ws/src
export PYTHONPATH=/home/sea_ws/src/training/legged_gym:/home/sea_ws/src/training/rsl_rl

python training/legged_gym/legged_gym/scripts/analyze_dynamic_training.py \
  --run_dir training/legged_gym/logs/Go2_pos_dynamic_complex/06_20_07-23-45_
```

输出目录：

```text
training/legged_gym/logs/Go2_pos_dynamic_complex/<run>/analysis/
```

主要输出：

| 文件 | 内容 |
|---|---|
| `report.md` | 可直接打开/截图的训练报告 |
| `train_summary.txt` | 关键数字摘要 |
| `rl_reward_length.png` | reward、episode length、time-to-goal |
| `rl_success_rates.png` | success、safe success、timeout |
| `rl_optimization.png` | value/surrogate/intervention/action std |
| `safety_events.png` | collision、near-miss |
| `safety_clearance_ttc.png` | TTC、动态余量、动态障碍数量 |
| `cbf_intervention.png` | safety layer 介入强度 |
| `reset_reasons_stacked.png` | reset 原因堆叠图 |
| `reward_breakdown.png` | reward 分项贡献 |
| `motion_mix.png` | 动态障碍物运动模式统计 |
| `throughput.png` | FPS 和训练耗时 |

---

## 六、模型评估

### 1. 单 checkpoint 固定评估

```bash
cd /home/sea_ws/src
export PYTHONPATH=/home/sea_ws/src/training/legged_gym:/home/sea_ws/src/training/rsl_rl

python training/legged_gym/legged_gym/scripts/evaluate_dynamic.py \
  --task go2_pos_dynamic_complex \
  --load_run 06_20_07-23-45_ \
  --checkpoint 500 \
  --num_episodes 50 \
  --headless
```

如果希望把评估结果写入文件：

```bash
python training/legged_gym/legged_gym/scripts/evaluate_dynamic.py \
  --task go2_pos_dynamic_complex \
  --load_run 06_20_07-23-45_ \
  --checkpoint 500 \
  --num_episodes 50 \
  --headless \
  --output_csv training/legged_gym/logs/Go2_pos_dynamic_complex/06_20_07-23-45_/analysis/eval_500.csv \
  --output_summary training/legged_gym/logs/Go2_pos_dynamic_complex/06_20_07-23-45_/analysis/eval_500_summary.txt
```

### 2. 多 checkpoint 对比评估

推荐不要只看最后一个模型，而是比较 `model_300.pt`、`model_400.pt`、`model_500.pt`：

```bash
cd /home/sea_ws/src
export PYTHONPATH=/home/sea_ws/src/training/legged_gym:/home/sea_ws/src/training/rsl_rl
export OMP_NUM_THREADS=8

python training/legged_gym/legged_gym/scripts/evaluate_checkpoints.py \
  --task go2_pos_dynamic_complex \
  --load_run 06_20_07-23-45_ \
  --checkpoints 300 400 500 \
  --num_episodes 50 \
  --headless
```

输出文件位于对应 run 的 `analysis/`：

| 文件 | 内容 |
|---|---|
| `eval_checkpoints.csv` | 多 checkpoint 指标表 |
| `eval_checkpoints_summary.md` | checkpoint 排名和推荐 |
| `eval_success_collision_bars.png` | success/collision 柱状图 |
| `eval_safety_metrics_bars.png` | near-miss/TTC/clearance/CBF 柱状图 |
| `eval_efficiency_bars.png` | timeout/duration/time-to-goal 柱状图 |

推荐模型排序逻辑：

```text
safe_success_rate 越高越好
success_rate 越高越好
dynamic_collision_count 越低越好
near_miss_count 越低越好
mean_time_to_goal 越低越好
```

---

## 七、视频录制与人工检查

### 1. 录制策略视频

```bash
cd /home/sea_ws/src
export PYTHONPATH=/home/sea_ws/src/training/legged_gym:/home/sea_ws/src/training/rsl_rl

xvfb-run -a python training/legged_gym/legged_gym/scripts/play_record.py \
  --task go2_pos_dynamic_complex \
  --load_run 06_20_07-23-45_ \
  --checkpoint 500 \
  --num_episodes 10 \
  --headless
```

输出目录：

```text
training/legged_gym/logs/Go2_pos_dynamic_complex/exported/
```

### 2. 视频检查重点

| 场景 | 观察问题 |
|---|---|
| 近目标阶段 | 是否在目标附近绕远、不收敛 |
| 横穿动态障碍物 | 是否从障碍物未来轨迹后方通过 |
| CBF 介入后 | 动作是减速/后让，还是被推向未来碰撞位置 |
| 成功 episode | 是否是真实稳定策略，而不是障碍物少导致的偶然成功 |

---

## 八、推荐实验流程

### 1. 快速 sanity check

```bash
python training/legged_gym/legged_gym/scripts/train.py \
  --task go2_pos_dynamic_complex \
  --num_envs 8 \
  --max_iterations 1 \
  --headless
```

确认：

- checkpoint 正常保存。
- 没有首步大面积 reset。
- `obs_shape` 仍为 `830`。

### 2. 正式训练

```bash
python training/legged_gym/legged_gym/scripts/train.py \
  --task go2_pos_dynamic_complex \
  --num_envs 64 \
  --max_iterations 500 \
  --headless \
  --analyze_after_train
```

### 3. 评估多个 checkpoint

```bash
python training/legged_gym/legged_gym/scripts/evaluate_checkpoints.py \
  --task go2_pos_dynamic_complex \
  --load_run <run_name> \
  --checkpoints 300 400 500 \
  --num_episodes 50 \
  --headless
```

### 4. 录制最佳 checkpoint

```bash
xvfb-run -a python training/legged_gym/legged_gym/scripts/play_record.py \
  --task go2_pos_dynamic_complex \
  --load_run <run_name> \
  --checkpoint <best_checkpoint> \
  --num_episodes 10 \
  --headless
```

### 5. 汇报材料建议

每轮实验建议至少保存并截图：

- `analysis/report.md`
- `analysis/rl_success_rates.png`
- `analysis/safety_events.png`
- `analysis/safety_clearance_ttc.png`
- `analysis/cbf_intervention.png`
- `analysis/reset_reasons_stacked.png`
- `analysis/eval_checkpoints_summary.md`
- 最佳 checkpoint 的策略视频

---

## 九、常见问题

### 1. Isaac Gym 退出时 `Segmentation fault` 或返回码 `139/-11`

如果 checkpoint、`train_metrics.csv`、评估 CSV、图表已经正常落盘，一般可按“Isaac Gym 清理阶段崩溃”处理。当前 `evaluate_checkpoints.py` 已对评估子进程的 `139/-11` 做兼容，只要结果 CSV 已写出就继续汇总。

### 2. RTX 4090 与 PhysX GPU

Isaac Gym Preview 4 的 PhysX GPU 内核较旧，在 RTX 4090 上可运行但有崩溃风险。若频繁崩溃：

- 降低 `--num_envs`，如 64、128、256。
- 或尝试 CPU PhysX，但训练会明显变慢。

### 3. `gymtorch` 首次加载很慢

首次 import 会 JIT 编译 C++ 扩展，通常需要 1 到 2 分钟。缓存目录：

```text
~/.cache/torch_extensions/
```

### 4. `pandas` 不存在

当前可视化脚本已经不依赖 `pandas`，只要有 `matplotlib` 即可：

```bash
python -c "import matplotlib; print(matplotlib.__version__)"
```

### 5. 无 GUI 服务器无法录像

使用 `xvfb-run -a` 包住 `play_record.py`：

```bash
xvfb-run -a python training/legged_gym/legged_gym/scripts/play_record.py ...
```

---

## 十、关键脚本索引

| 脚本 | 作用 |
|---|---|
| `training/legged_gym/legged_gym/scripts/train.py` | PPO 训练入口，支持训练后分析和 checkpoint 评估 |
| `training/legged_gym/legged_gym/scripts/analyze_dynamic_training.py` | 从 `train_metrics.csv` 生成训练图表和报告 |
| `training/legged_gym/legged_gym/scripts/evaluate_dynamic.py` | 单 checkpoint 固定评估 |
| `training/legged_gym/legged_gym/scripts/evaluate_checkpoints.py` | 多 checkpoint 对比评估和柱状图 |
| `training/legged_gym/legged_gym/scripts/play_record.py` | 策略视频录制 |
| `training/legged_gym/legged_gym/envs/go2/go2_pos_config.py` | Go2 任务配置、奖励权重、动态障碍参数 |
| `training/legged_gym/legged_gym/envs/base/legged_robot_pos_dynamic.py` | 动态障碍环境、观测、奖励、reset、统计指标 |
| `training/rsl_rl/rsl_rl/modules/cbf_lse_layer.py` | static/dynamic CBF safety layer |
| `training/rsl_rl/rsl_rl/modules/cbf_actor_critic.py` | PPO policy + CBF actor-critic |
