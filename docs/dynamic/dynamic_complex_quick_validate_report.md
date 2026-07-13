# SEA-Nav 复杂多动态障碍物环境小规模快速验证报告

> 日期：2026-06-20 | 环境：RTX 4090 (24GB) + Ubuntu 20.04

---

## 一、验证目标

在完成 `go2_pos_dynamic_1/2/3` 的小规模验证后，进一步对新实现的高密度复杂动态障碍物任务 `go2_pos_dynamic_complex` 做一轮快速验证，重点确认：

- 新环境是否能正常 `reset/step`
- 动态障碍物数量是否基本达到预期的 `6~10`
- 运动模式是否已经从单一往返直线扩展为更复杂轨迹
- 当前训练链路和录像链路是否已经具备继续推进的条件

本次目标是“快速确认复杂环境已经基本成形，并找出当前阻塞点”，不是追求最终成功率。

---

## 二、验证前背景

当前 `go2_pos_dynamic_complex` 的设计目标是：

- 默认 `6~10` 个动态障碍物
- 速度范围提升到 `0.35~0.65 m/s`
- 支持多种运动模式：
  - `linear_crossing`
  - `linear_diagonal`
  - `circular`
  - `figure_eight`
- 对复杂任务中的动态障碍物，采用“预定义轨迹直接驱动”的处理方式：
  - 不再要求动态障碍物之间、动态障碍物与静态障碍物之间进行真实碰撞避让
  - 允许动态障碍物在相遇时直接交叉穿过，以避免 PhysX 接触求解导致的局部抖动
- 观测从原来的 `550` 维扩展到 `830` 维：
  - 原始 `12 + 41 + 2`
  - 追加 `4 × 7` 的动态障碍 token

因此，这一任务已经不再与旧的 `dynamic_1/2/3` checkpoint 结构兼容，不能直接拿旧模型无缝恢复运行。

---

## 三、使用的命令

### 1. 配置与观测维度检查

```bash
python - <<'PY'
import sys
sys.path.insert(0, '/home/sea_ws/src/training/legged_gym')
sys.path.insert(0, '/home/sea_ws/src/training/rsl_rl')
from legged_gym.envs.go2.go2_pos_config import Go2PosDynamic1Cfg, Go2PosDynamicComplexCfg

cfg1 = Go2PosDynamic1Cfg()
cfgc = Go2PosDynamicComplexCfg()
print('dynamic1 count_range', cfg1.dynamic_obstacles.count_range, 'num_obs_one_step', cfg1.env.num_obs_one_step)
print('complex count_range', cfgc.dynamic_obstacles.count_range, 'num_obs_one_step', cfgc.env.num_obs_one_step)
PY
```

### 2. `dynamic_complex` 环境冒烟测试

```bash
python - <<'PY'
import sys
sys.path.insert(0, '/home/sea_ws/src/training/legged_gym')
sys.path.insert(0, '/home/sea_ws/src/training/rsl_rl')
import isaacgym
import torch
from legged_gym.envs import *
from legged_gym.utils import task_registry, get_args

args = get_args()
args.task='go2_pos_dynamic_complex'
args.headless=True
args.num_envs=32

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
print('obs_shape', tuple(obs.shape))
print('active_min', int(counts.min().item()))
print('active_max', int(counts.max().item()))
print('active_mean', float(counts.float().mean().item()))
print('motion_counts_first_env', env.dynamic_motion_counts[0].detach().cpu().tolist())
print('dones_sum', int(dones.sum().item()))
print('min_ttc_mean', float(env.min_ttc.mean().item()))
PY
```

### 3. 短程训练尝试（首次大规模尝试）

```bash
cd /home/sea_ws/src/training/legged_gym
export PYTHONPATH=/home/sea_ws/src/training/legged_gym:/home/sea_ws/src/training/rsl_rl:$PYTHONPATH
export WANDB_MODE=disabled

xvfb-run -a python legged_gym/scripts/train.py \
  --task go2_pos_dynamic_complex \
  --headless \
  --num_envs 64 \
  --max_iterations 12
```

### 4. 直接录像尝试

```bash
cd /home/sea_ws/src/training/legged_gym
export PYTHONPATH=/home/sea_ws/src/training/legged_gym:/home/sea_ws/src/training/rsl_rl:$PYTHONPATH

xvfb-run -a python legged_gym/scripts/play_record.py \
  --task go2_pos_dynamic_complex \
  --resume \
  --resume_experiment_name Go2_pos_rough \
  --load_run 06_06_03-46-25_ \
  --checkpoint 50
```

### 5. 最小训练闭环验证

```bash
cd /home/sea_ws/src/training/legged_gym
export PYTHONPATH=/home/sea_ws/src/training/legged_gym:/home/sea_ws/src/training/rsl_rl:$PYTHONPATH
export WANDB_MODE=disabled
export OMP_NUM_THREADS=8

python legged_gym/scripts/train.py \
  --task go2_pos_dynamic_complex \
  --headless \
  --num_envs 8 \
  --max_iterations 1
```

### 6. 复杂任务环境检查视频（启发式回退）

```bash
cd /home/sea_ws/src/training/legged_gym
export PYTHONPATH=/home/sea_ws/src/training/legged_gym:/home/sea_ws/src/training/rsl_rl:$PYTHONPATH

xvfb-run -a python legged_gym/scripts/play_record.py \
  --task go2_pos_dynamic_complex
```

### 7. 复杂任务首个 checkpoint 的策略录像

```bash
cd /home/sea_ws/src/training/legged_gym
export PYTHONPATH=/home/sea_ws/src/training/legged_gym:/home/sea_ws/src/training/rsl_rl:$PYTHONPATH

xvfb-run -a python legged_gym/scripts/play_record.py \
  --task go2_pos_dynamic_complex \
  --resume \
  --load_run 06_20_00-41-42_ \
  --checkpoint 1
```

### 8. 提高速度下限后的并行续训验证

```bash
cd /home/sea_ws/src/training/legged_gym
export PYTHONPATH=/home/sea_ws/src/training/legged_gym:/home/sea_ws/src/training/rsl_rl:$PYTHONPATH
export WANDB_MODE=disabled
export OMP_NUM_THREADS=8

python legged_gym/scripts/train.py \
  --task go2_pos_dynamic_complex \
  --headless \
  --resume \
  --load_run 06_20_00-41-42_ \
  --checkpoint 1 \
  --num_envs 32 \
  --max_iterations 6
```

### 9. 续训后模型录像

```bash
cd /home/sea_ws/src/training/legged_gym
export PYTHONPATH=/home/sea_ws/src/training/legged_gym:/home/sea_ws/src/training/rsl_rl:$PYTHONPATH

xvfb-run -a python legged_gym/scripts/play_record.py \
  --task go2_pos_dynamic_complex \
  --resume \
  --load_run 06_20_01-10-49_ \
  --checkpoint 7
```

### 10. 动态障碍物“允许交叉”录像复查

```bash
cd /home/sea_ws/src/training/legged_gym
export PYTHONPATH=/home/sea_ws/src/training/legged_gym:/home/sea_ws/src/training/rsl_rl:$PYTHONPATH
export OMP_NUM_THREADS=8

xvfb-run -a python legged_gym/scripts/play_record.py \
  --task go2_pos_dynamic_complex \
  --resume \
  --load_run 06_20_01-10-49_ \
  --checkpoint 7
```

---

## 四、验证结果

### 1. 配置检查结果

终端输出：

```text
dynamic1 count_range [3, 3] num_obs_one_step 83
complex count_range [6, 10] num_obs_one_step 83
```

结论：

- 新旧动态任务已经统一切换到新的 `83` 维单步观测格式
- `dynamic_1` 保持固定 `3` 个障碍物
- `dynamic_complex` 的设计目标已经明确设置为 `6~10`

### 2. 环境冒烟测试结果

第一次实现下，环境可以正常创建并完成首步 `step`，但实际激活的动态障碍物数量偏低，最低只到 `1`，与设计目标不一致。

修正动态障碍物填充逻辑后，再次验证得到：

```text
obs_shape (32, 830)
active_min 4
active_max 9
active_mean 6.5
motion_counts_first_env [3, 3, 0, 0]
dones_sum 6
min_ttc_mean 5.256013870239258
```

结论：

- 新环境已经能生成 `830` 维观测并正常 `step`
- 动态障碍物实际激活数量已经显著接近设计目标，平均达到 `6.5`
- 最高已达到 `9`
- 但最低仍会掉到 `4`，说明当前 obstacle 采样器还不够稳定
- 首步 `done_sum=6` 偏大，说明高密度复杂场景下仍存在较多初始化不稳定样本

### 3. 训练验证结果

短程训练命令可以正常进入环境创建阶段，但在训练真正开始后，会被复杂环境的动态障碍物重采样/更新逻辑拖慢，最终未完成有效迭代。

关键报错位置：

```text
File ".../legged_robot_pos_dynamic.py", line 1431, in reset_idx
  self._reset_root_states(normal_ids)
...
File ".../legged_robot_pos_dynamic.py", line 1079, in _init_dynamic_slot
  if not self._bbox_points_valid(env_idx, center, half_extent):
KeyboardInterrupt
```

以及：

```text
File ".../legged_robot_pos_dynamic.py", line 1583, in _post_physics_step_callback
  self._update_dynamic_obstacles()
...
File ".../legged_robot_pos_dynamic.py", line 1544, in _update_dynamic_obstacles
  motion_type = int(self.dynamic_motion_types[env_idx, slot_idx].item())
KeyboardInterrupt
```

结论：

- 当前 `dynamic_complex` 的主要瓶颈已经不是 PPO 或策略结构，而是复杂环境本身的 obstacle reset/update 开销
- 对于 quick validate 来说，这说明：
  - 环境设计方向是对的
  - 但在进入正式训练前，需要先做一次环境性能优化

### 4. 录像验证结果

直接尝试使用旧的 `Go2_pos_rough` checkpoint 录制 `dynamic_complex` 视频时，加载失败：

```text
RuntimeError: Error(s) in loading state_dict for DifferentiableSafeActorCritic:
  size mismatch for backbone.0.weight: copying a param with shape torch.Size([512, 71]) from checkpoint, the shape in current model is torch.Size([512, 99]).
  size mismatch for critic.0.weight: copying a param with shape torch.Size([512, 71]) from checkpoint, the shape in current model is torch.Size([512, 99]).
  size mismatch for encoder.0.weight: copying a param with shape torch.Size([512, 550]) from checkpoint, the shape in current model is torch.Size([512, 830]).
```

结论：

- 由于观测结构已经从 `550` 改到 `830`，旧 checkpoint 已无法直接用于 `dynamic_complex`
- 因此当前无法像 `dynamic_1/2/3` 那样，直接用已有策略录制复杂环境效果视频
- 必须先完成一版新的 `dynamic_complex` 训练，才能产出真实策略视频

### 5. 最小训练闭环结果

在将训练规模压缩到 `8 env / 1 iteration` 后，训练进程虽然最终返回码不是 `0`，但已经成功产出首个 `dynamic_complex` checkpoint：

```text
/home/sea_ws/src/training/legged_gym/logs/Go2_pos_dynamic_complex/06_20_00-41-42_/model_1.pt
```

结论：

- `go2_pos_dynamic_complex` 已经可以完成“建环境 -> 采样 -> PPO 更新 -> 保存 checkpoint”的最小闭环
- 当前问题更像是“大规模训练效率不足”，而不是“训练链路完全不通”

### 6. 启发式环境检查视频结果

为了在旧 checkpoint 不兼容时仍能检查复杂动态障碍物环境，本次补充修改了 `play_record.py`：

- 若 checkpoint 能正常加载，则按原逻辑录制策略视频
- 若 checkpoint 加载失败，则自动回退到一个轻量启发式导航器，继续录制环境检查视频

本次运行结果：

```text
Falling back to heuristic recorder because policy load failed: ...
Video will be saved to: /home/sea_ws/src/training/legged_gym/logs/Go2_pos_dynamic_complex/exported/go2_pos_dynamic_complex_heuristic_env_check.mp4
Recorder mode: heuristic
=== Episode 1/5 finished | success=0.00 dynamic_collision_count=0.00 ===
=== Episode 2/5 finished | success=0.00 dynamic_collision_count=0.00 ===
=== Episode 3/5 finished | success=0.00 dynamic_collision_count=0.00 ===
=== Episode 4/5 finished | success=0.00 dynamic_collision_count=0.00 ===
=== Episode 5/5 finished | success=0.00 dynamic_collision_count=0.00 ===
Video saved: .../go2_pos_dynamic_complex_heuristic_env_check.mp4  (2.7 MB, 215 frames)
```

结论：

- 即使没有可用的新策略模型，也已经可以直接录制复杂环境视频
- 这条链路足够用于检查：
  - 动态障碍物数量是否明显增多
  - 是否出现圆周和 8 字等复杂轨迹
  - 场景是否存在非线性穿行和多障碍交汇

### 7. 首个 `dynamic_complex` 策略录像结果

使用新产出的 `model_1.pt` 再次录像，已经可以正常加载 `830` 维策略模型：

```text
Loading model from: /home/sea_ws/src/training/legged_gym/logs/Go2_pos_dynamic_complex/06_20_00-41-42_/model_1.pt
Loaded policy from: /home/sea_ws/src/training/legged_gym/logs/Go2_pos_dynamic_complex/06_20_00-41-42_/model_1.pt
Video will be saved to: /home/sea_ws/src/training/legged_gym/logs/Go2_pos_dynamic_complex/exported/06_20_00-41-42_model_1.mp4
Recorder mode: policy
=== Episode 1/5 finished | success=0.00 dynamic_collision_count=0.00 ===
=== Episode 2/5 finished | success=0.00 dynamic_collision_count=0.00 ===
=== Episode 3/5 finished | success=0.00 dynamic_collision_count=0.00 ===
=== Episode 4/5 finished | success=0.00 dynamic_collision_count=0.00 ===
=== Episode 5/5 finished | success=0.00 dynamic_collision_count=1.00 ===
Video saved: .../06_20_00-41-42_model_1.mp4  (1.7 MB, 141 frames)
```

结论：

- 当前模型几乎还没有学会复杂动态避障，这是符合预期的，因为只训练了 `1 iteration`
- 但这已经证明：
  - 新任务模型可以被正确保存
  - 新任务模型可以被正确恢复
  - 复杂任务的策略录像链路已经打通

### 8. 提高速度下限后的并行续训结果

在将 `dynamic_complex` 的动态障碍物速度下限进一步提高到 `0.35 m/s` 后，本次继续尝试了更高并行环境数训练：

- `128 env`：能够完成环境创建和 checkpoint 恢复，但在当前复杂环境实现下，更像性能压力测试，未在可接受时间内形成有效短训结果
- `64 env`：同样可以启动，但仍然未在本轮验证窗口内产出新的训练模型
- `32 env`：成功完成一轮受控短训，并保存出新的续训模型

产物如下：

```text
/home/sea_ws/src/training/legged_gym/logs/Go2_pos_dynamic_complex/06_20_01-10-49_/model_7.pt
```

结论：

- 在更高速度下限 `0.35~0.65 m/s` 下，当前方案仍然可以继续训练，不是完全失稳
- 现阶段更合适的并行规模不是盲目拉高到 `64/128`，而是先在 `32 env` 这个可收敛、可产物化的区间做持续迭代
- 这说明当前真正约束训练效率的主要因素，依然是复杂环境本身的 reset/update 开销

### 9. `model_7.pt` 策略录像结果

使用新的续训模型 `model_7.pt` 录制复杂任务视频，加载和录像均成功：

```text
Loading model from: /home/sea_ws/src/training/legged_gym/logs/Go2_pos_dynamic_complex/06_20_01-10-49_/model_7.pt
Loaded policy from: /home/sea_ws/src/training/legged_gym/logs/Go2_pos_dynamic_complex/06_20_01-10-49_/model_7.pt
Video will be saved to: /home/sea_ws/src/training/legged_gym/logs/Go2_pos_dynamic_complex/exported/06_20_01-10-49_model_7.mp4
Recorder mode: policy
=== Episode 1/5 finished | success=0.00 dynamic_collision_count=1.00 ===
=== Episode 2/5 finished | success=0.00 dynamic_collision_count=1.00 ===
=== Episode 3/5 finished | success=0.00 dynamic_collision_count=1.00 ===
=== Episode 4/5 finished | success=0.00 dynamic_collision_count=0.00 ===
=== Episode 5/5 finished | success=0.00 dynamic_collision_count=0.00 ===
Video saved: .../06_20_01-10-49_model_7.mp4  (1.8 MB, 136 frames)
```

结论：

- 相比 `model_1.pt`，`model_7.pt` 依然没有达到有效避障策略，但已经完成了“在更高动态速度下继续续训并稳定导出模型/视频”的验证
- 这说明当前阶段的问题更多是“训练时长不够 + 环境性能仍偏慢”，而不是“提高速度下限后训练链路直接失效”

### 10. 动态障碍物“允许交叉”复查结果

针对视频中观察到的“动态障碍物原地高速抖动”问题，本次进一步调整了复杂任务中的动态障碍物处理方式：

- 复杂任务动态障碍物采用固定基座 actor，并继续由预定义轨迹直接刷新位姿
- 不再要求动态障碍物之间、动态障碍物与其他障碍物之间进行真实碰撞避让
- 训练中的动态碰撞事件仍然按机器人与动态障碍物的几何距离计算，不依赖 PhysX 接触结果

复查录像命令运行成功，结果如下：

```text
Loading model from: /home/sea_ws/src/training/legged_gym/logs/Go2_pos_dynamic_complex/06_20_01-10-49_/model_7.pt
Loaded policy from: /home/sea_ws/src/training/legged_gym/logs/Go2_pos_dynamic_complex/06_20_01-10-49_/model_7.pt
Video will be saved to: /home/sea_ws/src/training/legged_gym/logs/Go2_pos_dynamic_complex/exported/06_20_01-10-49_model_7.mp4
Recorder mode: policy
=== Episode 1/5 finished | success=0.00 dynamic_collision_count=1.00 ===
=== Episode 2/5 finished | success=0.00 dynamic_collision_count=0.00 ===
=== Episode 3/5 finished | success=0.00 dynamic_collision_count=1.00 ===
=== Episode 4/5 finished | success=0.00 dynamic_collision_count=0.00 ===
=== Episode 5/5 finished | success=0.00 dynamic_collision_count=1.00 ===
Video saved: .../06_20_01-10-49_model_7.mp4  (23.5 MB, 2009 frames)
```

结论：

- 新方案下录像链路稳定，没有再出现“改动后直接崩溃”的情况
- 复杂任务中的动态障碍物现在更接近“轨迹驱动目标”，允许相遇时直接穿过
- 这次改动的重点是消除由接触求解引起的异常抖动，而不是提升当前 `model_7.pt` 的避障成功率
- 是否还有残余视觉抖动，仍建议继续以这版新视频做人工逐帧检查

---

## 五、当前阶段结论

本次快速验证可以得出以下结论：

1. `go2_pos_dynamic_complex` 环境已经基本搭起来了。
- 高密度动态障碍物、多轨迹模式、动态 token、未来占据射线都已经进入代码实现。

2. 复杂环境的观测链路已经打通。
- 新任务单步观测为 `83` 维，总观测为 `830` 维。
- 这说明环境、策略输入和任务注册已经同步完成。

3. 复杂环境的 obstacle 数量已经“接近目标”，但还没有完全稳定达到目标。
- 当前平均 `6.5`
- 最高 `9`
- 最低仍会掉到 `4`

4. 当前最大的阻塞点不是策略，而是复杂环境的生成/更新性能。
- reset 阶段 obstacle 采样过慢
- step 阶段复杂轨迹更新仍有明显开销

5. 当前已经具备两类录像能力：
- 没有可用模型时，可回退为启发式环境检查视频
- 有新模型时，可直接录制 `dynamic_complex` 的真实策略视频
 - 在复杂任务中，动态障碍物现在允许按轨迹直接交叉穿过，不再要求处理它们之间的真实碰撞

6. 当前最主要的问题已经从“链路不通”转成“性能和训练效率不足”：
- 可以训练
- 可以保存模型
- 可以恢复模型
- 可以录视频
- 在更高速度下限下也能续训
- 但短时间内的策略效果仍然很弱

---

## 六、建议的下一步

建议接下来优先做下面 3 件事，而不是马上继续拉长训练：

1. 优化 `dynamic_complex` 环境性能
- 减少 Python 层逐 obstacle 的循环更新
- 简化或缓存复杂轨迹的 bbox 校验
- 让 reset 和 step 的 obstacle 更新更向张量化方向靠拢

2. 先做“可训练版复杂任务”，再做“最终高难版复杂任务”
- 例如先把复杂任务约束在 `6~8` 个障碍物
- 等训练链路稳定后，再放宽到 `6~10`

3. 以当前 `model_7.pt` 为起点，继续做 `20~100` iterations 的真正短训
- 下一轮 quick validate 的重点就可以从“链路打通”转为“是否出现初步成功样本”
- 并行环境数建议优先使用 `32 env`，而不是直接拉高到 `64/128 env`
 - 同时继续人工检查动态障碍物视频，确认“允许交叉”后已无明显原地抖动

---

## 七、视频与模型产物

| 产物 | 路径 | 大小 |
|---|---|---:|
| 首个 `dynamic_complex` checkpoint | `training/legged_gym/logs/Go2_pos_dynamic_complex/06_20_00-41-42_/model_1.pt` | 13 MB |
| 提高速度下限后的续训模型 | `training/legged_gym/logs/Go2_pos_dynamic_complex/06_20_01-10-49_/model_7.pt` | 13 MB |
| 启发式环境检查视频 | `training/legged_gym/logs/Go2_pos_dynamic_complex/exported/go2_pos_dynamic_complex_heuristic_env_check.mp4` | 2.7 MB |
| 首个策略录像 | `training/legged_gym/logs/Go2_pos_dynamic_complex/exported/06_20_00-41-42_model_1.mp4` | 1.8 MB |
| 续训后策略录像（允许交叉复查版） | `training/legged_gym/logs/Go2_pos_dynamic_complex/exported/06_20_01-10-49_model_7.mp4` | 23.5 MB |

---

## 七、本次 quick validate 的意义

虽然这次没有像 `dynamic_2/3` 那样直接拿到“训练后成功视频”，但它依然是必要且有价值的一步，因为它已经明确回答了三个关键问题：

1. 复杂动态环境是否已经真正进入代码实现？
- 答案是：已经进入。

2. 当前阻塞点是在算法，还是在环境工程？
- 答案是：主要在环境工程。

3. 提高速度下限后，当前方案还能不能继续训练？
- 答案是：能，但更适合先在 `32 env` 规模上持续续训。

因此，这次验证的定位更准确地说是：

**复杂多动态障碍物环境的工程可行性验证 + 提高速度下限后的初步训练可用性验证。**
