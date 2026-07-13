# SEA-Nav 第二周工作汇报

> 日期：2026-06-07  
> 项目：基于 SEA-Nav 的动态障碍物导航  
> 本周主题：从“静态房间导航 baseline”推进到“动态障碍物主动避障”的可行性验证

---

## 一、本周工作概述

本周的工作不是简单地“往仿真里加几个会动的箱子”，而是围绕一个核心问题展开：

**如何在尽量复用原始 SEA-Nav 架构的前提下，把它从“静态障碍物导航”推进到“动态障碍物主动避障”，并且保证方案在工程上可落地、训练上可推进、汇报上能自圆其说。**

本周完成的工作可以概括为四条主线：

1. 深入理解原始 `SEA-Nav` 的观测、训练、控制与安全约束机制，明确它为什么能做静态避障、又为什么不能直接处理真正的动态障碍物。
2. 在原仓库基础上完成动态任务环境改造，包括稀疏静态场景、动态障碍物 actor、动态观测融合、碰撞统计和分阶段任务注册。
3. 打通 `go2_pos_rough -> go2_pos_sparse_static -> go2_pos_dynamic_1 -> go2_pos_dynamic_2 -> go2_pos_dynamic_3` 的渐进式微调链路，并完成小规模训练与视频验证。
4. 反思当前方案与“真实传感器驱动的动态避障”之间的差异，明确当前实现的能力边界、合理性和后续升级方向。

本周的阶段性结论是：

- 原始 SEA-Nav 的主干算法不需要推倒重来；
- 动态任务的关键瓶颈不在网络结构，而在**观测语义**；
- 只要让动态障碍物以与原始 `rays` 兼容的形式进入观测，原始 PPO + CBF 框架就可以被继续利用；
- 当前 1/2/3 个动态障碍物的小规模验证已经说明方案可行，但 3 障碍物距离最终稳定交付仍有差距。

---

## 二、对原始 SEA-Nav 项目的理解

本周首先做的是对原始项目进行结构化梳理，而不是直接修改代码。

当前实际代码根目录是 `/home/sea_ws/src`。与导航任务最关键的模块如下：

- 环境注册入口：[envs/__init__.py](/home/sea_ws/src/training/legged_gym/legged_gym/envs/__init__.py:25)
- 原始导航环境：[legged_robot_pos.py](/home/sea_ws/src/training/legged_gym/legged_gym/envs/base/legged_robot_pos.py:1)
- 新增动态环境：[legged_robot_pos_dynamic.py](/home/sea_ws/src/training/legged_gym/legged_gym/envs/base/legged_robot_pos_dynamic.py:1)
- Go2 任务配置：[go2_pos_config.py](/home/sea_ws/src/training/legged_gym/legged_gym/envs/go2/go2_pos_config.py:1)
- 地形生成：[terrain.py](/home/sea_ws/src/training/legged_gym/legged_gym/utils/terrain.py:1)
- 训练脚本：[train.py](/home/sea_ws/src/training/legged_gym/legged_gym/scripts/train.py:1)
- 推理脚本：[play.py](/home/sea_ws/src/training/legged_gym/legged_gym/scripts/play.py:1)
- 录屏脚本：[play_record.py](/home/sea_ws/src/training/legged_gym/legged_gym/scripts/play_record.py:1)
- PPO + CBF 模块：[cbf_actor_critic.py](/home/sea_ws/src/training/rsl_rl/rsl_rl/modules/cbf_actor_critic.py:1)

### 2.1 原始任务的控制链路

原始 SEA-Nav 并不是一个“直接输出关节动作完成导航”的简单结构，而是一个带分层控制意味的系统：

1. 导航环境输出导航观测。
2. PPO 策略输出 3 维导航动作：`[vx, vy, yaw_rate]`。
3. 一个 locomotion 策略再把导航动作转成 12 个关节层面的控制信号。
4. CBF 安全层基于障碍物距离对导航动作做额外约束。

因此，原始系统已经具备“目标驱动 + 避障约束”的基本能力。  
如果动态任务失败，问题通常不在“网络完全不会避障”，而在“网络看到的障碍物语义不对”。

### 2.2 原始任务的观测结构

原始 Go2 导航任务配置中：

- 每步观测由 `12 维机器人状态 + 41 条 rays + 2 维目标信息` 组成；
- 再叠加 `10` 帧历史，因此总观测维度为 `550`。

这套设计本身非常适合局部避障导航，因为：

- `goal` 提供全局任务方向；
- `rays` 提供局部障碍物结构；
- 历史堆叠让策略具备一定的短时动态判断能力。

这里真正关键的是：**原始架构对 `rays` 的依赖非常强。**

### 2.3 原仓库中的 `rays` 到底是什么

这是本周理解项目时最关键、也是最容易误解的一点。

从名字看，`rays` 很容易被理解成“激光雷达”或“真实 2D 激光扫描”。  
但原始仓库里的 `rays` 并不是一个 Isaac Gym 中真实挂载的 LiDAR 传感器。

原始 `rays` 的生成过程在 [legged_robot_pos.py](/home/sea_ws/src/training/legged_gym/legged_gym/envs/base/legged_robot_pos.py:475) 到 [legged_robot_pos.py](/home/sea_ws/src/training/legged_gym/legged_gym/envs/base/legged_robot_pos.py:509)：

1. 先在机器人周围构造一批局部采样点；
2. 再去静态地形的 `height_samples` 中查询这些点的高度；
3. 将高于阈值的位置当作障碍物；
4. 再通过 `_grid2ray()` 把局部二值栅格转换为 41 条二维距离射线。

因此，原始 `rays` 更准确的描述是：

**基于静态 heightfield/heightmap 合成出来的理想化局部测距观测。**

这意味着它有两个重要性质：

1. 它不是“真实原始传感器回波”；
2. 它天然只反映**已经写入静态地形图**的障碍物。

这正是原始静态任务能成立的原因，也是它不能直接处理真正动态障碍物的根因。

---

## 三、为什么原始静态方案不能直接解决动态障碍物任务

这一部分是本周方案设计的逻辑起点。

从直觉上说，如果机器人前面有一个障碍物，不管它是静态还是动态，只要激光雷达能扫到，避障策略就应该能处理。  
这个判断在真实传感器系统里没有问题。

但原始 SEA-Nav 不是这么实现的。

### 3.1 原始方案能避的是“静态地图中的障碍物”

原始环境中的障碍物主要来自地形生成函数，例如 `hard_room`，它们最终进入了 `heightfield / trimesh`。  
而 `rays` 的来源正是这张静态地形图。

因此原始模型学到的是：

- “如果静态地图中的某个方向被障碍物占据，就调整自己的局部导航动作”

这是一种**基于静态局部占据结构的导航策略**。

### 3.2 如果只把障碍物做成会动的 actor，但不改观测，会发生什么

如果只是把动态障碍物作为独立 actor 加到仿真里，而不修改 `rays`：

- 物理世界里，障碍物确实会动；
- 机器人也可能真的撞上它；
- 但是策略输入里仍然没有这类障碍物的信息。

这样训练出来的策略很容易表现为：

- 视觉上“场景里有动态障碍物”
- 但行为上仍然是“静态导航”

因此，动态任务的关键不是“让障碍物动起来”，而是“让动态障碍物进入与原始模型兼容的观测语义”。

### 3.3 为什么不直接重做一套全新的动态感知

本周实际考虑过几条路线：

1. 直接换成真实 LiDAR / 深度相机输入  
优点是感知更真实，静态和动态障碍物天然统一。  
缺点是工程量大，观测语义变化大，原始 checkpoint 难以直接复用。

2. 每步在线修改 heightfield，让动态障碍物写进地形图  
优点是能复用原始 `_get_rays()`。  
缺点是 heightfield 本质是静态地形表示，不适合逐帧移动 actor，工程上非常别扭。

3. 保留原始 `rays` 结构，只为动态障碍物额外合成一个兼容的 `dynamic_rays`  
优点是：
   - 不改网络输入维度；
   - 不改动作维度；
   - 可以直接从静态模型继续微调；
   - 与原始 CBF 约束结构兼容。  
缺点是：
   - 当前实现不是“真实原始传感器回波”；
   - 需要额外说明其理想化假设。

本周最终选择的是第三条路线，因为它是最稳妥、最容易按时交付的一条工程路径。

---

## 四、本周方案设计原则

在明确问题本质之后，本周的方案设计遵循了四个原则。

### 4.1 主干算法尽量不动

本周没有修改：

- 策略网络输入输出结构
- PPO 训练主干
- CBF 模块接口
- 动作空间定义

保留这些结构的目的是最大程度复用原项目已经验证有效的部分。

### 4.2 优先复用静态 baseline 的能力

原始 `go2_pos_rough` 模型已经学会：

- 朝目标方向推进
- 根据局部障碍物结构做基本避障

动态任务并不是要从零开始学习“怎么走”，而是要在已有导航能力上补充“如何处理移动障碍物”的能力。  
因此，本周从一开始就把问题定义为“静态能力上的增量学习”，而不是“全新任务重训”。

### 4.3 动态障碍物观测要兼容原始 `rays`

如果直接增加新的观测模态或增加新的观测维度，会立即带来两个问题：

1. 原始 checkpoint 无法直接复用；
2. 训练不稳定时更难定位问题来自哪里。

因此本周刻意维持：

- 41 条 ray 数量不变；
- 总观测维度仍为 550；
- 动态障碍物只通过“改进 ray 语义”进入策略。

### 4.4 汇报上必须诚实区分“理想化感知”和“真实传感器”

这是本周后期最重视的一点。

当前实现确实没有把障碍物未来轨迹喂给策略，但它也不是严格意义上的“原始传感器点云输入”。  
更准确地说，它是：

**基于环境当前状态合成的理想化局部测距观测。**

这一定义既能说明它不依赖未来轨迹，又能诚实反映它与真实 LiDAR 仍有差距。

---

## 五、在原仓库基础上的具体实现

### 5.1 新增动态环境主类

新增文件：

- [legged_robot_pos_dynamic.py](/home/sea_ws/src/training/legged_gym/legged_gym/envs/base/legged_robot_pos_dynamic.py:1)

这个类继承自原始 `LeggedRobotPos`，扩展了以下能力：

- 稀疏房间中的起点/目标点重新采样；
- 每个 env 创建多个动态障碍物 actor；
- reset 时动态障碍物位置与速度随机化；
- 每步更新动态障碍物位置；
- 生成 `dynamic_rays` 并与静态 `rays` 融合；
- 统计动态碰撞次数；
- 动态碰撞触发 early reset 和单独惩罚。

关键函数包括：

- `_create_envs()`：[legged_robot_pos_dynamic.py](/home/sea_ws/src/training/legged_gym/legged_gym/envs/base/legged_robot_pos_dynamic.py:86)
- `_reset_dofs()`：[legged_robot_pos_dynamic.py](/home/sea_ws/src/training/legged_gym/legged_gym/envs/base/legged_robot_pos_dynamic.py:351)
- `_reset_dynamic_obstacles()`：[legged_robot_pos_dynamic.py](/home/sea_ws/src/training/legged_gym/legged_gym/envs/base/legged_robot_pos_dynamic.py:365)
- `_get_dynamic_rays()`：[legged_robot_pos_dynamic.py](/home/sea_ws/src/training/legged_gym/legged_gym/envs/base/legged_robot_pos_dynamic.py:520)
- `check_termination()`：[legged_robot_pos_dynamic.py](/home/sea_ws/src/training/legged_gym/legged_gym/envs/base/legged_robot_pos_dynamic.py:567)

### 5.2 新增稀疏静态房间

修改文件：

- [terrain.py](/home/sea_ws/src/training/legged_gym/legged_gym/utils/terrain.py:351)

新增：

- `dynamic_room_sparse_terrain_func()`

设计思路不是彻底取消静态障碍物，而是：

- 保留边界墙；
- 保留少量固定静态柱体；
- 去掉高密度随机静态障碍物。

这样做的原因是：

1. 场景仍然具有一定导航结构，不会退化成完全空房间；
2. 但不会像原始 `hard_room` 一样把训练难度抬得过高；
3. 更有利于把“动态避障能力”单独观察出来。

保留的 3 个静态柱体中心位置为：

- `(3.6, 2.6)`
- `(6.4, 5.0)`
- `(3.6, 7.4)`

### 5.3 新增任务配置与注册

修改文件：

- [go2_pos_config.py](/home/sea_ws/src/training/legged_gym/legged_gym/envs/go2/go2_pos_config.py:240)
- [envs/__init__.py](/home/sea_ws/src/training/legged_gym/legged_gym/envs/__init__.py:25)

新增任务：

- `go2_pos_sparse_static`
- `go2_pos_dynamic_1`
- `go2_pos_dynamic_2`
- `go2_pos_dynamic_3`

动态障碍物配置的核心参数为：

- 尺寸：`0.45 x 0.45 x 0.8`
- 观测近似半径：`ray_radius = 0.32`
- 车道中心：`x = [2.2, 5.0, 7.8]`
- 车道扰动：`lane_x_jitter = 0.15`
- 纵向活动范围：`y ∈ [1.2, 8.8]`
- 速度范围：`[0.15, 0.40] m/s`
- 运动模式：纵向往返，到边界反向

这里故意采用了较简单的运动模式，不是因为“机器人提前知道轨迹”，而是因为本周目标是先验证动态观测与渐进训练链路是否成立。

### 5.4 动态障碍物是如何进入观测的

这是本周最核心的实现点。

原始 `rays` 来源于静态地图采样。  
本周没有破坏原始静态感知，而是在其基础上增加一层动态障碍物感知：

1. 先照常计算 `static_rays`
2. 再把动态障碍物当前位置变换到机器人局部坐标系
3. 将障碍物用圆形占据近似
4. 对每条 ray 求与这些圆的最早交点距离，得到 `dynamic_rays`
5. 最后令：

```python
self.rays = torch.minimum(self.static_rays, self.dynamic_rays)
```

对应位置：

- 动态射线：[legged_robot_pos_dynamic.py](/home/sea_ws/src/training/legged_gym/legged_gym/envs/base/legged_robot_pos_dynamic.py:520)
- 融合逻辑：[legged_robot_pos_dynamic.py](/home/sea_ws/src/training/legged_gym/legged_gym/envs/base/legged_robot_pos_dynamic.py:558)

这一设计的优点是：

- 对策略来说，输入格式没有变；
- 对 CBF 来说，安全约束的使用方式也没有变；
- 对训练来说，可以直接从静态 checkpoint 继续微调。

### 5.5 当前 `dynamic_rays` 的合理性与局限

这里必须专门说明，因为这正是本周理解最深入、也最需要汇报时说清楚的地方。

当前 `dynamic_rays` 的计算依据是动态障碍物在仿真中的**当前真实位置**，而不是未来轨迹。  
因此当前实现：

- **没有使用未来信息**
- **没有把运动策略直接暴露给策略网络**

但它仍然是一个理想化实现，因为它默认可以无噪声、无遮挡地获得动态障碍物当前位姿，并据此合成局部测距。

所以本周方案的准确表述应该是：

**基于环境当前状态合成的理想化局部测距观测驱动的动态避障。**

而不是：

**完全基于真实激光雷达原始输入的动态避障。**

这个区分非常重要。它体现了本周方案设计中的一条原则：  
在按时完成项目目标的前提下，先把“动态避障闭环”跑通；同时在汇报中诚实说明当前感知层的理想化假设。

### 5.6 碰撞统计、奖励与终止条件

本周增加了三类统计量：

- `dynamic_collision_count`
- `body_collision_count`
- `total_collision_count`

动态碰撞的检测逻辑在 [legged_robot_pos_dynamic.py](/home/sea_ws/src/training/legged_gym/legged_gym/envs/base/legged_robot_pos_dynamic.py:571)。

同时增加了：

- 动态碰撞惩罚 `_reward_dynamic_collision()`
- 动态碰撞 early reset

对应位置：

- `check_termination()`：[legged_robot_pos_dynamic.py](/home/sea_ws/src/training/legged_gym/legged_gym/envs/base/legged_robot_pos_dynamic.py:567)
- `_reward_dynamic_collision()`：[legged_robot_pos_dynamic.py](/home/sea_ws/src/training/legged_gym/legged_gym/envs/base/legged_robot_pos_dynamic.py:620)

这样做是为了让训练目标与课程验收目标更一致：

- 不是只看 reward；
- 而是让“到达目标”和“减少动态碰撞”同时成为优化目标。

### 5.7 续训、推理、录屏脚本适配

本周还对工程链路做了较多补足，以保证任务能真正训练、评测、录制，而不是只停留在环境层面。

相关修改包括：

- `resume_experiment_name` 支持跨任务续训；
- `play.py` 和 `play_record.py` 支持新任务；
- 新增 `evaluate_dynamic.py`；
- 录屏脚本新增 top-down minimap overlay；
- 录屏场景支持更高交互密度的专用设置。

对应文件：

- [helpers.py](/home/sea_ws/src/training/legged_gym/legged_gym/utils/helpers.py:145)
- [task_registry.py](/home/sea_ws/src/training/legged_gym/legged_gym/utils/task_registry.py:159)
- [legged_robot_config.py](/home/sea_ws/src/training/legged_gym/legged_gym/envs/base/legged_robot_config.py:273)
- [play.py](/home/sea_ws/src/training/legged_gym/legged_gym/scripts/play.py:17)
- [play_record.py](/home/sea_ws/src/training/legged_gym/legged_gym/scripts/play_record.py:45)
- [play_record.py](/home/sea_ws/src/training/legged_gym/legged_gym/scripts/play_record.py:98)
- [evaluate_dynamic.py](/home/sea_ws/src/training/legged_gym/legged_gym/scripts/evaluate_dynamic.py:17)

---

## 六、训练策略设计

本周没有直接尝试“三个动态障碍物从零训练”，而是采用了更符合工程常识的分阶段微调路线。

### 6.1 设计逻辑

这样设计的原因是：

1. 原始静态模型已经具备朝目标移动和静态避障能力；
2. 新任务的核心新增难点是“动态障碍物感知与反应”；
3. 如果直接上 3 个动态障碍物，很难区分失败是来自：
   - 环境实现错误
   - 观测设计不合理
   - 奖励设计不稳定
   - 训练难度过高

因此本周采用的训练链路是：

1. `go2_pos_rough`
2. `go2_pos_sparse_static`
3. `go2_pos_dynamic_1`
4. `go2_pos_dynamic_2`
5. `go2_pos_dynamic_3`

### 6.2 关键训练命令

#### 阶段 1：稀疏静态适应

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

#### 阶段 2：1 个动态障碍物

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

#### 阶段 3：2 个动态障碍物

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

#### 阶段 4：3 个动态障碍物

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

这套链路的重点不是追求本周就拿到最终最优结果，而是：

- 先验证每一阶段都能承接上一阶段；
- 先验证“任务难度增加时，训练还能继续推进”；
- 先把动态任务的工程闭环打通。

---

## 七、训练与实现过程中解决的关键问题

### 7.1 多 actor 环境下的 DOF reset 索引错误

这是本周最典型、也最有代表性的工程问题。

#### 现象

- 单环境推理、录屏都正常；
- 但 `128 env` 训练时在 `reset()` 附近出现 GPU illegal memory access。

#### 根因

动态环境下每个 env 中不再只有机器人一个 actor，而是：

- 1 个机器人
- N 个动态障碍物

原始基类中的 `_reset_dofs()` 默认把 `env_ids` 直接传给 Isaac Gym 的 indexed reset 接口。  
在单 actor 环境下这通常没问题；但在多 actor 环境下，`env_id` 已经不等于“机器人 actor 在全局 root state 中的真实索引”。

于是 reset 写错对象，最终导致 GPU 状态破坏。

#### 修复

在动态环境中重写 `_reset_dofs()`，使用：

- `self.robot_actor_indices[env_ids]`

作为真正的 actor 索引传入。

位置：

- [legged_robot_pos_dynamic.py](/home/sea_ws/src/training/legged_gym/legged_gym/envs/base/legged_robot_pos_dynamic.py:351)

#### 价值

这个修复说明本周不是只停留在“能单 env 看着动起来”，而是把多环境训练真正跑通了。

### 7.2 视频可读性问题

在早期录屏中，虽然能看到机器人和障碍物，但不容易看出：

- 起点在哪里
- 目标点在哪里
- 当前是否真的完成了一轮导航

因此本周没有把录屏只当“附带产物”，而是专门重构了录屏可视化：

- 新增 top-down minimap overlay
- 显示 start / goal / robot / dynamic obstacles
- 显示 episode 编号、距离目标距离、碰撞统计

这使得视频更适合汇报和答辩，而不仅仅是自测。

### 7.3 交互稀疏问题

早期稀疏房间中有一部分 episode，机器人并不会明显遇到动态障碍物，因此展示效果不足。

对此本周又增加了“录制专用强交互设置”：

- 收紧起点/终点的 `y` 范围
- 收紧动态障碍物活动带
- 可启用 `force_interaction`

这样做的目的不是改训练逻辑，而是让演示视频更能体现“动态避障”的效果。

---

## 八、小规模验证结果

### 8.1 `go2_pos_dynamic_1`

基线对照结果：

- 直接用静态模型跑动态任务，录屏中 `5/5` 全失败；
- 说明原始静态权重不能直接当作动态任务答案。

微调后结果：

- 训练期成功率峰值大约达到 `0.52 ~ 0.67`；
- 录屏中 `5` 个 episode 有 `2` 个成功。

对应报告：

- [dynamic_quick_validate_report.md](/home/sea_ws/dynamic_quick_validate_report.md:1)

### 8.2 `go2_pos_dynamic_2`

短程微调结果表明：

- 双障碍物任务已经可以训练起来；
- 早期录屏中 `5/5` 全成功；
- 在后续更严格的录制专用强交互场景下，录屏表现变为 `3/5` 成功。

这说明：

- 双障碍物任务本身已经具有较好可行性；
- 但如果刻意提高交互密度，模型仍有较大优化空间。

### 8.3 `go2_pos_dynamic_3`

短程微调结果表明：

- 训练期成功率大约达到 `0.58 ~ 0.69`；
- 早期录屏中 `5` 个 episode 有 `3` 个成功；
- 在后续更严格的录制专用强交互场景下，录屏表现变为 `2/5` 成功。

这说明：

- 三障碍物任务已经具备“闭环可行性”；
- 但当前训练量和分布覆盖还不足以支持最终稳定交付。

### 8.4 当前验证结果应该如何解释

本周这些结果不应该被解释成“最终模型已经达标”，而应该解释成：

1. 动态任务环境实现是正确的；
2. 动态障碍物已经真正进入了策略观测；
3. 分阶段微调路线能够把 1 障碍物扩展到 2、3 障碍物；
4. 当前主要瓶颈已经从“能不能做出来”转移到“怎么进一步提高稳定性和泛化性”。

---

## 九、当前方案与真实传感器方案的关系

这一部分是本周报告中最需要讲透的地方，因为它直接决定汇报时的技术可信度。

### 9.1 当前方案不是“提前知道未来轨迹”

当前实现没有向策略提供：

- 动态障碍物未来轨迹
- 未来反向时机
- 未来速度计划

策略得到的仍然是当前局部观测和历史观测，因此它不是“拿到了未来信息再做规划”。

### 9.2 当前方案也不是“严格真实传感器输入”

当前 `dynamic_rays` 是基于障碍物当前真值位置合成的理想化测距。  
这意味着它比真实 LiDAR 更干净，因为它默认：

- 无噪声
- 无漏检
- 无误检
- 无遮挡推断误差
- 形状近似已知

因此，它是一种介于“纯仿真真值接口”和“真实原始传感器输入”之间的中间方案。

### 9.3 为什么本周仍然选择这条路线

因为本周的任务目标不是做一个完整的真实感知系统，而是：

- 先验证动态避障的训练闭环；
- 先证明原始 SEA-Nav 主干可以扩展到动态任务；
- 先拿到一条能持续推进的技术路线。

如果一开始就把目标定成“真实 LiDAR 点云 + 动态跟踪 + 噪声鲁棒性”，本周大概率只会停留在感知系统接线阶段，很难拿到可训练、可录屏、可汇报的结果。

### 9.4 后续更真实的升级方向

如果下一阶段要进一步贴近真实机器人感知，可以沿两条路线升级：

1. 使用仿真中的真实 scene raycast / depth sensor / LiDAR 生成局部 scan  
这样静态与动态障碍物会天然统一进入观测。

2. 基于深度/点云构建局部占据图，再从占据图生成与原始 `rays` 兼容的输入  
这样可以在保留原始观测形式的同时，把感知来源换成真实传感器。

因此，本周方案不是最终形态，但它是一个合理的过渡层：

- 工程上可落地；
- 训练上可推进；
- 研究上可继续向真实感知过渡。

---

## 十、当前阶段结论

本周最重要的结论有五点。

1. 对原始 SEA-Nav 的深入理解表明：动态任务的关键矛盾不在控制主干，而在观测语义。

2. 原始仓库中的 `rays` 并不是真实 LiDAR，而是由静态地形 heightfield 合成的局部测距，因此原始方法天然只能感知静态障碍物。

3. 本周通过“动态障碍物 actor + `dynamic_rays` 融合”的方式，在不改变策略输入输出维度的情况下，把原始系统扩展到了动态任务。

4. 当前方案没有向策略泄露未来轨迹信息，但它使用的是动态障碍物当前真值位置合成的理想化局部测距，因此应被准确表述为“基于环境状态合成的理想化局部测距观测驱动的动态避障”。

5. 1/2/3 个动态障碍物的小规模训练和录屏验证已经说明这条路线可行；接下来的重点不再是“能不能做”，而是“如何把三障碍物任务做得更稳、更真实、更可评测”。

---

## 十一、下一步计划

接下来建议重点推进四件事。

1. 延长 `go2_pos_dynamic_3` 的训练时长，优先提升三障碍物场景下的稳定性。

2. 固定随机种子，做更正式的 `20/50 episode` 评测，输出：
   - `success_rate`
   - `safe_success_rate`
   - `avg_dynamic_collision_count`
   - `avg_total_collision_count`
   - `timeout_rate`
   - `mean_time_to_goal`

3. 用增强后的录屏脚本重新录制正式 demo，展示：
   - 起点
   - 目标点
   - 动态障碍物
   - 机器人避障过程
   - 成功与失败对比

4. 如果导师后续更强调“真实传感器一致性”，则优先把 `dynamic_rays` 的来源从环境真值改成更真实的 scene raycast / 深度感知，而不是先改 PPO/CBF 网络结构。

---

## 十二、附：本周主要代码与产物

### 代码修改

- 动态环境主类：[legged_robot_pos_dynamic.py](/home/sea_ws/src/training/legged_gym/legged_gym/envs/base/legged_robot_pos_dynamic.py:1)
- 动态任务配置：[go2_pos_config.py](/home/sea_ws/src/training/legged_gym/legged_gym/envs/go2/go2_pos_config.py:240)
- 稀疏静态场景：[terrain.py](/home/sea_ws/src/training/legged_gym/legged_gym/utils/terrain.py:351)
- 任务注册：[envs/__init__.py](/home/sea_ws/src/training/legged_gym/legged_gym/envs/__init__.py:25)
- 续训支持：[helpers.py](/home/sea_ws/src/training/legged_gym/legged_gym/utils/helpers.py:145)
- 推理适配：[play.py](/home/sea_ws/src/training/legged_gym/legged_gym/scripts/play.py:17)
- 录屏增强：[play_record.py](/home/sea_ws/src/training/legged_gym/legged_gym/scripts/play_record.py:45)
- 评测脚本：[evaluate_dynamic.py](/home/sea_ws/src/training/legged_gym/legged_gym/scripts/evaluate_dynamic.py:17)

### 报告

- 原始 baseline 报告：[report.md](/home/sea_ws/report.md:1)
- 动态任务实施方案：[plan.md](/home/sea_ws/plan.md:1)
- 1 障碍物快速验证：[dynamic_quick_validate_report.md](/home/sea_ws/dynamic_quick_validate_report.md:1)
- 2/3 障碍物快速验证：[dynamic_23_quick_validate_report.md](/home/sea_ws/dynamic_23_quick_validate_report.md:1)

### 视频

- `dynamic_1`：`training/legged_gym/logs/Go2_pos_dynamic_1/exported/06_06_22-36-22_model_190.mp4`
- `dynamic_2`：`training/legged_gym/logs/Go2_pos_dynamic_2/exported/06_06_23-15-21_model_250.mp4`
- `dynamic_3`：`training/legged_gym/logs/Go2_pos_dynamic_3/exported/06_06_23-18-12_model_310.mp4`

---

## 十三、最后总结

从本周的推进过程看，这个项目最重要的收获不是“跑出了几个视频”，而是把问题真正想清楚了：

- 原始 SEA-Nav 为什么能做静态导航；
- 为什么原始 `rays` 不能直接处理动态障碍物；
- 为什么当前阶段应该优先改观测语义而不是重做网络；
- 为什么当前 `dynamic_rays` 方案是一个合理但理想化的中间方案；
- 为什么后续工作重点应该从“是否可行”转向“如何做得更稳、更真实、更可评测”。

也就是说，本周完成的不是一个表面上的 demo 拼接，而是一条从项目理解、方案设计、工程实现到实验验证都相对闭环的动态避障路线。

---

## 十四、补充：三障碍物正式强化训练与 10 次评测

在完成前面的 1/2/3 个动态障碍物小规模验证之后，又对最终目标任务 `go2_pos_dynamic_3` 做了一轮更正式的强化训练，并进行了 `10` 次 episode 的定量评测。

这一步的目标不是继续验证“方案是否可行”，而是回答更实际的问题：

- 在当前机器资源下，1~2 小时内能否训练出更强的三障碍物模型？
- 强化后的模型是否能在定量指标上明显优于之前的短训版本？

### 1. 机器资源评估

在正式训练前，对当前机器资源做了快速检查。

- GPU：`NVIDIA GeForce RTX 4090`
- 显存：`24 GB`
- CPU：`52` 物理核，`104` 线程
- 内存：约 `1 TB`

实际训练过程中，`go2_pos_dynamic_3` 在 `256 envs` 下的显存占用大致在 `4.6 GB ~ 9.3 GB`，远低于显存上限，因此资源瓶颈不在显存，而主要在 Isaac Gym 仿真吞吐和 PPO 更新速度。

结合历史训练日志和本次正式训练观察到的吞吐，单次 iteration 的时间大约在：

- `collection ≈ 1.6 ~ 2.1 s`
- `learning ≈ 0.4 ~ 0.6 s`

即每 `10` 个 iterations 大约需要 `22 ~ 24 s`。

基于这个速度，采用 `256 envs` 做一轮数百次 iteration 的强化训练，是可以在 1~2 小时内稳定完成的。

### 2. 正式训练参数设计

正式训练从已有的三障碍物短训模型继续，而不是重新从头训练。

起始模型：

- `training/legged_gym/logs/Go2_pos_dynamic_3/06_06_23-18-12_/model_310.pt`

采用的命令为：

```bash
conda activate sea_nav
cd /home/sea_ws/src

PYTHONUNBUFFERED=1 WANDB_MODE=disabled xvfb-run -a \
  python training/legged_gym/legged_gym/scripts/train.py \
  --task go2_pos_dynamic_3 \
  --headless \
  --resume \
  --load_run 06_06_23-18-12_ \
  --checkpoint 310 \
  --num_envs 256 \
  --max_iterations 400
```

这组参数的设计考虑是：

- `256 envs`：比之前小规模验证的 `128 envs` 更高效，但仍然足够稳，不会把 GPU 顶满到不可控。
- 从 `model_310.pt` 继续：复用已经具备基础三障碍物能力的模型，避免浪费训练预算。
- `400 iterations`：在当前吞吐下属于能在预算时间内完成、同时又足够带来明显提升的一档设置。

训练产物位于：

- 强化版模型：[model_500.pt](/home/sea_ws/src/training/legged_gym/logs/Go2_pos_dynamic_3/06_07_00-48-13_/model_500.pt)

对应路径：

- `training/legged_gym/logs/Go2_pos_dynamic_3/06_07_00-48-13_/model_500.pt`

### 3. 强化训练中的关键现象

这轮强化训练不是单调上升的，而是存在一定波动，这和前面小规模训练的结论一致：三障碍物任务仍然是一个相对敏感的场景。

不过训练中已经多次出现明显优于旧模型的指标窗口。例如：

- `Iteration 420`
  - `success = 0.8958`
  - `safe_success = 0.8333`
  - `avg_total_collision_count = 0.1875`

- `Iteration 510`
  - `success = 0.8542`
  - `safe_success = 0.7083`
  - `avg_total_collision_count = 0.2917`

- `Iteration 540`
  - `success = 0.7917`
  - `safe_success = 0.7917`
  - `avg_total_collision_count = 0.2188`

这说明强化训练确实在把策略往“更少碰撞、更稳定到达目标”的方向推。

### 4. 10 次 episode 定量评测

训练完成后，没有立即录视频，而是先用 `10` 次 episode 做定量评测，直接看成功率、碰撞次数和时间指标。

评测命令为：

```bash
conda activate sea_nav
cd /home/sea_ws/src

PYTHONUNBUFFERED=1 xvfb-run -a \
  python training/legged_gym/legged_gym/scripts/evaluate_dynamic.py \
  --task go2_pos_dynamic_3 \
  --resume \
  --load_run 06_07_00-48-13_ \
  --checkpoint 500 \
  --num_episodes 10
```

强化版 `model_500.pt` 的评测结果为：

| 指标 | 结果 |
|---|---:|
| `success_rate` | `1.0000` |
| `safe_success_rate` | `1.0000` |
| `avg_total_collision_count` | `0.0000` |
| `avg_dynamic_collision_count` | `0.0000` |
| `avg_body_collision_count` | `0.0000` |
| `timeout_rate` | `0.0000` |
| `mean_time_to_goal` | `13.4560 s` |

也就是说，在这 `10` 次测试中：

- `10/10` 全部成功
- `0` 次动态碰撞
- `0` 次机体碰撞
- `0` 次超时

这是当前阶段非常强的结果，已经明显强于之前的小规模三障碍物模型。

### 5. 与原始短训三障碍物模型对照

为了说明强化训练是否真的带来收益，又对旧模型 `model_310.pt` 做了同样的 `10` 次 episode 评测。

评测命令：

```bash
conda activate sea_nav
cd /home/sea_ws/src

PYTHONUNBUFFERED=1 xvfb-run -a \
  python training/legged_gym/legged_gym/scripts/evaluate_dynamic.py \
  --task go2_pos_dynamic_3 \
  --resume \
  --load_run 06_06_23-18-12_ \
  --checkpoint 310 \
  --num_episodes 10
```

旧模型 `model_310.pt` 的结果为：

| 指标 | `model_310.pt` |
|---|---:|
| `success_rate` | `0.7000` |
| `safe_success_rate` | `0.7000` |
| `avg_total_collision_count` | `0.4000` |
| `avg_dynamic_collision_count` | `0.3000` |
| `avg_body_collision_count` | `0.1000` |
| `timeout_rate` | `0.1000` |
| `mean_time_to_goal` | `22.7257 s` |

与强化版对比可以看出：

- 成功率从 `70%` 提升到 `100%`
- 平均总碰撞次数从 `0.4` 降到 `0`
- 平均动态碰撞次数从 `0.3` 降到 `0`
- 平均到达时间从 `22.73 s` 降到 `13.46 s`

这说明本次强化训练不是“只让训练曲线更好看”，而是在真正的 episode 评测上带来了明显提升。

### 6. 当前阶段结论更新

加入这轮正式强化训练后，可以对三障碍物任务做出更强的阶段性判断：

1. 在当前硬件条件下，`go2_pos_dynamic_3` 完全可以在 1~2 小时预算内做一轮有效强化训练。

2. 基于当前这套“动态 obstacle actor + dynamic rays 融合 + 分阶段微调”的方法，三障碍物任务不只是“可行”，而且已经能够在小规模正式评测中达到很好的结果。

3. 当前最值得继续保留和使用的三障碍物模型是：
   - [model_500.pt](/home/sea_ws/src/training/legged_gym/logs/Go2_pos_dynamic_3/06_07_00-48-13_/model_500.pt)

4. 后续如果继续做课程项目最终交付，下一步就不再是“先证明能不能成功”，而是：
   - 进一步扩大评测规模
   - 录制正式 demo
   - 讨论观测真实性和泛化能力
