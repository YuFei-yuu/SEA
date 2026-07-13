# SEA-Nav 第三周工作汇报

> 日期：2026-06-20  
> 项目：基于 SEA-Nav 的复杂多动态障碍物安全导航  
> 本周主题：从简单动态避障验证推进到 Motion-Aware SEA-Nav 复杂动态避障基座

---

## 一、本周工作概述

第二周已经完成了 `go2_pos_dynamic_1/2/3` 三类简单动态障碍物环境的复现和验证，说明在原 SEA-Nav 架构上加入动态障碍物是可行的。但导师提出的新任务更进一步：不仅要“有动态障碍物”，还要调研当前领域方法，思考动态障碍物处理机制，并改进 SEA-Nav，使其能够面向多动态障碍复杂环境实现更安全的避障。

因此本周工作的核心问题是：

**如何把 SEA-Nav 从简单动态障碍物场景推进到复杂多动态障碍物场景，同时让改造路线具有文献依据、工程可训练性和后续论文扩展空间。**

本周围绕这件事完成了五条主线：

1. 系统调研动态避障领域文献，梳理 VO/TTC、CBF、安全过滤器、Reach-Avoid、图网络和注意力建模等方法，并结合 SEA-Nav 当前架构做取舍。
2. 设计 Motion-Aware SEA-Nav 改造路线，在保持原 830 维复杂任务观测结构的前提下，引入相对速度、TTC 和 velocity-aware dynamic CBF。
3. 重构 `go2_pos_dynamic_complex` 仿真环境，将动态障碍物从在线重采样/在线推进改为 episode 初始化阶段预生成轨迹，降低 reset 和 step 阶段的工程瓶颈。
4. 完成复杂静态障碍物重新布局、动态障碍物轨迹检查、几何碰撞检测、训练闭环、评估脚本和录像脚本的持续修正。
5. 通过两轮 `64 env / 500 iterations` 训练定位问题：第一次训练 `success=0`，发现根因是 early reset 和接触力误触发；修复后 Easy complex 已经能学出非零成功率，但仍存在近目标绕远和动态障碍让行方向错误的问题。

本周阶段性结论是：

- 复杂动态环境已经基本搭建完成，训练、评估和视频检查链路已经打通。
- 只靠“动态障碍物加入 ray”不足以解决复杂动态避障，必须显式建模相对速度和短时碰撞风险。
- velocity-aware dynamic CBF 能作为 SEA-Nav 原安全层的自然扩展，但其安全距离、干预强度和奖励塑形需要与几何碰撞阈值严格对齐。
- 当前 Easy complex 已经从完全训练不动推进到能获得非零成功率，但策略行为仍不稳定，下一阶段重点应放在近目标收敛和 pass-behind 动态让行策略上。

---

## 二、文献调研与研究路线选择

本周首先阅读和整理了两版动态障碍物文献调研文档：

- [dynamic_obstacle_literature_review_v1.md](/home/sea_ws/src/docs/dynamic_obstacle_literature_review_v1.md:1)
- [dynamic_obstacle_literature_review_v2.md](/home/sea_ws/src/docs/dynamic_obstacle_literature_review_v2.md:1)

调研的目的不是泛泛列论文，而是回答一个具体问题：

**动态避障领域的前沿思想，哪些能短期落到 SEA-Nav 当前代码里，哪些应该作为后续论文增强方向保留。**

### 2.1 VO / TTC 类方法

DRL-VO、NavRL 等工作给出的核心启发是：动态避障不能只看当前距离，还必须看相对速度和未来短时碰撞风险。

传统静态避障中，离障碍物近就危险；但动态避障中，离得近不一定危险，正在远离的障碍物可以放松约束；离得稍远也不一定安全，快速横穿或迎面靠近的障碍物需要提前让行。因此，动态障碍物状态至少应包含：

```text
rel_x, rel_y, rel_vx, rel_vy, radius, ttc, valid
```

本周最终将这些信息做成 dynamic token，追加到复杂任务观测中。

### 2.2 CBF / Safety Filter 类方法

SEA-Nav 原始架构已经有 LSE-CBF 安全层，这是本项目最重要的可继承资产。Online CBF、One Filter、NavRL safety shield 等工作都说明：在动态环境中，策略不应该独自承担所有安全责任，而应采用：

```text
nominal policy 负责效率和目标推进
safety layer 负责最后安全约束
```

因此本周没有推翻原 PPO + CBF 框架，而是在原静态 ray CBF 后增加一个无训练参数的 `DynamicTokenCBFLayer`。这样做有三个好处：

- 与原 SEA-Nav 的安全过滤思想一致。
- 代码改动可控，便于调试和解释。
- 比直接引入全新 reachability network 更容易先跑通训练基座。

### 2.3 Reach-Avoid / Recovery Policy 类方法

Agile But Safe 和 One Filter 这类工作更进一步，用 reach-avoid value network 或 observation-conditioned safety filter 判断何时接管策略。它们的研究价值很高，也适合作为后续论文增强点。

但本周没有直接实现这类方法，原因是：

- 需要新的安全值函数训练目标，工程量明显更大。
- 当前复杂环境本身还没有稳定训练基座，过早加入 reachability 网络会让问题来源更难区分。
- 现阶段更应该先证明相对速度、TTC、动态 CBF 和 near-miss replay 在当前 SEA-Nav 上能形成可训练闭环。

### 2.4 图网络 / Transformer / Attention 类方法

HEIGHT、Intention-Aware CrowdNav、Subgoal Attention 等工作强调在人群环境中显式建模 agent-agent 交互。这类方法适合处理人群导航和社会导航问题，尤其适合动态障碍物数量变化、障碍物间有交互意图的场景。

本项目当前的动态障碍物是轨迹驱动的“可感知危险物”，暂时不模拟障碍物之间的真实交互意图。因此本周没有引入 GNN 或 Transformer，而是保留 top-K dynamic token 接口，为后续扩展留下空间。

### 2.5 本周最终研究路线

综合文献调研和仓库现状，本周将研究路线确定为：

```text
Motion-Aware SEA-Nav:
在原 SEA-Nav 的 PPO + LSE-CBF 基座上，
加入动态障碍物相对速度、TTC、动态 CBF 和 near-miss replay，
形成可解释、可训练、可逐步扩展的复杂动态避障框架。
```

这条路线的定位是：先做一个可靠的 motion-aware 安全导航基座，再考虑 GNN、Transformer、Reach-Avoid value network 或 recovery policy 等更大改造。

---

## 三、复杂动态障碍物环境搭建

### 3.1 从简单动态障碍到 complex 任务

第二周的 `go2_pos_dynamic_1/2/3` 主要验证简单动态避障可行性。本周进一步新增并推进 `go2_pos_dynamic_complex`，目标是：

- 支持多动态障碍物。
- 支持多种运动模式。
- 保留复杂静态障碍物结构。
- 保持观测维度和训练链路稳定。
- 可以用于后续 curriculum 训练和消融实验。

复杂任务的观测维度保持为：

```text
12 props + 41 rays + 2 goal + 4 * 7 dynamic tokens = 83 one-step obs
83 * 10 history = 830
```

dynamic token 语义为：

```text
[rel_x, rel_y, rel_vx, rel_vy, radius, ttc, valid]
```

这保证了策略既能看到原有 ray 结构，也能显式看到 top-K 动态障碍物的运动信息。

### 3.2 静态障碍物重新布局

最初复杂静态障碍物主要集中在房间中央，房间四周形成了一个较空旷的“外圈通道”。视频观察后发现，机器人可能利用这个空旷区域绕远路，从而回避真正的复杂避障问题。

因此本周调整了复杂房间中的静态障碍物布局，在整个房间内更大范围分布障碍物，形成更均匀的导航约束。当前配置位于 [go2_pos_config.py](/home/sea_ws/src/training/legged_gym/legged_gym/envs/go2/go2_pos_config.py:14)，`DYNAMIC_ROOM_COMPLEX_OBSTACLE_BOXES` 已包含 19 个障碍物盒子，覆盖房间下部、中部和上部区域。

这一步的意义是：让机器人不能简单“钻空子”，必须在静态结构和动态障碍物共同约束下学习导航。

### 3.3 动态障碍物改为 episode 预生成轨迹

早期 complex 环境采用在线采样和在线推进动态障碍物。在高密度场景中，这种方式出现两个问题：

- reset 和 step 中存在较多 Python 循环，性能瓶颈明显。
- 动态障碍物之间、动态障碍物和静态障碍物之间的接触/重采样会导致抖动和不稳定。

结合导师要求，本周将复杂任务动态障碍物改为：

```text
episode 初始化时一次性生成轨迹
step 阶段只按 trajectory step 读取当前位置和速度
忽略动态障碍物之间的碰撞和穿模
只保证机器人与动态障碍物的几何碰撞可以正常检测
```

核心配置位于 [go2_pos_config.py](/home/sea_ws/src/training/legged_gym/legged_gym/envs/go2/go2_pos_config.py:434)：

```python
trajectory_mode = "episode_precomputed"
trajectory_extra_horizon = 0.8
disable_simulation_contacts = True
```

环境中新增的主要轨迹缓存包括：

```text
dynamic_traj_pos
dynamic_traj_vel
dynamic_traj_step
```

形状为：

```text
[num_envs, max_dynamic_obstacles, trajectory_len, 2]
```

这一步解决了两个关键工程问题：动态障碍物运动更稳定，训练过程不再被在线重采样和 PhysX 接触求解拖慢。

### 3.4 动态障碍物运动模式

当前复杂任务支持：

- `linear_crossing`
- `linear_diagonal`
- `circular`
- `figure_eight`

其中线性轨迹改成闭合 ping-pong 轨迹，圆形和 8 字轨迹使用预采样相位轨迹。这样既保留了足够多的动态交互模式，也保证了每个 episode 内障碍物运动是可复现、可检查的。

本周还录制了障碍物检查视频和策略视频，用于确认动态障碍物不是原地抖动，而是在预生成轨迹上运动。

---

## 四、网络与安全层修改

### 4.1 保留原 SEA-Nav 主干

本周没有推翻原始策略结构，而是尽量延续 SEA-Nav 的设计：

```text
history obs
  -> actor encoder
  -> nominal nav action u_bar = [vx, vy, yaw_rate]
  -> static LSE-CBF
  -> dynamic token CBF
  -> safe action u_s
```

这样的选择有两个考虑：

- 从研究上看，它延续了 SEA-Nav “策略 + 安全层”的核心思想。
- 从工程上看，它避免了在复杂环境还不稳定时引入过多新模块，方便定位问题。

### 4.2 新增 velocity-aware DynamicTokenCBFLayer

本周在 [cbf_lse_layer.py](/home/sea_ws/src/training/rsl_rl/rsl_rl/modules/cbf_lse_layer.py:69) 中新增了 `DynamicTokenCBFLayer`。

对每个 dynamic token，定义：

```text
p_rel = p_obs - p_robot
v_rel = v_obs - v_robot
h = ||p_rel||^2 - d_safe^2
dot(h) = 2 * p_rel^T * (v_obs - u_robot)
dot(h) + alpha * h >= 0
```

直观含义是：

- 如果障碍物远离机器人，约束会放松。
- 如果障碍物正在靠近，CBF 会更早修正机器人速度。
- yaw action 暂时保持透传，只对 `[vx, vy]` 做安全投影。

在 [cbf_actor_critic.py](/home/sea_ws/src/training/rsl_rl/rsl_rl/modules/cbf_actor_critic.py:157) 中，动作处理顺序改为：

```text
u_bar
  -> static/ray LSE-CBF
  -> u_static_safe
  -> dynamic token CBF
  -> u_s
```

同时保留：

```text
u_bar
u_static_safe
u_s
```

用于后续分析 intervention loss 和安全层介入行为。

### 4.3 新增训练和评估指标

为了不再只看 success，本周补充了多项日志：

```text
active_dynamic_count
min_dynamic_clearance
dynamic_cbf_intervention_rate
shield_intervention_step_rate
dynamic_cbf_intervention_step_rate
reset_goal
reset_contact50
reset_terminate_contact
reset_dynamic_collision
```

这些指标帮助定位训练失败到底是：

- 动态障碍物碰撞太多。
- 静态/机身接触导致 reset。
- 安全层过度介入。
- episode 太短，没有有效学习样本。
- 目标推进奖励不足。

这也是本周后半段能够定位 `success=0` 根因的关键。

---

## 五、排错过程与关键问题修复

### 5.1 问题一：复杂环境初始版本训练过慢

最初复杂环境虽然能 reset/step，但在 `64 env` 训练时明显卡顿。分析后发现，动态障碍物在线初始化和在线推进中存在较多逐 env、逐 slot 的 Python 逻辑，尤其是障碍物间距检查、bbox 有效性检查和线性轨迹推进。

解决方式：

- 将复杂任务动态障碍物改为 episode 预生成轨迹。
- step 阶段用张量索引更新位置和速度。
- 不再因为动态障碍物互相穿过或穿过静态障碍物而重采样。

结果：

```text
32 env / 200 step 可以正常运行
obs_shape = (32, 830)
dynamic_traj_step 随 episode 前进
训练不再卡在动态障碍物初始化/推进循环上
```

### 5.2 问题二：动态障碍物原地抖动

视频检查时发现部分动态障碍物几乎在原地抖动，没有按预期运动。这个问题可能来自两个方面：

- 动态障碍物仍被 PhysX 接触求解影响。
- 在线重采样/边界反弹逻辑在局部区域反复触发。

解决方式：

- complex 任务启用 `disable_simulation_contacts = True`。
- 动态障碍物由预生成轨迹驱动，不再依赖在线碰撞反弹。
- inactive slot 统一放到远处，避免参与 ray 和碰撞。

结果：

- 动态障碍物运动更平滑。
- 录像中能观察到障碍物沿预生成轨迹运动。
- 后续机器人碰撞仍由几何距离触发，不依赖动态障碍物 PhysX 接触。

### 5.3 问题三：第一次 `64 env / 500 iterations` 训练完全失败

第一次完整训练 run 为：

```text
training/legged_gym/logs/Go2_pos_dynamic_complex/06_20_05-14-50_
```

训练结果：

```text
success = 0
safe_success = 0
avg_dynamic_collision_count 接近 0
avg_body_collision_count 很低
timeout_rate = 0
```

表面看像是安全层太保守，策略不敢走。但进一步检查发现 episode 极短：

```text
mean_episode_length 约 9 step
mean_episode_duration 约 0.17 s
```

这说明策略不是“走了很久没到”，而是刚开始就被 reset，几乎没有有效 PPO 经验。

### 5.4 问题四：contact force hard reset 误触发

继续排查 reset 原因后，发现主要问题是 `contact_force > 50` 的 hard reset 太宽泛。原逻辑对所有刚体检查横向接触力，导致足端正常落地、摩擦、初始化冲击也可能触发 reset。

修复方式：

- 新增 reset reason 指标，记录每种 reset 来源。
- 将 hard contact reset 收窄到 `termination_contact_indices`，主要是 base/head 等真正危险部位。
- 增加 `hard_contact_warmup_steps = 10`，初始化前几步只记录不 reset。

修复后 `16 env / 30 step` 冒烟结果：

```text
obs_shape (16, 830)
active_min 2
active_max 4
first_step_dones 0
done_total_30 0
reset_contact50 0.0
reset_spawn_collision 0.0
reset_dynamic_collision 0.0
```

这一步非常关键：它说明之前训练失败的首要原因不是算法，而是环境终止逻辑切断了训练样本。

### 5.5 问题五：Isaac Gym 退出阶段 segmentation fault

多次训练和评估结束时出现：

```text
Segmentation fault (core dumped)
return code 139
```

但检查发现：

- checkpoint 已正常保存。
- `train_metrics.csv` 已正常写出。
- `torch.load()` 可以读取 checkpoint。
- 策略录像可以加载模型并运行。

因此目前判断这是 Isaac Gym 析构/清理阶段问题，而不是 PPO 更新失败或 Python 逻辑错误。本周记录中均按“训练完成、退出清理阶段崩溃”处理。

---

## 六、训练实验与结果分析

### 6.1 Easy complex curriculum

修复 reset 逻辑后，没有直接继续 Hard complex，而是将默认训练难度调成 Easy 起点：

```text
dynamic obstacle count_range = [2, 4]
speed_range = [0.15, 0.35]
enable_near_miss_replay = False
goal_reached_time = 20
goal_progress = 6.0
```

这样做的理由是：在复杂动态环境中，如果随机初始化策略一开始就面对 6-10 个中高速障碍物，PPO 很容易只学到保守动作或被安全层完全接管。先在 Easy 阶段建立目标推进能力，再逐步恢复难度更符合 curriculum learning 思路。

### 6.2 修复后的 `64 env / 500 iterations`

训练 run：

```text
training/legged_gym/logs/Go2_pos_dynamic_complex/06_20_07-23-45_
```

保存产物：

```text
model_200.pt
model_300.pt
model_400.pt
model_500.pt
train_metrics.csv
```

最后训练窗口关键指标：

```text
last success:                       0.2917
last safe_success:                  0.2917
last mean_episode_length:         853.91
last dynamic_collision_count:       0.7083
last shield_intervention_step_rate: 0.9973
last dynamic_cbf_intervention_step_rate: 0.3016
```

训练过程最好窗口：

```text
max success:      1.0000 @ iteration 250
max safe_success: 1.0000 @ iteration 250
```

最后 10 个记录窗口平均：

```text
success:                 0.3542
safe_success:            0.1208
mean_episode_length:   744.89
reset_dynamic_collision: 0.1479
reset_terminate_contact: 0.4635
```

这说明修复后已经不再是 `success=0` 的死局，episode length 也从十步以内提升到数百步，PPO 已经能获得有效长轨迹样本。

### 6.3 `model_500` 固定评估和录像

`model_500` 的 40/50 episode 快速评估结果：

```text
success_rate:                0.4000
safe_success_rate:           0.2750
avg_dynamic_collision_count: 0.3250
avg_body_collision_count:    0.7000
avg_active_dynamic_count:    3.0750
```

录制视频：

```text
training/legged_gym/logs/Go2_pos_dynamic_complex/exported/06_20_07-23-45_model_500.mp4
```

视频信息：

```text
10 episodes
9621 frames
1000 x 1000
30 FPS
约 128 MB
```

视频中 10 个 episode 里成功 6 个，失败 4 个。这个结果说明当前策略已经具备一定避障能力，但仍不是稳定策略。

### 6.4 视频暴露出的行为问题

人工检查视频后发现两个典型问题：

```text
1. 目标点已经很近，但机器人绕了较大远路，没有稳定向目标收敛。
2. 动态障碍物从右向左运动时，机器人有时仍向左侧躲避，反而撞到障碍物未来位置。
```

这两个问题说明当前策略并不只是“训练不够久”，而是 reward 和 safety layer 对两个关键行为塑形不足：

- 近目标阶段缺少强收敛约束和绕远惩罚。
- 动态障碍阶段缺少 pass-behind 或 future-clearance 方向性奖励。

另一个重要发现是：

```text
dynamic collision threshold = 0.65
DynamicTokenCBFLayer 默认安全距离约为 radius 0.32 + margin 0.20 = 0.52
```

也就是说，当前 CBF 认为可接受的距离可能已经小于几何碰撞判定阈值。这会导致“安全层放行，但环境判定碰撞”的不一致，需要优先修复。

### 6.5 06-21 奖励函数二次优化

根据 `model_500` 视频暴露出的近目标绕远和横穿动态障碍让行方向错误，本次没有直接增加动态障碍物数量，而是先把问题拆成三个更可控的子问题：

```text
1. 安全层边界是否和环境碰撞判定一致。
2. 近目标阶段是否仍然强制机器人朝目标收敛。
3. 横穿动态障碍场景中，策略是否知道应该从障碍物未来轨迹后方通过。
```

因此奖励函数调整的核心思路不是简单“多给到达奖励”或“多罚碰撞”，而是把不同阶段的行为分开塑形：

```text
远离目标时：继续强调 goal_progress，保证整体推进。
靠近目标时：压制绕圈、横向漂移和反向离目标运动。
动态交互时：用相对速度和 TTC 引导 pass-behind，而不是只按当前距离避障。
安全层上：让 CBF 安全距离不小于几何碰撞阈值，避免训练信号互相矛盾。
```

具体修改如下。

第一，修复动态 CBF 与碰撞阈值不一致问题：

```text
dynamic_collision threshold = 0.65
DynamicTokenCBFLayer safety_margin = 0.35
dynamic obstacle radius 约 0.32
实际动态安全距离约为 0.32 + 0.35 = 0.67
```

这样动态 CBF 的默认安全边界略大于环境动态碰撞阈值，避免出现“CBF 认为安全、环境却判碰撞”的反向训练信号。

第二，缩短近目标驻留要求并提高紧目标奖励：

```text
goal_reached_time: 20 -> 12
reach_pos_target_tight: 10.0 -> 15.0
```

这一改动的目的，是让机器人在进入目标邻域后更快完成 episode，而不是在目标附近继续探索大幅绕行路径。

第三，新增近目标局部奖励和惩罚：

```text
near_goal_radial_reward = 4.0
negative_progress_penalty = -5.0
near_goal_orbit_penalty = -4.0
near_goal_command_penalty = -2.0
near_goal_stop_reward = 4.0
```

其中各项分工如下：

- `near_goal_radial_reward`：只在近目标区域内奖励朝目标方向的径向速度，避免策略只靠切向运动刷存活时间。
- `negative_progress_penalty`：当机器人已经接近目标却远离目标时给惩罚，直接针对“到目标旁边又绕出去”的问题。
- `near_goal_orbit_penalty`：惩罚近目标区域内过大的切向速度，以及径向接近不足的绕圈行为。
- `near_goal_command_penalty`：从动作命令层面惩罚近目标时过快、横向、反向的导航指令，促使 nominal policy 自己给出更稳定的收敛动作。
- `near_goal_stop_reward`：在距离目标很近且机体速度、偏航角速度较小时给奖励，鼓励到点后稳定停住。

第四，新增动态让行方向奖励：

```text
dynamic_avoid_direction_reward = 1.0
ttc_threshold = 1.2
min_obstacle_speed = 0.05
min_lateral_speed = 0.03
```

该奖励使用 dynamic token 中的：

```text
rel_pos, rel_vel, radius, ttc, valid
```

结合当前机器人速度估计障碍物运动方向。当 TTC 较小且障碍物确实在横穿时，计算机器人相对障碍物运动方向的位置关系：

```text
pass_score = dot(rel_pos, obstacle_direction) / ||rel_pos||
```

在当前 token 定义 `rel_pos = obstacle_pos - robot_pos` 下，`pass_score > 0` 表示机器人更倾向于位于障碍物运动方向的后方，也就是从动态障碍物未来轨迹后方通过；`pass_score < 0` 则更可能抢到障碍物前方或朝障碍物未来位置绕行。因此该奖励用于修正视频中“障碍物从右向左运动，机器人仍向左绕导致撞到未来位置”的问题。

第五，补充诊断日志和可视化字段：

```text
ubar_goal_angle
ustatic_goal_angle
usafe_goal_angle
ubar_norm
usafe_norm
future_dynamic_clearance
pass_behind_score
```

这些字段可以帮助区分问题来源：

```text
如果 u_bar 已经偏离目标方向，说明 nominal policy 学偏了。
如果 u_bar 正常但 u_s 偏离，说明 CBF 修正过强或方向不合理。
如果 pass_behind_score 长期为负，说明动态让行策略仍在抢障碍物前方。
如果 future_dynamic_clearance 下降而当前 clearance 尚可，说明策略缺少前瞻性。
```

### 6.6 128 env / 500 iterations 训练与 model_400 录像

完成上述奖励和日志修改后，启动了一轮 `128 env / 500 iterations` 训练：

```text
training/legged_gym/logs/Go2_pos_dynamic_complex/06_21_04-28-54_
```

训练保存了：

```text
model_200.pt
model_300.pt
model_400.pt
model_500.pt
train_metrics.csv
analysis/report.md
analysis/*.png
```

分析报告显示：

```text
last_iteration: 490
last_timesteps: 3016704
best_success: 1.0000
best_safe_success: 0.9167
```

从 checkpoint 对应的训练窗口看，`model_400` 是本轮最值得录像检查的已保存模型：

```text
iteration 400 / model_400:
success:                 0.8958
safe_success:            0.8958
dynamic_collision_count: 0.0208
body_collision_count:    0.0833
total_collision_count:   0.1042
time_to_goal:           16.2950 s
pass_behind_score:       0.1614
```

但训练后段出现明显退化，最终窗口变为：

```text
iteration 490:
success:                 0.5417
safe_success:            0.2083
dynamic_collision_count: 0.4583
total_collision_count:   0.8542
pass_behind_score:      -0.1172
```

这说明本次 reward 方向是有效的，尤其在 400 iteration 附近形成了更安全的策略；但继续训练到 500 iteration 后，策略又出现动态安全性下降和 pass-behind 方向退化。后续不能默认选择最后一个 checkpoint，而应基于 50 episode 评估和视频选择最佳 checkpoint。

随后录制 `model_400` 的 10 episode 视频：

```text
training/legged_gym/logs/Go2_pos_dynamic_complex/exported/06_21_04-28-54_model_400.mp4
```

视频信息：

```text
10 episodes
13838 frames
1000 x 1000
30 FPS
约 193 MB
```

脚本输出中 10 个 episode 的结果为：

```text
成功 6 次
失败 4 次
动态碰撞只出现在第 4 个 episode
```

阶段性判断：

- 相比原 `model_500`，`model_400` 的训练窗口安全指标明显更好，说明近目标奖励、CBF 边界对齐和 pass-behind 奖励方向基本正确。
- 录像仍未达到稳定验收标准，10 episode 中仍有 4 次失败，说明策略还没有完全解决近目标收敛和复杂交互下的泛化问题。
- 后段训练退化提示当前奖励权重可能存在竞争：策略一方面被近目标奖励牵引，另一方面仍受动态避障、安全层修正、接触终止和探索噪声影响，训练到后期可能从较优行为漂移出去。
- 下一阶段必须把“checkpoint 选择、50 episode 固定评估、视频诊断、消融对比”作为闭环，而不是只看最终训练曲线。

---

## 七、本周形成的阶段性认识

### 7.1 动态避障的关键不是“障碍物会动”，而是“策略理解运动趋势”

一开始容易把动态避障理解成在仿真中加入 moving actor。但本周的文献调研和实验都说明，仅有 moving actor 不够。真正关键的是：

- 观测中是否有相对速度。
- 奖励中是否有短时碰撞风险。
- 安全层是否考虑障碍物未来运动。
- 策略是否学会从障碍物未来轨迹后方通过。

因此后续工作不能只增加障碍物数量，而要让机器人具备 motion-aware 决策能力。

### 7.2 安全层既是优势，也可能造成训练偏差

SEA-Nav 的 CBF 安全层让策略在训练早期不至于大量撞障碍，是一个很强的优势。但本周训练也显示：

```text
shield_intervention_step_rate 长期接近 1.0
```

这意味着实际执行动作大量来自安全层修正，而不是 nominal policy 自己学会安全动作。如果不处理这个问题，策略可能形成依赖安全层的行为，表现为：

- 到达效率低。
- 目标附近绕远。
- 动态障碍物让行方向不稳定。

后续需要增加 `u_bar / u_static_safe / u_s` 的方向分析，让 nominal policy 学会提出更接近安全动作的命令。

### 7.3 工程诊断指标非常重要

第一次 500 iterations 训练失败时，如果只看 `success=0`，容易误判为算法失败。但通过新增 reset reason 后，发现真正问题是 contact reset 误触发导致 episode 极短。

这说明复杂强化学习任务中，不能只看最终成功率。必须同时记录：

```text
episode length
reset reason
collision type
dynamic collision
body collision
near miss
TTC
CBF intervention
time to goal
```

这些指标是把“训练失败”变成“可解释问题”的前提。

### 7.4 当前路线具有可解释的后续扩展空间

本周没有直接上 GNN、Transformer 或 Reach-Avoid value network，但当前设计已经预留了扩展接口：

- dynamic token 可以接 GNN 或 attention encoder。
- near-miss replay 可以扩展成 ABS 风格恢复策略训练。
- dynamic CBF 可以扩展成 observation-conditioned safety filter。
- future-clearance reward 可以进一步发展成 VO-like differentiable risk。

这使得本周工作既是工程基座，也是后续研究路线的起点。

---

## 八、下一步优化计划

经过 06-21 这轮奖励调整后，下一阶段不应急着进入 Medium/Hard complex，而应先把 Easy complex 中已经出现的较优策略稳定下来。当前最关键的问题已经从“能不能训练出成功率”转变为：

```text
如何防止后段训练退化；
如何让 model_400 附近的安全成功表现可复现；
如何进一步减少近目标失败和动态交互失败。
```

### 8.1 先做固定评估，确认 model_400 是否真正优于 model_500

当前 `model_400` 的训练窗口指标明显优于 `model_500` 末尾窗口，但训练窗口不是严格固定评估。下一步首先要对 `model_300/model_400/model_500` 做固定 50 episode 评估：

```text
evaluate_checkpoints.py:
model_300.pt
model_400.pt
model_500.pt
每个 checkpoint 50 episodes
```

重点看：

```text
success_rate
safe_success_rate
avg_dynamic_collision_count
avg_body_collision_count
avg_near_miss_count
avg_time_to_goal
avg_future_dynamic_clearance
avg_pass_behind_score
```

如果固定评估也确认 `model_400` 最优，则后续以 `model_400` 作为 Easy complex 当前基线，而不是使用最后的 `model_500`。

### 8.2 稳定训练后段，降低 400 iteration 后的策略漂移

本轮训练在 400 iteration 附近达到较好表现，490 iteration 明显退化。下一步需要围绕训练稳定性做小步实验：

```text
A. 降低后半程学习率或缩短单轮训练到 400 iterations。
B. 保存更密集 checkpoint，例如 300/350/400/450/500。
C. 观察 entropy、action_std、surrogate_loss 和 intervention_loss，判断是否探索噪声导致后期破坏已学策略。
D. 若后段退化重复出现，考虑在 300 iteration 后降低 entropy_coef 或启用 early-stop checkpoint selection。
```

验收标准不是最终 iteration 最好，而是固定评估下能稳定复现：

```text
safe_success_rate >= 0.45
avg_dynamic_collision_count <= 0.25
avg_body_collision_count <= 0.5
视频中不再频繁近目标大绕行
```

### 8.3 近目标奖励继续做消融，而不是一次性加大所有权重

当前近目标奖励已经加入，但 10 episode 录像仍有失败。下一步需要单独检查近目标行为：

```text
统计失败 episode 中最后 5 秒的 distance 曲线。
统计 near_goal_orbit_penalty、negative_progress_penalty、near_goal_stop_reward 的实际贡献。
对比 u_bar_goal_angle 与 usafe_goal_angle，判断绕远来自 policy 还是 CBF。
```

候选调参方向：

```text
near_goal_stop_reward: 4.0 -> 5.0 或 6.0
near_goal_orbit_penalty: -4.0 -> -5.0
near_goal_command_penalty 保持谨慎，避免过强导致近目标不敢动
goal_reached_time 保持 12，暂不继续降低，防止误判到达
```

如果视频中仍出现“离目标很近但继续绕行”，优先增强 `near_goal_orbit_penalty` 和 `near_goal_stop_reward`；如果出现“近目标停在外圈不进去”，则优先增强 `near_goal_radial_reward` 和 `reach_pos_target_tight`。

### 8.4 动态让行方向继续围绕 pass-behind 做验证

`model_400` 窗口中 `pass_behind_score` 为正，末尾窗口变为负，这说明 `dynamic_avoid_direction_reward` 有作用，但还不稳定。下一步应从视频和指标两侧验证：

```text
横穿障碍物 episode 中单独记录 pass_behind_score。
检查 pass_behind_score < 0 的 episode 是否更容易 near miss 或 dynamic collision。
按 motion_type 分组统计，优先看 linear_crossing 和 linear_diagonal。
```

候选调参方向：

```text
dynamic_avoid_direction_reward: 1.0 -> 1.5
ttc_threshold: 1.2 -> 1.5
near_miss ttc_threshold: 0.8 -> 1.0
near_miss: -0.3 -> -0.5
```

调参原则是先增强 direction reward，再增强 near-miss 惩罚。原因是只加大碰撞/near-miss 惩罚，策略可能学到保守停顿；而 direction reward 能告诉策略“往哪边让”。

### 8.5 保持 Easy complex，暂缓开启 near-miss replay 和 Medium curriculum

当前 Easy complex 仍没有稳定达到验收标准，因此暂时不建议立刻：

```text
增加动态障碍物数量到 [3, 5]
提高速度到 [0.20, 0.45]
开启 enable_near_miss_replay
引入 GNN/Transformer/Reach-Avoid 新网络
```

推荐顺序是：

```text
A0: 固定评估 model_300/400/500，选出真实最佳 checkpoint。
A1: 以最佳 checkpoint 对应配置复训 3 个随机种子，确认 model_400 优势是否可复现。
A2: 近目标奖励小消融，只动一个权重，观察近目标绕远是否下降。
A3: pass-behind 奖励小消融，只动一个权重，观察横穿障碍方向是否改善。
A4: 达到 Easy complex 验收后，再开启 near-miss replay。
A5: replay 稳定后再进入 Medium curriculum。
```

### 8.6 训练记录与可视化要求

后续每轮训练仍需严格保存和检查：

```text
train_metrics.csv
analysis/report.md
analysis/rl_success_rates.png
analysis/safety_events.png
analysis/safety_clearance_ttc.png
analysis/action_goal_alignment.png
analysis/dynamic_direction.png
analysis/reward_breakdown.png
eval_checkpoints.csv
eval_checkpoints_summary.md
最佳 checkpoint 10 episode 视频
```

尤其要把 `dynamic_direction.png`、`action_goal_alignment.png` 和录像一起看。仅凭训练曲线不能判断近目标绕远和动态让行方向是否真正改善。

---

## 九、本周产物汇总

### 9.1 文档产物

```text
docs/dynamic_obstacle_literature_review_v1.md
docs/dynamic_obstacle_literature_review_v2.md
docs/dynamic_complex_quick_validate_report.md
docs/motion_aware_dynamic_complex_validate_report.md
docs/week3report.md
```

### 9.2 代码相关产物

主要涉及：

```text
training/legged_gym/legged_gym/envs/base/legged_robot_pos_dynamic.py
training/legged_gym/legged_gym/envs/go2/go2_pos_config.py
training/legged_gym/legged_gym/scripts/evaluate_dynamic.py
training/legged_gym/legged_gym/scripts/play_record.py
training/rsl_rl/rsl_rl/modules/cbf_actor_critic.py
training/rsl_rl/rsl_rl/modules/cbf_lse_layer.py
training/rsl_rl/rsl_rl/runners/on_policy_runner.py
```

### 9.3 模型与视频产物

第一次失败训练：

```text
training/legged_gym/logs/Go2_pos_dynamic_complex/06_20_05-14-50_
```

修复后 Easy complex 训练：

```text
training/legged_gym/logs/Go2_pos_dynamic_complex/06_20_07-23-45_
```

奖励函数二次优化后训练：

```text
training/legged_gym/logs/Go2_pos_dynamic_complex/06_21_04-28-54_
```

关键文件：

```text
model_200.pt
model_300.pt
model_400.pt
model_500.pt
train_metrics.csv
analysis/report.md
analysis/*.png
```

`model_500` 视频：

```text
training/legged_gym/logs/Go2_pos_dynamic_complex/exported/06_20_07-23-45_model_500.mp4
```

奖励函数二次优化后的 `model_400` 视频：

```text
training/legged_gym/logs/Go2_pos_dynamic_complex/exported/06_21_04-28-54_model_400.mp4
```

---

## 十、组会汇报时可强调的要点

本周工作可以概括为三句话：

第一，文献调研后明确了动态避障不是简单加 moving obstacle，而是要让 SEA-Nav 具备对相对速度、TTC 和未来短时风险的建模能力。

第二，工程上已经把 `go2_pos_dynamic_complex` 从不稳定的在线动态障碍物环境，改造成 episode 预生成轨迹、几何碰撞检测、动态 token 观测、velocity-aware CBF 的复杂动态避障基座。

第三，训练中经历了 `success=0` 的失败，但通过新增 reset reason 和训练指标，定位到 contact reset 误触发和 curriculum 过难问题；修复后 Easy complex 已经能学出非零成功率，并进一步通过近目标奖励、pass-behind 奖励和 CBF 边界对齐，在 `model_400` 附近获得了更好的安全成功窗口。

这说明本周的主要贡献不是单个指标提升，而是完成了从文献理解、方案取舍、环境重构、安全层扩展、训练诊断、奖励函数定向修复到下一步消融计划的完整闭环。下一阶段要重点解决的是训练后段退化、固定评估复现和视频中的残余失败行为。
