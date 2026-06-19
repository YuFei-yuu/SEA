下面这几篇可以作为你向导师汇报的“精华文献池”。我建议不要泛泛讲很多动态避障论文，而是围绕 **SEA-Nav 如何从静态/密集障碍导航扩展到动态障碍导航** 来组织。

## 一、SEA-Nav 当前基座能做什么、缺什么

SEA-Nav 项目本身是一个四足机器人导航强化学习框架，代码仓库包含 `training/legged_gym` 和 `training/rsl_rl`，训练入口是 `training/legged_gym/legged_gym/scripts/train.py --headless`，测试入口是 `play.py`。仓库 README 明确说明它基于 Isaac Gym、legged_gym 和 rsl_rl。([GitHub][1])

SEA-Nav 论文的核心是：用 PPO 训练高层导航策略，策略输出速度命令，再经过一个 **可微 LSE-CBF safety layer** 投影成安全速度；同时引入 **Adaptive Collision-State Initialization, ACSI**，也就是把碰撞前的关键危险状态反复拿来训练，提高密集障碍避障样本效率。([arXiv][2]) 当前论文中的观测包括机器人速度、角速度、局部目标点、41 条 2D LiDAR range，并用 10 帧历史观测编码；动作是高层速度命令 `[vx, vy, wz]`。([arXiv][2])

这说明 SEA-Nav 很适合作为动态避障基座，因为它已经有：**LiDAR 历史输入、PPO、速度命令控制、CBF 安全层、碰撞关键状态重放**。但它缺少动态避障最关键的两类信息：**障碍物速度/运动趋势** 和 **未来短时碰撞风险**。

---

## 二、建议重点读的 6 篇文献

| 文献                                                                                                        | 核心思想                                                                                                                                              | 对你拓展 SEA-Nav 的启发                                                              |
| --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| **DRL-VO: Learning to Navigate Through Crowded Dynamic Scenes Using Velocity Obstacles**, 2023            | 用短历史 LiDAR、附近行人运动学信息和 sub-goal 作为输入，在 RL reward 中加入 Velocity Obstacle 项，引导机器人主动避开行人。实验包含最多 55 个行人，并做了真实机器人零重训迁移。([arXiv][3])                      | 最直接。你可以把 SEA-Nav 的静态距离惩罚升级为 **VO / TTC 风险奖励**，并在观测中加入障碍物相对速度。                 |
| **Intention Aware Robot Crowd Navigation with Attention-Based Interaction Graph**, 2022 / ICRA 2023       | 用 recurrent graph neural network + attention 建模机器人和人群之间的时空交互，并预测动态 agent 的未来轨迹，再把预测结果接入 model-free RL，避免机器人闯入他人的预期路径。([arXiv][4])                 | 如果你想做得更“具身智能/社会导航”，可以加一个 **K 个动态障碍物的图注意力编码器**，而不是只用 LiDAR range。              |
| **NavRL: Learning Safe Flight in Dynamic Environments**, 2024                                             | 用 PPO 学习 UAV 在静态与动态障碍中的导航，并引入受 Velocity Obstacles 启发的 safety shield，减少神经网络黑箱策略的失败。([arXiv][5])                                                    | 给 SEA-Nav 很好的工程路线：**RL 负责导航决策，VO/CBF 类安全层负责最后兜底**。                            |
| **Agile But Safe: Learning Collision-Free High-Speed Legged Locomotion**, RSS 2024                        | 四足机器人中使用 agile policy + recovery policy，并用 reach-avoid value network 决定何时切换到恢复策略，能在静态和动态障碍中高速安全运动。([arXiv][6])                                    | 与 SEA-Nav 同属四足机器人方向。启发是可以在 SEA-Nav 外再加一个 **危险恢复策略 / safety override policy**。 |
| **One Filter to Deploy Them All: Robust Safety for Quadrupedal Navigation in Unknown Environments**, 2024 | 提出 observation-conditioned reachability-based safety filter，用 LiDAR 输入动态构造安全区域，并在必要时覆盖 nominal controller，适用于不同四足控制器。([arXiv][7])                 | 很适合做 SEA-Nav 的“安全层升级版”：把当前 LSE-CBF shield 扩展成 **观测条件化的动态安全过滤器**。              |
| **REASAN: Learning Reactive Safe Navigation for Legged Robots**, 2025                                     | 面向复杂动态环境的四足反应式安全导航系统，包含 locomotion、safety shielding、navigation 三个 RL policy，以及处理 LiDAR 点云的 transformer-based exteroceptive estimator。([arXiv][8]) | 如果你想做系统级拓展，可以参考它的模块化：**导航策略 + 安全屏障 + 感知估计器**，比单一 end-to-end policy 更稳。        |

可选补充一篇：**Omni-Perception: Omnidirectional Collision Avoidance for Legged Locomotion in Dynamic Environments**, 2025。它强调直接用时序 3D LiDAR 点云做端到端四足避障，提出 PD-RiskNet 处理近端/远端风险，并支持 Isaac Gym 等仿真平台。([arXiv][9]) 如果你后续想从 SEA-Nav 的 2D sparse LiDAR 扩展到 3D LiDAR 或全向动态避障，这篇很有价值。

---

## 三、当前动态避障方法可以概括成 4 类

**第一类：反应式几何方法，如 VO / ORCA / DWA / MPC。**
这类方法显式利用障碍物相对位置和相对速度，计算哪些速度会导致未来碰撞。DRL-VO 的综述部分也把传统方法分成 costmap + DWA、Velocity Obstacle、MPC、人群交互模型等，并指出这些方法可解释但往往需要手调参数。([arXiv][3])

**第二类：强化学习直接学习动态避障策略。**
方法通常把 LiDAR、目标点、机器人状态、动态障碍物位置速度输入网络，让 PPO/SAC/IQL 等算法学习速度命令。DRL-VO、NavRL 都属于这一类，但它们不是“纯黑箱 RL”，而是把 VO、safety shield、sub-goal 等结构性先验加入策略或奖励中。([arXiv][3])

**第三类：预测/图网络/注意力建模。**
在人群导航中，单纯看到当前位置不够，因为人会继续移动。Intention-Aware CrowdNav 预测动态 agent 的未来轨迹，并用注意力图网络建模机器人和人群的时空交互。([arXiv][4]) 这一类适合“社会导航”“具身智能导航”表述。

**第四类：安全过滤器 / CBF / Reachability。**
SEA-Nav 已经采用 CBF shield；ABS 和 One Filter 进一步把安全性做成可学习的 reach-avoid value 或 observation-conditioned safety filter。([arXiv][2]) 这一类最适合和 SEA-Nav 结合，因为 SEA-Nav 本身就是“RL policy + safety layer”。

---

## 四、我建议你的研究切入点

最合适的题目方向可以叫：

**Motion-Aware SEA-Nav: Dynamic Obstacle Avoidance for Quadruped Navigation with Velocity-Aware CBF and Critical Near-Miss Replay**

中文可以表述为：

**面向动态障碍物的运动感知 SEA-Nav：结合相对速度安全屏障与近碰撞状态重放的四足机器人导航方法**

核心创新不必太大，但要清晰：

原 SEA-Nav 主要利用 LiDAR 距离约束避障，你可以把它扩展为 **距离 + 相对速度 + 短时预测碰撞风险**。这样既延续 SEA-Nav 的 CBF/ACSI 思路，又引入动态避障领域最主流的 VO/TTC 思想。

---

## 五、具体怎么在 SEA-Nav 上加动态避障

### 1. 仿真环境加入动态障碍物

在 Isaac Gym 环境里增加若干 moving obstacle actor，比如圆柱、球、简化人形或者移动机器人。初期不需要做人群仿真，先做 3 类场景：

| 场景                           | 作用             |
| ---------------------------- | -------------- |
| 横穿型 obstacle crossing        | 测机器人能不能提前减速/绕行 |
| 迎面型 head-on obstacle         | 测相对速度风险判断      |
| 随机游走 random moving obstacles | 测泛化能力          |

每个动态障碍物维护状态：

```text
p_obs = [x, y]
v_obs = [vx, vy]
radius_obs
```

训练时随机化数量、速度、半径、起点、目标点。

---

### 2. 观测空间增加“运动信息”

SEA-Nav 现有观测已经有 LiDAR range 历史，但 range 历史只能间接反映运动。建议增加两种低成本特征：

```text
dynamic_obs_features:
[
  dx, dy,          # 障碍物相对机器人位置
  dvx, dvy,        # 障碍物相对机器人速度
  radius,
  ttc              # time-to-collision
] × K nearest obstacles
```

如果暂时不做目标检测/跟踪，也可以先从仿真中直接读动态障碍物 actor 的 root state，作为 privileged-but-realizable observation。论文里说明真实部署时可由 LiDAR tracking / multi-object tracking 替代。

更轻量的版本是只加 LiDAR range-rate：

```text
rho_dot_i = (rho_i(t) - rho_i(t-1)) / dt
```

这样可以判断某个方向的障碍物是在靠近还是远离。

---

### 3. 把 SEA-Nav 的 CBF 改成动态 CBF

原 SEA-Nav 的 CBF 更像是对 LiDAR 距离做静态安全约束。动态障碍下，应该考虑相对速度。对第 `j` 个动态障碍物，可定义：

```text
p_rel = p_obs - p_robot
v_rel = v_obs - v_robot
h_j = ||p_rel||^2 - d_safe^2
```

动态 CBF 约束为：

```text
dot(h_j) + alpha * h_j >= 0
```

其中：

```text
dot(h_j) = 2 * p_rel^T * v_rel
```

如果策略输出的是机器人速度 `u = [vx, vy, wz]`，那么约束可以转成对 `vx, vy` 的限制。直觉是：
当障碍物离得近且正在靠近时，CBF 会强制策略减速、侧向绕行或停止；当障碍物远离时，约束会放松。

为了继承 SEA-Nav 的做法，可以仍然用 LSE 聚合多个障碍物约束：

```text
h_dyn = -1/k * log(sum_j exp(-k * h_j))
```

这样就不是只关心最近一个障碍物，而是平滑融合多个动态障碍的风险。

---

### 4. 奖励函数增加 VO / TTC 风险项

参考 DRL-VO，不建议只用碰撞惩罚，因为动态避障需要“提前让路”。可以加入：

```text
r_collision_dyn = -C                    # 撞到动态障碍
r_near_miss = -w1 * max(0, d_safe - d_min)
r_ttc = -w2 * exp(-TTC / tau)            # TTC 越小惩罚越大
r_progress = +w3 * goal_progress
r_smooth = -w4 * ||u_t - u_{t-1}||^2
```

其中 TTC 可以粗略计算为：

```text
TTC = - (p_rel · v_rel) / ||v_rel||^2
```

只在 `p_rel · v_rel < 0`，也就是双方正在接近时启用。

这个奖励比“撞了才罚”更适合动态障碍，因为它会鼓励策略提前规避。

---

### 5. 把 ACSI 扩展成 Near-Miss Replay

SEA-Nav 原来的 ACSI 是碰撞后回放碰撞前状态。动态避障里更好的做法是：

```text
如果 min_distance < threshold 或 TTC < threshold：
    记录 t - H 到 t 的状态片段
    后续 reset 到这段 near-miss 之前
```

也就是说，不一定等真的撞上，只要出现“差点撞上”的危险交互，就把它作为高价值训练样本反复训练。

这可以作为你的一个主要创新点：
**从 collision-state replay 扩展到 dynamic near-miss replay。**

---

## 六、推荐实验设计

### Baseline

| 方法                                       | 说明                            |
| ---------------------------------------- | ----------------------------- |
| SEA-Nav 原版                               | 不加动态观测、不加动态奖励、不改 CBF          |
| SEA-Nav + 动态障碍训练                         | 只加 moving obstacles，验证原方法是否失效 |
| SEA-Nav + TTC/VO reward                  | 只加动态风险奖励                      |
| SEA-Nav + Dynamic CBF                    | 只加动态安全层                       |
| SEA-Nav + Dynamic CBF + Near-Miss Replay | 你的完整方法                        |

### 指标

| 指标                       | 含义                    |
| ------------------------ | --------------------- |
| Success Rate             | 到达目标比例                |
| Collision Rate           | 与静态/动态障碍碰撞率           |
| Dynamic Collision Rate   | 单独统计动态障碍碰撞            |
| Minimum Distance         | 全程最小人/障碍距离            |
| TTC Violation Rate       | TTC 小于阈值的比例           |
| Time to Goal             | 到达时间                  |
| Path Length              | 路径长度                  |
| Average Speed            | 平均速度                  |
| Shield Intervention Rate | CBF/safety layer 介入比例 |

### 场景难度

| 难度     | 设置                 |
| ------ | ------------------ |
| Easy   | 1–2 个低速动态障碍        |
| Medium | 3–5 个中速障碍，含横穿      |
| Hard   | 5–10 个动态障碍，含迎面和交叉流 |
| OOD    | 训练没见过的速度、密度、障碍半径   |

---

## 七、最终建议你汇报时这样讲

你可以把研究路线凝练成三句话：

第一，SEA-Nav 已经通过 **PPO + LSE-CBF + ACSI** 解决了密集静态障碍中的安全高效导航问题，但动态障碍需要建模障碍物的运动趋势，而不只是距离。([arXiv][2])

第二，2022 年以来的动态导航文献主要有三条路线：**VO/TTC 引导的强化学习**、**图网络/轨迹预测的人群交互建模**、**CBF/reachability/safety filter 安全屏障**；其中 DRL-VO、NavRL、ABS、One Filter 与 SEA-Nav 最契合。([arXiv][3])

第三，本项目可以在 SEA-Nav 上做一个轻量但清晰的拓展：加入动态障碍仿真、相对速度/TTC 观测、VO/TTC 风险奖励、动态 CBF safety layer，以及 near-miss replay，从而形成一个“运动感知 SEA-Nav”框架。

[1]: https://github.com/11chens/SEA-Nav-Code "GitHub - 11chens/SEA-Nav-Code · GitHub"
[2]: https://arxiv.org/abs/2603.09460 "SEA-Nav: Efficient Policy Learning for Safe and Agile Quadruped Navigation in Cluttered Environments"
[3]: https://arxiv.org/abs/2301.06512 "DRL-VO: Learning to Navigate Through Crowded Dynamic Scenes Using Velocity Obstacles"
[4]: https://arxiv.org/abs/2203.01821 "[2203.01821] Intention Aware Robot Crowd Navigation with Attention-Based Interaction Graph"
[5]: https://arxiv.org/abs/2409.15634 "NavRL: Learning Safe Flight in Dynamic Environments"
[6]: https://arxiv.org/abs/2401.17583 "[2401.17583] Agile But Safe: Learning Collision-Free High-Speed Legged Locomotion"
[7]: https://arxiv.org/abs/2412.09989 "[2412.09989] One Filter to Deploy Them All: Robust Safety for Quadrupedal Navigation in Unknown Environments"
[8]: https://arxiv.org/abs/2512.09537 "[2512.09537] REASAN: Learning Reactive Safe Navigation for Legged Robots"
[9]: https://arxiv.org/abs/2505.19214 "Omni-Perception: Omnidirectional Collision Avoidance for Legged Locomotion in Dynamic Environments"
