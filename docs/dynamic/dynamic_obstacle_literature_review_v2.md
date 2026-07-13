下面这几篇可以作为你向导师汇报的“精华文献池”。我建议不要泛泛讲很多动态避障论文，而是围绕 **SEA-Nav 如何从静态/密集障碍导航扩展到动态障碍导航** 来组织。

## 一、SEA-Nav 当前基座能做什么、缺什么

SEA-Nav 项目本身是一个四足机器人导航强化学习框架，代码仓库包含 `training/legged_gym` 和 `training/rsl_rl`，训练入口是 `training/legged_gym/legged_gym/scripts/train.py --headless`，测试入口是 `play.py`。仓库 README 明确说明它基于 Isaac Gym、legged_gym 和 rsl_rl。([GitHub][1])

SEA-Nav 论文的核心是：用 PPO 训练高层导航策略，策略输出速度命令，再经过一个 **可微 LSE-CBF safety layer** 投影成安全速度；同时引入 **Adaptive Collision-State Initialization, ACSI**，也就是把碰撞前的关键危险状态反复拿来训练，提高密集障碍避障样本效率。([arXiv][2]) 当前论文中的观测包括机器人速度、角速度、局部目标点、41 条 2D LiDAR range，并用 10 帧历史观测编码；动作是高层速度命令 `[vx, vy, wz]`。([arXiv][2])

这说明 SEA-Nav 很适合作为动态避障基座，因为它已经有：**LiDAR 历史输入、PPO、速度命令控制、CBF 安全层、碰撞关键状态重放**。但它缺少动态避障最关键的两类信息：**障碍物速度/运动趋势** 和 **未来短时碰撞风险**。

---

## 二、建议重点读的 6 篇文献

| 文   献      | 核心思想     | 对你拓展 SEA-Nav 的启发   |
| ------- | ------- | -------- |
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



[1]: https://github.com/11chens/SEA-Nav-Code "GitHub - 11chens/SEA-Nav-Code · GitHub"
[2]: https://arxiv.org/abs/2603.09460 "SEA-Nav: Efficient Policy Learning for Safe and Agile Quadruped Navigation in Cluttered Environments"
[3]: https://arxiv.org/abs/2301.06512 "DRL-VO: Learning to Navigate Through Crowded Dynamic Scenes Using Velocity Obstacles"
[4]: https://arxiv.org/abs/2203.01821 "[2203.01821] Intention Aware Robot Crowd Navigation with Attention-Based Interaction Graph"
[5]: https://arxiv.org/abs/2409.15634 "NavRL: Learning Safe Flight in Dynamic Environments"
[6]: https://arxiv.org/abs/2401.17583 "[2401.17583] Agile But Safe: Learning Collision-Free High-Speed Legged Locomotion"
[7]: https://arxiv.org/abs/2412.09989 "[2412.09989] One Filter to Deploy Them All: Robust Safety for Quadrupedal Navigation in Unknown Environments"
[8]: https://arxiv.org/abs/2512.09537 "[2512.09537] REASAN: Learning Reactive Safe Navigation for Legged Robots"
[9]: https://arxiv.org/abs/2505.19214 "Omni-Perception: Omnidirectional Collision Avoidance for Legged Locomotion in Dynamic Environments"
