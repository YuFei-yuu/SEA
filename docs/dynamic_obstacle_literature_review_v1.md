# 动态障碍物导航文献调研与项目改进方向

> 调研目标：了解 2022 年以来强化学习 / 具身智能在动态障碍物环境中的导航方法，为 SEA-Nav 项目（PPO + CBF 安全层，四足机器人导航）提供改进思路。

---

## 目录

- [动态障碍物导航文献调研与项目改进方向](#动态障碍物导航文献调研与项目改进方向)
  - [目录](#目录)
  - [1. 当前项目状态概述](#1-当前项目状态概述)
  - [2. 精选文献详细整理](#2-精选文献详细整理)
    - [2.1 Agile But Safe (RSS 2024)](#21-agile-but-safe-rss-2024)
      - [关键内容](#关键内容)
    - [2.2 Online CBF for Multi-Agent Navigation (MRS 2023)](#22-online-cbf-for-multi-agent-navigation-mrs-2023)
      - [关键内容](#关键内容-1)
    - [2.3 HEIGHT (2024)](#23-height-2024)
      - [关键内容](#关键内容-2)
    - [2.4 Subgoal-Driven Navigation with Attention (2023)](#24-subgoal-driven-navigation-with-attention-2023)
      - [关键内容](#关键内容-3)
    - [2.5 HJ Reachability in RL: A Survey (IEEE OJ-CSYS 2024)](#25-hj-reachability-in-rl-a-survey-ieee-oj-csys-2024)
      - [关键内容](#关键内容-4)
    - [2.6 DRL Navigation in Crowded Environments (JIRS 2024)](#26-drl-navigation-in-crowded-environments-jirs-2024)
      - [关键内容](#关键内容-5)
  - [3. 文献对比总览](#3-文献对比总览)
  - [4. 项目改进方向](#4-项目改进方向)
    - [4.1 短期改进（基于现有架构微调）](#41-短期改进基于现有架构微调)
      - [改进 1：在线自适应 Alpha 网络](#改进-1在线自适应-alpha-网络)
      - [改进 2：障碍物速度编码](#改进-2障碍物速度编码)
      - [改进 3：射线 Attention 机制](#改进-3射线-attention-机制)
    - [4.2 中期改进（架构升级）](#42-中期改进架构升级)
      - [改进 4：Reach-Avoid 安全值网络](#改进-4reach-avoid-安全值网络)
      - [改进 5：异质图历史编码器](#改进-5异质图历史编码器)
      - [改进 6：动态障碍物轨迹预测模块](#改进-6动态障碍物轨迹预测模块)
    - [4.3 长期改进（范式革新）](#43-长期改进范式革新)
      - [改进 7：多模态动态障碍物场景](#改进-7多模态动态障碍物场景)
      - [改进 8：Sim-to-Real 动态场景部署](#改进-8sim-to-real-动态场景部署)
      - [改进 9：安全证书的形式化保证](#改进-9安全证书的形式化保证)
  - [建议实施路线图](#建议实施路线图)
  - [参考文献](#参考文献)

---

## 1. 当前项目状态概述

SEA-Nav 项目的核心架构：

```
观测(550维: 本体状态+41射线+目标方向, x10历史帧)
  → MLP历史编码器(16维隐变量)
    → Backbone(512→256→128, ELU)
      → 导航头 → u_bar [vx, vy, yaw_rate]
      → Alpha头 → alpha (softplus)
      → LSE-CBF安全层 → 修正为安全动作 u_s
        → 低通滤波 → 运动策略JIT模型 → PD控制器 → 关节力矩
```

**已实现能力：**
- 静态障碍物感知：高度场 → 网格射线投射（41条射线，-120°~+120°）
- 动态障碍物：简单纵向往复运动（0.15~0.40 m/s），圆形近似
- 射线融合：`rays = torch.minimum(static_rays, dynamic_rays)`

**核心不足：**
1. 动态障碍物射线融合仅取瞬时最小值，无轨迹预测
2. CBF 参数 alpha 训练后固定，不随环境变化自适应
3. 动态障碍物速度信息未显式编码到观测中
4. 历史编码器为简单 MLP，无法建模障碍物交互关系

---

## 2. 精选文献详细整理

### 2.1 Agile But Safe (RSS 2024)

| 项目 | 内容 |
|------|------|
| **标题** | Agile But Safe: Learning Collision-Free High-Speed Legged Locomotion |
| **作者** | Tairan He, Chong Zhang, Wenli Xiao, Guanqi He, Changliu Liu, Guanya Shi |
| **机构** | CMU / ETH Zürich |
| **原文链接** | https://arxiv.org/abs/2401.17583 |
| **项目页面** | https://agile-but-safe.github.io/ |
| **发表** | Robotics: Science and Systems (RSS), 2024 |

#### 关键内容

**1. 双策略框架 (Dual-Policy)**
- **敏捷策略 (Agile Policy)**：高速目标导向导航，最高 3.1 m/s
- **恢复策略 (Recovery Policy)**：当碰撞风险升高时接管控制，确保安全
- 两个策略由一个学习的 **值网络** 控制切换，而非硬编码规则

**2. Reach-Avoid (RA) 值网络**
- 学习预测"从当前状态出发，使用敏捷策略会在未来多久发生碰撞"
- 既作为策略切换的决策依据，又为恢复策略提供梯度引导
- 本质上是一个 **学习的安全证书**，比传统 CBF 更灵活

**3. 射线预测网络**
- 将深度图映射为 11 条稀疏射线距离（比 SEA-Nav 的 41 条更少）
- 高效的外感知表示，适合机载计算（Jetson Orin NX）

**4. 实验验证**
- 平台：宇树 Go1（与当前项目 Go2 同系列）
- 动态障碍物场景：行人、婴儿车、摇动的腿
- 鲁棒性：12 kg 负载（等同自重）、雪地、外部撞击
- 全部计算在 Jetson Orin NX 上完成

**核心图示（逻辑再现）：**
```
深度图 ──→ 射线预测网络 ──→ 11条稀疏射线 ──┬──→ 敏捷策略 (v, ω)
                                            │
                                            ├──→ RA值网络 V(s) ──→ 切换信号
                                            │
                                            └──→ 恢复策略 (v, ω)
```

---

### 2.2 Online CBF for Multi-Agent Navigation (MRS 2023)

| 项目 | 内容 |
|------|------|
| **标题** | Online Control Barrier Functions for Decentralized Multi-Agent Navigation |
| **作者** | Zhan Gao, Guang Yang, Amanda Prorok |
| **机构** | University of Cambridge |
| **原文链接** | https://arxiv.org/abs/2303.04313 |
| **发表** | International Symposium on Multi-Robot and Multi-Agent Systems (MRS), 2023 |

#### 关键内容

**1. 核心问题**
- 传统 CBF 使用 **固定超参数**，在动态环境中表现差：
  - 保守设置 → 机器人永远绕行，无法到达目标（freezing robot problem）
  - 激进设置 → 不安全的控制量，尤其在拥挤/高速场景
- 手工调参在动态场景中无法奏效

**2. 在线 CBF 方案**
- 用 **RL 学习 CBF 调参策略**，根据局部感知实时调整
- 参数化方式：**图神经网络 (GNN)**，具有平移不变性和排列等变性
- 完全去中心化：每个 agent 仅根据局部观测决策

**3. 为什么用模型无关 RL**
- CBF 参数 → 导航性能的映射难以解析建模
- RL 直接从交互经验中学习何时保守、何时激进
- GNN 能泛化到不同数量的障碍物/agent

**4. 技术路线（与 SEA-Nav 对比）：**
```
传统 CBF（SEA-Nav 当前）：
  固定 alpha ──→ CBF-Layer ──→ 安全动作

Online CBF（本文）：
  RL策略(GNN) ──→ 在线调整 CBF参数 ──→ 自适应CBF-Layer ──→ 安全动作
      ↑
  局部感知（障碍物位置/速度）
```

---

### 2.3 HEIGHT (2024)

| 项目 | 内容 |
|------|------|
| **标题** | HEIGHT: Heterogeneous Interaction Graph Transformer for Robot Navigation in Crowded and Constrained Environments |
| **作者** | Shuijing Liu, Haochen Xia, Katherine Driggs-Campbell 等 |
| **机构** | UIUC |
| **原文链接** | https://arxiv.org/abs/2411.12150 |
| **发表** | arXiv preprint, November 2024 |

#### 关键内容

**1. 核心洞察：异质交互建模**
- 传统方法将所有障碍物**同质化**处理（如 SEA-Nav 将所有动态障碍物仅视为"更近的射线"）
- 实际上存在三种不同的交互类型：
  - **Robot ↔ Human**：需要预测人类意图并主动让行
  - **Human ↔ Human**：人群内部的交互会影响群体运动
  - **Obstacle ↔ Agent**：静态障碍物仅需避让，无需预测

**2. 模型架构**
```
观测序列 ──→ 异质时空图构建
                │
    ┌───────────┼───────────┐
    │           │           │
 Robot-Human  Human-Human  Obstacle-Agent
   边           边           边
    │           │           │
    └───────────┼───────────┘
                │
    多注意力头 Transformer
                │
            GRU 时序建模
                │
          PPO 策略输出
```

**3. 关键结果**
- 88% 成功率 vs. 64% 基线（A* + CNN）
- **零样本 sim-to-real 泛化**到未见过的拥挤密度
- 在真实环境（Turtlebot + 行人）验证

---

### 2.4 Subgoal-Driven Navigation with Attention (2023)

| 项目 | 内容 |
|------|------|
| **标题** | Subgoal-Driven Navigation in Dynamic Environments Using Attention-Based Deep Reinforcement Learning |
| **作者** | Jorge de Heuvel, Weixian Shi, Xiangyu Zeng, Maren Bennewitz |
| **机构** | University of Bonn |
| **原文链接** | https://arxiv.org/abs/2303.01443 |
| **发表** | arXiv, March 2023 |

#### 关键内容

**1. 层次化设计**
- **Subgoal Agent**：处理 2D LiDAR 扫描（1080 点），输出子目标位置 `(x, y)`
- **Motion Agent**：使用局部路径规划到达子目标，处理底层运动控制
- 两个 agent 独立训练，组合使用

**2. Attention 机制**
- 在 LiDAR 扫描序列上应用 self-attention
- 策略**自动学会关注**扫描中与避障最相关的区域（如快速靠近的行人方向）
- 无需显式标注哪些障碍物是动态的

**3. 为什么不需要显式轨迹预测**
- Attention 权重自动捕捉时序变化（某方向的 LiDAR 点变近 = 有东西靠近）
- 策略隐式学会了从 LiDAR 的时序变化中推断危险程度

**4. 部署验证**
- 真实 Turtlebot 机器人 + 行人
- 仅用 LiDAR，无显式障碍物检测/跟踪

```
LiDAR扫描(1080点) ──→ Self-Attention编码 ──→ Subgoal Agent ──→ 子目标
                                                                    │
                                                              Motion Agent
                                                                    │
                                                            电机控制指令
```

---

### 2.5 HJ Reachability in RL: A Survey (IEEE OJ-CSYS 2024)

| 项目 | 内容 |
|------|------|
| **标题** | Hamilton-Jacobi Reachability in Reinforcement Learning: A Survey |
| **作者** | Milan Ganai, Sicun Gao, Sylvia L. Herbert |
| **机构** | UC San Diego |
| **原文链接** | https://arxiv.org/abs/2407.09645 |
| **发表** | IEEE Open Journal of Control Systems, Vol. 3, pp. 310–324, 2024 |

#### 关键内容

**1. HJ 可达性与 CBF 的关系**
- CBF 本质上是 HJ 可达性的一种**简化/保守近似**
- HJ 可达性提供严格的"安全/不安全"区域划分
- CBF 通过构造屏障函数来保证前向不变性，是 HJ 的充分条件

**2. 传统 HJ 的局限与突破**
- 传统方法受限于维度灾难（~6D），无法处理高维 RL 策略
- 最新进展：**学习 HJ 值函数**，扩展到 112 维状态空间
- 学习方法分为两类：
  - **在线学习**：与策略同时训练 (e.g., RESPO, NeurIPS 2023)
  - **离线学习**：先学习安全证书，再约束策略

**3. 与动态障碍物的关联**
- HJ 可达性天然支持 **reach-avoid 任务**（到达目标 + 避开障碍物）
- 可处理 **非合作动态障碍物**（不假设障碍物有合作行为）
- 处理 **不确定性**（感知噪声、模型误差）有理论保证

**4. 对 SEA-Nav 的启示**
```
当前方案：
  LSE-CBF层 (解析解, 固定alpha) ──→ 瞬时安全约束

HJ可达性增强方案：
  学习HJ值函数 ──→ 预测未来N步的安全区域 ──→ 约束策略 或 在线修正
```

---

### 2.6 DRL Navigation in Crowded Environments (JIRS 2024)

| 项目 | 内容 |
|------|------|
| **标题** | A Comprehensive Review of Mobile Robot Navigation Using Deep Reinforcement Learning Algorithms in Crowded Environments |
| **作者** | H. Le, S. Saeedvand, C.C. Hsu |
| **机构** | — |
| **原文链接** | https://link.springer.com/article/10.1007/s10846-024-02198-w |
| **发表** | Journal of Intelligent & Robotic Systems, 110(4), 158, 2024 |

#### 关键内容

**1. 综述范围**
- 覆盖 DQN / DDQN / DDPG / SAC / PPO / A3C 等主流 DRL 算法
- 分类框架：自主式 / 规划式 / SLAM 式导航
- 拥挤环境特指含多个动态 agent 的场景

**2. 方法分类**

| 类别 | 代表方法 | 特点 |
|------|---------|------|
| 传感器融合 | LSTM + CNN + LiDAR | 利用时序信息处理动态变化 |
| 社交感知 | SARL, CADRL | 建模行人的社交规范 |
| 层次化 | 高层规划 + 低层控制 | 解耦复杂任务 |
| 安全约束 | CBF, HJ Reachability | 提供安全保障 |
| 注意力机制 | Transformer, GAT | 自适应关注关键障碍物 |

**3. 该综述指出的开放问题**
1. **Sim-to-Real 鸿沟**：绝大多数工作停留在仿真
2. **多机器人协作**：去中心化动态避障通信效率低
3. **动态环境复杂度**：简单往复运动 ≠ 真实动态场景
4. **奖励函数设计**：如何平衡安全、效率、平滑性
5. **安全性保证**：学习策略缺乏形式化安全保证

---

## 3. 文献对比总览

| 维度 | Agile But Safe | Online CBF | HEIGHT | Subgoal-Attn | HJ Survey | DRL Survey |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| **年份** | 2024 | 2023 | 2024 | 2023 | 2024 | 2024 |
| **发表** | RSS | MRS | arXiv | arXiv | IEEE OJ-CSYS | JIRS |
| **方法类型** | 双策略+值网络 | RL在线调参 | 异质图Transformer | 层次化+Attention | 综述 | 综述 |
| **安全保证** | Reach-Avoid学习 | CBF(自适应) | PPO(无形式化) | 无形式化 | HJ可达性 | 总结各种 |
| **动态障碍物** | ✅ 真实行人 | ✅ 多Agent | ✅ 人群 | ✅ 行人 | ✅ 理论支持 | ✅ 全面覆盖 |
| **硬件平台** | Go1 四足 | 仿真多机器人 | Turtlebot | Turtlebot | — | — |
| **sim-to-real** | ✅ | ❌ | ✅ | ✅ | — | — |
| **与SEA-Nav关联度** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |

---

## 4. 项目改进方向

### 4.1 短期改进（基于现有架构微调）

#### 改进 1：在线自适应 Alpha 网络

**参考**：Online CBF (Gao et al., 2023)

**当前问题**：alpha 在训练后固定，不随局部障碍物密度/速度调整

**改进方案**：
- 新增小型 `AlphaAdapter` 网络
- 输入：当前 41 条射线 + 动态障碍物速度/位置（如有）
- 输出：时变 alpha 偏置量 `delta_alpha`
- 最终 alpha = `alpha_base + delta_alpha`
- 可独立于主策略训练，或与主策略联合微调

**涉及文件**：
- `training/rsl_rl/rsl_rl/modules/cbf_actor_critic.py` — 修改 AlphaHead
- `training/rsl_rl/rsl_rl/modules/cbf_lse_layer.py` — CBF layer 接受时变 alpha

```
当前：
  obs → AlphaHead → alpha(固定) → CBF-Layer

改进后：
  obs → AlphaHead → alpha_base ──┬──→ CBF-Layer
  rays + dyn_vel → AlphaAdapter → delta_alpha ──┘
```

#### 改进 2：障碍物速度编码

**参考**：HEIGHT (Liu et al., 2024)

**当前问题**：`_get_dynamic_rays()` 计算了障碍物位置但未编码速度信息；观测中无法区分"静止障碍物"和"正在靠近的障碍物"

**改进方案**：
- 在 `LeggedRobotPosDynamic` 中增加 `_get_dynamic_velocity_encoding()` 方法
- 对每个动态障碍物，计算其在机器人局部坐标系中的相对速度 `[vx_rel, vy_rel]`
- 将速度编码（N×2 维，N 为最大障碍物数）追加到观测向量末尾
- 修改配置文件中的 `num_observations` 相应增加

**涉及文件**：
- `training/legged_gym/legged_gym/envs/base/legged_robot_pos_dynamic.py`
- `training/legged_gym/legged_gym/envs/go2/go2_pos_config.py`

```
当前观测：
  [本体状态(12) | 射线(41) | 目标(2)] × 10帧历史

改进后观测：
  [本体状态(12) | 射线(41) | 目标(2) | 动态障碍物速度(3×2)] × 10帧历史
```

#### 改进 3：射线 Attention 机制

**参考**：Subgoal-Attention (de Heuvel et al., 2023)

**当前问题**：41 条射线均匀分布于 -120°~+120°，策略无法自适应关注危险方向

**改进方案**：
- 在历史编码器中加入轻量级 self-attention 层
- 对每条射线的 10 帧历史做 attention，自动学习关注"快速变近"的方向
- 输出加权后的射线特征，供 backbone 使用
- 可选：将 attention 权重可视化，辅助调试和理解策略行为

**涉及文件**：
- `training/rsl_rl/rsl_rl/modules/cbf_actor_critic.py` — 修改 encoder

```
当前编码器：
  550维 × 10帧 → MLP → 16维隐变量

改进后编码器：
  射线(41×10) → Self-Attention → 加权射线特征 ─┐
  本体(12×10) + 目标(2×10) ──────────────────→ Concat → MLP → 16维隐变量
```

---

### 4.2 中期改进（架构升级）

#### 改进 4：Reach-Avoid 安全值网络

**参考**：Agile But Safe (He et al., 2024)

**当前问题**：LSE-CBF 安全层仅基于当前时刻的射线距离计算约束，无法预测"未来可能碰撞"

**改进方案**：
- 训练一个 **Reach-Avoid 值网络** $V_{RA}(s)$，预测未来 N 步内的碰撞概率
- 替代或增强当前 CBF-Layer：
  - **选项 A（增强）**：$V_{RA}(s)$ 输出作为 CBF 的额外约束或 margin 调整
  - **选项 B（替代）**：用 $V_{RA}(s)$ 的梯度直接修正导航动作，不依赖解析 CBF
- RA 值网络与主策略联合训练（类似 critic）
- 训练数据可从碰撞回放缓冲区获取（SEA-Nav 已有 collision replay 机制）

**涉及文件**：
- `training/rsl_rl/rsl_rl/modules/cbf_actor_critic.py` — 新增 RAValueHead
- `training/rsl_rl/rsl_rl/modules/cbf_lse_layer.py` — 与 RA 值网络集成
- `training/rsl_rl/rsl_rl/algorithms/ppo.py` — 新增 RA 值网络损失

```
当前CBF-Layer：
  u_bar ──→ LSE-CBF(rays, alpha) ──→ u_s

增强方案A (CBF + RA)：
  u_bar ──→ LSE-CBF(rays, alpha_adaptive + V_RA) ──→ u_s
                ↑
  obs ──→ RA值网络 ──→ V_RA(s)

替代方案B (纯RA)：
  u_bar ──→ RA-guided修正 ──→ u_s
                ↑
  obs ──→ RA值网络 ──→ V_RA(s), ∇V_RA
```

#### 改进 5：异质图历史编码器

**参考**：HEIGHT (Liu et al., 2024)

**当前问题**：MLP 历史编码器将所有观测（射线、本体状态、目标）展平为单一向量，无法建模障碍物之间的空间关系

**改进方案**：
- 将历史编码器重构为 **异质图结构**：
  - **节点**：机器人、每条射线端点、每个动态障碍物
  - **边**：
    - 机器人 ↔ 射线（距离信息）
    - 射线 ↔ 射线（相邻角度的空间连续性）
    - 机器人 ↔ 动态障碍物（相对位置 + 速度）
    - 动态障碍物 ↔ 动态障碍物（交互影响）
- 用轻量级 GAT (Graph Attention Network) 处理异质图
- 输出：图级嵌入作为历史隐变量

**涉及文件**：
- `training/rsl_rl/rsl_rl/modules/cbf_actor_critic.py` — 重构 HistoryEncoder
- 新建 `training/rsl_rl/rsl_rl/modules/hetero_graph_encoder.py`

```
当前：
  展平向量 [550×10] → MLP → 16维

改进后：
  ┌── 机器人节点 ──┐
  ├── 射线节点×41 ─┤ → 异质GAT → 图池化 → 16维
  ├── 障碍物节点×N ┤
  └── 目标节点 ────┘
```

#### 改进 6：动态障碍物轨迹预测模块

**参考**：HJ Survey (Ganai et al., 2024) 中的 reach-avoid 形式化; DRL Survey 中的预测-规划框架

**当前问题**：动态障碍物仅用当前帧的圆形近似，完全不考虑未来轨迹

**改进方案**：
- 维护每个动态障碍物过去 K 帧的位置历史
- 训练一个轻量级 **轨迹预测网络**（可用简单的线性预测或小型 MLP/LSTM）
- 预测未来 M 帧的位置
- 在射线融合时，使用**预测位置**而非当前位置：
  - `rays = torch.minimum(static_rays, current_dynamic_rays, future_dynamic_rays_1, future_dynamic_rays_2, ...)`
- 可选：将预测的不确定性也编码到观测中

**涉及文件**：
- `training/legged_gym/legged_gym/envs/base/legged_robot_pos_dynamic.py`
- 新建 `training/legged_gym/legged_gym/utils/trajectory_predictor.py`

---

### 4.3 长期改进（范式革新）

#### 改进 7：多模态动态障碍物场景

**参考**：HEIGHT, DRL Survey 中的 crowded environments

- 增加多种运动模式的动态障碍物：随机游走、追踪机器人、交叉穿越
- 增加障碍物数量（当前最多 3 个 → 扩展到 10+）
- 加入障碍物间的社交交互规则（如 ORCA 策略控制的智能体）

#### 改进 8：Sim-to-Real 动态场景部署

**参考**：Agile But Safe (Go1 实机), Subgoal-Attention (Turtlebot + 人), HEIGHT (零样本迁移)

- 当前项目纯仿真。参考 Agile But Safe 的部署方式：
  - 域随机化（摩擦力、质量、延迟）
  - 深度图 → 射线蒸馏网络
  - Jetson 机载部署

#### 改进 9：安全证书的形式化保证

**参考**：HJ Reachability Survey

- 当前 CBF 提供了一定的安全保障，但对动态障碍物没有形式化保证
- 用学习的 HJ 值函数作为策略的安全过滤器
- 可结合置信度校准（conformal prediction）处理预测不确定性

---

## 建议实施路线图

```
Phase 1（2-3周）─ 短期改进
  ├── 改进2: 障碍物速度编码 → 观测增强
  ├── 改进3: 射线Attention → 编码器增强
  └── 实验对比 baseline vs improved

Phase 2（3-4周）─ 中期改进
  ├── 改进1: 在线自适应Alpha → CBF增强
  ├── 改进4: Reach-Avoid值网络 → 安全层升级
  └── 实验对比 CBF vs CBF+RA

Phase 3（4-6周）─ 深度改进
  ├── 改进5: 异质图历史编码器
  ├── 改进6: 轨迹预测模块
  └── 综合评估

Phase 4（长期）
  ├── 改进7: 多模态动态场景
  ├── 改进8: Sim-to-Real
  └── 改进9: 形式化安全保证
```

---

## 参考文献

1. He T, Zhang C, Xiao W, et al. Agile But Safe: Learning Collision-Free High-Speed Legged Locomotion. RSS 2024. [arXiv:2401.17583](https://arxiv.org/abs/2401.17583)
2. Gao Z, Yang G, Prorok A. Online Control Barrier Functions for Decentralized Multi-Agent Navigation. MRS 2023. [arXiv:2303.04313](https://arxiv.org/abs/2303.04313)
3. Liu S, Xia H, Driggs-Campbell K, et al. HEIGHT: Heterogeneous Interaction Graph Transformer for Robot Navigation in Crowded and Constrained Environments. arXiv 2024. [arXiv:2411.12150](https://arxiv.org/abs/2411.12150)
4. de Heuvel J, Shi W, Zeng X, Bennewitz M. Subgoal-Driven Navigation in Dynamic Environments Using Attention-Based Deep Reinforcement Learning. arXiv 2023. [arXiv:2303.01443](https://arxiv.org/abs/2303.01443)
5. Ganai M, Gao S, Herbert SL. Hamilton-Jacobi Reachability in Reinforcement Learning: A Survey. IEEE OJ-CSYS 2024. [arXiv:2407.09645](https://arxiv.org/abs/2407.09645)
6. Le H, Saeedvand S, Hsu CC. A Comprehensive Review of Mobile Robot Navigation Using Deep Reinforcement Learning Algorithms in Crowded Environments. JIRS 2024. [DOI:10.1007/s10846-024-02198-w](https://doi.org/10.1007/s10846-024-02198-w)


