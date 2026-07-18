# 调试记录

## 仿真环境

早期场景包含窄通道、台阶两侧结构墙和四级窄台阶。台阶由独立 heightfield box 写入，视觉检查发现相邻踏面及台阶与平台之间可能出现空隙；同时，台阶前仅有 3 个低矮障碍物，机器人可以从稀疏障碍物的两侧轻易绕开。

### 调试与修改过程

#### 1. 重建台阶与高台

- 删除 `structural_boxes` 中用于窄通道和台阶两侧的结构墙。
- 将台阶改为 5 级：每级高度 `0.08 m`、踏面深度 `0.30 m`，总高度 `0.40 m`、总深度 `1.50 m`。
- 台阶从房间局部坐标 `x=4.80 m` 开始，第 5 级结束于 `x=6.30 m`。
- 将高台改为从 `x=6.30 m` 直接连续延伸，平台高度 `0.40 m`；不再使用独立 box 与台阶拼接。
- 场景生成改用连续的 heightfield cell 区间写入每一级踏面和后方平台，避免由 box 中心点取整产生的浮空缝隙。
- 因外墙占据两侧各 `0.30 m`，台阶的实际可通行宽度为 `9.40 m`，横跨全部可通行区域。

#### 2. 加密台阶前低矮障碍物

- 保留原先 3 个低矮盒体，并新增 30 个固定盒体，当前总数为 33 个。
- 新增障碍物采用交错阵列，覆盖台阶前低处的 `x=1.45–3.85 m`、`y=0.85–9.15 m` 区域；含原始盒体后的整体覆盖范围约为 `x=1.20–4.10 m`、`y=0.68–9.33 m`。
- 低矮障碍物高度固定为 `0.08 m`、`0.10 m` 或 `0.12 m`，尺寸与原有障碍物一致量级。
- 对所有障碍物执行边界与两两不重叠检查，避免侵入外墙、台阶或起点附近的必要缓冲空间。

### 最终场景

| 项目 | 当前规格 |
| --- | --- |
| 房间 | `10 m × 10 m`，相邻房间由 `0.30 m` 厚、`1.0 m` 高外墙分隔 |
| 窄通道/侧墙 | 已删除；没有台阶两侧结构墙 |
| 低矮障碍物 | 33 个固定盒体，高 `0.08–0.12 m`，台阶前交错密集分布 |
| 台阶 | 5 级，踏面高度依次为 `0.08/0.16/0.24/0.32/0.40 m`、深 `0.30 m` |
| 台阶位置 | 从 `x=4.80 m` 至 `x=6.30 m`，宽度为外墙内侧全部可通行宽度 `9.40 m` |
| 高台 | 自 `x=6.30 m` 无缝连接，平台高 `0.40 m`，延伸至后侧外墙内缘 |

录制单房间地形巡览视频来检查仿真环境，在 `/home/sea_ws/src` 下执行：

```bash
export PYTHONPATH=/home/sea_ws/src/training/legged_gym:/home/sea_ws/src/training/rsl_rl

python training/legged_gym/legged_gym/scripts/record_depth_stairs_terrain.py \
  --task go2_pos_depth_stairs --headless --fps 30 \
  --frames_per_shot 120 --side_hold_frames 300 \
  --output_video /home/sea_ws/artifacts/depth_stairs/dense_low_obstacles_stairs_tour.mp4
```

最新视频为：

- `/home/sea_ws/artifacts/depth_stairs/dense_low_obstacles_stairs_tour.mp4`
- 时长 `26 s`，分辨率 `960 × 540`，帧率 `30 FPS`。
- 前半段展示房间、密集低矮障碍物及台阶；末尾 `16–26 s` 为固定的近侧视检查镜头，画面标记为 `SIDE VIEW: stair continuity check`。

相关文件：

- 场景配置：`/home/sea_ws/src/training/legged_gym/legged_gym/envs/go2/go2_pos_config.py`
- Heightfield 场景生成：`/home/sea_ws/src/training/legged_gym/legged_gym/utils/terrain.py` 中的 `depth_stairs_room_terrain_func`
- 任务环境和指标：`/home/sea_ws/src/training/legged_gym/legged_gym/envs/base/legged_robot_pos_depth_stairs.py`
- 静态地形巡览脚本：`/home/sea_ws/src/training/legged_gym/legged_gym/scripts/record_depth_stairs_terrain.py`
- 策略运行与深度图录像脚本：`/home/sea_ws/src/training/legged_gym/legged_gym/scripts/record_depth_stairs.py`


## 低层步态控制诊断（2026-07-18）

在开始导航策略训练前，检查当前冻结的 Go2 SLR 低层步态控制器能否处理新的五级台阶。此检查绕过尚未训练的导航策略，直接将固定速度命令送入环境的导航动作接口；命令经过现有滤波与限幅后，由同一冻结 SLR 控制器生成关节动作。因此，结果反映低层控制器与台阶地形的组合能力，不受深度感知或高层导航训练状态影响。

### 检查设置

- 使用 `go2_pos_depth_stairs`，关闭传感器噪声、推扰、摩擦/质量/初始姿态随机化，并关闭接触终止。
- 每个测试从房间中心线 `y=5.00 m` 出发；上行从 `x=0.90 m` 向高台前进，下行从高台 `x=7.20 m` 返回低台。
- 分别测试固定命令速度 `0.25`、`0.40`、`0.55`、`0.70 m/s`，每个方向运行 `1500` 个仿真步。
- 上行通过条件为到达高台起点 `x>=6.30 m` 且机身高度达到平台判据；同时不得出现跌倒或静止超时。
- 执行脚本：`training/legged_gym/legged_gym/scripts/diagnose_depth_stairs_locomotion.py`；原始结果：`/home/sea_ws/artifacts/depth_stairs/locomotion_stair_diagnosis.csv`。

### 结果

| 方向 | 速度（m/s） | 最远局部 x（m） | 是否到达目标台面 | 跌倒 | 静止超时 |
| --- | ---: | ---: | --- | --- | --- |
| 上行 | 0.25 | 2.774 | 否 | 否 | 是 |
| 上行 | 0.40 | 4.778 | 否 | 否 | 是 |
| 上行 | 0.55 | 2.636 | 否 | 否 | 是 |
| 上行 | 0.70 | 4.793 | 否 | 否 | 是 |
| 下行 | -0.25 | 3.568 | 是 | 否 | 是 |
| 下行 | -0.40 | 4.099 | 是 | 否 | 是 |
| 下行 | -0.55 | 3.735 | 是 | 否 | 是 |
| 下行 | -0.70 | 3.745 | 是 | 否 | 是 |

四个上行案例全部失败：最快的两个案例仅到达首级台阶前（首级从 `x=4.80 m` 开始），且机身最高高度始终约为平地高度 `0.42 m`，没有产生跨越首级所需的上升。下行可回到低台，但测试继续前进后也触发静止超时，不能视为完整的稳定上下台阶能力。

### 结论与限制

当前导航接口只接受 `[vx, vy, yaw_rate]`，第三项为偏航角速度而不是 `vz`；高层不能直接下达“斜向上速度”或机身高度命令。台阶上的垂直运动必须由低层步态根据地形接触、摆腿高度和机身控制自主产生。

本检查证明当前冻结步态控制器不能仅凭固定平面前进命令跨越当前首级 `0.08 m` 台阶；它不证明经过重新调整的、位置/接触感知的变速命令或具备抬脚与机身高度适应能力的步态控制器也无法上台阶。正式导航训练前必须重新调整或重训低层步态控制，并以重复的接近、上行、平台稳定和下行测试验证。

