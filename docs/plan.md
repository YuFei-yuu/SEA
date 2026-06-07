# SEA-Nav 动态障碍物导航实施方案

## 当前实现
- 新增稀疏静态场景 `dynamic_room_sparse`：10m x 10m 房间、四周边界墙、3 个固定静态方柱。
- 新增任务：`go2_pos_sparse_static`、`go2_pos_dynamic_1`、`go2_pos_dynamic_2`、`go2_pos_dynamic_3`。
- 动态障碍物作为额外 actor 创建，不写入 heightfield；观测仍使用原有 41 条 ray，并将动态障碍物解析射线融合为 `final_rays = min(static_rays, dynamic_rays)`。
- 训练网络、动作维度、PPO/CBF 主干保持不变。
- 新增动态碰撞统计：`dynamic_collision_count`、`body_collision_count`、`total_collision_count`，并支持动态碰撞 early reset。
- 新增评测脚本 `training/legged_gym/legged_gym/scripts/evaluate_dynamic.py`。

## 任务配置
- `go2_pos_sparse_static`：稀疏静态房间，0 个动态障碍物。
- `go2_pos_dynamic_1`：稀疏静态房间，1 个动态障碍物。
- `go2_pos_dynamic_2`：稀疏静态房间，2 个动态障碍物。
- `go2_pos_dynamic_3`：稀疏静态房间，3 个动态障碍物。

## 动态障碍物参数
- 尺寸：`0.45 x 0.45 x 0.8`
- 射线近似半径：`0.32`
- 车道中心：`x = [2.2, 5.0, 7.8]`
- 左右扰动：`lane_x_jitter = 0.15`
- 运动范围：`y in [1.2, 8.8]`
- 速度范围：`[0.15, 0.40] m/s`
- 运动模式：仅纵向往返，到边界反向

## 建议训练顺序
1. 阶段 0：环境验证
- `python training/legged_gym/legged_gym/scripts/play.py --task go2_pos_dynamic_1 --resume --resume_experiment_name Go2_pos_rough`
- 目标：确认障碍物会动、ray 会变化、碰撞统计会增长。

2. 阶段 1：稀疏静态适应
- `python training/legged_gym/legged_gym/scripts/train.py --task go2_pos_sparse_static --resume --resume_experiment_name Go2_pos_rough`

3. 阶段 2：1 个动态障碍物
- `python training/legged_gym/legged_gym/scripts/train.py --task go2_pos_dynamic_1 --resume --resume_experiment_name Go2_pos_sparse_static`

4. 阶段 3：2 个动态障碍物
- `python training/legged_gym/legged_gym/scripts/train.py --task go2_pos_dynamic_2 --resume --resume_experiment_name Go2_pos_dynamic_1`

5. 阶段 4：3 个动态障碍物
- `python training/legged_gym/legged_gym/scripts/train.py --task go2_pos_dynamic_3 --resume --resume_experiment_name Go2_pos_dynamic_2`

## 评测与录像
- 评测：
  - `python training/legged_gym/legged_gym/scripts/evaluate_dynamic.py --task go2_pos_dynamic_3 --resume --num_episodes 50`
- 录像：
  - `xvfb-run -a python training/legged_gym/legged_gym/scripts/play_record.py --task go2_pos_dynamic_3 --resume`

## 评测指标
- `success_rate`
- `safe_success_rate`
- `avg_total_collision_count`
- `avg_dynamic_collision_count`
- `avg_body_collision_count`
- `timeout_rate`
- `mean_time_to_goal`

## 需要优先验证的点
- 动态障碍物 root state 每步更新且边界反向正常。
- `final_rays` 在障碍物进入前方视野时明显短于 `static_rays`。
- `dynamic_collision_count` 按事件计数，不会单次接触连续累加。
- `go2_pos_rough` 原任务仍能正常训练和推理。

## 关键可调参数说明

后续如果需要自行调整训练难度、录制视频效果或动态障碍物行为，优先看下面这些位置。

### 1. 起点与目标点采样

默认配置位置：

- `training/legged_gym/legged_gym/envs/go2/go2_pos_config.py`

关键字段：

- `start_x_left`
- `start_x_right`
- `goal_x_left`
- `goal_x_right`
- `start_y_range`
- `goal_y_range`
- `spawn_clearance`

作用：

- 控制机器人从房间左侧还是右侧出发；
- 控制目标点落在对侧哪个区域；
- 控制起点/目标点是否更靠近房间中间还是上下两侧；
- 控制起点和目标点与静态柱体之间的最小安全距离。

实际采样逻辑位置：

- `training/legged_gym/legged_gym/envs/base/legged_robot_pos_dynamic.py`
- 函数：`_sample_sparse_start_goal()`

当前默认行为：

- `50%` 左到右，`50%` 右到左；
- `x` 落在左右边界带；
- `y` 在给定范围内均匀随机；
- 如果太靠近静态柱体，会重新采样。

调参建议：

- 如果希望机器人更容易与动态障碍物相遇，可收紧 `start_y_range` 和 `goal_y_range`，让路径更集中在房间中部。
- 如果希望任务更随机，可扩大 `start_y_range` 和 `goal_y_range`。
- 如果希望起点/目标点更靠近边角，可直接改左右 `x` 范围。

### 2. 稀疏静态障碍物布局

默认配置位置：

- `training/legged_gym/legged_gym/envs/go2/go2_pos_config.py`
- `training/legged_gym/legged_gym/utils/terrain.py`

关键字段：

- `pillar_centers`
- `pillar_size`

作用：

- 控制 3 个静态柱体的数量、位置和尺寸。

当前默认布局：

- 柱体中心：`(3.6, 2.6)`、`(6.4, 5.0)`、`(3.6, 7.4)`
- 柱体尺寸：`0.8 x 0.8 x 0.6`

调参建议：

- 如果想让场景更难，可把柱体往中间收拢，缩窄可通行区域。
- 如果想让动态避障更突出、静态障碍影响更小，可把柱体进一步移向边缘。
- 如果只想改“视觉场景”，配置和 `terrain.py` 中的高度图绘制逻辑要同步改。

### 3. 动态障碍物数量与默认运动范围

默认配置位置：

- `training/legged_gym/legged_gym/envs/go2/go2_pos_config.py`

关键字段：

- `count`
- `lane_x`
- `lane_x_jitter`
- `y_min`
- `y_max`
- `speed_range`
- `ray_radius`
- `force_interaction`
- `force_interaction_jitter`

作用：

- `count`：控制障碍物数量，对应 `dynamic_1/2/3`
- `lane_x`：控制每个障碍物主要位于哪条纵向车道
- `lane_x_jitter`：控制车道左右扰动大小
- `y_min`/`y_max`：控制上下活动范围
- `speed_range`：控制移动速度
- `ray_radius`：控制感知时障碍物的圆形近似半径
- `force_interaction`：是否在 reset 时强制让至少一个障碍物更靠近机器人主路径

调参建议：

- 如果想让机器人更频繁遇到障碍物，可把 `lane_x` 调到机器人主通道附近，并缩小 `y` 范围。
- 如果想让任务更难，可提高 `speed_range` 上限或增大 `lane_x_jitter`。
- 如果想让障碍物在观测中“看起来更大”，可以增大 `ray_radius`。

### 4. 动态障碍物 reset 采样逻辑

实现位置：

- `training/legged_gym/legged_gym/envs/base/legged_robot_pos_dynamic.py`
- 函数：`_reset_dynamic_obstacles()`

当前行为：

- 每个障碍物先被分配到一条预定义车道；
- `x` 在车道中心附近随机扰动；
- `y` 在 `[y_min, y_max]` 内随机初始化；
- 初始运动方向随机；
- 速度在 `speed_range` 内随机采样；
- 若 `force_interaction=True`，第一个障碍物会被偏置到机器人起点与目标点之间的主通带附近。

适合修改的场景：

- 想把“纵向往返”改成“横向穿行”
- 想让多个障碍物从固定队形开始
- 想让障碍物初始位置更靠近起点、终点或中间区域
- 想让某个障碍物固定更容易与机器人相遇

### 5. 动态障碍物每步运动策略

实现位置：

- `training/legged_gym/legged_gym/envs/base/legged_robot_pos_dynamic.py`
- 函数：`_update_dynamic_obstacles()`

当前策略：

- 障碍物只沿 `y` 方向匀速运动；
- 到达 `y_min` 或 `y_max` 边界后反向；
- 没有加速度、没有转向、没有随机游走。

如果你后续要改成更复杂的运动模式，优先改这里，例如：

- 水平穿行
- 圆周运动
- 随机游走
- 多阶段路径点巡航
- 按时间随机切换速度

注意：

- 如果改得太复杂，训练难度会上升很多；
- 建议先保持观测结构不变，再逐步增加运动随机性。

### 6. 录制视频时的专用 override

位置：

- `training/legged_gym/legged_gym/scripts/play_record.py`
- 函数：`_apply_record_overrides()`

录制时当前会额外做的事情：

- 强制 `terrain.num_rows = 1`
- 强制 `terrain.num_cols = 1`
- 对动态任务把：
  - `start_y_range` 改成 `[4.2, 5.8]`
  - `goal_y_range` 改成 `[4.2, 5.8]`
  - `y_min` 改成 `3.0`
  - `y_max` 改成 `7.0`
  - `force_interaction = True`

这样做的目的是：

- 让录制视频时更容易出现明显的人机-障碍物交互；
- 避免视频里出现“机器人一路走过去但没有遇到动态障碍物”的情况。

如果你只想改“展示效果”，不想影响训练配置，优先改这里。

### 7. 录制视频中的可视化效果

位置：

- `training/legged_gym/legged_gym/scripts/play_record.py`
- 函数：`_draw_minimap_overlay()`

当前 overlay 会显示：

- 起点
- 目标点
- 机器人当前位置
- 动态障碍物位置
- episode 编号
- 距目标距离
- 碰撞计数

如果后续你想修改：

- 标记颜色
- 地图大小
- 是否显示静态柱体
- 是否显示路径连线
- 是否显示更多统计信息

就改这个函数。

### 8. 调参时的实用建议

- 想优先提高成功率：
  - 降低 `speed_range`
  - 收紧障碍物活动范围
  - 减少 `lane_x_jitter`

- 想让动态避障更明显：
  - 收紧 `start_y_range` / `goal_y_range`
  - 开启或加强 `force_interaction`
  - 把障碍物车道布置在主路径附近

- 想让任务更接近真实复杂场景：
  - 扩大起点/目标点范围
  - 增加障碍物速度随机性
  - 修改 `_update_dynamic_obstacles()`，引入更复杂运动模式

- 想只优化演示视频而不影响训练：
  - 优先改 `play_record.py` 的 override，不要先动 `go2_pos_config.py`
