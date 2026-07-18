# Go2 楼梯低障碍 Depth 导航实施计划

## 目标

构建一个固定拓扑、可随机化的复杂场景，使机器人从起点穿过密集低矮障碍物后登上五级无缝台阶、到达高台目标。机器人需基于 depth sensor 输入完成导航和避障。

### 具体要求
- 机器人需要使用 depth sensor、depth image、point cloud 或 height map 作为环境感知输入；
- 任务重点不是单纯平地导航，而是处理楼梯和低矮障碍物带来的感知与通过问题。

### 最低标准
- 成功搭建包含楼梯和低矮障碍物的仿真场景；
- 机器人能够从起点导航到目标点； 
- 机器人在导航过程中不能频繁撞上低矮物体； 
- 机器人需要展示至少一次通过楼梯或台阶区域的过程； 
- 提交传感器可视化结果、导航轨迹和 demo 视频。

### 当前已完成：仿真环境搭建

详见`/home/sea_ws/src/docs/depth/record.md`：

- 场景尺寸为 10 m x 10 m；每个房间保留厚 0.30 m、高 1.0 m 的外墙，用于与相邻房间隔开。已删除窄通道及台阶两侧的结构墙。
- 台阶从 `x=4.80 m` 开始，共五级；每级高 0.08 m、深 0.30 m，总高 0.40 m、总深 1.50 m。台阶填满两侧外墙之间的可通行宽度（9.40 m），并由连续 heightfield 切片生成，台阶之间及第 5 级与高台之间没有浮空或地面缝隙。
- 高台从 `x=6.30 m` 紧接最后一级台阶，平台高度为 0.40 m，并延伸到后侧外墙内缘。
- 台阶前平地区域布置 33 个固定低矮盒体，高度为 0.08-0.12 m，覆盖约 `x=1.20-4.10 m`、`y=0.68-9.33 m`；交错布局显著缩小从两侧绕过障碍物的空间。
- 正式评测使用 50 个未见随机种子，目标为成功率至少 60%、平均低障碍碰撞不超过 0.5 次/回合，并报告楼梯通过率、无碰撞成功率和到达时间。


## 当前感知接口

- 挂载一个前向深度相机：`160x90`、102 度水平视场、深度范围 0.1-5.0 m、10 Hz 更新。
- 因单相机视场只有约 100 度，策略使用 21 条覆盖约正负 50 度的前向射线，而不是原任务覆盖 240 度的 41 条真值射线。
- 采集深度图与几何真值射线对，训练单通道 ResNet-18：输入 `log2(depth)`，输出 21 个 `log2(range)`。数据按种子作 80/10/10 训练、验证、测试划分。
- 最终任务固定 `perception.mode=depth_predicted`；几何真值射线只允许用于数据标签、感知误差分析和 oracle 消融，不能写入 actor 观测。
- 最终导航观测为 `(12 本体状态 + 21 深度预测射线 + 2 目标状态) x 10` 历史帧，共 350 维。

## 实施路径

1. 新增 `go2_pos_depth_stairs` 任务、场景生成器、受限起点/目标采样、楼梯通过和低障碍碰撞指标；当前固定场景已完成无缝五级台阶和密集低障碍物布局。
2. 在任务环境中复用 ABS 的 GPU 深度相机生命周期，加入深度图缓存、相机同步和真值射线数据导出。
3. 实现数据采集脚本和 21 射线预测网络；先验证离线感知误差，再接入导航环境。
4. 重新调整或重训 Go2 低层步态控制器，使其能根据台阶接触实现摆腿抬升和机身高度适应；完成接近、上行、平台稳定和下行的重复运动诊断后，冻结通过验收的控制器。
5. 以通过验收的低层控制器为基础，依次训练 oracle 射线导航、深度预测射线导航和完整场景课程。
6. 实现 50 回合评测、轨迹 CSV/图片、深度图与射线对比图、外部视角加传感器画面的 demo 录像。

## 对照与交付

- 同一组 50 个种子对比：oracle 射线、深度预测射线、移除低障碍物场景。
- 记录感知射线 MAE/RMSE、成功率、低障碍碰撞、总碰撞、楼梯通过率和到达时间。
- 提交训练 checkpoint、评测 CSV/摘要、传感器可视化、导航轨迹和至少三条覆盖完整路线的成功视频。

## 约束

- 使用当前 SEA-Nav 的 Go2；低层步态控制器必须针对当前台阶重新调整并经运动诊断验收后冻结，不迁移 ABS 的 Go1 机器人或硬件部署代码。
- 本任务仅做静态复杂场景，不引入原动态障碍物 token 或动态 CBF 特权信息。
- 引用 ABS 的非商业许可代码和论文；实现仅用于课程/研究验证。

## 运行顺序

在 `/home/sea_ws/src` 下执行，并先激活已经跑通 SEA-Nav 的 Python 环境。

~~~bash
export PYTHONPATH=/home/sea_ws/src/training/legged_gym:/home/sea_ws/src/training/rsl_rl

python training/legged_gym/legged_gym/scripts/collect_depth_stairs.py \
  --task go2_pos_depth_stairs --headless --num_envs 64 \
  --output_dir training/legged_gym/depth_data/stairs_v1 --num_steps 2000

python training/legged_gym/legged_gym/scripts/train_depth_rays.py \
  --data_dir training/legged_gym/depth_data/stairs_v1 \
  --output training/legged_gym/depth_models/stairs_v1_best.pt

WANDB_MODE=disabled python training/legged_gym/legged_gym/scripts/train.py \
  --task go2_pos_depth_stairs --depth_mode oracle --headless \
  --num_envs 128 --max_iterations 1000

WANDB_MODE=disabled python training/legged_gym/legged_gym/scripts/train.py \
  --task go2_pos_depth_stairs --depth_mode depth_predicted \
  --depth_model training/legged_gym/depth_models/stairs_v1_best.pt \
  --headless --resume --load_run <oracle_run> --checkpoint <oracle_checkpoint> \
  --num_envs 128 --max_iterations 1000

python training/legged_gym/legged_gym/scripts/evaluate_depth_stairs.py \
  --task go2_pos_depth_stairs --depth_mode depth_predicted \
  --depth_model training/legged_gym/depth_models/stairs_v1_best.pt \
  --load_run <depth_run> --checkpoint <depth_checkpoint> --num_episodes 50 \
  --output_csv training/legged_gym/logs/Go2_pos_depth_stairs/eval.csv \
  --output_summary training/legged_gym/logs/Go2_pos_depth_stairs/eval.json

python training/legged_gym/legged_gym/scripts/record_depth_stairs.py \
  --task go2_pos_depth_stairs --depth_mode depth_predicted \
  --depth_model training/legged_gym/depth_models/stairs_v1_best.pt \
  --load_run <depth_run> --checkpoint <depth_checkpoint> --num_episodes 3 \
  --output_video training/legged_gym/logs/Go2_pos_depth_stairs/depth_demo.mp4 \
  --output_trajectory training/legged_gym/logs/Go2_pos_depth_stairs/depth_trajectory.csv
~~~

<oracle_run>、<oracle_checkpoint>、<depth_run> 和 <depth_checkpoint> 替换为对应训练产物。正式提交使用 depth_predicted 评测，不将 oracle 结果作为最终结果。
