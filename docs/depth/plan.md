# Go2 楼梯低障碍 Depth 导航实施计划

## 目标

在 SEA-Nav 的 Go2 分层导航框架中构建一个固定拓扑、可随机化的复杂场景。机器人从起点穿过低矮障碍物和窄通道后登上四级台阶、到达高台目标；最终导航策略只使用前向深度相机派生的距离射线、目标和本体状态，不使用场景真值射线或 height map。

## 场景与验收

- 场景尺寸为 10 m x 10 m，包含 1 m 高边界墙、0.06-0.14 m 低矮盒体、0.90 m 窄通道、四级楼梯和 0.32 m 高平台。
- 楼梯每级高 0.08 m、深 0.30 m、宽 1.60 m；墙体保证机器人无法绕开楼梯到达目标。
- 正式评测使用 50 个未见随机种子，目标为成功率至少 60%、平均低障碍碰撞不超过 0.5 次/回合，并报告楼梯通过率、无碰撞成功率和到达时间。

## 感知接口

- 挂载一个前向深度相机：`160x90`、102 度水平视场、深度范围 0.1-5.0 m、10 Hz 更新。
- 因单相机视场只有约 100 度，策略使用 21 条覆盖约正负 50 度的前向射线，而不是原任务覆盖 240 度的 41 条真值射线。
- 采集深度图与几何真值射线对，训练单通道 ResNet-18：输入 `log2(depth)`，输出 21 个 `log2(range)`。数据按种子作 80/10/10 训练、验证、测试划分。
- 最终任务固定 `perception.mode=depth_predicted`；几何真值射线只允许用于数据标签、感知误差分析和 oracle 消融，不能写入 actor 观测。
- 最终导航观测为 `(12 本体状态 + 21 深度预测射线 + 2 目标状态) x 10` 历史帧，共 350 维。

## 实施路径

1. 新增 `go2_pos_depth_stairs` 任务、场景生成器、受限起点/目标采样、楼梯通过和低障碍碰撞指标。
2. 在任务环境中复用 ABS 的 GPU 深度相机生命周期，加入深度图缓存、相机同步和真值射线数据导出。
3. 实现数据采集脚本和 21 射线预测网络；先验证离线感知误差，再接入导航环境。
4. 以当前冻结 Go2 低层 JIT 步态控制器为基础，依次训练 oracle 射线导航、深度预测射线导航和完整场景课程。
5. 实现 50 回合评测、轨迹 CSV/图片、深度图与射线对比图、外部视角加传感器画面的 demo 录像。

## 对照与交付

- 同一组 50 个种子对比：oracle 射线、深度预测射线、移除低障碍物场景。
- 记录感知射线 MAE/RMSE、成功率、低障碍碰撞、总碰撞、楼梯通过率和到达时间。
- 提交训练 checkpoint、评测 CSV/摘要、传感器可视化、导航轨迹和至少三条覆盖完整路线的成功视频。

## 约束

- 使用当前 SEA-Nav 的 Go2 与冻结低层步态控制器；不迁移 ABS 的 Go1 机器人或硬件部署代码。
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
