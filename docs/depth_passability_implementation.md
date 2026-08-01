# 楼梯场景导航：Depth Sensor 与通过性判断实施方案

## 目标

在当前目标房间中完成包含五级楼梯、0.40 m 平台和低矮障碍物的双向导航。冻结低层 locomotion，只训练上层导航；上层必须通过 Depth Sensor 感知低矮障碍物和楼梯，避免把已知地图直接作为部署输入。

最终观测链路为：

```text
160x90 clean depth -> DepthRayNet -> 21 条前向射线
                              -> (12 本体 + 21 射线 + 2 目标) x 10 = 350 维
                              -> [vx, vy, yaw_rate]
```

## 已确定的设计

- 场景范围：当前目标房间，随机起点、目标横向位置和上下行方向；障碍物布局固定。
- 训练流程：局部目标驱动 Teacher -> BC/DAgger -> 短程 PPO 微调。
- Teacher：直达目标优先；只有当前直达线被低矮障碍物阻挡时才选择左右切点，按实时位姿重规划，不再按起点绑定整条 A* 路线。
- 通过性判断：共享策略特征后增加四分类辅助头：`free`、`low_obstacle_bypass`、`stair_passable`、`blocked`。辅助头只用于训练和统计，不改变部署动作接口。
- 深度鲁棒性：当前阶段只使用 clean depth，不加入噪声、缺失和截断增强；后续如需要真实传感器迁移，再单独增加增强实验。
- 验收：上行、下行各 20 个未见 seed，至少 15/20 成功；成功回合零低障碍碰撞，并完整离开楼梯。

## 实现模块

1. `go2_pos_depth_stairs_passability`：真实深度相机、预测射线、双向起点和通过性标签。
2. `local_room_teacher.py`：GPU 批量局部直达/左右绕行选择器和 Teacher 诊断指标。
3. `DepthRayNet` 数据链路：按 episode/seed 采集深度-射线对，训练、验证和测试分组隔离。
4. `DifferentiableSafeActorCritic`、`RolloutStorage`、`PPO`：可选通过性辅助头和交叉熵损失。
5. 训练、评测、Teacher 诊断及中文记录文档。

## 关键约束

- 低层模型路径、观测布局、关节顺序、PD 参数和 50 Hz 控制频率不得改变。
- oracle 射线用于 Teacher 标签、8192 环境策略预训练和消融；最终微调、验收与部署 actor 必须显式使用 `depth_predicted`，checkpoint 不能停留在 oracle 阶段。
- `go2_pos_stairs_minimal` 保留为历史基线；不能把其已知地图 AABB 射线当作最终 Depth Sensor 结果。
- Teacher 的路径效率必须单独记录，防止“先横移、再前进”的固定模板重新进入训练数据。

## 运行顺序

```bash
cd /home/sea_ws/src
export PYTHONPATH=/home/sea_ws/src/training/legged_gym:/home/sea_ws/src/training/rsl_rl
export WANDB_MODE=disabled

# 1. Depth 数据
python training/legged_gym/legged_gym/scripts/collect_depth_stairs.py \
  --task go2_pos_depth_stairs_passability --headless --num_envs 64 \
  --output_dir training/legged_gym/depth_data/stairs_clean --num_steps 2000

# 2. DepthRayNet
python training/legged_gym/legged_gym/scripts/train_depth_rays.py \
  --data_dir training/legged_gym/depth_data/stairs_clean \
  --output training/legged_gym/depth_models/stairs_clean_best.pt

# 3. Teacher 路径诊断
python training/legged_gym/legged_gym/scripts/diagnose_depth_teacher.py \
  --task go2_pos_depth_stairs_passability --headless \
  --output_csv training/legged_gym/logs/Go2_pos_depth_stairs_passability/teacher.csv

# 4. Teacher BC/DAgger + oracle PPO 预训练（oracle 模式自动关闭相机）
python training/legged_gym/legged_gym/scripts/train_depth_passability.py \
  --task go2_pos_depth_stairs_passability --headless --num_envs 8192 \
  --depth_mode oracle --teacher_steps 3000 --ppo_iterations 100 \
  --output_summary training/legged_gym/logs/Go2_pos_depth_stairs_passability/pretrain.json

# 5. 8192 环境 PPO 大并行 smoke test
python training/legged_gym/legged_gym/scripts/train.py \
  --task go2_pos_depth_stairs_passability --headless --num_envs 8192 --max_iterations 1 \
  --depth_mode oracle --no_wandb

# 6. 真实 DepthRayNet 链路功能 smoke
python training/legged_gym/legged_gym/scripts/train.py \
  --task go2_pos_depth_stairs_passability --headless --num_envs 32 --max_iterations 2 \
  --depth_mode depth_predicted \
  --depth_model training/legged_gym/depth_models/stairs_clean_best.pt \
  --init_checkpoint <oracle_pretrain_run>/model_100.pt

# 7. 真实深度短程 PPO 微调；当前 24 GiB 机器从 1024 环境起跑
python training/legged_gym/legged_gym/scripts/train.py \
  --task go2_pos_depth_stairs_passability --headless --num_envs 1024 \
  --depth_mode depth_predicted \
  --depth_model training/legged_gym/depth_models/stairs_clean_best.pt \
  --depth_update_hz 2 \
  --init_checkpoint <oracle_pretrain_run>/model_100.pt

# 8. 最终评测（训练完成后替换 run/checkpoint）
python training/legged_gym/legged_gym/scripts/evaluate_depth_stairs.py \
  --task go2_pos_depth_stairs_passability --headless \
  --depth_mode depth_predicted \
  --depth_model training/legged_gym/depth_models/stairs_balanced_weighted_best.pt \
  --eval_seed_base 200000 \
  --num_episodes 40 \
  --output_csv training/legged_gym/logs/Go2_pos_depth_stairs_passability/final.csv \
  --output_summary training/legged_gym/logs/Go2_pos_depth_stairs_passability/final.json
```

## Teacher 通过标准

- 无障碍直达样本的 lateral-only action 比例不超过 15%。
- 首次有效前向速度时间不超过 0.75 s。
- `mean_plan_ratio`（局部绕行计划长度/目标直线距离）中位数不超过 1.35，95 分位不超过 1.60。
- 低矮障碍阻挡时必须出现左右绕行候选；楼梯区必须保持正向机体系速度。
