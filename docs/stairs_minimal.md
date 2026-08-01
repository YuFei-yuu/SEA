# SEA-Nav 最简双向台阶导航

## 任务与接口

- `go2_blind_stair_loco`：SEA Go2 原生盲式低层任务，actor/critic 均使用 45 维本体观测，输出 12 维外部关节顺序动作。
- `go2_blind_stair_loco_forward_finetune`：从既有低层权重进行头朝前正向下行微调，观测、动作和控制契约与原低层任务完全一致。
- `go2_pos_stairs_minimal`：保留 350 维上层观测和三维速度接口，21 条射线由房间墙体和 20 个低障碍物的二维 AABB 计算；AABB 膨胀 `0.15 m`，台阶和高台不进入障碍射线。
- 低层控制固定为 50 Hz、action scale 0.25、`Kp=30`、`Kd=0.75`。TorchScript 加载时逐项检查观测布局、12 个关节顺序、默认角、PD、频率和模型哈希。
- 房间保留五级 `0.08 m x 0.30 m` 台阶、`0.40 m` 高台以及原 33 个低矮盒体中的 20 个稀疏子集。盒体中心、完整 XY 尺寸和 `0.08/0.10/0.12 m` 高度均直接沿用原配置，不再缩放；减少数量用于保证相邻障碍之间有足够通行间隔。不创建相机，也不训练深度网络。
- 每个方向的起点均在横向直线上随机采样：上行 `x=0.70, y=1.00..9.00`，下行 `x=8.60, y=1.00..9.00`。终点固定为上行 `(7.20, 5.00)`、下行 `(0.70, 5.00)`；下行起点后移用于留出斜向接近台阶的距离。
- 训练 teacher 使用固定地图 A* 预判 20 个障碍物的通过性，再对路径做可见性简化。原 350 维布局不变，其中原有 2 维目标项指向当前 A* 局部航点；actor 不接收额外特权维度。台阶入口 `(4.40, 4.55)` 与出口 `(7.10, 4.85)` 横向错开 `0.30 m`，形成约 `6.3°` 的斜向台阶段，同时保持机头朝前。
- 固定低层使用 `Go2_blind_stair_loco_forward_finetune/branches/from_2800_forward_lr1e4_0731a/exports/blind_stair_loco_iter_0250.pt` 及同名 JSON 元数据。

终止原因仅有 `success / stair_stuck / fall_or_contact / timeout / other_stuck`。低障碍物接触并入 `fall_or_contact`，同时单独记录 `collision_class=low_obstacle`；正常台阶足端接触不终止。`episode_length >= max_episode_length` 即 timeout；timeout 与 success 同步触发时只记录 timeout。成功还要求零低障碍碰撞、正确台面高度、目标平面距离小于 0.5 m、机身朝向目标、基座越过台阶外缘至少 0.8 m、四足完全进入目标平地，并连续保持 12 步。

## 环境

在 `/home/sea_ws/src` 执行：

```bash
conda activate sea_nav
export PYTHONPATH=/home/sea_ws/src/training/legged_gym:/home/sea_ws/src/training/rsl_rl
export WANDB_MODE=disabled
```

## Smoke Test

低层先运行 32 environments、2 iterations：

```bash
python training/legged_gym/legged_gym/scripts/train.py \
  --task go2_blind_stair_loco --headless --num_envs 32 --max_iterations 2
```

再验证 8192 environments 的单次 PPO 更新：

```bash
python training/legged_gym/legged_gym/scripts/train.py \
  --task go2_blind_stair_loco --headless --num_envs 8192 --max_iterations 1
```

低层模型导出后，对上层执行相同的 `32 x 2` 和 `8192 x 1` 检查；命令中的旧 checkpoint 与新任务严格保持 350 维兼容：

```bash
python training/legged_gym/legged_gym/scripts/train.py \
  --task go2_pos_stairs_minimal --headless --num_envs 32 --max_iterations 2 \
  --resume --resume_experiment_name Go2_pos_depth_stairs \
  --load_run 07_14_06-25-22_ --checkpoint 200

python training/legged_gym/legged_gym/scripts/train.py \
  --task go2_pos_stairs_minimal --headless --num_envs 8192 --max_iterations 1 \
  --resume --resume_experiment_name Go2_pos_depth_stairs \
  --load_run 07_14_06-25-22_ --checkpoint 200
```

## 低层训练

正式入口会训练到 iteration 500，之后每 100 iterations 导出并执行目标房间 gate。每组默认 20 次，8 组分别覆盖上/下行和 `0.25/0.40/0.55/0.70 m/s`；每组至少 18 次成功且零跌倒、零卡死时立即停止：

```bash
python training/legged_gym/legged_gym/scripts/train_blind_stair_loco.py \
  --num_envs 8192 --first_gate_iteration 500 \
  --gate_interval 100 --max_iterations 3000
```

默认产物：

- `training/legged_gym/legged_gym/ctrl_model/blind_stair_loco.pt`
- `training/legged_gym/legged_gym/ctrl_model/blind_stair_loco.pt.json`
- `training/legged_gym/logs/Go2_blind_stair_loco/target_room_gates/`

也可手动导出和 gate：

```bash
python training/legged_gym/legged_gym/scripts/export_blind_stair_loco.py \
  --checkpoint training/legged_gym/logs/Go2_blind_stair_loco/<run>/model_<iteration>.pt \
  --output training/legged_gym/legged_gym/ctrl_model/blind_stair_loco.pt

python training/legged_gym/legged_gym/scripts/gate_blind_stair_loco.py \
  --task go2_pos_stairs_minimal --headless \
  --output_csv /tmp/blind_stair_gate.csv \
  --output_summary /tmp/blind_stair_gate.json
```

## 上层评测与微调

最终上层策略用无特权观测的 actor 拟合固定地图斜向航路 teacher。teacher 只用于训练监督，部署 actor 仍只接收原有 350 维观测并输出 `[vx, vy,yaw_rate]`；固定低层不参与训练。PPO 在早期试验中会破坏行为克隆初始化，因此当前交付流程不接 PPO：

```bash
# DAgger 阶段从局部航点 BC 权重初始化，逐步转为 actor 自己驱动
python training/legged_gym/legged_gym/scripts/train_stairs_minimal.py \
  --task go2_pos_stairs_minimal --headless --num_envs 8192 \
  --teacher_pretrain_steps 3000 --teacher_update_interval 2 \
  --teacher_actor_rollout_fraction 1.0 \
  --max_iterations 0 --no_gate \
  --init_checkpoint training/legged_gym/logs/Go2_pos_stairs_minimal/08_01_03-24-29_/model_teacher_pretrained.pt
```

本次产出的可用模型为：

```text
training/legged_gym/logs/Go2_pos_stairs_minimal/08_01_03-34-44_/model_teacher_pretrained.pt
```

用未参与训练的 seed 分别完成上行、下行各 20 回合，并统计斜向与三维联合动作比例：

```bash
python training/legged_gym/legged_gym/scripts/evaluate_stairs_minimal.py \
  --task go2_pos_stairs_minimal --headless \
  --checkpoint_path training/legged_gym/logs/Go2_pos_stairs_minimal/08_01_03-34-44_/model_teacher_pretrained.pt \
  --episodes_per_direction 20 --success_threshold 0.75 \
  --up_seed_base 12000 --down_seed_base 13000 \
  --output_csv training/legged_gym/logs/Go2_pos_stairs_minimal/08_01_03-34-44_/eval_unseen_20x20.csv \
  --output_summary training/legged_gym/logs/Go2_pos_stairs_minimal/08_01_03-34-44_/eval_unseen_20x20.json
```

2026-08-01 最终 actor 实测上行 `20/20`、下行 `20/20`，零低障碍/墙体碰撞。上行/下行平均斜向命令比例为 `31.9% / 39.5%`，`vx/vy/yaw` 同时非零比例为 `21.2% / 25.7%`，台阶段侧向命令比例为 `15.0% / 47.9%`。40 回合均进入障碍区且没有绕房间边缘。

冻结低层的三维联合命令探针：

```bash
python training/legged_gym/legged_gym/scripts/diagnose_stairs_minimal_omnidirectional.py \
  --task go2_pos_stairs_minimal --headless \
  --output_csv training/legged_gym/logs/Go2_pos_stairs_minimal/omnidirectional_locomotion_probe_model250/results.csv \
  --output_json training/legged_gym/logs/Go2_pos_stairs_minimal/omnidirectional_locomotion_probe_model250/summary.json
```

6 组平地联合命令覆盖 `[0.30, ±0.20, 0]`、`[0.30, ±0.18, ±0.30]` 和 `[0.18, ±0.25, ±0.25]`。各轴响应方向全部正确，零低障碍/墙体碰撞，证明原来的横纵分段来自上层 teacher，而不是冻结低层不支持全向命令。

若行为克隆策略低于门槛，只继续增加 teacher 拟合步数并重新使用独立 seed 评测，不接 PPO：

```bash
python training/legged_gym/legged_gym/scripts/train_stairs_minimal.py \
  --task go2_pos_stairs_minimal --headless --num_envs 8192 \
  --teacher_pretrain_steps 1000 --teacher_update_interval 2 \
  --max_iterations 0 --no_gate \
  --init_checkpoint <current_bc_run>/model_teacher_pretrained.pt
```

评测微调后的 checkpoint 时显式切换实验目录：

```bash
python training/legged_gym/legged_gym/scripts/evaluate_stairs_minimal.py \
  --task go2_pos_stairs_minimal --headless \
  --resume_experiment_name Go2_pos_stairs_minimal \
  --load_run <minimal_run> --checkpoint <checkpoint> \
  --episodes_per_direction 50 \
  --output_csv training/legged_gym/logs/Go2_pos_stairs_minimal/final_eval.csv \
  --output_summary training/legged_gym/logs/Go2_pos_stairs_minimal/final_eval.json
```

## 低层人工审查视频

低层 checkpoint 的标准审查方式使用 `record_blind_stair_loco.py`，不是下文的上层导航录制器。当前录制契约为：

- 使用课程第 6 行的五级 `0.08 m x 0.30 m` 训练台阶。
- 关闭观测噪声、摩擦/质量随机化和外力扰动。
- 上下行都发送机体系正向 `vx>0`；下行以 yaw=`pi` 使头部朝向世界下台阶方向。
- 基座越过台阶外缘至少 0.8 m、四足越过外缘至少 0.05 m、足端位于目标平面、朝向误差不超过 20 度，并连续保持 12 步后才记为 `fully_cleared`。
- 视频采用近距离跟随视角，默认 `960x540 @ 50 FPS`；画面显示 iteration、正向速度命令、局部位置、基座高度、朝向对齐值和四足离阶状态。
- 每段视频同时输出逐步 CSV 和方向汇总 JSON，JSON 记录 checkpoint 路径、iteration 和 SHA256。任一试验未完整离阶时脚本返回失败。

容器内带相机录制需要虚拟显示，不能使用关闭 graphics 的纯 headless 路径。单速度上下行复核命令：

```bash
CHECKPOINT=training/legged_gym/logs/Go2_blind_stair_loco_forward_finetune/<run>/model_<iteration>.pt
OUTPUT_DIR=training/legged_gym/logs/Go2_blind_stair_loco_forward_finetune/<run>/review_model_<iteration>_fixed_sim

xvfb-run -a -s '-screen 0 1280x720x24' \
  conda run -n sea_nav python \
  training/legged_gym/legged_gym/scripts/record_blind_stair_loco.py \
  --task go2_blind_stair_loco_forward_finetune \
  --direction up --speeds 0.40 \
  --checkpoint "$CHECKPOINT" --output_dir "$OUTPUT_DIR" \
  --max_steps 1000 --fps 50 --width 960 --height 540

xvfb-run -a -s '-screen 0 1280x720x24' \
  conda run -n sea_nav python \
  training/legged_gym/legged_gym/scripts/record_blind_stair_loco.py \
  --task go2_blind_stair_loco_forward_finetune \
  --direction down --speeds 0.40 \
  --checkpoint "$CHECKPOINT" --output_dir "$OUTPUT_DIR" \
  --max_steps 1000 --fps 50 --width 960 --height 540
```

省略 `--speeds` 时，每个方向依次录制 `0.25/0.40/0.55/0.70 m/s` 四段。人工审查应优先查看原始分段视频和 CSV/JSON；合并视频只用于便捷观看。

2026-07-31 的 model 250 复核产物：

```text
training/legged_gym/logs/Go2_blind_stair_loco_forward_finetune/07_31_17-18-25_from_2800_forward_lr1e4_0731a/review_model_250_fixed_sim/model_250_fixed_sim_review.mp4
```

该次 `0.40 m/s` 结果为：上行 `fully_cleared`（6.00 s），下行 `fully_cleared`（5.72 s）。

同一 TorchScript 在完整 `go2_pos_stairs_minimal` 目标房间中的固定速度复核：

```bash
python training/legged_gym/legged_gym/scripts/diagnose_stairs_minimal_locomotion.py \
  --task go2_pos_stairs_minimal --headless --speed 0.40 \
  --output_csv training/legged_gym/logs/Go2_pos_stairs_minimal/fixed_locomotion_probe_model250/results.csv \
  --output_json training/legged_gym/logs/Go2_pos_stairs_minimal/fixed_locomotion_probe_model250/summary.json
```

该诊断绕过上层策略。上行从台阶前平地 `x=3.90 m` 起步，下行从高台
`x=7.20 m` 起步且 yaw 为 `pi`，两个方向都发送机体系
`[vx, vy, yaw_rate]=[+0.40, 0, 0]`。实测上行 6.82 s、下行 6.46 s 均连续
12 步满足四足完全离阶；低障碍和墙体碰撞计数均为 0。下行最小朝向对齐值
为 0.9523，因此是机头在前的正向下台阶。

## 上层导航 Demo

每个方向批量录制 5 个随机起点的成功回合；任一回合不是 success 时脚本立即返回失败。视频只保留原始仿真画面：关闭目标点、射线、扫描点、碰撞点等调试可视化，也不叠加任何文字。需要通过虚拟显示启用 Isaac Gym 相机：

```bash
xvfb-run -a -s '-screen 0 1280x720x24' \
python training/legged_gym/legged_gym/scripts/record_stairs_minimal.py \
  --task go2_pos_stairs_minimal --direction up \
  --seeds 12001,12015,12000,12010,12018 \
  --checkpoint_path training/legged_gym/logs/Go2_pos_stairs_minimal/08_01_03-34-44_/model_teacher_pretrained.pt \
  --output_dir training/legged_gym/logs/Go2_pos_stairs_minimal/08_01_03-34-44_/videos_5_trials

xvfb-run -a -s '-screen 0 1280x720x24' \
python training/legged_gym/legged_gym/scripts/record_stairs_minimal.py \
  --task go2_pos_stairs_minimal --direction down \
  --seeds 13007,13019,13002,13018,13001 \
  --checkpoint_path training/legged_gym/logs/Go2_pos_stairs_minimal/08_01_03-34-44_/model_teacher_pretrained.pt \
  --output_dir training/legged_gym/logs/Go2_pos_stairs_minimal/08_01_03-34-44_/videos_5_trials
```

视频与同名轨迹 CSV 分别保存为 `up_trial_01..05` 和 `down_trial_01..05`。10 段均为 actor 成功回合；相机使用贴近机器人移动的侧向跟随视角，并按起点横向位置选择房间内侧，避免边界墙遮挡开场。视频无文字、目标点或射线标识。逐段轨迹的台阶段横向位移约 `0.21..1.46 m`，可直接核对机器人并非完全直上直下。按上行 1--5、下行 1--5 顺序拼接后的总视频为 `videos_5_trials/all_10_trials_side.mp4`。

最终提交以评测 CSV/JSON 的逐回合 `terminal_reason` 为准，不从聚合训练日志反推成功。验收要求上、下行各至少 15/20 成功，且 success 记录必须同时包含 `low_obstacle_collision_count=0`、`stair_crossed=1`、`fully_cleared=1`、`obstacle_field_crossed=1` 和 `whole_obstacle_zone_bypass=0`。

专用终止回归会在真实环境中构造台阶卡死、错误高度、到达后离开、最大步数同步成功和正常成功：

```bash
python training/legged_gym/legged_gym/scripts/regress_stairs_terminal.py \
  --task go2_pos_stairs_minimal --headless
```
