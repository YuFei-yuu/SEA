# SEA-Nav 最简双向台阶导航

## 任务与接口

- `go2_blind_stair_loco`：SEA Go2 原生盲式低层任务，actor/critic 均使用 45 维本体观测，输出 12 维外部关节顺序动作。
- `go2_blind_stair_loco_forward_finetune`：从既有低层权重进行头朝前正向下行微调，观测、动作和控制契约与原低层任务完全一致。
- `go2_pos_stairs_minimal`：保留 350 维上层观测和三维速度接口，21 条射线恒为最大距离，不创建相机、不查询高度感知结果。
- 低层控制固定为 50 Hz、action scale 0.25、`Kp=30`、`Kd=0.75`。TorchScript 加载时逐项检查观测布局、12 个关节顺序、默认角、PD、频率和模型哈希。
- 最简房间保留五级 `0.08 m x 0.30 m` 台阶和 `0.40 m` 高台，移除 33 个低矮盒体；原 `go2_pos_depth_stairs` 不变。

终止原因仅有 `success / stair_stuck / fall_or_contact / timeout / other_stuck`。`episode_length >= max_episode_length` 即 timeout；timeout 与 success 同步触发时只记录 timeout。成功还要求正确台面高度、目标平面距离小于 0.5 m、机身朝向目标、基座越过台阶外缘至少 0.8 m、四足完全进入目标平地，并连续保持 12 步。

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

先直接评测现有 `Go2_pos_depth_stairs/.../model_200.pt`。脚本默认加载该 checkpoint，并强制分别完成上行、下行各 50 回合：

```bash
python training/legged_gym/legged_gym/scripts/evaluate_stairs_minimal.py \
  --task go2_pos_stairs_minimal --headless \
  --episodes_per_direction 50 --success_threshold 0.60 \
  --output_csv training/legged_gym/logs/Go2_pos_stairs_minimal/baseline_eval.csv \
  --output_summary training/legged_gym/logs/Go2_pos_stairs_minimal/baseline_eval.json
```

只有任一方向低于 60% 时才启动分段微调。该入口先评测当前 checkpoint，之后每 100 iterations gate 一次，首个双向达标 checkpoint 即停止，最多新增 1000 iterations：

```bash
python training/legged_gym/legged_gym/scripts/train_stairs_minimal.py \
  --task go2_pos_stairs_minimal --headless --num_envs 8192 \
  --max_iterations 1000 --gate_interval 100 \
  --gate_episodes_per_direction 50 --gate_success_threshold 0.60
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

## 上层导航 Demo

分别录制一个成功上行和下行回合；脚本在回合不是 success 时返回失败，避免误提交 timeout 视频：

```bash
python training/legged_gym/legged_gym/scripts/record_stairs_minimal.py \
  --task go2_pos_stairs_minimal --direction up \
  --output_video training/legged_gym/logs/Go2_pos_stairs_minimal/up.mp4 \
  --output_trajectory training/legged_gym/logs/Go2_pos_stairs_minimal/up.csv

python training/legged_gym/legged_gym/scripts/record_stairs_minimal.py \
  --task go2_pos_stairs_minimal --direction down \
  --output_video training/legged_gym/logs/Go2_pos_stairs_minimal/down.mp4 \
  --output_trajectory training/legged_gym/logs/Go2_pos_stairs_minimal/down.csv
```

最终提交以评测 CSV/JSON 的逐回合 `terminal_reason` 为准，不从聚合训练日志反推成功。验收要求上、下行各至少 30/50 成功，且 success 记录必须同时包含 `stair_crossed=1`。

专用终止回归会在真实环境中构造台阶卡死、错误高度、到达后离开、最大步数同步成功和正常成功：

```bash
python training/legged_gym/legged_gym/scripts/regress_stairs_terminal.py \
  --task go2_pos_stairs_minimal --headless
```
