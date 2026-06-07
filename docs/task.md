# 动态障碍物导航 — 基于 SEA-Nav 任务文档

---

## 一、前置环境 (已完成，无需重新配置)

### 1.1 环境概况

通过ssh远程连接到服务器

| 组件 | 版本 |
|---|---|
| 服务器 | AutoDL + RTX 4090 (24GB) + Ubuntu 20.04 |
| Python | 3.8.20 (conda env: `sea_nav`) |
| PyTorch | 2.0.1+cu118 (支持 RTX 4090 sm_89) |
| CUDA Toolkit | 11.8 (conda-forge) |
| NVIDIA Driver | 570.124.04 |
| Isaac Gym | Preview 4 (安装在 `/tmp/isaacgym/`) |
| rsl_rl | 1.0.2 (editable install) |
| legged_gym | 1.0.0 (editable install) |
| 项目根目录 | `/home/sea_ws/src/` |
| VNC 桌面 | TigerVNC 端口 5901, 密码 `sea_nav_2026` |

### 1.2 使用环境

```bash
conda activate sea_nav
cd /home/sea_ws/src
```

### 1.3 已做的兼容性修复

| # | 文件 | 修改 | 原因 |
|---|---|---|---|
| 1 | `/tmp/isaacgym/python/isaacgym/torch_utils.py:135` | `np.float` → `np.float64` | numpy 1.24+ 废弃 |
| 2 | `training/legged_gym/legged_gym/envs/base/legged_robot_pos.py:133-135` | JIT `torch.jit.load()` 添加 `map_location=self.device` | CPU/GPU tensor 不匹配 |
| 3 | `training/legged_gym/legged_gym/envs/base/base_task.py:126-153` | `step_graphics()` 移到 viewer 条件外，按 Isaac Gym 规范顺序 `step_graphics → render_all_camera_sensors` | headless 模式下 camera sensor 画面不更新 |
| 4 | `training/legged_gym/legged_gym/scripts/train.py:51` | `config = print_config()` → `config = {}` | print_config 在某些环境崩溃 |

### 1.4 可用的辅助脚本

| 脚本 | 用途 |
|---|---|
| `scripts/play_record.py` | headless 视频录制（相机跟随机器人） |
| `scripts/play.py` | VNC 桌面实时交互仿真 |
| `scripts/train.py` | PPO+CBF 训练 |
| `guide.md` | 完整环境配置指南 |
| `report.md` | 上次训练报告 |
| `checkpoints_backup/` | 28 个原始 checkpoint 备份 |

---

## 二、VNC 调试指南

```bash
# 在本地终端 (Windows PowerShell) 建立 SSH 隧道
ssh -L 5901:localhost:5901 -p 22457 root@connect.cqa1.seetacloud.com

# TigerVNC Viewer 连接 localhost:5901，密码 sea_nav_2026

# VNC 桌面终端中运行仿真
conda activate sea_nav
cd /home/sea_ws/src
python training/legged_gym/legged_gym/scripts/play.py  # 实时 GUI
```

---

## 三、快速命令参考

```bash
conda activate sea_nav && cd /home/sea_ws/src

# 训练
WANDB_MODE=disabled xvfb-run -a python training/legged_gym/legged_gym/scripts/train.py --headless --num_envs 1024 --max_iterations 200

# 视频录制
xvfb-run -a python training/legged_gym/legged_gym/scripts/play_record.py

# 单步测试
python -X faulthandler -c "
import sys; sys.path.insert(0,'training/legged_gym')
from legged_gym.envs import *; from legged_gym.utils import get_args, task_registry
args=get_args(); args.headless=True; args.num_envs=2
env,_=task_registry.make_env(name='go2_pos_rough',args=args)
obs=env.get_observations()
for i in range(10):
    obs,_,_,_,_=env.step(torch.zeros(2,3,device=env.device))
print('OK')
"

# 检查 GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"
```
