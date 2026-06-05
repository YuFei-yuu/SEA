# SEA-Nav 环境配置与复现指南

> 适用环境：AutoDL / 云服务器，NVIDIA RTX 4090 (24GB)，Ubuntu 20.04

---

## 一、环境要求

| 组件 | 要求 | 备注 |
|---|---|---|
| 操作系统 | Ubuntu 20.04 LTS | Isaac Gym 仅支持 Linux |
| GPU | NVIDIA RTX 4090 或类似 (sm_89) | 至少 8GB 显存，推荐 24GB |
| NVIDIA 驱动 | >= 525 | `nvidia-smi` 确认 |
| Python | **3.8**（必须） | Isaac Gym Preview 4 限制 `<3.9` |
| Conda | Miniconda / Anaconda | 推荐用 conda 管理环境 |

---

## 二、创建 Conda 环境

```bash
# 1. 配置清华镜像（国内加速）
cat > ~/.condarc << 'EOF'
channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
show_channel_urls: true
EOF

# 2. 创建 Python 3.8 环境
conda create -n sea_nav python=3.8 -y
conda activate sea_nav
```

---

## 三、安装系统依赖

```bash
apt-get update
apt-get install -y build-essential libpython3.8-dev libgl1-mesa-glx xvfb
```

| 包 | 用途 |
|---|---|
| `build-essential` | C++ 编译器（gymtorch JIT 编译） |
| `libpython3.8-dev` | Python 开发头文件 |
| `libgl1-mesa-glx` | OpenGL 运行时 |
| `xvfb` | 虚拟 X 显示（headless 服务器录视频必需） |

---

## 四、安装 CUDA 11.8 Toolkit

```bash
conda install -c conda-forge cudatoolkit=11.8 -y
```

> RTX 4090 (Ada Lovelace, sm_89) 需要 CUDA >= 11.8。AutoDL 系统自带 CUDA 11.3 对 RTX 4090 不完全兼容。

---

## 五、安装 PyTorch 2.0.1

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install torch==2.0.1 torchvision==0.15.2 \
  --index-url https://download.pytorch.org/whl/cu118
```

> 选择 PyTorch 2.0.1（不是最新的 2.5+），以减少与 Isaac Gym gymtorch C++ API 的兼容性问题。

验证 PyTorch CUDA:
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# 期望输出: True \n NVIDIA GeForce RTX 4090
```

---

## 六、安装 Isaac Gym Preview 4

```bash
# 解压仓库自带的 Isaac Gym 包
cd /tmp
tar xzf /home/sea_ws/src/IsaacGym_Preview_4_Package.tar.gz
cd isaacgym/python
pip install -e .
```

### ⚠️ 兼容性修复 1: numpy.float 已弃用

Isaac Gym 的 `torch_utils.py` 使用了 `np.float`，在 NumPy >= 1.24 中被移除。

```bash
# 修复
sed -i 's/dtype=np\.float,/dtype=np.float64,/' /tmp/isaacgym/python/isaacgym/torch_utils.py
```

或手动编辑 `/tmp/isaacgym/python/isaacgym/torch_utils.py` 第 135 行：
```python
# 改前: def get_axis_params(value, axis_idx, x_value=0., dtype=np.float, n_dims=3):
# 改后: def get_axis_params(value, axis_idx, x_value=0., dtype=np.float64, n_dims=3):
```

---

## 七、安装项目本地包

```bash
# 安装 rsl_rl（PPO 算法 + CBF 安全层）
cd /home/sea_ws/src/training/rsl_rl
pip install -e .

# 安装 legged_gym（Isaac Gym 环境）
cd /home/sea_ws/src/training/legged_gym
pip install -e .
```

### ⚠️ 兼容性修复 2: JIT 模型 device 不匹配

TorchScript JIT 模型默认加载到 CPU，与 GPU 上的 Isaac Gym 张量冲突。

编辑 `training/legged_gym/legged_gym/envs/base/legged_robot_pos.py` 第 133-135 行：
```python
# 改前:
# self.slr_body = torch.jit.load(...)
# self.slr_encoder_vel = torch.jit.load(...)
# self.slr_encoder_latent = torch.jit.load(...)

# 改后:
self.slr_body = torch.jit.load(f"{LEGGED_GYM_ROOT_DIR}/legged_gym/ctrl_model/body_latest.jit", map_location=self.device)
self.slr_encoder_vel = torch.jit.load(f"{LEGGED_GYM_ROOT_DIR}/legged_gym/ctrl_model/encoder_vel.jit", map_location=self.device)
self.slr_encoder_latent = torch.jit.load(f"{LEGGED_GYM_ROOT_DIR}/legged_gym/ctrl_model/encoder_latent.jit", map_location=self.device)
```

### ⚠️ 兼容性修复 3: headless 模式下渲染管线缺失 step_graphics

Isaac Gym 的 `base_task.py` 中 `render()` 方法将 `step_graphics()` 放在了 `if self.viewer:` 块内。headless 模式下 viewer 为 None，`step_graphics()` 永不执行，导致 camera sensor 始终渲染机器人的初始位置（物理引擎最新位姿未同步到图形管线）。

编辑 `training/legged_gym/legged_gym/envs/base/base_task.py` 第 126-153 行的 `render()` 方法：

```python
# 改前 (step_graphics 在 viewer 块内，headless 时永不执行):
# def render(self, sync_frame_time=True):
#     self.gym.render_all_camera_sensors(self.sim)
#     if self.viewer:
#         ...
#         self.gym.step_graphics(self.sim)   # ← headless 时跳过
#         ...

# 改后 (step_graphics 无条件执行):
def render(self, sync_frame_time=True):
    # Sync physics state to graphics system (required for camera sensors
    # in both headless and non-headless modes)
    self.gym.step_graphics(self.sim)
    self.gym.render_all_camera_sensors(self.sim)

    if self.viewer:
        ...
        if self.enable_viewer_sync:
            self.gym.draw_viewer(self.viewer, self.sim, True)
            if sync_frame_time:
                self.gym.sync_frame_time(self.sim)
        else:
            self.gym.poll_viewer_events(self.viewer)
```

这是 Isaac Gym 官方文档指定的正确顺序：`simulate → fetch_results → step_graphics → render_all_camera_sensors`

### ⚠️ 兼容性修复 4: train.py 的 print_config 简化

原版 `train.py` 中 `print_config()` 调用 `class_to_dict(Go2PosRoughCfg.rewards)` 可能在部分环境下导致崩溃。

编辑 `training/legged_gym/legged_gym/scripts/train.py` 第 51 行：

```python
# 改前:
# config = print_config()

# 改后:
config = {}
```

---

## 八、安装其他 Python 依赖

```bash
pip install wandb matplotlib opencv-python scipy pyyaml pillow imageio ninja
```

---

## 九、验证环境

```bash
# 1. 验证 PyTorch CUDA
python -c "import torch; assert torch.cuda.is_available(); print('CUDA OK')"

# 2. 验证 Isaac Gym 模块
python -c "from isaacgym import gymapi, gymtorch, gymutil; print('Isaac Gym OK')"

# 3. 验证本地包
python -c "import legged_gym, rsl_rl; print('Packages OK')"
```

---

## 十、训练

```bash
# 激活环境
conda activate sea_nav
cd /home/sea_ws/src

# Headless 训练（无 GUI）
WANDB_MODE=disabled xvfb-run -a \
  python training/legged_gym/legged_gym/scripts/train.py \
  --headless --num_envs 2048 --max_iterations 2000

# 关键参数说明:
#   --headless         无图形界面模式
#   --num_envs N       并行环境数（2048=默认，显存不足时减半）
#   --max_iterations N 训练迭代数（默认 2000）
#   WANDB_MODE=disabled 禁用 wandb（没账号时用）
#   xvfb-run -a        提供虚拟 X 显示
```

训练日志和 checkpoint 保存在 `training/legged_gym/logs/Go2_pos_rough/` 下，以时间戳命名。

---

## 十一、可视化（视频录制）

使用 `play_record.py` 在 headless 模式下录制仿真视频：

```bash
conda activate sea_nav
cd /home/sea_ws/src

# 默认加载最新 checkpoint，录制 5 个 episode
xvfb-run -a python training/legged_gym/legged_gym/scripts/play_record.py

# 指定 checkpoint
xvfb-run -a python training/legged_gym/legged_gym/scripts/play_record.py \
  --load_run 06_06_XX-XX-XX_ --checkpoint 50
```

视频输出路径：`training/legged_gym/logs/Go2_pos_rough/exported/<run>_model_XX.mp4`

下载视频到本地即可观看四足机器人在障碍物环境中导航。

---

## 十二、实时 GUI 交互（VNC 远程桌面）

Isaac Gym 的 `play.py` 需要显示器才能打开仿真 GUI viewer。通过 VNC 可以在远程服务器上获得图形桌面，实时观察机器人导航、调试策略。

### 12.1 安装桌面环境 + VNC Server

```bash
apt-get install -y xfce4 xfce4-terminal tigervnc-standalone-server tigervnc-common
```

### 12.2 设置 VNC 密码

```bash
mkdir -p ~/.vnc
echo "your_password" | vncpasswd -f > ~/.vnc/passwd
chmod 600 ~/.vnc/passwd
```

### 12.3 创建 VNC 启动脚本

创建 `~/start_vnc.sh`：

```bash
#!/bin/bash
# Kill any existing VNC session on :1
vncserver -kill :1 2>/dev/null

# Start VNC server
vncserver :1 \
  -geometry 1920x1080 \
  -depth 24 \
  -localhost no \
  -SecurityTypes VncAuth \
  --I-Know-TigerVNC-is-Unsupported-2026

echo "VNC server running on :1 (port 5901)"
```

### 12.4 启动 VNC

```bash
chmod +x ~/start_vnc.sh
~/start_vnc.sh
```

### 12.5 本地 SSH 端口转发

在**本地电脑**终端执行（不是服务器）：

```bash
ssh -L 5901:localhost:5901 -p <SSH_PORT> root@<AUTODL_IP>
```

### 12.6 连接 VNC Viewer

在本地电脑打开 VNC 客户端（如 RealVNC、TigerVNC Viewer），连接：

```
localhost:5901
```

输入之前设置的 VNC 密码即可看到远程桌面。

### 12.7 在 VNC 桌面中运行仿真

在 VNC 桌面的终端中：

```bash
conda activate sea_nav
cd /home/sea_ws/src

# 运行原始 play.py（需要 GUI viewer）
python training/legged_gym/legged_gym/scripts/play.py

# 或训练（可观察 robot 在 viewer 中的行为）
python training/legged_gym/legged_gym/scripts/train.py
```

> **注意**: 原版 `play.py` 强制 `args.headless = False`（第 170 行），在 VNC 中有真正的 X display，因此可以正常打开 Isaac Gym viewer 窗口。

---

## 附录: 已知问题与解决

### RTX 4090 与 PhysX GPU

Isaac Gym Preview 4 的 `libPhysXGpu_64.so` 只包含 sm_70 的预编译 CUDA 内核。在 RTX 4090 (sm_89) 上，NVIDIA CUDA 驱动通过 JIT PTX 编译使其仍能运行，但有 segfault 风险。

**如果遇到 PhysX GPU 崩溃**：
- 降低 `--num_envs`（如 64、128、256）
- 或使用 CPU PhysX: 添加 `--sim_device=cpu`

### train.py 未修复前的 print_config 崩溃

原版 `train.py` 中 `print_config()` 调用会导致在部分环境下崩溃。已修复为 `config={}` 绕过。

### gymtorch 首次加载慢

`gymtorch` 在首次 import 时需要进行 C++ JIT 编译（约 1-2 分钟），这是正常的。编译结果缓存于 `~/.cache/torch_extensions/`，后续加载即快。
