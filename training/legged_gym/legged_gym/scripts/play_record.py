# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
import os

from legged_gym import LEGGED_GYM_ROOT_DIR
import isaacgym
from isaacgym import gymapi
import cv2
import torch

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


TOTAL_EPISODES = 5
FPS = 30
CAM_RES = (1000, 1000)
VIEW_OFFSET = (4.0, -4.0, 3.0)
MAX_FRAMES = 30000
DYNAMIC_TASKS = {
    "go2_pos_sparse_static",
    "go2_pos_dynamic_1",
    "go2_pos_dynamic_2",
    "go2_pos_dynamic_3",
}


def _has_saved_runs(experiment_name):
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", experiment_name)
    if not os.path.isdir(log_root):
        return False
    for entry in os.listdir(log_root):
        if entry == "exported":
            continue
        if os.path.isdir(os.path.join(log_root, entry)):
            return True
    return False


def _apply_record_overrides(env_cfg, args):
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.replay.enable_collision_replay = False
    env_cfg.visualization.ray_groups = {
        "guidance_navigation": ["guide", "guide_ray_marker"],
    }

    if args.task == "go2_pos_rough":
        env_cfg.terrain.terrain_types = ["hard_room"]
        env_cfg.terrain.terrain_proportions = [1.0]
        env_cfg.terrain.num_rows = 1
        env_cfg.terrain.num_cols = 1
        env_cfg.terrain.curriculum = True
        env_cfg.terrain.max_init_terrain_level = 3
    else:
        env_cfg.terrain.num_rows = 1
        env_cfg.terrain.num_cols = 1
        env_cfg.terrain.curriculum = False
        env_cfg.terrain.max_init_terrain_level = 0

    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.max_push_vel_xy = 0.0
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.added_mass_range = [0.0, 0.0]
    env_cfg.env.episode_length_s = 40
    env_cfg.env.debug_viz = True
    if hasattr(env_cfg.env, "stay_time"):
        env_cfg.env.stay_time = 500
    env_cfg.asset.terminate_after_contacts_on = []


def _maybe_set_resume_fallback(train_cfg, args):
    if args.resume_experiment_name is not None:
        train_cfg.runner.resume_experiment_name = args.resume_experiment_name
        return
    if args.task not in DYNAMIC_TASKS:
        return
    if not _has_saved_runs(train_cfg.runner.experiment_name):
        train_cfg.runner.resume_experiment_name = "Go2_pos_rough"


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    _apply_record_overrides(env_cfg, args)
    args.headless = True

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    train_cfg.runner.resume = True
    train_cfg.runner.load_run = args.load_run if args.load_run is not None else -1
    train_cfg.runner.checkpoint = args.checkpoint if args.checkpoint is not None else -1
    _maybe_set_resume_fallback(train_cfg, args)

    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    loaded_path = task_registry.loaded_policy_path
    print(f"Loaded policy from: {loaded_path}")

    cam_props = gymapi.CameraProperties()
    cam_props.width = CAM_RES[0]
    cam_props.height = CAM_RES[1]
    cam_handle = env.gym.create_camera_sensor(env.envs[0], cam_props)

    run_name = os.path.basename(os.path.dirname(loaded_path))
    ckpt_name = os.path.splitext(os.path.basename(loaded_path))[0]
    log_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name, "exported")
    os.makedirs(log_dir, exist_ok=True)
    video_path = os.path.join(log_dir, f"{run_name}{ckpt_name}.mp4")
    print(f"Video will be saved to: {video_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(video_path, fourcc, FPS, CAM_RES)
    print(f"Recording started (resolution={CAM_RES}, fps={FPS})")

    obs, _ = env.reset()
    episode_count = 0
    frame_count = 0

    with torch.no_grad():
        for _ in range(100 * int(env.max_episode_length)):
            root_states = env.root_states
            rx, ry, rz = root_states[0, 0], root_states[0, 1], root_states[0, 2]
            eye = gymapi.Vec3(rx + VIEW_OFFSET[0], ry + VIEW_OFFSET[1], rz + VIEW_OFFSET[2])
            target = gymapi.Vec3(rx, ry, rz + 0.5)
            env.gym.set_camera_location(cam_handle, env.envs[0], eye, target)

            actions = policy(obs.detach())
            obs, _, _, dones, infos = env.step(actions.detach())

            if frame_count < MAX_FRAMES:
                img = env.gym.get_camera_image(env.sim, env.envs[0], cam_handle, gymapi.IMAGE_COLOR)
                img = img.reshape((CAM_RES[1], CAM_RES[0], 4))[:, :, :3]
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                video.write(img_bgr)
                frame_count += 1
                if frame_count % 100 == 0:
                    print(f"  recorded {frame_count} frames ...")
            else:
                print(f"Reached max frames ({MAX_FRAMES}), stopping.")
                break

            if dones.any():
                episode_count += 1
                episode_info = infos.get("episode", {})
                success = float(episode_info.get("success", 0.0))
                dyn_col = float(episode_info.get("dynamic_collision_count", 0.0))
                print(f"=== Episode {episode_count}/{TOTAL_EPISODES} finished | success={success:.2f} dynamic_collision_count={dyn_col:.2f} ===")

            if episode_count >= TOTAL_EPISODES:
                print(f"Reached {TOTAL_EPISODES} episodes, stopping.")
                break

    video.release()
    file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
    print(f"Video saved: {video_path}  ({file_size_mb:.1f} MB, {frame_count} frames)")


if __name__ == "__main__":
    args = get_args()
    play(args)
