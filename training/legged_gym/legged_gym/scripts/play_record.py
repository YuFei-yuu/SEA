# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
#
# play_record.py — Headless video recording variant of play.py
# Runs policy inference in headless mode and records video via Isaac Gym camera sensor.
# Camera follows the robot automatically. Outputs an mp4 file.
#
# Usage:
#   xvfb-run -a python training/legged_gym/legged_gym/scripts/play_record.py
#   xvfb-run -a python training/legged_gym/legged_gym/scripts/play_record.py --load_run <run_dir> --checkpoint <N>

import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import numpy as np
import torch
import cv2
from isaacgym import gymapi

# ── Configurable settings ──────────────────────────────────────────────
TOTAL_EPISODES = 5          # stop after this many episodes
FPS = 30                    # video frame rate
CAM_RES = (1000, 1000)      # camera resolution
VIEW_OFFSET = (4.0, -4.0, 3.0)  # camera offset from robot (x, y, z)
MAX_FRAMES = 30000          # safety cap on total recorded frames
# ────────────────────────────────────────────────────────────────────────


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # ── Environment override for playground / recording ────────────────
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.terrain.terrain_types = ['hard_room']
    env_cfg.terrain.terrain_proportions = [1.0]
    env_cfg.asset.file = '{LEGGED_GYM_ROOT_DIR}/resources/go2_description/urdf/go2_description.urdf'
    env_cfg.replay.enable_collision_replay = False

    env_cfg.visualization.ray_groups = {
        "guidance_navigation": ["guide", "guide_ray_marker"],
    }

    if env_cfg.env.num_envs == 1:
        env_cfg.terrain.num_rows = 1
        env_cfg.terrain.num_cols = 1
        env_cfg.terrain.curriculum = True
        env_cfg.terrain.max_init_terrain_level = 3

    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = True
    env_cfg.domain_rand.max_push_vel_xy = 0.0
    env_cfg.domain_rand.randomize_base_mass = True
    env_cfg.domain_rand.added_mass_range = [0, 0]
    env_cfg.env.episode_length_s = 40
    env_cfg.env.stay_time = 500
    env_cfg.env.debug_viz = True
    env_cfg.asset.terminate_after_contacts_on = []

    # ── Force headless mode so it works without a physical display ─────
    args.headless = True

    # ── Create environment ─────────────────────────────────────────────
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # ── Load policy ────────────────────────────────────────────────────
    train_cfg.runner.resume = True
    if args.load_run is not None:
        train_cfg.runner.load_run = args.load_run
    else:
        train_cfg.runner.load_run = -1
    if args.checkpoint is not None:
        train_cfg.runner.checkpoint = args.checkpoint
    else:
        train_cfg.runner.checkpoint = -1

    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    loaded_path = task_registry.loaded_policy_path
    print(f'Loaded policy from: {loaded_path}')

    # ── Camera sensor setup ────────────────────────────────────────────
    cam_props = gymapi.CameraProperties()
    cam_props.width = CAM_RES[0]
    cam_props.height = CAM_RES[1]
    cam_handle = env.gym.create_camera_sensor(env.envs[0], cam_props)

    # ── Determine output video path ────────────────────────────────────
    run_name = os.path.basename(os.path.dirname(loaded_path))
    ckpt_name = os.path.splitext(os.path.basename(loaded_path))[0]
    log_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs',
                           train_cfg.runner.experiment_name, 'exported')
    os.makedirs(log_dir, exist_ok=True)
    # run_name already has trailing underscore, avoid double underscore
    video_path = os.path.join(log_dir, f'{run_name}{ckpt_name}.mp4')
    print(f'Video will be saved to: {video_path}')

    # ── Initialise video writer ────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_path, fourcc, FPS, CAM_RES)
    print(f'Recording started (resolution={CAM_RES}, fps={FPS})')

    # ── Simulation loop ────────────────────────────────────────────────
    obs = env.get_observations()
    env.reset()
    episode_count = 0
    frame_count = 0

    with torch.no_grad():
        for i in range(100 * int(env.max_episode_length)):
            # ── Move camera to follow robot BEFORE step ──────────────────
            root_states = env.root_states
            rx, ry, rz = root_states[0, 0], root_states[0, 1], root_states[0, 2]
            eye = gymapi.Vec3(rx + VIEW_OFFSET[0],
                              ry + VIEW_OFFSET[1],
                              rz + VIEW_OFFSET[2])
            target = gymapi.Vec3(rx, ry, rz + 0.5)
            env.gym.set_camera_location(cam_handle, env.envs[0], eye, target)

            actions = policy(obs.detach())
            obs, _, _, dones, _ = env.step(actions.detach())
            # step() → render() now correctly calls:
            #   step_graphics() → render_all_camera_sensors()
            # so the camera image reflects current physics state.

            # ── Grab frame (already rendered by render() above) ──────────
            if frame_count < MAX_FRAMES:
                img = env.gym.get_camera_image(env.sim, env.envs[0],
                                               cam_handle, gymapi.IMAGE_COLOR)
                img = img.reshape((CAM_RES[1], CAM_RES[0], 4))[:, :, :3]
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                video.write(img_bgr)
                frame_count += 1

                if frame_count % 100 == 0:
                    print(f"  recorded {frame_count} frames ...")

            else:
                print(f"Reached max frames ({MAX_FRAMES}), stopping.")
                break

            # ── Episode bookkeeping ────────────────────────────────────
            if dones.any():
                episode_count += 1
                print(f"=== Episode {episode_count}/{TOTAL_EPISODES} finished ===")

            if episode_count >= TOTAL_EPISODES:
                print(f"Reached {TOTAL_EPISODES} episodes, stopping.")
                break

    # ── Cleanup ────────────────────────────────────────────────────────
    video.release()
    file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
    print(f'Video saved: {video_path}  ({file_size_mb:.1f} MB, {frame_count} frames)')


if __name__ == '__main__':
    args = get_args()
    play(args)
