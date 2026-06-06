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
import numpy as np

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


TOTAL_EPISODES = 5
FPS = 30
CAM_RES = (1000, 1000)
VIEW_OFFSET = (4.0, -4.0, 3.0)
MAX_FRAMES = 30000
MINIMAP_SIZE = 260
MINIMAP_PAD = 24
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
        if hasattr(env_cfg, "sparse_room"):
            env_cfg.sparse_room.start_y_range = [4.2, 5.8]
            env_cfg.sparse_room.goal_y_range = [4.2, 5.8]
        if hasattr(env_cfg, "dynamic_obstacles"):
            env_cfg.dynamic_obstacles.y_min = 3.0
            env_cfg.dynamic_obstacles.y_max = 7.0
            env_cfg.dynamic_obstacles.force_interaction = True
            env_cfg.dynamic_obstacles.force_interaction_jitter = 0.45

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


def _draw_label(frame, text, origin, color, scale=0.55, thickness=1):
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _world_to_minimap(point_xy, room_origin_xy, room_size, canvas_origin, canvas_size):
    local = point_xy - room_origin_xy
    local = np.clip(local, 0.0, room_size)
    px = int(round(canvas_origin[0] + (local[0] / room_size) * canvas_size))
    py = int(round(canvas_origin[1] + canvas_size - (local[1] / room_size) * canvas_size))
    return px, py


def _draw_minimap_overlay(frame, env, episode_idx, frame_idx, latest_episode_info):
    if not hasattr(env, "room_origins"):
        return frame

    overlay = frame.copy()
    x0 = frame.shape[1] - MINIMAP_SIZE - MINIMAP_PAD
    y0 = MINIMAP_PAD
    x1 = frame.shape[1] - MINIMAP_PAD
    y1 = MINIMAP_PAD + MINIMAP_SIZE
    cv2.rectangle(overlay, (x0 - 12, y0 - 12), (x1 + 12, y1 + 86), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    room_size = float(env.terrain.env_length)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (215, 215, 215), 2)

    room_origin = env.room_origins[0].detach().cpu().numpy()
    robot_xy = env.root_states[0, :2].detach().cpu().numpy()
    start_xy = env.env_origins[0, :2].detach().cpu().numpy()
    goal_xy = env.position_targets[0, :2].detach().cpu().numpy()

    # Draw static pillars for sparse-room tasks.
    sparse_cfg = getattr(env.cfg, "sparse_room", None)
    if sparse_cfg is not None:
        half_x = sparse_cfg.pillar_size[0] * 0.5
        half_y = sparse_cfg.pillar_size[1] * 0.5
        for center_x, center_y in sparse_cfg.pillar_centers:
            p0 = _world_to_minimap(np.array([room_origin[0] + center_x - half_x, room_origin[1] + center_y - half_y]), room_origin, room_size, (x0, y0), MINIMAP_SIZE)
            p1 = _world_to_minimap(np.array([room_origin[0] + center_x + half_x, room_origin[1] + center_y + half_y]), room_origin, room_size, (x0, y0), MINIMAP_SIZE)
            cv2.rectangle(frame, (min(p0[0], p1[0]), min(p0[1], p1[1])), (max(p0[0], p1[0]), max(p0[1], p1[1])), (105, 105, 105), -1)

    if hasattr(env, "dynamic_root_states") and env.dynamic_root_states.shape[1] > 0:
        for obs_xy in env.dynamic_root_states[0, :, :2].detach().cpu().numpy():
            px, py = _world_to_minimap(obs_xy, room_origin, room_size, (x0, y0), MINIMAP_SIZE)
            cv2.circle(frame, (px, py), 8, (0, 140, 255), -1)
            cv2.circle(frame, (px, py), 10, (255, 255, 255), 1)

    start_px = _world_to_minimap(start_xy, room_origin, room_size, (x0, y0), MINIMAP_SIZE)
    goal_px = _world_to_minimap(goal_xy, room_origin, room_size, (x0, y0), MINIMAP_SIZE)
    robot_px = _world_to_minimap(robot_xy, room_origin, room_size, (x0, y0), MINIMAP_SIZE)
    cv2.circle(frame, start_px, 7, (0, 255, 255), -1)
    cv2.circle(frame, goal_px, 8, (255, 80, 80), -1)
    cv2.circle(frame, robot_px, 7, (80, 255, 80), -1)
    cv2.line(frame, start_px, goal_px, (160, 160, 160), 1, cv2.LINE_AA)

    dist_to_goal = float(torch.norm(env.position_targets[0, :2] - env.root_states[0, :2]).item())
    _draw_label(frame, f"Episode: {episode_idx + 1}/{TOTAL_EPISODES}", (x0 - 4, y1 + 24), (255, 255, 255))
    _draw_label(frame, f"Frame: {frame_idx}", (x0 - 4, y1 + 46), (255, 255, 255))
    _draw_label(frame, f"Dist->Goal: {dist_to_goal:.2f}m", (x0 - 4, y1 + 68), (255, 255, 255))

    if latest_episode_info is not None:
        success = float(latest_episode_info.get("success", 0.0))
        dyn_col = float(latest_episode_info.get("dynamic_collision_count", 0.0))
        total_col = float(latest_episode_info.get("total_collision_count", dyn_col))
        status = "SUCCESS" if success > 0.5 else "RUNNING"
        status_color = (80, 220, 80) if success > 0.5 else (230, 230, 230)
        _draw_label(frame, f"Status: {status}", (MINIMAP_PAD, 40), status_color, scale=0.8, thickness=2)
        _draw_label(frame, f"Dyn collisions: {dyn_col:.0f}", (MINIMAP_PAD, 72), (255, 255, 255))
        _draw_label(frame, f"Total collisions: {total_col:.0f}", (MINIMAP_PAD, 96), (255, 255, 255))

    legend_y = y0 + 18
    cv2.circle(frame, (x0 + 12, legend_y), 5, (80, 255, 80), -1)
    _draw_label(frame, "Robot", (x0 + 24, legend_y + 5), (255, 255, 255), scale=0.45)
    cv2.circle(frame, (x0 + 12, legend_y + 20), 5, (255, 80, 80), -1)
    _draw_label(frame, "Goal", (x0 + 24, legend_y + 25), (255, 255, 255), scale=0.45)
    cv2.circle(frame, (x0 + 12, legend_y + 40), 5, (0, 255, 255), -1)
    _draw_label(frame, "Start", (x0 + 24, legend_y + 45), (255, 255, 255), scale=0.45)
    cv2.circle(frame, (x0 + 12, legend_y + 60), 5, (0, 140, 255), -1)
    _draw_label(frame, "Dynamic obs", (x0 + 24, legend_y + 65), (255, 255, 255), scale=0.45)

    return frame


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
    latest_episode_info = None

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
                img_bgr = _draw_minimap_overlay(img_bgr, env, episode_count, frame_count, latest_episode_info)
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
                latest_episode_info = episode_info
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
