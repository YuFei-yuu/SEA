"""Record a deterministic passability-Teacher bypass demonstration.

This recorder deliberately executes the same closed-loop Teacher used by the
diagnostics.  It is a visual artifact for inspecting the sparse 15-obstacle
scene, not a learned-policy success claim.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np
from isaacgym import gymapi
from isaacgym.torch_utils import quat_rotate_inverse
import torch

from legged_gym.envs import *
from legged_gym.envs.base.legged_robot_pos_stairs_minimal import (
    TERMINAL_REASON_NAMES,
)
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.torch_math import yaw_quat


def _parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--direction", choices=("up", "down", "both"), default="both")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=390000)
    parser.add_argument("--video_goal_y", type=float, default=None, help="Optional fixed local goal y for a visually shallow recording route.")
    parser.add_argument("--video_start_y", type=float, default=None, help="Optional fixed local start y for a non-corner recording route.")
    parser.add_argument("--preset_route", action="store_true", help="Use a fixed shallow-y route for a visual bypass demonstration.")
    parser.add_argument("--max_steps", type=int, default=1800)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return known, get_args()


def _world_to_minimap(point_xy, room_origin, room_size, origin, size):
    local = np.clip(np.asarray(point_xy) - room_origin, 0.0, room_size)
    px = int(round(origin[0] + local[0] / room_size * size))
    py = int(round(origin[1] + size - local[1] / room_size * size))
    return px, py


def _overlay(frame, env, room_origin, direction, step, terminal_reason=None):
    height, width = frame.shape[:2]
    room_size = float(env.terrain.env_width)
    mini_size = 230
    pad = 18
    x0, y0 = width - mini_size - pad, pad
    cv2.rectangle(
        frame,
        (x0 - 10, y0 - 10),
        (x0 + mini_size + 10, y0 + mini_size + 85),
        (18, 18, 18),
        -1,
    )
    cv2.rectangle(frame, (x0, y0), (x0 + mini_size, y0 + mini_size), (220, 220, 220), 2)
    for box in env.cfg.terrain.low_obstacle_boxes:
        center_x, center_y, size_x, size_y, _ = [float(value) for value in box]
        p0 = _world_to_minimap(
            [room_origin[0] + center_x - size_x * 0.5, room_origin[1] + center_y - size_y * 0.5],
            room_origin,
            room_size,
            (x0, y0),
            mini_size,
        )
        p1 = _world_to_minimap(
            [room_origin[0] + center_x + size_x * 0.5, room_origin[1] + center_y + size_y * 0.5],
            room_origin,
            room_size,
            (x0, y0),
            mini_size,
        )
        cv2.rectangle(
            frame,
            (min(p0[0], p1[0]), min(p0[1], p1[1])),
            (max(p0[0], p1[0]), max(p0[1], p1[1])),
            (115, 115, 115),
            -1,
        )

    robot = env.root_states[0, :2].detach().cpu().numpy()
    goal = env.position_targets[0, :2].detach().cpu().numpy()
    start = env.env_origins[0, :2].detach().cpu().numpy()
    for point, color, radius in ((start, (0, 220, 220), 5), (goal, (70, 90, 255), 6), (robot, (70, 230, 70), 6)):
        cv2.circle(frame, _world_to_minimap(point, room_origin, room_size, (x0, y0), mini_size), radius, color, -1)
    cv2.line(
        frame,
        _world_to_minimap(start, room_origin, room_size, (x0, y0), mini_size),
        _world_to_minimap(goal, room_origin, room_size, (x0, y0), mini_size),
        (160, 160, 160),
        1,
    )

    local_x = float((env.root_states[0, 0] - env.room_origins[0, 0]).item())
    local_y = float((env.root_states[0, 1] - env.room_origins[0, 1]).item())
    passed = int(env.obstacle_field_crossed[0].item())
    stair = int(env.stair_crossed[0].item())
    cleared = int(env.fully_cleared[0].item())
    lines = (
        f"Teacher bypass | {direction} | step {step}",
        f"local (x,y)=({local_x:.2f}, {local_y:.2f})",
        f"obstacles={passed}  stair={stair}  clear={cleared}",
    )
    if terminal_reason is not None:
        lines = lines + (f"terminal={terminal_reason}",)
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (18, 32 + 28 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return frame


def _preset_route_actions(env, direction, route_index, goal_y):
    """Execute a fixed, shallow-y visual route without changing training Teacher logic."""
    start_y = float(env.episode_start_local_y[0].item())
    if direction == "up":
        points = (
            # Map-aware shallow bypass: advance at the initial y first, then
            # use two small lateral steps around the x=2 obstacle pair.
            (1.00, start_y),
            (1.40, 7.00),
            (1.70, 7.30),
            (2.10, 7.40),
            (2.40, 7.70),
            (4.20, 7.70),
            (4.40, 7.00),
            (4.80, 7.00),
            (7.10, goal_y),
        )
    else:
        if start_y < 7.0:
            # Low-start route: use two small, map-aligned lateral bypasses in
            # front of the x=3.85 and x=3.25 obstacle pair, hold the lower
            # corridor through the remaining zig-zag, and rise only after the
            # x=1.45 obstacle column is cleared.
            points = (
                (6.90, start_y),
                (4.40, start_y),
                (4.18, 5.30),
                (3.75, 5.30),
                (3.45, 5.00),
                (2.75, 5.00),
                (2.15, 5.05),
                (1.55, 5.05),
                (1.00, 5.10),
                (0.74, 5.10),
                (0.70, goal_y),
            )
        else:
            points = (
                (6.90, start_y),
                (6.70, 8.20),
                (4.20, 8.20),
                (3.50, 8.20),
                (2.90, 9.00),
                (1.80, 9.00),
                (1.10, goal_y),
                (0.85, goal_y),
            )
    index = min(route_index, len(points) - 1)
    local_xy = env.root_states[:, :2] - env.room_origins
    waypoint = torch.as_tensor(points[index], dtype=local_xy.dtype, device=local_xy.device)
    distance = torch.linalg.vector_norm(waypoint[None, :] - local_xy, dim=-1)
    if index < len(points) - 1 and bool((distance[0] < 0.20).item()):
        index += 1
        waypoint = torch.as_tensor(points[index], dtype=local_xy.dtype, device=local_xy.device)
    delta = waypoint[None, :] - local_xy
    norm = torch.linalg.vector_norm(delta, dim=-1).clamp(min=1.0e-5)
    path_direction = delta / norm[:, None]
    base_speed = 0.22 if direction == "down" and start_y < 7.0 else 0.22
    speed = torch.full((env.num_envs,), base_speed, device=env.device)
    lateral_speed = (
        0.26 if direction == "down" and start_y < 7.0
        else (0.24 if direction == "up" else 0.45)
    )
    speed = torch.where(
        torch.abs(path_direction[:, 1]) > 0.12,
        torch.full_like(speed, lateral_speed),
        speed,
    )
    local_x = local_xy[:, 0]
    stair_run = (local_x >= 4.50) & (local_x <= 6.60)
    speed = torch.where(stair_run, torch.full_like(speed, 0.22), speed)
    desired_world = path_direction * speed[:, None]
    body = quat_rotate_inverse(
        yaw_quat(env.base_quat),
        torch.cat((desired_world, torch.zeros(env.num_envs, 1, device=env.device)), dim=-1),
    )[:, :2]
    body[:, 0].clamp_(min=-0.20, max=0.55)
    body[:, 1].clamp_(min=-0.55, max=0.55)
    low_start_stair = direction == "down" and start_y < 7.0
    if low_start_stair:
        body[:, 0] = torch.where(
            stair_run, torch.full_like(body[:, 0], 0.32), body[:, 0]
        )
    body[:, 1] = torch.where(stair_run, torch.zeros_like(body[:, 1]), body[:, 1])
    teacher_action = env.get_navigation_teacher_actions()
    env.teacher_waypoint[:] = waypoint
    action = torch.cat((body, teacher_action[:, 2:3]), dim=-1)
    return action, index

def _record_direction(env, camera, camera_props, direction, seed, output_dir, record_args):
    env.cfg.depth_stairs.fixed_direction = 1 if direction == "up" else -1
    env.cfg.depth_stairs.eval_seed_base = int(seed)
    env.eval_seed_round.zero_()
    env.do_reset = True
    env.reset()
    env.do_reset = False

    room_origin = env.room_origins[0].detach().cpu().numpy().copy()
    output_video = os.path.join(output_dir, f"teacher_bypass_{direction}.mp4")
    output_csv = os.path.join(output_dir, f"teacher_bypass_{direction}.csv")
    video = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        record_args.fps,
        (record_args.width, record_args.height),
    )
    if not video.isOpened():
        raise RuntimeError(f"Could not open video output {output_video}")

    trajectory = []
    terminal_reason = "timeout"
    route_index = 0
    goal_y = float((env.position_targets[0, 1] - env.room_origins[0, 1]).item())
    for step in range(record_args.max_steps):
        eye = gymapi.Vec3(
            float(room_origin[0] + 3.8),
            float(room_origin[1] + 12.0),
            8.2,
        )
        target = gymapi.Vec3(
            float(room_origin[0] + 4.0),
            float(room_origin[1] + 5.0),
            0.15,
        )
        env.gym.set_camera_location(camera, env.envs[0], eye, target)
        with torch.no_grad():
            if record_args.preset_route:
                action, route_index = _preset_route_actions(
                    env, direction, route_index, goal_y
                )
            else:
                action = env.get_navigation_teacher_actions()
            action = torch.nan_to_num(action, nan=0.0, posinf=3.0, neginf=-3.0).clamp(-3.0, 3.0)
            _, _, _, dones, infos = env.step(action)
        local = (env.root_states[0, :2] - env.room_origins[0]).detach().cpu().tolist()
        trajectory.append(
            {
                "step": step,
                "x": local[0],
                "y": local[1],
                "vx": float(action[0, 0].item()),
                "vy": float(action[0, 1].item()),
                "yaw_rate": float(action[0, 2].item()),
            }
        )
        env.gym.step_graphics(env.sim)
        env.gym.render_all_camera_sensors(env.sim)
        rgba = env.gym.get_camera_image(
            env.sim, env.envs[0], camera, gymapi.IMAGE_COLOR
        ).reshape(record_args.height, record_args.width, 4)
        frame = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
        frame = _overlay(frame, env, room_origin, direction, step)
        video.write(frame)

        if bool(dones[0].item()):
            # Automatic reset is disabled.  Read the terminal code while the
            # completed episode is still resident, then finish this video.
            terminal_reason = TERMINAL_REASON_NAMES[int(env.terminal_reason[0].item())]
            frame = _overlay(frame, env, room_origin, direction, step, terminal_reason)
            # The last frame with the terminal label is written after the
            # outcome is available; it keeps the video self-describing.
            video.write(frame)
            break

    video.release()
    with open(output_csv, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("step", "x", "y", "vx", "vy", "yaw_rate"),
        )
        writer.writeheader()
        writer.writerows(trajectory)
    return {
        "direction": direction,
        "seed": int(seed),
        "terminal_reason": terminal_reason,
        "video": output_video,
        "trajectory": output_csv,
        "frames": len(trajectory),
    }


def main():
    record_args, args = _parse_args()
    if args.task != "go2_pos_depth_stairs_passability":
        raise ValueError("Use --task go2_pos_depth_stairs_passability.")

    env_cfg, _ = task_registry.get_cfgs(args.task)
    env_cfg.env.num_envs = 1
    env_cfg.env.disable_graphics = False
    env_cfg.env.debug_viz = False
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.depth_stairs.strict_terminal_rules = True
    env_cfg.depth_stairs.enable_stand_still_reset = True
    if record_args.preset_route:
        # Video-only acceptance keeps the robot visually close to the requested
        # endpoint instead of stopping at the default 0.5 m goal radius.
        env_cfg.rewards.reach_pos_target_tight_config.distance_threshold = 0.25
    if record_args.video_goal_y is not None:
        env_cfg.depth_stairs.goal_y_range = [float(record_args.video_goal_y), float(record_args.video_goal_y)]
    if record_args.video_start_y is not None:
        env_cfg.depth_stairs.start_y_range = [float(record_args.video_start_y), float(record_args.video_start_y)]
    env_cfg.perception.mode = "oracle"
    env_cfg.perception.render_depth_in_oracle = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.asset.terminate_after_contacts_on = []
    args.headless = True

    os.makedirs(record_args.output_dir, exist_ok=True)
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=env_cfg)
    camera_props = gymapi.CameraProperties()
    camera_props.width = record_args.width
    camera_props.height = record_args.height
    camera = env.gym.create_camera_sensor(env.envs[0], camera_props)

    directions = ("up", "down") if record_args.direction == "both" else (record_args.direction,)
    summaries = []
    for index, direction in enumerate(directions):
        summaries.append(
            _record_direction(
                env,
                camera,
                camera_props,
                direction,
                record_args.seed + index * 10000,
                record_args.output_dir,
                record_args,
            )
        )
        print(json.dumps(summaries[-1], ensure_ascii=False))
    summary_path = os.path.join(record_args.output_dir, "teacher_bypass_summary.json")
    with open(summary_path, "w", encoding="utf-8") as stream:
        json.dump(summaries, stream, indent=2, ensure_ascii=False)
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
