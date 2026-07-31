"""Test the frozen locomotion executor with fixed commands in the target room."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

import isaacgym
import torch
from isaacgym.torch_utils import quat_apply, quat_from_euler_xyz

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--speed", type=float, default=0.40)
    parser.add_argument("--max_steps", type=int, default=800)
    parser.add_argument("--hold_steps", type=int, default=12)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_json", required=True)
    known, remaining = parser.parse_known_args()
    if known.speed <= 0.0:
        raise ValueError("--speed must be positive in the robot body frame")
    sys.argv = [sys.argv[0]] + remaining
    return known, get_args()


def configure_environment(env_cfg):
    env_cfg.env.num_envs = 2
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.depth_stairs.fixed_direction = 0
    env_cfg.depth_stairs.eval_seed_base = -1
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_xy = False
    env_cfg.domain_rand.randomize_yaw = False
    env_cfg.domain_rand.randomize_roll = False
    env_cfg.domain_rand.randomize_pitch = False
    env_cfg.domain_rand.push_robots = False


def place_for_crossing(env):
    ids = torch.arange(2, device=env.device)
    env.root_states[ids] = env.base_init_state
    local_x = torch.tensor([3.90, 7.20], device=env.device)
    local_y = torch.full((2,), 5.00, device=env.device)
    surface_z = torch.tensor(
        [0.0, float(env.cfg.depth_stairs.platform_height)], device=env.device
    )
    yaw = torch.tensor([0.0, math.pi], device=env.device)
    zeros = torch.zeros(2, device=env.device)
    env.root_states[ids, 0] = env.room_origins[ids, 0] + local_x
    env.root_states[ids, 1] = env.room_origins[ids, 1] + local_y
    env.root_states[ids, 2] = float(env.base_init_state[2]) + surface_z
    env.root_states[ids, 3:7] = quat_from_euler_xyz(zeros, zeros, yaw)
    env.root_states[ids, 7:13] = 0.0
    env.base_quat[ids] = env.root_states[ids, 3:7]
    env.navigation_direction[ids] = torch.tensor([1, -1], device=env.device)
    env.position_targets[0] = torch.tensor(
        [
            env.room_origins[0, 0] + 7.30,
            env.room_origins[0, 1] + 5.00,
            float(env.base_init_state[2]) + float(env.cfg.depth_stairs.platform_height),
        ],
        device=env.device,
    )
    env.position_targets[1] = torch.tensor(
        [
            env.room_origins[1, 0] + 3.80,
            env.room_origins[1, 1] + 5.00,
            float(env.base_init_state[2]),
        ],
        device=env.device,
    )
    env._reset_dofs(ids)
    env._set_robot_root_states(ids)
    env.nav_actions_filtered.zero_()
    env.nav_actions_orig.zero_()
    env.episode_length_buf.zero_()
    env.low_obstacle_collision_count.zero_()
    env.wall_contact_count.zero_()
    env.stair_contact_count.zero_()
    env.stair_crossed.zero_()
    env.fully_cleared.zero_()
    env.terminal_reason.zero_()
    env.compute_observations()


def run_probe(env, speed, max_steps, hold_steps):
    env.reset()
    env.do_reset = False
    place_for_crossing(env)
    commands = torch.zeros(2, 3, device=env.device)
    commands[:, 0] = speed
    signs = torch.tensor([1.0, -1.0], device=env.device)
    target_surface = torch.tensor(
        [float(env.cfg.depth_stairs.platform_height), 0.0], device=env.device
    )
    hold = torch.zeros(2, dtype=torch.long, device=env.device)
    done = torch.zeros(2, dtype=torch.bool, device=env.device)
    outcome = ["max_steps", "max_steps"]
    first_clear_step = [-1, -1]
    terminal_step = [max_steps - 1, max_steps - 1]
    min_heading = torch.ones(2, device=env.device)
    max_directed_progress = torch.full((2,), -torch.inf, device=env.device)

    stair_start = float(env.cfg.terrain.stair_start_x)
    stair_end = float(env.cfg.depth_stairs.platform_start_x)
    base_clearance = float(env.cfg.depth_stairs.stair_clearance_distance)
    foot_margin = float(env.cfg.depth_stairs.foot_clearance_margin)
    heading_limit = math.cos(math.radians(float(env.cfg.depth_stairs.heading_tolerance_deg)))
    foot_height_tolerance = float(env.cfg.depth_stairs.foot_height_tolerance)

    with torch.no_grad():
        for step in range(max_steps):
            env.step(commands)
            local_x = env.root_states[:, 0] - env.room_origins[:, 0]
            feet_local_x = env.feet_pos[:, :, 0] - env.room_origins[:, None, 0]
            forward_body = torch.tensor(
                [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], device=env.device
            )
            forward_world = quat_apply(env.root_states[:, 3:7], forward_body)
            heading = signs * forward_world[:, 0]
            min_heading = torch.minimum(min_heading, heading)

            up_clear = (local_x >= stair_end + base_clearance) & torch.all(
                feet_local_x >= stair_end + foot_margin, dim=1
            )
            down_clear = (local_x <= stair_start - base_clearance) & torch.all(
                feet_local_x <= stair_start - foot_margin, dim=1
            )
            geometry_clear = torch.stack((up_clear[0], down_clear[1]))
            foot_surface_error = torch.abs(
                env.feet_pos[:, :, 2] - target_surface[:, None]
            )
            feet_on_target = torch.all(
                foot_surface_error <= foot_height_tolerance, dim=1
            )
            base_target_z = float(env.base_init_state[2]) + target_surface
            base_on_target = torch.abs(env.root_states[:, 2] - base_target_z) <= float(
                env.cfg.depth_stairs.height_tolerance
            )
            clear_now = geometry_clear & feet_on_target & base_on_target & (
                heading >= heading_limit
            )
            hold = torch.where(clear_now, hold + 1, torch.zeros_like(hold))
            directed_progress = signs * (local_x - torch.tensor([3.90, 7.20], device=env.device))
            max_directed_progress = torch.maximum(max_directed_progress, directed_progress)

            hard_failure = (
                env.reset_fall
                | env.reset_terminate_contact
                | (env.low_obstacle_collision_count > 0)
                | (env.wall_contact_count > 0)
            )
            for index in range(2):
                if done[index]:
                    continue
                if hard_failure[index]:
                    outcome[index] = "fall_or_contact"
                    terminal_step[index] = step
                    done[index] = True
                elif hold[index] >= hold_steps:
                    outcome[index] = "fully_cleared"
                    first_clear_step[index] = step
                    terminal_step[index] = step
                    done[index] = True
            commands[done] = 0.0
            if torch.all(done):
                break

    rows = []
    local_x = env.root_states[:, 0] - env.room_origins[:, 0]
    for index, direction in enumerate(("up", "down")):
        frames = terminal_step[index] + 1
        rows.append(
            {
                "direction": direction,
                "command_vx_body_mps": speed,
                "command_vy_body_mps": 0.0,
                "command_yaw_rate_rps": 0.0,
                "outcome": outcome[index],
                "fully_cleared": int(outcome[index] == "fully_cleared"),
                "duration_s": frames * float(env.dt),
                "final_local_x": float(local_x[index].item()),
                "final_base_z": float(env.root_states[index, 2].item()),
                "max_directed_progress": float(max_directed_progress[index].item()),
                "min_heading_alignment": float(min_heading[index].item()),
                "low_obstacle_collision_count": int(
                    env.low_obstacle_collision_count[index].item()
                ),
                "wall_contact_count": int(env.wall_contact_count[index].item()),
                "stair_contact_count": int(env.stair_contact_count[index].item()),
            }
        )
    return rows


def main():
    probe_args, args = parse_args()
    if args.task != "go2_pos_stairs_minimal":
        raise ValueError("Use --task go2_pos_stairs_minimal")
    args.headless = True
    env_cfg, _ = task_registry.get_cfgs(args.task)
    configure_environment(env_cfg)
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=env_cfg)
    rows = run_probe(env, probe_args.speed, probe_args.max_steps, probe_args.hold_steps)

    for path in (probe_args.output_csv, probe_args.output_json):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(probe_args.output_csv, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "task": args.task,
        "locomotion_model": os.path.abspath(
            str(env.cfg.loco.model_path).format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        ),
        "speed_mps": probe_args.speed,
        "passed": all(row["fully_cleared"] for row in rows),
        "results": rows,
    }
    with open(probe_args.output_json, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise RuntimeError("Frozen locomotion did not fully clear both stair directions")


if __name__ == "__main__":
    main()
