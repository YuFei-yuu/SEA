"""Run the fixed-command 90% gate for a blind stair locomotion policy."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

import isaacgym
import torch

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


SPEEDS = (0.25, 0.40, 0.55, 0.70)


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_summary", required=True)
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return known, get_args()


def main():
    gate_args, args = parse_args()
    if args.task != "go2_pos_stairs_minimal":
        raise ValueError("Use --task go2_pos_stairs_minimal")
    groups = [(direction, speed) for direction in ("up", "down") for speed in SPEEDS]
    num_envs = len(groups) * gate_args.trials
    env_cfg, _ = task_registry.get_cfgs(args.task)
    env_cfg.env.num_envs = num_envs
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.asset.terminate_after_contacts_on = []
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=env_cfg)
    env.reset()
    env.do_reset = False

    rows = []
    directions = []
    speeds = []
    trials = []
    for direction, speed in groups:
        for trial in range(gate_args.trials):
            directions.append(direction)
            speeds.append(speed)
            trials.append(trial)
    device = env.device
    is_up = torch.tensor([value == "up" for value in directions], device=device)
    speed_tensor = torch.tensor(speeds, device=device)
    ids = torch.arange(num_envs, device=device)
    env.root_states[:] = env.base_init_state
    env.root_states[:, 0] = env.room_origins[:, 0] + torch.where(
        is_up, torch.full_like(speed_tensor, 3.50), torch.full_like(speed_tensor, 7.20)
    )
    env.root_states[:, 1] = env.room_origins[:, 1] + 5.0
    env.root_states[:, 2] = env.base_init_state[2] + torch.where(
        is_up, torch.zeros_like(speed_tensor), torch.full_like(speed_tensor, 0.40)
    )
    env.root_states[:, 7:13] = 0.0
    env.navigation_direction[:] = torch.where(
        is_up,
        torch.ones(num_envs, dtype=torch.long, device=device),
        -torch.ones(num_envs, dtype=torch.long, device=device),
    )
    env.position_targets[:, 0] = env.room_origins[:, 0] + torch.where(
        is_up, torch.full_like(speed_tensor, 7.20), torch.full_like(speed_tensor, 3.50)
    )
    env.position_targets[:, 1] = env.room_origins[:, 1] + 5.0
    env.position_targets[:, 2] = env.base_init_state[2] + torch.where(
        is_up, torch.full_like(speed_tensor, 0.40), torch.zeros_like(speed_tensor)
    )
    env._set_robot_root_states(ids)
    env._reset_dofs(ids)
    env.nav_actions_filtered.zero_()
    env.actions_orig.zero_()
    env.stair_crossed.zero_()
    env.stair_progress_anchor.copy_(env.root_states[:, 0])
    env.stair_progress_steps.zero_()
    env.goal_hold_timer.zero_()
    env.stay_timer.zero_()
    env.episode_length_buf.zero_()
    commands = torch.zeros(num_envs, 3, device=device)
    commands[:, 0] = torch.where(is_up, speed_tensor, -speed_tensor)
    max_x = env.root_states[:, 0] - env.room_origins[:, 0]
    min_x = max_x.clone()
    max_z = env.root_states[:, 2].clone()
    min_z = max_z.clone()
    height_tolerance = float(env.cfg.depth_stairs.height_tolerance)
    fall = torch.zeros(num_envs, dtype=torch.bool, device=device)
    stuck = torch.zeros_like(fall)
    finished = torch.zeros_like(fall)
    success = torch.zeros_like(fall)
    crossing_hold = torch.zeros(num_envs, dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(gate_args.steps):
            env.step(commands)
            local_x = env.root_states[:, 0] - env.room_origins[:, 0]
            active = ~finished
            max_x = torch.where(active, torch.maximum(max_x, local_x), max_x)
            min_x = torch.where(active, torch.minimum(min_x, local_x), min_x)
            max_z = torch.where(
                active, torch.maximum(max_z, env.root_states[:, 2]), max_z
            )
            min_z = torch.where(
                active, torch.minimum(min_z, env.root_states[:, 2]), min_z
            )
            fall_now = env.fall_down & active
            stuck_now = env.stand_still_flag & active
            fall |= fall_now
            stuck |= stuck_now

            up_crossed = (local_x >= 6.30) & (
                torch.abs(env.root_states[:, 2] - (env.base_init_state[2] + 0.40))
                <= height_tolerance
            )
            down_crossed = (local_x <= 4.50) & (
                torch.abs(env.root_states[:, 2] - env.base_init_state[2])
                <= height_tolerance
            )
            crossed = torch.where(is_up, up_crossed, down_crossed) & active
            crossing_hold = torch.where(
                crossed, crossing_hold + 1, torch.zeros_like(crossing_hold)
            )
            success_now = (crossing_hold >= 12) & active & ~fall_now & ~stuck_now
            failed_now = (fall_now | stuck_now) & active
            success |= success_now
            finished |= success_now | failed_now
            commands[finished] = 0.0
            if torch.all(finished):
                break
    for index in range(num_envs):
        rows.append(
            {
                "direction": directions[index],
                "speed": speeds[index],
                "trial": trials[index],
                "success": int(success[index].item()),
                "fall": int(fall[index].item()),
                "stuck": int(stuck[index].item()),
                "max_local_x": float(max_x[index].item()),
                "min_local_x": float(min_x[index].item()),
                "max_z": float(max_z[index].item()),
                "min_z": float(min_z[index].item()),
            }
        )

    summaries = []
    required_successes = math.ceil(0.9 * gate_args.trials)
    for direction, speed in groups:
        selected = [row for row in rows if row["direction"] == direction and row["speed"] == speed]
        successes = sum(row["success"] for row in selected)
        summaries.append(
            {
                "direction": direction,
                "speed": speed,
                "successes": successes,
                "trials": len(selected),
                "success_rate": successes / len(selected),
                "falls": sum(row["fall"] for row in selected),
                "stuck": sum(row["stuck"] for row in selected),
                "required_successes": required_successes,
                "passes": successes >= required_successes
                and not any(row["fall"] or row["stuck"] for row in selected),
            }
        )
    summary = {"groups": summaries, "passes_gate": all(group["passes"] for group in summaries)}
    os.makedirs(os.path.dirname(os.path.abspath(gate_args.output_csv)), exist_ok=True)
    with open(gate_args.output_csv, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.makedirs(os.path.dirname(os.path.abspath(gate_args.output_summary)), exist_ok=True)
    with open(gate_args.output_summary, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
