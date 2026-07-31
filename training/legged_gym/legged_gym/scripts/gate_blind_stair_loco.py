"""Run the strict fixed-command gate for a blind stair locomotion policy."""

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

from legged_gym.envs import *
from legged_gym.envs.base.legged_robot_pos_stairs_minimal import (
    TERMINAL_FALL_OR_CONTACT,
    TERMINAL_NONE,
    TERMINAL_OTHER_STUCK,
    TERMINAL_REASON_NAMES,
    TERMINAL_STAIR_STUCK,
    TERMINAL_SUCCESS,
    TERMINAL_TIMEOUT,
)
from legged_gym.utils import get_args, task_registry


SPEEDS = (0.25, 0.40, 0.55, 0.70)


def gate_required_successes(trials):
    return math.ceil(0.95 * trials)


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--steps", type=int, default=2000)
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
    zeros = torch.zeros(num_envs, device=device)
    yaw = torch.where(is_up, zeros, torch.full_like(zeros, torch.pi))
    env.root_states[:, 3:7] = quat_from_euler_xyz(zeros, zeros, yaw)
    env.base_quat[:] = env.root_states[:, 3:7]
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
    env.terminal_reason.zero_()
    env.reset_buf.zero_()
    env.time_out_buf.zero_()
    commands = torch.zeros(num_envs, 3, device=device)
    commands[:, 0] = speed_tensor
    max_x = env.root_states[:, 0] - env.room_origins[:, 0]
    min_x = max_x.clone()
    max_z = env.root_states[:, 2].clone()
    min_z = max_z.clone()
    finished = torch.zeros(num_envs, dtype=torch.bool, device=device)
    recorded_reason = torch.zeros(num_envs, dtype=torch.long, device=device)
    min_command_vx = torch.full((num_envs,), float("inf"), device=device)
    min_heading_alignment = torch.ones(num_envs, device=device)
    success_base_clearance = torch.full((num_envs,), float("nan"), device=device)
    success_all_feet_clear = torch.zeros(num_envs, dtype=torch.bool, device=device)
    success_hold_steps = torch.zeros_like(env.goal_hold_timer)
    heading_threshold = math.cos(math.radians(float(env.cfg.depth_stairs.heading_tolerance_deg)))
    body_forward = torch.zeros(num_envs, 3, device=device)
    body_forward[:, 0] = 1.0

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
            forward_world = quat_apply(env.root_states[:, 3:7], body_forward)
            heading_alignment = forward_world[:, 0] * env.navigation_direction.float()
            min_heading_alignment = torch.where(
                active,
                torch.minimum(min_heading_alignment, heading_alignment),
                min_heading_alignment,
            )
            actual_command_vx = env.slr_commands[:, 0]
            min_command_vx = torch.where(
                active,
                torch.minimum(min_command_vx, actual_command_vx),
                min_command_vx,
            )

            done_now = env.reset_buf & active
            if torch.any(done_now & (env.terminal_reason == TERMINAL_NONE)):
                raise RuntimeError("Gate observed a done episode without a terminal reason")
            recorded_reason[done_now] = env.terminal_reason[done_now]
            success_now = done_now & (env.terminal_reason == TERMINAL_SUCCESS)
            up_clearance = local_x - float(env.cfg.depth_stairs.platform_start_x)
            down_clearance = float(env.cfg.terrain.stair_start_x) - local_x
            clearance = torch.where(is_up, up_clearance, down_clearance)
            success_base_clearance[success_now] = clearance[success_now]
            success_all_feet_clear[success_now] = env.fully_cleared[success_now]
            success_hold_steps[success_now] = env.goal_hold_timer[success_now]
            finished |= done_now
            commands[finished] = 0.0
            if torch.all(finished):
                break
    if not torch.all(finished):
        unfinished = int((~finished).sum().item())
        raise RuntimeError(
            f"Gate ended with {unfinished} ongoing episodes; increase --steps instead of "
            "treating them as success or timeout"
        )

    if torch.any((recorded_reason == TERMINAL_SUCCESS) & (recorded_reason == TERMINAL_TIMEOUT)):
        raise RuntimeError("Gate recorded an episode as both success and timeout")

    base_clearance_required = float(env.cfg.depth_stairs.stair_clearance_distance)
    for index in range(num_envs):
        reason_code = int(recorded_reason[index].item())
        reason = TERMINAL_REASON_NAMES[reason_code]
        success = reason_code == TERMINAL_SUCCESS
        valid_success = (
            not success
            or (
                float(success_base_clearance[index].item()) >= base_clearance_required
                and bool(success_all_feet_clear[index].item())
                and int(success_hold_steps[index].item()) >= int(env.cfg.env.goal_reached_time)
            )
        )
        rows.append(
            {
                "direction": directions[index],
                "speed": speeds[index],
                "trial": trials[index] + 1,
                "terminal_reason": reason,
                "success": int(success),
                "fall_or_contact": int(reason_code == TERMINAL_FALL_OR_CONTACT),
                "stair_stuck": int(reason_code == TERMINAL_STAIR_STUCK),
                "timeout": int(reason_code == TERMINAL_TIMEOUT),
                "other_stuck": int(reason_code == TERMINAL_OTHER_STUCK),
                "min_command_vx": float(min_command_vx[index].item()),
                "command_vx_positive": int(float(min_command_vx[index].item()) > 0.0),
                "min_heading_alignment": float(min_heading_alignment[index].item()),
                "heading_within_20deg": int(
                    float(min_heading_alignment[index].item()) >= heading_threshold
                ),
                "success_base_clearance_m": (
                    float(success_base_clearance[index].item()) if success else ""
                ),
                "success_all_feet_clear": int(success_all_feet_clear[index].item()),
                "success_hold_steps": int(success_hold_steps[index].item()),
                "success_validation_passes": int(valid_success),
                "success_timeout_overlap": 0,
                "max_local_x": float(max_x[index].item()),
                "min_local_x": float(min_x[index].item()),
                "max_z": float(max_z[index].item()),
                "min_z": float(min_z[index].item()),
            }
        )

    summaries = []
    required_successes = gate_required_successes(gate_args.trials)
    for direction, speed in groups:
        selected = [row for row in rows if row["direction"] == direction and row["speed"] == speed]
        successes = sum(row["success"] for row in selected)
        reason_counts = {
            name: sum(row["terminal_reason"] == name for row in selected)
            for name in TERMINAL_REASON_NAMES.values()
            if name != "ongoing"
        }
        summaries.append(
            {
                "direction": direction,
                "speed": speed,
                "successes": successes,
                "trials": len(selected),
                "success_rate": successes / len(selected),
                "falls_or_contacts": sum(row["fall_or_contact"] for row in selected),
                "stair_stuck": sum(row["stair_stuck"] for row in selected),
                "timeouts": sum(row["timeout"] for row in selected),
                "other_stuck": sum(row["other_stuck"] for row in selected),
                "terminal_reason_counts": reason_counts,
                "command_vx_violations": sum(
                    not row["command_vx_positive"] for row in selected
                ),
                "heading_violations": sum(
                    not row["heading_within_20deg"] for row in selected
                ),
                "success_validation_violations": sum(
                    not row["success_validation_passes"] for row in selected
                ),
                "required_successes": required_successes,
                "passes": successes >= required_successes
                and not any(
                    row["fall_or_contact"]
                    or row["stair_stuck"]
                    or not row["command_vx_positive"]
                    or not row["heading_within_20deg"]
                    or not row["success_validation_passes"]
                    or row["success_timeout_overlap"]
                    for row in selected
                ),
            }
        )
    metadata = env.blind_stair_policy.metadata
    summary = {
        "schema_version": 2,
        "policy_model": os.path.abspath(env.blind_stair_policy.model_path),
        "policy_metadata": os.path.abspath(env.blind_stair_policy.metadata_path),
        "model_sha256": metadata["model_sha256"],
        "checkpoint_path": metadata["checkpoint_path"],
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "trials_per_group": gate_args.trials,
        "required_successes_per_group": required_successes,
        "heading_tolerance_deg": float(env.cfg.depth_stairs.heading_tolerance_deg),
        "base_clearance_m": base_clearance_required,
        "hold_steps": int(env.cfg.env.goal_reached_time),
        "groups": summaries,
        "failure_reason_totals": {
            name: sum(row["terminal_reason"] == name for row in rows)
            for name in TERMINAL_REASON_NAMES.values()
            if name != "ongoing"
        },
        "passes_gate": all(group["passes"] for group in summaries),
    }
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
