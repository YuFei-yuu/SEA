"""Measure frozen low-level tracking for simultaneous vx/vy/yaw commands."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

import isaacgym
import torch
from isaacgym.torch_utils import quat_from_euler_xyz

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


COMMAND_CASES = (
    ("diagonal_left", 0.30, 0.20, 0.00),
    ("diagonal_right", 0.30, -0.20, 0.00),
    ("curve_left", 0.30, 0.18, 0.30),
    ("curve_right", 0.30, -0.18, -0.30),
    ("crab_turn_left", 0.18, 0.25, 0.25),
    ("crab_turn_right", 0.18, -0.25, -0.25),
)


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--warmup_steps", type=int, default=75)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_json", required=True)
    known, remaining = parser.parse_known_args()
    if known.steps <= known.warmup_steps or known.warmup_steps < 0:
        raise ValueError("Require --steps > --warmup_steps >= 0")
    sys.argv = [sys.argv[0]] + remaining
    return known, get_args()


def configure_environment(env_cfg):
    env_cfg.env.num_envs = len(COMMAND_CASES)
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.depth_stairs.fixed_direction = 1
    env_cfg.depth_stairs.eval_seed_base = -1
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_xy = False
    env_cfg.domain_rand.randomize_yaw = False
    env_cfg.domain_rand.randomize_roll = False
    env_cfg.domain_rand.randomize_pitch = False
    env_cfg.domain_rand.push_robots = False


def place_on_open_platform(env):
    count = len(COMMAND_CASES)
    ids = torch.arange(count, device=env.device)
    env.root_states[ids] = env.base_init_state
    env.root_states[ids, 0] = env.room_origins[ids, 0] + 7.25
    env.root_states[ids, 1] = env.room_origins[ids, 1] + 5.00
    env.root_states[ids, 2] = (
        float(env.base_init_state[2]) + float(env.cfg.depth_stairs.platform_height)
    )
    zeros = torch.zeros(count, device=env.device)
    env.root_states[ids, 3:7] = quat_from_euler_xyz(zeros, zeros, zeros)
    env.root_states[ids, 7:13] = 0.0
    env.base_quat[ids] = env.root_states[ids, 3:7]
    env.position_targets[ids, 0] = env.room_origins[ids, 0] + 9.25
    env.position_targets[ids, 1] = env.room_origins[ids, 1] + 5.00
    env.position_targets[ids, 2] = env.root_states[ids, 2]
    env._reset_dofs(ids)
    env._set_robot_root_states(ids)
    env.nav_actions_filtered.zero_()
    env.nav_actions_orig.zero_()
    env.episode_length_buf.zero_()
    env.low_obstacle_collision_count.zero_()
    env.wall_contact_count.zero_()
    env.terminal_reason.zero_()
    env.compute_observations()


def signed_response_ok(command, measured, threshold=0.06):
    if abs(command) < 1.0e-6:
        return True
    return math.copysign(1.0, measured) == math.copysign(1.0, command) and abs(measured) >= threshold


def run_probe(env, steps, warmup_steps):
    env.reset()
    env.do_reset = False
    place_on_open_platform(env)
    commands = torch.tensor(
        [case[1:] for case in COMMAND_CASES], dtype=torch.float32, device=env.device
    )
    samples = []
    max_gravity_z = torch.full((len(COMMAND_CASES),), -1.0, device=env.device)
    max_roll_pitch_rate = torch.zeros(len(COMMAND_CASES), device=env.device)
    with torch.no_grad():
        for step in range(steps):
            env.step(commands)
            max_gravity_z = torch.maximum(max_gravity_z, env.projected_gravity[:, 2])
            max_roll_pitch_rate = torch.maximum(
                max_roll_pitch_rate, torch.linalg.vector_norm(env.base_ang_vel[:, :2], dim=-1)
            )
            if step >= warmup_steps:
                samples.append(
                    torch.cat((env.base_lin_vel[:, :2], env.base_ang_vel[:, 2:3]), dim=-1).clone()
                )
    measured = torch.stack(samples).mean(dim=0)
    mae = torch.stack(samples).sub(commands.unsqueeze(0)).abs().mean(dim=0)
    rows = []
    for index, (name, vx, vy, yaw_rate) in enumerate(COMMAND_CASES):
        actual = measured[index].detach().cpu().tolist()
        errors = mae[index].detach().cpu().tolist()
        vx_ok = signed_response_ok(vx, actual[0])
        vy_ok = signed_response_ok(vy, actual[1])
        yaw_ok = signed_response_ok(yaw_rate, actual[2])
        stable = (
            float(max_gravity_z[index].item()) < -0.80
            and float(max_roll_pitch_rate[index].item()) < 3.0
            and int(env.low_obstacle_collision_count[index].item()) == 0
            and int(env.wall_contact_count[index].item()) == 0
        )
        rows.append(
            {
                "case": name,
                "command_vx": vx,
                "command_vy": vy,
                "command_yaw_rate": yaw_rate,
                "mean_vx": actual[0],
                "mean_vy": actual[1],
                "mean_yaw_rate": actual[2],
                "mae_vx": errors[0],
                "mae_vy": errors[1],
                "mae_yaw_rate": errors[2],
                "vx_response_ok": int(vx_ok),
                "vy_response_ok": int(vy_ok),
                "yaw_response_ok": int(yaw_ok),
                "stable": int(stable),
                "low_obstacle_collision_count": int(env.low_obstacle_collision_count[index].item()),
                "wall_contact_count": int(env.wall_contact_count[index].item()),
                "max_projected_gravity_z": float(max_gravity_z[index].item()),
                "max_roll_pitch_rate": float(max_roll_pitch_rate[index].item()),
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
    rows = run_probe(env, probe_args.steps, probe_args.warmup_steps)
    passed = all(
        row["vx_response_ok"]
        and row["vy_response_ok"]
        and row["yaw_response_ok"]
        and row["stable"]
        for row in rows
    )
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
        "steps": probe_args.steps,
        "warmup_steps": probe_args.warmup_steps,
        "passed": passed,
        "cases": rows,
    }
    with open(probe_args.output_json, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2))
    if not passed:
        raise RuntimeError("Frozen locomotion failed the simultaneous vx/vy/yaw probe")


if __name__ == "__main__":
    main()
