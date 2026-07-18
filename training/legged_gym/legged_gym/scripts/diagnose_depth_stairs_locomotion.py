"""Probe the frozen locomotion controller on the depth-stair terrain.

The navigation policy is bypassed. Fixed velocity commands are passed through
the same SLR controller used by the task, so a failed ascent is evidence about
low-level mobility rather than high-level navigation learning.
"""
from __future__ import annotations

import argparse
import csv
import os

from isaacgym import gymapi
import torch

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


def _parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--steps", type=int, default=1500)
    known, remaining = parser.parse_known_args()
    import sys

    sys.argv = [sys.argv[0]] + remaining
    return known, get_args()


def _prepare_env(env_cfg, num_envs):
    env_cfg.env.num_envs = num_envs
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_xy = False
    env_cfg.domain_rand.randomize_yaw = False
    env_cfg.domain_rand.randomize_roll = False
    env_cfg.domain_rand.randomize_pitch = False
    env_cfg.asset.terminate_after_contacts_on = []
    env_cfg.depth_stairs.start_x_range = [0.90, 0.90]
    env_cfg.depth_stairs.start_y_range = [5.00, 5.00]
    env_cfg.depth_stairs.goal_x_range = [7.80, 7.80]
    env_cfg.depth_stairs.goal_y_range = [5.00, 5.00]


def _place_robot(env, local_x, local_y, height):
    ids = torch.arange(env.num_envs, device=env.device)
    env.root_states[:] = env.base_init_state
    env.root_states[:, 0] = env.room_origins[:, 0] + local_x
    env.root_states[:, 1] = env.room_origins[:, 1] + local_y
    env.root_states[:, 2] = env.base_init_state[2] + height
    env.root_states[:, 7:13] = 0.0
    env._set_robot_root_states(ids)
    env._reset_dofs(ids)
    env.nav_actions_filtered[:] = 0.0
    env.compute_observations()


def _run_ascent(env, speeds, steps):
    obs, _ = env.reset()
    env.do_reset = False
    _place_robot(env, local_x=0.90, local_y=5.00, height=0.0)
    device = env.device
    commands = torch.zeros(len(speeds), 3, device=device)
    commands[:, 0] = torch.tensor(speeds, device=device)
    start_x = (env.root_states[:, 0] - env.room_origins[:, 0]).clone()
    max_x = start_x.clone()
    max_z = env.root_states[:, 2].clone()
    fell = torch.zeros(len(speeds), dtype=torch.bool, device=device)
    fall_down = torch.zeros(len(speeds), dtype=torch.bool, device=device)
    stuck = torch.zeros(len(speeds), dtype=torch.bool, device=device)

    with torch.no_grad():
        for _ in range(steps):
            _, _, _, dones, _ = env.step(commands)
            local_x = env.root_states[:, 0] - env.room_origins[:, 0]
            max_x = torch.maximum(max_x, local_x)
            max_z = torch.maximum(max_z, env.root_states[:, 2])
            fell |= dones
            fall_down |= env.reset_fall
            stuck |= env.reset_stand_still

    platform_x = env.cfg.depth_stairs.platform_start_x
    platform_z = env.base_init_state[2] + env.cfg.depth_stairs.platform_height - 0.12
    reached_platform = (max_x >= platform_x) & (max_z >= platform_z)
    return [
        {
            "direction": "up",
            "command_vx": speed,
            "start_x": float(start_x[idx].cpu()),
            "max_local_x": float(max_x[idx].cpu()),
            "max_z": float(max_z[idx].cpu()),
            "reached_platform": int(reached_platform[idx].cpu()),
            "terminated": int(fell[idx].cpu()),
            "fall_down": int(fall_down[idx].cpu()),
            "stand_still": int(stuck[idx].cpu()),
        }
        for idx, speed in enumerate(speeds)
    ]


def _run_descent(env, speeds, steps):
    obs, _ = env.reset()
    env.do_reset = False
    _place_robot(
        env,
        local_x=7.20,
        local_y=5.00,
        height=env.cfg.depth_stairs.platform_height,
    )

    commands = torch.zeros(len(speeds), 3, device=env.device)
    commands[:, 0] = -torch.tensor(speeds, device=env.device)
    start_x = (env.root_states[:, 0] - env.room_origins[:, 0]).clone()
    min_x = start_x.clone()
    min_z = env.root_states[:, 2].clone()
    fell = torch.zeros(len(speeds), dtype=torch.bool, device=env.device)
    fall_down = torch.zeros(len(speeds), dtype=torch.bool, device=env.device)
    stuck = torch.zeros(len(speeds), dtype=torch.bool, device=env.device)

    with torch.no_grad():
        for _ in range(steps):
            _, _, _, dones, _ = env.step(commands)
            local_x = env.root_states[:, 0] - env.room_origins[:, 0]
            min_x = torch.minimum(min_x, local_x)
            min_z = torch.minimum(min_z, env.root_states[:, 2])
            fell |= dones
            fall_down |= env.reset_fall
            stuck |= env.reset_stand_still

    stair_x = env.cfg.terrain.stair_start_x
    reached_floor = (min_x < stair_x - 0.30) & (min_z <= env.base_init_state[2] + 0.08)
    return [
        {
            "direction": "down",
            "command_vx": -speed,
            "start_x": float(start_x[idx].cpu()),
            "max_local_x": float(min_x[idx].cpu()),
            "max_z": float(min_z[idx].cpu()),
            "reached_platform": int(reached_floor[idx].cpu()),
            "terminated": int(fell[idx].cpu()),
            "fall_down": int(fall_down[idx].cpu()),
            "stand_still": int(stuck[idx].cpu()),
        }
        for idx, speed in enumerate(speeds)
    ]


def main():
    diagnose_args, args = _parse_args()
    if args.task != "go2_pos_depth_stairs":
        raise ValueError("Use --task go2_pos_depth_stairs for locomotion diagnosis.")
    args.headless = True
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    speeds = [0.25, 0.40, 0.55, 0.70]
    _prepare_env(env_cfg, len(speeds))
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    rows = _run_ascent(env, speeds, diagnose_args.steps)
    rows.extend(_run_descent(env, speeds, diagnose_args.steps))
    os.makedirs(os.path.dirname(os.path.abspath(diagnose_args.output_csv)), exist_ok=True)
    with open(diagnose_args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
