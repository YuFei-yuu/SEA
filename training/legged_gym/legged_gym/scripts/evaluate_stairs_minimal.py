"""Evaluate bidirectional minimal stair navigation with exclusive outcomes."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import isaacgym
import torch

from legged_gym.envs import *
from legged_gym.envs.base.legged_robot_pos_stairs_minimal import (
    COLLISION_CLASS_NAMES,
    TERMINAL_REASON_NAMES,
    TERMINAL_SUCCESS,
)
from legged_gym.utils import get_args, task_registry


BASELINE_EXPERIMENT = "Go2_pos_depth_stairs"
BASELINE_RUN = "07_14_06-25-22_"
BASELINE_CHECKPOINT = 200


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--episodes_per_direction", type=int, default=20)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_summary", required=True)
    parser.add_argument("--success_threshold", type=float, default=0.75)
    parser.add_argument("--up_seed_base", type=int, default=1000)
    parser.add_argument("--down_seed_base", type=int, default=2000)
    parser.add_argument(
        "--checkpoint_path",
        help="Load an explicit weight-only checkpoint, including a teacher-pretrained file.",
    )
    parser.add_argument(
        "--teacher",
        action="store_true",
        help="Evaluate the privileged training teacher instead of a checkpoint.",
    )
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return known, get_args()


def main():
    eval_args, args = parse_args()
    if args.task != "go2_pos_stairs_minimal":
        raise ValueError("Use --task go2_pos_stairs_minimal")
    args.headless = True
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    env_cfg.env.num_envs = min(32, eval_args.episodes_per_direction)
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.depth_stairs.fixed_direction = 1
    env_cfg.depth_stairs.eval_seed_base = eval_args.up_seed_base
    if eval_args.checkpoint_path:
        train_cfg.runner.resume = False
        args.resume = False
        args.init_checkpoint = eval_args.checkpoint_path
    else:
        train_cfg.runner.resume = True
        train_cfg.runner.resume_experiment_name = (
            args.resume_experiment_name or BASELINE_EXPERIMENT
        )
        train_cfg.runner.load_run = args.load_run if args.load_run is not None else -1
        if args.load_run is None:
            train_cfg.runner.load_run = BASELINE_RUN
        train_cfg.runner.checkpoint = (
            args.checkpoint if args.checkpoint is not None else BASELINE_CHECKPOINT
        )

    env, _ = task_registry.make_env(args.task, args=args, env_cfg=env_cfg)
    if eval_args.teacher:
        policy = lambda _obs: env.get_navigation_teacher_actions()
        policy_path = "privileged_known_room_teacher"
    else:
        runner, _ = task_registry.make_alg_runner(
            env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=None
        )
        policy = runner.get_inference_policy(device=env.device)
        policy_path = getattr(task_registry, "loaded_policy_path", "")
    rows = []
    counts = {"up": 0, "down": 0}
    seen_seeds = {"up": set(), "down": set()}
    num_envs = env.num_envs
    obstacle_x_min, obstacle_x_max = env.cfg.depth_stairs.obstacle_field_x_range
    bypass_y_margin = float(env.cfg.depth_stairs.obstacle_bypass_y_margin)
    room_width = float(env.terrain.env_width)
    field_entered = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    field_crossed = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    edge_bypass = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    field_y_min = torch.full((num_envs,), float("inf"), device=env.device)
    field_y_max = torch.full((num_envs,), float("-inf"), device=env.device)
    action_steps = torch.zeros(num_envs, dtype=torch.long, device=env.device)
    diagonal_steps = torch.zeros_like(action_steps)
    full_3d_steps = torch.zeros_like(action_steps)
    yaw_steps = torch.zeros_like(action_steps)
    stair_steps = torch.zeros_like(action_steps)
    stair_diagonal_steps = torch.zeros_like(action_steps)

    with torch.no_grad():
        for name, direction_value in (("up", 1), ("down", -1)):
            env.cfg.depth_stairs.fixed_direction = direction_value
            seed_base = eval_args.up_seed_base if name == "up" else eval_args.down_seed_base
            env.cfg.depth_stairs.eval_seed_base = seed_base
            env.eval_seed_round.zero_()
            field_entered.zero_()
            field_crossed.zero_()
            edge_bypass.zero_()
            field_y_min.fill_(float("inf"))
            field_y_max.fill_(float("-inf"))
            action_steps.zero_()
            diagonal_steps.zero_()
            full_3d_steps.zero_()
            yaw_steps.zero_()
            stair_steps.zero_()
            stair_diagonal_steps.zero_()
            obs, _ = env.reset()
            while counts[name] < eval_args.episodes_per_direction:
                # Accumulate the state before stepping: completed envs are
                # reset inside env.step(), so their terminal root state is no
                # longer available afterward.
                local_xy = env.root_states[:, :2] - env.room_origins
                in_field = (local_xy[:, 0] >= obstacle_x_min) & (
                    local_xy[:, 0] <= obstacle_x_max
                )
                field_entered |= in_field
                field_y_min = torch.where(
                    in_field, torch.minimum(field_y_min, local_xy[:, 1]), field_y_min
                )
                field_y_max = torch.where(
                    in_field, torch.maximum(field_y_max, local_xy[:, 1]), field_y_max
                )
                edge_bypass |= in_field & (
                    (local_xy[:, 1] < bypass_y_margin)
                    | (local_xy[:, 1] > room_width - bypass_y_margin)
                )
                # Upward episodes must leave the field on its stair side;
                # downward episodes must enter it from the high-platform side.
                # The latter is intentional: the low target interval ends at
                # x=1.10, while the success distance tolerance may terminate
                # the body slightly before the x=1.15 field boundary.
                field_crossed |= (
                    (direction_value > 0) & (local_xy[:, 0] >= obstacle_x_max)
                ) | (
                    (direction_value < 0) & (local_xy[:, 0] <= obstacle_x_max)
                )
                actions = policy(obs)
                diagonal = (torch.abs(actions[:, 0]) > 0.08) & (
                    torch.abs(actions[:, 1]) > 0.06
                )
                turning = torch.abs(actions[:, 2]) > 0.05
                in_stairs = (local_xy[:, 0] >= 4.35) & (local_xy[:, 0] <= 7.15)
                action_steps += 1
                diagonal_steps += diagonal.long()
                full_3d_steps += (diagonal & turning).long()
                yaw_steps += turning.long()
                stair_steps += in_stairs.long()
                stair_diagonal_steps += (
                    in_stairs & (torch.abs(actions[:, 1]) > 0.025)
                ).long()
                obs, _, _, _, infos = env.step(actions)
                outcomes = infos.get("episode_outcomes")
                if outcomes is None:
                    continue
                for batch_index, (direction, seed, reason, crossed, steps) in enumerate(zip(
                    outcomes["direction"].tolist(),
                    outcomes["seed"].tolist(),
                    outcomes["terminal_reason"].tolist(),
                    outcomes["stair_crossed"].tolist(),
                    outcomes["episode_steps"].tolist(),
                )):
                    if counts[name] >= eval_args.episodes_per_direction:
                        break
                    if direction != direction_value:
                        raise RuntimeError("Evaluation episode direction changed unexpectedly")
                    if not seed_base <= seed < seed_base + eval_args.episodes_per_direction:
                        continue
                    if seed in seen_seeds[name]:
                        continue
                    if reason == TERMINAL_SUCCESS and not crossed:
                        raise RuntimeError("A successful episode did not cross the stairs")
                    low_count = int(outcomes["low_obstacle_collision_count"][batch_index].item())
                    wall_count = int(outcomes["wall_contact_count"][batch_index].item())
                    stair_count = int(outcomes["stair_contact_count"][batch_index].item())
                    collision_class = int(outcomes["collision_class"][batch_index].item())
                    obstacle_index = int(
                        outcomes["low_obstacle_collision_index"][batch_index].item()
                    )
                    fully_cleared = int(outcomes["fully_cleared"][batch_index].item())
                    correct_height = int(outcomes["correct_goal_height"][batch_index].item())
                    env_index = int(outcomes["env_ids"][batch_index].item())
                    entered = int(field_entered[env_index].item())
                    crossed_field = int(field_crossed[env_index].item())
                    bypassed = int(edge_bypass[env_index].item())
                    if field_y_min[env_index].isfinite():
                        y_min = float(field_y_min[env_index].item())
                        y_max = float(field_y_max[env_index].item())
                    else:
                        y_min = float("nan")
                        y_max = float("nan")
                    action_count = max(int(action_steps[env_index].item()), 1)
                    stair_count_steps = max(int(stair_steps[env_index].item()), 1)
                    seen_seeds[name].add(seed)
                    counts[name] += 1
                    rows.append(
                        {
                            "episode": counts[name],
                            "seed": int(seed),
                            "direction": name,
                            "success": int(reason == TERMINAL_SUCCESS),
                            "terminal_reason": TERMINAL_REASON_NAMES[int(reason)],
                            "collision_class": COLLISION_CLASS_NAMES[int(collision_class)],
                            "low_obstacle_collision_index": obstacle_index,
                            "terminal_local_x": float(
                                outcomes["terminal_local_x"][batch_index].item()
                            ),
                            "terminal_local_y": float(
                                outcomes["terminal_local_y"][batch_index].item()
                            ),
                            "low_obstacle_collision_count": low_count,
                            "wall_contact_count": wall_count,
                            "stair_contact_count": stair_count,
                            "stair_crossed": int(crossed),
                            "fully_cleared": fully_cleared,
                            "correct_goal_height": correct_height,
                            "obstacle_field_entered": entered,
                            "obstacle_field_crossed": crossed_field,
                            "whole_obstacle_zone_bypass": bypassed,
                            "obstacle_field_y_min": y_min,
                            "obstacle_field_y_max": y_max,
                            "diagonal_command_step_rate": float(
                                diagonal_steps[env_index].item() / action_count
                            ),
                            "full_3d_command_step_rate": float(
                                full_3d_steps[env_index].item() / action_count
                            ),
                            "yaw_command_step_rate": float(
                                yaw_steps[env_index].item() / action_count
                            ),
                            "stair_diagonal_command_step_rate": float(
                                stair_diagonal_steps[env_index].item() / stair_count_steps
                            ),
                            "episode_steps": int(steps),
                            "episode_duration_s": float(steps * env.dt),
                        }
                    )
                    field_entered[env_index] = False
                    field_crossed[env_index] = False
                    edge_bypass[env_index] = False
                    field_y_min[env_index] = float("inf")
                    field_y_max[env_index] = float("-inf")
                    action_steps[env_index] = 0
                    diagonal_steps[env_index] = 0
                    full_3d_steps[env_index] = 0
                    yaw_steps[env_index] = 0
                    stair_steps[env_index] = 0
                    stair_diagonal_steps[env_index] = 0

    os.makedirs(os.path.dirname(os.path.abspath(eval_args.output_csv)), exist_ok=True)
    with open(eval_args.output_csv, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "task": args.task,
        "checkpoint_path": policy_path,
        "episodes_per_direction": eval_args.episodes_per_direction,
        "success_threshold": eval_args.success_threshold,
        "seed_ranges": {
            "up": [eval_args.up_seed_base, eval_args.up_seed_base + eval_args.episodes_per_direction - 1],
            "down": [eval_args.down_seed_base, eval_args.down_seed_base + eval_args.episodes_per_direction - 1],
        },
        "directions": {},
    }
    for direction in ("up", "down"):
        selected = [row for row in rows if row["direction"] == direction]
        success_rate = sum(row["success"] for row in selected) / len(selected)
        safe_success_rate = sum(
            row["success"]
            and row["low_obstacle_collision_count"] == 0
            and row["stair_crossed"]
            and row["fully_cleared"]
            and row["obstacle_field_entered"]
            and row["obstacle_field_crossed"]
            and not row["whole_obstacle_zone_bypass"]
            for row in selected
        ) / len(selected)
        reasons = {
            reason: sum(row["terminal_reason"] == reason for row in selected)
            for reason in TERMINAL_REASON_NAMES.values()
            if reason != "ongoing"
        }
        summary["directions"][direction] = {
            "success_rate": success_rate,
            "safe_success_rate": safe_success_rate,
            "zero_obstacle_collision_successes": sum(
                row["success"] and row["low_obstacle_collision_count"] == 0 for row in selected
            ),
            "stair_crossed_successes": sum(
                row["success"] and row["stair_crossed"] and row["fully_cleared"] for row in selected
            ),
            "obstacle_field_crossed_successes": sum(
                row["success"] and row["obstacle_field_entered"] and row["obstacle_field_crossed"]
                and not row["whole_obstacle_zone_bypass"]
                for row in selected
            ),
            "whole_obstacle_zone_bypasses": sum(
                row["whole_obstacle_zone_bypass"] for row in selected
            ),
            "mean_diagonal_command_step_rate": sum(
                row["diagonal_command_step_rate"] for row in selected
            ) / len(selected),
            "mean_full_3d_command_step_rate": sum(
                row["full_3d_command_step_rate"] for row in selected
            ) / len(selected),
            "mean_yaw_command_step_rate": sum(
                row["yaw_command_step_rate"] for row in selected
            ) / len(selected),
            "mean_stair_diagonal_command_step_rate": sum(
                row["stair_diagonal_command_step_rate"] for row in selected
            ) / len(selected),
            "collision_classes": {
                collision: sum(row["collision_class"] == collision for row in selected)
                for collision in COLLISION_CLASS_NAMES.values()
                if collision != "none"
            },
            "passes": safe_success_rate >= eval_args.success_threshold,
            "terminal_reasons": reasons,
        }
    summary["passes_acceptance"] = all(
        value["passes"] for value in summary["directions"].values()
    )
    os.makedirs(os.path.dirname(os.path.abspath(eval_args.output_summary)), exist_ok=True)
    with open(eval_args.output_summary, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
