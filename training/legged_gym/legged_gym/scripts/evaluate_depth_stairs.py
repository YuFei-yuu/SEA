"""Evaluate the Go2 depth-stair navigation policy and write reproducible episode metrics."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

from isaacgym import gymapi
import torch

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


def _parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_summary", required=True)
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return known, get_args()


def _value(info, key):
    value = info.get(key, 0.0)
    return float(value.detach().cpu().item()) if isinstance(value, torch.Tensor) else float(value)


def main():
    eval_args, args = _parse_args()
    if args.task != "go2_pos_depth_stairs":
        raise ValueError("Use --task go2_pos_depth_stairs for this evaluator.")
    perception_mode = args.depth_mode or "depth_predicted"
    if perception_mode not in ("oracle", "depth_predicted"):
        raise ValueError("Use --depth_mode oracle or --depth_mode depth_predicted.")
    if perception_mode == "depth_predicted" and not args.depth_model:
        raise ValueError("depth_predicted evaluation requires --depth_model <best.pt>.")

    args.headless = True
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.asset.terminate_after_contacts_on = []
    train_cfg.runner.resume = True
    train_cfg.runner.load_run = args.load_run if args.load_run is not None else -1
    train_cfg.runner.checkpoint = args.checkpoint if args.checkpoint is not None else -1

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    runner, _ = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = runner.get_inference_policy(device=env.device)
    obs, _ = env.reset()

    fieldnames = [
        "episode",
        "success",
        "safe_success",
        "depth_safe_success",
        "low_obstacle_collision_count",
        "body_collision_count",
        "total_collision_count",
        "stair_pass_rate",
        "depth_ray_mae",
        "episode_duration",
        "time_to_goal",
        "timeout",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(eval_args.output_csv)), exist_ok=True)
    rows = []
    completed = 0
    with torch.no_grad():
        while completed < eval_args.num_episodes:
            action = policy(obs.detach())
            obs, _, _, dones, infos = env.step(action.detach())
            if not dones.any():
                continue
            info = infos.get("episode", {})
            row = {"episode": completed + 1}
            for key in fieldnames[1:]:
                row[key] = _value(info, key)
            rows.append(row)
            completed += 1
            print(
                f"episode={completed:03d} success={row['success']:.0f} "
                f"stair={row['stair_pass_rate']:.0f} low_collision="
                f"{row['low_obstacle_collision_count']:.2f} depth_mae="
                f"{row['depth_ray_mae']:.3f}m"
            )

    with open(eval_args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "task": args.task,
        "perception_mode": perception_mode,
        "depth_model": args.depth_model,
        "checkpoint_path": getattr(task_registry, "loaded_policy_path", ""),
        "num_episodes": len(rows),
        "success_rate": sum(row["success"] for row in rows) / len(rows),
        "safe_success_rate": sum(row["depth_safe_success"] for row in rows) / len(rows),
        "stair_pass_rate": sum(row["stair_pass_rate"] for row in rows) / len(rows),
        "avg_low_obstacle_collision_count": sum(
            row["low_obstacle_collision_count"] for row in rows
        ) / len(rows),
        "avg_total_collision_count": sum(
            row["total_collision_count"] for row in rows
        ) / len(rows),
        "mean_depth_ray_mae": sum(row["depth_ray_mae"] for row in rows) / len(rows),
        "mean_time_to_goal": sum(row["time_to_goal"] for row in rows) / len(rows),
    }
    os.makedirs(os.path.dirname(os.path.abspath(eval_args.output_summary)), exist_ok=True)
    with open(eval_args.output_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
