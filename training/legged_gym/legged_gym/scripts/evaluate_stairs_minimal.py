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
    TERMINAL_REASON_NAMES,
    TERMINAL_SUCCESS,
)
from legged_gym.utils import get_args, task_registry


BASELINE_EXPERIMENT = "Go2_pos_depth_stairs"
BASELINE_RUN = "07_14_06-25-22_"
BASELINE_CHECKPOINT = 200


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--episodes_per_direction", type=int, default=50)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_summary", required=True)
    parser.add_argument("--success_threshold", type=float, default=0.60)
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
    runner, _ = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=None
    )
    policy = runner.get_inference_policy(device=env.device)
    rows = []
    counts = {"up": 0, "down": 0}

    with torch.no_grad():
        for name, direction_value in (("up", 1), ("down", -1)):
            env.cfg.depth_stairs.fixed_direction = direction_value
            obs, _ = env.reset()
            while counts[name] < eval_args.episodes_per_direction:
                actions = policy(obs)
                obs, _, _, _, infos = env.step(actions)
                outcomes = infos.get("episode_outcomes")
                if outcomes is None:
                    continue
                for direction, reason, crossed, steps in zip(
                    outcomes["direction"].tolist(),
                    outcomes["terminal_reason"].tolist(),
                    outcomes["stair_crossed"].tolist(),
                    outcomes["episode_steps"].tolist(),
                ):
                    if counts[name] >= eval_args.episodes_per_direction:
                        break
                    if direction != direction_value:
                        raise RuntimeError("Evaluation episode direction changed unexpectedly")
                    if reason == TERMINAL_SUCCESS and not crossed:
                        raise RuntimeError("A successful episode did not cross the stairs")
                    counts[name] += 1
                    rows.append(
                        {
                            "episode": counts[name],
                            "direction": name,
                            "success": int(reason == TERMINAL_SUCCESS),
                            "terminal_reason": TERMINAL_REASON_NAMES[int(reason)],
                            "stair_crossed": int(crossed),
                            "episode_steps": int(steps),
                            "episode_duration_s": float(steps * env.dt),
                        }
                    )

    os.makedirs(os.path.dirname(os.path.abspath(eval_args.output_csv)), exist_ok=True)
    with open(eval_args.output_csv, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "task": args.task,
        "checkpoint_path": getattr(task_registry, "loaded_policy_path", ""),
        "episodes_per_direction": eval_args.episodes_per_direction,
        "success_threshold": eval_args.success_threshold,
        "directions": {},
    }
    for direction in ("up", "down"):
        selected = [row for row in rows if row["direction"] == direction]
        success_rate = sum(row["success"] for row in selected) / len(selected)
        reasons = {
            reason: sum(row["terminal_reason"] == reason for row in selected)
            for reason in TERMINAL_REASON_NAMES.values()
            if reason != "ongoing"
        }
        summary["directions"][direction] = {
            "success_rate": success_rate,
            "passes": success_rate >= eval_args.success_threshold,
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
