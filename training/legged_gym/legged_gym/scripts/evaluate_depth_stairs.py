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
from legged_gym.envs.base.legged_robot_pos_stairs_minimal import TERMINAL_REASON_NAMES
from legged_gym.utils import get_args, task_registry


def _parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--eval_seed_base", type=int, default=100000)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_summary", required=True)
    parser.add_argument(
        "--disable_cbf",
        action="store_true",
        help="Disable the policy CBF for an apples-to-apples actor ablation.",
    )
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return known, get_args()


def _value(info, key):
    value = info.get(key, 0.0)
    return float(value.detach().cpu().item()) if isinstance(value, torch.Tensor) else float(value)


def main():
    eval_args, args = _parse_args()
    if args.task not in {"go2_pos_depth_stairs", "go2_pos_depth_stairs_passability"}:
        raise ValueError("Use a depth-stairs task for this evaluator.")
    perception_mode = args.depth_mode or "depth_predicted"
    if perception_mode not in ("oracle", "depth_predicted"):
        raise ValueError("Use --depth_mode oracle or --depth_mode depth_predicted.")
    if perception_mode == "depth_predicted" and not args.depth_model:
        raise ValueError("depth_predicted evaluation requires --depth_model <best.pt>.")

    args.headless = True
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    if args.task == "go2_pos_depth_stairs_passability":
        # Use an explicit one-environment schedule.  With automatic resets,
        # Isaac Gym resets the next episode before ``step`` returns done, so
        # changing direction after done would affect the following episode.
        env_cfg.depth_stairs.fixed_direction = 1
        env_cfg.depth_stairs.eval_seed_base = eval_args.eval_seed_base
        env_cfg.depth_stairs.strict_terminal_rules = True
        env_cfg.depth_stairs.enable_stand_still_reset = True
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
    if hasattr(runner.alg.actor_critic, "enable_shield"):
        runner.alg.actor_critic.enable_shield = not eval_args.disable_cbf
    policy = runner.get_inference_policy(device=env.device)
    # Keep the terminal episode available for metric collection.  We reset
    # the completed environment explicitly below after selecting its next
    # direction.
    env.do_reset = False
    obs, _ = env.reset()

    fieldnames = [
        "episode",
        "direction",
        "episode_seed",
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
        "stair_crossed",
        "fully_cleared",
        "obstacle_field_crossed",
        "passability_target_mean",
        "teacher_lateral_only_steps",
        "terminal_reason",
        "terminal_reason_name",
        "stair_approached",
        "goal_reached",
        "cbf_intervention_norm_mean",
        "cbf_intervention_rate",
        "action_filter_delta_mean",
        "action_filter_delta_rate",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(eval_args.output_csv)), exist_ok=True)
    rows = []
    completed = 0
    with torch.no_grad():
        while completed < eval_args.num_episodes:
            action = policy(obs.detach())
            if hasattr(env, "record_policy_action_diagnostics"):
                env.record_policy_action_diagnostics(
                    getattr(runner.alg.actor_critic, "u_bar", None),
                    getattr(
                        runner.alg.actor_critic,
                        "u_safe",
                        getattr(runner.alg.actor_critic, "u_s", None),
                    ),
                )
            obs, _, _, dones, infos = env.step(action.detach())
            if not dones.any():
                continue

            # ``reset_idx`` both snapshots the current episode and creates
            # the next one.  Set the direction before calling it so the
            # next reset is deterministic and alternates up/down episodes.
            if args.task == "go2_pos_depth_stairs_passability":
                env.cfg.depth_stairs.fixed_direction = (
                    1 if (completed + 1) % 2 == 0 else -1
                )
            env.reset_idx(dones.nonzero(as_tuple=False).flatten())
            info = env.extras.get("episode", {})
            row = {"episode": completed + 1}
            for key in fieldnames[1:]:
                if key == "terminal_reason_name":
                    reason = int(_value(info, "terminal_reason"))
                    row[key] = TERMINAL_REASON_NAMES.get(reason, "unknown")
                else:
                    row[key] = _value(info, key)
            rows.append(row)
            completed += 1
            print(
                f"episode={completed:03d} success={row['success']:.0f} "
                f"stair={row['stair_pass_rate']:.0f} low_collision="
                f"{row['low_obstacle_collision_count']:.2f} depth_mae="
                f"{row['depth_ray_mae']:.3f}m terminal={row['terminal_reason']}"
            )
            env.compute_observations()
            obs = env.get_observations()

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
        "stair_cross_rate": sum(row["stair_crossed"] for row in rows) / len(rows),
        "fully_cleared_rate": sum(row["fully_cleared"] for row in rows) / len(rows),
        "obstacle_field_cross_rate": sum(row["obstacle_field_crossed"] for row in rows) / len(rows),
        "stair_approach_rate": sum(row["stair_approached"] for row in rows) / len(rows),
        "goal_reached_rate": sum(row["goal_reached"] for row in rows) / len(rows),
        "mean_cbf_intervention_norm": sum(row["cbf_intervention_norm_mean"] for row in rows) / len(rows),
        "cbf_intervention_rate": sum(row["cbf_intervention_rate"] for row in rows) / len(rows),
        "mean_action_filter_delta": sum(row["action_filter_delta_mean"] for row in rows) / len(rows),
        "action_filter_delta_rate": sum(row["action_filter_delta_rate"] for row in rows) / len(rows),
        "terminal_reason_counts": {
            name: sum(row["terminal_reason_name"] == name for row in rows)
            for name in sorted(set(row["terminal_reason_name"] for row in rows))
        },
    }
    up_rows = [row for row in rows if row["direction"] > 0]
    down_rows = [row for row in rows if row["direction"] < 0]
    summary["up_episodes"] = len(up_rows)
    summary["down_episodes"] = len(down_rows)
    summary["up_success_rate"] = (
        sum(row["depth_safe_success"] for row in up_rows) / len(up_rows)
        if up_rows
        else 0.0
    )
    summary["down_success_rate"] = (
        sum(row["depth_safe_success"] for row in down_rows) / len(down_rows)
        if down_rows
        else 0.0
    )
    os.makedirs(os.path.dirname(os.path.abspath(eval_args.output_summary)), exist_ok=True)
    with open(eval_args.output_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
