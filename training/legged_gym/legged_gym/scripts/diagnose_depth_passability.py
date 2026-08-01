"""Fixed-seed closed-loop diagnostics for Teacher and oracle actors.

The script deliberately keeps completed environments frozen instead of
auto-resetting them.  This makes every row a first-terminal snapshot with the
same seed, direction and scene across Teacher/CBF-on/CBF-off comparisons.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

from isaacgym import gymapi  # noqa: F401  # initialize Isaac Gym before torch
import torch
from legged_gym.envs import *
from legged_gym.envs.base.legged_robot_pos_stairs_minimal import TERMINAL_REASON_NAMES
from legged_gym.utils import get_args, task_registry


def _parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--episodes_per_direction", type=int, default=4)
    parser.add_argument("--eval_seed_base", type=int, default=140000)
    parser.add_argument("--mode", choices=("teacher", "oracle_actor"), default="teacher")
    parser.add_argument("--direction", choices=("up", "down", "both"), default="both")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--disable_cbf", action="store_true")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--max_steps", type=int, default=800)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_summary", required=True)
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return known, get_args()


def _scalar(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _configure(env_cfg, args, direction, seed_base, num_envs):
    env_cfg.env.num_envs = num_envs
    env_cfg.perception.mode = "oracle"
    env_cfg.perception.render_depth_in_oracle = False
    env_cfg.depth_stairs.fixed_direction = direction
    env_cfg.depth_stairs.eval_seed_base = seed_base
    env_cfg.depth_stairs.strict_terminal_rules = True
    env_cfg.depth_stairs.enable_stand_still_reset = True
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.asset.terminate_after_contacts_on = []


def _run_one(mode, direction, record_args, args):
    count = int(record_args.episodes_per_direction)
    env_cfg, train_cfg = task_registry.get_cfgs("go2_pos_depth_stairs_passability")
    _configure(env_cfg, args, direction, record_args.eval_seed_base, count)
    args.headless = True
    runner = None
    if mode == "oracle_actor":
        train_cfg.runner.resume = True
        train_cfg.runner.load_run = args.load_run if args.load_run is not None else -1
        train_cfg.runner.checkpoint = args.checkpoint if args.checkpoint is not None else -1
    env, _ = task_registry.make_env(
        "go2_pos_depth_stairs_passability", args=args, env_cfg=env_cfg
    )
    if mode == "oracle_actor":
        runner, _ = task_registry.make_alg_runner(
            env=env, name="go2_pos_depth_stairs_passability", args=args, train_cfg=train_cfg
        )
        actor = runner.alg.actor_critic
        actor.enable_shield = not record_args.disable_cbf
        policy = runner.get_inference_policy(device=env.device)
    else:
        policy = None
        actor = None

    # Runner construction may call env.reset once.  Start the measured batch
    # explicitly after clearing the seed counter so internal initialization
    # resets cannot shift the requested fixed seed range.
    env.do_reset = False
    env.eval_seed_round.zero_()
    env.reset_idx(torch.arange(count, device=env.device))
    zeros = torch.zeros(count, 3, device=env.device)
    env.nav_actions_orig = zeros.clone()
    env.nav_actions_filtered = zeros.clone()
    env.nav_actions_after_clip = zeros.clone()
    env.slr_commands = zeros.clone()
    env.actions_orig = torch.zeros(count, env.num_actions, device=env.device)
    env.update_percetion()
    env.compute_observations()
    obs = env.get_observations()
    active = torch.ones(count, dtype=torch.bool, device=env.device)
    rows = []
    with torch.no_grad():
        for _ in range(record_args.max_steps):
            if mode == "teacher":
                action = env.get_navigation_teacher_actions()
                env.record_policy_action_diagnostics(action, action)
            else:
                action = policy(obs.detach())
                env.record_policy_action_diagnostics(
                    getattr(actor, "u_bar", None),
                    getattr(actor, "u_safe", getattr(actor, "u_s", None)),
                )
            # Keep an unstable checkpoint from passing NaN/Inf into PhysX;
            # the resulting trajectory is still recorded as a failed timeout.
            action = torch.nan_to_num(action, nan=0.0, posinf=3.0, neginf=-3.0)
            action = action.clamp(min=-3.0, max=3.0)
            action = torch.where(active[:, None], action, torch.zeros_like(action))
            if record_args.trace and mode == "teacher" and (_ % 20 == 0):
                local = env.root_states[0, :2] - env.room_origins[0]
                print(
                    f"trace step={_:04d} local=({local[0].item():.3f},{local[1].item():.3f}) "
                    f"wp=({env.teacher_waypoint[0,0].item():.3f},{env.teacher_waypoint[0,1].item():.3f}) "
                    f"label={int(env.passability_targets[0].item())} action={action[0].tolist()}"
                )
            obs, _, _, dones, _ = env.step(action.detach())
            finished = dones.bool() & active
            for index in finished.nonzero(as_tuple=False).flatten().tolist():
                reason = int(env.terminal_reason[index].item())
                length = max(int(env.episode_length_buf[index].item()), 1)
                cbf_mean = env.policy_cbf_intervention_sum[index] / float(length)
                cbf_rate = env.policy_cbf_intervention_steps[index] / float(length)
                filter_mean = env.action_filter_delta_sum[index] / float(length)
                filter_rate = env.action_filter_delta_steps[index] / float(length)
                low = int(env.low_obstacle_collision_count[index].item())
                success = int(env.goal_reached_flag[index].item())
                cleared = int(env.fully_cleared[index].item())
                rows.append(
                    {
                        "mode": "oracle_actor_no_cbf" if mode == "oracle_actor" and record_args.disable_cbf else mode,
                        "cbf_enabled": int(mode == "oracle_actor" and actor.enable_shield),
                        "direction": int(env.navigation_direction[index].item()),
                        "episode_seed": int(env.episode_seed[index].item()),
                        "terminal_local_x": float((env.root_states[index, 0] - env.room_origins[index, 0]).item()),
                        "terminal_local_y": float((env.root_states[index, 1] - env.room_origins[index, 1]).item()),
                        "start_local_y": float(env.episode_start_local_y[index].item()),
                        "goal_local_y": float((env.position_targets[index, 1] - env.room_origins[index, 1]).item()),
                        "terminal_reason": reason,
                        "terminal_reason_name": TERMINAL_REASON_NAMES.get(reason, "unknown"),
                        "success": success,
                        "safe_success": int(success and low == 0 and cleared),
                        "obstacle_field_crossed": int(env.obstacle_field_crossed[index].item()),
                        "stair_approached": int(env.stair_approached[index].item()),
                        "stair_crossed": int(env.stair_crossed[index].item()),
                        "fully_cleared": cleared,
                        "goal_reached": success,
                        "low_obstacle_collision_count": low,
                        "body_collision_count": int(env.body_collision_count[index].item()),
                        "total_collision_count": int(env.total_collision_count[index].item()),
                        "episode_steps": length,
                        "episode_duration": float(length * env.dt),
                        "cbf_intervention_norm_mean": _scalar(cbf_mean),
                        "cbf_intervention_rate": _scalar(cbf_rate),
                        "action_filter_delta_mean": _scalar(filter_mean),
                        "action_filter_delta_rate": _scalar(filter_rate),
                        "teacher_lateral_only_ratio": float(
                            env.teacher_lateral_only_count[index].item() / length
                        ),
                        "teacher_lateral_only_total_ratio": float(
                            env.teacher_lateral_only_total_count[index].item() / length
                        ),
                    }
                )
            active &= ~finished
            if not bool(active.any()):
                break
    # A policy that never emits a terminal signal before the diagnostic
    # horizon is an explicit timeout, not an empty evaluation.  Preserve the
    # partial stage/collision metrics so fixed-seed gates count it as failure.
    for index in active.nonzero(as_tuple=False).flatten().tolist():
        reason = TERMINAL_TIMEOUT
        length = max(int(record_args.max_steps), 1)
        cbf_mean = env.policy_cbf_intervention_sum[index] / float(length)
        cbf_rate = env.policy_cbf_intervention_steps[index] / float(length)
        filter_mean = env.action_filter_delta_sum[index] / float(length)
        filter_rate = env.action_filter_delta_steps[index] / float(length)
        low = int(env.low_obstacle_collision_count[index].item())
        success = int(env.goal_reached_flag[index].item())
        cleared = int(env.fully_cleared[index].item())
        rows.append(
            {
                "mode": "oracle_actor_no_cbf" if mode == "oracle_actor" and record_args.disable_cbf else mode,
                "cbf_enabled": int(mode == "oracle_actor" and actor.enable_shield),
                "direction": int(env.navigation_direction[index].item()),
                "episode_seed": int(env.episode_seed[index].item()),
                "terminal_local_x": float((env.root_states[index, 0] - env.room_origins[index, 0]).item()),
                "terminal_local_y": float((env.root_states[index, 1] - env.room_origins[index, 1]).item()),
                "start_local_y": float(env.episode_start_local_y[index].item()),
                "goal_local_y": float((env.position_targets[index, 1] - env.room_origins[index, 1]).item()),
                "terminal_reason": reason,
                "terminal_reason_name": TERMINAL_REASON_NAMES[reason],
                "success": success,
                "safe_success": int(success and low == 0 and cleared),
                "obstacle_field_crossed": int(env.obstacle_field_crossed[index].item()),
                "stair_approached": int(env.stair_approached[index].item()),
                "stair_crossed": int(env.stair_crossed[index].item()),
                "fully_cleared": cleared,
                "goal_reached": success,
                "low_obstacle_collision_count": low,
                "body_collision_count": int(env.body_collision_count[index].item()),
                "total_collision_count": int(env.total_collision_count[index].item()),
                "episode_steps": length,
                "episode_duration": float(length * env.dt),
                "cbf_intervention_norm_mean": _scalar(cbf_mean),
                "cbf_intervention_rate": _scalar(cbf_rate),
                "action_filter_delta_mean": _scalar(filter_mean),
                "action_filter_delta_rate": _scalar(filter_rate),
                "teacher_lateral_only_ratio": float(
                    env.teacher_lateral_only_count[index].item() / length
                ),
                "teacher_lateral_only_total_ratio": float(
                    env.teacher_lateral_only_total_count[index].item() / length
                ),
            }
        )
    return rows


def _write_results(rows, record_args, args):

    os.makedirs(os.path.dirname(os.path.abspath(record_args.output_csv)), exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["mode"]
    with open(record_args.output_csv, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    by_mode = {}
    for mode in sorted(set(row["mode"] for row in rows)):
        subset = [row for row in rows if row["mode"] == mode]
        by_mode[mode] = {
            "episodes": len(subset),
            "safe_success_rate": sum(row["safe_success"] for row in subset) / max(len(subset), 1),
            "terminal_reason_counts": {
                name: sum(row["terminal_reason_name"] == name for row in subset)
                for name in sorted(set(row["terminal_reason_name"] for row in subset))
            },
            "obstacle_field_cross_rate": sum(row["obstacle_field_crossed"] for row in subset) / max(len(subset), 1),
            "stair_approach_rate": sum(row["stair_approached"] for row in subset) / max(len(subset), 1),
            "stair_cross_rate": sum(row["stair_crossed"] for row in subset) / max(len(subset), 1),
            "fully_cleared_rate": sum(row["fully_cleared"] for row in subset) / max(len(subset), 1),
            "goal_reached_rate": sum(row["goal_reached"] for row in subset) / max(len(subset), 1),
            "mean_cbf_intervention_norm": sum(row["cbf_intervention_norm_mean"] for row in subset) / max(len(subset), 1),
            "mean_action_filter_delta": sum(row["action_filter_delta_mean"] for row in subset) / max(len(subset), 1),
        }
    summary = {
        "task": args.task,
        "eval_seed_base": record_args.eval_seed_base,
        "episodes_per_direction": record_args.episodes_per_direction,
        "modes": by_mode,
    }
    os.makedirs(os.path.dirname(os.path.abspath(record_args.output_summary)), exist_ok=True)
    with open(record_args.output_summary, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2))


def main():
    record_args, args = _parse_args()
    if args.task != "go2_pos_depth_stairs_passability":
        raise ValueError("Use --task go2_pos_depth_stairs_passability.")

    if record_args.compare:
        # Isaac Gym can leave GPU resources live after destroying a simulator.
        # Run each comparison in a short-lived child process and aggregate its
        # CSVs, preserving the exact seed/direction pair without segfault-prone
        # simulator reuse.
        import subprocess
        import tempfile

        rows = []
        directions = ("up", "down")
        jobs = [("teacher", False), ("oracle_actor", False), ("oracle_actor", True)]
        with tempfile.TemporaryDirectory(prefix="depth_passability_compare_") as directory:
            for mode, disable_cbf in jobs:
                for direction in directions:
                    child_csv = os.path.join(directory, f"{mode}_{disable_cbf}_{direction}.csv")
                    child_json = os.path.join(directory, f"{mode}_{disable_cbf}_{direction}.json")
                    command = [
                        sys.executable,
                        __file__,
                        "--task", args.task,
                        "--headless",
                        "--mode", mode,
                        "--direction", direction,
                        "--episodes_per_direction", str(record_args.episodes_per_direction),
                        "--eval_seed_base", str(record_args.eval_seed_base),
                        "--max_steps", str(record_args.max_steps),
                        "--output_csv", child_csv,
                        "--output_summary", child_json,
                    ]
                    if disable_cbf:
                        command.append("--disable_cbf")
                    if args.load_run is not None:
                        command.extend(["--load_run", str(args.load_run)])
                    if args.checkpoint is not None:
                        command.extend(["--checkpoint", str(args.checkpoint)])
                    subprocess.run(command, check=False)
                    if os.path.exists(child_csv):
                        with open(child_csv, newline="", encoding="utf-8") as stream:
                            rows.extend(list(csv.DictReader(stream)))
        # Child CSV values are strings; normalize the fields used by the
        # summary while retaining the row-level artifact exactly.
        numeric = {
            "cbf_enabled", "direction", "episode_seed", "terminal_reason", "success",
            "safe_success", "obstacle_field_crossed", "stair_approached", "stair_crossed",
            "fully_cleared", "goal_reached", "low_obstacle_collision_count",
            "body_collision_count", "total_collision_count", "episode_steps",
            "episode_duration", "cbf_intervention_norm_mean", "cbf_intervention_rate",
            "action_filter_delta_mean", "action_filter_delta_rate", "teacher_lateral_only_ratio",
            "teacher_lateral_only_total_ratio",
            "terminal_local_x", "terminal_local_y", "start_local_y", "goal_local_y",
        }
        for row in rows:
            for key in numeric:
                if key in row:
                    row[key] = float(row[key]) if "." in row[key] else int(row[key])
        _write_results(rows, record_args, args)
        return

    directions = (1, -1) if record_args.direction == "both" else ((1,) if record_args.direction == "up" else (-1,))
    rows = []
    for direction in directions:
        rows.extend(_run_one(record_args.mode, direction, record_args, args))
    _write_results(rows, record_args, args)


if __name__ == "__main__":
    main()
