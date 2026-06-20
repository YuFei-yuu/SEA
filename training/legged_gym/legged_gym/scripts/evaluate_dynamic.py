# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
import argparse
import csv
import os
import sys

from legged_gym import LEGGED_GYM_ROOT_DIR
import isaacgym
import torch

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


DYNAMIC_TASKS = {
    "go2_pos_sparse_static",
    "go2_pos_dynamic_1",
    "go2_pos_dynamic_2",
    "go2_pos_dynamic_3",
    "go2_pos_dynamic_complex",
}


def _parse_eval_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--output_summary", type=str, default=None)
    known_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    base_args = get_args()
    base_args.num_episodes = known_args.num_episodes
    base_args.output_csv = known_args.output_csv
    base_args.output_summary = known_args.output_summary
    return base_args


def _has_saved_runs(experiment_name):
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", experiment_name)
    if not os.path.isdir(log_root):
        return False
    for entry in os.listdir(log_root):
        if entry == "exported":
            continue
        if os.path.isdir(os.path.join(log_root, entry)):
            return True
    return False


def _apply_eval_overrides(env_cfg, args):
    env_cfg.env.num_envs = 1
    env_cfg.replay.enable_collision_replay = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.max_push_vel_xy = 0.0
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.added_mass_range = [0.0, 0.0]
    env_cfg.env.debug_viz = False
    env_cfg.asset.terminate_after_contacts_on = []

    if args.task == "go2_pos_rough":
        env_cfg.terrain.terrain_types = ["hard_room"]
        env_cfg.terrain.terrain_proportions = [1.0]
        env_cfg.terrain.num_rows = 1
        env_cfg.terrain.num_cols = 1
        env_cfg.terrain.curriculum = True
        env_cfg.terrain.max_init_terrain_level = 3
    else:
        env_cfg.terrain.num_rows = 1
        env_cfg.terrain.num_cols = 1
        env_cfg.terrain.curriculum = False
        env_cfg.terrain.max_init_terrain_level = 0


def _maybe_set_resume_fallback(train_cfg, args):
    if args.resume_experiment_name is not None:
        train_cfg.runner.resume_experiment_name = args.resume_experiment_name
        return
    if args.task not in DYNAMIC_TASKS:
        return
    if not _has_saved_runs(train_cfg.runner.experiment_name):
        train_cfg.runner.resume_experiment_name = "Go2_pos_rough"
        print("No checkpoint found for current task, falling back to Go2_pos_rough.")


def _scalar(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _write_summary_files(summary, output_csv=None, output_summary=None):
    if output_csv is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        write_header = not os.path.isfile(output_csv) or os.path.getsize(output_csv) == 0
        with open(output_csv, "a", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(summary.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(summary)

    if output_summary is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_summary)), exist_ok=True)
        with open(output_summary, "w", encoding="utf-8") as f:
            for key, value in summary.items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.4f}\n")
                else:
                    f.write(f"{key}: {value}\n")


def evaluate(args):
    args.headless = True
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    _apply_eval_overrides(env_cfg, args)

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    train_cfg.runner.resume = True
    train_cfg.runner.load_run = args.load_run if args.load_run is not None else -1
    train_cfg.runner.checkpoint = args.checkpoint if args.checkpoint is not None else -1
    _maybe_set_resume_fallback(train_cfg, args)

    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    loaded_path = task_registry.loaded_policy_path
    obs, _ = env.reset()

    stats = {
        "success": [],
        "safe_success": [],
        "total_collision_count": [],
        "dynamic_collision_count": [],
        "body_collision_count": [],
        "near_miss_count": [],
        "min_ttc": [],
        "shield_intervention_rate": [],
        "active_dynamic_count": [],
        "min_dynamic_clearance": [],
        "future_dynamic_clearance": [],
        "pass_behind_score": [],
        "dynamic_cbf_intervention_rate": [],
        "timeout": [],
        "episode_duration": [],
        "time_to_goal": [],
    }

    completed = 0
    with torch.no_grad():
        while completed < args.num_episodes:
            actions = policy(obs.detach())
            obs, _, _, dones, infos = env.step(actions.detach())
            if not dones.any():
                continue

            episode_info = infos.get("episode", {})
            success = _scalar(episode_info.get("success", 0.0))
            safe_success = _scalar(episode_info.get("safe_success", 0.0))
            total_collision_count = _scalar(episode_info.get("total_collision_count", 0.0))
            dynamic_collision_count = _scalar(episode_info.get("dynamic_collision_count", 0.0))
            body_collision_count = _scalar(episode_info.get("body_collision_count", 0.0))
            near_miss_count = _scalar(episode_info.get("near_miss_count", 0.0))
            min_ttc = _scalar(episode_info.get("min_ttc", 0.0))
            shield_intervention_rate = _scalar(
                episode_info.get("shield_intervention_rate", 0.0)
            )
            active_dynamic_count = _scalar(episode_info.get("active_dynamic_count", 0.0))
            min_dynamic_clearance = _scalar(episode_info.get("min_dynamic_clearance", 0.0))
            future_dynamic_clearance = _scalar(
                episode_info.get("future_dynamic_clearance", 0.0)
            )
            pass_behind_score = _scalar(episode_info.get("pass_behind_score", 0.0))
            dynamic_cbf_intervention_rate = _scalar(
                episode_info.get("dynamic_cbf_intervention_rate", 0.0)
            )
            timeout = _scalar(episode_info.get("timeout", 0.0))
            episode_duration = _scalar(episode_info.get("episode_duration", 0.0))
            time_to_goal = _scalar(episode_info.get("time_to_goal", 0.0))

            stats["success"].append(success)
            stats["safe_success"].append(safe_success)
            stats["total_collision_count"].append(total_collision_count)
            stats["dynamic_collision_count"].append(dynamic_collision_count)
            stats["body_collision_count"].append(body_collision_count)
            stats["near_miss_count"].append(near_miss_count)
            stats["min_ttc"].append(min_ttc)
            stats["shield_intervention_rate"].append(shield_intervention_rate)
            stats["active_dynamic_count"].append(active_dynamic_count)
            stats["min_dynamic_clearance"].append(min_dynamic_clearance)
            stats["future_dynamic_clearance"].append(future_dynamic_clearance)
            stats["pass_behind_score"].append(pass_behind_score)
            stats["dynamic_cbf_intervention_rate"].append(dynamic_cbf_intervention_rate)
            stats["timeout"].append(timeout)
            stats["episode_duration"].append(episode_duration)
            if success > 0.5:
                stats["time_to_goal"].append(time_to_goal)

            completed += 1
            print(
                f"Episode {completed:03d}/{args.num_episodes} | "
                f"success={success:.0f} safe_success={safe_success:.0f} "
                f"dyn_col={dynamic_collision_count:.2f} body_col={body_collision_count:.2f} "
                f"near_miss={near_miss_count:.2f} min_ttc={min_ttc:.2f} "
                f"active_dyn={active_dynamic_count:.1f} min_dyn_clear={min_dynamic_clearance:.2f} "
                f"future_clear={future_dynamic_clearance:.2f} pass_score={pass_behind_score:.2f} "
                f"total_col={total_collision_count:.2f} timeout={timeout:.0f} duration={episode_duration:.2f}s"
            )

    def mean(key):
        values = stats[key]
        return sum(values) / len(values) if values else 0.0

    mean_time_to_goal = (
        sum(stats["time_to_goal"]) / len(stats["time_to_goal"])
        if stats["time_to_goal"]
        else 0.0
    )
    summary = {
        "task": args.task,
        "load_run": args.load_run if args.load_run is not None else "",
        "checkpoint": int(args.checkpoint) if args.checkpoint is not None else -1,
        "checkpoint_path": loaded_path,
        "seed": int(args.seed) if args.seed is not None else -1,
        "num_episodes": int(args.num_episodes),
        "success_rate": mean("success"),
        "safe_success_rate": mean("safe_success"),
        "avg_total_collision_count": mean("total_collision_count"),
        "avg_dynamic_collision_count": mean("dynamic_collision_count"),
        "avg_body_collision_count": mean("body_collision_count"),
        "avg_near_miss_count": mean("near_miss_count"),
        "avg_min_ttc": mean("min_ttc"),
        "avg_shield_intervention_rate": mean("shield_intervention_rate"),
        "avg_active_dynamic_count": mean("active_dynamic_count"),
        "avg_min_dynamic_clearance": mean("min_dynamic_clearance"),
        "avg_future_dynamic_clearance": mean("future_dynamic_clearance"),
        "avg_pass_behind_score": mean("pass_behind_score"),
        "avg_dynamic_cbf_intervention_rate": mean("dynamic_cbf_intervention_rate"),
        "timeout_rate": mean("timeout"),
        "mean_episode_duration": mean("episode_duration"),
        "mean_time_to_goal": mean_time_to_goal,
    }

    print("\nEvaluation summary")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    _write_summary_files(
        summary,
        output_csv=getattr(args, "output_csv", None),
        output_summary=getattr(args, "output_summary", None),
    )
    return summary


if __name__ == "__main__":
    evaluate(_parse_eval_args())
