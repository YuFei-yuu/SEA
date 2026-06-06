# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
import argparse
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
}


def _parse_eval_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--num_episodes", type=int, default=50)
    known_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    base_args = get_args()
    base_args.num_episodes = known_args.num_episodes
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
            timeout = _scalar(episode_info.get("timeout", 0.0))
            episode_duration = _scalar(episode_info.get("episode_duration", 0.0))
            time_to_goal = _scalar(episode_info.get("time_to_goal", 0.0))

            stats["success"].append(success)
            stats["safe_success"].append(safe_success)
            stats["total_collision_count"].append(total_collision_count)
            stats["dynamic_collision_count"].append(dynamic_collision_count)
            stats["body_collision_count"].append(body_collision_count)
            stats["timeout"].append(timeout)
            stats["episode_duration"].append(episode_duration)
            if success > 0.5:
                stats["time_to_goal"].append(time_to_goal)

            completed += 1
            print(
                f"Episode {completed:03d}/{args.num_episodes} | "
                f"success={success:.0f} safe_success={safe_success:.0f} "
                f"dyn_col={dynamic_collision_count:.2f} body_col={body_collision_count:.2f} "
                f"total_col={total_collision_count:.2f} timeout={timeout:.0f} duration={episode_duration:.2f}s"
            )

    def mean(key):
        values = stats[key]
        return sum(values) / len(values) if values else 0.0

    print("\nEvaluation summary")
    print(f"task: {args.task}")
    print(f"checkpoint_path: {loaded_path}")
    print(f"seed: {args.seed}")
    print(f"num_episodes: {args.num_episodes}")
    print(f"success_rate: {mean('success'):.4f}")
    print(f"safe_success_rate: {mean('safe_success'):.4f}")
    print(f"avg_total_collision_count: {mean('total_collision_count'):.4f}")
    print(f"avg_dynamic_collision_count: {mean('dynamic_collision_count'):.4f}")
    print(f"avg_body_collision_count: {mean('body_collision_count'):.4f}")
    print(f"timeout_rate: {mean('timeout'):.4f}")
    if stats["time_to_goal"]:
        print(f"mean_time_to_goal: {sum(stats['time_to_goal']) / len(stats['time_to_goal']):.4f}")
    else:
        print("mean_time_to_goal: 0.0000")


if __name__ == "__main__":
    evaluate(_parse_eval_args())
