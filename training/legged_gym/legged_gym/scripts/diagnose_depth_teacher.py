"""Measure local Teacher efficiency and fixed lateral-template behavior."""

from __future__ import annotations

import argparse
import csv
import os
import sys

from isaacgym import gymapi
import torch

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


def _parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--steps", type=int, default=160)
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return known, get_args()


def main():
    record_args, args = _parse_args()
    if args.task != "go2_pos_depth_stairs_passability":
        raise ValueError("Use --task go2_pos_depth_stairs_passability.")
    args.headless = True
    env_cfg, _ = task_registry.get_cfgs(args.task)
    env_cfg.env.num_envs = args.num_envs or 32
    env_cfg.perception.mode = "oracle"
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=env_cfg)
    obs, _ = env.reset()
    lateral = torch.zeros(env.num_envs, device=env.device)
    forward_ready = torch.zeros(env.num_envs, device=env.device)
    distance_sum = torch.zeros(env.num_envs, device=env.device)
    plan_ratio_sum = torch.zeros(env.num_envs, device=env.device)
    traveled_distance = torch.zeros(env.num_envs, device=env.device)
    first_forward = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
    initial_distance = torch.linalg.vector_norm(
        env.position_targets[:, :2] - env.root_states[:, :2], dim=-1
    )
    previous_xy = env.root_states[:, :2].clone()
    active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    valid_steps = torch.zeros(env.num_envs, device=env.device)
    last_distance = initial_distance.clone()
    with torch.no_grad():
        for step in range(record_args.steps):
            actions = env.get_navigation_teacher_actions()
            lateral += (
                active & (actions[:, 0] < 0.08) & (actions[:, 1].abs() > 0.08)
            ).float()
            forward_ready += (active & (actions[:, 0] > 0.08)).float()
            first_forward = torch.where(
                active & (first_forward < 0) & (actions[:, 0] > 0.08),
                torch.full_like(first_forward, step),
                first_forward,
            )
            goal_distance = torch.linalg.vector_norm(
                env.position_targets[:, :2] - env.root_states[:, :2], dim=-1
            )
            distance_sum += active.float() * goal_distance
            plan_ratio_sum += active.float() * (
                env.teacher_path_distance / goal_distance.clamp(min=0.10)
            )
            valid_steps += active.float()
            last_distance = torch.where(active, goal_distance, last_distance)
            obs, _, _, dones, _ = env.step(actions)
            current_xy = env.root_states[:, :2]
            completed = dones.bool()
            traveled_distance += (active & ~completed).float() * torch.linalg.vector_norm(
                current_xy - previous_xy, dim=-1
            )
            active &= ~completed
            previous_xy = current_xy.clone()
    net_progress = initial_distance - last_distance
    denominator = valid_steps.clamp(min=1.0)
    rows = []
    for idx in range(env.num_envs):
        rows.append(
            {
                "env_id": idx,
                "lateral_only_steps": float(lateral[idx].cpu()),
                "lateral_only_ratio": float((lateral[idx] / denominator[idx]).cpu()),
                "forward_ready_steps": float(forward_ready[idx].cpu()),
                "first_forward_step": int(first_forward[idx].cpu()),
                "first_forward_s": float(
                    torch.where(
                        first_forward[idx] >= 0,
                        first_forward[idx].float() * env.dt,
                        torch.full((), -1.0, device=env.device),
                    ).cpu()
                ),
                "goal_progress": float(net_progress[idx].cpu()),
                "episode_steps_observed": int(valid_steps[idx].cpu()),
                "mean_goal_distance": float((distance_sum[idx] / denominator[idx]).cpu()),
                "mean_plan_ratio": float((plan_ratio_sum[idx] / denominator[idx]).cpu()),
                "traveled_over_progress": float(
                    (traveled_distance[idx] / net_progress[idx].clamp(min=0.10)).cpu()
                ),
            }
        )
    os.makedirs(os.path.dirname(os.path.abspath(record_args.output_csv)), exist_ok=True)
    with open(record_args.output_csv, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {record_args.output_csv}")


if __name__ == "__main__":
    main()
