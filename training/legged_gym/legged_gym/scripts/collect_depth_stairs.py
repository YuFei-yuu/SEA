"""Collect paired Go2 depth images and oracle navigation rays for the stair task."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from isaacgym import gymapi
import torch

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


def _parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_steps", type=int, default=2000)
    parser.add_argument("--sample_interval", type=int, default=5)
    parser.add_argument("--shard_size", type=int, default=4096)
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return known, get_args()


def _write_shard(output_dir, shard_index, depths, rays):
    if not depths:
        return
    path = os.path.join(output_dir, f"depth_rays_{shard_index:04d}.npz")
    np.savez_compressed(
        path,
        depth=np.concatenate(depths, axis=0).astype(np.float32),
        rays=np.concatenate(rays, axis=0).astype(np.float32),
    )
    print(f"Saved {path}")


def main():
    record_args, args = _parse_args()
    if args.task != "go2_pos_depth_stairs":
        raise ValueError("Use --task go2_pos_depth_stairs for depth-ray collection.")

    os.makedirs(record_args.output_dir, exist_ok=True)
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg.perception.mode = "oracle"
    env_cfg.perception.model_path = ""
    env_cfg.env.num_envs = args.num_envs or min(env_cfg.env.num_envs, 64)
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs, _ = env.reset()

    metadata = {
        "task": args.task,
        "seed": int(env_cfg.seed),
        "num_rays": int(env.rays.shape[1]),
        "camera_resolution": list(env_cfg.sensors.depth_cam.resolution),
        "depth_min": float(env_cfg.sensors.depth_cam.min_),
        "depth_max": float(env_cfg.sensors.depth_cam.max_),
        "perception_mode": "oracle",
    }
    with open(os.path.join(record_args.output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    depths, rays = [], []
    shard_index = 0
    samples_in_shard = 0
    for step in range(record_args.num_steps):
        actions = torch.zeros(
            env.num_envs, env.num_nav_actions, device=env.device, dtype=torch.float
        )
        actions[:, 0].uniform_(0.10, 0.55)
        actions[:, 1].uniform_(-0.25, 0.25)
        actions[:, 2].uniform_(-0.35, 0.35)
        obs, _, _, _, _ = env.step(actions)

        if step == 0 or step % record_args.sample_interval != 0:
            continue
        depths.append(env.cam_obs.detach().cpu().numpy())
        rays.append(env.oracle_rays.detach().cpu().numpy())
        samples_in_shard += env.num_envs
        if samples_in_shard >= record_args.shard_size:
            _write_shard(record_args.output_dir, shard_index, depths, rays)
            shard_index += 1
            depths, rays, samples_in_shard = [], [], 0

    _write_shard(record_args.output_dir, shard_index, depths, rays)


if __name__ == "__main__":
    main()
