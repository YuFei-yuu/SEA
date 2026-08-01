"""Record one successful direction of the minimal stair navigation task."""

from __future__ import annotations

import argparse
import csv
import os
import sys

import cv2
from isaacgym import gymapi
import torch

from legged_gym.envs import *
from legged_gym.envs.base.legged_robot_pos_stairs_minimal import TERMINAL_REASON_NAMES
from legged_gym.envs.base.legged_robot_pos_stairs_minimal import TERMINAL_SUCCESS
from legged_gym.utils import get_args, task_registry


BASELINE_EXPERIMENT = "Go2_pos_depth_stairs"
BASELINE_RUN = "07_14_06-25-22_"
BASELINE_CHECKPOINT = 200


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--direction", choices=("up", "down"), required=True)
    parser.add_argument("--output_video")
    parser.add_argument("--output_trajectory")
    parser.add_argument(
        "--output_dir",
        help="Directory for trial_XX.mp4/csv outputs when recording multiple trials.",
    )
    parser.add_argument("--num_trials", type=int, default=1)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--seeds",
        help="Comma-separated per-trial seeds; overrides --seed and --num_trials.",
    )
    parser.add_argument(
        "--checkpoint_path",
        help="Load an explicit weight-only checkpoint, including a teacher-pretrained file.",
    )
    known, remaining = parser.parse_known_args()
    if known.seeds:
        known.trial_seeds = tuple(
            int(value.strip()) for value in known.seeds.split(",") if value.strip()
        )
        if not known.trial_seeds:
            raise ValueError("--seeds must contain at least one integer")
        known.num_trials = len(known.trial_seeds)
    else:
        known.trial_seeds = None
    if known.num_trials < 1:
        raise ValueError("--num_trials must be positive")
    sys.argv = [sys.argv[0]] + remaining
    return known, get_args()


def _record_trial(env, policy, camera, camera_props, record_args, trial_index, seed,
                  output_video, output_trajectory):
    os.makedirs(os.path.dirname(os.path.abspath(output_video)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_trajectory)), exist_ok=True)
    video = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        record_args.fps,
        (camera_props.width, camera_props.height),
    )
    if not video.isOpened():
        raise RuntimeError(f"Could not open video output: {output_video}")

    # Reset the deterministic seed counter for every trial so the requested
    # five recordings use five distinct transverse start positions.
    env.cfg.depth_stairs.eval_seed_base = seed
    env.eval_seed_round.zero_()
    obs, _ = env.reset()
    start_local_y = float(
        (env.root_states[0, 1] - env.room_origins[0, 1]).item()
    )
    # Stay on the room-interior side of the robot. This prevents the boundary
    # wall from hiding low/high transverse starts while preserving a side view.
    camera_side = 1.0 if start_local_y < 0.5 * float(env.terrain.env_width) else -1.0
    trajectory = []
    max_steps = record_args.max_steps or int(env.max_episode_length)
    terminal_reason = "incomplete"
    with torch.no_grad():
        for step in range(max_steps):
            root = env.root_states[0, :3]
            eye = gymapi.Vec3(
                float(root[0]),
                float(root[1] + camera_side * 2.25),
                float(root[2] + 1.05),
            )
            camera_target = gymapi.Vec3(
                float(root[0]), float(root[1]), float(root[2] + 0.05)
            )
            env.gym.set_camera_location(camera, env.envs[0], eye, camera_target)
            actions = policy(obs)
            obs, _, _, _, infos = env.step(actions)
            outcomes = infos.get("episode_outcomes")
            if outcomes is not None:
                # The vectorized environment has already reset the robot when
                # step() returns. Do not append that next-episode pose/frame to
                # the completed trial's trajectory and video.
                terminal_reason = TERMINAL_REASON_NAMES[int(outcomes["terminal_reason"][0])]
                break
            xyz = (env.root_states[0, :3] - torch.cat(
                (env.room_origins[0], torch.zeros(1, device=env.device))
            )).detach().cpu().tolist()
            command = actions[0].detach().cpu().tolist()
            trajectory.append(
                {
                    "step": step,
                    "x": xyz[0],
                    "y": xyz[1],
                    "z": xyz[2],
                    "vx": command[0],
                    "vy": command[1],
                    "yaw_rate": command[2],
                }
            )
            env.gym.render_all_camera_sensors(env.sim)
            rgba = env.gym.get_camera_image(
                env.sim, env.envs[0], camera, gymapi.IMAGE_COLOR
            ).reshape(camera_props.height, camera_props.width, 4)
            frame = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
            video.write(frame)
    video.release()
    with open(output_trajectory, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("step", "x", "y", "z", "vx", "vy", "yaw_rate"),
        )
        writer.writeheader()
        writer.writerows(trajectory)
    print(f"trial={trial_index + 1} seed={seed} terminal_reason={terminal_reason}")
    print(f"Saved {output_video}")
    print(f"Saved {output_trajectory}")
    if terminal_reason != TERMINAL_REASON_NAMES[TERMINAL_SUCCESS]:
        raise RuntimeError(
            f"The recorded {record_args.direction} trial {trial_index + 1} "
            f"was not successful: {terminal_reason}"
        )


def main():
    record_args, args = parse_args()
    if args.task != "go2_pos_stairs_minimal":
        raise ValueError("Use --task go2_pos_stairs_minimal")
    if record_args.num_trials > 1 and not record_args.output_dir:
        raise ValueError("--output_dir is required when --num_trials is greater than one")
    if record_args.num_trials == 1 and (
        not record_args.output_video or not record_args.output_trajectory
    ):
        raise ValueError("Single-trial recording requires --output_video and --output_trajectory")

    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    env_cfg.env.num_envs = 1
    env_cfg.env.disable_graphics = False
    env_cfg.env.debug_viz = False
    env_cfg.visualization.draw_position_target = False
    env_cfg.visualization.draw_rays = False
    env_cfg.visualization.draw_scan_dots = False
    env_cfg.visualization.draw_collision_points = False
    env_cfg.visualization.draw_dynamic_obstacles = False
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.depth_stairs.fixed_direction = 1 if record_args.direction == "up" else -1
    default_seed = 1000 if record_args.direction == "up" else 2000
    base_seed = record_args.seed if record_args.seed is not None else default_seed
    env_cfg.depth_stairs.eval_seed_base = base_seed
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    if record_args.checkpoint_path:
        train_cfg.runner.resume = False
        args.resume = False
        args.init_checkpoint = record_args.checkpoint_path
    else:
        train_cfg.runner.resume = True
        train_cfg.runner.resume_experiment_name = (
            args.resume_experiment_name or BASELINE_EXPERIMENT
        )
        train_cfg.runner.load_run = args.load_run or BASELINE_RUN
        train_cfg.runner.checkpoint = (
            args.checkpoint if args.checkpoint is not None else BASELINE_CHECKPOINT
        )
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=env_cfg)
    runner, _ = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=None
    )
    policy = runner.get_inference_policy(device=env.device)

    camera_props = gymapi.CameraProperties()
    camera_props.width = 960
    camera_props.height = 540
    camera = env.gym.create_camera_sensor(env.envs[0], camera_props)
    for trial_index in range(record_args.num_trials):
        trial_seed = (
            record_args.trial_seeds[trial_index]
            if record_args.trial_seeds is not None
            else base_seed + trial_index
        )
        if record_args.num_trials == 1:
            output_video = record_args.output_video
            output_trajectory = record_args.output_trajectory
        else:
            os.makedirs(record_args.output_dir, exist_ok=True)
            stem = f"{record_args.direction}_trial_{trial_index + 1:02d}"
            output_video = os.path.join(record_args.output_dir, f"{stem}.mp4")
            output_trajectory = os.path.join(record_args.output_dir, f"{stem}.csv")
        _record_trial(
            env,
            policy,
            camera,
            camera_props,
            record_args,
            trial_index,
            trial_seed,
            output_video,
            output_trajectory,
        )


if __name__ == "__main__":
    main()
