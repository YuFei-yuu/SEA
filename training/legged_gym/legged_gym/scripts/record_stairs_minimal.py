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
    parser.add_argument("--output_video", required=True)
    parser.add_argument("--output_trajectory", required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max_steps", type=int, default=None)
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return known, get_args()


def main():
    record_args, args = parse_args()
    if args.task != "go2_pos_stairs_minimal":
        raise ValueError("Use --task go2_pos_stairs_minimal")
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    env_cfg.env.num_envs = 1
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.depth_stairs.fixed_direction = 1 if record_args.direction == "up" else -1
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
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
    os.makedirs(os.path.dirname(os.path.abspath(record_args.output_video)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(record_args.output_trajectory)), exist_ok=True)
    video = cv2.VideoWriter(
        record_args.output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        record_args.fps,
        (camera_props.width, camera_props.height),
    )
    obs, _ = env.reset()
    trajectory = []
    max_steps = record_args.max_steps or int(env.max_episode_length)
    terminal_reason = "incomplete"
    with torch.no_grad():
        for step in range(max_steps):
            root = env.root_states[0, :3]
            room = env.room_origins[0]
            eye = gymapi.Vec3(
                float(root[0] - 3.0), float(room[1] - 3.5), float(root[2] + 2.2)
            )
            target = gymapi.Vec3(float(root[0] + 0.8), float(root[1]), float(root[2]))
            env.gym.set_camera_location(camera, env.envs[0], eye, target)
            obs, _, _, _, infos = env.step(policy(obs))
            xyz = env.root_states[0, :3].detach().cpu().tolist()
            trajectory.append({"step": step, "x": xyz[0], "y": xyz[1], "z": xyz[2]})
            env.gym.render_all_camera_sensors(env.sim)
            rgba = env.gym.get_camera_image(
                env.sim, env.envs[0], camera, gymapi.IMAGE_COLOR
            ).reshape(camera_props.height, camera_props.width, 4)
            frame = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
            cv2.putText(
                frame,
                f"direction: {record_args.direction}",
                (24, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            video.write(frame)
            outcomes = infos.get("episode_outcomes")
            if outcomes is not None:
                terminal_reason = TERMINAL_REASON_NAMES[int(outcomes["terminal_reason"][0])]
                break
    video.release()
    with open(record_args.output_trajectory, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("step", "x", "y", "z"))
        writer.writeheader()
        writer.writerows(trajectory)
    print(f"terminal_reason={terminal_reason}")
    print(f"Saved {record_args.output_video}")
    print(f"Saved {record_args.output_trajectory}")
    if terminal_reason != TERMINAL_REASON_NAMES[TERMINAL_SUCCESS]:
        raise RuntimeError(
            f"The recorded {record_args.direction} episode was not successful: {terminal_reason}"
        )


if __name__ == "__main__":
    main()
