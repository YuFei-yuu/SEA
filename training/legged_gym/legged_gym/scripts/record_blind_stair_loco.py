"""Record fixed-speed blind locomotion trials on the 0.08 m training stairs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys

import cv2
from isaacgym import gymapi
import torch

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from rsl_rl.modules import BlindLocomotionActorCritic


DEFAULT_SPEEDS = (0.25, 0.40, 0.55, 0.70)


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--direction", choices=("up", "down"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return known, get_args()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_policy(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy = BlindLocomotionActorCritic(
        num_actions=12,
        num_actor_obs=45,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
    ).to(device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()
    return policy.act_inference, int(checkpoint["iter"])


def configure_environment(env_cfg, direction):
    env_cfg.env.num_envs = 1
    env_cfg.env.disable_graphics = False
    env_cfg.env.episode_length_s = 20
    env_cfg.terrain.num_rows = 10
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.min_init_terrain_level = 6
    env_cfg.terrain.max_init_terrain_level = 6
    env_cfg.terrain.terrain_proportions = (
        [0.0, 1.0, 0.0] if direction == "up" else [0.0, 0.0, 1.0]
    )
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False


def put_label(frame, direction, speed, trial, step, local_x, z):
    lines = (
        f"model 2800 | {direction} | trial {trial}/4",
        f"command vx: {speed if direction == 'up' else -speed:+.2f} m/s",
        f"step: {step:04d} | local x: {local_x:+.2f} m | base z: {z:.2f} m",
    )
    overlay = frame.copy()
    cv2.rectangle(overlay, (14, 12), (590, 116), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.58, frame, 0.42, 0, frame)
    for line_index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (28, 42 + line_index * 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )


def reset_trial(env, env_id):
    env.do_reset = True
    env.reset_idx(env_id)
    env.do_reset = False
    env.compute_observations()
    return env.get_observations()


def record_trial(
    env,
    policy,
    camera,
    video_path,
    trajectory_path,
    direction,
    speed,
    trial,
    fps,
    max_steps,
    frame_size,
):
    env_id = torch.arange(1, device=env.device)
    obs = reset_trial(env, env_id)
    command_x = speed if direction == "up" else -speed
    direction_sign = 1.0 if direction == "up" else -1.0
    env.commands.zero_()
    env.commands[:, 0] = command_x
    env.compute_observations()
    obs = env.get_observations()

    origin = env.env_origins[0].detach().cpu().tolist()
    video = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        frame_size,
    )
    if not video.isOpened():
        raise RuntimeError(f"Could not open video output: {video_path}")

    trajectory = []
    crossing_hold = 0
    outcome = "max_steps"
    with torch.no_grad():
        for step in range(max_steps):
            env.commands.zero_()
            env.commands[:, 0] = command_x
            env.compute_observations()
            obs = env.get_observations()
            root_before_step = env.root_states[0, :3].detach().cpu().tolist()
            eye = gymapi.Vec3(
                float(root_before_step[0] - 1.1 * direction_sign),
                float(root_before_step[1] - 2.8),
                float(root_before_step[2] + 1.25),
            )
            target = gymapi.Vec3(
                float(root_before_step[0] + 0.25 * direction_sign),
                float(root_before_step[1]),
                float(root_before_step[2] - 0.08),
            )
            env.gym.set_camera_location(camera, env.envs[0], eye, target)
            actions = policy(obs)
            obs, _, _, dones, _ = env.step(actions)

            xyz = env.root_states[0, :3].detach().cpu().tolist()
            local_x = xyz[0] - origin[0]
            directed_x = direction_sign * local_x
            crossed = directed_x >= 3.15 and abs(xyz[2] - 0.29) <= 0.20
            crossing_hold = crossing_hold + 1 if crossed else 0
            trajectory.append(
                {
                    "step": step,
                    "time_s": step * env.dt,
                    "command_vx": command_x,
                    "x": xyz[0],
                    "y": xyz[1],
                    "z": xyz[2],
                    "local_x": local_x,
                    "crossed": int(crossed),
                }
            )

            env.gym.render_all_camera_sensors(env.sim)
            rgba = env.gym.get_camera_image(
                env.sim, env.envs[0], camera, gymapi.IMAGE_COLOR
            ).reshape(frame_size[1], frame_size[0], 4)
            frame = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
            put_label(frame, direction, speed, trial, step, local_x, xyz[2])
            video.write(frame)

            if crossing_hold >= 12:
                outcome = "crossed"
                break
            if bool(dones[0].item()):
                outcome = "timeout" if bool(env.time_out_buf[0].item()) else "fall_or_contact"
                break

    video.release()
    with open(trajectory_path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(trajectory[0]))
        writer.writeheader()
        writer.writerows(trajectory)

    directed_positions = [direction_sign * row["local_x"] for row in trajectory]
    return {
        "trial": trial,
        "direction": direction,
        "speed_mps": speed,
        "command_vx": command_x,
        "outcome": outcome,
        "frames": len(trajectory),
        "duration_s": len(trajectory) * env.dt,
        "max_directed_local_x": max(directed_positions),
        "final_local_x": trajectory[-1]["local_x"],
        "final_base_z": trajectory[-1]["z"],
        "video": os.path.abspath(video_path),
        "trajectory": os.path.abspath(trajectory_path),
    }


def main():
    record_args, args = parse_args()
    if args.task != "go2_blind_stair_loco":
        raise ValueError("Use --task go2_blind_stair_loco")
    checkpoint_path = os.path.abspath(record_args.checkpoint)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)
    os.makedirs(record_args.output_dir, exist_ok=True)

    env_cfg, _ = task_registry.get_cfgs(args.task)
    configure_environment(env_cfg, record_args.direction)
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=env_cfg)
    # Keep every recording on the fixed 0.08 m row.
    env.cfg.terrain.curriculum = False
    policy, checkpoint_iteration = load_policy(checkpoint_path, env.device)
    if checkpoint_iteration != 2800:
        raise ValueError(
            f"Expected checkpoint iteration 2800, got {checkpoint_iteration}"
        )

    camera_props = gymapi.CameraProperties()
    camera_props.width = record_args.width
    camera_props.height = record_args.height
    camera = env.gym.create_camera_sensor(env.envs[0], camera_props)
    if camera < 0:
        raise RuntimeError("Could not create the Isaac Gym review camera")

    results = []
    for trial, speed in enumerate(DEFAULT_SPEEDS, start=1):
        speed_tag = f"{int(round(speed * 100)):03d}"
        stem = f"{record_args.direction}_{trial:02d}_{speed_tag}cms"
        result = record_trial(
            env,
            policy,
            camera,
            os.path.join(record_args.output_dir, stem + ".mp4"),
            os.path.join(record_args.output_dir, stem + ".csv"),
            record_args.direction,
            speed,
            trial,
            record_args.fps,
            record_args.max_steps,
            (record_args.width, record_args.height),
        )
        results.append(result)
        print(json.dumps(result, indent=2))

    summary = {
        "task": args.task,
        "checkpoint": checkpoint_path,
        "checkpoint_iteration": checkpoint_iteration,
        "checkpoint_sha256": sha256(checkpoint_path),
        "terrain_step_height_m": 0.08,
        "terrain_tread_depth_m": 0.30,
        "direction": record_args.direction,
        "trials": results,
    }
    summary_path = os.path.join(
        record_args.output_dir, f"{record_args.direction}_summary.json"
    )
    with open(summary_path, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(f"Saved summary: {os.path.abspath(summary_path)}")


if __name__ == "__main__":
    main()
