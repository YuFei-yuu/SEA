"""Record a static fly-through of one depth-stair terrain room."""
from __future__ import annotations

import argparse
import os
import sys

from isaacgym import gymapi
import cv2
import numpy as np

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


def _parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output_video", required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frames_per_shot", type=int, default=120)
    parser.add_argument("--side_hold_frames", type=int, default=300)
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return known, get_args()


def _lerp(start, end, fraction):
    return start * (1.0 - fraction) + end * fraction


def _write_shot(
    env, camera, props, video, eye_start, eye_end, target_start, target_end, frames, label=""
):
    for frame in range(frames):
        fraction = frame / max(frames - 1, 1)
        eye = _lerp(eye_start, eye_end, fraction)
        target = _lerp(target_start, target_end, fraction)
        env.gym.set_camera_location(
            camera,
            env.envs[0],
            gymapi.Vec3(*eye),
            gymapi.Vec3(*target),
        )
        env.gym.step_graphics(env.sim)
        env.gym.render_all_camera_sensors(env.sim)
        rgba = env.gym.get_camera_image(
            env.sim, env.envs[0], camera, gymapi.IMAGE_COLOR
        ).reshape(props.height, props.width, 4)
        image = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
        if label:
            cv2.putText(
                image,
                label,
                (24, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        video.write(image)


def main():
    record_args, args = _parse_args()
    if args.task != "go2_pos_depth_stairs":
        raise ValueError("Use --task go2_pos_depth_stairs for terrain recording.")

    args.headless = True
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.perception.mode = "oracle"
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_xy = False
    env_cfg.domain_rand.randomize_yaw = False
    env_cfg.domain_rand.randomize_roll = False
    env_cfg.domain_rand.randomize_pitch = False
    env_cfg.asset.terminate_after_contacts_on = []

    os.makedirs(os.path.dirname(os.path.abspath(record_args.output_video)), exist_ok=True)
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.reset()

    props = gymapi.CameraProperties()
    props.width = 960
    props.height = 540
    camera = env.gym.create_camera_sensor(env.envs[0], props)
    video = cv2.VideoWriter(
        record_args.output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        record_args.fps,
        (props.width, props.height),
    )
    if not video.isOpened():
        raise RuntimeError(f"Could not open video output: {record_args.output_video}")

    origin = env.room_origins[0].detach().cpu().numpy()
    ox, oy = float(origin[0]), float(origin[1])
    frames = record_args.frames_per_shot

    # Establish the full room, scan the low-obstacle route, then inspect stairs and platform.
    _write_shot(
        env,
        camera,
        props,
        video,
        np.array([ox + 1.0, oy + 0.8, 8.0]),
        np.array([ox + 8.2, oy + 1.8, 6.6]),
        np.array([ox + 5.0, oy + 5.0, 0.0]),
        np.array([ox + 6.0, oy + 5.0, 0.2]),
        frames,
    )
    _write_shot(
        env,
        camera,
        props,
        video,
        np.array([ox + 0.7, oy + 1.7, 2.5]),
        np.array([ox + 4.0, oy + 2.2, 2.0]),
        np.array([ox + 1.8, oy + 5.0, 0.2]),
        np.array([ox + 5.1, oy + 5.0, 0.2]),
        frames,
    )
    _write_shot(
        env,
        camera,
        props,
        video,
        np.array([ox + 3.9, oy + 2.2, 1.7]),
        np.array([ox + 5.7, oy + 2.2, 1.9]),
        np.array([ox + 5.2, oy + 5.0, 0.2]),
        np.array([ox + 6.6, oy + 5.0, 0.4]),
        frames,
    )
    _write_shot(
        env,
        camera,
        props,
        video,
        np.array([ox + 5.8, oy + 2.4, 2.1]),
        np.array([ox + 7.7, oy + 4.0, 2.3]),
        np.array([ox + 6.3, oy + 5.0, 0.4]),
        np.array([ox + 8.3, oy + 5.0, 0.4]),
        frames,
    )
    _write_shot(
        env,
        camera,
        props,
        video,
        np.array([ox + 5.55, oy + 2.0, 1.0]),
        np.array([ox + 5.55, oy + 2.0, 1.0]),
        np.array([ox + 5.55, oy + 5.0, 0.20]),
        np.array([ox + 5.55, oy + 5.0, 0.20]),
        record_args.side_hold_frames,
        "SIDE VIEW: stair continuity check",
    )
    video.release()
    print(f"Saved {record_args.output_video}")


if __name__ == "__main__":
    main()
