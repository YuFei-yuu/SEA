"""Record Go2 depth-stair navigation with side-by-side external and depth-camera views."""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
from isaacgym import gymapi
import torch

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


def _parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output_video", required=True)
    parser.add_argument("--output_trajectory", required=True)
    parser.add_argument("--num_episodes", type=int, default=3)
    parser.add_argument("--fps", type=int, default=30)
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return known, get_args()


def _depth_panel(depth, min_depth, max_depth, target_height):
    normalized = 1.0 - (depth - min_depth) / (max_depth - min_depth)
    image = np.uint8(np.clip(normalized, 0.0, 1.0) * 255.0)
    panel = cv2.applyColorMap(image, cv2.COLORMAP_TURBO)
    width = int(round(panel.shape[1] * target_height / panel.shape[0]))
    return cv2.resize(panel, (width, target_height), interpolation=cv2.INTER_NEAREST)


def main():
    record_args, args = _parse_args()
    if args.task != "go2_pos_depth_stairs":
        raise ValueError("Use --task go2_pos_depth_stairs for this recorder.")
    if args.depth_mode not in (None, "depth_predicted") or not args.depth_model:
        raise ValueError("Recording requires --depth_mode depth_predicted --depth_model <best.pt>.")

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.asset.terminate_after_contacts_on = []
    train_cfg.runner.resume = True
    train_cfg.runner.load_run = args.load_run if args.load_run is not None else -1
    train_cfg.runner.checkpoint = args.checkpoint if args.checkpoint is not None else -1

    os.makedirs(os.path.dirname(os.path.abspath(record_args.output_video)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(record_args.output_trajectory)), exist_ok=True)
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    runner, _ = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = runner.get_inference_policy(device=env.device)

    camera_props = gymapi.CameraProperties()
    camera_props.width = 960
    camera_props.height = 540
    external_camera = env.gym.create_camera_sensor(env.envs[0], camera_props)
    panel_width = int(round(camera_props.height * 160 / 90))
    video = cv2.VideoWriter(
        record_args.output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        record_args.fps,
        (camera_props.width + panel_width, camera_props.height),
    )

    obs, _ = env.reset()
    trajectory = []
    episodes = 0
    max_steps = int(env.max_episode_length) * 100
    with torch.no_grad():
        for _ in range(max_steps):
            root = env.root_states[0, :3].detach().cpu().numpy()
            eye = gymapi.Vec3(float(root[0] - 3.5), float(root[1] - 3.5), float(root[2] + 2.4))
            target = gymapi.Vec3(float(root[0]), float(root[1]), float(root[2] + 0.30))
            env.gym.set_camera_location(external_camera, env.envs[0], eye, target)

            action = policy(obs.detach())
            obs, _, _, dones, infos = env.step(action.detach())
            trajectory.append(env.root_states[0, :3].detach().cpu().numpy().copy())

            env.gym.render_all_camera_sensors(env.sim)
            rgba = env.gym.get_camera_image(
                env.sim, env.envs[0], external_camera, gymapi.IMAGE_COLOR
            ).reshape(camera_props.height, camera_props.width, 4)
            external = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
            depth = env.cam_obs[0].detach().cpu().numpy()
            panel = _depth_panel(
                depth,
                env_cfg.sensors.depth_cam.min_,
                env_cfg.sensors.depth_cam.max_,
                camera_props.height,
            )
            mae = float(env.depth_ray_mae[0].detach().cpu().item())
            cv2.putText(
                external,
                f"depth-ray MAE: {mae:.3f} m",
                (24, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            video.write(np.concatenate([external, panel], axis=1))

            if dones.any():
                episodes += 1
                info = infos.get("episode", {})
                print(
                    f"episode={episodes} success={float(info.get('success', 0.0)):.0f} "
                    f"stair={float(info.get('stair_pass_rate', 0.0)):.0f}"
                )
                if episodes >= record_args.num_episodes:
                    break

    video.release()
    np.savetxt(
        record_args.output_trajectory,
        np.asarray(trajectory),
        delimiter=",",
        header="x,y,z",
        comments="",
    )
    print(f"Saved {record_args.output_video}")
    print(f"Saved {record_args.output_trajectory}")


if __name__ == "__main__":
    main()
