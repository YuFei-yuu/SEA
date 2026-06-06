# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
import os

from legged_gym import LEGGED_GYM_ROOT_DIR
import isaacgym
from isaacgym import gymapi
import cv2
import torch

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


DYNAMIC_TASKS = {
    "go2_pos_sparse_static",
    "go2_pos_dynamic_1",
    "go2_pos_dynamic_2",
    "go2_pos_dynamic_3",
}


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


def _apply_play_overrides(env_cfg, args):
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.replay.enable_collision_replay = False

    env_cfg.visualization.ray_groups = {
        "guidance_navigation": ["guide", "guide_ray_marker"],
    }

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

    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.max_push_vel_xy = 0.0
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.added_mass_range = [0.0, 0.0]
    env_cfg.env.episode_length_s = 40
    env_cfg.env.debug_viz = True
    if hasattr(env_cfg.env, "stay_time"):
        env_cfg.env.stay_time = 500
    env_cfg.asset.terminate_after_contacts_on = []


def _maybe_set_resume_fallback(train_cfg, args):
    if args.resume_experiment_name is not None:
        train_cfg.runner.resume_experiment_name = args.resume_experiment_name
        return
    if args.task not in DYNAMIC_TASKS:
        return
    if not _has_saved_runs(train_cfg.runner.experiment_name):
        train_cfg.runner.resume_experiment_name = "Go2_pos_rough"


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    _apply_play_overrides(env_cfg, args)

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    train_cfg.runner.resume = True
    train_cfg.runner.load_run = args.load_run if args.load_run is not None else -1
    train_cfg.runner.checkpoint = args.checkpoint if args.checkpoint is not None else -1
    _maybe_set_resume_fallback(train_cfg, args)

    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    print("Loaded policy from:", task_registry.loaded_policy_path)

    camera_props = gymapi.CameraProperties()
    camera_props.width = 1000
    camera_props.height = 1000
    camera_handle = env.gym.create_camera_sensor(env.envs[0], camera_props)
    env.gym.set_camera_location(
        camera_handle,
        env.envs[0],
        gymapi.Vec3(5.0, 5.0, 7.0),
        gymapi.Vec3(4.99, 5.0, 0.0),
    )

    record_video = False
    save_images = False
    total_episodes = 10
    video = None
    current_frame = 0
    max_frames = 20000

    obs, _ = env.reset()
    episode_count = 0

    with torch.no_grad():
        for _ in range(100 * int(env.max_episode_length)):
            actions = policy(obs.detach())
            obs, _, _, dones, infos = env.step(actions.detach())
            env.gym.set_camera_location(
                camera_handle,
                env.envs[0],
                gymapi.Vec3(5.0, 5.0, 7.0),
                gymapi.Vec3(4.99, 5.0, 0.0),
            )

            if dones.any():
                episode_count += 1
                episode_info = infos.get("episode", {})
                success = float(episode_info.get("success", 0.0))
                dyn_col = float(episode_info.get("dynamic_collision_count", 0.0))
                print(f"Episode {episode_count} finished | success={success:.2f} dynamic_collision_count={dyn_col:.2f}")

            if episode_count == total_episodes:
                print(f"Reached {total_episodes} episodes, stopping.")
                if video is not None:
                    video.release()
                break

            if (record_video or save_images) and current_frame < max_frames:
                env.gym.render_all_camera_sensors(env.sim)
                img = env.gym.get_camera_image(env.sim, env.envs[0], camera_handle, gymapi.IMAGE_COLOR)
                img = img.reshape((camera_props.height, camera_props.width, 4))[:, :, :3]
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                if record_video:
                    if video is None:
                        fps = 50
                        output_path = os.path.join(
                            LEGGED_GYM_ROOT_DIR,
                            "logs",
                            train_cfg.runner.experiment_name,
                            "exported",
                            f"{train_cfg.runner.load_run}_{train_cfg.runner.checkpoint}.mp4",
                        )
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (camera_props.width, camera_props.height))
                        print(f"Recording video to {output_path}")
                    video.write(img_bgr)

                if save_images:
                    img_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name, "exported", "frames")
                    os.makedirs(img_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(img_dir, f"frame_{current_frame:04d}.png"), img_bgr)

                current_frame += 1
                if current_frame % 100 == 0:
                    print(f"Recorded {current_frame}/{max_frames} frames")
            elif (record_video or save_images) and current_frame >= max_frames:
                if video is not None:
                    video.release()
                    video = None
                record_video = False
                save_images = False


if __name__ == "__main__":
    args = get_args()
    args.headless = False
    play(args)
