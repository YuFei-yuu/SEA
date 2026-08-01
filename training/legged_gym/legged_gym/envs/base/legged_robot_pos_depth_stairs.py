"""Depth-camera stair and low-obstacle navigation task for Go2."""
from __future__ import annotations

import math
import os

import numpy as np
import torch

from isaacgym import gymtorch
from isaacgym.torch_utils import quat_rotate_inverse

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot_pos_dynamic import LeggedRobotPosDynamic
from legged_gym.depth_ray import load_depth_ray_model
from legged_gym.utils.torch_math import quat_apply_yaw, yaw_quat


class LeggedRobotPosDepthStairs(LeggedRobotPosDynamic):
    """Static stair task whose actor receives only depth-predicted forward rays."""

    def render(self, sync_frame_time=True):
        """Avoid a second camera render on every headless simulation step.

        ``_render_depth_and_predict`` explicitly renders cameras when the
        configured sensor update is due.  The generic headless render path
        would otherwise render all cameras again after every physics step,
        making large-environment depth PPO prohibitively slow.
        """
        if self.viewer is None:
            return
        super().render(sync_frame_time)

    def _stairs_cfg(self):
        return self.cfg.depth_stairs

    def _get_env_origins(self):
        self.custom_origins = True
        self.env_origins = torch.zeros(
            self.num_envs, 3, device=self.device, requires_grad=False
        )
        self.position_targets = torch.zeros(
            self.num_envs, 3, device=self.device, requires_grad=False
        )
        self.room_origins = torch.zeros(
            self.num_envs, 2, device=self.device, requires_grad=False
        )
        self.terrain_levels = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.terrain_types = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.goal_levels = torch.zeros(self.num_envs, device=self.device)
        self.max_terrain_level = self.cfg.terrain.num_rows
        self.ori_z = torch.zeros(self.num_envs, 1, device=self.device)
        self._resample_depth_stairs_navigation(
            torch.arange(self.num_envs, device=self.device)
        )

    def _resample_depth_stairs_navigation(self, env_ids):
        cfg = self._stairs_cfg()
        if len(env_ids) == 0:
            return
        rows = torch.randint(
            0, self.cfg.terrain.num_rows, (len(env_ids),), device=self.device
        )
        cols = torch.randint(
            0, self.cfg.terrain.num_cols, (len(env_ids),), device=self.device
        )
        self.terrain_levels[env_ids] = rows
        self.terrain_types[env_ids] = cols
        for local_idx, env_id in enumerate(env_ids.tolist()):
            room_origin_x = rows[local_idx].item() * self.terrain.env_length
            room_origin_y = cols[local_idx].item() * self.terrain.env_width
            self.room_origins[env_id] = torch.tensor(
                [room_origin_x, room_origin_y], device=self.device
            )
            start_x = float(np.random.uniform(*cfg.start_x_range))
            start_y = float(np.random.uniform(*cfg.start_y_range))
            goal_x = float(np.random.uniform(*cfg.goal_x_range))
            goal_y = float(np.random.uniform(*cfg.goal_y_range))
            self.env_origins[env_id, :2] = torch.tensor(
                [room_origin_x + start_x, room_origin_y + start_y],
                device=self.device,
            )
            self.position_targets[env_id] = torch.tensor(
                [
                    room_origin_x + goal_x,
                    room_origin_y + goal_y,
                    cfg.platform_height + 0.50,
                ],
                device=self.device,
            )

    def _init_buffers(self):
        super()._init_buffers()
        cfg = self._stairs_cfg()
        perception_cfg = self.cfg.perception
        self.perception_mode = str(perception_cfg.mode)
        if self.perception_mode not in {"oracle", "depth_predicted"}:
            raise ValueError("perception.mode must be 'oracle' or 'depth_predicted'.")
        if (
            self.perception_mode == "depth_predicted"
            and not getattr(self, "depth_camera_enabled", False)
        ):
            raise RuntimeError(
                "depth_predicted mode requires sensors.depth_cam.enable=True"
            )

        self.oracle_rays = self.rays.clone()
        self.depth_rays = self.rays.clone()
        self.depth_ray_mae = torch.zeros(self.num_envs, device=self.device)
        self.depth_ray_error_sum = torch.zeros(self.num_envs, device=self.device)
        self.depth_ray_error_count = torch.zeros(self.num_envs, device=self.device)
        self.last_depth_render_step = -1
        camera_update_hz = float(perception_cfg.update_hz)
        if self.perception_mode == "oracle":
            camera_update_hz = float(perception_cfg.oracle_camera_update_hz)
        self.camera_update_steps = max(1, int(round(1.0 / (camera_update_hz * self.dt))))
        self.depth_ray_model = None
        if self.perception_mode == "depth_predicted":
            model_path = str(perception_cfg.model_path).format(
                LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR
            )
            self.depth_ray_model = load_depth_ray_model(
                model_path, self.device, self.ray_angles.shape[0]
            )

        self.low_obstacle_boxes = torch.tensor(
            self.cfg.terrain.low_obstacle_boxes,
            dtype=torch.float,
            device=self.device,
        )
        self.low_obstacle_collision_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.low_obstacle_collision_event = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.low_obstacle_collision_cooldown = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.stair_passed = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    def _render_depth_and_predict(self):
        if not getattr(self, "depth_camera_enabled", False):
            return
        if self.common_step_counter % self.camera_update_steps != 0:
            return

        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        for env_id, camera_tensor in enumerate(self.camera_tensors):
            self.cam_obs[env_id].copy_(
                (-camera_tensor).clip(
                    min=self.cfg.sensors.depth_cam.min_,
                    max=self.cfg.sensors.depth_cam.max_,
                )
            )
        self.gym.end_access_image_tensors(self.sim)
        self.last_depth_render_step = self.common_step_counter

        if self.depth_ray_model is None:
            return
        with torch.no_grad():
            depth_input = torch.log2(
                self.cam_obs.clamp(
                    min=self.cfg.sensors.depth_cam.min_,
                    max=self.cfg.sensors.depth_cam.max_,
                )
            ).unsqueeze(1)
            self.depth_rays.copy_(
                torch.exp2(self.depth_ray_model(depth_input)).clip(
                    min=self.cfg.sensors.ray2d.min_dist,
                    max=self.cfg.sensors.ray2d.max_dist,
                )
            )

    def update_percetion(self):
        self._render_depth_and_predict()
        super().update_percetion()

    def _get_rays(self, env_ids=None):
        if env_ids is not None:
            raise NotImplementedError("Depth-stair ray queries are batched over all envs.")

        points = quat_apply_yaw(
            self.base_quat.repeat(1, self.num_height_points), self.height_points
        ) + self.root_states[:, :3].unsqueeze(1)
        points += self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].reshape(-1).clip(0, self.height_samples.shape[0] - 2)
        py = points[:, :, 1].reshape(-1).clip(0, self.height_samples.shape[1] - 2)
        heights = torch.maximum(
            self.height_samples[px, py], self.height_samples[px + 1, py]
        )
        self.measured_heights = heights.view(
            self.num_envs, self.len_x, self.len_y
        ) * self.terrain.cfg.vertical_scale
        center_height = self.measured_heights[:, self.c_x, self.c_y].view(
            self.num_envs, 1, 1
        )
        obstacle_mask = self.measured_heights > (
            center_height + self.cfg.perception.obstacle_height_threshold
        )
        self.static_rays = self._grid2ray(obstacle_mask.float()) * (
            self.cfg.terrain.measure_resolution
        )
        self.oracle_rays.copy_(self.static_rays)
        if self.perception_mode == "oracle":
            self.rays.copy_(self.oracle_rays)
        else:
            self.rays.copy_(self.depth_rays)
        self.dynamic_rays.fill_(self.cfg.sensors.ray2d.max_dist)
        self.predicted_dynamic_rays.copy_(self.dynamic_rays)

    def _inside_low_obstacle(self):
        local_xy = self.root_states[:, :2] - self.room_origins
        inflation = float(self._stairs_cfg().collision_inflation)
        inside = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for box in self.low_obstacle_boxes:
            center_x, center_y, size_x, size_y, _ = box
            inside |= (
                (torch.abs(local_xy[:, 0] - center_x) <= size_x * 0.5 + inflation)
                & (torch.abs(local_xy[:, 1] - center_y) <= size_y * 0.5 + inflation)
            )
        return inside

    def check_termination(self):
        super().check_termination()
        cfg = self._stairs_cfg()
        local_x = self.root_states[:, 0] - self.room_origins[:, 0]
        platform_reached = (
            (local_x >= cfg.platform_start_x)
            & (
                self.root_states[:, 2]
                >= self.base_init_state[2] + cfg.platform_height - 0.12
            )
        )
        self.stair_passed |= platform_reached

        self.low_obstacle_collision_cooldown = torch.clamp(
            self.low_obstacle_collision_cooldown - 1, min=0
        )
        collision_now = self.body_collision_event & self._inside_low_obstacle()
        self.low_obstacle_collision_event = collision_now & (
            self.low_obstacle_collision_cooldown == 0
        )
        self.low_obstacle_collision_count += self.low_obstacle_collision_event.long()
        self.low_obstacle_collision_cooldown[
            self.low_obstacle_collision_event
        ] = int(cfg.collision_cooldown_steps)

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        low_collision_count = self.low_obstacle_collision_count[env_ids].float().clone()
        stair_passed = self.stair_passed[env_ids].float().clone()
        depth_ray_mae = self.depth_ray_mae[env_ids].clone()
        success = self.goal_reached_flag[env_ids].float().clone()
        self._resample_depth_stairs_navigation(env_ids)

        super().reset_idx(env_ids)
        episode = self.extras["episode"]
        episode["low_obstacle_collision_count"] = torch.mean(low_collision_count)
        episode["stair_pass_rate"] = torch.mean(stair_passed)
        episode["depth_ray_mae"] = torch.mean(depth_ray_mae)
        episode["depth_safe_success"] = torch.mean(
            success * (low_collision_count == 0).float() * stair_passed
        )

        self.low_obstacle_collision_count[env_ids] = 0
        self.low_obstacle_collision_event[env_ids] = False
        self.low_obstacle_collision_cooldown[env_ids] = 0
        self.stair_passed[env_ids] = False
        self.depth_ray_mae[env_ids] = 0.0
        self.depth_ray_error_sum[env_ids] = 0.0
        self.depth_ray_error_count[env_ids] = 0.0

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        if self.perception_mode == "depth_predicted":
            ray_error = torch.mean(torch.abs(self.depth_rays - self.oracle_rays), dim=1)
            self.depth_ray_error_sum += ray_error
            self.depth_ray_error_count += 1.0
            self.depth_ray_mae = self.depth_ray_error_sum / self.depth_ray_error_count.clamp(
                min=1.0
            )
