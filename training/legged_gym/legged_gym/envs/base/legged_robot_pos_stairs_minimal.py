"""Bidirectional stair navigation with known-map 2-D footprint rays."""

from __future__ import annotations

import math

import torch
from isaacgym.torch_utils import quat_apply, quat_from_euler_xyz, quat_rotate_inverse

from legged_gym.envs.base.legged_robot_pos_dynamic import LeggedRobotPosDynamic
from legged_gym.utils.fixed_room_planner import build_bidirectional_route_templates
from legged_gym.utils.footprint_rays import ray_aabb_distances, room_footprint_boxes
from legged_gym.utils.torch_math import quat_apply_yaw, yaw_quat


TERMINAL_NONE = 0
TERMINAL_SUCCESS = 1
TERMINAL_STAIR_STUCK = 2
TERMINAL_FALL_OR_CONTACT = 3
TERMINAL_TIMEOUT = 4
TERMINAL_OTHER_STUCK = 5

TERMINAL_REASON_NAMES = {
    TERMINAL_NONE: "ongoing",
    TERMINAL_SUCCESS: "success",
    TERMINAL_STAIR_STUCK: "stair_stuck",
    TERMINAL_FALL_OR_CONTACT: "fall_or_contact",
    TERMINAL_TIMEOUT: "timeout",
    TERMINAL_OTHER_STUCK: "other_stuck",
}

COLLISION_CLASS_NAMES = {
    0: "none",
    1: "low_obstacle",
    2: "wall",
    3: "stair",
}


def deterministic_uniform_from_seed(seeds, salt, low, high):
    """Map integer episode seeds to deterministic uniform values on any device."""
    state = torch.remainder(
        seeds.long() * 1_103_515_245 + 12_345 + int(salt) * 2_654_435_761,
        2_147_483_647,
    )
    unit = state.to(torch.float64) / 2_147_483_647.0
    return (float(low) + (float(high) - float(low)) * unit).to(torch.float32)


def update_consecutive_timer(timer, condition):
    """Increment while a condition stays true and reset immediately otherwise."""
    return torch.where(condition, timer + 1, torch.zeros_like(timer))


def stair_success_eligible(distance, correct_height, stair_crossed, distance_threshold):
    return (distance < distance_threshold) & correct_height & stair_crossed


def stair_fully_cleared(
    base_x,
    feet_x,
    up,
    stair_start_x,
    stair_end_x,
    base_clearance,
    foot_margin,
):
    """Require the base and all four feet to enter the destination deck."""
    up_clear = (base_x >= stair_end_x + base_clearance) & torch.all(
        feet_x >= stair_end_x + foot_margin, dim=1
    )
    down_clear = (base_x <= stair_start_x - base_clearance) & torch.all(
        feet_x <= stair_start_x - foot_margin, dim=1
    )
    return torch.where(up, up_clear, down_clear)


def timeout_reached(episode_length, max_episode_length):
    return episode_length >= max_episode_length


def stair_progress_is_stuck(
    in_stair, progress_steps, directed_progress, window_steps, min_progress
):
    return (
        in_stair
        & (progress_steps >= window_steps)
        & (directed_progress < min_progress)
    )


def exclusive_terminal_masks(
    success_candidate,
    stair_stuck_candidate,
    fall_or_contact_candidate,
    timeout_candidate,
    other_stuck_candidate,
):
    """Apply deterministic failure-first precedence to terminal candidates."""
    fall = fall_or_contact_candidate
    stair_stuck = stair_stuck_candidate & ~fall
    timeout = timeout_candidate & ~fall & ~stair_stuck
    other_stuck = other_stuck_candidate & ~fall & ~stair_stuck & ~timeout
    success = success_candidate & ~fall & ~stair_stuck & ~timeout & ~other_stuck
    return success, stair_stuck, fall, timeout, other_stuck


class LeggedRobotPosStairsMinimal(LeggedRobotPosDynamic):
    """Bidirectional navigation over a known room with footprint rays."""

    def _get_env_origins(self):
        self.custom_origins = True
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device)
        self.position_targets = torch.zeros(self.num_envs, 3, device=self.device)
        self.room_origins = torch.zeros(self.num_envs, 2, device=self.device)
        self.terrain_levels = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.terrain_types = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.goal_levels = torch.zeros(self.num_envs, device=self.device)
        self.max_terrain_level = self.cfg.terrain.num_rows
        self.ori_z = torch.zeros(self.num_envs, 1, device=self.device)
        self.navigation_direction = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        self.episode_seed = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.eval_seed_round = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.direction_reset_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._resample_sparse_navigation(torch.arange(self.num_envs, device=self.device))

    def _resample_sparse_navigation(self, env_ids):
        if len(env_ids) == 0:
            return
        cfg = self.cfg.depth_stairs
        count = len(env_ids)
        fixed_direction = int(getattr(cfg, "fixed_direction", 0))
        eval_seed_base = int(getattr(cfg, "eval_seed_base", -1))
        seeded_eval = fixed_direction != 0 and eval_seed_base >= 0
        if seeded_eval:
            seeds = eval_seed_base + env_ids.long() + self.eval_seed_round[env_ids] * self.num_envs
            self.eval_seed_round[env_ids] += 1
            self.episode_seed[env_ids] = seeds
            rows = torch.remainder(seeds, self.cfg.terrain.num_rows)
            cols = torch.remainder(seeds // self.cfg.terrain.num_rows, self.cfg.terrain.num_cols)
        else:
            seeds = None
            self.episode_seed[env_ids] = -1
            rows = torch.randint(0, self.cfg.terrain.num_rows, (count,), device=self.device)
            cols = torch.randint(0, self.cfg.terrain.num_cols, (count,), device=self.device)
        if fixed_direction == 0:
            parity = torch.remainder(
                env_ids.long() + self.direction_reset_count[env_ids], 2
            )
            direction = torch.where(
                parity == 0, torch.ones_like(parity), -torch.ones_like(parity)
            )
            self.direction_reset_count[env_ids] += 1
        else:
            direction = torch.full(
                (count,), 1 if fixed_direction > 0 else -1, dtype=torch.long, device=self.device
            )

        room_x = rows.float() * self.terrain.env_length
        room_y = cols.float() * self.terrain.env_width
        if seeded_eval:
            start_y = deterministic_uniform_from_seed(seeds, 1, *cfg.start_y_range)
            goal_y = deterministic_uniform_from_seed(seeds, 2, *cfg.goal_y_range)
            up_start_x = deterministic_uniform_from_seed(
                seeds, 3, *getattr(cfg, "up_start_x_range", cfg.start_x_range)
            )
            up_goal_x = deterministic_uniform_from_seed(
                seeds, 4, *getattr(cfg, "up_goal_x_range", cfg.goal_x_range)
            )
            down_start_x = deterministic_uniform_from_seed(
                seeds, 5, *getattr(cfg, "down_start_x_range", cfg.goal_x_range)
            )
            down_goal_x = deterministic_uniform_from_seed(
                seeds, 6, *getattr(cfg, "down_goal_x_range", cfg.start_x_range)
            )
        else:
            start_y = torch.empty(count, device=self.device).uniform_(*cfg.start_y_range)
            goal_y = torch.empty(count, device=self.device).uniform_(*cfg.goal_y_range)
            up_start_x = torch.empty(count, device=self.device).uniform_(
                *getattr(cfg, "up_start_x_range", cfg.start_x_range)
            )
            up_goal_x = torch.empty(count, device=self.device).uniform_(
                *getattr(cfg, "up_goal_x_range", cfg.goal_x_range)
            )
            down_start_x = torch.empty(count, device=self.device).uniform_(
                *getattr(cfg, "down_start_x_range", cfg.goal_x_range)
            )
            down_goal_x = torch.empty(count, device=self.device).uniform_(
                *getattr(cfg, "down_goal_x_range", cfg.start_x_range)
            )
        up = direction > 0
        start_x = torch.where(up, up_start_x, down_start_x)
        goal_x = torch.where(up, up_goal_x, down_goal_x)
        start_surface_z = torch.where(
            up,
            torch.zeros(count, device=self.device),
            torch.full((count,), float(cfg.platform_height), device=self.device),
        )
        goal_surface_z = torch.where(
            up,
            torch.full((count,), float(cfg.platform_height), device=self.device),
            torch.zeros(count, device=self.device),
        )

        self.terrain_levels[env_ids] = rows
        self.terrain_types[env_ids] = cols
        self.navigation_direction[env_ids] = direction
        self.room_origins[env_ids, 0] = room_x
        self.room_origins[env_ids, 1] = room_y
        self.env_origins[env_ids, 0] = room_x + start_x
        self.env_origins[env_ids, 1] = room_y + start_y
        self.env_origins[env_ids, 2] = start_surface_z
        self.position_targets[env_ids, 0] = room_x + goal_x
        self.position_targets[env_ids, 1] = room_y + goal_y
        self.position_targets[env_ids, 2] = self.base_init_state[2] + goal_surface_z
        if not hasattr(self, "episode_start_local_y"):
            self.episode_start_local_y = torch.zeros(self.num_envs, device=self.device)
        self.episode_start_local_y[env_ids] = start_y
        if hasattr(self, "route_templates"):
            self._assign_navigation_routes(env_ids)

    def _init_buffers(self):
        super()._init_buffers()
        self.footprint_inflation = float(getattr(self.cfg.depth_stairs, "footprint_inflation", 0.15))
        self.footprint_centers, self.footprint_half_extents = room_footprint_boxes(
            self.cfg.terrain.low_obstacle_boxes,
            room_size=float(self.terrain.env_length),
            wall_thickness=float(getattr(self.cfg.terrain, "boundary_wall_thickness", 0.30)),
            inflation=self.footprint_inflation,
            device=self.device,
        )
        low_boxes = torch.tensor(
            self.cfg.terrain.low_obstacle_boxes,
            dtype=torch.float32,
            device=self.device,
        )
        self.low_obstacle_centers = low_boxes[:, :2]
        self.low_obstacle_half_extents = low_boxes[:, 2:4] * 0.5
        self.low_obstacle_heights = low_boxes[:, 4]
        self.low_obstacle_collision_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.wall_contact_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.stair_contact_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.low_obstacle_collision_cooldown = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.last_low_obstacle_contact = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.last_wall_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_stair_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.collision_class_code = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.low_obstacle_collision_index = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.stair_crossed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.fully_cleared = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.stair_progress_anchor = self.root_states[:, 0].clone()
        self.stair_progress_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.terminal_reason = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.correct_goal_height = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        cfg = self.cfg.depth_stairs
        templates, lengths, start_values, _ = build_bidirectional_route_templates(
            self.cfg.terrain.low_obstacle_boxes,
            start_y_range=cfg.start_y_range,
            route_bins=int(cfg.route_bins),
            low_start_x=float(cfg.up_start_x_range[0]),
            high_start_x=float(cfg.down_start_x_range[0]),
            up_goal=(float(cfg.up_goal_x_range[0]), float(cfg.goal_y_range[0])),
            down_goal=(float(cfg.down_goal_x_range[0]), float(cfg.goal_y_range[0])),
            low_staging=tuple(cfg.route_low_staging),
            high_staging=tuple(cfg.route_high_staging),
            room_size=float(self.terrain.env_length),
            resolution=float(cfg.route_grid_resolution),
            obstacle_inflation=float(cfg.route_obstacle_inflation),
            boundary_margin=float(cfg.route_boundary_margin),
        )
        self.route_templates = torch.tensor(templates, device=self.device)
        self.route_template_lengths = torch.tensor(lengths, dtype=torch.long, device=self.device)
        self.route_start_values = torch.tensor(start_values, device=self.device)
        self.navigation_route = torch.zeros(
            self.num_envs, templates.shape[2], 2, device=self.device
        )
        self.navigation_route_length = torch.ones(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.navigation_route_progress = torch.ones(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._assign_navigation_routes(torch.arange(self.num_envs, device=self.device))

    def _assign_navigation_routes(self, env_ids):
        if len(env_ids) == 0:
            return
        start_min = self.route_start_values[0]
        bin_spacing = self.route_start_values[1] - self.route_start_values[0]
        bins = torch.round(
            (self.episode_start_local_y[env_ids] - start_min) / bin_spacing
        ).long().clamp(0, self.route_start_values.numel() - 1)
        directions = torch.where(
            self.navigation_direction[env_ids] > 0,
            torch.zeros_like(bins),
            torch.ones_like(bins),
        )
        self.navigation_route[env_ids] = self.route_templates[directions, bins]
        self.navigation_route_length[env_ids] = self.route_template_lengths[directions, bins]
        self.navigation_route_progress[env_ids] = torch.minimum(
            torch.ones_like(bins), self.navigation_route_length[env_ids] - 1
        )

    def _reset_root_states(self, env_ids):
        super()._reset_root_states(env_ids)
        if len(env_ids) == 0:
            return
        zeros = torch.zeros(len(env_ids), device=self.device)
        yaw = torch.where(
            self.navigation_direction[env_ids] > 0,
            zeros,
            torch.full_like(zeros, torch.pi),
        )
        self.root_states[env_ids, 3:7] = quat_from_euler_xyz(zeros, zeros, yaw)
        self.base_quat[env_ids] = self.root_states[env_ids, 3:7]
        self._set_robot_root_states(env_ids)

    def _get_rays(self, env_ids=None):
        if env_ids is None:
            configured_ids = getattr(self, "perception_env_ids", None)
            ids = (
                torch.arange(self.num_envs, device=self.device)
                if configured_ids is None
                else configured_ids
            )
        else:
            ids = env_ids
        if ids.numel() == 0:
            return
        angles = self.ray_angles.to(self.device)
        local_directions = torch.stack(
            (torch.cos(angles), torch.sin(angles)), dim=-1
        ).unsqueeze(0).expand(ids.numel(), -1, -1)
        yaw = self.base_quat[ids].unsqueeze(1).expand(-1, angles.numel(), -1)
        local_directions_3d = torch.cat(
            (local_directions, torch.zeros_like(local_directions[..., :1])), dim=-1
        )
        world_directions = quat_apply_yaw(yaw.reshape(-1, 4), local_directions_3d.reshape(-1, 3))
        world_directions = world_directions[:, :2].view(ids.numel(), angles.numel(), 2)
        # Box centers are room-local; translate them to each room before querying.
        translated_centers = self.room_origins[ids, None, :] + self.footprint_centers[None, :, :]
        distances = ray_aabb_distances(
            self.root_states[ids, :2],
            world_directions,
            translated_centers,
            self.footprint_half_extents[None, :, :].expand(ids.numel(), -1, -1),
            float(self.cfg.sensors.ray2d.min_dist),
            float(self.cfg.sensors.ray2d.max_dist),
        )
        ray_values = distances.min(dim=-1).values
        self.rays[ids] = ray_values
        self.static_rays[ids] = ray_values
        self.dynamic_rays[ids] = float(self.cfg.sensors.ray2d.max_dist)
        self.predicted_dynamic_rays[ids] = float(self.cfg.sensors.ray2d.max_dist)

    def _check_spawn_collision(self):
        # The configured start intervals are before/after the obstacle field.
        return

    def _current_navigation_waypoint(self):
        batch = torch.arange(self.num_envs, device=self.device)
        local_xy = self.root_states[:, :2] - self.room_origins
        progress = torch.minimum(
            self.navigation_route_progress, self.navigation_route_length - 1
        )
        waypoint = self.navigation_route[batch, progress]
        waypoint_distance = torch.linalg.vector_norm(waypoint - local_xy, dim=-1)
        advance = (
            (waypoint_distance < float(self.cfg.depth_stairs.route_waypoint_tolerance))
            & (progress < self.navigation_route_length - 1)
        )
        progress = torch.where(advance, progress + 1, progress)
        self.navigation_route_progress[:] = progress
        return self.navigation_route[batch, progress]

    def _get_perception(self):
        """Keep the 350-D layout but guide it with the fixed map's local waypoint."""
        self.rays_rand = self.rays.clone()
        self.rays_hist = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            torch.stack([self.rays_rand] * self.cfg.env.his_len, dim=1),
            torch.cat([self.rays_hist[:, 1:], self.rays_rand.unsqueeze(1)], dim=1),
        )
        waypoint_world = self._current_navigation_waypoint() + self.room_origins
        pos_diff = torch.zeros(self.num_envs, 3, device=self.device)
        pos_diff[:, :2] = waypoint_world - self.root_states[:, :2]
        self.goal_local_pos = quat_rotate_inverse(
            yaw_quat(self.base_quat), pos_diff
        )[:, :2]
        self.goal_hist = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            torch.stack([self.goal_local_pos] * self.cfg.env.his_len, dim=1),
            torch.cat([self.goal_hist[:, 1:], self.goal_local_pos.unsqueeze(1)], dim=1),
        )

    def get_navigation_teacher_actions(self):
        """Follow precomputed traversable routes with simultaneous vx/vy/yaw."""
        cfg = self.cfg.depth_stairs
        local_xy = self.root_states[:, :2] - self.room_origins
        up = self.navigation_direction > 0
        x = local_xy[:, 0]
        waypoint = self._current_navigation_waypoint()
        target_vector = waypoint - local_xy
        target_distance = torch.linalg.vector_norm(target_vector, dim=-1).clamp(min=1.0e-5)
        path_direction = target_vector / target_distance.unsqueeze(-1)

        in_obstacle_field = (local_xy[:, 0] >= 1.15) & (local_xy[:, 0] <= 4.50)
        target_speed = torch.where(
            in_obstacle_field,
            torch.full_like(x, float(cfg.teacher_obstacle_speed)),
            torch.full_like(x, float(cfg.teacher_speed)),
        )
        target_speed = torch.where(
            torch.abs(path_direction[:, 1]) > 0.15,
            torch.full_like(target_speed, float(cfg.teacher_diagonal_speed)),
            target_speed,
        )
        yaw = torch.atan2(
            2.0
            * (
                self.base_quat[:, 3] * self.base_quat[:, 2]
                + self.base_quat[:, 0] * self.base_quat[:, 1]
            ),
            1.0
            - 2.0
            * (
                self.base_quat[:, 1] * self.base_quat[:, 1]
                + self.base_quat[:, 2] * self.base_quat[:, 2]
            ),
        )
        distance = torch.norm(self.position_targets[:, :2] - self.root_states[:, :2], dim=-1)
        forward_yaw = torch.where(up, torch.zeros_like(yaw), torch.full_like(yaw, torch.pi))
        path_yaw = torch.atan2(path_direction[:, 1], path_direction[:, 0])
        blend = float(cfg.teacher_heading_path_blend)
        heading_x = (1.0 - blend) * torch.cos(forward_yaw) + blend * torch.cos(path_yaw)
        heading_y = (1.0 - blend) * torch.sin(forward_yaw) + blend * torch.sin(path_yaw)
        desired_yaw = torch.atan2(heading_y, heading_x)
        stair_crossing = (x >= 4.35) & (x <= 7.15)
        stair_blend = float(cfg.teacher_stair_heading_path_blend)
        stair_heading_x = (
            (1.0 - stair_blend) * torch.cos(forward_yaw)
            + stair_blend * torch.cos(path_yaw)
        )
        stair_heading_y = (
            (1.0 - stair_blend) * torch.sin(forward_yaw)
            + stair_blend * torch.sin(path_yaw)
        )
        stair_yaw = torch.atan2(stair_heading_y, stair_heading_x)
        desired_yaw = torch.where(stair_crossing, stair_yaw, desired_yaw)
        desired_yaw = torch.where(distance < 0.50, forward_yaw, desired_yaw)
        yaw_error = torch.atan2(torch.sin(desired_yaw - yaw), torch.cos(desired_yaw - yaw))
        yaw_rate = (float(cfg.teacher_heading_gain) * yaw_error).clip(
            min=-float(cfg.teacher_max_yaw_rate), max=float(cfg.teacher_max_yaw_rate)
        )
        target_speed = torch.where(
            stair_crossing,
            torch.full_like(target_speed, float(cfg.teacher_stair_speed)),
            target_speed,
        )
        desired_world_velocity = path_direction * target_speed.unsqueeze(-1)
        body_velocity = quat_apply_yaw(
            torch.cat((-self.base_quat[:, :3], self.base_quat[:, 3:4]), dim=-1),
            torch.cat(
                (desired_world_velocity, torch.zeros_like(desired_world_velocity[:, :1])),
                dim=-1,
            ),
        )[:, :2]
        body_velocity[:, 0].clamp_(min=0.0, max=float(cfg.teacher_speed))
        body_velocity[:, 1].clamp_(
            min=-float(cfg.teacher_max_lateral_speed),
            max=float(cfg.teacher_max_lateral_speed),
        )
        body_velocity[:, 1] = torch.where(
            stair_crossing,
            body_velocity[:, 1].clamp(
                min=-float(cfg.teacher_stair_max_lateral_speed),
                max=float(cfg.teacher_stair_max_lateral_speed),
            ),
            body_velocity[:, 1],
        )
        actions = torch.cat((body_velocity, yaw_rate.unsqueeze(-1)), dim=-1)
        stop_scale = (distance / 0.60).clip(min=0.0, max=1.0)
        stop_scale = torch.where(
            self.fully_cleared, stop_scale, torch.ones_like(stop_scale)
        )
        actions[:, :2] *= stop_scale.unsqueeze(-1)
        actions[:, 2] *= (distance > 0.20).float()
        return actions

    def _update_static_collision_metrics(self):
        """Classify static contacts geometrically because PhysX has no terrain IDs."""
        self.low_obstacle_collision_cooldown.sub_(1).clamp_(min=0)
        if self.penalised_contact_indices.numel() > 0:
            body_contact = torch.any(
                torch.linalg.vector_norm(
                    self.contact_forces[:, self.penalised_contact_indices, :], dim=-1
                ) > 1.0,
                dim=1,
            )
        else:
            body_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        foot_contact = self.contact_forces[:, self.feet_indices, 2] > 1.0

        root_xy = self.root_states[:, :2] - self.room_origins
        feet_xy = self.feet_pos[:, :, :2] - self.room_origins[:, None, :]
        root_box_delta = torch.abs(
            root_xy[:, None, :] - self.low_obstacle_centers[None, :, :]
        )
        root_in_boxes = torch.all(
            root_box_delta
            <= self.low_obstacle_half_extents[None, :, :] + self.footprint_inflation,
            dim=-1,
        )
        low_root = root_in_boxes.any(dim=1)
        foot_box_delta = torch.abs(
            feet_xy[:, :, None, :] - self.low_obstacle_centers[None, None, :, :]
        )
        feet_on_box = torch.all(
            foot_box_delta <= self.low_obstacle_half_extents[None, None, :, :] + 0.03,
            dim=-1,
        )
        foot_contact_boxes = (foot_contact[:, :, None] & feet_on_box).any(dim=1)
        low_foot_contact = foot_contact_boxes.any(dim=1)
        low_contact = ((body_contact & low_root) | low_foot_contact) & ~self.initial_
        low_event = low_contact & ~self.last_low_obstacle_contact & (
            self.low_obstacle_collision_cooldown == 0
        )
        self.low_obstacle_collision_count += low_event.long()
        self.low_obstacle_collision_cooldown[low_event] = 10
        contact_boxes = foot_contact_boxes | (body_contact[:, None] & root_in_boxes)
        contact_box_index = torch.argmax(contact_boxes.long(), dim=1)
        self.low_obstacle_collision_index[low_event] = contact_box_index[low_event]
        self.last_low_obstacle_contact = low_contact

        wall_limit = float(getattr(self.cfg.terrain, "boundary_wall_thickness", 0.30))
        wall_margin = wall_limit + self.footprint_inflation
        wall_contact = body_contact & (
            (root_xy[:, 0] <= wall_margin)
            | (root_xy[:, 0] >= self.terrain.env_length - wall_margin)
            | (root_xy[:, 1] <= wall_margin)
            | (root_xy[:, 1] >= self.terrain.env_width - wall_margin)
        ) & ~self.initial_
        stair_root = (
            (root_xy[:, 0] >= float(self.cfg.terrain.stair_start_x))
            & (root_xy[:, 0] <= float(self.cfg.depth_stairs.platform_start_x))
        )
        stair_contact = (body_contact & stair_root) | (
            foot_contact & (
                (feet_xy[:, :, 0] >= float(self.cfg.terrain.stair_start_x))
                & (feet_xy[:, :, 0] <= float(self.cfg.depth_stairs.platform_start_x))
            )
        ).any(dim=1)
        stair_contact &= ~self.initial_
        wall_event = wall_contact & ~self.last_wall_contact & ~low_event
        stair_event = stair_contact & ~self.last_stair_contact & ~low_event
        self.wall_contact_count += wall_event.long()
        self.stair_contact_count += stair_event.long()
        self.last_wall_contact = wall_contact
        self.last_stair_contact = stair_contact
        self.collision_class_code[low_event] = 1
        self.collision_class_code[wall_event] = 2
        self.collision_class_code[(~wall_event) & stair_event] = 3
        return low_event

    def _stair_stuck_candidate(self, local_x):
        cfg = self.cfg.depth_stairs
        in_stair = (local_x >= cfg.stair_stuck_x_range[0]) & (
            local_x <= cfg.stair_stuck_x_range[1]
        )
        self.stair_progress_anchor = torch.where(
            in_stair & (self.stair_progress_steps > 0),
            self.stair_progress_anchor,
            self.root_states[:, 0],
        )
        self.stair_progress_steps = torch.where(
            in_stair, self.stair_progress_steps + 1, torch.zeros_like(self.stair_progress_steps)
        )
        directed_progress = (
            self.root_states[:, 0] - self.stair_progress_anchor
        ) * self.navigation_direction.float()
        window_complete = self.stair_progress_steps >= int(cfg.stair_stuck_window_steps)
        stuck = stair_progress_is_stuck(
            in_stair,
            self.stair_progress_steps,
            directed_progress,
            int(cfg.stair_stuck_window_steps),
            cfg.stair_stuck_min_progress,
        )
        restart = window_complete & ~stuck
        self.stair_progress_anchor = torch.where(
            restart, self.root_states[:, 0], self.stair_progress_anchor
        )
        self.stair_progress_steps = torch.where(
            restart, torch.zeros_like(self.stair_progress_steps), self.stair_progress_steps
        )
        return stuck

    def check_termination(self):
        self.extras.pop("episode_outcomes", None)
        cfg = self.cfg.depth_stairs
        self.initial_ = self.episode_length_buf <= 1
        low_obstacle_collision = self._update_static_collision_metrics()
        local_x = self.root_states[:, 0] - self.room_origins[:, 0]
        up = self.navigation_direction > 0
        high_height_error = torch.abs(
            self.root_states[:, 2] - (self.base_init_state[2] + cfg.platform_height)
        )
        low_height_error = torch.abs(self.root_states[:, 2] - self.base_init_state[2])
        high_enough = high_height_error <= cfg.height_tolerance
        low_enough = low_height_error <= cfg.height_tolerance
        feet_local_x = self.feet_pos[:, :, 0] - self.room_origins[:, None, 0]
        position_cleared = stair_fully_cleared(
            local_x,
            feet_local_x,
            up,
            float(self.cfg.terrain.stair_start_x),
            float(cfg.platform_start_x),
            float(cfg.stair_clearance_distance),
            float(cfg.foot_clearance_margin),
        )
        target_surface_z = torch.where(
            up,
            torch.full_like(local_x, float(cfg.platform_height)),
            torch.zeros_like(local_x),
        )
        feet_on_target_height = torch.all(
            torch.abs(self.feet_pos[:, :, 2] - target_surface_z[:, None])
            <= float(cfg.foot_height_tolerance),
            dim=1,
        )
        body_forward = torch.zeros(self.num_envs, 3, device=self.device)
        body_forward[:, 0] = 1.0
        forward_world = quat_apply(self.root_states[:, 3:7], body_forward)
        heading_alignment = forward_world[:, 0] * self.navigation_direction.float()
        heading_aligned = heading_alignment >= math.cos(
            math.radians(float(cfg.heading_tolerance_deg))
        )
        self.fully_cleared = position_cleared & feet_on_target_height & heading_aligned
        correct_height = torch.where(up, high_enough, low_enough)
        crossed_now = self.fully_cleared & correct_height
        self.stair_crossed |= crossed_now

        self.correct_goal_height = correct_height
        success_eligible = stair_success_eligible(
            self.distance,
            correct_height,
            self.stair_crossed & self.fully_cleared,
            self.cfg.rewards.reach_pos_target_tight_config.distance_threshold,
        )
        self.goal_hold_timer = update_consecutive_timer(
            self.goal_hold_timer, success_eligible
        )
        success_candidate = self.goal_hold_timer >= self.cfg.env.goal_reached_time

        terminate_contact = torch.any(
            torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0,
            dim=1,
        ) & ~self.initial_
        fall = self.projected_gravity[:, 2] > -0.8
        fall_or_contact_candidate = fall | terminate_contact | low_obstacle_collision
        timeout_candidate = timeout_reached(
            self.episode_length_buf, self.max_episode_length
        )
        stair_stuck_candidate = self._stair_stuck_candidate(local_x) & ~self.initial_
        # A known-room teacher first aligns laterally at the low/high spawn
        # line. It intentionally holds vx=0 during that phase; do not call
        # this transient alignment a stair failure.
        teacher_alignment_pending = (
            (
                (up & (local_x < 1.20))
                | ((~up) & (local_x > 6.80))
            )
            & (torch.abs((self.root_states[:, 1] - self.room_origins[:, 1]) - 4.70)
               > float(cfg.teacher_alignment_tolerance))
        )
        stair_stuck_candidate &= ~teacher_alignment_pending

        low_motion = (torch.norm(self.base_lin_vel[:, :2], dim=1) < 0.1) & (
            torch.abs(self.base_ang_vel[:, 2]) < 0.1
        )
        outside_stairs = (local_x < cfg.stair_stuck_x_range[0]) | (
            local_x > cfg.stair_stuck_x_range[1]
        )
        count_other_stuck = low_motion & outside_stairs & (
            self.episode_length_buf > int(0.1 * self.max_episode_length)
        )
        self.stay_timer = update_consecutive_timer(self.stay_timer, count_other_stuck)
        other_stuck_candidate = self.stay_timer >= self.cfg.env.stay_time

        success, stair_stuck, fall_or_contact, timeout, other_stuck = exclusive_terminal_masks(
            success_candidate,
            stair_stuck_candidate,
            fall_or_contact_candidate,
            timeout_candidate,
            other_stuck_candidate,
        )
        self.terminal_reason.zero_()
        self.terminal_reason[success] = TERMINAL_SUCCESS
        self.terminal_reason[stair_stuck] = TERMINAL_STAIR_STUCK
        self.terminal_reason[fall_or_contact] = TERMINAL_FALL_OR_CONTACT
        self.terminal_reason[timeout] = TERMINAL_TIMEOUT
        self.terminal_reason[other_stuck] = TERMINAL_OTHER_STUCK

        self.goal_reached_flag = success
        self.stand_still_flag = stair_stuck | other_stuck
        # Stuck episodes are explicit failures and must receive the termination penalty.
        # Timeouts remain truncations and are deliberately excluded.
        self.terminate_buf = fall_or_contact | stair_stuck | other_stuck
        self.time_out_buf = timeout
        self.fall_down = fall
        self.reset_buf = success | stair_stuck | fall_or_contact | timeout | other_stuck
        self.reach_goal = success_eligible
        self.extras["bad_masks"] = self.initial_

        self.reset_goal = success.clone()
        self.reset_stand_still = (stair_stuck | other_stuck).clone()
        self.reset_timeout = timeout.clone()
        self.reset_fall = fall.clone()
        self.reset_contact50.zero_()
        self.reset_initial_contact50.zero_()
        self.reset_spawn_collision.zero_()
        self.reset_terminate_contact = terminate_contact.clone()
        self.reset_dynamic_collision.zero_()
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        if torch.any(success & timeout):
            raise RuntimeError("A stair navigation episode cannot be both success and timeout")
        if torch.any(self.reset_buf & (self.terminal_reason == TERMINAL_NONE)):
            raise RuntimeError("Every completed stair navigation episode needs one terminal reason")

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        outcomes = {
            "env_ids": env_ids.clone(),
            "direction": self.navigation_direction[env_ids].clone(),
            "seed": self.episode_seed[env_ids].clone(),
            "terminal_reason": self.terminal_reason[env_ids].clone(),
            "stair_crossed": self.stair_crossed[env_ids].clone(),
            "episode_steps": self.episode_length_buf[env_ids].clone(),
            "low_obstacle_collision_count": self.low_obstacle_collision_count[env_ids].clone(),
            "wall_contact_count": self.wall_contact_count[env_ids].clone(),
            "stair_contact_count": self.stair_contact_count[env_ids].clone(),
            "collision_class": self.collision_class_code[env_ids].clone(),
            "low_obstacle_collision_index": self.low_obstacle_collision_index[env_ids].clone(),
            "terminal_local_x": (
                self.root_states[env_ids, 0] - self.room_origins[env_ids, 0]
            ).clone(),
            "terminal_local_y": (
                self.root_states[env_ids, 1] - self.room_origins[env_ids, 1]
            ).clone(),
            "fully_cleared": self.fully_cleared[env_ids].clone(),
            "correct_goal_height": self.correct_goal_height[env_ids].clone(),
        }
        direction = outcomes["direction"]
        reason = outcomes["terminal_reason"]
        low_count = outcomes["low_obstacle_collision_count"].float()
        wall_count = outcomes["wall_contact_count"].float()
        stair_count = outcomes["stair_contact_count"].float()
        super().reset_idx(env_ids)
        self.extras["episode_outcomes"] = outcomes
        episode = self.extras["episode"]
        up = direction > 0
        down = direction < 0
        episode["up_success"] = (
            (reason[up] == TERMINAL_SUCCESS).float().mean() if torch.any(up) else torch.tensor(0.0, device=self.device)
        )
        episode["down_success"] = (
            (reason[down] == TERMINAL_SUCCESS).float().mean()
            if torch.any(down)
            else torch.tensor(0.0, device=self.device)
        )
        episode["stair_stuck_rate"] = (reason == TERMINAL_STAIR_STUCK).float().mean()
        episode["exclusive_timeout_rate"] = (reason == TERMINAL_TIMEOUT).float().mean()
        episode["low_obstacle_collision_count"] = low_count.mean()
        episode["wall_contact_count"] = wall_count.mean()
        episode["stair_contact_count"] = stair_count.mean()
        episode["safe_success"] = (
            (reason == TERMINAL_SUCCESS) & (low_count == 0.0) & outcomes["fully_cleared"]
        ).float().mean()
        self.stair_crossed[env_ids] = False
        self.fully_cleared[env_ids] = False
        self.stair_progress_anchor[env_ids] = self.root_states[env_ids, 0]
        self.stair_progress_steps[env_ids] = 0
        self.terminal_reason[env_ids] = TERMINAL_NONE
        self.correct_goal_height[env_ids] = False
        self.low_obstacle_collision_count[env_ids] = 0
        self.wall_contact_count[env_ids] = 0
        self.stair_contact_count[env_ids] = 0
        self.low_obstacle_collision_cooldown[env_ids] = 0
        self.last_low_obstacle_contact[env_ids] = False
        self.last_wall_contact[env_ids] = False
        self.last_stair_contact[env_ids] = False
        self.collision_class_code[env_ids] = 0
        self.low_obstacle_collision_index[env_ids] = -1
        self.actions_orig[env_ids] = 0.0

    def _reward_reach_pos_target_tight(self):
        reward = super()._reward_reach_pos_target_tight()
        return reward * self.correct_goal_height * self.stair_crossed * self.fully_cleared

    def _reward_teacher_action_tracking(self):
        command_error = torch.sum(
            torch.square(self.nav_actions_after_clip - self.get_navigation_teacher_actions()),
            dim=-1,
        )
        return torch.exp(-4.0 * command_error)
