"""Depth-sensor stair navigation with explicit passability supervision."""

from __future__ import annotations

import torch
from isaacgym.torch_utils import quat_from_euler_xyz, quat_rotate_inverse, torch_rand_float

from legged_gym.envs.base.legged_robot_pos_depth_stairs import (
    LeggedRobotPosDepthStairs,
)
from legged_gym.envs.base.legged_robot_pos_stairs_minimal import (
    TERMINAL_FALL_OR_CONTACT,
    TERMINAL_NONE,
    TERMINAL_OTHER_STUCK,
    TERMINAL_STAIR_STUCK,
    TERMINAL_SUCCESS,
    TERMINAL_TIMEOUT,
    TERMINAL_REASON_NAMES,
    stair_fully_cleared,
)
from legged_gym.utils.local_room_teacher import choose_local_waypoint, segment_clearance
from legged_gym.utils.torch_math import yaw_quat


class LeggedRobotPosDepthStairsPassability(LeggedRobotPosDepthStairs):
    """Bidirectional target-room task driven by predicted depth rays.

    The geometry is used only for teacher labels, terminal checks and offline
    diagnostics.  The actor observation continues to contain camera-predicted
    rays, proprioception and the local goal.
    """

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        mode = str(cfg.perception.mode)
        render_oracle = bool(
            getattr(cfg.perception, "render_depth_in_oracle", False)
        )
        cfg.sensors.depth_cam.enable = mode == "depth_predicted" or (
            mode == "oracle" and render_oracle
        )
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _resample_depth_stairs_navigation(self, env_ids):
        cfg = self._stairs_cfg()
        if len(env_ids) == 0:
            return
        if not hasattr(self, "navigation_direction"):
            self.navigation_direction = torch.ones(
                self.num_envs, dtype=torch.long, device=self.device
            )
            self.episode_seed = torch.full(
                (self.num_envs,), -1, dtype=torch.long, device=self.device
            )
            self.eval_seed_round = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )
            self.direction_reset_count = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )

        fixed_direction = int(getattr(cfg, "fixed_direction", 0))
        eval_seed_base = int(getattr(cfg, "eval_seed_base", -1))
        count = len(env_ids)
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
                (count,), 1 if fixed_direction > 0 else -1,
                dtype=torch.long,
                device=self.device,
            )

        seeded = eval_seed_base >= 0
        if seeded:
            seeds = (
                eval_seed_base
                + env_ids.long()
                + self.eval_seed_round[env_ids] * self.num_envs
            )
            self.eval_seed_round[env_ids] += 1
            self.episode_seed[env_ids] = seeds
            modulus = 2_147_483_647

            def seeded_uniform(salt, low, high):
                state = torch.remainder(
                    seeds * 1_103_515_245
                    + 12_345
                    + int(salt) * 2_654_435_761,
                    modulus,
                ).to(torch.float32)
                unit = state / float(modulus)
                return float(low) + unit * (float(high) - float(low))

            # All reset geometry, including x and terrain tile, is tied to the
            # episode seed.  This makes CBF/Teacher/actor comparisons exactly
            # repeatable instead of only fixing the lateral coordinate.
            start_y = seeded_uniform(7, *cfg.start_y_range)
            goal_y = seeded_uniform(11, *cfg.goal_y_range)
            up_start_x = seeded_uniform(13, *cfg.up_start_x_range)
            up_goal_x = seeded_uniform(17, *cfg.up_goal_x_range)
            down_start_x = seeded_uniform(19, *cfg.down_start_x_range)
            down_goal_x = seeded_uniform(23, *cfg.down_goal_x_range)
            rows = torch.remainder(seeds * 31 + 17, self.cfg.terrain.num_rows).long()
            cols = torch.remainder(seeds * 37 + 29, self.cfg.terrain.num_cols).long()
        else:
            self.episode_seed[env_ids] = -1
            start_y = torch.empty(count, device=self.device).uniform_(*cfg.start_y_range)
            goal_y = torch.empty(count, device=self.device).uniform_(*cfg.goal_y_range)
            up_start_x = torch.empty(count, device=self.device).uniform_(*cfg.up_start_x_range)
            up_goal_x = torch.empty(count, device=self.device).uniform_(*cfg.up_goal_x_range)
            down_start_x = torch.empty(count, device=self.device).uniform_(*cfg.down_start_x_range)
            down_goal_x = torch.empty(count, device=self.device).uniform_(*cfg.down_goal_x_range)
            rows = torch.randint(0, self.cfg.terrain.num_rows, (count,), device=self.device)
            cols = torch.randint(0, self.cfg.terrain.num_cols, (count,), device=self.device)

        if not seeded:
            self.episode_seed[env_ids] = (
                env_ids.long() + self.direction_reset_count[env_ids] * self.num_envs
            )

        up = direction > 0
        start_x = torch.where(up, up_start_x, down_start_x)
        goal_x = torch.where(up, up_goal_x, down_goal_x)
        start_surface = torch.where(
            up, torch.zeros(count, device=self.device), torch.full((count,), float(cfg.platform_height), device=self.device)
        )
        goal_surface = torch.where(
            up, torch.full((count,), float(cfg.platform_height), device=self.device), torch.zeros(count, device=self.device)
        )
        self.terrain_levels[env_ids] = rows
        self.terrain_types[env_ids] = cols
        self.navigation_direction[env_ids] = direction
        self.room_origins[env_ids, 0] = rows.float() * self.terrain.env_length
        self.room_origins[env_ids, 1] = cols.float() * self.terrain.env_width
        self.env_origins[env_ids, 0] = self.room_origins[env_ids, 0] + start_x
        self.env_origins[env_ids, 1] = self.room_origins[env_ids, 1] + start_y
        self.env_origins[env_ids, 2] = start_surface
        self.position_targets[env_ids, 0] = self.room_origins[env_ids, 0] + goal_x
        self.position_targets[env_ids, 1] = self.room_origins[env_ids, 1] + goal_y
        self.position_targets[env_ids, 2] = self.base_init_state[2] + goal_surface if hasattr(self, "base_init_state") else goal_surface + 0.42
        self.episode_start_local_y[env_ids] = start_y

    def _get_env_origins(self):
        self.episode_start_local_y = torch.zeros(self.num_envs, device=self.device)
        super()._get_env_origins()
        # ``_get_env_origins`` is called while constructing the simulator and
        # is not an episode reset.  Do not consume the first deterministic
        # evaluation seed before ``env.reset()`` starts the episode.
        if hasattr(self, "eval_seed_round"):
            self.eval_seed_round.zero_()

    def _check_spawn_collision(self):
        local_xy = self.root_states[:, :2] - self.room_origins
        inside_obstacle = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        inflation = float(getattr(self.cfg.depth_stairs, "collision_inflation", 0.18))
        for box in self.low_obstacle_boxes:
            center_x, center_y, size_x, size_y, _ = box
            inside_obstacle |= (
                torch.abs(local_xy[:, 0] - center_x) <= size_x * 0.5 + inflation
            ) & (
                torch.abs(local_xy[:, 1] - center_y) <= size_y * 0.5 + inflation
            )
        wall_margin = 0.45
        inside_wall = (
            (local_xy[:, 0] <= wall_margin)
            | (local_xy[:, 0] >= self.terrain.env_length - wall_margin)
            | (local_xy[:, 1] <= wall_margin)
            | (local_xy[:, 1] >= self.terrain.env_width - wall_margin)
        )
        self.reset_spawn_collision = self.initial_ & (inside_obstacle | inside_wall)
        self.reset_buf |= self.reset_spawn_collision

    def _reset_root_states(self, env_ids):
        if len(env_ids) == 0:
            return
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        if int(getattr(self.cfg.depth_stairs, "eval_seed_base", -1)) >= 0:
            self.root_states[env_ids, 7:13] = 0.0
        else:
            self.root_states[env_ids, 7:13] = torch_rand_float(
                -0.5, 0.5, (len(env_ids), 6), device=self.device
            )
        direction = self.navigation_direction[env_ids]
        yaw = torch.where(
            direction > 0,
            torch.zeros_like(direction, dtype=torch.float),
            torch.full_like(direction, torch.pi, dtype=torch.float),
        )
        self.root_states[env_ids, 3:7] = quat_from_euler_xyz(
            torch.zeros_like(yaw), torch.zeros_like(yaw), yaw
        )
        self.base_quat[env_ids] = self.root_states[env_ids, 3:7]
        self._set_robot_root_states(env_ids)
        if hasattr(self, "_reset_dynamic_obstacles"):
            self._reset_dynamic_obstacles(env_ids)

    def _init_buffers(self):
        super()._init_buffers()
        boxes = torch.as_tensor(
            self.cfg.terrain.low_obstacle_boxes, dtype=torch.float, device=self.device
        ).reshape(-1, 5)
        self.teacher_centers = boxes[:, :2]
        self.teacher_half_extents = boxes[:, 2:4] * 0.5
        self.teacher_waypoint = torch.zeros(self.num_envs, 2, device=self.device)
        self.teacher_waypoint_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.lane_escape_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.lane_escape_y = torch.zeros(self.num_envs, device=self.device)
        self.teacher_lateral_escape_sign = torch.zeros(self.num_envs, device=self.device)
        self.teacher_path_distance = torch.zeros(self.num_envs, device=self.device)
        self.passability_targets = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.passability_update_step = -1
        self.stair_crossed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.fully_cleared = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.obstacle_field_crossed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.teacher_lateral_only_count = torch.zeros(self.num_envs, device=self.device)
        self.teacher_lateral_only_total_count = torch.zeros(self.num_envs, device=self.device)
        self.teacher_forward_ready_steps = torch.zeros(self.num_envs, device=self.device)
        self.stair_approached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._stair_approach_rewarded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._stair_clear_rewarded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._obstacle_field_rewarded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.stage_progress = torch.zeros(self.num_envs, device=self.device)
        self.stair_progress_reward = torch.zeros(self.num_envs, device=self.device)
        self.obstacle_progress_reward = torch.zeros(self.num_envs, device=self.device)
        self.goal_approach_reward = torch.zeros(self.num_envs, device=self.device)
        self.terminal_reason = torch.full(
            (self.num_envs,), TERMINAL_NONE, dtype=torch.long, device=self.device
        )
        self.policy_cbf_intervention = torch.zeros(self.num_envs, device=self.device)
        self.policy_cbf_intervention_sum = torch.zeros(self.num_envs, device=self.device)
        self.policy_cbf_intervention_steps = torch.zeros(self.num_envs, device=self.device)
        self.action_filter_delta = torch.zeros(self.num_envs, device=self.device)
        self.action_filter_delta_sum = torch.zeros(self.num_envs, device=self.device)
        self.action_filter_delta_steps = torch.zeros(self.num_envs, device=self.device)
        self.prev_directed_x = torch.zeros(self.num_envs, device=self.device)
        self.stage_prev_distance = torch.zeros(self.num_envs, device=self.device)
        self.teacher_action_raw = torch.zeros(self.num_envs, 3, device=self.device)
        self.teacher_action_after_filter = torch.zeros(self.num_envs, 3, device=self.device)

    def record_policy_action_diagnostics(self, u_bar, u_safe):
        """Record the actual CBF delta before the environment action filter."""
        if u_bar is None or u_safe is None:
            return
        delta = torch.linalg.vector_norm(u_safe - u_bar, dim=-1).detach()
        self.policy_cbf_intervention = delta
        active = (self.episode_length_buf > 1).float()
        self.policy_cbf_intervention_sum += delta * active
        self.policy_cbf_intervention_steps += (delta > 1.0e-3).float() * active

    def step(self, nav_actions):
        # Keep the previous filtered command so Teacher targets can use the
        # same low-pass/clip contract as actor commands.
        self.nav_actions_filtered_prev = self.nav_actions_filtered.clone()
        # Count the action that is actually submitted to the environment once
        # per control step.  Reward computation may query the Teacher again;
        # counting there would inflate the lateral-only diagnostic.
        lateral_only = (
            (nav_actions[:, 0] < 0.08) & (torch.abs(nav_actions[:, 1]) > 0.08)
        ).float()
        self.teacher_lateral_only_total_count += lateral_only
        local_x = self.root_states[:, 0] - self.room_origins[:, 0]
        obstacle_stage = (~self.fully_cleared) & (
            local_x < float(getattr(self.cfg.depth_stairs, "platform_start_x", 6.30))
        )
        self.teacher_lateral_only_count += lateral_only * obstacle_stage.float()
        self.teacher_forward_ready_steps += (nav_actions[:, 0] > 0.08).float()
        return super().step(nav_actions)

    def _update_passability_targets(self):
        if self.passability_update_step == self.common_step_counter:
            return
        local_xy = self.root_states[:, :2] - self.room_origins
        goals = self.position_targets[:, :2] - self.room_origins
        waypoint, _, path_distance, labels = choose_local_waypoint(
            local_xy,
            goals,
            self.teacher_centers,
            self.teacher_half_extents,
            inflation=float(getattr(self.cfg.depth_stairs, "teacher_obstacle_inflation", 0.42)),
            lateral_margin=float(getattr(self.cfg.depth_stairs, "teacher_lateral_margin", 0.18)),
            min_directed_progress=float(
                getattr(self.cfg.depth_stairs, "teacher_min_directed_progress", 0.05)
            ),
            room_size=float(self.terrain.env_width),
            preferred_lateral_sign=self.teacher_lateral_escape_sign,
            min_clearance=float(
                getattr(self.cfg.depth_stairs, "teacher_min_clearance", 0.0)
            ),
        )
        previous_waypoint = self.teacher_waypoint.clone()
        previous_delta = previous_waypoint - local_xy
        previous_distance = torch.linalg.vector_norm(previous_delta, dim=-1)
        previous_directed = previous_delta[:, 0] * torch.sign(goals[:, 0] - local_xy[:, 0])
        previous_free, _ = segment_clearance(
            local_xy,
            previous_waypoint,
            self.teacher_centers,
            self.teacher_half_extents,
            inflation=float(getattr(self.cfg.depth_stairs, "teacher_obstacle_inflation", 0.42)),
        )
        hold_previous = (
            self.teacher_waypoint_valid
            & (previous_distance > 0.12)
            & (previous_directed > 0.10)
            & previous_free[:, 0]
        )
        waypoint = torch.where(hold_previous[:, None], previous_waypoint, waypoint)
        stair_start = float(self.cfg.terrain.stair_start_x)
        stair_end = stair_start + float(self.cfg.terrain.stair_count) * float(self.cfg.terrain.stair_tread)
        stair_region = (local_xy[:, 0] >= stair_start - 0.30) & (local_xy[:, 0] <= stair_end + 0.30)
        labels = torch.where(stair_region, torch.full_like(labels, 2), labels)
        up = self.navigation_direction > 0
        room_center = 0.5 * float(self.terrain.env_width)
        wall_lane = max(
            0.85,
            float(getattr(self.cfg.depth_stairs, "teacher_obstacle_inflation", 0.42))
            + float(getattr(self.cfg.depth_stairs, "teacher_lateral_margin", 0.18))
            + 0.45,
        )
        lane_candidate = (labels == 1) & (
            (torch.abs(waypoint[:, 1] - wall_lane) < 0.06)
            | (torch.abs(waypoint[:, 1] - (room_center - 0.30)) < 0.06)
            | (torch.abs(waypoint[:, 1] - (room_center + 0.30)) < 0.06)
            | (torch.abs(waypoint[:, 1] - (float(self.terrain.env_width) - wall_lane)) < 0.06)
        )
        self.lane_escape_active[:] = lane_candidate
        self.lane_escape_y = torch.where(
            lane_candidate, waypoint[:, 1], self.lane_escape_y
        )
        self.teacher_waypoint[:] = waypoint
        self.teacher_waypoint_valid[:] = True
        new_lateral_escape = (labels == 1) & (
            torch.abs(waypoint[:, 1] - local_xy[:, 1]) >= 0.20
        )
        self.teacher_lateral_escape_sign[:] = torch.where(
            new_lateral_escape,
            torch.sign(waypoint[:, 1] - local_xy[:, 1]),
            torch.where(labels == 1, self.teacher_lateral_escape_sign, torch.zeros_like(labels, dtype=torch.float)),
        )
        self.teacher_path_distance[:] = path_distance
        self.passability_targets[:] = labels
        self.extras["passability_targets"] = labels.detach()
        self.passability_update_step = self.common_step_counter

    def update_percetion(self):
        super().update_percetion()
        if hasattr(self, "nav_actions_orig") and hasattr(self, "nav_actions_after_clip"):
            self.action_filter_delta = torch.linalg.vector_norm(
                self.nav_actions_after_clip - self.nav_actions_orig, dim=-1
            )
            active = (self.episode_length_buf > 1).float()
            self.action_filter_delta_sum += self.action_filter_delta * active
            self.action_filter_delta_steps += (self.action_filter_delta > 1.0e-3).float() * active
        self._update_passability_targets()

    def get_passability_targets(self):
        self._update_passability_targets()
        return self.passability_targets

    def get_navigation_teacher_actions(self):
        self._update_passability_targets()
        local_xy = self.root_states[:, :2] - self.room_origins
        goals = self.position_targets[:, :2] - self.room_origins
        target_vector = self.teacher_waypoint - local_xy
        target_distance = torch.linalg.vector_norm(target_vector, dim=-1).clamp(min=1.0e-5)
        direction = target_vector / target_distance.unsqueeze(-1)
        base_speed = float(getattr(self.cfg.depth_stairs, "teacher_speed", 0.40))
        bypass_speed = float(
            getattr(self.cfg.depth_stairs, "teacher_bypass_speed", base_speed)
        )
        bypass = self.passability_targets == 1
        directed_waypoint_x = target_vector[:, 0] * self.navigation_direction.float()
        lateral_escape = bypass & (
            torch.abs(target_vector[:, 1]) >= 0.25
        ) & (directed_waypoint_x <= 0.10)
        backtrack_lateral = bypass & (directed_waypoint_x < -0.02) & (
            torch.abs(target_vector[:, 1]) >= 0.20
        )
        speed_tensor = torch.where(
            self.lane_escape_active | lateral_escape,
            torch.full_like(target_distance, bypass_speed),
            torch.full_like(target_distance, base_speed),
        )
        stair_state = self.passability_targets == 2
        stair_speed = torch.where(
            self.navigation_direction > 0,
            torch.full_like(target_distance, float(getattr(self.cfg.depth_stairs, "teacher_stair_speed", 0.22))),
            torch.full_like(target_distance, float(getattr(self.cfg.depth_stairs, "teacher_down_stair_speed", 0.40))),
        )
        stair_min_forward = torch.where(
            self.navigation_direction > 0,
            torch.full_like(target_distance, float(getattr(self.cfg.depth_stairs, "teacher_stair_min_forward_speed", 0.18))),
            torch.full_like(target_distance, float(getattr(self.cfg.depth_stairs, "teacher_down_stair_min_forward_speed", 0.32))),
        )
        stair_start = float(self.cfg.terrain.stair_start_x)
        stair_end = stair_start + float(self.cfg.terrain.stair_count) * float(self.cfg.terrain.stair_tread)
        down_high_platform_hold = (
            (self.navigation_direction < 0)
            & (local_xy[:, 0] > stair_end + .20)
            & ~self.fully_cleared
            & ~stair_state
        )
        speed_tensor = torch.where(
            stair_state,
            stair_speed,
            speed_tensor,
        )
        down_platform_lateral = (
            (self.navigation_direction < 0)
            & self.fully_cleared
            & (local_xy[:, 0] < stair_start - 1.30)
            & ~bypass
            & (torch.abs(target_vector[:, 1]) > 0.20)
        )
        down_pre_obstacle_hold = (
            (self.navigation_direction < 0)
            & self.fully_cleared
            & (local_xy[:, 0] >= stair_start - 1.30)
        )
        speed_tensor = torch.where(
            down_platform_lateral,
            torch.full_like(target_distance, float(getattr(self.cfg.depth_stairs, "teacher_down_platform_speed", 0.40))),
            speed_tensor,
        )
        desired_world = direction * speed_tensor.unsqueeze(-1)
        body = quat_rotate_inverse(
            yaw_quat(self.base_quat),
            torch.cat((desired_world, torch.zeros(self.num_envs, 1, device=self.device)), dim=-1),
        )[:, :2]
        down_lane_error = torch.full_like(local_xy[:, 1], float(getattr(self.cfg.depth_stairs, "teacher_down_stair_lane_y", 8.20))) - local_xy[:, 1]
        down_lane_world = torch.stack((torch.zeros_like(down_lane_error), down_lane_error), dim=-1)
        down_lane_body = quat_rotate_inverse(
            yaw_quat(self.base_quat),
            torch.cat((down_lane_world, torch.zeros(self.num_envs, 1, device=self.device)), dim=-1),
        )[:, :2]
        body[:, 0].clamp_(min=-0.25, max=bypass_speed)
        body[:, 1].clamp_(
            min=-float(getattr(self.cfg.depth_stairs, "teacher_max_lateral_speed", 0.40)),
            max=float(getattr(self.cfg.depth_stairs, "teacher_max_lateral_speed", 0.40)),
        )
        stair_lateral_speed = float(
            getattr(self.cfg.depth_stairs, "teacher_stair_lateral_speed", 0.08)
        )
        body[:, 1] = torch.where(
            stair_state,
            body[:, 1].clamp(min=-stair_lateral_speed, max=stair_lateral_speed),
            body[:, 1],
        )
        blocked = self.passability_targets == 3
        backtracking = bypass & (directed_waypoint_x < -0.02)
        goal_overshoot = (~bypass) & (directed_waypoint_x < -0.02)
        minimum_forward = torch.where(
            bypass,
            torch.where(
                lateral_escape | backtracking,
                torch.full_like(
                    body[:, 0],
                    float(getattr(self.cfg.depth_stairs, "teacher_bypass_min_forward_speed", -0.08)),
                ),
                torch.full_like(
                    body[:, 0],
                    float(getattr(self.cfg.depth_stairs, "teacher_min_forward_speed", 0.12)),
                ),
            ),
            torch.full_like(body[:, 0], float(getattr(self.cfg.depth_stairs, "teacher_clear_min_forward_speed", 0.24))),
        )
        minimum_forward = torch.where(
            stair_state,
            stair_min_forward,
            minimum_forward,
        )
        minimum_forward = torch.where(
            down_high_platform_hold,
            torch.full_like(body[:, 0], float(getattr(self.cfg.depth_stairs, "teacher_down_high_platform_forward_speed", 0.35))),
            minimum_forward,
        )
        minimum_forward = torch.where(
            backtracking,
            torch.full_like(
                body[:, 0],
                float(getattr(self.cfg.depth_stairs, "teacher_backtrack_speed", -0.35)),
            ),
            minimum_forward,
        )
        # On the destination platform, keep a small forward component while
        # correcting lateral goal error after a few centimetres of overshoot.
        minimum_forward = torch.where(
            goal_overshoot & ~self.fully_cleared,
            torch.full_like(body[:, 0], 0.08),
            minimum_forward,
        )
        platform_goal_lateral = (~bypass) & self.fully_cleared & (
            torch.abs(target_vector[:, 1]) > 0.20
        )
        up_platform_hold = (self.navigation_direction > 0) & self.fully_cleared
        up_platform_undershoot = up_platform_hold & (
            local_xy[:, 0] < goals[:, 0] - 0.05
        )
        up_platform_overshoot = up_platform_hold & (
            local_xy[:, 0] > goals[:, 0] + 0.10
        )
        up_platform_ready = up_platform_hold & ~up_platform_undershoot & ~up_platform_overshoot
        minimum_forward = torch.where(
            platform_goal_lateral | up_platform_hold,
            torch.zeros_like(minimum_forward),
            minimum_forward,
        )
        minimum_forward = torch.where(
            up_platform_undershoot,
            torch.full_like(minimum_forward, 0.12),
            minimum_forward,
        )
        minimum_forward = torch.where(
            up_platform_overshoot,
            torch.full_like(minimum_forward, -0.15),
            minimum_forward,
        )
        minimum_forward = torch.where(
            up_platform_ready,
            torch.zeros_like(minimum_forward),
            minimum_forward,
        )
        # The frozen locomotion policy has little authority for reverse vx.
        # For a backtracking waypoint that also has a real lateral escape,
        # hold vx at zero and let the lateral command clear the margin.
        minimum_forward = torch.where(
            backtracking & (lateral_escape | backtrack_lateral),
            torch.zeros_like(minimum_forward),
            minimum_forward,
        )
        minimum_forward = torch.where(blocked, torch.zeros_like(minimum_forward), minimum_forward)
        body[:, 0] = torch.maximum(body[:, 0], minimum_forward)
        recovery_lateral_speed = min(
            bypass_speed,
            float(getattr(self.cfg.depth_stairs, "teacher_max_lateral_speed", 0.40)),
        )
        body[:, 0] = torch.where(backtrack_lateral, torch.zeros_like(body[:, 0]), body[:, 0])
        body[:, 1] = torch.where(
            backtrack_lateral,
            torch.sign(target_vector[:, 1]) * recovery_lateral_speed,
            body[:, 1],
        )
        body[:, 0] = torch.where(up_platform_ready, torch.zeros_like(body[:, 0]), body[:, 0])
        body[:, 1] = torch.where(
            down_high_platform_hold, down_lane_body[:, 1].clamp(min=-0.40, max=0.40), body[:, 1]
        )
        body[:, 1] = torch.where(
            down_pre_obstacle_hold, torch.zeros_like(body[:, 1]), body[:, 1]
        )
        # Once the down robot reaches the target-side x boundary, do not command reverse body vx into the wall while finishing the lateral goal.
        down_goal_lateral = (
            (self.navigation_direction < 0)
            & self.fully_cleared
            & (local_xy[:, 0] <= goals[:, 0] + 0.05)
            & (torch.abs(target_vector[:, 1]) > 0.20)
        )
        body[:, 0] = torch.where(down_goal_lateral, torch.zeros_like(body[:, 0]), body[:, 0])
        # Keep the frozen locomotion heading aligned with the navigation axis.
        # Lateral bypass remains a body-y command, so this prevents sideways
        # motion from drifting the base yaw and starving the world-y command.
        yaw = torch.atan2(
            2.0 * (
                self.base_quat[:, 3] * self.base_quat[:, 2]
                + self.base_quat[:, 0] * self.base_quat[:, 1]
            ),
            1.0
            - 2.0 * (
                self.base_quat[:, 1] * self.base_quat[:, 1]
                + self.base_quat[:, 2] * self.base_quat[:, 2]
            ),
        )
        desired_yaw = torch.where(
            self.navigation_direction > 0,
            torch.zeros_like(yaw),
            torch.full_like(yaw, torch.pi),
        )
        yaw_error = torch.atan2(
            torch.sin(desired_yaw - yaw), torch.cos(desired_yaw - yaw)
        )
        yaw_rate = (
            float(getattr(self.cfg.depth_stairs, "teacher_heading_gain", 1.50))
            * yaw_error
        ).clamp(
            min=-float(getattr(self.cfg.depth_stairs, "teacher_max_yaw_rate", 1.0)),
            max=float(getattr(self.cfg.depth_stairs, "teacher_max_yaw_rate", 1.0)),
        )
        distance_to_goal = torch.linalg.vector_norm(goals - local_xy, dim=-1)
        actions = torch.cat((body, yaw_rate[:, None]), dim=-1)
        # Do not stop in the final 20 cm until the base and all feet have
        # cleared the stairs; otherwise the frozen locomotion can settle just
        # short of the strict full-clearance boundary.
        stop_at_goal = (distance_to_goal <= 0.20) & self.fully_cleared
        actions *= (~stop_at_goal).float().unsqueeze(-1)
        self.teacher_action_raw[:] = actions
        return actions

    def _reward_teacher_action_tracking(self):
        teacher = self.get_navigation_teacher_actions()
        previous = getattr(self, "nav_actions_filtered_prev", self.nav_actions_filtered)
        alpha = float(getattr(self.cfg.commands, "alpha", 1.0))
        teacher_orig = torch.clip(teacher, -3.0, 3.0)
        teacher_filtered = alpha * teacher_orig + (1.0 - alpha) * previous
        self.teacher_action_after_filter[:] = torch.clip(
            teacher_filtered, min=self.nav_clip_min, max=self.nav_clip_max
        )
        command_error = torch.sum(
            torch.square(
                self.nav_actions_after_clip - self.teacher_action_after_filter
            ),
            dim=-1,
        )
        return torch.exp(-4.0 * command_error)

    def check_termination(self):
        previous_goal_hold_timer = self.goal_hold_timer.clone()
        previous_stay_timer = self.stay_timer.clone()
        super().check_termination()
        self.goal_hold_timer = torch.where(
            self.reach_goal,
            previous_goal_hold_timer + 1,
            torch.zeros_like(previous_goal_hold_timer),
        )
        raw_goal_reached = self.goal_hold_timer >= self.cfg.env.goal_reached_time
        self.stay_timer = torch.where(
            self.static,
            previous_stay_timer + 1,
            torch.zeros_like(previous_stay_timer),
        )
        self.stand_still_flag = self.stay_timer >= self.cfg.env.stay_time
        self.reset_stand_still = self.stand_still_flag.clone()
        cfg = self.cfg.depth_stairs
        local_x = self.root_states[:, 0] - self.room_origins[:, 0]
        stair_start = float(self.cfg.terrain.stair_start_x)
        stair_end = stair_start + float(self.cfg.terrain.stair_count) * float(self.cfg.terrain.stair_tread)
        up = self.navigation_direction > 0
        correct_height = torch.where(
            up,
            self.root_states[:, 2] >= self.base_init_state[2] + float(cfg.platform_height) - 0.12,
            self.root_states[:, 2] <= self.base_init_state[2] + 0.12,
        )
        feet_local_x = self.feet_pos[:, :, 0] - self.room_origins[:, None, 0]
        position_cleared = stair_fully_cleared(
            local_x,
            feet_local_x,
            up,
            stair_start,
            stair_end,
            float(getattr(cfg, "stair_clearance_distance", 0.80)),
            float(getattr(cfg, "foot_clearance_margin", 0.05)),
        )
        target_surface_z = torch.where(
            up,
            torch.full_like(local_x, float(cfg.platform_height)),
            torch.zeros_like(local_x),
        )
        feet_on_target = torch.all(
            torch.abs(self.feet_pos[:, :, 2] - target_surface_z[:, None])
            <= float(getattr(cfg, "foot_height_tolerance", 0.12)),
            dim=1,
        )
        passed = position_cleared & correct_height
        fully_cleared_now = passed & feet_on_target
        self.stair_crossed |= passed
        self.fully_cleared |= fully_cleared_now
        obstacle_field_now = torch.where(up, local_x >= 4.50, local_x <= 1.15)
        self.obstacle_field_crossed |= obstacle_field_now
        stair_start_directed = torch.where(up, local_x, -local_x)
        previous_directed = self.prev_directed_x.clone()
        directed_delta = (stair_start_directed - previous_directed).clamp(-0.20, 0.20)
        self.prev_directed_x[:] = stair_start_directed
        self.stage_progress[:] = directed_delta.clamp(min=0.0)
        self.obstacle_progress_reward[:] = self.stage_progress * (
            stair_start_directed < torch.where(up, torch.full_like(local_x, 4.50), torch.full_like(local_x, 8.85))
        ).float()
        stair_region = torch.where(
            up,
            (local_x >= stair_start - 0.30) & (local_x <= stair_end + 0.30),
            (local_x <= stair_end + 0.30) & (local_x >= stair_start - 0.30),
        )
        self.stair_approached |= stair_region
        self.stair_progress_reward[:] = self.stage_progress * stair_region.float()
        self.goal_approach_reward[:] = (
            self.stage_prev_distance - self.distance
        ).clamp(min=0.0, max=0.20)
        self.stage_prev_distance[:] = self.distance
        valid_goal = raw_goal_reached & self.fully_cleared & (
            self.low_obstacle_collision_count == 0
        )
        strict_rules = bool(getattr(cfg, "strict_terminal_rules", False))
        enable_stand_still = bool(getattr(cfg, "enable_stand_still_reset", strict_rules))
        low_collision_failure = strict_rules & (self.low_obstacle_collision_event | (self.low_obstacle_collision_count > 0))
        fall_or_contact = (
            self.fall_down
            | self.reset_terminate_contact
            | self.reset_contact50
            | self.reset_dynamic_collision
            | low_collision_failure
        )
        stair_stuck = self.stand_still_flag & enable_stand_still & ~self.fully_cleared
        timeout = self.time_out_buf
        other_stuck = self.stand_still_flag & enable_stand_still & self.fully_cleared
        success, stair_stuck, fall_or_contact, timeout, other_stuck = (
            valid_goal & ~fall_or_contact & ~stair_stuck & ~timeout,
            stair_stuck & ~fall_or_contact,
            fall_or_contact,
            timeout & ~fall_or_contact & ~stair_stuck,
            other_stuck,
        )
        self.terminal_reason.fill_(TERMINAL_NONE)
        self.terminal_reason[success] = TERMINAL_SUCCESS
        self.terminal_reason[stair_stuck] = TERMINAL_STAIR_STUCK
        self.terminal_reason[fall_or_contact] = TERMINAL_FALL_OR_CONTACT
        self.terminal_reason[timeout] = TERMINAL_TIMEOUT
        self.terminal_reason[other_stuck] = TERMINAL_OTHER_STUCK
        known_non_goal_reset = (
            fall_or_contact | stair_stuck | timeout | other_stuck | self.reset_spawn_collision
        )
        self.reset_buf = success | known_non_goal_reset
        self.goal_reached_flag = success
        self.reset_goal = self.goal_reached_flag.clone()
        self.reset_stand_still = stair_stuck | other_stuck
        self.reset_timeout = timeout
        self.reset_fall = self.fall_down
        self.reset_terminate_contact = fall_or_contact

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        stair = self.stair_crossed[env_ids].float().clone()
        cleared = self.fully_cleared[env_ids].float().clone()
        field = self.obstacle_field_crossed[env_ids].float().clone()
        labels = self.passability_targets[env_ids].clone()
        teacher_lateral = self.teacher_lateral_only_count[env_ids].clone()
        teacher_lateral_total = self.teacher_lateral_only_total_count[env_ids].clone()
        success = self.goal_reached_flag[env_ids].float().clone()
        low_collisions = self.low_obstacle_collision_count[env_ids].float().clone()
        directions = self.navigation_direction[env_ids].float().clone()
        seeds = self.episode_seed[env_ids].float().clone()
        reasons = self.terminal_reason[env_ids].clone()
        cbf_mean = self.policy_cbf_intervention_sum[env_ids] / self.episode_length_buf[env_ids].float().clamp(min=1.0)
        cbf_steps = self.policy_cbf_intervention_steps[env_ids] / self.episode_length_buf[env_ids].float().clamp(min=1.0)
        filter_mean = self.action_filter_delta_sum[env_ids] / self.episode_length_buf[env_ids].float().clamp(min=1.0)
        filter_steps = self.action_filter_delta_steps[env_ids] / self.episode_length_buf[env_ids].float().clamp(min=1.0)
        super().reset_idx(env_ids)
        episode = self.extras.setdefault("episode", {})
        episode["direction"] = directions.mean()
        episode["episode_seed"] = seeds.mean()
        episode["stair_pass_rate"] = stair.mean()
        episode["stair_crossed"] = stair.mean()
        episode["fully_cleared"] = cleared.mean()
        episode["obstacle_field_crossed"] = field.mean()
        episode["depth_safe_success"] = torch.mean(
            success * (low_collisions == 0).float() * cleared
        )
        episode["passability_target_mean"] = labels.float().mean()
        episode["teacher_lateral_only_steps"] = teacher_lateral.mean()
        episode["teacher_lateral_only_total_steps"] = teacher_lateral_total.mean()
        episode["terminal_reason"] = reasons.float().mean()
        episode["cbf_intervention_norm_mean"] = cbf_mean.mean()
        episode["cbf_intervention_rate"] = cbf_steps.mean()
        episode["action_filter_delta_mean"] = filter_mean.mean()
        episode["action_filter_delta_rate"] = filter_steps.mean()
        episode["stair_approached"] = self.stair_approached[env_ids].float().mean()
        episode["goal_reached"] = success.mean()
        self.stair_crossed[env_ids] = False
        self.fully_cleared[env_ids] = False
        self.obstacle_field_crossed[env_ids] = False
        self.teacher_lateral_only_count[env_ids] = 0.0
        self.teacher_lateral_only_total_count[env_ids] = 0.0
        self.teacher_forward_ready_steps[env_ids] = 0.0
        self.teacher_waypoint_valid[env_ids] = False
        self.lane_escape_active[env_ids] = False
        self.lane_escape_y[env_ids] = 0.0
        self.teacher_lateral_escape_sign[env_ids] = 0.0
        self.stair_approached[env_ids] = False
        self._stair_approach_rewarded[env_ids] = False
        self._stair_clear_rewarded[env_ids] = False
        self._obstacle_field_rewarded[env_ids] = False
        self.stage_progress[env_ids] = 0.0
        self.stair_progress_reward[env_ids] = 0.0
        self.obstacle_progress_reward[env_ids] = 0.0
        self.goal_approach_reward[env_ids] = 0.0
        self.policy_cbf_intervention[env_ids] = 0.0
        self.policy_cbf_intervention_sum[env_ids] = 0.0
        self.policy_cbf_intervention_steps[env_ids] = 0.0
        self.action_filter_delta[env_ids] = 0.0
        self.action_filter_delta_sum[env_ids] = 0.0
        self.action_filter_delta_steps[env_ids] = 0.0
        self.terminal_reason[env_ids] = TERMINAL_NONE
        self.prev_directed_x[env_ids] = torch.where(
            self.navigation_direction[env_ids] > 0,
            self.root_states[env_ids, 0] - self.room_origins[env_ids, 0],
            -(self.root_states[env_ids, 0] - self.room_origins[env_ids, 0]),
        )
        self.stage_prev_distance[env_ids] = self.distance[env_ids]
        self.passability_update_step = -1

    def _reward_obstacle_field_progress(self):
        return self.obstacle_progress_reward

    def _reward_stair_approach(self):
        reward = self.stair_approached.float() * (self._stair_approach_rewarded == 0).float()
        self._stair_approach_rewarded |= self.stair_approached
        return reward

    def _reward_stair_crossing(self):
        return self.stair_progress_reward

    def _reward_stair_clearance(self):
        reward = self.fully_cleared.float() * (self._stair_clear_rewarded == 0).float()
        self._stair_clear_rewarded |= self.fully_cleared
        return reward

    def _reward_goal_approach(self):
        return self.goal_approach_reward
