"""Minimal bidirectional stair navigation without exteroceptive observations."""

from __future__ import annotations

import math

import torch
from isaacgym.torch_utils import quat_apply, quat_from_euler_xyz

from legged_gym.envs.base.legged_robot_pos_dynamic import LeggedRobotPosDynamic


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
    """Goal navigation over the fixed five-step room with constant open-space rays."""

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
        self._resample_sparse_navigation(torch.arange(self.num_envs, device=self.device))

    def _resample_sparse_navigation(self, env_ids):
        if len(env_ids) == 0:
            return
        cfg = self.cfg.depth_stairs
        count = len(env_ids)
        rows = torch.randint(0, self.cfg.terrain.num_rows, (count,), device=self.device)
        cols = torch.randint(0, self.cfg.terrain.num_cols, (count,), device=self.device)
        fixed_direction = int(getattr(cfg, "fixed_direction", 0))
        if fixed_direction == 0:
            direction = torch.where(
                torch.rand(count, device=self.device) < 0.5,
                torch.ones(count, dtype=torch.long, device=self.device),
                -torch.ones(count, dtype=torch.long, device=self.device),
            )
        else:
            direction = torch.full(
                (count,), 1 if fixed_direction > 0 else -1, dtype=torch.long, device=self.device
            )

        room_x = rows.float() * self.terrain.env_length
        room_y = cols.float() * self.terrain.env_width
        start_y = torch.empty(count, device=self.device).uniform_(*cfg.start_y_range)
        goal_y = torch.empty(count, device=self.device).uniform_(*cfg.goal_y_range)
        low_x = torch.empty(count, device=self.device).uniform_(*cfg.start_x_range)
        high_x = torch.empty(count, device=self.device).uniform_(*cfg.goal_x_range)
        up = direction > 0
        start_x = torch.where(up, low_x, high_x)
        goal_x = torch.where(up, high_x, low_x)
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

    def _init_buffers(self):
        super()._init_buffers()
        self.stair_crossed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.fully_cleared = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.stair_progress_anchor = self.root_states[:, 0].clone()
        self.stair_progress_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.terminal_reason = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.correct_goal_height = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
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
        if env_ids is not None:
            self.rays[env_ids] = self.cfg.sensors.ray2d.max_dist
            return
        self.rays.fill_(self.cfg.sensors.ray2d.max_dist)
        self.static_rays.fill_(self.cfg.sensors.ray2d.max_dist)
        self.dynamic_rays.fill_(self.cfg.sensors.ray2d.max_dist)
        self.predicted_dynamic_rays.fill_(self.cfg.sensors.ray2d.max_dist)

    def _check_spawn_collision(self):
        # Starts are fixed in the clear center corridor; no height query is needed.
        return

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
        fall_or_contact_candidate = fall | terminate_contact
        timeout_candidate = timeout_reached(
            self.episode_length_buf, self.max_episode_length
        )
        stair_stuck_candidate = self._stair_stuck_candidate(local_x) & ~self.initial_

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
            "terminal_reason": self.terminal_reason[env_ids].clone(),
            "stair_crossed": self.stair_crossed[env_ids].clone(),
            "episode_steps": self.episode_length_buf[env_ids].clone(),
        }
        direction = outcomes["direction"]
        reason = outcomes["terminal_reason"]
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
        self.stair_crossed[env_ids] = False
        self.fully_cleared[env_ids] = False
        self.stair_progress_anchor[env_ids] = self.root_states[env_ids, 0]
        self.stair_progress_steps[env_ids] = 0
        self.terminal_reason[env_ids] = TERMINAL_NONE
        self.correct_goal_height[env_ids] = False
        self.actions_orig[env_ids] = 0.0

    def _reward_reach_pos_target_tight(self):
        reward = super()._reward_reach_pos_target_tight()
        return reward * self.correct_goal_height * self.stair_crossed * self.fully_cleared
