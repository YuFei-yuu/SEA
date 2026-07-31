"""Proprioceptive Go2 velocity locomotion task for stair training."""

import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_from_euler_xyz

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.low_level import (
    GO2_EXTERNAL_JOINT_ORDER,
    GO2_INTERNAL_TO_EXTERNAL_INDEX,
    build_blind_stair_observation,
)


class LeggedRobotBlindStairLoco(LeggedRobot):
    """Train the exact 45-D observation contract consumed by navigation."""

    def _get_env_origins(self):
        super()._get_env_origins()
        min_level = int(getattr(self.cfg.terrain, "min_init_terrain_level", 0))
        max_level = int(self.cfg.terrain.max_init_terrain_level)
        if min_level < 0 or max_level >= self.cfg.terrain.num_rows or min_level > max_level:
            raise ValueError(
                f"Invalid blind stair initial terrain range [{min_level}, {max_level}]"
            )
        if min_level > 0:
            self.terrain_levels = torch.randint(
                min_level,
                max_level + 1,
                (self.num_envs,),
                device=self.device,
            )
            self.env_origins[:] = self.terrain_origins[
                self.terrain_levels, self.terrain_types
            ]

    def _init_buffers(self):
        super()._init_buffers()
        external_joint_order = tuple(
            self.dof_names[index] for index in GO2_INTERNAL_TO_EXTERNAL_INDEX
        )
        if external_joint_order != GO2_EXTERNAL_JOINT_ORDER:
            raise ValueError(
                "Runtime Go2 URDF joint order does not match blind stair training contract"
            )

    def _get_noise_scale_vec(self, cfg):
        noise = torch.zeros_like(self.obs_buf[0])
        scales = cfg.noise.noise_scales
        level = cfg.noise.noise_level
        noise[0:3] = scales.ang_vel * level * self.obs_scales.ang_vel
        noise[3:6] = scales.gravity * level
        noise[9:21] = scales.dof_pos * level * self.obs_scales.dof_pos
        noise[21:33] = scales.dof_vel * level * self.obs_scales.dof_vel
        self.add_noise = cfg.noise.add_noise
        return noise

    def _terrain_column_boundaries(self):
        proportions = self.cfg.terrain.terrain_proportions
        total = sum(proportions)
        flat_end = int(round(self.cfg.terrain.num_cols * proportions[0] / total))
        up_end = flat_end + int(
            round(self.cfg.terrain.num_cols * proportions[1] / total)
        )
        return flat_end, up_end

    def _reset_root_states(self, env_ids):
        super()._reset_root_states(env_ids)
        if len(env_ids) == 0:
            return
        flat_end, up_end = self._terrain_column_boundaries()
        types = self.terrain_types[env_ids]
        up_ids = env_ids[(types >= flat_end) & (types < up_end)]
        down_ids = env_ids[types >= up_end]
        for stair_ids, x_offset in ((up_ids, 1.0), (down_ids, -1.0)):
            if len(stair_ids) == 0:
                continue
            self.root_states[stair_ids, 0] = self.env_origins[stair_ids, 0] + x_offset
            self.root_states[stair_ids, 1] = self.env_origins[stair_ids, 1] + torch.empty(
                len(stair_ids), device=self.device
            ).uniform_(-0.5, 0.5)
        if len(down_ids) > 0:
            zeros = torch.zeros(len(down_ids), device=self.device)
            yaw = torch.full_like(zeros, torch.pi)
            self.root_states[down_ids, 3:7] = quat_from_euler_xyz(zeros, zeros, yaw)
            self.base_quat[down_ids] = self.root_states[down_ids, 3:7]
        stair_ids = torch.cat((up_ids, down_ids))
        if len(stair_ids) > 0:
            stair_ids_int32 = stair_ids.to(dtype=torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.root_states),
                gymtorch.unwrap_tensor(stair_ids_int32),
                len(stair_ids_int32),
            )

    def _resample_commands(self, env_ids):
        super()._resample_commands(env_ids)
        if len(env_ids) == 0:
            return
        flat_end, up_end = self._terrain_column_boundaries()
        types = self.terrain_types[env_ids]
        up_ids = env_ids[(types >= flat_end) & (types < up_end)]
        down_ids = env_ids[types >= up_end]
        gate_speeds = torch.tensor((0.25, 0.40, 0.55, 0.70), device=self.device)
        for stair_ids in (up_ids, down_ids):
            if len(stair_ids) == 0:
                continue
            if bool(getattr(self.cfg.commands, "low_speed_focus", False)):
                # Preserve all gate speeds while concentrating the final stage
                # on the command that is most likely to stall at a stair edge.
                speed_indices = torch.randint(
                    1, len(gate_speeds), (len(stair_ids),), device=self.device
                )
                low_speed = torch.rand(len(stair_ids), device=self.device) < 0.7
                speed_indices[low_speed] = 0
            else:
                speed_indices = torch.randint(
                    0, len(gate_speeds), (len(stair_ids),), device=self.device
                )
            # Both stair directions use forward body-frame velocity. Downward
            # world motion is produced by spawning the robot at yaw=pi.
            self.commands[stair_ids, 0] = gate_speeds[speed_indices]
            self.commands[stair_ids, 1:3] = 0.0

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        self.contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact

    def compute_observations(self):
        self.obs_buf = build_blind_stair_observation(
            self.base_ang_vel,
            self.projected_gravity,
            self.commands[:, :3],
            self.reindex(self.dof_pos - self.default_dof_pos),
            self.reindex(self.dof_vel),
            self.reindex(self.actions),
            lin_vel_scale=self.obs_scales.lin_vel,
            ang_vel_scale=self.obs_scales.ang_vel,
            dof_pos_scale=self.obs_scales.dof_pos,
            dof_vel_scale=self.obs_scales.dof_vel,
        )
        if self.add_noise:
            self.obs_buf += (2.0 * torch.rand_like(self.obs_buf) - 1.0) * self.noise_scale_vec

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if len(env_ids) > 0:
            self.actions[env_ids] = 0.0
            self.last_actions[env_ids] = 0.0

    def check_termination(self):
        contact = torch.any(
            torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0,
            dim=1,
        )
        self.time_out_buf = self.episode_length_buf >= self.max_episode_length
        self.reset_buf = contact | self.time_out_buf
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def _reward_tracking_lin_vel(self):
        error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-error / self.cfg.rewards.tracking_sigma)

    def _reward_directional_progress(self):
        """Keep a dense velocity gradient when a robot stalls at a stair edge."""
        command_xy = self.commands[:, :2]
        command_norm = torch.norm(command_xy, dim=1)
        command_direction = command_xy / torch.clamp(command_norm[:, None], min=0.1)
        progress_velocity = torch.sum(command_direction * self.base_lin_vel[:, :2], dim=1)
        progress_velocity = torch.clamp(progress_velocity, min=-1.0, max=1.0)
        return progress_velocity * (command_norm > 0.1)

    def _reward_joint_pos_penalty(self):
        penalty = torch.norm(self.dof_pos - self.default_dof_pos, dim=1)
        upright = torch.clamp(-self.projected_gravity[:, 2], 0.0, 0.7) / 0.7
        return penalty * upright

    def _reward_feet_height_body(self):
        target_height = float(self.cfg.rewards.feet_height_body_target)
        relative_height = self.feet_pos[:, :, 2] - self.root_states[:, None, 2]
        relative_xy_velocity = self.feet_vel[:, :, :2] - self.root_states[:, None, 7:9]
        moving_weight = torch.tanh(2.0 * torch.norm(relative_xy_velocity, dim=2))
        swing_weight = (~self.contact_filt).float()
        penalty = torch.sum(
            torch.square(relative_height - target_height) * moving_weight * swing_weight,
            dim=1,
        )
        return penalty * (torch.norm(self.commands[:, :2], dim=1) > 0.1)

    def _reward_feet_slip(self):
        tangential_speed_sq = torch.sum(torch.square(self.feet_vel[:, :, :2]), dim=2)
        return torch.sum(tangential_speed_sq * self.contact_filt.float(), dim=1)

    def _reward_torque_peaks(self):
        threshold = float(self.cfg.rewards.soft_torque_peak) * self.torque_limits
        excess = torch.clamp(torch.abs(self.torques) - threshold, min=0.0)
        return torch.sum(torch.square(excess), dim=1)

    def _reward_undesired_contacts(self):
        contacts = torch.norm(
            self.contact_forces[:, self.penalised_contact_indices, :], dim=-1
        ) > 1.0
        return torch.sum(contacts.float(), dim=1)
