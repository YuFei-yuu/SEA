from time import time
import os
import numpy as np

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot_pos import LeggedRobotPos
from legged_gym.utils.torch_math import yaw_quat, quat_apply_yaw, circle_ray_query


class LeggedRobotPosDynamic(LeggedRobotPos):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _dynamic_cfg(self):
        return getattr(self.cfg, "dynamic_obstacles", None)

    def _sparse_cfg(self):
        return getattr(self.cfg, "sparse_room", None)

    def _dynamic_enabled(self):
        cfg = self._dynamic_cfg()
        return cfg is not None and getattr(cfg, "enable", False) and getattr(cfg, "count", 0) > 0

    def _get_env_origins(self):
        self.custom_origins = True
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        self.position_targets = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        self.room_origins = torch.zeros(self.num_envs, 2, device=self.device, requires_grad=False)
        self.terrain_levels = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.terrain_types = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.goal_levels = torch.zeros(self.num_envs, device=self.device)
        self.max_terrain_level = self.cfg.terrain.num_rows
        self.ori_z = torch.zeros(self.num_envs, 1, device=self.device)
        self._resample_sparse_navigation(torch.arange(self.num_envs, device=self.device))

    def _is_point_clear_of_pillars(self, x, y):
        cfg = self._sparse_cfg()
        half_x = cfg.pillar_size[0] * 0.5
        half_y = cfg.pillar_size[1] * 0.5
        clearance = cfg.spawn_clearance
        for center in cfg.pillar_centers:
            if abs(x - center[0]) <= half_x + clearance and abs(y - center[1]) <= half_y + clearance:
                return False
        return True

    def _sample_sparse_start_goal(self):
        cfg = self._sparse_cfg()
        for _ in range(512):
            left_to_right = np.random.rand() < 0.5
            if left_to_right:
                sx = np.random.uniform(cfg.start_x_left[0], cfg.start_x_left[1])
                gx = np.random.uniform(cfg.goal_x_right[0], cfg.goal_x_right[1])
            else:
                sx = np.random.uniform(cfg.start_x_right[0], cfg.start_x_right[1])
                gx = np.random.uniform(cfg.goal_x_left[0], cfg.goal_x_left[1])
            sy = np.random.uniform(cfg.start_y_range[0], cfg.start_y_range[1])
            gy = np.random.uniform(cfg.goal_y_range[0], cfg.goal_y_range[1])
            if self._is_point_clear_of_pillars(sx, sy) and self._is_point_clear_of_pillars(gx, gy):
                return (sx, sy), (gx, gy)
        return (1.4, 5.0), (8.6, 5.0)

    def _resample_sparse_navigation(self, env_ids):
        if len(env_ids) == 0:
            return
        rows = torch.randint(0, self.cfg.terrain.num_rows, (len(env_ids),), device=self.device)
        cols = torch.randint(0, self.cfg.terrain.num_cols, (len(env_ids),), device=self.device)
        self.terrain_levels[env_ids] = rows
        self.terrain_types[env_ids] = cols
        for local_idx, env_id in enumerate(env_ids.tolist()):
            row = int(rows[local_idx])
            col = int(cols[local_idx])
            room_origin_x = row * self.terrain.env_length
            room_origin_y = col * self.terrain.env_width
            self.room_origins[env_id, 0] = room_origin_x
            self.room_origins[env_id, 1] = room_origin_y
            start_xy, goal_xy = self._sample_sparse_start_goal()
            self.env_origins[env_id, 0] = room_origin_x + start_xy[0]
            self.env_origins[env_id, 1] = room_origin_y + start_xy[1]
            self.position_targets[env_id, 0] = room_origin_x + goal_xy[0]
            self.position_targets[env_id, 1] = room_origin_y + goal_xy[1]

    def _create_envs(self):
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        self.dynamic_obs_count = int(getattr(self._dynamic_cfg(), "count", 0)) if self._dynamic_enabled() else 0
        obstacle_asset = None
        if self.dynamic_obs_count > 0:
            obs_cfg = self._dynamic_cfg()
            obstacle_options = gymapi.AssetOptions()
            obstacle_options.disable_gravity = getattr(obs_cfg, "kinematic", True)
            obstacle_options.fix_base_link = False
            obstacle_options.density = 1.0
            obstacle_options.angular_damping = 0.0
            obstacle_options.linear_damping = 0.0
            obstacle_asset = self.gym.create_box(
                self.sim,
                obs_cfg.size[0],
                obs_cfg.size[1],
                obs_cfg.size[2],
                obstacle_options,
            )

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.dynamic_actor_handles = []
        self.envs = []
        self.robot_actor_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.dynamic_actor_indices = torch.zeros(self.num_envs, self.dynamic_obs_count, dtype=torch.long, device=self.device)

        self.gym.set_light_parameters(self.sim, 3, gymapi.Vec3(0.5, 0.5, 0.5), gymapi.Vec3(0.3, 0.3, 0.3), gymapi.Vec3(-1, -1, -1))
        self.gym.set_light_parameters(self.sim, 0, gymapi.Vec3(0.4, 0.4, 0.4), gymapi.Vec3(0.5, 0.5, 0.5), gymapi.Vec3(5, 5, 15))

        for i in range(self.num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)

            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)

            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)
            self.robot_actor_indices[i] = self.gym.get_actor_index(env_handle, actor_handle, gymapi.DOMAIN_SIM)

            env_dynamic_handles = []
            for dyn_idx in range(self.dynamic_obs_count):
                dyn_pose = gymapi.Transform()
                dyn_pose.p = gymapi.Vec3(float(self.room_origins[i, 0] + 5.0), float(self.room_origins[i, 1] + 5.0), float(self._dynamic_cfg().size[2] * 0.5))
                dyn_handle = self.gym.create_actor(env_handle, obstacle_asset, dyn_pose, f"dynamic_obstacle_{dyn_idx}", i, 0, 0)
                self.gym.set_rigid_body_color(env_handle, dyn_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.95, 0.45, 0.15))
                env_dynamic_handles.append(dyn_handle)
                self.dynamic_actor_indices[i, dyn_idx] = self.gym.get_actor_index(env_handle, dyn_handle, gymapi.DOMAIN_SIM)
            self.dynamic_actor_handles.append(env_dynamic_handles)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    def _init_buffers(self):
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self._root_state_tensor = gymtorch.wrap_tensor(actor_root_state)
        self.actors_per_env = 1 + self.dynamic_obs_count
        self.all_root_states = self._root_state_tensor.view(self.num_envs, self.actors_per_env, 13)
        self.root_states = self.all_root_states[:, 0, :]
        self.dynamic_root_states = self.all_root_states[:, 1:, :]

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self.feet_pos = self.rigid_body_states[:, self.feet_indices, 0:3]
        self.feet_vel = self.rigid_body_states[:, self.feet_indices, 7:10]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.contact_filt = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.last_base_twist = torch.zeros_like(self.root_states[:, 7:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        self.dof_bias = torch.zeros(self.num_envs, self.num_actions, device=self.device)

        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()

        # LeggedRobotPos buffers
        self.obs_history_buf = torch.zeros(self.num_envs, self.cfg.env.his_len, self.cfg.env.num_obs_one_step, device=self.device, dtype=torch.float)
        self.slr_obs_buf = torch.zeros(self.num_envs, self.cfg.loco.num_obs_buf, device=self.device, dtype=torch.float)
        self.slr_obs_hist = torch.zeros(self.num_envs, self.cfg.loco.his_len, self.cfg.loco.num_obs_buf, device=self.device, dtype=torch.float)
        self.base_lin_vel_pred = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
        self.actions_orig = self.actions.clone()
        self._init_replay_buffers()

        self.pos_hist = torch.zeros(self.num_envs, self.cfg.env.his_len, 2, device=self.device, dtype=torch.float)
        self.delay_goal = torch.zeros(self.num_envs, 2, device=self.device, requires_grad=False)
        self.goal_hist = torch.zeros(self.num_envs, self.cfg.env.his_len, 2, device=self.device, dtype=torch.float)
        self.heading_targets = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)

        self.len_y = len(self.cfg.terrain.measured_points_y)
        self.len_x = len(self.cfg.terrain.measured_points_x)
        self.c_y = int(self.len_y / 2)
        try:
            self.c_x = self.cfg.terrain.measured_points_x.index(0.0)
        except ValueError:
            self.c_x = np.argmin(np.abs(np.array(self.cfg.terrain.measured_points_x)))

        self.reach_goal = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool, requires_grad=False)
        self.distance = torch.zeros(self.num_envs, device=self.device, dtype=torch.float, requires_grad=False)
        self.goal_local_pos = torch.zeros(self.num_envs, 2, device=self.device, requires_grad=False)

        self.ray_angles = torch.arange(
            start=self.cfg.sensors.ray2d.theta_start,
            end=self.cfg.sensors.ray2d.theta_end,
            step=self.cfg.sensors.ray2d.theta_step,
            device=self.device,
        )
        self.rays = torch.ones(self.num_envs, self.ray_angles.shape[0], dtype=torch.float, device=self.device, requires_grad=False) * 5.0
        self.static_rays = self.rays.clone()
        self.dynamic_rays = self.rays.clone()
        self.delay_rays = self.rays.clone()
        self.nav_clip_min = torch.tensor([
            self.cfg.commands.ranges.limit_vx[0],
            self.cfg.commands.ranges.limit_vy[0],
            self.cfg.commands.ranges.limit_vyaw[0],
        ], dtype=torch.float, device=self.device, requires_grad=False)
        self.nav_clip_max = torch.tensor([
            self.cfg.commands.ranges.limit_vx[1],
            self.cfg.commands.ranges.limit_vy[1],
            self.cfg.commands.ranges.limit_vyaw[1],
        ], dtype=torch.float, device=self.device, requires_grad=False)
        self.nav_actions_filtered = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        self.rays_hist = torch.ones(self.num_envs, self.cfg.env.his_len, self.ray_angles.shape[0], device=self.device, dtype=torch.float) * 5.0
        self.goal_hold_timer = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)
        self.stay_timer = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)
        self.goal_reached_flag = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.stand_still_flag = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # Dynamic obstacle buffers
        self.dynamic_local_pos = torch.zeros(self.num_envs, self.dynamic_obs_count, 2, device=self.device, dtype=torch.float)
        self.dynamic_dirs = torch.ones(self.num_envs, self.dynamic_obs_count, device=self.device, dtype=torch.float)
        self.dynamic_speeds = torch.zeros(self.num_envs, self.dynamic_obs_count, device=self.device, dtype=torch.float)
        self.body_collision_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.dynamic_collision_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.total_collision_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.body_collision_cooldown = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.dynamic_collision_cooldown = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.body_collision_event = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.dynamic_collision_event = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

    def _set_actor_root_states_indexed(self, actor_indices):
        if actor_indices.numel() == 0:
            return
        actor_indices_int32 = actor_indices.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_state_tensor),
            gymtorch.unwrap_tensor(actor_indices_int32),
            len(actor_indices_int32),
        )

    def _set_robot_root_states(self, env_ids):
        self._set_actor_root_states_indexed(self.robot_actor_indices[env_ids])

    def _set_dynamic_root_states(self, env_ids):
        if self.dynamic_obs_count == 0 or len(env_ids) == 0:
            return
        self._set_actor_root_states_indexed(self.dynamic_actor_indices[env_ids].reshape(-1))

    def _reset_dofs(self, env_ids):
        if len(env_ids) == 0:
            return
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(1.0, 1.0, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        actor_indices_int32 = self.robot_actor_indices[env_ids].to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(actor_indices_int32),
            len(actor_indices_int32),
        )

    def _reset_dynamic_obstacles(self, env_ids):
        if self.dynamic_obs_count == 0 or len(env_ids) == 0:
            return
        cfg = self._dynamic_cfg()
        lane_x = torch.tensor(cfg.lane_x[:self.dynamic_obs_count], device=self.device, dtype=torch.float).unsqueeze(0).repeat(len(env_ids), 1)
        lane_x += torch_rand_float(-cfg.lane_x_jitter, cfg.lane_x_jitter, (len(env_ids), self.dynamic_obs_count), device=self.device)
        lane_x = lane_x.clamp(min=0.8, max=9.2)
        y_local = torch_rand_float(cfg.y_min, cfg.y_max, (len(env_ids), self.dynamic_obs_count), device=self.device)
        dirs = torch.where(torch.rand(len(env_ids), self.dynamic_obs_count, device=self.device) > 0.5, 1.0, -1.0)
        speeds = torch_rand_float(cfg.speed_range[0], cfg.speed_range[1], (len(env_ids), self.dynamic_obs_count), device=self.device)

        self.dynamic_local_pos[env_ids, :, 0] = lane_x
        self.dynamic_local_pos[env_ids, :, 1] = y_local
        self.dynamic_dirs[env_ids] = dirs
        self.dynamic_speeds[env_ids] = speeds

        world_xy = self.room_origins[env_ids].unsqueeze(1) + self.dynamic_local_pos[env_ids]
        self.dynamic_root_states[env_ids, :, 0:2] = world_xy
        self.dynamic_root_states[env_ids, :, 2] = cfg.size[2] * 0.5
        self.dynamic_root_states[env_ids, :, 3:7] = 0.0
        self.dynamic_root_states[env_ids, :, 6] = 1.0
        self.dynamic_root_states[env_ids, :, 7:10] = 0.0
        self.dynamic_root_states[env_ids, :, 8] = self.dynamic_dirs[env_ids] * self.dynamic_speeds[env_ids]
        self.dynamic_root_states[env_ids, :, 10:13] = 0.0
        self._set_dynamic_root_states(env_ids)

    def _reset_root_states(self, env_ids):
        self._resample_sparse_navigation(env_ids)
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device)

        if self.cfg.domain_rand.randomize_yaw:
            yaw = torch.zeros_like(self.root_states[env_ids, 3]).uniform_(self.cfg.domain_rand.init_yaw_range[0], self.cfg.domain_rand.init_yaw_range[1])
        else:
            yaw = torch.zeros_like(self.root_states[env_ids, 3])
        if self.cfg.domain_rand.randomize_roll:
            roll = torch.zeros_like(self.root_states[env_ids, 3]).uniform_(self.cfg.domain_rand.init_roll_range[0], self.cfg.domain_rand.init_roll_range[1])
        else:
            roll = torch.zeros_like(self.root_states[env_ids, 3])
        if self.cfg.domain_rand.randomize_pitch:
            pitch = torch.zeros_like(self.root_states[env_ids, 3]).uniform_(self.cfg.domain_rand.init_pitch_range[0], self.cfg.domain_rand.init_pitch_range[1])
        else:
            pitch = torch.zeros_like(self.root_states[env_ids, 3])
        self.root_states[env_ids, 3:7] = quat_from_euler_xyz(roll, pitch, yaw)
        self.base_quat[env_ids] = self.root_states[env_ids, 3:7]
        self._set_robot_root_states(env_ids)
        self._reset_dynamic_obstacles(env_ids)

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        episode_len_s = self.episode_length_buf[env_ids].float() * self.dt
        success = self.goal_reached_flag[env_ids].float()
        timeout = self.time_out_buf[env_ids].float()
        total_collisions = self.total_collision_count[env_ids].float()
        body_collisions = self.body_collision_count[env_ids].float()
        dynamic_collisions = self.dynamic_collision_count[env_ids].float()
        safe_success = success * (total_collisions == 0).float()

        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.obs_history_buf[env_ids, :, :] = 0.
        self.slr_obs_hist[env_ids, :, :] = 0.
        self.rays_hist[env_ids, :, :] = 5.
        self.pos_hist[env_ids, :, :] = 0.
        self.goal_hist[env_ids, :, :] = 0.
        self.reset_buf[env_ids] = 1
        self.goal_reached_flag[env_ids] = 0
        self.stand_still_flag[env_ids] = 0
        self.goal_hold_timer[env_ids] = 0
        self.stay_timer[env_ids] = 0
        self.reach_goal[env_ids] = 0
        self.nav_actions_filtered[env_ids] = 0.
        self.contact_filt[env_ids] = False
        self.last_contacts[env_ids] = False
        self.collision_occurred[env_ids] = False
        self.last_collision_active[env_ids] = False
        self.num_collisions[env_ids] = 0
        self.collision_pos_hist[env_ids] = 0
        self.body_collision_count[env_ids] = 0
        self.dynamic_collision_count[env_ids] = 0
        self.total_collision_count[env_ids] = 0
        self.body_collision_cooldown[env_ids] = 0
        self.dynamic_collision_cooldown[env_ids] = 0
        self.body_collision_event[env_ids] = False
        self.dynamic_collision_event[env_ids] = False

        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        self.extras["episode"]["success"] = torch.mean(success)
        self.extras["episode"]["safe_success"] = torch.mean(safe_success)
        self.extras["episode"]["timeout"] = torch.mean(timeout)
        self.extras["episode"]["dynamic_collision_count"] = torch.mean(dynamic_collisions)
        self.extras["episode"]["body_collision_count"] = torch.mean(body_collisions)
        self.extras["episode"]["total_collision_count"] = torch.mean(total_collisions)
        self.extras["episode"]["episode_duration"] = torch.mean(episode_len_s)
        if torch.any(success > 0):
            self.extras["episode"]["time_to_goal"] = torch.mean(episode_len_s[success > 0])
        else:
            self.extras["episode"]["time_to_goal"] = torch.tensor(0.0, device=self.device)
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def _update_dynamic_obstacles(self):
        if self.dynamic_obs_count == 0:
            return
        cfg = self._dynamic_cfg()
        self.dynamic_local_pos[:, :, 1] += self.dynamic_dirs * self.dynamic_speeds * self.dt
        hit_low = self.dynamic_local_pos[:, :, 1] < cfg.y_min
        hit_high = self.dynamic_local_pos[:, :, 1] > cfg.y_max
        bounce = hit_low | hit_high
        self.dynamic_dirs = torch.where(bounce, -self.dynamic_dirs, self.dynamic_dirs)
        self.dynamic_local_pos[:, :, 1] = torch.where(hit_low, 2 * cfg.y_min - self.dynamic_local_pos[:, :, 1], self.dynamic_local_pos[:, :, 1])
        self.dynamic_local_pos[:, :, 1] = torch.where(hit_high, 2 * cfg.y_max - self.dynamic_local_pos[:, :, 1], self.dynamic_local_pos[:, :, 1])
        world_xy = self.room_origins.unsqueeze(1) + self.dynamic_local_pos
        self.dynamic_root_states[:, :, 0:2] = world_xy
        self.dynamic_root_states[:, :, 2] = cfg.size[2] * 0.5
        self.dynamic_root_states[:, :, 7:10] = 0.0
        self.dynamic_root_states[:, :, 8] = self.dynamic_dirs * self.dynamic_speeds
        self._set_dynamic_root_states(torch.arange(self.num_envs, device=self.device))

    def _post_physics_step_callback(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        self.contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        self._update_dynamic_obstacles()
        self.update_percetion()

    def update_percetion(self):
        self.distance = torch.norm(self.position_targets[:, :2] - self.root_states[:, :2], dim=1)
        self.far_goal = self.distance > 0.5
        self._get_rays()

    def _get_dynamic_rays(self):
        if self.dynamic_obs_count == 0:
            return torch.ones_like(self.rays) * self.cfg.sensors.ray2d.max_dist
        robot_pos = self.root_states[:, :3].unsqueeze(1)
        local_pos = quat_rotate_inverse(yaw_quat(self.base_quat).unsqueeze(1).repeat(1, self.dynamic_obs_count, 1).view(-1, 4), (self.dynamic_root_states[:, :, :3] - robot_pos).reshape(-1, 3))
        centers = local_pos[:, :2].view(self.num_envs, self.dynamic_obs_count, 2)
        x0 = torch.zeros((self.num_envs, 1), device=self.device)
        y0 = torch.zeros((self.num_envs, 1), device=self.device)
        theta = self.ray_angles.unsqueeze(0).repeat(self.num_envs, 1)
        ray_dists = []
        for idx in range(self.dynamic_obs_count):
            ray_dists.append(circle_ray_query(x0, y0, theta, centers[:, idx, :], self._dynamic_cfg().ray_radius, self.cfg.sensors.ray2d.min_dist, self.cfg.sensors.ray2d.max_dist))
        return torch.stack(ray_dists, dim=0).min(dim=0).values

    def _get_rays(self, env_ids=None):
        if not hasattr(self.terrain, 'height_points'):
            self.height_points = self._init_height_points()

        if env_ids is not None:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights = torch.max(heights1, heights2)
        self.measured_heights = heights.view(self.num_envs, self.len_x, self.len_y) * self.terrain.cfg.vertical_scale
        center_height = self.measured_heights[:, self.c_x, self.c_y].unsqueeze(1).unsqueeze(2)
        raw_heights = torch.where(self.measured_heights > center_height + 0.1, 1.0, 0.0)
        self.static_rays = self._grid2ray(raw_heights) * self.cfg.terrain.measure_resolution
        self.dynamic_rays = self._get_dynamic_rays()
        self.rays = torch.minimum(self.static_rays, self.dynamic_rays)

    def _update_collision_events(self, collision_now, cooldown_buf, count_buf):
        cooldown_buf[:] = torch.clamp(cooldown_buf - 1, min=0)
        new_event = collision_now & (cooldown_buf == 0)
        count_buf += new_event.long()
        cooldown_buf[new_event] = self.cfg.rewards.dynamic_collision_config.cooldown_steps
        return new_event

    def check_termination(self):
        self.initial_ = self.episode_length_buf <= 1
        self.reach_goal = self.distance < self.cfg.rewards.reach_pos_target_tight_config.distance_threshold

        self.terminate_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :2], dim=-1) > 1.0, dim=1)
        self.terminate_buf &= (~self.initial_)
        self.reset_buf = self.terminate_buf.clone()
        self.reset_buf |= torch.any(torch.norm(self.contact_forces[:, :, :2], dim=-1) > 50.0, dim=1)
        if self.initial_.any():
            self._check_spawn_collision()
        self.extras["bad_masks"] = self.initial_

        body_collision_now = torch.any(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :2], dim=-1) > 1.0, dim=1)
        body_collision_now &= (~self.initial_)
        self.body_collision_event = self._update_collision_events(body_collision_now, self.body_collision_cooldown, self.body_collision_count)

        if self.dynamic_obs_count > 0:
            dist = torch.norm(self.dynamic_root_states[:, :, :2] - self.root_states[:, None, :2], dim=-1)
            dynamic_collision_now = torch.any(dist < self.cfg.rewards.dynamic_collision_config.threshold, dim=1)
            dynamic_collision_now &= (~self.initial_)
            self.dynamic_collision_event = self._update_collision_events(dynamic_collision_now, self.dynamic_collision_cooldown, self.dynamic_collision_count)
        else:
            self.dynamic_collision_event = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        self.total_collision_count = self.body_collision_count + self.dynamic_collision_count
        self.collision_occurred |= body_collision_now | self.dynamic_collision_event
        self.last_collision_active = body_collision_now

        if torch.any(body_collision_now):
            self._update_collision_hist(body_collision_now)

        if self.cfg.rewards.dynamic_collision_config.early_reset:
            self.reset_buf |= self.dynamic_collision_event
            self.terminate_buf |= self.dynamic_collision_event

        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.fall_down = self.projected_gravity[:, 2] > -0.8

        v_low = (torch.norm(self.base_lin_vel[:, :2], dim=-1) < 0.1) & (torch.abs(self.base_ang_vel[:, 2]) < 0.1)
        d_low = torch.norm(self.root_states[:, :2] - self.pos_hist[:, 0, :2], dim=-1) < 0.2
        self.not_just_reset = (self.episode_length_buf / self.max_episode_length) > 0.1
        self.static = (v_low | d_low) & self.not_just_reset

        self.goal_hold_timer = self.goal_hold_timer + self.reach_goal.int()
        self.stay_timer = self.stay_timer + self.static.int()
        self.goal_reached_flag = self.goal_hold_timer >= self.cfg.env.goal_reached_time
        self.stand_still_flag = self.stay_timer >= self.cfg.env.stay_time

        self.reset_buf |= self.goal_reached_flag
        self.reset_buf |= self.stand_still_flag
        self.reset_buf |= self.time_out_buf
        self.reset_buf |= self.fall_down

    def _reward_dynamic_collision(self):
        return self.dynamic_collision_event.float()
