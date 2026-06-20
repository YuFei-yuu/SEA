import math
import os
import numpy as np

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot_pos import LeggedRobotPos
from legged_gym.utils.torch_math import yaw_quat, quat_apply_yaw, circle_ray_query


class LeggedRobotPosDynamic(LeggedRobotPos):
    MOTION_LINEAR_CROSSING = 0
    MOTION_LINEAR_DIAGONAL = 1
    MOTION_CIRCULAR = 2
    MOTION_FIGURE_EIGHT = 3

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _dynamic_cfg(self):
        return getattr(self.cfg, "dynamic_obstacles", None)

    def _sparse_cfg(self):
        return getattr(self.cfg, "sparse_room", None)

    def _dynamic_enabled(self):
        cfg = self._dynamic_cfg()
        return (
            cfg is not None
            and getattr(cfg, "enable", False)
            and getattr(cfg, "count", 0) > 0
        )

    def _dynamic_count_range(self):
        cfg = self._dynamic_cfg()
        count = int(getattr(cfg, "count", 0))
        count_range = getattr(cfg, "count_range", [count, count])
        if count_range is None:
            return count, count
        low = int(count_range[0])
        high = int(count_range[1])
        high = min(high, count)
        low = min(low, high)
        return low, high

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
        self._resample_sparse_navigation(torch.arange(self.num_envs, device=self.device))

    def _is_point_clear_of_pillars(self, x, y, extra_clearance=0.0):
        cfg = self._sparse_cfg()
        clearance = cfg.spawn_clearance + extra_clearance
        obstacle_boxes = getattr(cfg, "obstacle_boxes", None)
        if obstacle_boxes is None:
            obstacle_boxes = [
                (center[0], center[1], cfg.pillar_size[0], cfg.pillar_size[1], cfg.pillar_size[2])
                for center in cfg.pillar_centers
            ]
        for center_x, center_y, size_x, size_y, _ in obstacle_boxes:
            half_x = size_x * 0.5
            half_y = size_y * 0.5
            if (
                abs(x - center_x) <= half_x + clearance
                and abs(y - center_y) <= half_y + clearance
            ):
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
            if self._is_point_clear_of_pillars(sx, sy) and self._is_point_clear_of_pillars(
                gx, gy
            ):
                return (sx, sy), (gx, gy)
        return (1.4, 5.0), (8.6, 5.0)

    def _resample_sparse_navigation(self, env_ids):
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
        asset_path = self.cfg.asset.file.format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR
        )
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = (
            self.cfg.asset.replace_cylinder_with_capsule
        )
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

        self.dynamic_obs_count = (
            int(getattr(self._dynamic_cfg(), "count", 0)) if self._dynamic_enabled() else 0
        )
        obstacle_asset = None
        if self.dynamic_obs_count > 0:
            obs_cfg = self._dynamic_cfg()
            obstacle_options = gymapi.AssetOptions()
            obstacle_options.disable_gravity = getattr(obs_cfg, "kinematic", True)
            obstacle_options.fix_base_link = getattr(obs_cfg, "fixed_base_link", False)
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

        base_init_state_list = (
            self.cfg.init_state.pos
            + self.cfg.init_state.rot
            + self.cfg.init_state.lin_vel
            + self.cfg.init_state.ang_vel
        )
        self.base_init_state = to_torch(
            base_init_state_list, device=self.device, requires_grad=False
        )
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        self.actor_handles = []
        self.dynamic_actor_handles = []
        self.envs = []
        self.robot_actor_indices = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.dynamic_actor_indices = torch.zeros(
            self.num_envs, self.dynamic_obs_count, dtype=torch.long, device=self.device
        )

        self.gym.set_light_parameters(
            self.sim,
            3,
            gymapi.Vec3(0.5, 0.5, 0.5),
            gymapi.Vec3(0.3, 0.3, 0.3),
            gymapi.Vec3(-1, -1, -1),
        )
        self.gym.set_light_parameters(
            self.sim,
            0,
            gymapi.Vec3(0.4, 0.4, 0.4),
            gymapi.Vec3(0.5, 0.5, 0.5),
            gymapi.Vec3(5, 5, 15),
        )

        for i in range(self.num_envs):
            env_handle = self.gym.create_env(
                self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs))
            )
            pos = self.env_origins[i].clone()
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(
                rigid_shape_props_asset, i
            )
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(
                env_handle,
                robot_asset,
                start_pose,
                self.cfg.asset.name,
                i,
                self.cfg.asset.self_collisions,
                0,
            )

            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(
                env_handle, actor_handle, body_props, recomputeInertia=True
            )

            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)
            self.robot_actor_indices[i] = self.gym.get_actor_index(
                env_handle, actor_handle, gymapi.DOMAIN_SIM
            )

            env_dynamic_handles = []
            for dyn_idx in range(self.dynamic_obs_count):
                dyn_pose = gymapi.Transform()
                dyn_pose.p = gymapi.Vec3(
                    float(self.room_origins[i, 0] + 5.0),
                    float(self.room_origins[i, 1] + 5.0),
                    float(self._dynamic_cfg().size[2] * 0.5),
                )
                dyn_handle = self.gym.create_actor(
                    env_handle,
                    obstacle_asset,
                    dyn_pose,
                    f"dynamic_obstacle_{dyn_idx}",
                    i,
                    int(getattr(self._dynamic_cfg(), "collision_filter", 0)),
                    0,
                )
                self.gym.set_rigid_body_color(
                    env_handle,
                    dyn_handle,
                    0,
                    gymapi.MESH_VISUAL_AND_COLLISION,
                    gymapi.Vec3(0.95, 0.45, 0.15),
                )
                env_dynamic_handles.append(dyn_handle)
                self.dynamic_actor_indices[i, dyn_idx] = self.gym.get_actor_index(
                    env_handle, dyn_handle, gymapi.DOMAIN_SIM
                )
            self.dynamic_actor_handles.append(env_dynamic_handles)

        self.feet_indices = torch.zeros(
            len(feet_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], feet_names[i]
            )

        self.penalised_contact_indices = torch.zeros(
            len(penalized_contact_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], penalized_contact_names[i]
            )

        self.termination_contact_indices = torch.zeros(
            len(termination_contact_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], termination_contact_names[i]
            )

    def _init_replay_buffers(self):
        super()._init_replay_buffers()
        self.replay_dynamic_root_states = torch.zeros(
            self.num_envs,
            self.replay_len,
            self.dynamic_obs_count,
            13,
            device=self.device,
            dtype=torch.float,
        )
        self.replay_dynamic_local_pos = torch.zeros(
            self.num_envs,
            self.replay_len,
            self.dynamic_obs_count,
            2,
            device=self.device,
            dtype=torch.float,
        )
        self.replay_dynamic_velocity_vectors = torch.zeros(
            self.num_envs,
            self.replay_len,
            self.dynamic_obs_count,
            2,
            device=self.device,
            dtype=torch.float,
        )
        self.replay_dynamic_motion_types = torch.full(
            (self.num_envs, self.replay_len, self.dynamic_obs_count),
            -1,
            device=self.device,
            dtype=torch.long,
        )
        self.replay_dynamic_centers = torch.zeros(
            self.num_envs,
            self.replay_len,
            self.dynamic_obs_count,
            2,
            device=self.device,
            dtype=torch.float,
        )
        self.replay_dynamic_phase = torch.zeros(
            self.num_envs,
            self.replay_len,
            self.dynamic_obs_count,
            device=self.device,
            dtype=torch.float,
        )
        self.replay_dynamic_phase_speed = torch.zeros(
            self.num_envs,
            self.replay_len,
            self.dynamic_obs_count,
            device=self.device,
            dtype=torch.float,
        )
        self.replay_dynamic_shape_params = torch.zeros(
            self.num_envs,
            self.replay_len,
            self.dynamic_obs_count,
            4,
            device=self.device,
            dtype=torch.float,
        )
        self.replay_dynamic_active_mask = torch.zeros(
            self.num_envs,
            self.replay_len,
            self.dynamic_obs_count,
            device=self.device,
            dtype=torch.bool,
        )
        self.replay_dynamic_traj_step = torch.zeros(
            self.num_envs,
            self.replay_len,
            device=self.device,
            dtype=torch.long,
        )

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
        self.all_root_states = self._root_state_tensor.view(
            self.num_envs, self.actors_per_env, 13
        )
        self.root_states = self.all_root_states[:, 0, :]
        self.dynamic_root_states = self.all_root_states[:, 1:, :]

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state_tensor).view(
            self.num_envs, -1, 13
        )
        self.feet_pos = self.rigid_body_states[:, self.feet_indices, 0:3]
        self.feet_vel = self.rigid_body_states[:, self.feet_indices, 7:10]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(
            self.num_envs, -1, 3
        )
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(
            get_axis_params(-1.0, self.up_axis_idx), device=self.device
        ).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1.0, 0.0, 0.0], device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.torques = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.p_gains = torch.zeros(
            self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.d_gains = torch.zeros(
            self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.actions = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_actions = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.feet_air_time = torch.zeros(
            self.num_envs,
            self.feet_indices.shape[0],
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.commands = torch.zeros(
            self.num_envs,
            self.cfg.commands.num_commands,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.commands_scale = torch.tensor(
            [self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],
            device=self.device,
            requires_grad=False,
        )
        self.last_contacts = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )
        self.contact_filt = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 10:13]
        )
        self.last_base_twist = torch.zeros_like(self.root_states[:, 7:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.default_dof_pos = torch.zeros(
            self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
        )
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
                self.p_gains[i] = 0.0
                self.d_gains[i] = 0.0
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        self.dof_bias = torch.zeros(self.num_envs, self.num_actions, device=self.device)

        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()

        self.obs_history_buf = torch.zeros(
            self.num_envs,
            self.cfg.env.his_len,
            self.cfg.env.num_obs_one_step,
            device=self.device,
            dtype=torch.float,
        )
        self.slr_obs_buf = torch.zeros(
            self.num_envs, self.cfg.loco.num_obs_buf, device=self.device, dtype=torch.float
        )
        self.slr_obs_hist = torch.zeros(
            self.num_envs,
            self.cfg.loco.his_len,
            self.cfg.loco.num_obs_buf,
            device=self.device,
            dtype=torch.float,
        )
        self.base_lin_vel_pred = torch.zeros(
            self.num_envs, 3, device=self.device, dtype=torch.float
        )
        self.actions_orig = self.actions.clone()
        self._init_replay_buffers()

        self.pos_hist = torch.zeros(
            self.num_envs, self.cfg.env.his_len, 2, device=self.device, dtype=torch.float
        )
        self.delay_goal = torch.zeros(
            self.num_envs, 2, device=self.device, requires_grad=False
        )
        self.goal_hist = torch.zeros(
            self.num_envs, self.cfg.env.his_len, 2, device=self.device, dtype=torch.float
        )
        self.heading_targets = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False
        )

        self.len_y = len(self.cfg.terrain.measured_points_y)
        self.len_x = len(self.cfg.terrain.measured_points_x)
        self.c_y = int(self.len_y / 2)
        try:
            self.c_x = self.cfg.terrain.measured_points_x.index(0.0)
        except ValueError:
            self.c_x = np.argmin(np.abs(np.array(self.cfg.terrain.measured_points_x)))

        self.reach_goal = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool, requires_grad=False
        )
        self.distance = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float, requires_grad=False
        )
        self.prev_distance = torch.zeros_like(self.distance)
        self.goal_local_pos = torch.zeros(
            self.num_envs, 2, device=self.device, requires_grad=False
        )

        self.ray_angles = torch.arange(
            start=self.cfg.sensors.ray2d.theta_start,
            end=self.cfg.sensors.ray2d.theta_end,
            step=self.cfg.sensors.ray2d.theta_step,
            device=self.device,
        )
        self.rays = torch.ones(
            self.num_envs,
            self.ray_angles.shape[0],
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        ) * 5.0
        self.static_rays = self.rays.clone()
        self.dynamic_rays = self.rays.clone()
        self.predicted_dynamic_rays = self.rays.clone()
        self.delay_rays = self.rays.clone()
        self.nav_clip_min = torch.tensor(
            [
                self.cfg.commands.ranges.limit_vx[0],
                self.cfg.commands.ranges.limit_vy[0],
                self.cfg.commands.ranges.limit_vyaw[0],
            ],
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.nav_clip_max = torch.tensor(
            [
                self.cfg.commands.ranges.limit_vx[1],
                self.cfg.commands.ranges.limit_vy[1],
                self.cfg.commands.ranges.limit_vyaw[1],
            ],
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.nav_actions_filtered = torch.zeros(
            self.num_envs, 3, device=self.device, requires_grad=False
        )
        self.rays_hist = torch.ones(
            self.num_envs,
            self.cfg.env.his_len,
            self.ray_angles.shape[0],
            device=self.device,
            dtype=torch.float,
        ) * 5.0
        self.goal_hold_timer = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)
        self.stay_timer = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)
        self.goal_reached_flag = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.stand_still_flag = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )

        token_k = getattr(self.cfg.env, "dynamic_token_k", 0)
        token_dim = getattr(self.cfg.env, "dynamic_token_dim", 7)
        self.dynamic_tokens = torch.zeros(
            self.num_envs, token_k, token_dim, device=self.device, dtype=torch.float
        )
        self.dynamic_token_mask = torch.zeros(
            self.num_envs, token_k, device=self.device, dtype=torch.bool
        )

        self.dynamic_local_pos = torch.zeros(
            self.num_envs, self.dynamic_obs_count, 2, device=self.device, dtype=torch.float
        )
        self.dynamic_velocity_vectors = torch.zeros(
            self.num_envs, self.dynamic_obs_count, 2, device=self.device, dtype=torch.float
        )
        self.dynamic_centers = torch.zeros(
            self.num_envs, self.dynamic_obs_count, 2, device=self.device, dtype=torch.float
        )
        self.dynamic_anchor_points = torch.zeros(
            self.num_envs, self.dynamic_obs_count, 2, device=self.device, dtype=torch.float
        )
        self.dynamic_shape_params = torch.zeros(
            self.num_envs, self.dynamic_obs_count, 4, device=self.device, dtype=torch.float
        )
        self.dynamic_phase = torch.zeros(
            self.num_envs, self.dynamic_obs_count, device=self.device, dtype=torch.float
        )
        self.dynamic_phase_speed = torch.zeros(
            self.num_envs, self.dynamic_obs_count, device=self.device, dtype=torch.float
        )
        self.dynamic_motion_types = torch.full(
            (self.num_envs, self.dynamic_obs_count),
            -1,
            device=self.device,
            dtype=torch.long,
        )
        self.dynamic_active_mask = torch.zeros(
            self.num_envs, self.dynamic_obs_count, device=self.device, dtype=torch.bool
        )
        self.dynamic_size_xy = torch.tensor(
            self._dynamic_cfg().size[:2] if self.dynamic_obs_count > 0 else [0.0, 0.0],
            device=self.device,
            dtype=torch.float,
        )
        self.dynamic_radius = float(getattr(self._dynamic_cfg(), "ray_radius", 0.32))
        self.body_collision_count = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.dynamic_collision_count = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.total_collision_count = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.body_collision_cooldown = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.dynamic_collision_cooldown = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.body_collision_event = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.dynamic_collision_event = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.near_miss_event = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.near_miss_occurred = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.near_miss_count = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.last_near_miss_active = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.min_ttc = torch.full(
            (self.num_envs,), 10.0, device=self.device, dtype=torch.float
        )
        self.predicted_min_clearance = torch.full(
            (self.num_envs,), 5.0, device=self.device, dtype=torch.float
        )
        self.min_dynamic_clearance = torch.full(
            (self.num_envs,), 5.0, device=self.device, dtype=torch.float
        )
        self.shield_intervention_rate = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float
        )
        self.dynamic_cbf_intervention_rate = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float
        )
        self.shield_intervention_sum = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float
        )
        self.dynamic_cbf_intervention_sum = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float
        )
        self.shield_intervention_steps = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float
        )
        self.dynamic_cbf_intervention_steps = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float
        )
        self.reset_goal = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.reset_stand_still = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.reset_timeout = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.reset_fall = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.reset_contact50 = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.reset_initial_contact50 = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.reset_spawn_collision = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.reset_terminate_contact = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.reset_dynamic_collision = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.dynamic_motion_counts = torch.zeros(
            self.num_envs, 4, device=self.device, dtype=torch.long
        )
        traj_len = int(
            self.max_episode_length
            + math.ceil(
                float(getattr(self._dynamic_cfg(), "trajectory_extra_horizon", 0.0))
                / max(self.dt, 1e-6)
            )
            + 2
        )
        self.dynamic_traj_pos = torch.zeros(
            self.num_envs,
            self.dynamic_obs_count,
            traj_len,
            2,
            device=self.device,
            dtype=torch.float,
        )
        self.dynamic_traj_vel = torch.zeros_like(self.dynamic_traj_pos)
        self.dynamic_traj_step = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )

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

    def _check_spawn_collision(self):
        before_reset = self.reset_buf.clone()
        super()._check_spawn_collision()
        self.reset_spawn_collision = self.reset_buf & (~before_reset) & self.initial_

    def _reset_dofs(self, env_ids):
        if len(env_ids) == 0:
            return
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(
            1.0, 1.0, (len(env_ids), self.num_dof), device=self.device
        )
        self.dof_vel[env_ids] = 0.0

        actor_indices_int32 = self.robot_actor_indices[env_ids].to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(actor_indices_int32),
            len(actor_indices_int32),
        )

    def _local_robot_start_goal(self, env_ids):
        robot_local = self.env_origins[env_ids, :2] - self.room_origins[env_ids]
        goal_local = self.position_targets[env_ids, :2] - self.room_origins[env_ids]
        return robot_local, goal_local

    def _room_bounds(self):
        cfg = self._dynamic_cfg()
        margin = float(getattr(cfg, "obstacle_wall_margin", 0.5))
        x_min = 0.8 + margin
        x_max = 9.2 - margin
        y_min = cfg.y_min + margin
        y_max = cfg.y_max - margin
        return x_min, x_max, y_min, y_max

    def _sample_motion_types(self, num_samples):
        cfg = self._dynamic_cfg()
        probs_dict = getattr(cfg, "motion_type_probs", None)
        names = [
            "linear_crossing",
            "linear_diagonal",
            "circular",
            "figure_eight",
        ]
        if probs_dict is None:
            probs = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        else:
            probs = torch.tensor(
                [float(probs_dict.get(name, 0.0)) for name in names],
                device=self.device,
                dtype=torch.float,
            )
        probs = probs / probs.sum().clamp(min=1e-6)
        return torch.multinomial(probs, num_samples, replacement=True)

    def _point_is_valid_dynamic_spawn(self, env_idx, point_xy, extra_clearance=0.0):
        cfg = self._dynamic_cfg()
        robot_local, goal_local = self._local_robot_start_goal(
            torch.tensor([env_idx], device=self.device)
        )
        start_pt = robot_local[0]
        goal_pt = goal_local[0]
        start_clearance = float(getattr(cfg, "spawn_start_clearance", getattr(cfg, "spawn_goal_clearance", 1.2))) + extra_clearance
        goal_clearance = float(getattr(cfg, "spawn_goal_clearance", 1.2)) + extra_clearance
        if torch.norm(point_xy - start_pt) < start_clearance:
            return False
        if torch.norm(point_xy - goal_pt) < goal_clearance:
            return False
        if not self._is_point_clear_of_pillars(
            float(point_xy[0].item()), float(point_xy[1].item()), extra_clearance
        ):
            return False
        return True

    def _bbox_points_valid(self, env_idx, center_xy, half_extent):
        offsets = torch.tensor(
            [
                [half_extent[0], half_extent[1]],
                [half_extent[0], -half_extent[1]],
                [-half_extent[0], half_extent[1]],
                [-half_extent[0], -half_extent[1]],
            ],
            device=self.device,
            dtype=torch.float,
        )
        points = center_xy.unsqueeze(0) + offsets
        x_min, x_max, y_min, y_max = self._room_bounds()
        if torch.any(points[:, 0] < x_min) or torch.any(points[:, 0] > x_max):
            return False
        if torch.any(points[:, 1] < y_min) or torch.any(points[:, 1] > y_max):
            return False
        clearance = float(getattr(self._dynamic_cfg(), "obstacle_wall_margin", 0.5))
        for pt in points:
            if not self._point_is_valid_dynamic_spawn(env_idx, pt, extra_clearance=clearance):
                return False
        return True

    def _sample_candidate_center(self, env_idx, motion_type):
        cfg = self._dynamic_cfg()
        x_min, x_max, y_min, y_max = self._room_bounds()
        interaction_band = float(getattr(cfg, "interaction_band_half_width", 1.25))
        robot_local, goal_local = self._local_robot_start_goal(
            torch.tensor([env_idx], device=self.device)
        )
        corridor_mid_y = 0.5 * (robot_local[0, 1] + goal_local[0, 1])
        if motion_type == self.MOTION_LINEAR_CROSSING:
            x = torch.empty(1, device=self.device).uniform_(x_min, x_max)[0]
            y = torch.empty(1, device=self.device).uniform_(
                max(y_min, corridor_mid_y - interaction_band),
                min(y_max, corridor_mid_y + interaction_band),
            )[0]
            return torch.stack([x, y], dim=0)
        if motion_type == self.MOTION_LINEAR_DIAGONAL:
            edge = torch.randint(0, 4, (1,), device=self.device).item()
            if edge == 0:
                return torch.tensor(
                    [x_min, float(torch.empty(1).uniform_(y_min, y_max).item())],
                    device=self.device,
                )
            if edge == 1:
                return torch.tensor(
                    [x_max, float(torch.empty(1).uniform_(y_min, y_max).item())],
                    device=self.device,
                )
            if edge == 2:
                return torch.tensor(
                    [float(torch.empty(1).uniform_(x_min, x_max).item()), y_min],
                    device=self.device,
                )
            return torch.tensor(
                [float(torch.empty(1).uniform_(x_min, x_max).item()), y_max],
                device=self.device,
            )
        x = torch.empty(1, device=self.device).uniform_(x_min, x_max)[0]
        y = torch.empty(1, device=self.device).uniform_(y_min, y_max)[0]
        return torch.stack([x, y], dim=0)

    def _sample_linear_diagonal_velocity(self, speed):
        jitter_deg = float(getattr(self._dynamic_cfg(), "diagonal_heading_jitter_deg", 20.0))
        base_angles = torch.tensor(
            [
                math.pi / 4.0,
                3.0 * math.pi / 4.0,
                -math.pi / 4.0,
                -3.0 * math.pi / 4.0,
            ],
            device=self.device,
        )
        base_angle = base_angles[torch.randint(0, 4, (1,), device=self.device).item()]
        jitter = float(torch.empty(1).uniform_(-jitter_deg, jitter_deg).item()) * math.pi / 180.0
        angle = base_angle + jitter
        return torch.tensor(
            [speed * math.cos(angle), speed * math.sin(angle)],
            device=self.device,
            dtype=torch.float,
        )

    def _traj_half_extent(self, motion_type, shape_params):
        if motion_type == self.MOTION_CIRCULAR:
            radius = shape_params[0]
            return torch.tensor([radius, radius], device=self.device)
        if motion_type == self.MOTION_FIGURE_EIGHT:
            return torch.tensor([shape_params[0], shape_params[1]], device=self.device)
        return self.dynamic_size_xy * 0.5

    def _wrap_phase(self, phase):
        return torch.remainder(phase, 2.0 * math.pi)

    def _uses_precomputed_dynamic_trajectories(self):
        return (
            self.dynamic_obs_count > 0
            and getattr(self._dynamic_cfg(), "trajectory_mode", "online")
            == "episode_precomputed"
        )

    def _trajectory_len(self):
        return self.dynamic_traj_pos.shape[2]

    def _sample_precomputed_center(self, env_idx, margin=0.0):
        cfg = self._dynamic_cfg()
        x_min, x_max, y_min, y_max = self._room_bounds()
        x_min += margin
        x_max -= margin
        y_min += margin
        y_max -= margin
        robot_local, goal_local = self._local_robot_start_goal(
            torch.tensor([env_idx], device=self.device)
        )
        start_pt = robot_local[0]
        goal_pt = goal_local[0]
        start_clearance = float(getattr(cfg, "spawn_start_clearance", getattr(cfg, "spawn_goal_clearance", 1.0)))
        goal_clearance = float(getattr(cfg, "spawn_goal_clearance", 1.0))
        force_interaction = bool(getattr(cfg, "force_interaction", False))
        interaction_band = float(getattr(cfg, "interaction_band_half_width", 1.25))
        jitter = float(getattr(cfg, "force_interaction_jitter", 0.45))
        corridor_mid_y = 0.5 * (start_pt[1] + goal_pt[1])

        for _ in range(32):
            x = torch.empty(1, device=self.device).uniform_(x_min, x_max)[0]
            if force_interaction:
                low_y = torch.clamp(corridor_mid_y - jitter, min=y_min, max=y_max)
                high_y = torch.clamp(corridor_mid_y + jitter, min=y_min, max=y_max)
            else:
                low_y = max(y_min, float((corridor_mid_y - interaction_band).item()))
                high_y = min(y_max, float((corridor_mid_y + interaction_band).item()))
            low_y = float(low_y.item() if torch.is_tensor(low_y) else low_y)
            high_y = float(high_y.item() if torch.is_tensor(high_y) else high_y)
            if low_y > high_y:
                low_y, high_y = y_min, y_max
            y = torch.empty(1, device=self.device).uniform_(low_y, high_y)[0]
            center = torch.stack([x, y])
            if torch.norm(center - start_pt) >= start_clearance and torch.norm(center - goal_pt) >= goal_clearance:
                return center

        return torch.tensor(
            [0.5 * (x_min + x_max), 0.5 * (y_min + y_max)],
            device=self.device,
            dtype=torch.float,
        )

    def _precompute_linear_pingpong(self, start, velocity, lower, upper, steps):
        t = torch.arange(steps, device=self.device, dtype=torch.float) * self.dt
        speed = torch.norm(velocity).clamp(min=1e-6)
        direction = velocity / speed
        lower_s = torch.sum(lower * direction)
        upper_s = torch.sum(upper * direction)
        use_lower_as_base = lower_s <= upper_s
        base = torch.where(use_lower_as_base, lower, upper)
        min_s = torch.minimum(lower_s, upper_s)
        max_s = torch.maximum(lower_s, upper_s)
        span = (max_s - min_s).clamp(min=0.2)
        signed0 = torch.sum(start * direction) - min_s
        signed = signed0 + speed * t
        period = 2.0 * span
        phase = torch.remainder(signed, period)
        dist = torch.where(phase <= span, phase, period - phase)
        pos = base.unsqueeze(0) + dist.unsqueeze(1) * direction.unsqueeze(0)
        vel_sign = torch.where(phase <= span, 1.0, -1.0)
        vel = vel_sign.unsqueeze(1) * speed * direction.unsqueeze(0)
        return pos, vel

    def _precompute_parametric_slot(self, env_idx, motion_type, speed, phase):
        cfg = self._dynamic_cfg()
        x_min, x_max, y_min, y_max = self._room_bounds()
        steps = self._trajectory_len()
        t = torch.arange(steps, device=self.device, dtype=torch.float) * self.dt
        pos = torch.zeros(steps, 2, device=self.device, dtype=torch.float)
        vel = torch.zeros_like(pos)
        center = torch.zeros(2, device=self.device, dtype=torch.float)
        anchor = torch.zeros(2, device=self.device, dtype=torch.float)
        shape_params = torch.zeros(4, device=self.device, dtype=torch.float)
        velocity = torch.zeros(2, device=self.device, dtype=torch.float)
        phase_speed = 0.0

        if motion_type == self.MOTION_LINEAR_CROSSING:
            center = self._sample_precomputed_center(env_idx)
            direction = 1.0 if torch.rand(1, device=self.device).item() > 0.5 else -1.0
            velocity = torch.tensor([0.0, direction * speed], device=self.device)
            lower = torch.tensor([center[0], y_min], device=self.device)
            upper = torch.tensor([center[0], y_max], device=self.device)
            pos, vel = self._precompute_linear_pingpong(center, velocity, lower, upper, steps)
            anchor = pos[0].clone()
            center = pos[0].clone()
        elif motion_type == self.MOTION_LINEAR_DIAGONAL:
            center = self._sample_precomputed_center(env_idx)
            velocity = self._sample_linear_diagonal_velocity(speed)
            direction = velocity / torch.norm(velocity).clamp(min=1e-6)
            candidates = []
            eps = 1e-6
            for axis, low, high in ((0, x_min, x_max), (1, y_min, y_max)):
                if abs(float(direction[axis].item())) > eps:
                    candidates.append((low - center[axis]) / direction[axis])
                    candidates.append((high - center[axis]) / direction[axis])
            candidates = torch.stack(candidates)
            forward = candidates[candidates >= 0.0].min()
            backward = candidates[candidates <= 0.0].max()
            lower = center + backward * direction
            upper = center + forward * direction
            pos, vel = self._precompute_linear_pingpong(center, velocity, lower, upper, steps)
            anchor = pos[0].clone()
            center = pos[0].clone()
        elif motion_type == self.MOTION_CIRCULAR:
            radius_min, radius_max = getattr(cfg, "circle_radius_range", [0.6, 1.3])
            max_radius = max(0.2, min(radius_max, 0.5 * min(x_max - x_min, y_max - y_min) - 0.05))
            radius = float(torch.empty(1, device=self.device).uniform_(radius_min, max_radius).item())
            anchor = self._sample_precomputed_center(env_idx, margin=radius)
            direction = 1.0 if torch.rand(1, device=self.device).item() > 0.5 else -1.0
            phase_speed = direction * speed / max(radius, 1e-3)
            theta = phase + phase_speed * t
            pos = anchor.unsqueeze(0) + torch.stack(
                [radius * torch.cos(theta), radius * torch.sin(theta)], dim=-1
            )
            vel = torch.stack(
                [-radius * phase_speed * torch.sin(theta), radius * phase_speed * torch.cos(theta)],
                dim=-1,
            )
            center = anchor.clone()
            shape_params[0] = radius
            shape_params[1] = phase_speed
        else:
            scale_min, scale_max = getattr(cfg, "figure_eight_scale_range", [0.7, 1.2])
            max_scale = max(0.2, min(scale_max, 0.5 * min(x_max - x_min, y_max - y_min) - 0.05))
            scale_x = float(torch.empty(1, device=self.device).uniform_(scale_min, max_scale).item())
            scale_y = float(torch.empty(1, device=self.device).uniform_(scale_min, max_scale).item())
            anchor = self._sample_precomputed_center(env_idx, margin=max(scale_x, scale_y))
            direction = 1.0 if torch.rand(1, device=self.device).item() > 0.5 else -1.0
            phase_speed = direction * speed / max(max(scale_x, scale_y), 1e-3)
            theta = phase + phase_speed * t
            pos = anchor.unsqueeze(0) + torch.stack(
                [scale_x * torch.sin(theta), scale_y * torch.sin(2.0 * theta)],
                dim=-1,
            )
            vel = torch.stack(
                [
                    scale_x * phase_speed * torch.cos(theta),
                    2.0 * scale_y * phase_speed * torch.cos(2.0 * theta),
                ],
                dim=-1,
            )
            center = anchor.clone()
            shape_params[0] = scale_x
            shape_params[1] = scale_y
            shape_params[2] = phase_speed

        pos[:, 0].clamp_(x_min, x_max)
        pos[:, 1].clamp_(y_min, y_max)
        return pos, vel, center, anchor, shape_params, velocity, phase_speed

    def _generate_precomputed_dynamic_trajectories(self, env_ids):
        if self.dynamic_obs_count == 0 or len(env_ids) == 0:
            return
        low_count, high_count = self._dynamic_count_range()
        self.dynamic_active_mask[env_ids] = False
        self.dynamic_motion_types[env_ids] = -1
        self.dynamic_local_pos[env_ids] = 0.0
        self.dynamic_velocity_vectors[env_ids] = 0.0
        self.dynamic_centers[env_ids] = 0.0
        self.dynamic_anchor_points[env_ids] = 0.0
        self.dynamic_shape_params[env_ids] = 0.0
        self.dynamic_phase[env_ids] = 0.0
        self.dynamic_phase_speed[env_ids] = 0.0
        self.dynamic_motion_counts[env_ids] = 0
        self.dynamic_traj_pos[env_ids] = 0.0
        self.dynamic_traj_vel[env_ids] = 0.0
        self.dynamic_traj_step[env_ids] = 0

        active_counts = torch.randint(
            low_count, high_count + 1, (len(env_ids),), device=self.device
        )
        for local_i, env_id in enumerate(env_ids.tolist()):
            desired = int(active_counts[local_i].item())
            motion_types = self._sample_motion_types(desired)
            for slot_idx in range(desired):
                motion_type = int(motion_types[slot_idx].item())
                speed = float(
                    torch_rand_float(
                        self._dynamic_cfg().speed_range[0],
                        self._dynamic_cfg().speed_range[1],
                        (1, 1),
                        device=self.device,
                    )[0, 0].item()
                )
                phase = float(torch.empty(1, device=self.device).uniform_(0.0, 2.0 * math.pi).item())
                pos, vel, center, anchor, shape_params, velocity, phase_speed = (
                    self._precompute_parametric_slot(env_id, motion_type, speed, phase)
                )
                self.dynamic_traj_pos[env_id, slot_idx] = pos
                self.dynamic_traj_vel[env_id, slot_idx] = vel
                self.dynamic_motion_types[env_id, slot_idx] = motion_type
                self.dynamic_active_mask[env_id, slot_idx] = True
                self.dynamic_centers[env_id, slot_idx] = center
                self.dynamic_anchor_points[env_id, slot_idx] = anchor
                self.dynamic_shape_params[env_id, slot_idx] = shape_params
                self.dynamic_phase[env_id, slot_idx] = phase
                self.dynamic_phase_speed[env_id, slot_idx] = phase_speed
                self.dynamic_local_pos[env_id, slot_idx] = pos[0]
                self.dynamic_velocity_vectors[env_id, slot_idx] = vel[0]

        self._refresh_motion_counts(env_ids)

    def _activate_dynamic_slot(
        self,
        env_idx,
        slot_idx,
        motion_type,
        center,
        velocity,
        phase,
        phase_speed,
        shape_params,
        anchor=None,
    ):
        if anchor is None:
            anchor = center
        self.dynamic_motion_types[env_idx, slot_idx] = int(motion_type)
        self.dynamic_centers[env_idx, slot_idx] = center
        self.dynamic_anchor_points[env_idx, slot_idx] = anchor
        self.dynamic_shape_params[env_idx, slot_idx] = shape_params
        self.dynamic_phase[env_idx, slot_idx] = self._wrap_phase(
            torch.tensor(phase, device=self.device, dtype=torch.float)
        )
        self.dynamic_phase_speed[env_idx, slot_idx] = phase_speed
        self.dynamic_active_mask[env_idx, slot_idx] = True
        self.dynamic_velocity_vectors[env_idx, slot_idx] = velocity
        self._update_slot_pose_from_state(env_idx, slot_idx)

    def _generate_fallback_centers(self, env_idx):
        x_candidates = [1.8, 2.6, 4.8, 5.6, 7.4, 8.2]
        y_candidates = [1.8, 3.2, 4.8, 6.4, 8.0]
        centers = []
        for x in x_candidates:
            for y in y_candidates:
                point = torch.tensor([x, y], device=self.device, dtype=torch.float)
                if self._point_is_valid_dynamic_spawn(env_idx, point, extra_clearance=0.15):
                    centers.append(point)
        return centers

    def _fallback_fill_dynamic_slots(self, env_idx, desired, actual):
        cfg = self._dynamic_cfg()
        clearance = max(0.65, float(getattr(cfg, "spawn_min_separation", 0.9)) - 0.2)
        centers = self._generate_fallback_centers(env_idx)
        slot_idx = actual
        for center in centers:
            if slot_idx >= desired or slot_idx >= self.dynamic_obs_count:
                break
            existing_mask = self.dynamic_active_mask[env_idx]
            if existing_mask.any():
                existing = self.dynamic_centers[env_idx, existing_mask]
                if torch.any(torch.norm(existing - center.unsqueeze(0), dim=-1) < clearance):
                    continue
            speed = float(
                torch_rand_float(
                    cfg.speed_range[0], cfg.speed_range[1], (1, 1), device=self.device
                )[0, 0].item()
            )
            motion_cycle = [
                self.MOTION_LINEAR_CROSSING,
                self.MOTION_LINEAR_DIAGONAL,
                self.MOTION_CIRCULAR,
                self.MOTION_FIGURE_EIGHT,
            ]
            motion_type = motion_cycle[slot_idx % len(motion_cycle)]
            phase = float(torch.empty(1).uniform_(0.0, 2.0 * math.pi).item())
            shape_params = torch.zeros(4, device=self.device, dtype=torch.float)
            velocity = torch.zeros(2, device=self.device, dtype=torch.float)
            phase_speed = 0.0
            if motion_type == self.MOTION_LINEAR_CROSSING:
                velocity = torch.tensor(
                    [0.0, speed if torch.rand(1).item() > 0.5 else -speed],
                    device=self.device,
                    dtype=torch.float,
                )
            elif motion_type == self.MOTION_LINEAR_DIAGONAL:
                velocity = self._sample_linear_diagonal_velocity(speed)
            elif motion_type == self.MOTION_CIRCULAR:
                radius = float(
                    torch.empty(1).uniform_(
                        getattr(cfg, "circle_radius_range", [0.6, 1.3])[0],
                        min(0.95, getattr(cfg, "circle_radius_range", [0.6, 1.3])[1]),
                    ).item()
                )
                shape_params[0] = radius
                phase_speed = (1.0 if torch.rand(1).item() > 0.5 else -1.0) * speed / max(
                    radius, 1e-3
                )
                shape_params[1] = phase_speed
                if not self._bbox_points_valid(
                    env_idx, center, torch.tensor([radius, radius], device=self.device)
                ):
                    motion_type = self.MOTION_LINEAR_CROSSING
                    velocity = torch.tensor(
                        [0.0, speed if torch.rand(1).item() > 0.5 else -speed],
                        device=self.device,
                        dtype=torch.float,
                    )
                    shape_params.zero_()
                    phase_speed = 0.0
            else:
                scale = float(
                    torch.empty(1).uniform_(
                        getattr(cfg, "figure_eight_scale_range", [0.7, 1.2])[0],
                        min(0.9, getattr(cfg, "figure_eight_scale_range", [0.7, 1.2])[1]),
                    ).item()
                )
                shape_params[0] = scale
                shape_params[1] = scale
                phase_speed = (1.0 if torch.rand(1).item() > 0.5 else -1.0) * speed / max(
                    scale, 1e-3
                )
                shape_params[2] = phase_speed
                if not self._bbox_points_valid(
                    env_idx, center, torch.tensor([scale, scale], device=self.device)
                ):
                    motion_type = self.MOTION_LINEAR_DIAGONAL
                    velocity = self._sample_linear_diagonal_velocity(speed)
                    shape_params.zero_()
                    phase_speed = 0.0

            self._activate_dynamic_slot(
                env_idx,
                slot_idx,
                motion_type,
                center,
                velocity,
                phase,
                phase_speed,
                shape_params,
                anchor=center,
            )
            slot_idx += 1
        return slot_idx

    def _init_dynamic_slot(self, env_idx, slot_idx):
        cfg = self._dynamic_cfg()
        motion_type = int(self._sample_motion_types(1)[0].item())
        speed = float(
            torch_rand_float(
                cfg.speed_range[0], cfg.speed_range[1], (1, 1), device=self.device
            )[0, 0].item()
        )
        max_attempts = int(getattr(cfg, "max_spawn_attempts", 32))
        clearance = float(getattr(cfg, "spawn_min_separation", 0.9))

        for attempt in range(max_attempts):
            current_motion = motion_type if attempt < max_attempts - 1 else self.MOTION_LINEAR_CROSSING
            center = self._sample_candidate_center(env_idx, current_motion)
            phase = float(torch.empty(1).uniform_(0.0, 2.0 * math.pi).item())
            shape_params = torch.zeros(4, device=self.device, dtype=torch.float)
            velocity = torch.zeros(2, device=self.device, dtype=torch.float)
            anchor = center.clone()
            phase_speed = 0.0

            if current_motion == self.MOTION_LINEAR_CROSSING:
                velocity = torch.tensor(
                    [0.0, speed if torch.rand(1).item() > 0.5 else -speed],
                    device=self.device,
                    dtype=torch.float,
                )
                anchor = center.clone()
            elif current_motion == self.MOTION_LINEAR_DIAGONAL:
                velocity = self._sample_linear_diagonal_velocity(speed)
                anchor = center.clone()
            elif current_motion == self.MOTION_CIRCULAR:
                radius_min, radius_max = getattr(cfg, "circle_radius_range", [0.6, 1.3])
                radius = float(torch.empty(1).uniform_(radius_min, radius_max).item())
                shape_params[0] = radius
                direction = 1.0 if torch.rand(1).item() > 0.5 else -1.0
                phase_speed = direction * speed / max(radius, 1e-3)
                shape_params[1] = phase_speed
            else:
                scale_min, scale_max = getattr(cfg, "figure_eight_scale_range", [0.7, 1.2])
                scale_x = float(torch.empty(1).uniform_(scale_min, scale_max).item())
                scale_y = float(torch.empty(1).uniform_(scale_min, scale_max).item())
                shape_params[0] = scale_x
                shape_params[1] = scale_y
                direction = 1.0 if torch.rand(1).item() > 0.5 else -1.0
                phase_speed = direction * speed / max(max(scale_x, scale_y), 1e-3)
                shape_params[2] = phase_speed

            half_extent = self._traj_half_extent(current_motion, shape_params)
            if not self._bbox_points_valid(env_idx, center, half_extent):
                continue

            existing_mask = self.dynamic_active_mask[env_idx]
            if existing_mask.any():
                existing = self.dynamic_centers[env_idx, existing_mask]
                if torch.any(torch.norm(existing - center.unsqueeze(0), dim=-1) < clearance):
                    continue

            self._activate_dynamic_slot(
                env_idx,
                slot_idx,
                current_motion,
                center,
                velocity,
                phase,
                phase_speed,
                shape_params,
                anchor=anchor,
            )
            return True
        return False

    def _update_slot_pose_from_state(self, env_idx, slot_idx):
        motion_type = int(self.dynamic_motion_types[env_idx, slot_idx].item())
        anchor = self.dynamic_anchor_points[env_idx, slot_idx]
        phase = self.dynamic_phase[env_idx, slot_idx]
        shape = self.dynamic_shape_params[env_idx, slot_idx]
        velocity = self.dynamic_velocity_vectors[env_idx, slot_idx]
        pos = torch.zeros(2, device=self.device, dtype=torch.float)
        vel = torch.zeros(2, device=self.device, dtype=torch.float)

        if motion_type == self.MOTION_LINEAR_CROSSING:
            pos = self.dynamic_centers[env_idx, slot_idx]
            vel = velocity
        elif motion_type == self.MOTION_LINEAR_DIAGONAL:
            pos = self.dynamic_centers[env_idx, slot_idx]
            vel = velocity
        elif motion_type == self.MOTION_CIRCULAR:
            radius = shape[0]
            omega = self.dynamic_phase_speed[env_idx, slot_idx]
            pos = anchor + torch.tensor(
                [radius * torch.cos(phase), radius * torch.sin(phase)],
                device=self.device,
            )
            vel = torch.tensor(
                [-radius * omega * torch.sin(phase), radius * omega * torch.cos(phase)],
                device=self.device,
            )
        elif motion_type == self.MOTION_FIGURE_EIGHT:
            scale_x = shape[0]
            scale_y = shape[1]
            omega = self.dynamic_phase_speed[env_idx, slot_idx]
            pos = anchor + torch.tensor(
                [scale_x * torch.sin(phase), scale_y * torch.sin(2.0 * phase)],
                device=self.device,
            )
            vel = torch.tensor(
                [
                    scale_x * omega * torch.cos(phase),
                    2.0 * scale_y * omega * torch.cos(2.0 * phase),
                ],
                device=self.device,
            )
        self.dynamic_local_pos[env_idx, slot_idx] = pos
        self.dynamic_velocity_vectors[env_idx, slot_idx] = vel

    def _refresh_motion_counts(self, env_ids):
        if self.dynamic_obs_count == 0 or len(env_ids) == 0:
            return
        self.dynamic_motion_counts[env_ids] = 0
        for motion_type in range(self.dynamic_motion_counts.shape[1]):
            mask = self.dynamic_active_mask[env_ids] & (
                self.dynamic_motion_types[env_ids] == motion_type
            )
            self.dynamic_motion_counts[env_ids, motion_type] = mask.sum(dim=1)

    def _reset_dynamic_obstacles(self, env_ids):
        if self.dynamic_obs_count == 0 or len(env_ids) == 0:
            return
        if self._uses_precomputed_dynamic_trajectories():
            self._generate_precomputed_dynamic_trajectories(env_ids)
            self._sync_dynamic_root_states(env_ids)
            return
        low_count, high_count = self._dynamic_count_range()
        self.dynamic_active_mask[env_ids] = False
        self.dynamic_motion_types[env_ids] = -1
        self.dynamic_local_pos[env_ids] = 0.0
        self.dynamic_velocity_vectors[env_ids] = 0.0
        self.dynamic_centers[env_ids] = 0.0
        self.dynamic_anchor_points[env_ids] = 0.0
        self.dynamic_shape_params[env_ids] = 0.0
        self.dynamic_phase[env_ids] = 0.0
        self.dynamic_phase_speed[env_ids] = 0.0
        self.dynamic_motion_counts[env_ids] = 0

        active_counts = torch.randint(
            low_count, high_count + 1, (len(env_ids),), device=self.device
        )
        for local_i, env_id in enumerate(env_ids.tolist()):
            desired = int(active_counts[local_i].item())
            actual = 0
            for slot_idx in range(self.dynamic_obs_count):
                if actual >= desired:
                    break
                if self._init_dynamic_slot(env_id, slot_idx):
                    actual += 1
            if actual < desired:
                actual = self._fallback_fill_dynamic_slots(env_id, desired, actual)

        self._refresh_motion_counts(env_ids)

        self._sync_dynamic_root_states(env_ids)

    def _sync_dynamic_root_states(self, env_ids=None):
        if self.dynamic_obs_count == 0:
            return
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if len(env_ids) == 0:
            return
        world_xy = self.room_origins[env_ids].unsqueeze(1) + self.dynamic_local_pos[env_ids]
        self.dynamic_root_states[env_ids, :, 0:2] = world_xy
        self.dynamic_root_states[env_ids, :, 2] = self._dynamic_cfg().size[2] * 0.5
        self.dynamic_root_states[env_ids, :, 3:7] = 0.0
        self.dynamic_root_states[env_ids, :, 6] = 1.0
        self.dynamic_root_states[env_ids, :, 7:9] = self.dynamic_velocity_vectors[env_ids]
        self.dynamic_root_states[env_ids, :, 9] = 0.0
        self.dynamic_root_states[env_ids, :, 10:13] = 0.0
        inactive = ~self.dynamic_active_mask[env_ids]
        if inactive.any():
            far_x_value = self.terrain.env_length * self.cfg.terrain.num_rows + 50.0
            far_y_value = self.terrain.env_width * self.cfg.terrain.num_cols + 50.0
            far_x = torch.full_like(self.dynamic_root_states[env_ids, :, 0], far_x_value)
            far_y = torch.full_like(self.dynamic_root_states[env_ids, :, 1], far_y_value)
            far_z = torch.full_like(self.dynamic_root_states[env_ids, :, 2], -10.0)
            self.dynamic_root_states[env_ids, :, 0] = torch.where(
                inactive, far_x, self.dynamic_root_states[env_ids, :, 0]
            )
            self.dynamic_root_states[env_ids, :, 1] = torch.where(
                inactive, far_y, self.dynamic_root_states[env_ids, :, 1]
            )
            self.dynamic_root_states[env_ids, :, 2] = torch.where(
                inactive, far_z, self.dynamic_root_states[env_ids, :, 2]
            )
            self.dynamic_root_states[env_ids, :, 7:10] = torch.where(
                inactive.unsqueeze(-1),
                torch.zeros_like(self.dynamic_root_states[env_ids, :, 7:10]),
                self.dynamic_root_states[env_ids, :, 7:10],
            )
        self._set_dynamic_root_states(env_ids)

    def _reset_root_states(self, env_ids):
        self._resample_sparse_navigation(env_ids)
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        self.root_states[env_ids, 7:13] = torch_rand_float(
            -0.5, 0.5, (len(env_ids), 6), device=self.device
        )

        if self.cfg.domain_rand.randomize_yaw:
            yaw = torch.zeros_like(self.root_states[env_ids, 3]).uniform_(
                self.cfg.domain_rand.init_yaw_range[0],
                self.cfg.domain_rand.init_yaw_range[1],
            )
        else:
            yaw = torch.zeros_like(self.root_states[env_ids, 3])
        if self.cfg.domain_rand.randomize_roll:
            roll = torch.zeros_like(self.root_states[env_ids, 3]).uniform_(
                self.cfg.domain_rand.init_roll_range[0],
                self.cfg.domain_rand.init_roll_range[1],
            )
        else:
            roll = torch.zeros_like(self.root_states[env_ids, 3])
        if self.cfg.domain_rand.randomize_pitch:
            pitch = torch.zeros_like(self.root_states[env_ids, 3]).uniform_(
                self.cfg.domain_rand.init_pitch_range[0],
                self.cfg.domain_rand.init_pitch_range[1],
            )
        else:
            pitch = torch.zeros_like(self.root_states[env_ids, 3])
        self.root_states[env_ids, 3:7] = quat_from_euler_xyz(roll, pitch, yaw)
        self.base_quat[env_ids] = self.root_states[env_ids, 3:7]
        self._set_robot_root_states(env_ids)
        self._reset_dynamic_obstacles(env_ids)

    def _update_replay_buffer(self):
        super()._update_replay_buffer()
        if self.dynamic_obs_count == 0:
            return
        init_mask = (self.episode_length_buf <= 1)[:, None, None, None]
        init_mask_small = (self.episode_length_buf <= 1)[:, None, None]
        self.replay_dynamic_root_states = torch.where(
            init_mask,
            torch.stack([self.dynamic_root_states] * self.replay_len, dim=1),
            torch.cat(
                [self.replay_dynamic_root_states[:, 1:], self.dynamic_root_states.unsqueeze(1)],
                dim=1,
            ),
        )
        self.replay_dynamic_local_pos = torch.where(
            init_mask,
            torch.stack([self.dynamic_local_pos] * self.replay_len, dim=1),
            torch.cat(
                [self.replay_dynamic_local_pos[:, 1:], self.dynamic_local_pos.unsqueeze(1)],
                dim=1,
            ),
        )
        self.replay_dynamic_velocity_vectors = torch.where(
            init_mask,
            torch.stack([self.dynamic_velocity_vectors] * self.replay_len, dim=1),
            torch.cat(
                [
                    self.replay_dynamic_velocity_vectors[:, 1:],
                    self.dynamic_velocity_vectors.unsqueeze(1),
                ],
                dim=1,
            ),
        )
        self.replay_dynamic_centers = torch.where(
            init_mask,
            torch.stack([self.dynamic_centers] * self.replay_len, dim=1),
            torch.cat(
                [self.replay_dynamic_centers[:, 1:], self.dynamic_centers.unsqueeze(1)],
                dim=1,
            ),
        )
        self.replay_dynamic_motion_types = torch.where(
            init_mask_small,
            torch.stack([self.dynamic_motion_types] * self.replay_len, dim=1),
            torch.cat(
                [
                    self.replay_dynamic_motion_types[:, 1:],
                    self.dynamic_motion_types.unsqueeze(1),
                ],
                dim=1,
            ),
        )
        self.replay_dynamic_phase = torch.where(
            init_mask_small,
            torch.stack([self.dynamic_phase] * self.replay_len, dim=1),
            torch.cat([self.replay_dynamic_phase[:, 1:], self.dynamic_phase.unsqueeze(1)], dim=1),
        )
        self.replay_dynamic_phase_speed = torch.where(
            init_mask_small,
            torch.stack([self.dynamic_phase_speed] * self.replay_len, dim=1),
            torch.cat(
                [self.replay_dynamic_phase_speed[:, 1:], self.dynamic_phase_speed.unsqueeze(1)],
                dim=1,
            ),
        )
        self.replay_dynamic_shape_params = torch.where(
            init_mask,
            torch.stack([self.dynamic_shape_params] * self.replay_len, dim=1),
            torch.cat(
                [self.replay_dynamic_shape_params[:, 1:], self.dynamic_shape_params.unsqueeze(1)],
                dim=1,
            ),
        )
        self.replay_dynamic_active_mask = torch.where(
            init_mask_small,
            torch.stack([self.dynamic_active_mask] * self.replay_len, dim=1),
            torch.cat(
                [
                    self.replay_dynamic_active_mask[:, 1:],
                    self.dynamic_active_mask.unsqueeze(1),
                ],
                dim=1,
            ),
        )
        step_mask = (self.episode_length_buf <= 1)[:, None]
        self.replay_dynamic_traj_step = torch.where(
            step_mask,
            torch.stack([self.dynamic_traj_step] * self.replay_len, dim=1),
            torch.cat(
                [self.replay_dynamic_traj_step[:, 1:], self.dynamic_traj_step.unsqueeze(1)],
                dim=1,
            ),
        )

    def _reset_collision_replay(self, env_ids):
        undo_range = getattr(self.cfg.replay, "undo_steps_range", [40, 80])
        undo_steps = torch.randint(
            undo_range[0], undo_range[1], (len(env_ids),), device=self.device
        )
        current_len = self.episode_length_buf[env_ids]
        undo_steps = torch.min(undo_steps.long(), current_len.long())
        undo_steps = torch.min(
            undo_steps, torch.tensor(self.replay_len - 1, device=self.device)
        )
        valid_replay = undo_steps > 20
        replay_ids = env_ids[valid_replay]
        fallback_ids = env_ids[~valid_replay]

        if len(fallback_ids) > 0:
            self._reset_dofs(fallback_ids)
            self._reset_root_states(fallback_ids)
            self.is_replay[fallback_ids] = False

        if len(replay_ids) == 0:
            return

        self.is_replay[replay_ids] = True
        indices = -undo_steps[valid_replay]
        self.root_states[replay_ids] = self.replay_root_states[replay_ids, indices]
        self.dof_pos[replay_ids] = self.replay_dof_pos[replay_ids, indices]
        self.dof_vel[replay_ids] = self.replay_dof_vel[replay_ids, indices]
        self.dynamic_root_states[replay_ids] = self.replay_dynamic_root_states[
            replay_ids, indices
        ]
        self.dynamic_local_pos[replay_ids] = self.replay_dynamic_local_pos[
            replay_ids, indices
        ]
        self.dynamic_velocity_vectors[replay_ids] = self.replay_dynamic_velocity_vectors[
            replay_ids, indices
        ]
        self.dynamic_motion_types[replay_ids] = self.replay_dynamic_motion_types[
            replay_ids, indices
        ]
        self.dynamic_centers[replay_ids] = self.replay_dynamic_centers[replay_ids, indices]
        self.dynamic_phase[replay_ids] = self.replay_dynamic_phase[replay_ids, indices]
        self.dynamic_phase_speed[replay_ids] = self.replay_dynamic_phase_speed[
            replay_ids, indices
        ]
        self.dynamic_shape_params[replay_ids] = self.replay_dynamic_shape_params[
            replay_ids, indices
        ]
        self.dynamic_active_mask[replay_ids] = self.replay_dynamic_active_mask[
            replay_ids, indices
        ]
        self.dynamic_traj_step[replay_ids] = self.replay_dynamic_traj_step[
            replay_ids, indices
        ]
        self._set_robot_root_states(replay_ids)
        robot_actor_indices_int32 = self.robot_actor_indices[replay_ids].to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(robot_actor_indices_int32),
            len(robot_actor_indices_int32),
        )
        self._set_dynamic_root_states(replay_ids)

    def _should_use_replay(self, env_ids):
        enable_collision = getattr(self.cfg.replay, "enable_collision_replay", False)
        enable_near_miss = getattr(self.cfg.replay, "enable_near_miss_replay", False)
        is_collision = self.collision_occurred[env_ids]
        is_near_miss = self.near_miss_occurred[env_ids] | (self.near_miss_count[env_ids] > 0)
        is_success = self.goal_reached_flag[env_ids]
        is_timeout = self.time_out_buf[env_ids]
        prob_replay = getattr(self.cfg.replay, "replay_prob", 0.8)
        wants_replay = torch.rand(len(env_ids), device=self.device) < prob_replay
        replay_trigger = (enable_collision & is_collision) | (enable_near_miss & is_near_miss)
        return wants_replay & replay_trigger & (~is_success) & (~is_timeout)

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        episode_len_s = self.episode_length_buf[env_ids].float() * self.dt
        success = self.goal_reached_flag[env_ids].float()
        timeout = self.time_out_buf[env_ids].float()
        total_collisions = self.total_collision_count[env_ids].float()
        body_collisions = self.body_collision_count[env_ids].float()
        dynamic_collisions = self.dynamic_collision_count[env_ids].float()
        near_miss = self.near_miss_count[env_ids].float()
        min_ttc = self.min_ttc[env_ids]
        shield_rate = self.shield_intervention_rate[env_ids].clone()
        dynamic_shield_rate = self.dynamic_cbf_intervention_rate[env_ids].clone()
        shield_mean = self.shield_intervention_sum[env_ids] / torch.clamp(
            self.episode_length_buf[env_ids].float(), min=1.0
        )
        dynamic_shield_mean = self.dynamic_cbf_intervention_sum[env_ids] / torch.clamp(
            self.episode_length_buf[env_ids].float(), min=1.0
        )
        shield_step_rate = self.shield_intervention_steps[env_ids] / torch.clamp(
            self.episode_length_buf[env_ids].float(), min=1.0
        )
        dynamic_shield_step_rate = self.dynamic_cbf_intervention_steps[env_ids] / torch.clamp(
            self.episode_length_buf[env_ids].float(), min=1.0
        )
        reset_goal = self.reset_goal[env_ids].float()
        reset_stand_still = self.reset_stand_still[env_ids].float()
        reset_timeout = self.reset_timeout[env_ids].float()
        reset_fall = self.reset_fall[env_ids].float()
        reset_contact50 = self.reset_contact50[env_ids].float()
        reset_initial_contact50 = self.reset_initial_contact50[env_ids].float()
        reset_spawn_collision = self.reset_spawn_collision[env_ids].float()
        reset_terminate_contact = self.reset_terminate_contact[env_ids].float()
        reset_dynamic_collision = self.reset_dynamic_collision[env_ids].float()
        active_dynamic_count = self.dynamic_active_mask[env_ids].sum(dim=1).float()
        min_dynamic_clearance = self.min_dynamic_clearance[env_ids].clone()
        motion_counts = self.dynamic_motion_counts[env_ids].clone()
        safe_success = success * (total_collisions == 0).float()

        wants_replay = self._should_use_replay(env_ids)
        replay_ids = env_ids[wants_replay]
        normal_ids = env_ids[~wants_replay]

        if len(replay_ids) > 0:
            self._reset_collision_replay(replay_ids)
        if len(normal_ids) > 0:
            self._reset_dofs(normal_ids)
            self._reset_root_states(normal_ids)
            self.is_replay[normal_ids] = False

        self.last_actions[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.obs_history_buf[env_ids, :, :] = 0.0
        self.slr_obs_hist[env_ids, :, :] = 0.0
        self.rays_hist[env_ids, :, :] = 5.0
        self.pos_hist[env_ids, :, :] = 0.0
        self.goal_hist[env_ids, :, :] = 0.0
        self.dynamic_tokens[env_ids] = 0.0
        self.dynamic_token_mask[env_ids] = False
        self.reset_buf[env_ids] = 1
        self.goal_reached_flag[env_ids] = 0
        self.stand_still_flag[env_ids] = 0
        self.goal_hold_timer[env_ids] = 0
        self.stay_timer[env_ids] = 0
        self.reach_goal[env_ids] = 0
        self.nav_actions_filtered[env_ids] = 0.0
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
        self.near_miss_event[env_ids] = False
        self.near_miss_occurred[env_ids] = False
        self.near_miss_count[env_ids] = 0
        self.last_near_miss_active[env_ids] = False
        self.min_ttc[env_ids] = 10.0
        self.predicted_min_clearance[env_ids] = 5.0
        self.min_dynamic_clearance[env_ids] = 5.0
        self.shield_intervention_rate[env_ids] = 0.0
        self.dynamic_cbf_intervention_rate[env_ids] = 0.0
        self.shield_intervention_sum[env_ids] = 0.0
        self.dynamic_cbf_intervention_sum[env_ids] = 0.0
        self.shield_intervention_steps[env_ids] = 0.0
        self.dynamic_cbf_intervention_steps[env_ids] = 0.0
        self.prev_distance[env_ids] = self.distance[env_ids]
        self.reset_goal[env_ids] = False
        self.reset_stand_still[env_ids] = False
        self.reset_timeout[env_ids] = False
        self.reset_fall[env_ids] = False
        self.reset_contact50[env_ids] = False
        self.reset_initial_contact50[env_ids] = False
        self.reset_spawn_collision[env_ids] = False
        self.reset_terminate_contact[env_ids] = False
        self.reset_dynamic_collision[env_ids] = False

        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            )
            self.episode_sums[key][env_ids] = 0.0
        self.extras["episode"]["success"] = torch.mean(success)
        self.extras["episode"]["safe_success"] = torch.mean(safe_success)
        self.extras["episode"]["timeout"] = torch.mean(timeout)
        self.extras["episode"]["dynamic_collision_count"] = torch.mean(dynamic_collisions)
        self.extras["episode"]["body_collision_count"] = torch.mean(body_collisions)
        self.extras["episode"]["total_collision_count"] = torch.mean(total_collisions)
        self.extras["episode"]["near_miss_count"] = torch.mean(near_miss)
        self.extras["episode"]["min_ttc"] = torch.mean(min_ttc)
        self.extras["episode"]["shield_intervention_rate"] = torch.mean(shield_rate)
        self.extras["episode"]["shield_intervention_mean"] = torch.mean(shield_mean)
        self.extras["episode"]["shield_intervention_step_rate"] = torch.mean(shield_step_rate)
        self.extras["episode"]["active_dynamic_count"] = torch.mean(active_dynamic_count)
        self.extras["episode"]["min_dynamic_clearance"] = torch.mean(min_dynamic_clearance)
        self.extras["episode"]["dynamic_cbf_intervention_rate"] = torch.mean(dynamic_shield_rate)
        self.extras["episode"]["dynamic_cbf_intervention_mean"] = torch.mean(dynamic_shield_mean)
        self.extras["episode"]["dynamic_cbf_intervention_step_rate"] = torch.mean(dynamic_shield_step_rate)
        self.extras["episode"]["reset_goal"] = torch.mean(reset_goal)
        self.extras["episode"]["reset_stand_still"] = torch.mean(reset_stand_still)
        self.extras["episode"]["reset_timeout"] = torch.mean(reset_timeout)
        self.extras["episode"]["reset_fall"] = torch.mean(reset_fall)
        self.extras["episode"]["reset_contact50"] = torch.mean(reset_contact50)
        self.extras["episode"]["reset_initial_contact50"] = torch.mean(reset_initial_contact50)
        self.extras["episode"]["reset_spawn_collision"] = torch.mean(reset_spawn_collision)
        self.extras["episode"]["reset_terminate_contact"] = torch.mean(reset_terminate_contact)
        self.extras["episode"]["reset_dynamic_collision"] = torch.mean(reset_dynamic_collision)
        self.extras["episode"]["episode_duration"] = torch.mean(episode_len_s)
        if torch.any(success > 0):
            self.extras["episode"]["time_to_goal"] = torch.mean(episode_len_s[success > 0])
        else:
            self.extras["episode"]["time_to_goal"] = torch.tensor(
                0.0, device=self.device
            )
        motion_names = [
            "linear_crossing",
            "linear_diagonal",
            "circular",
            "figure_eight",
        ]
        for idx, name in enumerate(motion_names):
            self.extras["episode"][f"motion_count_{name}"] = torch.mean(
                motion_counts[:, idx].float()
            )
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def _advance_linear_slot(self, env_idx, slot_idx):
        cfg = self._dynamic_cfg()
        center = self.dynamic_centers[env_idx, slot_idx]
        velocity = self.dynamic_velocity_vectors[env_idx, slot_idx]
        next_center = center + velocity * self.dt
        x_min, x_max, y_min, y_max = self._room_bounds()
        motion_type = int(self.dynamic_motion_types[env_idx, slot_idx].item())
        if motion_type == self.MOTION_LINEAR_CROSSING:
            if next_center[1] < y_min or next_center[1] > y_max:
                velocity[1] = -velocity[1]
                next_center[1] = center[1] + velocity[1] * self.dt
        else:
            if (
                next_center[0] < x_min
                or next_center[0] > x_max
                or next_center[1] < y_min
                or next_center[1] > y_max
            ):
                speed = torch.norm(velocity).item()
                anchor = self._sample_candidate_center(env_idx, self.MOTION_LINEAR_DIAGONAL)
                self.dynamic_centers[env_idx, slot_idx] = anchor
                self.dynamic_anchor_points[env_idx, slot_idx] = anchor
                self.dynamic_velocity_vectors[env_idx, slot_idx] = self._sample_linear_diagonal_velocity(
                    speed
                )
                self._update_slot_pose_from_state(env_idx, slot_idx)
                return
        self.dynamic_centers[env_idx, slot_idx] = next_center
        self.dynamic_anchor_points[env_idx, slot_idx] = next_center
        self.dynamic_velocity_vectors[env_idx, slot_idx] = velocity
        self._update_slot_pose_from_state(env_idx, slot_idx)

    def _update_dynamic_obstacles(self):
        if self.dynamic_obs_count == 0:
            return
        if self._uses_precomputed_dynamic_trajectories():
            self.dynamic_traj_step = torch.clamp(
                self.dynamic_traj_step + 1, max=self._trajectory_len() - 1
            )
            batch = torch.arange(self.num_envs, device=self.device)
            self.dynamic_local_pos[:] = self.dynamic_traj_pos[
                batch, :, self.dynamic_traj_step
            ]
            self.dynamic_velocity_vectors[:] = self.dynamic_traj_vel[
                batch, :, self.dynamic_traj_step
            ]
            self._sync_dynamic_root_states()
            return
        active_env_ids = torch.arange(self.num_envs, device=self.device)
        for env_idx in active_env_ids.tolist():
            for slot_idx in range(self.dynamic_obs_count):
                if not self.dynamic_active_mask[env_idx, slot_idx]:
                    continue
                motion_type = int(self.dynamic_motion_types[env_idx, slot_idx].item())
                if motion_type in (self.MOTION_LINEAR_CROSSING, self.MOTION_LINEAR_DIAGONAL):
                    self._advance_linear_slot(env_idx, slot_idx)
                else:
                    self.dynamic_phase[env_idx, slot_idx] = self._wrap_phase(
                        self.dynamic_phase_speed[env_idx, slot_idx] * self.dt
                        + self.dynamic_phase[env_idx, slot_idx]
                    )
                    self._update_slot_pose_from_state(env_idx, slot_idx)

        self._sync_dynamic_root_states(active_env_ids)

    def _post_physics_step_callback(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        self.contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        self._update_dynamic_obstacles()
        self.update_percetion()

    def update_percetion(self):
        previous_distance = self.distance.clone()
        self.distance = torch.norm(
            self.position_targets[:, :2] - self.root_states[:, :2], dim=1
        )
        self.prev_distance = torch.where(
            self.episode_length_buf <= 1, self.distance, previous_distance
        )
        self.far_goal = self.distance > 0.5
        self._get_rays()
        self._update_dynamic_tokens()
        if hasattr(self, "nav_actions_orig") and hasattr(self, "nav_actions_after_clip"):
            intervention = torch.norm(
                self.nav_actions_after_clip - self.nav_actions_orig, dim=-1
            )
            self.shield_intervention_rate = intervention
            active_step = (self.episode_length_buf > 1).float()
            self.shield_intervention_sum += intervention * active_step
            self.shield_intervention_steps += (intervention > 1e-3).float() * active_step
            if self.dynamic_obs_count > 0:
                rel_pos = self.dynamic_root_states[:, :, :2] - self.root_states[:, None, :2]
                rel_vel = self.dynamic_velocity_vectors - self.nav_actions_after_clip[:, None, :2]
                rel_speed_sq = (rel_vel ** 2).sum(dim=-1).clamp(min=1e-6)
                closing = -(rel_pos * rel_vel).sum(dim=-1)
                ttc = torch.where(closing > 0.0, closing / rel_speed_sq, torch.full_like(closing, 10.0))
                ttc = torch.where(self.dynamic_active_mask, ttc, torch.full_like(ttc, 10.0))
                dynamic_intervention = torch.where(
                    ttc.min(dim=-1).values < self.cfg.rewards.ttc_risk_config.threshold,
                    intervention,
                    torch.zeros_like(intervention),
                )
                self.dynamic_cbf_intervention_rate = dynamic_intervention
                self.dynamic_cbf_intervention_sum += dynamic_intervention * active_step
                self.dynamic_cbf_intervention_steps += (dynamic_intervention > 1e-3).float() * active_step
            else:
                self.dynamic_cbf_intervention_rate[:] = 0.0

    def _predict_local_positions(self, horizon):
        if self._uses_precomputed_dynamic_trajectories():
            offset = int(round(float(horizon) / max(self.dt, 1e-6)))
            pred_step = torch.clamp(
                self.dynamic_traj_step + offset,
                min=0,
                max=self._trajectory_len() - 1,
            )
            batch = torch.arange(self.num_envs, device=self.device)
            local_pos = self.dynamic_traj_pos[batch, :, pred_step]
            velocity = self.dynamic_traj_vel[batch, :, pred_step]
            far_local = torch.ones_like(local_pos) * 20.0
            local_pos = torch.where(self.dynamic_active_mask.unsqueeze(-1), local_pos, far_local)
            velocity = torch.where(
                self.dynamic_active_mask.unsqueeze(-1),
                velocity,
                torch.zeros_like(velocity),
            )
            return local_pos, velocity
        local_pos = self.dynamic_local_pos.clone()
        velocity = self.dynamic_velocity_vectors.clone()
        if self.dynamic_obs_count == 0:
            return local_pos, velocity
        active = self.dynamic_active_mask
        for motion_type in (
            self.MOTION_CIRCULAR,
            self.MOTION_FIGURE_EIGHT,
        ):
            mask = active & (self.dynamic_motion_types == motion_type)
            if mask.any():
                indices = mask.nonzero(as_tuple=False)
                for idx in indices.tolist():
                    env_idx, slot_idx = idx
                    phase = self.dynamic_phase[env_idx, slot_idx] + self.dynamic_phase_speed[
                        env_idx, slot_idx
                    ] * horizon
                    shape = self.dynamic_shape_params[env_idx, slot_idx]
                    center = self.dynamic_centers[env_idx, slot_idx]
                    if motion_type == self.MOTION_CIRCULAR:
                        radius = shape[0]
                        omega = self.dynamic_phase_speed[env_idx, slot_idx]
                        local_pos[env_idx, slot_idx] = center + torch.tensor(
                            [radius * torch.cos(phase), radius * torch.sin(phase)],
                            device=self.device,
                        )
                        velocity[env_idx, slot_idx] = torch.tensor(
                            [
                                -radius * omega * torch.sin(phase),
                                radius * omega * torch.cos(phase),
                            ],
                            device=self.device,
                        )
                    else:
                        scale_x = shape[0]
                        scale_y = shape[1]
                        omega = self.dynamic_phase_speed[env_idx, slot_idx]
                        local_pos[env_idx, slot_idx] = center + torch.tensor(
                            [scale_x * torch.sin(phase), scale_y * torch.sin(2.0 * phase)],
                            device=self.device,
                        )
                        velocity[env_idx, slot_idx] = torch.tensor(
                            [
                                scale_x * omega * torch.cos(phase),
                                2.0 * scale_y * omega * torch.cos(2.0 * phase),
                            ],
                            device=self.device,
                        )
        linear_mask = active & (
            (self.dynamic_motion_types == self.MOTION_LINEAR_CROSSING)
            | (self.dynamic_motion_types == self.MOTION_LINEAR_DIAGONAL)
        )
        if linear_mask.any():
            local_pos[linear_mask] = local_pos[linear_mask] + velocity[linear_mask] * horizon
        return local_pos, velocity

    def _ray_query_local_centers(self, centers):
        x0 = torch.zeros((self.num_envs, 1), device=self.device)
        y0 = torch.zeros((self.num_envs, 1), device=self.device)
        theta = self.ray_angles.unsqueeze(0).repeat(self.num_envs, 1)
        ray_dists = []
        for idx in range(self.dynamic_obs_count):
            ray_dists.append(
                circle_ray_query(
                    x0,
                    y0,
                    theta,
                    centers[:, idx, :],
                    self.dynamic_radius,
                    self.cfg.sensors.ray2d.min_dist,
                    self.cfg.sensors.ray2d.max_dist,
                )
            )
        if len(ray_dists) == 0:
            return torch.ones_like(self.rays) * self.cfg.sensors.ray2d.max_dist
        return torch.stack(ray_dists, dim=0).min(dim=0).values

    def _get_dynamic_rays(self):
        if self.dynamic_obs_count == 0:
            return torch.ones_like(self.rays) * self.cfg.sensors.ray2d.max_dist
        robot_pos = self.root_states[:, :3].unsqueeze(1)
        local_pos = quat_rotate_inverse(
            yaw_quat(self.base_quat)
            .unsqueeze(1)
            .repeat(1, self.dynamic_obs_count, 1)
            .view(-1, 4),
            (self.dynamic_root_states[:, :, :3] - robot_pos).reshape(-1, 3),
        )
        centers = local_pos[:, :2].view(self.num_envs, self.dynamic_obs_count, 2)
        current_rays = self._ray_query_local_centers(centers)
        self.dynamic_rays = current_rays

        future_horizons = getattr(self._dynamic_cfg(), "future_horizons", [])
        if not future_horizons:
            self.predicted_dynamic_rays = current_rays
            return current_rays

        ray_candidates = [current_rays]
        min_clearance = torch.full(
            (self.num_envs,), self.cfg.sensors.ray2d.max_dist, device=self.device
        )
        for horizon in future_horizons:
            pred_local_pos, _ = self._predict_local_positions(float(horizon))
            pred_world = self.room_origins.unsqueeze(1) + pred_local_pos
            rel_world = torch.cat(
                [
                    pred_world - self.root_states[:, None, :2],
                    torch.zeros(
                        self.num_envs,
                        self.dynamic_obs_count,
                        1,
                        device=self.device,
                        dtype=torch.float,
                    ),
                ],
                dim=-1,
            )
            local_pred = quat_rotate_inverse(
                yaw_quat(self.base_quat)
                .unsqueeze(1)
                .repeat(1, self.dynamic_obs_count, 1)
                .view(-1, 4),
                rel_world.reshape(-1, 3),
            )
            pred_centers = local_pred[:, :2].view(
                self.num_envs, self.dynamic_obs_count, 2
            )
            pred_rays = self._ray_query_local_centers(pred_centers)
            ray_candidates.append(pred_rays)
            clearance = torch.norm(pred_centers, dim=-1) - self.dynamic_radius
            clearance = torch.where(
                self.dynamic_active_mask,
                clearance,
                torch.full_like(clearance, self.cfg.sensors.ray2d.max_dist),
            )
            min_clearance = torch.minimum(min_clearance, clearance.min(dim=-1).values)

        self.predicted_min_clearance = min_clearance
        self.predicted_dynamic_rays = torch.stack(ray_candidates, dim=0).min(dim=0).values
        return self.predicted_dynamic_rays

    def _get_rays(self, env_ids=None):
        if not hasattr(self.terrain, "height_points"):
            self.height_points = self._init_height_points()

        if env_ids is not None:
            points = quat_apply_yaw(
                self.base_quat[env_ids].repeat(1, self.num_height_points),
                self.height_points[env_ids],
            ) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(
                self.base_quat.repeat(1, self.num_height_points), self.height_points
            ) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights = torch.max(heights1, heights2)
        self.measured_heights = (
            heights.view(self.num_envs, self.len_x, self.len_y)
            * self.terrain.cfg.vertical_scale
        )
        center_height = self.measured_heights[:, self.c_x, self.c_y].unsqueeze(1).unsqueeze(2)
        raw_heights = torch.where(self.measured_heights > center_height + 0.1, 1.0, 0.0)
        self.static_rays = self._grid2ray(raw_heights) * self.cfg.terrain.measure_resolution
        dynamic_future_rays = self._get_dynamic_rays()
        self.rays = torch.minimum(self.static_rays, dynamic_future_rays)

    def _compute_min_ttc(self):
        if self.dynamic_obs_count == 0:
            self.min_ttc[:] = 10.0
            return self.min_ttc
        rel_pos_world = self.dynamic_root_states[:, :, :2] - self.root_states[:, None, :2]
        rel_vel_world = self.dynamic_velocity_vectors - self.base_lin_vel[:, None, :2]
        rel_speed_sq = (rel_vel_world ** 2).sum(dim=-1).clamp(min=1e-6)
        closing = -(rel_pos_world * rel_vel_world).sum(dim=-1)
        ttc = closing / rel_speed_sq
        ttc = torch.where(closing > 0.0, ttc, torch.full_like(ttc, 10.0))
        ttc = torch.where(self.dynamic_active_mask, ttc, torch.full_like(ttc, 10.0))
        self.min_ttc = ttc.min(dim=-1).values
        return self.min_ttc

    def _update_dynamic_tokens(self):
        token_k = getattr(self.cfg.env, "dynamic_token_k", 0)
        if token_k == 0 or self.dynamic_obs_count == 0:
            return
        min_ttc = self._compute_min_ttc()
        rel_pos_world = self.dynamic_root_states[:, :, :2] - self.root_states[:, None, :2]
        local_quat = yaw_quat(self.base_quat).unsqueeze(1).repeat(
            1, self.dynamic_obs_count, 1
        ).view(-1, 4)
        rel_local = quat_rotate_inverse(
            local_quat,
            torch.cat(
                [
                    rel_pos_world,
                    torch.zeros(
                        self.num_envs,
                        self.dynamic_obs_count,
                        1,
                        device=self.device,
                        dtype=torch.float,
                    ),
                ],
                dim=-1,
            ).reshape(-1, 3),
        )[:, :2].view(self.num_envs, self.dynamic_obs_count, 2)
        rel_vel_local = quat_rotate_inverse(
            local_quat,
            torch.cat(
                [
                    self.dynamic_velocity_vectors - self.base_lin_vel[:, None, :2],
                    torch.zeros(
                        self.num_envs,
                        self.dynamic_obs_count,
                        1,
                        device=self.device,
                        dtype=torch.float,
                    ),
                ],
                dim=-1,
            ).reshape(-1, 3),
        )[:, :2].view(self.num_envs, self.dynamic_obs_count, 2)
        distances = torch.norm(rel_local, dim=-1)
        ttc = torch.full_like(distances, 10.0)
        if self.dynamic_obs_count > 0:
            rel_pos_world = self.dynamic_root_states[:, :, :2] - self.root_states[:, None, :2]
            rel_vel_world = self.dynamic_velocity_vectors - self.base_lin_vel[:, None, :2]
            rel_speed_sq = (rel_vel_world ** 2).sum(dim=-1).clamp(min=1e-6)
            closing = -(rel_pos_world * rel_vel_world).sum(dim=-1)
            raw_ttc = closing / rel_speed_sq
            ttc = torch.where(closing > 0.0, raw_ttc, torch.full_like(raw_ttc, 10.0))
            ttc = torch.where(self.dynamic_active_mask, ttc, torch.full_like(ttc, 10.0))

        risk_scores = ttc + 0.2 * distances
        risk_scores = torch.where(
            self.dynamic_active_mask, risk_scores, torch.full_like(risk_scores, 1e6)
        )
        topk = min(token_k, self.dynamic_obs_count)
        selected = torch.topk(risk_scores, k=topk, dim=-1, largest=False)
        self.dynamic_tokens[:] = 0.0
        self.dynamic_token_mask[:] = False
        radius_value = torch.full(
            (self.num_envs, self.dynamic_obs_count),
            self.dynamic_radius,
            device=self.device,
            dtype=torch.float,
        )
        for token_idx in range(topk):
            obs_idx = selected.indices[:, token_idx]
            batch_idx = torch.arange(self.num_envs, device=self.device)
            valid = self.dynamic_active_mask[batch_idx, obs_idx]
            self.dynamic_token_mask[:, token_idx] = valid
            self.dynamic_tokens[:, token_idx, 0:2] = rel_local[batch_idx, obs_idx]
            self.dynamic_tokens[:, token_idx, 2:4] = rel_vel_local[batch_idx, obs_idx]
            self.dynamic_tokens[:, token_idx, 4] = radius_value[batch_idx, obs_idx]
            self.dynamic_tokens[:, token_idx, 5] = ttc[batch_idx, obs_idx]
            self.dynamic_tokens[:, token_idx, 6] = valid.float()

    def _get_perception(self):
        self.rays_rand = self.rays.clone() + torch.rand_like(self.rays) * 0.0
        self.rays_hist = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            torch.stack([self.rays_rand] * self.cfg.env.his_len, dim=1),
            torch.cat([self.rays_hist[:, 1:], self.rays_rand.unsqueeze(1)], dim=1),
        )
        pos_diff = self.position_targets - self.root_states[:, 0:3]
        self.goal_local_pos = quat_rotate_inverse(yaw_quat(self.base_quat), pos_diff)[:, :2]
        self.goal_hist = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            torch.stack([self.goal_local_pos] * self.cfg.env.his_len, dim=1),
            torch.cat([self.goal_hist[:, 1:], self.goal_local_pos.unsqueeze(1)], dim=1),
        )

    def compute_observations(self):
        self._update_replay_buffer()
        self.prop_buf = torch.cat(
            (
                self.projected_gravity,
                self.slr_commands[:, :3] * self.commands_scale[:3],
                self.base_lin_vel * 1.0,
                self.base_ang_vel * 1.0,
            ),
            dim=-1,
        )

        noise_scales = self.cfg.noise.noise_scales
        noise_vec = torch.cat(
            (
                torch.ones(3) * noise_scales.gravity,
                torch.zeros(3),
                torch.ones(3) * noise_scales.lin_vel * 1.0,
                torch.ones(3) * noise_scales.ang_vel * 1.0,
            ),
            dim=0,
        )
        if self.cfg.noise.add_noise:
            self.prop_buf += (2 * torch.rand_like(self.prop_buf) - 1) * noise_vec.to(
                self.device
            )

        self._get_perception()

        env_ids = (
            self.episode_length_buf % int(self.cfg.commands.delay_time / self.dt) == 0
        ).nonzero(as_tuple=False).flatten()
        if len(env_ids) != 0:
            resample_time_idx = -torch.randint(
                2, 4, (len(env_ids),), device=self.device
            ) - 1
            self.delay_rays[env_ids] = self.rays_hist[env_ids, resample_time_idx, :]
            self.delay_goal[env_ids] = self.goal_hist[env_ids, resample_time_idx, :]

        env_ids = (self.episode_length_buf % 10 == 0).nonzero(as_tuple=False).flatten()
        self.pos_hist[env_ids] = torch.where(
            (self.episode_length_buf[env_ids] <= 1)[:, None, None],
            torch.stack([self.root_states[env_ids, :2]] * self.cfg.env.his_len, dim=1),
            torch.cat(
                [self.pos_hist[env_ids, 1:], self.root_states[env_ids, :2].unsqueeze(1)],
                dim=1,
            ),
        )

        dynamic_tokens_flat = self.dynamic_tokens.view(self.num_envs, -1)
        obs_buf = torch.cat(
            (
                self.prop_buf,
                torch.log2(self.delay_rays.clip(min=0.1, max=5.0)),
                self.delay_goal,
                dynamic_tokens_flat,
            ),
            dim=-1,
        )

        self.obs_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            torch.stack([obs_buf] * self.cfg.env.his_len, dim=1),
            torch.cat([self.obs_history_buf[:, 1:], obs_buf.unsqueeze(1)], dim=1),
        )
        self.obs_buf = self.obs_history_buf.view(self.num_envs, -1)

    def _update_collision_events(self, collision_now, cooldown_buf, count_buf):
        cooldown_buf[:] = torch.clamp(cooldown_buf - 1, min=0)
        new_event = collision_now & (cooldown_buf == 0)
        count_buf += new_event.long()
        cooldown_buf[new_event] = self.cfg.rewards.dynamic_collision_config.cooldown_steps
        return new_event

    def check_termination(self):
        self.initial_ = self.episode_length_buf <= 1
        self.reach_goal = (
            self.distance < self.cfg.rewards.reach_pos_target_tight_config.distance_threshold
        )

        terminate_contact_now = torch.any(
            torch.norm(self.contact_forces[:, self.termination_contact_indices, :2], dim=-1)
            > 1.0,
            dim=1,
        )
        self.terminate_buf = terminate_contact_now & (~self.initial_)
        self.reset_buf = self.terminate_buf.clone()

        if self.termination_contact_indices.numel() > 0:
            hard_contact_now = torch.any(
                torch.norm(
                    self.contact_forces[:, self.termination_contact_indices, :2], dim=-1
                )
                > 50.0,
                dim=1,
            )
        else:
            hard_contact_now = torch.zeros(
                self.num_envs, device=self.device, dtype=torch.bool
            )
        warmup_steps = int(getattr(self.cfg.env, "hard_contact_warmup_steps", 10))
        hard_contact_ready = self.episode_length_buf > warmup_steps
        hard_contact_reset = hard_contact_now & hard_contact_ready
        self.reset_buf |= hard_contact_reset
        if self.initial_.any():
            self._check_spawn_collision()
        self.extras["bad_masks"] = self.initial_

        body_collision_now = torch.any(
            torch.norm(self.contact_forces[:, self.penalised_contact_indices, :2], dim=-1)
            > 1.0,
            dim=1,
        )
        body_collision_now &= ~self.initial_
        self.body_collision_event = self._update_collision_events(
            body_collision_now, self.body_collision_cooldown, self.body_collision_count
        )

        if self.dynamic_obs_count > 0:
            dist = torch.norm(
                self.dynamic_root_states[:, :, :2] - self.root_states[:, None, :2], dim=-1
            )
            dist = torch.where(
                self.dynamic_active_mask,
                dist,
                torch.full_like(dist, 100.0),
            )
            self.min_dynamic_clearance = dist.min(dim=1).values - self.dynamic_radius
            dynamic_collision_now = torch.any(
                dist < self.cfg.rewards.dynamic_collision_config.threshold, dim=1
            )
            dynamic_collision_now &= ~self.initial_
            self.dynamic_collision_event = self._update_collision_events(
                dynamic_collision_now,
                self.dynamic_collision_cooldown,
                self.dynamic_collision_count,
            )
        else:
            self.min_dynamic_clearance[:] = 5.0
            self.dynamic_collision_event = torch.zeros(
                self.num_envs, device=self.device, dtype=torch.bool
            )

        near_cfg = self.cfg.rewards.near_miss_config
        self._compute_min_ttc()
        near_miss_now = (
            (self.min_ttc < near_cfg.ttc_threshold)
            | (self.predicted_min_clearance < near_cfg.clearance_threshold)
        ) & (~self.initial_)
        near_miss_onset = near_miss_now & (~self.last_near_miss_active)
        self.near_miss_event = near_miss_now
        self.near_miss_occurred |= near_miss_now
        self.near_miss_count += near_miss_onset.long()
        self.last_near_miss_active = near_miss_now

        self.total_collision_count = self.body_collision_count + self.dynamic_collision_count
        self.collision_occurred |= body_collision_now | self.dynamic_collision_event
        self.last_collision_active = body_collision_now

        if torch.any(body_collision_now):
            self._update_collision_hist(body_collision_now)

        dynamic_collision_reset = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        if self.cfg.rewards.dynamic_collision_config.early_reset:
            dynamic_collision_reset = self.dynamic_collision_event
            self.reset_buf |= dynamic_collision_reset
            self.terminate_buf |= dynamic_collision_reset

        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.fall_down = self.projected_gravity[:, 2] > -0.8

        v_low = (torch.norm(self.base_lin_vel[:, :2], dim=-1) < 0.1) & (
            torch.abs(self.base_ang_vel[:, 2]) < 0.1
        )
        d_low = (
            torch.norm(self.root_states[:, :2] - self.pos_hist[:, 0, :2], dim=-1) < 0.2
        )
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

        self.reset_goal = self.goal_reached_flag.clone()
        self.reset_stand_still = self.stand_still_flag.clone()
        self.reset_timeout = self.time_out_buf.clone()
        self.reset_fall = self.fall_down.clone()
        self.reset_contact50 = hard_contact_reset.clone()
        self.reset_initial_contact50 = hard_contact_now & (~hard_contact_ready)
        self.reset_terminate_contact = self.terminate_buf & terminate_contact_now
        self.reset_dynamic_collision = dynamic_collision_reset.clone()

    def _reward_goal_progress(self):
        cfg = getattr(self.cfg.rewards, "goal_progress_config", None)
        max_progress = float(getattr(cfg, "max_progress", 0.25)) if cfg is not None else 0.25
        progress = (self.prev_distance - self.distance).clip(min=-max_progress, max=max_progress)
        return progress * (self.episode_length_buf > 1).float()

    def _reward_dynamic_collision(self):
        return self.dynamic_collision_event.float()

    def _reward_ttc_risk(self):
        cfg = self.cfg.rewards.ttc_risk_config
        ttc = self.min_ttc
        return torch.where(
            ttc < cfg.threshold,
            (cfg.threshold - ttc) / max(cfg.threshold - cfg.saturation, 1e-3),
            torch.zeros_like(ttc),
        ).clip(min=0.0)

    def _reward_near_miss(self):
        cfg = self.cfg.rewards.near_miss_config
        clearance_pen = (cfg.clearance_threshold - self.predicted_min_clearance).clip(
            min=0.0
        )
        ttc_pen = (cfg.ttc_threshold - self.min_ttc).clip(min=0.0)
        return clearance_pen + 0.5 * ttc_pen

    def _reward_close_obst_vel(self):
        rew_cfg = self.cfg.rewards.close_obst_vel_config
        front_clearance = torch.min(self.static_rays, self.predicted_dynamic_rays).min(
            dim=-1
        ).values
        dir_alignment = self._get_guidance_nav_alignment(fov_deg=rew_cfg.fov_deg)
        x_vel = self.base_lin_vel[:, 0].clip(min=0.0)
        safe_vel_limit = (front_clearance * rew_cfg.safe_vel_scale).clip(
            max=rew_cfg.safe_vel_max
        )
        reward_vel_clamped = torch.min(x_vel, safe_vel_limit)
        reward_base = dir_alignment * reward_vel_clamped
        overspeed = (x_vel - safe_vel_limit).clip(min=0.0)
        overspeed_penalty = overspeed * rew_cfg.overspeed_penalty_weight
        reward = (reward_base - overspeed_penalty).clip(min=0.0)
        reach_bonus = rew_cfg.reach_bonus_weight / (1.0 + 2 * torch.square(self.distance))
        return reward * self.far_goal + (~self.far_goal) * reach_bonus
