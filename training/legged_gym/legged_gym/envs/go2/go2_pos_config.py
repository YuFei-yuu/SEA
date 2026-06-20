# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfgPPO
from legged_gym.envs.base.legged_robot_pos_config import LeggedRobotPosCfg
import numpy as np


DYNAMIC_ROOM_SPARSE_OBSTACLE_BOXES = (
    (3.6, 2.6, 0.8, 0.8, 0.6),
    (6.4, 5.0, 0.8, 0.8, 0.6),
    (3.6, 7.4, 0.8, 0.8, 0.6),
)


DYNAMIC_ROOM_COMPLEX_OBSTACLE_BOXES = (
    (1.6, 2.0, 0.7, 0.7, 0.6),
    (2.9, 1.2, 1.0, 0.4, 0.6),
    (4.7, 1.9, 0.5, 1.1, 0.6),
    (6.5, 1.2, 1.2, 0.4, 0.6),
    (8.4, 2.0, 0.7, 0.7, 0.6),
    (1.4, 3.9, 0.4, 1.1, 0.6),
    (3.1, 4.3, 0.4, 1.4, 0.6),
    (5.2, 3.6, 0.8, 0.5, 0.6),
    (6.8, 4.5, 0.5, 1.1, 0.6),
    (8.6, 4.0, 0.4, 1.2, 0.6),
    (1.6, 6.2, 0.7, 0.7, 0.6),
    (3.3, 6.9, 1.0, 0.4, 0.6),
    (5.0, 5.9, 0.7, 0.7, 0.6),
    (6.6, 6.8, 1.1, 0.4, 0.6),
    (8.4, 6.1, 0.7, 0.7, 0.6),
    (2.4, 8.1, 1.1, 0.4, 0.6),
    (4.4, 7.8, 0.5, 1.0, 0.6),
    (6.3, 8.1, 1.1, 0.4, 0.6),
    (7.9, 7.7, 0.4, 1.0, 0.6),
)


class Go2PosRoughCfg(LeggedRobotPosCfg):
    class loco:
        num_obs_buf = 45
        his_len = 10
        class normalization:
            class obs_scales:
                lin_vel = 2.0
                ang_vel = 0.25
                dof_pos = 1.0
                dof_vel = 0.05

    class env(LeggedRobotPosCfg.env):
        goal_reached_time = 150
        stay_time = 150
        his_len = 10
        num_nav_actions = 3
        num_props = 12
        num_rays = 41
        num_goal_obs = 2
        dynamic_token_k = 0
        dynamic_token_dim = 7
        num_dynamic_obs = dynamic_token_k * dynamic_token_dim
        num_obs_one_step = num_props + num_rays + num_goal_obs + num_dynamic_obs
        num_observations = num_obs_one_step * his_len
        num_envs = 2048
        episode_length_s = 60
        debug_viz = True

    class replay:
        enable_collision_replay = True
        replay_prob = 0.8
        early_reset_prob_range = [0.1, 0.5]
        undo_steps_range = [100, 150]
        max_collision_points = 10

    class visualization:
        draw_scan_dots = False
        draw_rays = True
        draw_collision_points = True
        draw_position_target = True
        draw_dynamic_obstacles = True
        ray_groups = {
            "guidance_navigation": ["guide", "guide_ray_marker"],
        }
        points = {
            "ray_pink": [0.015, 4, (1.0, 0.2, 1.0)],
            "ray_yellow": [0.015, 4, (1.0, 1.0, 0.0)],
            "ray_red": [0.015, 4, (1.0, 0.0, 0.0)],
            "ray_green": [0.015, 4, (0.0, 1.0, 0.0)],
            "goal_marker": [0.2, 8, (0.0, 0.0, 1.0)],
            "collision_marker": [0.10, 4, (1.0, 0.0, 0.0)],
            "scan_dot_obs": [0.05, 4, (1.0, 0.0, 0.0)],
            "scan_dot_safe": [0.05, 4, (0.0, 1.0, 0.0)],
            "guide_ray_marker": [0.06, 6, (0.0, 1.0, 1.0)],
        }

    class init_state(LeggedRobotPosCfg.init_state):
        pos = [0.0, 0.0, 0.42]
        default_joint_angles = {
            'FL_hip_joint': 0.1,
            'RL_hip_joint': 0.1,
            'FR_hip_joint': -0.1,
            'RR_hip_joint': -0.1,
            'FL_thigh_joint': 0.8,
            'RL_thigh_joint': 1.0,
            'FR_thigh_joint': 0.8,
            'RR_thigh_joint': 1.0,
            'FL_calf_joint': -1.5,
            'RL_calf_joint': -1.5,
            'FR_calf_joint': -1.5,
            'RR_calf_joint': -1.5,
        }

    class commands(LeggedRobotPosCfg.commands):
        curriculum = False
        max_curriculum = 1.
        num_commands = 3
        delay_time = 0.1
        alpha = 0.5
        class ranges(LeggedRobotPosCfg.commands.ranges):
            limit_vx = [-0.5, 2.0]
            limit_vy = [-1.0, 1.0]
            limit_vyaw = [-1.0, 1.0]

    class control(LeggedRobotPosCfg.control):
        control_type = 'P'
        stiffness = {'joint': 30.}
        damping = {'joint': 0.75}
        action_scale = 0.25
        decimation = 4

    class asset(LeggedRobotPosCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/go2_description/urdf/go2_description_v8.urdf'
        flip_visual_attachments = True
        fix_base_link = False
        name = "Go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf", "Head_upper", "Head_lower", "base"]
        terminate_after_contacts_on = ["base", "Head_upper", "Head_lower"]
        self_collisions = 1

    class terrain(LeggedRobotPosCfg.terrain):
        terrain_types = ['hard_room']
        terrain_proportions = [1.0]
        num_rows = 10
        num_cols = 10
        measure_heights = True

    class domain_rand:
        randomize_friction = True
        friction_range = [-0.2, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1.5, 1.5]
        push_robots = True
        push_interval_s = 2.5
        max_push_vel_xy = 0.0
        randomize_yaw = True
        randomize_roll = False
        randomize_pitch = False
        init_yaw_range = [-3.14, 3.14]
        init_roll_range = [-0.1, 0.1]
        init_pitch_range = [-0.1, 0.1]
        randomize_xy = True
        init_x_range = [-0.5, 0.5]
        init_y_range = [-0.5, 0.5]
        randomize_velo = False
        init_vlinx_range = [-0.5, 0.5]
        init_vliny_range = [-0.5, 0.5]
        init_vlinz_range = [-0.5, 0.5]
        init_vang_range = [-0.5, 0.5]

    class sensors:
        class ray2d:
            enable = False
            log2 = True
            min_dist = 0.1
            max_dist = 3.0
            theta_start = -2 * np.pi / 3
            theta_end = 2 * np.pi / 3 + 0.0001
            theta_step = np.pi / 30

    class normalization:
        class obs_scales:
            lin_vel = 1.0
            ang_vel = 1.0
            dof_pos = 1.0
            dof_vel = 0.2
            height_measurements = 2.0
            ray2d = 1.0
        clip_observations = 100.
        clip_actions = 100.

    class noise:
        add_noise = True
        noise_level = 1.0
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.0
            lin_vel = 0.1
            ang_vel = 0.1
            gravity = 0.05
            height_measurements = 0.1

    class rewards:
        class scales:
            termination = -100.0
            collision = -4.0
            close_obst_vel = 5.0
            stuck = -5.0
            velo_dir = 4.0
            reach_pos_target_tight = 10.0
            ang_vel_xy = -0.05
            proximity = 0.0

        class proximity_config:
            slope = 15.0
            min_dist = 0.1
            narrow_slope = 10.0
            speed_limit_scale = 1.0
            speed_limit_min = 0.1
            speed_limit_max = 1.0
            rear_x_range = [-0.4, -0.1]
            rear_y_range = [-0.3, 0.3]
            obstacle_height_th = 0.15
            rear_penalty_weight = 1.0
            speed_penalty_weight = 2.0

        class velo_dir_config:
            target_speed_max = 0.5
            target_speed_scale = 1.0
            orbit_penalty_weight = 1.0
            reach_bonus_weight = 1.0

        class reach_pos_target_tight_config:
            distance_threshold = 0.5
            reach_bonus_weight = 1.0

        class close_obst_vel_config:
            max_rays_clip = 2.0
            kernel_size = 5
            fov_deg = 150.0
            safe_vel_scale = 1.0
            safe_vel_max = 0.5
            reach_bonus_weight = 1.0
            overspeed_penalty_weight = 0.2

        class stuck_config:
            fov_deg = 120.0
            move_dist_threshold = 0.1
            dead_end_threshold = 1.0
            backward_vel_threshold = 0.0
            turn_vel_threshold = 1.0

        class collision_config:
            contact_force_th = 0.1
            vel_square_scale = 4.0
            base_weight = 6.0
            head_weight = 10.0
            leg_weight = 10.0
            generic_weight = 1.0

        pos_level = 'normal'
        soft_dof_pos_limit = 0.95
        base_height_target = 0.25
        only_positive_rewards = False
        position_target_sigma_soft = 2.0
        position_target_sigma_tight = 0.5
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.85
        max_contact_force = 100.


class Go2PosDynamicBaseCfg(Go2PosRoughCfg):
    class env(Go2PosRoughCfg.env):
        num_envs = 512
        episode_length_s = 40
        goal_reached_time = 100
        stay_time = 200
        dynamic_token_k = 4
        dynamic_token_dim = 7
        num_dynamic_obs = dynamic_token_k * dynamic_token_dim
        num_obs_one_step = (
            Go2PosRoughCfg.env.num_props
            + Go2PosRoughCfg.env.num_rays
            + Go2PosRoughCfg.env.num_goal_obs
            + num_dynamic_obs
        )
        num_observations = num_obs_one_step * Go2PosRoughCfg.env.his_len

    class replay:
        enable_collision_replay = False
        enable_near_miss_replay = False
        replay_prob = 0.0
        early_reset_prob_range = [0.0, 0.0]
        undo_steps_range = [1, 2]
        max_collision_points = 10

    class terrain(Go2PosRoughCfg.terrain):
        terrain_types = ['dynamic_room_sparse']
        terrain_proportions = [1.0]
        num_rows = 4
        num_cols = 4
        curriculum = False
        max_init_terrain_level = 0
        obstacle_boxes = DYNAMIC_ROOM_SPARSE_OBSTACLE_BOXES
        boundary_wall_thickness = 0.4

    class commands(Go2PosRoughCfg.commands):
        class ranges(Go2PosRoughCfg.commands.ranges):
            limit_vx = [-0.3, 1.0]
            limit_vy = [-0.7, 0.7]
            limit_vyaw = [-0.7, 0.7]

    class sparse_room:
        obstacle_boxes = DYNAMIC_ROOM_SPARSE_OBSTACLE_BOXES
        pillar_centers = [(3.6, 2.6), (6.4, 5.0), (3.6, 7.4)]
        pillar_size = [0.8, 0.8, 0.6]
        start_x_left = [1.0, 1.8]
        start_x_right = [8.2, 9.0]
        goal_x_left = [1.0, 1.8]
        goal_x_right = [8.2, 9.0]
        start_y_range = [1.5, 8.5]
        goal_y_range = [1.5, 8.5]
        spawn_clearance = 0.8

    class dynamic_obstacles:
        enable = True
        count = 3
        count_range = [3, 3]
        size = [0.45, 0.45, 0.8]
        ray_radius = 0.32
        kinematic = True
        disable_simulation_contacts = False
        fixed_base_link = False
        collision_filter = 0
        lane_x = [2.2, 5.0, 7.8]
        lane_x_jitter = 0.15
        y_min = 1.2
        y_max = 8.8
        speed_range = [0.15, 0.40]
        speed_range_eval_hard = [0.35, 0.80]
        motion_type_probs = {
            "linear_crossing": 1.0,
            "linear_diagonal": 0.0,
            "circular": 0.0,
            "figure_eight": 0.0,
        }
        trajectory_mode = "online"
        trajectory_extra_horizon = 0.0
        future_horizons = [0.2, 0.4, 0.6]
        circle_radius_range = [0.6, 1.3]
        figure_eight_scale_range = [0.7, 1.2]
        diagonal_heading_jitter_deg = 20.0
        spawn_min_separation = 0.9
        spawn_start_clearance = 1.2
        spawn_goal_clearance = 1.2
        obstacle_wall_margin = 0.5
        max_spawn_attempts = 32
        interaction_band_half_width = 1.25
        force_interaction = False
        force_interaction_jitter = 0.45

    class rewards(Go2PosRoughCfg.rewards):
        class scales(Go2PosRoughCfg.rewards.scales):
            dynamic_collision = -8.0
            ttc_risk = -2.0
            near_miss = -1.0

        class close_obst_vel_config(Go2PosRoughCfg.rewards.close_obst_vel_config):
            safe_vel_max = 0.30

        class dynamic_collision_config:
            threshold = 0.60
            cooldown_steps = 10
            early_reset = True

        class ttc_risk_config:
            threshold = 1.5
            saturation = 0.25

        class near_miss_config:
            ttc_threshold = 1.0
            clearance_threshold = 0.6
            replay_clearance_threshold = 0.6
            replay_ttc_threshold = 1.0


class Go2PosSparseStaticCfg(Go2PosDynamicBaseCfg):
    class dynamic_obstacles(Go2PosDynamicBaseCfg.dynamic_obstacles):
        enable = False
        count = 0


class Go2PosDynamic1Cfg(Go2PosDynamicBaseCfg):
    class dynamic_obstacles(Go2PosDynamicBaseCfg.dynamic_obstacles):
        count = 1


class Go2PosDynamic2Cfg(Go2PosDynamicBaseCfg):
    class dynamic_obstacles(Go2PosDynamicBaseCfg.dynamic_obstacles):
        count = 2


class Go2PosDynamic3Cfg(Go2PosDynamicBaseCfg):
    class dynamic_obstacles(Go2PosDynamicBaseCfg.dynamic_obstacles):
        count = 3
        count_range = [3, 3]


class Go2PosDynamicComplexCfg(Go2PosDynamicBaseCfg):
    class env(Go2PosDynamicBaseCfg.env):
        num_envs = 384
        episode_length_s = 45
        goal_reached_time = 12
        hard_contact_warmup_steps = 10

    class terrain(Go2PosDynamicBaseCfg.terrain):
        obstacle_boxes = DYNAMIC_ROOM_COMPLEX_OBSTACLE_BOXES

    class sparse_room(Go2PosDynamicBaseCfg.sparse_room):
        obstacle_boxes = DYNAMIC_ROOM_COMPLEX_OBSTACLE_BOXES
        spawn_clearance = 0.55

    class replay(Go2PosDynamicBaseCfg.replay):
        enable_collision_replay = False
        enable_near_miss_replay = False
        replay_prob = 0.4
        undo_steps_range = [30, 60]

    class dynamic_obstacles(Go2PosDynamicBaseCfg.dynamic_obstacles):
        count = 8
        count_range = [2, 4]
        speed_range = [0.15, 0.35]
        trajectory_mode = "episode_precomputed"
        trajectory_extra_horizon = 0.8
        disable_simulation_contacts = True
        fixed_base_link = False
        collision_filter = 1
        spawn_min_separation = 0.75
        spawn_start_clearance = 1.6
        spawn_goal_clearance = 1.0
        obstacle_wall_margin = 0.35
        max_spawn_attempts = 24
        motion_type_probs = {
            "linear_crossing": 0.75,
            "linear_diagonal": 0.15,
            "circular": 0.10,
            "figure_eight": 0.00,
        }

    class rewards(Go2PosDynamicBaseCfg.rewards):
        class scales(Go2PosDynamicBaseCfg.rewards.scales):
            goal_progress = 6.0
            reach_pos_target_tight = 15.0
            near_goal_radial_reward = 4.0
            negative_progress_penalty = -5.0
            near_goal_orbit_penalty = -4.0
            near_goal_command_penalty = -2.0
            near_goal_stop_reward = 4.0
            ttc_risk = -0.8
            near_miss = -0.3
            dynamic_avoid_direction_reward = 1.0

        class goal_progress_config:
            max_progress = 0.25

        class near_goal_radial_reward_config:
            near_goal_distance = 1.6
            target_speed_scale = 0.8
            target_speed_max = 0.45

        class negative_progress_penalty_config:
            near_goal_distance = 1.8
            max_negative_progress = 0.30

        class near_goal_orbit_penalty_config:
            near_goal_distance = 1.6
            tangential_speed_threshold = 0.08
            radial_speed_deadband = 0.18
            radial_deficit_weight = 0.5

        class near_goal_command_penalty_config:
            near_goal_distance = 1.6
            target_speed_scale = 0.65
            target_speed_min = 0.05
            target_speed_max = 0.45
            tangential_weight = 0.7
            wrong_way_weight = 1.0

        class near_goal_stop_reward_config:
            near_goal_distance = 0.7
            speed_threshold = 0.35
            yaw_rate_threshold = 0.60

        class dynamic_avoid_direction_reward_config:
            ttc_threshold = 1.2
            min_lateral_speed = 0.03
            min_obstacle_speed = 0.05

        class close_obst_vel_config(Go2PosDynamicBaseCfg.rewards.close_obst_vel_config):
            safe_vel_max = 0.45

        class dynamic_collision_config(Go2PosDynamicBaseCfg.rewards.dynamic_collision_config):
            threshold = 0.65
            early_reset = True

        class near_miss_config(Go2PosDynamicBaseCfg.rewards.near_miss_config):
            ttc_threshold = 0.8
            clearance_threshold = 0.5
            replay_clearance_threshold = 0.6
            replay_ttc_threshold = 1.0


class Go2PosRoughCfgPPO(LeggedRobotCfgPPO):
    runner_class_name = 'OnPolicyRunner'
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.003

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'Go2_pos_rough'
        policy_class_name = 'DifferentiableSafeActorCritic'
        algorithm_class_name = 'PPO'
        max_iterations = 2000


class Go2PosSparseStaticCfgPPO(Go2PosRoughCfgPPO):
    class runner(Go2PosRoughCfgPPO.runner):
        run_name = ''
        experiment_name = 'Go2_pos_sparse_static'


class Go2PosDynamic1CfgPPO(Go2PosRoughCfgPPO):
    class runner(Go2PosRoughCfgPPO.runner):
        run_name = ''
        experiment_name = 'Go2_pos_dynamic_1'


class Go2PosDynamic2CfgPPO(Go2PosRoughCfgPPO):
    class runner(Go2PosRoughCfgPPO.runner):
        run_name = ''
        experiment_name = 'Go2_pos_dynamic_2'


class Go2PosDynamic3CfgPPO(Go2PosRoughCfgPPO):
    class runner(Go2PosRoughCfgPPO.runner):
        run_name = ''
        experiment_name = 'Go2_pos_dynamic_3'


class Go2PosDynamicComplexCfgPPO(Go2PosRoughCfgPPO):
    class policy(Go2PosRoughCfgPPO.policy):
        dynamic_cbf_default_safe_radius = 0.65
        dynamic_cbf_safety_margin = 0.35
        dynamic_cbf_damping_factor = 1.0

    class runner(Go2PosRoughCfgPPO.runner):
        run_name = ''
        experiment_name = 'Go2_pos_dynamic_complex'
