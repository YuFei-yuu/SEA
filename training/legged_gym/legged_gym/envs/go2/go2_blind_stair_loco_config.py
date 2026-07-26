"""Go2 blind stair locomotion configuration."""

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class Go2BlindStairLocoCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 8192
        num_observations = 45
        num_privileged_obs = None
        num_actions = 12
        num_nav_actions = 12
        num_props = 45
        his_len = 1
        episode_length_s = 20
        debug_viz = False
        disable_graphics = True

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "trimesh"
        measure_heights = False
        curriculum = True
        terrain_length = 10.0
        terrain_width = 10.0
        num_rows = 10
        num_cols = 10
        min_init_terrain_level = 0
        max_init_terrain_level = 2
        terrain_types = ["flat", "blind_stair_up", "blind_stair_down"]
        terrain_proportions = [0.2, 0.4, 0.4]
        blind_stair_heights = (0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.10)
        blind_stair_count = 5
        blind_stair_tread = 0.30
        blind_stair_platform_size = 3.0

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        num_commands = 4
        resampling_time = 10.0
        heading_command = False
        low_speed_focus = False

        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-0.7, 0.7]
            ang_vel_yaw = [-0.7, 0.7]
            heading = [-3.14, 3.14]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.42]
        default_joint_angles = {
            "FL_hip_joint": 0.1,
            "FL_thigh_joint": 0.8,
            "FL_calf_joint": -1.5,
            "FR_hip_joint": -0.1,
            "FR_thigh_joint": 0.8,
            "FR_calf_joint": -1.5,
            "RL_hip_joint": 0.1,
            "RL_thigh_joint": 1.0,
            "RL_calf_joint": -1.5,
            "RR_hip_joint": -0.1,
            "RR_thigh_joint": 1.0,
            "RR_calf_joint": -1.5,
        }

    class control(LeggedRobotCfg.control):
        control_type = "P"
        stiffness = {"joint": 30.0}
        damping = {"joint": 0.75}
        action_scale = 0.25
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = f"{LEGGED_GYM_ROOT_DIR}/resources/go2_description/urdf/go2_description_v8.urdf"
        name = "Go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf", "Head_upper", "Head_lower", "base"]
        terminate_after_contacts_on = ["base", "Head_upper", "Head_lower"]
        self_collisions = 1
        flip_visual_attachments = True

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.8, 1.2]
        randomize_base_mass = False
        push_robots = False
        randomize_yaw = False
        randomize_roll = False
        randomize_pitch = False

    class rewards(LeggedRobotCfg.rewards):
        class scales:
            termination = -100.0
            tracking_lin_vel = 3.0
            tracking_ang_vel = 1.5
            directional_progress = 2.0
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            torques = -2.5e-5
            powers = -2.0e-5
            dof_acc = -2.5e-7
            action_rate = -0.01
            dof_pos_limits = -5.0
            feet_air_time = 0.1
            feet_contact_forces = -1.5e-4
            stand_still = -2.0
            joint_pos_penalty = -1.0
            feet_height_body = -5.0
            undesired_contacts = -1.0

        only_positive_rewards = False
        tracking_sigma = 0.25
        soft_dof_pos_limit = 0.95
        max_contact_force = 100.0
        feet_height_body_target = -0.20

    class normalization(LeggedRobotCfg.normalization):
        clip_observations = 100.0
        clip_actions = 100.0

        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05

    class noise(LeggedRobotCfg.noise):
        add_noise = True

        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            dof_pos = 0.01
            dof_vel = 1.5
            ang_vel = 0.2
            gravity = 0.05

    class sim(LeggedRobotCfg.sim):
        class physx(LeggedRobotCfg.sim.physx):
            max_gpu_contact_pairs = 44_000_000
            default_buffer_size_multiplier = 5


class Go2BlindStairLocoCfgPPO(LeggedRobotCfgPPO):
    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"

    class algorithm(LeggedRobotCfgPPO.algorithm):
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1.0e-3
        gamma = 0.99
        lam = 0.95
        enable_action_range_regularization = False
        enable_smoothness_regularization = False

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = "BlindLocomotionActorCritic"
        experiment_name = "Go2_blind_stair_loco"
        run_name = ""
        num_steps_per_env = 24
        max_iterations = 3000
        save_interval = 100
