"""Minimal bidirectional stair navigation configuration."""

from legged_gym.envs.go2.go2_pos_config import (
    Go2PosDepthStairsCfg,
    Go2PosDepthStairsCfgPPO,
)


class Go2PosStairsMinimalCfg(Go2PosDepthStairsCfg):
    class loco(Go2PosDepthStairsCfg.loco):
        backend = "blind_stair"
        model_path = (
            "{LEGGED_GYM_ROOT_DIR}/logs/Go2_blind_stair_loco_forward_finetune/branches/"
            "from_2800_forward_lr1e4_0731a/exports/blind_stair_loco_iter_0250.pt"
        )
        metadata_path = (
            "{LEGGED_GYM_ROOT_DIR}/logs/Go2_blind_stair_loco_forward_finetune/branches/"
            "from_2800_forward_lr1e4_0731a/exports/blind_stair_loco_iter_0250.pt.json"
        )

    class env(Go2PosDepthStairsCfg.env):
        num_envs = 8192
        episode_length_s = 35
        goal_reached_time = 12
        stay_time = 250
        hard_contact_warmup_steps = 10
        disable_graphics = True

    class terrain(Go2PosDepthStairsCfg.terrain):
        measure_heights = False
        # Keep the original box dimensions, but use a sparse 20-box subset so
        # the Go2 has real, local avoidance choices instead of a blocked wall.
        low_obstacle_scale = 1.0
        low_obstacle_boxes = tuple(
            box
            for index, box in enumerate(Go2PosDepthStairsCfg.terrain.low_obstacle_boxes)
            if index in (
                1,
                3, 4, 5, 6, 7,
                9, 11, 13, 14,
                16, 18, 20,
                21, 23, 25, 26,
                27, 29, 30,
            )
        )

    class sensors(Go2PosDepthStairsCfg.sensors):
        class depth_cam(Go2PosDepthStairsCfg.sensors.depth_cam):
            enable = False

    class perception(Go2PosDepthStairsCfg.perception):
        mode = "constant_open"
        model_path = ""

    class depth_stairs(Go2PosDepthStairsCfg.depth_stairs):
        fixed_direction = 0
        eval_seed_base = -1
        start_x_range = [0.70, 0.70]
        # Starts are sampled along one transverse line. Goals are fixed so
        # success measures navigation, not target randomization.
        start_y_range = [1.00, 9.00]
        goal_x_range = [7.20, 7.20]
        goal_y_range = [5.00, 5.00]
        up_start_x_range = [0.70, 0.70]
        up_goal_x_range = [7.20, 7.20]
        # Give descending episodes enough open high-platform distance to use
        # simultaneous vx/vy/yaw before aligning head-first with the stairs.
        down_start_x_range = [8.60, 8.60]
        down_goal_x_range = [0.70, 0.70]
        # The position policy settles the Go2 base near 0.28 m although it is
        # spawned at 0.42 m. This still leaves a 0.24 m margin to the wrong deck.
        height_tolerance = 0.16
        stair_clearance_distance = 0.80
        foot_clearance_margin = 0.05
        foot_height_tolerance = 0.18
        footprint_inflation = 0.15
        # Evaluation guardrails: successful navigation must traverse the
        # obstacle field, rather than leaving the room through an edge lane.
        obstacle_field_x_range = [1.15, 4.50]
        obstacle_bypass_y_margin = 0.55
        heading_tolerance_deg = 20.0
        stair_stuck_x_range = [4.00, 7.10]
        stair_stuck_window_steps = 100
        stair_stuck_min_progress = 0.15
        teacher_speed = 0.40
        teacher_obstacle_speed = 0.30
        teacher_diagonal_speed = 0.30
        teacher_stair_speed = 0.40
        teacher_lateral_gain = 3.00
        teacher_lateral_damping = 1.00
        teacher_max_lateral_speed = 0.40
        teacher_heading_gain = 1.50
        teacher_max_yaw_rate = 0.60
        teacher_alignment_speed = 0.00
        teacher_alignment_tolerance = 0.08
        route_bins = 33
        route_grid_resolution = 0.10
        route_obstacle_inflation = 0.42
        route_boundary_margin = 0.40
        # A 0.30 m lateral offset across the 2.70 m stair span yields a mild
        # 6.3 degree diagonal while keeping the robot head-first.
        route_low_staging = [4.40, 4.55]
        route_high_staging = [7.10, 4.85]
        route_waypoint_tolerance = 0.18
        teacher_heading_path_blend = 0.45
        teacher_stair_heading_path_blend = 0.80
        teacher_stair_max_lateral_speed = 0.12

    class domain_rand(Go2PosDepthStairsCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.8, 1.2]
        randomize_base_mass = False
        randomize_xy = False
        randomize_yaw = False
        randomize_roll = False
        randomize_pitch = False
        push_robots = False

    class rewards(Go2PosDepthStairsCfg.rewards):
        class scales(Go2PosDepthStairsCfg.rewards.scales):
            stuck = 0.0
            termination = -50.0
            teacher_action_tracking = 4.0

    class sim(Go2PosDepthStairsCfg.sim):
        class physx(Go2PosDepthStairsCfg.sim.physx):
            max_gpu_contact_pairs = 44_000_000
            default_buffer_size_multiplier = 5


class Go2PosStairsMinimalCfgPPO(Go2PosDepthStairsCfgPPO):
    class policy(Go2PosDepthStairsCfgPPO.policy):
        enable_shield = True

    class runner(Go2PosDepthStairsCfgPPO.runner):
        experiment_name = "Go2_pos_stairs_minimal"
        max_iterations = 1000
        save_interval = 100
