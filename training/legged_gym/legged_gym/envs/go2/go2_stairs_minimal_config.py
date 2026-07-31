"""Minimal bidirectional stair navigation configuration."""

from legged_gym.envs.go2.go2_pos_config import (
    Go2PosDepthStairsCfg,
    Go2PosDepthStairsCfgPPO,
)


class Go2PosStairsMinimalCfg(Go2PosDepthStairsCfg):
    class loco(Go2PosDepthStairsCfg.loco):
        backend = "blind_stair"
        model_path = "{LEGGED_GYM_ROOT_DIR}/legged_gym/ctrl_model/blind_stair_loco.pt"
        metadata_path = "{LEGGED_GYM_ROOT_DIR}/legged_gym/ctrl_model/blind_stair_loco.pt.json"

    class env(Go2PosDepthStairsCfg.env):
        num_envs = 8192
        episode_length_s = 35
        goal_reached_time = 12
        stay_time = 250
        hard_contact_warmup_steps = 10
        disable_graphics = True

    class terrain(Go2PosDepthStairsCfg.terrain):
        measure_heights = False
        low_obstacle_boxes = ()

    class sensors(Go2PosDepthStairsCfg.sensors):
        class depth_cam(Go2PosDepthStairsCfg.sensors.depth_cam):
            enable = False

    class perception(Go2PosDepthStairsCfg.perception):
        mode = "constant_open"
        model_path = ""

    class depth_stairs(Go2PosDepthStairsCfg.depth_stairs):
        fixed_direction = 0
        start_x_range = [3.30, 3.70]
        start_y_range = [4.60, 5.40]
        goal_x_range = [7.00, 7.40]
        goal_y_range = [4.60, 5.40]
        # The position policy settles the Go2 base near 0.28 m although it is
        # spawned at 0.42 m. This still leaves a 0.24 m margin to the wrong deck.
        height_tolerance = 0.16
        stair_clearance_distance = 0.80
        foot_clearance_margin = 0.05
        foot_height_tolerance = 0.18
        heading_tolerance_deg = 20.0
        stair_stuck_x_range = [4.00, 7.10]
        stair_stuck_window_steps = 100
        stair_stuck_min_progress = 0.15

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
            termination = -10.0

    class sim(Go2PosDepthStairsCfg.sim):
        class physx(Go2PosDepthStairsCfg.sim.physx):
            max_gpu_contact_pairs = 44_000_000
            default_buffer_size_multiplier = 5


class Go2PosStairsMinimalCfgPPO(Go2PosDepthStairsCfgPPO):
    class policy(Go2PosDepthStairsCfgPPO.policy):
        enable_shield = False

    class runner(Go2PosDepthStairsCfgPPO.runner):
        experiment_name = "Go2_pos_stairs_minimal"
        max_iterations = 1000
        save_interval = 100
