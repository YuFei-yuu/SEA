"""Configuration for the production Depth Sensor stair task."""

from legged_gym.envs.go2.go2_pos_config import (
    Go2PosDepthStairsCfg,
    Go2PosRoughCfgPPO,
)


# Keep obstacles distributed over the full low-floor approach while leaving
# several independent lateral corridors for the depth teacher to discover.
_PASSABILITY_OBSTACLE_INDICES = (
    # Keep 15 obstacles, with three early-entry blockers and a denser middle
    # zig-zag.  This makes the bypass visible while retaining edge corridors.
    3, 5, 6, 9, 11, 13, 15, 17, 18, 21, 22, 23, 27, 30, 31,
)

_PASSABILITY_OBSTACLE_Y_REMAP = {
    3: 2.20,
    9: 6.80,
    15: 7.20,
    21: 5.60,
    27: 3.00,
}


class Go2PosDepthStairsPassabilityCfg(Go2PosDepthStairsCfg):
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

    class asset(Go2PosDepthStairsCfg.asset):
        # Low-obstacle contacts are observed and penalized by the navigation
        # task; early curriculum episodes must not reset on the first contact.
        terminate_after_contacts_on = []

    class terrain(Go2PosDepthStairsCfg.terrain):
        # Preserve each obstacle height while making the x/y footprint 0.9x.
        # A sparse, full-width subset keeps the scene useful for bypass
        # learning without creating a nearly continuous obstacle wall.
        low_obstacle_boxes = tuple(
            (x, _PASSABILITY_OBSTACLE_Y_REMAP.get(index, y), size_x * 0.90, size_y * 0.90, height)
            for index, (x, y, size_x, size_y, height) in enumerate(
                Go2PosDepthStairsCfg.terrain.low_obstacle_boxes
            )
            if index in _PASSABILITY_OBSTACLE_INDICES
        )

    class env(Go2PosDepthStairsCfg.env):
        num_envs = 8192
        episode_length_s = 35
        goal_reached_time = 12

    class depth_stairs(Go2PosDepthStairsCfg.depth_stairs):
        fixed_direction = 0
        eval_seed_base = -1
        # Start close to the upper room wall, outside the central obstacle
        # band; the random goal y remains broad so the route still needs
        # lateral bypassing after entering the room.
        start_y_range = [8.80, 9.20]
        # Keep the destination on the central platform band so wall-side
        # starts must later make a visible lateral bypass without ending in a
        # near-wall platform alignment deadlock.
        goal_y_range = [3.00, 6.50]
        # Leave enough look-ahead before the first obstacle column so bypass
        # actions can combine forward and lateral motion from the first step.
        up_start_x_range = [0.75, 0.90]
        up_goal_x_range = [7.20, 7.20]
        down_start_x_range = [8.60, 8.60]
        down_goal_x_range = [0.70, 0.70]
        # Match the physical collision inflation plus the robot footprint so
        # the receding-horizon segment stays clear before contact is reported.
        teacher_obstacle_inflation = 0.30
        teacher_lateral_margin = 0.30
        teacher_min_directed_progress = 0.50
        # Keep a small geometric margin while allowing recovery waypoints
        # when the robot starts just outside an inflated obstacle.
        teacher_min_clearance = 0.05
        teacher_speed = 0.25
        teacher_bypass_speed = 0.80
        teacher_stair_speed = 0.22
        teacher_down_stair_speed = 0.40
        # The stair run is full-width. Keep the frozen locomotion aligned with the tread until all feet clear it; lateral bypass resumes on the destination platform.
        teacher_stair_lateral_speed = 0.0
        teacher_min_forward_speed = 0.12
        teacher_bypass_min_forward_speed = 0.12
        teacher_backtrack_speed = -0.35
        teacher_clear_min_forward_speed = 0.24
        teacher_stair_min_forward_speed = 0.18
        teacher_down_stair_min_forward_speed = 0.32
        # After full stair clearance, lateral goal alignment may span several metres; use the existing safe bypass speed so it finishes within the episode horizon.
        teacher_down_platform_speed = 0.80
        teacher_down_stair_lane_y = 8.20
        teacher_down_high_platform_forward_speed = 0.35
        teacher_max_lateral_speed = 0.80
        platform_start_x = 6.30
        stair_clearance_distance = 0.80
        foot_clearance_margin = 0.05
        foot_height_tolerance = 0.12
        collision_inflation = 0.05
        # Curriculum runs keep collisions as penalties so the actor can learn
        # to recover.  The evaluator enables strict terminal rules explicitly.
        strict_terminal_rules = False
        enable_stand_still_reset = False

    class perception(Go2PosDepthStairsCfg.perception):
        # 8192-env PPO pretraining defaults to oracle rays without cameras.
        # Final fine-tuning/evaluation must explicitly select depth_predicted.
        mode = "oracle"
        model_path = ""
        render_depth_in_oracle = False

    class domain_rand(Go2PosDepthStairsCfg.domain_rand):
        randomize_friction = False
        randomize_base_mass = False
        randomize_xy = False
        randomize_yaw = False
        randomize_roll = False
        randomize_pitch = False
        push_robots = False

    class rewards(Go2PosDepthStairsCfg.rewards):
        class scales(Go2PosDepthStairsCfg.rewards.scales):
            teacher_action_tracking = 4.0
            obstacle_field_progress = 2.0
            stair_approach = 1.5
            stair_crossing = 3.0
            stair_clearance = 8.0
            goal_approach = 1.5


class Go2PosDepthStairsPassabilityCfgPPO(Go2PosRoughCfgPPO):
    class policy(Go2PosRoughCfgPPO.policy):
        num_passability_classes = 4
        enable_shield = True

    class algorithm(Go2PosRoughCfgPPO.algorithm):
        learning_rate = 2.0e-4
        schedule = "fixed"
        entropy_coef = 0.001
        num_learning_epochs = 3
        passability_loss_coef = 0.25

    class runner(Go2PosRoughCfgPPO.runner):
        experiment_name = "Go2_pos_depth_stairs_passability"
        policy_class_name = "DifferentiableSafeActorCritic"
        max_iterations = 600
        save_interval = 50
