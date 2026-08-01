"""Pure tests for blind locomotion and minimal stair terminal semantics."""

import json
import os
import tempfile
from collections import deque
from types import SimpleNamespace
import unittest

import isaacgym
from isaacgym import terrain_utils
import numpy as np
import torch

from legged_gym.envs.go2.go2_blind_stair_loco_config import Go2BlindStairLocoCfg
from legged_gym.envs.go2.go2_blind_stair_loco_config import (
    Go2BlindStairForwardFinetuneCfg,
    Go2BlindStairForwardFinetuneCfgPPO,
)
from legged_gym.envs.go2.go2_pos_config import Go2PosDepthStairsCfg
from legged_gym.envs.go2.go2_stairs_minimal_config import Go2PosStairsMinimalCfg
from legged_gym.envs.base.legged_robot_pos_stairs_minimal import (
    COLLISION_CLASS_NAMES,
    deterministic_uniform_from_seed,
    exclusive_terminal_masks,
    stair_fully_cleared,
    stair_progress_is_stuck,
    stair_success_eligible,
    timeout_reached,
    update_consecutive_timer,
)
from legged_gym.utils.footprint_rays import ray_aabb_distances, room_footprint_boxes
from legged_gym.utils.local_room_teacher import choose_local_waypoint, segment_clearance
from legged_gym.utils.fixed_room_planner import (
    build_bidirectional_route_templates,
    build_occupancy_grid,
    segment_is_free,
)
from legged_gym.low_level import (
    BlindStairPolicy,
    GO2_EXTERNAL_JOINT_ORDER,
    build_blind_stair_metadata,
    build_blind_stair_observation,
)
from legged_gym.scripts.gate_blind_stair_loco import gate_required_successes
from legged_gym.utils.terrain import Terrain
from rsl_rl.algorithms import PPO
from rsl_rl.modules import BlindLocomotionActorCritic, DifferentiableSafeActorCritic
from rsl_rl.runners.on_policy_runner import OnPolicyRunner


DEFAULT_ANGLES = (-0.1, 0.8, -1.5, 0.1, 0.8, -1.5, -0.1, 1.0, -1.5, 0.1, 1.0, -1.5)
EXPECTED = {
    "observation_scales": {"lin_vel": 2.0, "ang_vel": 0.25, "dof_pos": 1.0, "dof_vel": 0.05},
    "joint_order": GO2_EXTERNAL_JOINT_ORDER,
    "default_joint_angles": DEFAULT_ANGLES,
    "control": {"frequency_hz": 50.0, "action_scale": 0.25, "stiffness": 30.0, "damping": 0.75},
}


class TestMinimalStairs(unittest.TestCase):
    def test_local_teacher_keeps_clear_goal_direct(self):
        starts = torch.tensor([[0.0, 1.0]])
        goals = torch.tensor([[5.0, 1.0]])
        centers = torch.tensor([[2.0, 5.0]])
        extents = torch.tensor([[0.25, 0.25]])
        waypoint, direct, _, label = choose_local_waypoint(starts, goals, centers, extents)
        self.assertTrue(direct.item())
        self.assertEqual(label.item(), 0)
        self.assertTrue(torch.allclose(waypoint, goals))

    def test_local_teacher_uses_forward_lateral_corner_when_blocked(self):
        starts = torch.tensor([[0.0, 5.0]])
        goals = torch.tensor([[5.0, 5.0]])
        centers = torch.tensor([[2.0, 5.0]])
        extents = torch.tensor([[0.25, 0.25]])
        waypoint, direct, _, label = choose_local_waypoint(starts, goals, centers, extents)
        self.assertFalse(direct.item())
        self.assertEqual(label.item(), 1)
        self.assertGreater(waypoint[0, 0].item(), 0.0)
        self.assertGreater(abs(waypoint[0, 1].item() - 5.0), 0.25)
        clear, _ = segment_clearance(starts, waypoint, centers, extents)
        self.assertTrue(clear.item())
    def test_seeded_eval_sampling_is_deterministic_and_bounded(self):
        seeds = torch.tensor([1000, 1001, 2000])
        first = deterministic_uniform_from_seed(seeds, 7, 0.75, 1.10)
        second = deterministic_uniform_from_seed(seeds, 7, 0.75, 1.10)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.all((first >= 0.75) & (first <= 1.10)))
        self.assertEqual(torch.unique(first).numel(), len(seeds))

    def test_footprint_ray_hits_aabb_and_rejects_parallel_miss(self):
        origins = torch.tensor([[0.0, 0.0]])
        directions = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]])
        centers = torch.tensor([[2.0, 0.0]])
        half_extents = torch.tensor([[0.5, 0.5]])
        distances = ray_aabb_distances(
            origins, directions, centers, half_extents, min_dist=0.1, max_dist=5.0
        )
        self.assertTrue(torch.allclose(distances[0, :, 0], torch.tensor([1.5, 5.0, 5.0])))
        self.assertTrue(torch.isfinite(distances).all())

    def test_room_footprint_contains_walls_and_20_full_size_obstacles(self):
        cfg = Go2PosStairsMinimalCfg().terrain
        centers, half_extents = room_footprint_boxes(cfg.low_obstacle_boxes)
        self.assertEqual(tuple(centers.shape), (24, 2))
        self.assertEqual(tuple(half_extents.shape), (24, 2))
        self.assertTrue(torch.all(half_extents >= 0.15))
        self.assertEqual(len(cfg.low_obstacle_boxes), 20)
        original_boxes = set(Go2PosDepthStairsCfg().terrain.low_obstacle_boxes)
        self.assertTrue(all(box in original_boxes for box in cfg.low_obstacle_boxes))
        self.assertEqual(len(set(cfg.low_obstacle_boxes)), 20)
        self.assertTrue(torch.all(centers[4:, 0] < cfg.stair_start_x))

        forward_clearance = ray_aabb_distances(
            torch.tensor([[4.70, 5.0]]),
            torch.tensor([[[1.0, 0.0]]]),
            centers,
            half_extents,
            min_dist=0.1,
            max_dist=5.0,
        ).min(dim=-1).values.item()
        self.assertGreater(forward_clearance, 4.5)

        resolution = 0.05
        cells = int(10.0 / resolution)
        occupied = np.zeros((cells, cells), dtype=bool)
        for center, extent in zip(centers.numpy(), half_extents.numpy()):
            lower = np.floor((center - extent) / resolution).astype(int).clip(0, cells)
            upper = np.ceil((center + extent) / resolution).astype(int).clip(0, cells)
            occupied[lower[0] : upper[0], lower[1] : upper[1]] = True
        start = (int(0.95 / resolution), int(5.0 / resolution))
        target_x = int(4.60 / resolution)
        queue = deque([start])
        visited = {start}
        reached = False
        while queue:
            x, y = queue.popleft()
            if x >= target_x:
                reached = True
                break
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbor = (x + dx, y + dy)
                if (
                    0 <= neighbor[0] < cells
                    and 0 <= neighbor[1] < cells
                    and not occupied[neighbor]
                    and neighbor not in visited
                ):
                    visited.add(neighbor)
                    queue.append(neighbor)
        self.assertTrue(reached)

    def test_fixed_map_routes_are_clear_and_diagonal(self):
        cfg = Go2PosStairsMinimalCfg()
        templates, lengths, _, _ = build_bidirectional_route_templates(
            cfg.terrain.low_obstacle_boxes,
            start_y_range=cfg.depth_stairs.start_y_range,
            route_bins=cfg.depth_stairs.route_bins,
            low_start_x=cfg.depth_stairs.up_start_x_range[0],
            high_start_x=cfg.depth_stairs.down_start_x_range[0],
            up_goal=(cfg.depth_stairs.up_goal_x_range[0], cfg.depth_stairs.goal_y_range[0]),
            down_goal=(cfg.depth_stairs.down_goal_x_range[0], cfg.depth_stairs.goal_y_range[0]),
            low_staging=tuple(cfg.depth_stairs.route_low_staging),
            high_staging=tuple(cfg.depth_stairs.route_high_staging),
            obstacle_inflation=cfg.depth_stairs.route_obstacle_inflation,
        )
        occupied = build_occupancy_grid(
            cfg.terrain.low_obstacle_boxes,
            obstacle_inflation=cfg.depth_stairs.footprint_inflation,
        )
        self.assertEqual(templates.shape[:2], (2, cfg.depth_stairs.route_bins))
        self.assertTrue(np.all(lengths >= 4))
        for direction in range(2):
            for route_index in range(cfg.depth_stairs.route_bins):
                route = templates[direction, route_index, : lengths[direction, route_index]]
                for start, goal in zip(route[:-1], route[1:]):
                    # Only the low-floor route is rasterized with obstacles;
                    # the high platform and stair span are intentionally open.
                    if max(start[0], goal[0]) <= cfg.depth_stairs.route_low_staging[0] + 1e-5:
                        rounded_start = tuple(round(float(value), 4) for value in start)
                        rounded_goal = tuple(round(float(value), 4) for value in goal)
                        self.assertTrue(
                            segment_is_free(occupied, rounded_start, rounded_goal)
                        )
        stair_delta = np.asarray(cfg.depth_stairs.route_high_staging) - np.asarray(
            cfg.depth_stairs.route_low_staging
        )
        stair_angle = abs(np.degrees(np.arctan2(stair_delta[1], stair_delta[0])))
        self.assertGreater(stair_angle, 5.0)
        self.assertLess(stair_angle, cfg.depth_stairs.heading_tolerance_deg)

    def test_down_start_has_room_for_diagonal_platform_approach(self):
        cfg = Go2PosStairsMinimalCfg().depth_stairs
        self.assertEqual(cfg.up_goal_x_range, [7.20, 7.20])
        self.assertEqual(cfg.down_start_x_range, [8.60, 8.60])
        self.assertGreater(cfg.down_start_x_range[0] - cfg.route_high_staging[0], 1.0)

    def test_collision_class_contract(self):
        self.assertEqual(COLLISION_CLASS_NAMES[0], "none")
        self.assertEqual(COLLISION_CLASS_NAMES[1], "low_obstacle")
        self.assertEqual(COLLISION_CLASS_NAMES[2], "wall")
        self.assertEqual(COLLISION_CLASS_NAMES[3], "stair")

    def test_forward_finetune_contract(self):
        env_cfg = Go2BlindStairForwardFinetuneCfg()
        train_cfg = Go2BlindStairForwardFinetuneCfgPPO()
        self.assertEqual(env_cfg.env.num_observations, 45)
        self.assertEqual(env_cfg.env.num_actions, 12)
        self.assertEqual(env_cfg.terrain.terrain_proportions, [0.1, 0.2, 0.7])
        self.assertEqual(train_cfg.algorithm.learning_rate, 1.0e-4)
        self.assertEqual(train_cfg.algorithm.schedule, "fixed")
        self.assertEqual(gate_required_successes(20), 19)

    def test_weight_only_load_resets_iteration_and_keeps_fresh_optimizer(self):
        source = torch.nn.Linear(3, 2)
        source_optimizer = torch.optim.Adam(source.parameters(), lr=1.0e-3)
        source(torch.ones(1, 3)).sum().backward()
        source_optimizer.step()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = os.path.join(directory, "model_2800.pt")
            torch.save(
                {
                    "model_state_dict": source.state_dict(),
                    "optimizer_state_dict": source_optimizer.state_dict(),
                    "iter": 2800,
                    "infos": None,
                },
                checkpoint_path,
            )
            target = torch.nn.Linear(3, 2)
            fresh_optimizer = torch.optim.Adam(target.parameters(), lr=1.0e-4)
            runner = OnPolicyRunner.__new__(OnPolicyRunner)
            runner.device = "cpu"
            runner.alg = SimpleNamespace(
                actor_critic=target,
                optimizer=fresh_optimizer,
            )
            runner.current_learning_iteration = 99
            runner.load(
                checkpoint_path,
                load_optimizer=False,
                reset_iteration=True,
            )
            self.assertEqual(runner.current_learning_iteration, 0)
            self.assertEqual(fresh_optimizer.param_groups[0]["lr"], 1.0e-4)
            self.assertEqual(len(fresh_optimizer.state), 0)
            for source_value, target_value in zip(source.parameters(), target.parameters()):
                self.assertTrue(torch.equal(source_value, target_value))

    def test_blind_actor_shape(self):
        policy = BlindLocomotionActorCritic(num_actions=12, num_actor_obs=45)
        self.assertEqual(tuple(policy.act_inference(torch.zeros(4, 45)).shape), (4, 12))

    def test_target_difficulty_uses_exact_five_step_geometry(self):
        cfg = Go2BlindStairLocoCfg().terrain
        terrain = terrain_utils.SubTerrain(
            "test",
            width=int(cfg.terrain_width / cfg.horizontal_scale),
            length=int(cfg.terrain_length / cfg.horizontal_scale),
            vertical_scale=cfg.vertical_scale,
            horizontal_scale=cfg.horizontal_scale,
        )
        builder = Terrain.__new__(Terrain)
        builder.cfg = cfg
        builder.blind_stair_down_terrain_func(terrain, difficulty=0.6)
        heights = np.unique(terrain.height_field_raw) * cfg.vertical_scale
        self.assertTrue(
            np.allclose(heights, np.asarray([0.0, 0.08, 0.16, 0.24, 0.32, 0.40]))
        )
        centerline = terrain.height_field_raw[:, terrain.width // 2]
        transitions = np.flatnonzero(np.diff(centerline) != 0)
        self.assertEqual(len(transitions), 10)
        self.assertTrue(np.all(np.diff(transitions[:5]) == 3))

    def test_blind_observation_layout_and_scaling(self):
        terms = [torch.full((2, width), float(index + 1)) for index, width in enumerate((3, 3, 3, 12, 12, 12))]
        observation = build_blind_stair_observation(
            *terms,
            lin_vel_scale=2.0,
            ang_vel_scale=0.25,
            dof_pos_scale=1.0,
            dof_vel_scale=0.05,
        )
        self.assertEqual(tuple(observation.shape), (2, 45))
        self.assertTrue(torch.allclose(observation[:, 0:3], torch.full((2, 3), 0.25)))
        self.assertTrue(torch.allclose(observation[:, 3:6], torch.full((2, 3), 2.0)))
        self.assertTrue(
            torch.allclose(
                observation[:, 6:9],
                torch.tensor([[6.0, 6.0, 0.75], [6.0, 6.0, 0.75]]),
            )
        )
        self.assertTrue(torch.allclose(observation[:, 21:33], torch.full((2, 12), 0.25)))
        self.assertTrue(torch.allclose(observation[:, 33:45], torch.full((2, 12), 6.0)))

    def test_low_level_ppo_update_accepts_twelve_actions(self):
        policy = BlindLocomotionActorCritic(num_actions=12, num_actor_obs=45)
        algorithm = PPO(
            policy,
            num_learning_epochs=1,
            num_mini_batches=1,
            enable_action_range_regularization=False,
            enable_smoothness_regularization=False,
        )
        algorithm.init_storage(4, 4, [45], [12])
        obs = torch.randn(4, 45)
        for _ in range(4):
            actions = algorithm.act(obs, obs)
            next_obs = torch.randn(4, 45)
            algorithm.process_env_step(
                next_obs,
                torch.randn(4),
                torch.zeros(4, dtype=torch.bool),
                {},
            )
            obs = next_obs
        algorithm.compute_returns(obs)
        losses = algorithm.update()
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in losses))

    def test_minimal_navigation_disables_cbf_intervention(self):
        policy = DifferentiableSafeActorCritic(
            num_actions=3,
            num_props=12,
            num_rays=21,
            num_goal_obs=2,
            num_dynamic_obs=0,
            his_len=10,
            ray_fov_deg=100.0,
            enable_shield=False,
        )
        output = policy.act_inference(torch.randn(4, 350))
        self.assertTrue(torch.equal(output, policy.u_bar))
        self.assertTrue(torch.equal(policy.u_s, policy.u_bar))

    def test_passability_head_is_shared_with_navigation_features(self):
        policy = DifferentiableSafeActorCritic(
            num_actions=3,
            num_props=12,
            num_rays=21,
            num_goal_obs=2,
            num_dynamic_obs=0,
            his_len=10,
            ray_fov_deg=100.0,
            enable_shield=False,
            num_passability_classes=4,
        )
        output = policy.act_inference(torch.randn(5, 350))
        self.assertEqual(tuple(output.shape), (5, 3))
        self.assertEqual(tuple(policy.passability_logits.shape), (5, 4))

    def test_ppo_updates_passability_auxiliary_loss(self):
        policy = DifferentiableSafeActorCritic(
            num_actions=3,
            num_props=12,
            num_rays=21,
            num_goal_obs=2,
            num_dynamic_obs=0,
            his_len=10,
            enable_shield=False,
            num_passability_classes=4,
        )
        algorithm = PPO(
            policy,
            num_learning_epochs=1,
            num_mini_batches=1,
            enable_action_range_regularization=False,
            enable_smoothness_regularization=False,
            passability_loss_coef=0.25,
        )
        algorithm.init_storage(4, 4, [350], [3])
        obs = torch.randn(4, 350)
        labels = torch.tensor([0, 1, 2, 3])
        for _ in range(4):
            actions = algorithm.act(obs, obs, passability_targets=labels)
            next_obs = torch.randn(4, 350)
            algorithm.process_env_step(
                next_obs,
                torch.randn(4),
                torch.zeros(4, dtype=torch.bool),
                {},
            )
            obs = next_obs
        algorithm.compute_returns(obs)
        algorithm.update()
        self.assertTrue(np.isfinite(algorithm.last_passability_loss))
        self.assertGreater(algorithm.last_passability_loss, 0.0)

    def test_goal_timer_is_consecutive(self):
        timer = torch.tensor([5, 5])
        timer = update_consecutive_timer(timer, torch.tensor([True, False]))
        self.assertEqual(timer.tolist(), [6, 0])

        timer = update_consecutive_timer(timer, torch.tensor([False, True]))
        self.assertEqual(timer.tolist(), [0, 1])

    def test_wrong_height_cannot_succeed(self):
        eligible = stair_success_eligible(
            torch.tensor([0.1, 0.1]),
            torch.tensor([False, True]),
            torch.tensor([True, False]),
            0.5,
        )
        self.assertEqual(eligible.tolist(), [False, False])

    def test_timeout_beats_success(self):
        masks = exclusive_terminal_masks(
            torch.tensor([True]),
            torch.tensor([False]),
            torch.tensor([False]),
            torch.tensor([True]),
            torch.tensor([False]),
        )
        success, _, _, timeout, _ = masks
        self.assertFalse(success.item())
        self.assertTrue(timeout.item())

    def test_timeout_triggers_exactly_at_limit(self):
        reached = timeout_reached(torch.tensor([99, 100, 101]), 100)
        self.assertEqual(reached.tolist(), [False, True, True])

    def test_stair_stuck_requires_full_window_and_insufficient_progress(self):
        stuck = stair_progress_is_stuck(
            torch.tensor([True, True, True, False]),
            torch.tensor([100, 99, 100, 100]),
            torch.tensor([0.14, 0.01, 0.15, 0.01]),
            100,
            0.15,
        )
        self.assertEqual(stuck.tolist(), [True, False, False, False])

    def test_full_clearance_requires_base_and_all_feet_on_destination_deck(self):
        cleared = stair_fully_cleared(
            torch.tensor([7.10, 7.10, 4.00]),
            torch.tensor(
                [
                    [6.36, 6.36, 6.36, 6.20],
                    [6.36, 6.36, 6.36, 6.36],
                    [4.70, 4.70, 4.70, 4.70],
                ]
            ),
            torch.tensor([True, True, False]),
            stair_start_x=4.80,
            stair_end_x=6.30,
            base_clearance=0.80,
            foot_margin=0.05,
        )
        self.assertEqual(cleared.tolist(), [False, True, True])

    def test_full_clearance_rejects_base_near_stair_edge(self):
        cleared = stair_fully_cleared(
            torch.tensor([6.90, 4.20]),
            torch.tensor([[6.40] * 4, [4.70] * 4]),
            torch.tensor([True, False]),
            stair_start_x=4.80,
            stair_end_x=6.30,
            base_clearance=0.80,
            foot_margin=0.05,
        )
        self.assertEqual(cleared.tolist(), [False, False])

    def test_terminal_masks_are_exclusive(self):
        candidates = torch.ones(3, dtype=torch.bool)
        masks = exclusive_terminal_masks(candidates, candidates, candidates, candidates, candidates)
        self.assertTrue(torch.all(torch.stack(masks).sum(dim=0) == 1))

    def test_metadata_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = os.path.join(directory, "checkpoint.pt")
            model_path = os.path.join(directory, "policy.pt")
            actor = BlindLocomotionActorCritic(num_actions=12, num_actor_obs=45).actor.eval()
            torch.save({"model_state_dict": {}}, checkpoint_path)
            torch.jit.script(actor).save(model_path)
            metadata = build_blind_stair_metadata(model_path, checkpoint_path, DEFAULT_ANGLES)
            metadata["joint_order"][0] = "wrong_joint"
            metadata_path = model_path + ".json"
            with open(metadata_path, "w", encoding="utf-8") as stream:
                json.dump(metadata, stream)
            with self.assertRaisesRegex(ValueError, "joint order"):
                BlindStairPolicy(model_path, metadata_path, "cpu", EXPECTED)


if __name__ == "__main__":
    unittest.main()
