"""Pure tests for blind locomotion and minimal stair terminal semantics."""

import json
import os
import tempfile
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
from legged_gym.envs.base.legged_robot_pos_stairs_minimal import (
    exclusive_terminal_masks,
    stair_fully_cleared,
    stair_progress_is_stuck,
    stair_success_eligible,
    timeout_reached,
    update_consecutive_timer,
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
