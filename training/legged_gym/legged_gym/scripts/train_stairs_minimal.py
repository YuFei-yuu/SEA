"""Gate and, only if needed, fine-tune minimal bidirectional stair navigation."""

from __future__ import annotations

import argparse
import json
import os
import sys

import isaacgym
import torch

from legged_gym.envs import *
from legged_gym.envs.base.legged_robot_pos_stairs_minimal import (
    TERMINAL_REASON_NAMES,
    TERMINAL_SUCCESS,
)
from legged_gym.utils import get_args, task_registry


BASELINE_EXPERIMENT = "Go2_pos_depth_stairs"
BASELINE_RUN = "07_14_06-25-22_"
BASELINE_CHECKPOINT = 200


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gate_interval", type=int, default=100)
    parser.add_argument("--gate_episodes_per_direction", type=int, default=50)
    parser.add_argument("--gate_success_threshold", type=float, default=0.60)
    known, remaining = parser.parse_known_args()
    if known.gate_interval <= 0:
        raise ValueError("--gate_interval must be positive")
    sys.argv = [sys.argv[0]] + remaining
    return known, get_args()


def evaluate_policy(env, policy, episodes_per_direction):
    results = {}
    original_direction = int(env.cfg.depth_stairs.fixed_direction)
    policy_owner = getattr(policy, "__self__", None)
    if policy_owner is not None:
        policy_owner.eval()
    with torch.no_grad():
        for name, direction_value in (("up", 1), ("down", -1)):
            env.cfg.depth_stairs.fixed_direction = direction_value
            obs, _ = env.reset()
            reasons = []
            while len(reasons) < episodes_per_direction:
                obs, _, _, _, infos = env.step(policy(obs))
                outcomes = infos.get("episode_outcomes")
                if outcomes is None:
                    continue
                for direction, reason, crossed in zip(
                    outcomes["direction"].tolist(),
                    outcomes["terminal_reason"].tolist(),
                    outcomes["stair_crossed"].tolist(),
                ):
                    if len(reasons) >= episodes_per_direction:
                        break
                    if direction != direction_value:
                        raise RuntimeError("Gate episode direction changed unexpectedly")
                    if reason == TERMINAL_SUCCESS and not crossed:
                        raise RuntimeError("Gate observed success without a stair crossing")
                    reasons.append(int(reason))
            success_rate = sum(reason == TERMINAL_SUCCESS for reason in reasons) / len(reasons)
            results[name] = {
                "success_rate": success_rate,
                "terminal_reasons": {
                    reason_name: sum(
                        reason == reason_code for reason in reasons
                    )
                    for reason_code, reason_name in TERMINAL_REASON_NAMES.items()
                    if reason_code != 0
                },
            }
    env.cfg.depth_stairs.fixed_direction = original_direction
    env.reset()
    return results


def main():
    gate_args, args = parse_args()
    if args.task != "go2_pos_stairs_minimal":
        raise ValueError("Use --task go2_pos_stairs_minimal")

    env, _ = task_registry.make_env(args.task, args=args)
    _, train_cfg = task_registry.get_cfgs(args.task)
    train_cfg.runner.resume = True
    train_cfg.runner.resume_experiment_name = (
        args.resume_experiment_name or BASELINE_EXPERIMENT
    )
    train_cfg.runner.load_run = args.load_run or BASELINE_RUN
    train_cfg.runner.checkpoint = (
        args.checkpoint if args.checkpoint is not None else BASELINE_CHECKPOINT
    )
    runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = runner.get_inference_policy(device=env.device)
    history = []

    def run_gate(additional_iterations):
        directions = evaluate_policy(
            env, policy, gate_args.gate_episodes_per_direction
        )
        passed = all(
            result["success_rate"] >= gate_args.gate_success_threshold
            for result in directions.values()
        )
        record = {
            "checkpoint_iteration": int(runner.current_learning_iteration),
            "additional_iterations": int(additional_iterations),
            "passes": passed,
            "directions": directions,
        }
        history.append(record)
        print(json.dumps(record, indent=2))
        return passed

    passed = run_gate(0)
    trained = 0
    budget = int(train_cfg.runner.max_iterations)
    while not passed and trained < budget:
        chunk = min(gate_args.gate_interval, budget - trained)
        runner.learn(
            num_learning_iterations=chunk,
            init_at_random_ep_len=(trained == 0),
            config={},
        )
        trained += chunk
        passed = run_gate(trained)

    output_path = os.path.join(runner.log_dir, "gate_history.json")
    with open(output_path, "w", encoding="utf-8") as stream:
        json.dump(
            {
                "success_threshold": gate_args.gate_success_threshold,
                "episodes_per_direction": gate_args.gate_episodes_per_direction,
                "passed": passed,
                "history": history,
            },
            stream,
            indent=2,
        )
    print(f"Saved gate history: {output_path}")
    if not passed:
        raise RuntimeError(f"Bidirectional gate did not pass within {budget} iterations")


if __name__ == "__main__":
    main()
