"""Gate and, only if needed, fine-tune minimal bidirectional stair navigation."""

from __future__ import annotations

import argparse
import json
import os
import sys

import isaacgym
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.envs.base.legged_robot_pos_stairs_minimal import (
    TERMINAL_REASON_NAMES,
    TERMINAL_SUCCESS,
)
from legged_gym.utils import get_args, task_registry


BASELINE_EXPERIMENT = "Go2_pos_depth_stairs"
BASELINE_RUN = "07_14_06-25-22_"
BASELINE_CHECKPOINT = 200
BASELINE_INIT_CHECKPOINT = os.path.join(
    LEGGED_GYM_ROOT_DIR,
    "logs",
    BASELINE_EXPERIMENT,
    BASELINE_RUN,
    f"model_{BASELINE_CHECKPOINT}.pt",
)


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gate_interval", type=int, default=50)
    parser.add_argument("--gate_episodes_per_direction", type=int, default=20)
    parser.add_argument("--gate_success_threshold", type=float, default=0.75)
    parser.add_argument(
        "--skip_initial_gate",
        action="store_true",
        help="Train one gate interval before running the first bidirectional gate.",
    )
    parser.add_argument(
        "--no_gate",
        action="store_true",
        help="Train and save the requested iterations without running a gate.",
    )
    parser.add_argument(
        "--teacher_pretrain_steps",
        type=int,
        default=0,
        help="Number of privileged-teacher environment steps before PPO.",
    )
    parser.add_argument("--teacher_update_interval", type=int, default=4)
    parser.add_argument(
        "--teacher_actor_rollout_fraction",
        type=float,
        default=0.0,
        help=(
            "Final actor share in teacher pretraining rollouts. Values above zero "
            "perform an online DAgger-style transition from teacher to actor states."
        ),
    )
    known, remaining = parser.parse_known_args()
    if known.gate_interval <= 0:
        raise ValueError("--gate_interval must be positive")
    if known.teacher_pretrain_steps < 0 or known.teacher_update_interval <= 0:
        raise ValueError("Teacher pretraining arguments must be non-negative/positive")
    if not 0.0 <= known.teacher_actor_rollout_fraction <= 1.0:
        raise ValueError("--teacher_actor_rollout_fraction must be in [0, 1]")
    sys.argv = [sys.argv[0]] + remaining
    return known, get_args()


def evaluate_policy(env, policy, episodes_per_direction):
    results = {}
    original_direction = int(env.cfg.depth_stairs.fixed_direction)
    original_perception_ids = getattr(env, "perception_env_ids", None)
    gate_env_count = min(32, env.num_envs)
    gate_env_ids = torch.arange(gate_env_count, device=env.device)
    env.perception_env_ids = gate_env_ids
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
                for env_id, direction, reason, crossed, low_count, fully_cleared in zip(
                    outcomes["env_ids"].tolist(),
                    outcomes["direction"].tolist(),
                    outcomes["terminal_reason"].tolist(),
                    outcomes["stair_crossed"].tolist(),
                    outcomes["low_obstacle_collision_count"].tolist(),
                    outcomes["fully_cleared"].tolist(),
                ):
                    if len(reasons) >= episodes_per_direction:
                        break
                    if env_id >= gate_env_count:
                        continue
                    if direction != direction_value:
                        raise RuntimeError("Gate episode direction changed unexpectedly")
                    if reason == TERMINAL_SUCCESS and (
                        not crossed or not fully_cleared or low_count != 0
                    ):
                        raise RuntimeError(
                            "Gate observed an unsafe success without full, collision-free clearance"
                        )
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
    env.perception_env_ids = original_perception_ids
    env.reset()
    return results


def pretrain_with_teacher(env, runner, steps, update_interval, actor_rollout_fraction=0.0):
    """Behavior-clone safe commands while the teacher drives the simulation."""
    if steps <= 0:
        return []
    actor_critic = runner.alg.actor_critic
    optimizer = runner.alg.optimizer
    actor_critic.train()
    actor_critic.std.data.fill_(0.25)
    obs, _ = env.reset()
    losses = []
    for step in range(steps):
        with torch.no_grad():
            target_actions = env.get_navigation_teacher_actions()
        if step % update_interval == 0:
            predicted_actions = actor_critic.forward(obs)
            safe_loss = torch.mean(torch.square(predicted_actions - target_actions))
            nominal_loss = torch.mean(
                torch.square(actor_critic.u_bar - target_actions)
            )
            loss = safe_loss + 0.10 * nominal_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().item()))
        with torch.no_grad():
            actor_actions = actor_critic.forward(obs)
            transition_steps = max(int(0.6 * steps), 1)
            actor_share = min(float(step) / transition_steps, 1.0) * float(
                actor_rollout_fraction
            )
            rollout_actions = torch.lerp(target_actions, actor_actions, actor_share)
            obs, _, _, _, _ = env.step(rollout_actions)
        if (step + 1) % 100 == 0 or step + 1 == steps:
            recent = losses[-min(25, len(losses)) :]
            print(
                f"Teacher pretrain {step + 1}/{steps}: "
                f"loss={sum(recent) / max(len(recent), 1):.6f}"
            )
    return losses


def main():
    gate_args, args = parse_args()
    if args.task != "go2_pos_stairs_minimal":
        raise ValueError("Use --task go2_pos_stairs_minimal")

    env, _ = task_registry.make_env(args.task, args=args)
    _, train_cfg = task_registry.get_cfgs(args.task)
    if args.resume:
        train_cfg.runner.resume = True
        train_cfg.runner.resume_experiment_name = (
            args.resume_experiment_name or BASELINE_EXPERIMENT
        )
        train_cfg.runner.load_run = args.load_run or BASELINE_RUN
        train_cfg.runner.checkpoint = (
            args.checkpoint if args.checkpoint is not None else BASELINE_CHECKPOINT
        )
    else:
        train_cfg.runner.resume = False
        args.init_checkpoint = args.init_checkpoint or BASELINE_INIT_CHECKPOINT
    runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    teacher_losses = pretrain_with_teacher(
        env,
        runner,
        gate_args.teacher_pretrain_steps,
        gate_args.teacher_update_interval,
        gate_args.teacher_actor_rollout_fraction,
    )
    if teacher_losses:
        teacher_path = os.path.join(runner.log_dir, "model_teacher_pretrained.pt")
        runner.save(teacher_path)
        print(f"Saved teacher-pretrained checkpoint: {teacher_path}")
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

    if gate_args.no_gate:
        budget = int(train_cfg.runner.max_iterations)
        runner.learn(
            num_learning_iterations=budget,
            init_at_random_ep_len=True,
            config={},
        )
        output_path = os.path.join(runner.log_dir, "gate_history.json")
        with open(output_path, "w", encoding="utf-8") as stream:
            json.dump(
                {
                    "success_threshold": gate_args.gate_success_threshold,
                    "episodes_per_direction": gate_args.gate_episodes_per_direction,
                    "passed": None,
                    "skipped": True,
                    "history": [],
                },
                stream,
                indent=2,
            )
        print(f"Saved gate history (skipped): {output_path}")
        return

    passed = False if gate_args.skip_initial_gate else run_gate(0)
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
