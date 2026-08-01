"""Teacher BC/DAgger pretraining followed by optional short PPO training."""

from __future__ import annotations

import argparse
import json
import os
import sys

from isaacgym import gymapi
import torch

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


def _parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--teacher_steps", type=int, default=3000)
    parser.add_argument("--teacher_update_interval", type=int, default=2)
    parser.add_argument("--actor_rollout_fraction", type=float, default=1.0)
    parser.add_argument("--ppo_iterations", type=int, default=100)
    parser.add_argument("--no_ppo", action="store_true")
    parser.add_argument(
        "--curriculum_stage",
        choices=list("ABCDEFG"),
        default="F",
        help="A flat obstacles, B up stairs, C down stairs, D/E mixed, F oracle, G depth_predicted.",
    )
    parser.add_argument("--strict_terminal_rules", action="store_true")
    parser.add_argument("--enable_stand_still_reset", action="store_true")
    parser.add_argument("--output_summary", required=True)
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return known, get_args()


def _apply_curriculum_stage(env_cfg, stage):
    """Apply a bounded task curriculum before Isaac Gym creates the terrain."""
    stage = str(stage).upper()
    depth = env_cfg.depth_stairs
    terrain = env_cfg.terrain
    original_boxes = tuple(terrain.low_obstacle_boxes)
    if stage == "A":
        depth.fixed_direction = 1
        depth.up_goal_x_range = [4.20, 4.20]
        terrain.low_obstacle_boxes = original_boxes[:6]
    elif stage == "B":
        depth.fixed_direction = 1
        terrain.low_obstacle_boxes = ()
    elif stage == "C":
        depth.fixed_direction = -1
        terrain.low_obstacle_boxes = ()
    elif stage == "D":
        depth.fixed_direction = 1
        terrain.low_obstacle_boxes = original_boxes
    elif stage == "E":
        depth.fixed_direction = -1
        terrain.low_obstacle_boxes = original_boxes
    elif stage == "F":
        depth.fixed_direction = 0
        terrain.low_obstacle_boxes = original_boxes
        env_cfg.perception.mode = "oracle"
    elif stage == "G":
        depth.fixed_direction = 0
        terrain.low_obstacle_boxes = original_boxes
        env_cfg.perception.mode = "depth_predicted"
    else:
        raise ValueError(f"Unknown curriculum stage {stage!r}")
    depth.curriculum_stage = stage


def main():
    train_args, args = _parse_args()
    if args.task != "go2_pos_depth_stairs_passability":
        raise ValueError("Use --task go2_pos_depth_stairs_passability.")
    env_cfg, _ = task_registry.get_cfgs(args.task)
    _apply_curriculum_stage(env_cfg, train_args.curriculum_stage)
    env_cfg.depth_stairs.strict_terminal_rules = train_args.strict_terminal_rules
    env_cfg.depth_stairs.enable_stand_still_reset = train_args.enable_stand_still_reset
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=env_cfg)
    _, train_cfg = task_registry.get_cfgs(args.task)
    runner, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    actor_critic = runner.alg.actor_critic
    optimizer = runner.alg.optimizer
    actor_critic.train()
    actor_critic.std.data.fill_(0.25)
    obs, _ = env.reset()
    losses = []
    for step in range(train_args.teacher_steps):
        with torch.no_grad():
            target = env.get_navigation_teacher_actions()
            labels = env.get_passability_targets()
            actor_action = actor_critic.forward(obs)
            rollout_fraction = min(1.0, step / max(0.6 * train_args.teacher_steps, 1)) * train_args.actor_rollout_fraction
            rollout_action = torch.lerp(target, actor_action, rollout_fraction)
        if step % max(train_args.teacher_update_interval, 1) == 0:
            prediction = actor_critic.forward(obs)
            action_loss = torch.mean((prediction - target) ** 2)
            logits = actor_critic.passability_logits
            passability_loss = torch.nn.functional.cross_entropy(logits, labels)
            loss = action_loss + 0.25 * passability_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        obs, _, _, _, _ = env.step(rollout_action)
    teacher_path = os.path.join(runner.log_dir, "model_teacher_pretrained.pt")
    runner.save(teacher_path)
    if not train_args.no_ppo and train_args.ppo_iterations > 0:
        runner.learn(train_args.ppo_iterations, init_at_random_ep_len=True, config={})
    os.makedirs(os.path.dirname(os.path.abspath(train_args.output_summary)), exist_ok=True)
    with open(train_args.output_summary, "w", encoding="utf-8") as stream:
        json.dump(
            {
                "task": args.task,
                "teacher_steps": train_args.teacher_steps,
                "ppo_iterations": 0 if train_args.no_ppo else train_args.ppo_iterations,
                "teacher_loss_mean": sum(losses) / max(len(losses), 1),
                "teacher_checkpoint": teacher_path,
                "curriculum_stage": train_args.curriculum_stage,
                "strict_terminal_rules": train_args.strict_terminal_rules,
                "enable_stand_still_reset": train_args.enable_stand_still_reset,
            },
            stream,
            indent=2,
        )
    print(teacher_path)


if __name__ == "__main__":
    main()
