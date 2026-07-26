"""Run targeted timeout and success regressions in the real stair environment."""

from __future__ import annotations

import json

import isaacgym
import torch

from legged_gym.envs import *
from legged_gym.envs.base.legged_robot_pos_stairs_minimal import (
    TERMINAL_NONE,
    TERMINAL_REASON_NAMES,
    TERMINAL_STAIR_STUCK,
    TERMINAL_SUCCESS,
    TERMINAL_TIMEOUT,
)
from legged_gym.utils import get_args, task_registry


def main():
    args = get_args()
    if args.task != "go2_pos_stairs_minimal":
        raise ValueError("Use --task go2_pos_stairs_minimal")
    args.headless = True
    env_cfg, _ = task_registry.get_cfgs(args.task)
    env_cfg.env.num_envs = 5
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=env_cfg)
    env.do_reset = False
    env.reset()

    def clear_terminal_state():
        env.contact_forces.zero_()
        env.projected_gravity.zero_()
        env.projected_gravity[:, 2] = -1.0
        env.base_lin_vel.zero_()
        env.base_lin_vel[:, 0] = 0.5
        env.base_ang_vel.zero_()
        env.navigation_direction.fill_(1)
        env.distance.fill_(1.0)
        env.stair_crossed.zero_()
        env.stair_progress_steps.zero_()
        env.stair_progress_anchor.copy_(env.root_states[:, 0])
        env.goal_hold_timer.zero_()
        env.stay_timer.zero_()
        env.terminal_reason.zero_()
        env.episode_length_buf.fill_(10)

    clear_terminal_state()
    leave_id = 2
    env.root_states[leave_id, 0] = env.room_origins[leave_id, 0] + 7.2
    env.root_states[leave_id, 2] = env.base_init_state[2] + 0.40
    env.distance[leave_id] = 0.1
    env.stair_crossed[leave_id] = True
    for _ in range(6):
        env.check_termination()
    if int(env.goal_hold_timer[leave_id]) != 6:
        raise RuntimeError("Goal hold timer did not accumulate while continuously eligible")
    env.distance[leave_id] = 1.0
    env.check_termination()
    if int(env.goal_hold_timer[leave_id]) != 0:
        raise RuntimeError("Goal hold timer did not reset immediately after leaving")

    clear_terminal_state()
    room_x = env.room_origins[:, 0]

    stair_stuck_id = 0
    env.root_states[stair_stuck_id, 0] = room_x[stair_stuck_id] + 5.0
    env.stair_progress_anchor[stair_stuck_id] = env.root_states[stair_stuck_id, 0] - 0.10
    env.stair_progress_steps[stair_stuck_id] = (
        env.cfg.depth_stairs.stair_stuck_window_steps - 1
    )

    wrong_height_id = 1
    env.root_states[wrong_height_id, 0] = room_x[wrong_height_id] + 7.2
    env.root_states[wrong_height_id, 2] = env.base_init_state[2]
    env.distance[wrong_height_id] = 0.1
    env.stair_crossed[wrong_height_id] = True

    left_goal_id = 2
    env.root_states[left_goal_id, 0] = room_x[left_goal_id] + 3.5
    env.root_states[left_goal_id, 2] = env.base_init_state[2]

    simultaneous_id = 3
    success_id = 4
    for env_id in (simultaneous_id, success_id):
        env.root_states[env_id, 0] = room_x[env_id] + 7.2
        env.root_states[env_id, 2] = env.base_init_state[2] + 0.40
        env.distance[env_id] = 0.1
        env.stair_crossed[env_id] = True
        env.goal_hold_timer[env_id] = env.cfg.env.goal_reached_time - 1
    env.episode_length_buf[simultaneous_id] = int(env.max_episode_length)
    env.episode_length_buf[success_id] = int(env.max_episode_length) - 1

    env.check_termination()
    expected = [
        TERMINAL_STAIR_STUCK,
        TERMINAL_NONE,
        TERMINAL_NONE,
        TERMINAL_TIMEOUT,
        TERMINAL_SUCCESS,
    ]
    actual = env.terminal_reason.tolist()
    if actual != expected:
        raise RuntimeError(f"Terminal regression mismatch: expected {expected}, got {actual}")
    if bool(env.reset_buf[wrong_height_id]) or bool(env.reset_buf[left_goal_id]):
        raise RuntimeError("Wrong-height or left-goal scenario terminated incorrectly")
    if bool(env.goal_reached_flag[simultaneous_id]):
        raise RuntimeError("A simultaneous timeout was incorrectly marked successful")

    print(
        json.dumps(
            {
                "stair_stuck": TERMINAL_REASON_NAMES[actual[stair_stuck_id]],
                "wrong_height": TERMINAL_REASON_NAMES[actual[wrong_height_id]],
                "arrive_then_leave": TERMINAL_REASON_NAMES[actual[left_goal_id]],
                "success_and_timeout_same_step": TERMINAL_REASON_NAMES[
                    actual[simultaneous_id]
                ],
                "valid_success": TERMINAL_REASON_NAMES[actual[success_id]],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
