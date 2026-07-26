"""Train, export, and target-room gate blind stair locomotion in stages."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from legged_gym import LEGGED_GYM_ROOT_DIR


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=8192)
    parser.add_argument("--first_gate_iteration", type=int, default=500)
    parser.add_argument("--gate_interval", type=int, default=100)
    parser.add_argument("--max_iterations", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--resume_run", default=None)
    parser.add_argument("--resume_iteration", type=int, default=0)
    parser.add_argument("--sim_device", default="cuda:0")
    parser.add_argument("--rl_device", default="cuda:0")
    parser.add_argument("--pipeline", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument(
        "--output_model",
        default=os.path.join(
            LEGGED_GYM_ROOT_DIR, "legged_gym", "ctrl_model", "blind_stair_loco.pt"
        ),
    )
    parser.add_argument(
        "--gate_dir",
        default=os.path.join(
            LEGGED_GYM_ROOT_DIR, "logs", "Go2_blind_stair_loco", "target_room_gates"
        ),
    )
    args = parser.parse_args()
    if not 0 < args.first_gate_iteration <= args.max_iterations:
        raise ValueError("first gate iteration must be within the training budget")
    if args.gate_interval <= 0:
        raise ValueError("--gate_interval must be positive")
    if (args.resume_run is None) != (args.resume_iteration == 0):
        raise ValueError("--resume_run and a positive --resume_iteration must be used together")
    if not 0 <= args.resume_iteration < args.max_iterations:
        raise ValueError("--resume_iteration must be below --max_iterations")
    return args


def newest_run(log_root, existing):
    candidates = [path for path in log_root.iterdir() if path.is_dir() and path not in existing]
    if not candidates:
        raise RuntimeError(f"Training did not create a run under {log_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def run_checked(command, env, cwd):
    print(" ".join(str(part) for part in command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def main():
    args = parse_args()
    repo_root = Path(LEGGED_GYM_ROOT_DIR).parents[1]
    scripts_dir = Path(LEGGED_GYM_ROOT_DIR) / "legged_gym" / "scripts"
    log_root = Path(LEGGED_GYM_ROOT_DIR) / "logs" / "Go2_blind_stair_loco"
    log_root.mkdir(parents=True, exist_ok=True)
    gate_dir = Path(args.gate_dir).resolve()
    gate_dir.mkdir(parents=True, exist_ok=True)
    output_model = Path(args.output_model).resolve()
    output_metadata = Path(str(output_model) + ".json")

    process_env = os.environ.copy()
    python_paths = [
        str(Path(LEGGED_GYM_ROOT_DIR)),
        str(repo_root / "training" / "rsl_rl"),
    ]
    if process_env.get("PYTHONPATH"):
        python_paths.append(process_env["PYTHONPATH"])
    process_env["PYTHONPATH"] = os.pathsep.join(python_paths)
    process_env["WANDB_MODE"] = "disabled"

    current_iteration = args.resume_iteration
    current_run = log_root / args.resume_run if args.resume_run is not None else None
    if current_run is not None:
        checkpoint = current_run / f"model_{current_iteration}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Resume checkpoint is missing: {checkpoint}")
    next_chunk = (
        args.first_gate_iteration if current_iteration == 0 else args.gate_interval
    )
    history_path = gate_dir / "gate_history.json"
    if history_path.is_file() and current_iteration > 0:
        with open(history_path, "r", encoding="utf-8") as stream:
            history = json.load(stream)
    else:
        history = []
    while current_iteration < args.max_iterations:
        chunk = min(next_chunk, args.max_iterations - current_iteration)
        existing = set(path for path in log_root.iterdir() if path.is_dir())
        train_command = [
            sys.executable,
            str(scripts_dir / "train.py"),
            "--task",
            "go2_blind_stair_loco",
            "--headless",
            "--num_envs",
            str(args.num_envs),
            "--max_iterations",
            str(chunk),
            "--seed",
            str(args.seed),
            "--sim_device",
            args.sim_device,
            "--rl_device",
            args.rl_device,
            "--pipeline",
            args.pipeline,
            "--run_name",
            "staged_gate",
        ]
        if current_run is not None:
            train_command.extend(
                [
                    "--resume",
                    "--load_run",
                    current_run.name,
                    "--checkpoint",
                    str(current_iteration),
                ]
            )
            if current_iteration >= 1800:
                train_command.append("--down_stair_focus")
            if current_iteration >= 2700:
                train_command.append("--low_speed_focus")
            min_level = min(6, max(0, current_iteration // 100 - 1))
            max_level = min(9, min_level + 3)
            train_command.extend(
                [
                    "--min_terrain_level",
                    str(min_level),
                    "--max_terrain_level",
                    str(max_level),
                ]
            )
        run_checked(train_command, process_env, repo_root)
        current_iteration += chunk
        current_run = newest_run(log_root, existing)
        checkpoint = current_run / f"model_{current_iteration}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Expected staged checkpoint is missing: {checkpoint}")

        run_checked(
            [
                sys.executable,
                str(scripts_dir / "export_blind_stair_loco.py"),
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(output_model),
            ],
            process_env,
            repo_root,
        )

        gate_csv = gate_dir / f"gate_{current_iteration}.csv"
        gate_summary = gate_dir / f"gate_{current_iteration}.json"
        run_checked(
            [
                sys.executable,
                str(scripts_dir / "gate_blind_stair_loco.py"),
                "--task",
                "go2_pos_stairs_minimal",
                "--headless",
                "--loco_model",
                str(output_model),
                "--loco_metadata",
                str(output_metadata),
                "--sim_device",
                args.sim_device,
                "--rl_device",
                args.rl_device,
                "--pipeline",
                args.pipeline,
                "--output_csv",
                str(gate_csv),
                "--output_summary",
                str(gate_summary),
            ],
            process_env,
            repo_root,
        )
        with open(gate_summary, "r", encoding="utf-8") as stream:
            result = json.load(stream)
        history.append(
            {
                "iteration": current_iteration,
                "checkpoint": str(checkpoint),
                "model": str(output_model),
                "model_metadata": str(output_metadata),
                **result,
            }
        )
        with open(history_path, "w", encoding="utf-8") as stream:
            json.dump(history, stream, indent=2)
        if result["passes_gate"]:
            print(f"Blind stair locomotion passed at iteration {current_iteration}")
            return
        next_chunk = args.gate_interval

    raise RuntimeError(
        f"Blind stair locomotion did not pass the target-room gate by iteration {current_iteration}"
    )


if __name__ == "__main__":
    main()
