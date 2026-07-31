"""Weight-only forward-descent finetuning with checkpoint-local strict gates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from legged_gym import LEGGED_GYM_ROOT_DIR


TASK = "go2_blind_stair_loco_forward_finetune"
EXPERIMENT = "Go2_blind_stair_loco_forward_finetune"
BASELINE_SHA256 = "98abb6cf55147f7e937d775d1b87a10f96c575e7d1e256c9be971d43320b83f6"
DEFAULT_BASELINE = os.path.join(
    LEGGED_GYM_ROOT_DIR,
    "logs",
    "Go2_blind_stair_loco",
    "07_27_05-04-43_staged_gate",
    "model_2800.pt",
)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_checkpoint", default=DEFAULT_BASELINE)
    parser.add_argument("--branch_name", default="from_2800_forward_lr1e4_0731")
    parser.add_argument("--num_envs", type=int, default=8192)
    parser.add_argument("--first_gate_iteration", type=int, default=50)
    parser.add_argument("--gate_interval", type=int, choices=(50, 100), default=50)
    parser.add_argument("--max_iterations", type=int, default=600)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--resume_run", default=None)
    parser.add_argument("--resume_iteration", type=int, default=0)
    parser.add_argument("--sim_device", default="cuda:0")
    parser.add_argument("--rl_device", default="cuda:0")
    parser.add_argument("--pipeline", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--artifact_root", default=None)
    args = parser.parse_args()
    if args.trials != 20:
        raise ValueError("Formal forward-finetune gates require exactly 20 trials per group")
    if not 0 < args.first_gate_iteration <= args.max_iterations:
        raise ValueError("--first_gate_iteration must be inside the training budget")
    if args.max_iterations > 600:
        raise ValueError("Forward finetune may add at most 600 iterations")
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


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)


def main():
    args = parse_args()
    repo_root = Path(LEGGED_GYM_ROOT_DIR).parents[1]
    scripts_dir = Path(LEGGED_GYM_ROOT_DIR) / "legged_gym" / "scripts"
    log_root = Path(LEGGED_GYM_ROOT_DIR) / "logs" / EXPERIMENT
    log_root.mkdir(parents=True, exist_ok=True)
    artifact_root = (
        Path(args.artifact_root).resolve()
        if args.artifact_root
        else log_root / "branches" / args.branch_name
    )
    gates_dir = artifact_root / "gates"
    exports_dir = artifact_root / "exports"
    manifest_path = artifact_root / "branch_manifest.json"
    history_path = gates_dir / "gate_history.json"

    baseline = Path(args.baseline_checkpoint).resolve()
    if not baseline.is_file():
        raise FileNotFoundError(f"Baseline checkpoint is missing: {baseline}")
    baseline_sha = sha256(baseline)
    if baseline_sha != BASELINE_SHA256:
        raise ValueError(
            f"Baseline checkpoint SHA256 mismatch: expected {BASELINE_SHA256}, got {baseline_sha}"
        )

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
    current_run = log_root / args.resume_run if args.resume_run else None
    if current_run is not None:
        checkpoint = current_run / f"model_{current_iteration}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Resume checkpoint is missing: {checkpoint}")

    if manifest_path.exists() and current_iteration == 0:
        raise FileExistsError(
            f"Branch artifacts already exist and will not be overwritten: {manifest_path}"
        )
    if not manifest_path.exists():
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
        write_json(
            manifest_path,
            {
                "schema_version": 1,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "task": TASK,
                "experiment": EXPERIMENT,
                "branch_name": args.branch_name,
                "git_commit_at_start": git_commit,
                "baseline_checkpoint": str(baseline),
                "baseline_checkpoint_sha256": baseline_sha,
                "initialization": "network_weights_only",
                "optimizer_state_loaded": False,
                "iteration_reset_to_zero": True,
                "learning_rate": 1.0e-4,
                "learning_rate_schedule": "fixed",
                "num_envs": args.num_envs,
                "terrain_proportions": {"flat": 0.10, "up": 0.20, "down": 0.70},
                "down_commands_body_vx_mps": [0.25, 0.40, 0.55, 0.70],
                "down_spawn_yaw_rad": "pi",
                "actor_observation_dim": 45,
                "actor_action_dim": 12,
                "external_perception": False,
                "gate_trials_per_group": args.trials,
                "gate_required_successes_per_group": 19,
                "max_finetune_iterations": 600,
            },
        )

    if history_path.is_file():
        with open(history_path, "r", encoding="utf-8") as stream:
            history = json.load(stream)
    else:
        history = []

    next_chunk = args.first_gate_iteration if current_iteration == 0 else args.gate_interval
    while current_iteration < args.max_iterations:
        chunk = min(next_chunk, args.max_iterations - current_iteration)
        target_iteration = current_iteration + chunk
        export_path = exports_dir / f"blind_stair_loco_iter_{target_iteration:04d}.pt"
        gate_csv = gates_dir / f"gate_{target_iteration:04d}.csv"
        gate_json = gates_dir / f"gate_{target_iteration:04d}.json"
        for output in (export_path, Path(str(export_path) + ".json"), gate_csv, gate_json):
            if output.exists():
                raise FileExistsError(f"Refusing to overwrite branch artifact: {output}")

        existing = {path for path in log_root.iterdir() if path.is_dir()}
        train_command = [
            sys.executable,
            str(scripts_dir / "train.py"),
            "--task",
            TASK,
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
            args.branch_name,
        ]
        if current_run is None:
            train_command.extend(["--init_checkpoint", str(baseline)])
        else:
            train_command.extend(
                [
                    "--resume",
                    "--load_run",
                    current_run.name,
                    "--checkpoint",
                    str(current_iteration),
                ]
            )
        run_checked(train_command, process_env, repo_root)
        current_iteration = target_iteration
        current_run = newest_run(log_root, existing)
        checkpoint = current_run / f"model_{current_iteration}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Expected checkpoint is missing: {checkpoint}")

        run_checked(
            [
                sys.executable,
                str(scripts_dir / "export_blind_stair_loco.py"),
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(export_path),
            ],
            process_env,
            repo_root,
        )
        run_checked(
            [
                sys.executable,
                str(scripts_dir / "gate_blind_stair_loco.py"),
                "--task",
                "go2_pos_stairs_minimal",
                "--headless",
                "--loco_model",
                str(export_path),
                "--loco_metadata",
                str(export_path) + ".json",
                "--sim_device",
                args.sim_device,
                "--rl_device",
                args.rl_device,
                "--pipeline",
                args.pipeline,
                "--trials",
                str(args.trials),
                "--output_csv",
                str(gate_csv),
                "--output_summary",
                str(gate_json),
            ],
            process_env,
            repo_root,
        )
        with open(gate_json, "r", encoding="utf-8") as stream:
            gate_result = json.load(stream)
        history.append(
            {
                "finetune_iteration": current_iteration,
                "training_run": current_run.name,
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256(checkpoint),
                "torchscript": str(export_path.resolve()),
                "metadata": str(Path(str(export_path) + ".json").resolve()),
                "gate_csv": str(gate_csv.resolve()),
                "gate_json": str(gate_json.resolve()),
                "gate": gate_result,
            }
        )
        write_json(history_path, history)
        if gate_result["passes_gate"]:
            write_json(
                artifact_root / "candidate.json",
                history[-1],
            )
            print(f"Forward stair finetune passed the strict gate at iteration {current_iteration}")
            return
        next_chunk = args.gate_interval

    raise RuntimeError(
        f"Forward stair finetune did not pass the strict gate by iteration {current_iteration}"
    )


if __name__ == "__main__":
    main()
