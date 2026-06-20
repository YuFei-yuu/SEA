import argparse
import csv
import os
import subprocess
import sys
from typing import Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry


Summary = Dict[str, object]


def _parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoints", type=int, nargs="+", required=True)
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--analysis_dir", type=str, default=None)
    known_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    base_args = get_args()
    base_args.checkpoints = known_args.checkpoints
    base_args.num_episodes = known_args.num_episodes
    base_args.analysis_dir = known_args.analysis_dir
    return base_args


def _default_analysis_dir(args) -> str:
    _, train_cfg = task_registry.get_cfgs(name=args.task)
    load_run = args.load_run
    if load_run is None or load_run == -1:
        load_run = str(train_cfg.runner.load_run)
    if load_run is None or load_run == "-1":
        log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name)
        runs = [
            entry
            for entry in os.listdir(log_root)
            if entry != "exported" and os.path.isdir(os.path.join(log_root, entry))
        ]
        runs.sort()
        if not runs:
            raise FileNotFoundError(f"No runs found in {log_root}")
        load_run = runs[-1]
    run_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name, load_run)
    analysis_dir = os.path.join(run_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    return analysis_dir


def _as_float(row: Summary, key: str) -> float:
    try:
        return float(row.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _plot_bar_group(
    rows: Sequence[Summary],
    analysis_dir: str,
    filename: str,
    columns: Sequence[str],
    title: str,
    ylabel: str,
) -> bool:
    present = [column for column in columns if any(column in row for row in rows)]
    if not present:
        return False

    labels = [str(int(_as_float(row, "checkpoint"))) for row in rows]
    x_values = list(range(len(labels)))
    width = 0.8 / max(len(present), 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, column in enumerate(present):
        offsets = [
            value + (idx - (len(present) - 1) / 2.0) * width
            for value in x_values
        ]
        ax.bar(offsets, [_as_float(row, column) for row in rows], width=width, label=column)
    ax.set_title(title)
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_values)
    ax.set_xticklabels(labels)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(analysis_dir, filename), dpi=160)
    plt.close(fig)
    return True


def _rank_checkpoints(rows: Sequence[Summary]) -> List[Summary]:
    return sorted(
        rows,
        key=lambda row: (
            -_as_float(row, "safe_success_rate"),
            -_as_float(row, "success_rate"),
            _as_float(row, "avg_dynamic_collision_count"),
            _as_float(row, "avg_near_miss_count"),
            _as_float(row, "mean_time_to_goal"),
        ),
    )


def _write_csv(rows: Sequence[Summary], csv_path: str):
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_csv_rows(csv_path: str) -> List[Summary]:
    if not os.path.isfile(csv_path):
        return []
    with open(csv_path, newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def _evaluate_checkpoint_subprocess(args, checkpoint: int, output_csv: str):
    before_rows = len(_read_csv_rows(output_csv))
    script_path = os.path.join(
        LEGGED_GYM_ROOT_DIR,
        "legged_gym",
        "scripts",
        "evaluate_dynamic.py",
    )
    cmd = [
        sys.executable,
        script_path,
        "--task",
        args.task,
        "--checkpoint",
        str(checkpoint),
        "--num_episodes",
        str(args.num_episodes),
        "--output_csv",
        output_csv,
        "--headless",
    ]
    if args.load_run is not None:
        cmd.extend(["--load_run", str(args.load_run)])
    if getattr(args, "resume_experiment_name", None) is not None:
        cmd.extend(["--resume_experiment_name", str(args.resume_experiment_name)])
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])

    env = os.environ.copy()
    pythonpath = [
        os.path.join(LEGGED_GYM_ROOT_DIR),
        os.path.abspath(os.path.join(LEGGED_GYM_ROOT_DIR, "..", "rsl_rl")),
        env.get("PYTHONPATH", ""),
    ]
    env["PYTHONPATH"] = ":".join([path for path in pythonpath if path])
    result = subprocess.run(cmd, env=env, check=False)
    after_rows = len(_read_csv_rows(output_csv))
    if result.returncode not in (0, 139, -11) or after_rows <= before_rows:
        raise RuntimeError(
            f"Checkpoint {checkpoint} evaluation failed with return code {result.returncode}"
        )


def _markdown_table(rows: Sequence[Summary]) -> List[str]:
    columns = [
        "checkpoint",
        "success_rate",
        "safe_success_rate",
        "avg_dynamic_collision_count",
        "avg_near_miss_count",
        "mean_time_to_goal",
    ]
    present = [column for column in columns if any(column in row for row in rows)]
    lines = [
        "| " + " | ".join(present) + " |",
        "| " + " | ".join(["---:" for _ in present]) + " |",
    ]
    for row in rows:
        values = []
        for column in present:
            value = row.get(column, "")
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _write_report(rows: Sequence[Summary], analysis_dir: str, generated_plots: Sequence[str]) -> str:
    ranking = _rank_checkpoints(rows)
    best = ranking[0]
    report_path = os.path.join(analysis_dir, "eval_checkpoints_summary.md")
    lines: List[str] = [
        "# Checkpoint Evaluation Summary",
        "",
        f"- best_checkpoint: `{int(_as_float(best, 'checkpoint'))}`",
        f"- best_safe_success_rate: `{_as_float(best, 'safe_success_rate'):.4f}`",
        f"- best_success_rate: `{_as_float(best, 'success_rate'):.4f}`",
        "",
        "## Ranking",
        "",
    ]
    lines.extend(_markdown_table(ranking))
    lines.extend(["", "## Figures", ""])
    for filename in generated_plots:
        lines.append(f"![{filename}]({filename})")
        lines.append("")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return report_path


def main():
    args = _parse_args()
    args.headless = True
    analysis_dir = args.analysis_dir or _default_analysis_dir(args)
    os.makedirs(analysis_dir, exist_ok=True)

    temp_csv = os.path.join(analysis_dir, "eval_checkpoints_raw.csv")
    if os.path.exists(temp_csv):
        os.remove(temp_csv)
    for checkpoint in args.checkpoints:
        print(f"\n=== Evaluating checkpoint {checkpoint} ===")
        _evaluate_checkpoint_subprocess(args, checkpoint, temp_csv)

    summaries = _read_csv_rows(temp_csv)
    if not summaries:
        raise RuntimeError("No checkpoint summaries were produced.")

    csv_path = os.path.join(analysis_dir, "eval_checkpoints.csv")
    _write_csv(summaries, csv_path)

    generated = []
    if _plot_bar_group(
        summaries,
        analysis_dir,
        "eval_success_collision_bars.png",
        [
            "success_rate",
            "safe_success_rate",
            "avg_dynamic_collision_count",
            "avg_body_collision_count",
        ],
        "Success and Collision by Checkpoint",
        "Rate / Count",
    ):
        generated.append("eval_success_collision_bars.png")
    if _plot_bar_group(
        summaries,
        analysis_dir,
        "eval_safety_metrics_bars.png",
        [
            "avg_near_miss_count",
            "avg_min_ttc",
            "avg_min_dynamic_clearance",
            "avg_future_dynamic_clearance",
            "avg_dynamic_cbf_intervention_rate",
        ],
        "Safety Metrics by Checkpoint",
        "Metric value",
    ):
        generated.append("eval_safety_metrics_bars.png")
    if _plot_bar_group(
        summaries,
        analysis_dir,
        "eval_dynamic_direction_bars.png",
        [
            "avg_pass_behind_score",
            "avg_future_dynamic_clearance",
            "avg_min_dynamic_clearance",
        ],
        "Dynamic Direction Metrics by Checkpoint",
        "Metric value",
    ):
        generated.append("eval_dynamic_direction_bars.png")
    if _plot_bar_group(
        summaries,
        analysis_dir,
        "eval_efficiency_bars.png",
        [
            "timeout_rate",
            "mean_episode_duration",
            "mean_time_to_goal",
            "avg_active_dynamic_count",
        ],
        "Efficiency Metrics by Checkpoint",
        "Metric value",
    ):
        generated.append("eval_efficiency_bars.png")

    report_path = _write_report(summaries, analysis_dir, generated)
    best_checkpoint = int(_as_float(_rank_checkpoints(summaries)[0], "checkpoint"))
    print(f"\neval_csv: {csv_path}")
    print(f"eval_report: {report_path}")
    print(f"recommended_checkpoint: {best_checkpoint}")


if __name__ == "__main__":
    main()
