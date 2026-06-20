import argparse
import csv
import os
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


Rows = List[Dict[str, float]]
PlotSpec = Tuple[str, Sequence[Tuple[str, str]], str]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--smooth_window", type=int, default=5)
    return parser.parse_args()


def ensure_analysis_dir(run_dir: str) -> str:
    analysis_dir = os.path.join(run_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    return analysis_dir


def _to_float(value: str) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def read_metrics(run_dir: str) -> Tuple[Rows, List[str]]:
    metrics_path = os.path.join(run_dir, "train_metrics.csv")
    if not os.path.isfile(metrics_path):
        raise FileNotFoundError(f"train_metrics.csv not found: {metrics_path}")
    with open(metrics_path, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        rows = [
            {key: _to_float(value) for key, value in row.items()}
            for row in reader
        ]
    if not rows:
        raise ValueError(f"No rows in {metrics_path}")
    return rows, fieldnames


def has_column(fieldnames: Iterable[str], column: str) -> bool:
    return column in fieldnames


def get_column(rows: Rows, column: str) -> List[float]:
    return [row.get(column, 0.0) for row in rows]


def x_axis(rows: Rows, fieldnames: Sequence[str]) -> Tuple[List[float], str]:
    if has_column(fieldnames, "timesteps"):
        return get_column(rows, "timesteps"), "Timesteps"
    return get_column(rows, "iteration"), "Iteration"


def smooth_values(values: Sequence[float], window: int) -> List[float]:
    if window <= 1:
        return list(values)
    smoothed: List[float] = []
    for idx in range(len(values)):
        start = max(0, idx + 1 - window)
        chunk = values[start : idx + 1]
        smoothed.append(sum(chunk) / max(len(chunk), 1))
    return smoothed


def existing_columns(fieldnames: Iterable[str], columns: Iterable[str]) -> List[str]:
    field_set = set(fieldnames)
    return [column for column in columns if column in field_set]


def plot_curve_group(
    rows: Rows,
    fieldnames: Sequence[str],
    analysis_dir: str,
    filename: str,
    columns: Sequence[Tuple[str, str]],
    title: str,
    smooth_window: int,
) -> bool:
    present = [(column, label) for column, label in columns if has_column(fieldnames, column)]
    if not present:
        return False

    x, xlabel = x_axis(rows, fieldnames)
    fig, axes = plt.subplots(len(present), 1, figsize=(11, 2.8 * len(present)), sharex=True)
    if len(present) == 1:
        axes = [axes]

    for ax, (column, label) in zip(axes, present):
        values = get_column(rows, column)
        ax.plot(x, values, alpha=0.35, linewidth=1.1, label=label)
        ax.plot(
            x,
            smooth_values(values, smooth_window),
            linewidth=2.0,
            label=f"{label} (smoothed)",
        )
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    axes[-1].set_xlabel(xlabel)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(os.path.join(analysis_dir, filename), dpi=160)
    plt.close(fig)
    return True


def plot_stacked_area(
    rows: Rows,
    fieldnames: Sequence[str],
    analysis_dir: str,
    filename: str,
    columns: Sequence[str],
    title: str,
) -> bool:
    present = existing_columns(fieldnames, columns)
    if not present:
        return False

    x, xlabel = x_axis(rows, fieldnames)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.stackplot(x, [get_column(rows, column) for column in present], labels=present, alpha=0.82)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Rate / Count")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(analysis_dir, filename), dpi=160)
    plt.close(fig)
    return True


def plot_reward_breakdown(
    rows: Rows,
    fieldnames: Sequence[str],
    analysis_dir: str,
    smooth_window: int,
) -> bool:
    reward_columns = [column for column in fieldnames if column.startswith("rew_")]
    if not reward_columns:
        return False

    x, xlabel = x_axis(rows, fieldnames)
    fig, ax = plt.subplots(figsize=(12, 6))
    for column in reward_columns:
        ax.plot(
            x,
            smooth_values(get_column(rows, column), smooth_window),
            linewidth=1.6,
            label=column,
        )
    ax.set_title("Reward Breakdown")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Episode reward contribution")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(analysis_dir, "reward_breakdown.png"), dpi=160)
    plt.close(fig)
    return True


def plot_all_curves(
    rows: Rows,
    fieldnames: Sequence[str],
    analysis_dir: str,
    smooth_window: int,
) -> List[str]:
    plot_specs: List[PlotSpec] = [
        (
            "rl_reward_length.png",
            [
                ("mean_reward", "Mean Reward"),
                ("mean_episode_length", "Mean Episode Length"),
                ("episode_duration", "Episode Duration"),
                ("time_to_goal", "Time To Goal"),
            ],
            "RL Reward and Episode Efficiency",
        ),
        (
            "rl_success_rates.png",
            [
                ("success", "Success Rate"),
                ("safe_success", "Safe Success Rate"),
                ("timeout", "Timeout Rate"),
            ],
            "Success and Timeout",
        ),
        (
            "rl_optimization.png",
            [
                ("value_loss", "Value Loss"),
                ("surrogate_loss", "Surrogate Loss"),
                ("regularization_loss", "Regularization Loss"),
                ("smooth_loss", "Smooth Loss"),
                ("intervention_loss", "Intervention Loss"),
                ("mean_action_std", "Mean Action Std"),
            ],
            "PPO Optimization",
        ),
        (
            "safety_events.png",
            [
                ("dynamic_collision_count", "Dynamic Collision Count"),
                ("body_collision_count", "Body Collision Count"),
                ("total_collision_count", "Total Collision Count"),
                ("near_miss_count", "Near Miss Count"),
            ],
            "Safety Events",
        ),
        (
            "safety_clearance_ttc.png",
            [
                ("min_ttc", "Min TTC"),
                ("min_dynamic_clearance", "Min Dynamic Clearance"),
                ("future_dynamic_clearance", "Future Dynamic Clearance"),
                ("active_dynamic_count", "Active Dynamic Count"),
            ],
            "Dynamic Clearance and TTC",
        ),
        (
            "dynamic_direction.png",
            [
                ("pass_behind_score", "Pass Behind Score"),
                ("future_dynamic_clearance", "Future Dynamic Clearance"),
                ("min_dynamic_clearance", "Min Dynamic Clearance"),
            ],
            "Dynamic Avoidance Direction",
        ),
        (
            "action_goal_alignment.png",
            [
                ("ubar_goal_angle", "u_bar Goal Angle"),
                ("ustatic_goal_angle", "u_static Goal Angle"),
                ("usafe_goal_angle", "u_safe Goal Angle"),
                ("ubar_norm", "u_bar Norm"),
                ("usafe_norm", "u_safe Norm"),
            ],
            "Action Goal Alignment",
        ),
        (
            "cbf_intervention.png",
            [
                ("shield_intervention_rate", "Shield Intervention Rate"),
                ("shield_intervention_mean", "Shield Intervention Mean"),
                ("shield_intervention_step_rate", "Shield Intervention Step Rate"),
                ("dynamic_cbf_intervention_rate", "Dynamic CBF Intervention Rate"),
                ("dynamic_cbf_intervention_mean", "Dynamic CBF Intervention Mean"),
                ("dynamic_cbf_intervention_step_rate", "Dynamic CBF Intervention Step Rate"),
            ],
            "CBF Intervention",
        ),
        (
            "throughput.png",
            [
                ("fps", "FPS"),
                ("collection_time", "Collection Time"),
                ("learn_time", "Learn Time"),
                ("mean_num_sim", "Mean Num Sim"),
            ],
            "Training Throughput",
        ),
        (
            "motion_mix.png",
            [
                ("motion_count_linear_crossing", "Linear Crossing"),
                ("motion_count_linear_diagonal", "Linear Diagonal"),
                ("motion_count_circular", "Circular"),
                ("motion_count_figure_eight", "Figure Eight"),
            ],
            "Dynamic Obstacle Motion Mix",
        ),
    ]

    generated: List[str] = []
    for filename, columns, title in plot_specs:
        if plot_curve_group(rows, fieldnames, analysis_dir, filename, columns, title, smooth_window):
            generated.append(filename)

    reset_columns = [
        "reset_goal",
        "reset_stand_still",
        "reset_timeout",
        "reset_fall",
        "reset_contact50",
        "reset_initial_contact50",
        "reset_spawn_collision",
        "reset_terminate_contact",
        "reset_dynamic_collision",
    ]
    if plot_stacked_area(
        rows,
        fieldnames,
        analysis_dir,
        "reset_reasons_stacked.png",
        reset_columns,
        "Reset Reasons",
    ):
        generated.append("reset_reasons_stacked.png")

    if plot_reward_breakdown(rows, fieldnames, analysis_dir, smooth_window):
        generated.append("reward_breakdown.png")

    return generated


def max_column(rows: Rows, fieldnames: Sequence[str], column: str) -> float:
    if not has_column(fieldnames, column):
        return 0.0
    return max(get_column(rows, column))


def min_column(rows: Rows, fieldnames: Sequence[str], column: str) -> float:
    if not has_column(fieldnames, column):
        return 0.0
    return min(get_column(rows, column))


def summarize(rows: Rows, fieldnames: Sequence[str], analysis_dir: str) -> Dict[str, float]:
    last = rows[-1]
    summary = {
        "num_points": float(len(rows)),
        "last_iteration": last.get("iteration", 0.0),
        "last_timesteps": last.get("timesteps", 0.0),
        "last_mean_reward": last.get("mean_reward", 0.0),
        "last_mean_episode_length": last.get("mean_episode_length", 0.0),
        "last_success": last.get("success", 0.0),
        "last_safe_success": last.get("safe_success", 0.0),
        "last_dynamic_collision_count": last.get("dynamic_collision_count", 0.0),
        "last_total_collision_count": last.get("total_collision_count", 0.0),
        "last_near_miss_count": last.get("near_miss_count", 0.0),
        "last_min_ttc": last.get("min_ttc", 0.0),
        "last_min_dynamic_clearance": last.get("min_dynamic_clearance", 0.0),
        "last_future_dynamic_clearance": last.get("future_dynamic_clearance", 0.0),
        "last_pass_behind_score": last.get("pass_behind_score", 0.0),
        "last_ubar_goal_angle": last.get("ubar_goal_angle", 0.0),
        "last_ustatic_goal_angle": last.get("ustatic_goal_angle", 0.0),
        "last_usafe_goal_angle": last.get("usafe_goal_angle", 0.0),
        "last_shield_intervention_step_rate": last.get("shield_intervention_step_rate", 0.0),
        "last_dynamic_cbf_intervention_step_rate": last.get(
            "dynamic_cbf_intervention_step_rate", 0.0
        ),
        "best_success": max_column(rows, fieldnames, "success"),
        "best_safe_success": max_column(rows, fieldnames, "safe_success"),
        "min_dynamic_collision_count": min_column(rows, fieldnames, "dynamic_collision_count"),
        "min_total_collision_count": min_column(rows, fieldnames, "total_collision_count"),
    }

    lines = [f"run_dir: {analysis_dir.rsplit('/analysis', 1)[0]}"]
    for key, value in summary.items():
        if key in {"num_points", "last_iteration", "last_timesteps"}:
            lines.append(f"{key}: {int(value)}")
        else:
            lines.append(f"{key}: {value:.4f}")

    summary_path = os.path.join(analysis_dir, "train_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return summary


def write_report(
    run_dir: str,
    analysis_dir: str,
    summary: Dict[str, float],
    generated_plots: Sequence[str],
) -> str:
    report_path = os.path.join(analysis_dir, "report.md")
    lines = [
        "# Training Visualization Report",
        "",
        f"- run_dir: `{run_dir}`",
        f"- num_points: `{int(summary.get('num_points', 0))}`",
        f"- last_iteration: `{int(summary.get('last_iteration', 0))}`",
    ]
    if summary.get("last_timesteps", 0) > 0:
        lines.append(f"- last_timesteps: `{int(summary['last_timesteps'])}`")

    lines.extend(
        [
            "",
            "## Key Metrics",
            "",
            "| Metric | Value |",
            "|---|---:|",
        ]
    )
    for key in [
        "last_mean_reward",
        "last_mean_episode_length",
        "last_success",
        "last_safe_success",
        "last_dynamic_collision_count",
        "last_total_collision_count",
        "last_near_miss_count",
        "last_min_ttc",
        "last_min_dynamic_clearance",
        "best_success",
        "best_safe_success",
    ]:
        lines.append(f"| {key} | {summary.get(key, 0.0):.4f} |")

    lines.extend(["", "## Figures", ""])
    for filename in generated_plots:
        lines.append(f"![{filename}]({filename})")
        lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return report_path


def main():
    args = parse_args()
    analysis_dir = ensure_analysis_dir(args.run_dir)
    rows, fieldnames = read_metrics(args.run_dir)
    generated_plots = plot_all_curves(rows, fieldnames, analysis_dir, args.smooth_window)
    summary = summarize(rows, fieldnames, analysis_dir)
    report_path = write_report(args.run_dir, analysis_dir, summary, generated_plots)
    print(f"analysis_dir: {analysis_dir}")
    print(f"report_path: {report_path}")


if __name__ == "__main__":
    main()
