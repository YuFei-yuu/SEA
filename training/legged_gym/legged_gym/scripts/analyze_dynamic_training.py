import argparse
import os
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    return parser.parse_args()


def ensure_analysis_dir(run_dir: str) -> str:
    analysis_dir = os.path.join(run_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    return analysis_dir


def read_metrics(run_dir: str) -> pd.DataFrame:
    metrics_path = os.path.join(run_dir, "train_metrics.csv")
    if not os.path.isfile(metrics_path):
        raise FileNotFoundError(f"train_metrics.csv not found: {metrics_path}")
    return pd.read_csv(metrics_path)


def smooth_series(series: pd.Series, window: int = 5) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def plot_curves(df: pd.DataFrame, analysis_dir: str):
    iteration = df["iteration"]

    plot_specs = [
        ("reward_success_curves.png", [
            ("mean_reward", "Mean Reward"),
            ("success", "Success Rate"),
            ("safe_success", "Safe Success Rate"),
        ], "Reward and Success"),
        ("collision_safety_curves.png", [
            ("dynamic_collision_count", "Dynamic Collision Count"),
            ("body_collision_count", "Body Collision Count"),
            ("total_collision_count", "Total Collision Count"),
            ("near_miss_count", "Near Miss Count"),
        ], "Collision and Near Miss"),
        ("clearance_ttc_curves.png", [
            ("min_ttc", "Min TTC"),
            ("min_dynamic_clearance", "Min Dynamic Clearance"),
            ("dynamic_cbf_intervention_rate", "Dynamic CBF Intervention Rate"),
            ("shield_intervention_rate", "Shield Intervention Rate"),
        ], "Clearance, TTC and Intervention"),
        ("optimization_curves.png", [
            ("value_loss", "Value Loss"),
            ("surrogate_loss", "Surrogate Loss"),
            ("regularization_loss", "Regularization Loss"),
            ("intervention_loss", "Intervention Loss"),
        ], "Optimization"),
        ("efficiency_curves.png", [
            ("mean_episode_length", "Mean Episode Length"),
            ("episode_duration", "Episode Duration"),
            ("time_to_goal", "Time To Goal"),
            ("active_dynamic_count", "Active Dynamic Count"),
        ], "Episode Efficiency"),
    ]

    for filename, columns, title in plot_specs:
        fig, axes = plt.subplots(len(columns), 1, figsize=(10, 3 * len(columns)), sharex=True)
        if len(columns) == 1:
            axes = [axes]
        for ax, (column, label) in zip(axes, columns):
            if column not in df.columns:
                ax.text(0.5, 0.5, f"Missing column: {column}", ha="center", va="center")
                ax.set_axis_off()
                continue
            ax.plot(iteration, df[column], alpha=0.35, linewidth=1.2, label=label)
            ax.plot(iteration, smooth_series(df[column]), linewidth=2.0, label=f"{label} (smoothed)")
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")
        axes[-1].set_xlabel("Iteration")
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(os.path.join(analysis_dir, filename), dpi=160)
        plt.close(fig)


def summarize(df: pd.DataFrame, analysis_dir: str):
    last = df.iloc[-1]
    best_success = float(df["success"].max()) if "success" in df.columns else 0.0
    best_safe_success = float(df["safe_success"].max()) if "safe_success" in df.columns else 0.0
    min_dynamic_collision = float(df["dynamic_collision_count"].min()) if "dynamic_collision_count" in df.columns else 0.0
    min_total_collision = float(df["total_collision_count"].min()) if "total_collision_count" in df.columns else 0.0

    lines: List[str] = [
        f"run_dir: {analysis_dir.rsplit('/analysis', 1)[0]}",
        f"num_points: {len(df)}",
        f"last_iteration: {int(last['iteration'])}",
        f"last_mean_reward: {float(last['mean_reward']):.4f}",
        f"last_success: {float(last['success']) if 'success' in df.columns else 0.0:.4f}",
        f"last_safe_success: {float(last['safe_success']) if 'safe_success' in df.columns else 0.0:.4f}",
        f"last_dynamic_collision_count: {float(last['dynamic_collision_count']) if 'dynamic_collision_count' in df.columns else 0.0:.4f}",
        f"last_total_collision_count: {float(last['total_collision_count']) if 'total_collision_count' in df.columns else 0.0:.4f}",
        f"last_near_miss_count: {float(last['near_miss_count']) if 'near_miss_count' in df.columns else 0.0:.4f}",
        f"last_min_ttc: {float(last['min_ttc']) if 'min_ttc' in df.columns else 0.0:.4f}",
        f"last_min_dynamic_clearance: {float(last['min_dynamic_clearance']) if 'min_dynamic_clearance' in df.columns else 0.0:.4f}",
        f"best_success: {best_success:.4f}",
        f"best_safe_success: {best_safe_success:.4f}",
        f"min_dynamic_collision_count: {min_dynamic_collision:.4f}",
        f"min_total_collision_count: {min_total_collision:.4f}",
    ]
    summary_path = os.path.join(analysis_dir, "train_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    analysis_dir = ensure_analysis_dir(args.run_dir)
    df = read_metrics(args.run_dir)
    plot_curves(df, analysis_dir)
    summarize(df, analysis_dir)
    print(f"analysis_dir: {analysis_dir}")


if __name__ == "__main__":
    main()
