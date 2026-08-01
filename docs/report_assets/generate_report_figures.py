#!/usr/bin/env python3
"""Generate reproducible figures for docs/report.md from workspace artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "axes.unicode_minus": False,
        "axes.titleweight": "bold",
        "axes.edgecolor": "#30363d",
        "axes.labelcolor": "#24292f",
        "xtick.color": "#424a53",
        "ytick.color": "#424a53",
        "figure.facecolor": "white",
        "axes.facecolor": "#fbfcfd",
        "savefig.facecolor": "white",
    }
)

BLUE = "#2f6f9f"
GREEN = "#27864b"
ORANGE = "#d9822b"
RED = "#c43d3d"
PURPLE = "#7654a6"
TEAL = "#148a8a"
GRAY = "#6e7781"
LIGHT_GRAY = "#d8dee4"


def load_json(relative_path: str):
    with (ROOT / relative_path).open(encoding="utf-8") as stream:
        return json.load(stream)


def load_csv(relative_path: str):
    with (ROOT / relative_path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def save(fig, name: str):
    fig.savefig(OUT / name, dpi=190, bbox_inches="tight")
    plt.close(fig)


def style_axis(ax, grid_axis="y"):
    ax.grid(True, axis=grid_axis, color="#d8dee4", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def architecture_figure():
    fig, ax = plt.subplots(figsize=(17.5, 7.2))
    ax.set_xlim(0, 17.5)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    def box(x, y, w, h, title, body, color, title_size=11):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.025,rounding_size=0.08",
            linewidth=1.5,
            edgecolor=color,
            facecolor="white",
        )
        ax.add_patch(patch)
        ax.add_patch(Rectangle((x, y + h - 0.36), w, 0.36, color=color, alpha=0.13, lw=0))
        ax.text(x + 0.13, y + h - 0.18, title, fontsize=title_size, weight="bold", va="center", color="#20252b")
        ax.text(x + 0.13, y + h - 0.53, body, fontsize=8.4, va="top", color="#424a53", linespacing=1.35)
        return patch

    def arrow(x1, y1, x2, y2, text=None, color="#59636e", bend=0.0):
        arr = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=13,
            lw=1.5,
            color=color,
            connectionstyle=f"arc3,rad={bend}",
        )
        ax.add_patch(arr)
        if text:
            ax.text(
                (x1 + x2) / 2,
                (y1 + y2) / 2 + 0.12,
                text,
                fontsize=8.5,
                ha="center",
                color=color,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.0},
            )

    ax.text(0.25, 6.82, "Hierarchical Stair Navigation: Perception, Navigation and Frozen Locomotion", fontsize=18, weight="bold", color="#1f2328")
    ax.text(0.25, 6.46, "Oracle/Teacher provide training supervision; deployment keeps Depth -> 21 rays -> 350-D actor -> 3-D command -> 12 joints", fontsize=10.5, color="#59636e")

    box(0.3, 4.2, 2.3, 1.45, "Depth Perception", "160 x 90 clean depth\nDepthRayNet\n10 Hz deploy / 2 Hz train", BLUE)
    box(3.15, 4.2, 2.5, 1.45, "Rays & Observation", "21 forward rays\n12 proprio + 21 rays + 2 goal\n10-frame stack = 350-D", TEAL)
    box(6.2, 4.2, 2.55, 1.45, "Navigation Actor", "DifferentiableSafeActorCritic\n[vx, vy, yaw_rate]\n4-class passability head", PURPLE)
    box(9.3, 4.2, 2.55, 1.45, "Safety Interface", "CBF / passability gate\nlow-pass filter + clipping\nseparate intervention logs", ORANGE)
    box(12.4, 4.2, 2.8, 1.45, "Frozen Locomotion", "45-D proprio -> 12 actions\nTorchScript contract check\n50 Hz", GREEN)
    box(15.75, 4.2, 1.45, 1.45, "Go2", "scale 0.25\nKp = 30\nKd = 0.75", RED, title_size=10.5)

    arrow(2.6, 4.93, 3.15, 4.93)
    arrow(5.65, 4.93, 6.2, 4.93)
    arrow(8.75, 4.93, 9.3, 4.93)
    arrow(11.85, 4.93, 12.4, 4.93)
    arrow(15.2, 4.93, 15.75, 4.93)

    box(0.9, 1.55, 3.6, 1.35, "Local-goal Teacher", "direct-first, then AABB corners/lanes\nanalytic segment-AABB + chunking\nreplan each step with stair constraints", BLUE)
    box(5.1, 1.55, 3.0, 1.35, "BC / DAgger", "action MSE + passability CE\n8192-env Oracle pretraining\ngradual actor rollout takeover", PURPLE)
    box(8.9, 1.55, 3.0, 1.35, "PPO & Curriculum A-G", "flat -> one-way stair -> full room\nOracle -> depth_predicted\nauxiliary loss inside PPO", ORANGE)
    box(12.7, 1.55, 4.0, 1.35, "Strict Evaluation", "success / stair_stuck / contact / timeout\nobstacle, foot clearance, goal hold\nper-episode CSV/JSON + videos", GREEN)

    arrow(4.5, 2.23, 5.1, 2.23)
    arrow(8.1, 2.23, 8.9, 2.23)
    arrow(11.9, 2.23, 12.7, 2.23)
    arrow(6.55, 2.9, 7.25, 4.2, "actor init", PURPLE, -0.05)
    arrow(10.4, 2.9, 10.55, 4.2, "online update", ORANGE, 0.03)
    arrow(2.3, 4.2, 2.3, 2.9, "oracle rays / labels", BLUE)

    ax.text(0.35, 0.55, "Interface boundary: navigation only outputs 3-D commands; observation layout, joint order, PD gains, frequency and model hash are runtime-validated.", fontsize=10.2, color="#30363d")
    save(fig, "architecture_overview.png")


def locomotion_gate_figure():
    gate_dir = "training/legged_gym/logs/Go2_blind_stair_loco_forward_finetune/branches/from_2800_forward_lr1e4_0731a/gates"
    iterations = [50, 100, 150, 200, 250]
    gates = [load_json(f"{gate_dir}/gate_{value:04d}.json") for value in iterations]
    labels = [f"Up\n{speed:.2f}" for speed in (0.25, 0.40, 0.55, 0.70)] + [f"Down\n{speed:.2f}" for speed in (0.25, 0.40, 0.55, 0.70)]
    matrix = np.asarray([[group["success_rate"] for group in gate["groups"]] for gate in gates])
    average = matrix.mean(axis=1)
    minimum = matrix.min(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1, 1.75]})
    ax = axes[0]
    ax.plot(iterations, average * 100, marker="o", lw=2.4, color=BLUE, label="Mean of 8 groups")
    ax.plot(iterations, minimum * 100, marker="s", lw=2.1, color=ORANGE, label="Worst speed group")
    ax.axhline(95, color=GREEN, ls="--", lw=1.4, label="Per-group target: 95%")
    for x, y in zip(iterations, average * 100):
        ax.text(x, y + 2.2, f"{y:.1f}%", ha="center", fontsize=8.5, color=BLUE)
    ax.set_title("Forward-Descending Locomotion Gate")
    ax.set_xlabel("Fine-tuning iteration")
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(-3, 106)
    ax.legend(frameon=False, fontsize=8.7, loc="lower right")
    style_axis(ax)

    ax = axes[1]
    image = ax.imshow(matrix * 100, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_title("8 Strict Fixed-command Groups per Checkpoint (20 trials/cell)")
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_yticks(np.arange(len(iterations)), [str(value) for value in iterations])
    ax.set_xlabel("Direction / speed (m/s)")
    ax.set_ylabel("Fine-tuning iteration")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col] * 100
            ax.text(col, row, f"{value:.0f}", ha="center", va="center", fontsize=8.4, color="white" if value < 55 else "#182026", weight="bold")
    cbar = fig.colorbar(image, ax=ax, fraction=0.026, pad=0.02)
    cbar.set_label("Success rate (%)")
    fig.suptitle("Frozen Locomotion Selection: 5 Checkpoints and 800 Strict Gate Trials", fontsize=15, weight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "locomotion_gate_progress.png")


def minimal_navigation_figure():
    summary = load_json("training/legged_gym/logs/Go2_pos_stairs_minimal/08_01_03-34-44_/eval_unseen_20x20.json")
    omni = load_json("training/legged_gym/logs/Go2_pos_stairs_minimal/omnidirectional_locomotion_probe_model250/summary.json")
    directions = [summary["directions"]["up"], summary["directions"]["down"]]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), gridspec_kw={"width_ratios": [0.75, 1.35, 1.2]})
    ax = axes[0]
    values = [item["safe_success_rate"] * 100 for item in directions]
    bars = ax.bar(["Up", "Down"], values, color=[BLUE, GREEN], width=0.58)
    ax.set_title("Unseen-seed Safe Success")
    ax.set_ylabel("Safe success rate (%)")
    ax.set_ylim(0, 112)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.0f}%\n(20/20)", ha="center", fontsize=9)
    style_axis(ax)

    ax = axes[1]
    categories = ["Diagonal", "vx/vy/yaw\nall nonzero", "Stair-stage\ndiagonal"]
    up = [directions[0]["mean_diagonal_command_step_rate"], directions[0]["mean_full_3d_command_step_rate"], directions[0]["mean_stair_diagonal_command_step_rate"]]
    down = [directions[1]["mean_diagonal_command_step_rate"], directions[1]["mean_full_3d_command_step_rate"], directions[1]["mean_stair_diagonal_command_step_rate"]]
    x = np.arange(len(categories))
    width = 0.34
    ax.bar(x - width / 2, np.asarray(up) * 100, width, label="Up", color=BLUE)
    ax.bar(x + width / 2, np.asarray(down) * 100, width, label="Down", color=GREEN)
    ax.set_xticks(x, categories)
    ax.set_ylabel("Command step ratio (%)")
    ax.set_title("Joint Command Behavior")
    ax.legend(frameon=False)
    style_axis(ax)

    ax = axes[2]
    command = []
    response = []
    axis_name = []
    for case in omni["cases"]:
        for key, mean_key, name in (("command_vx", "mean_vx", "vx"), ("command_vy", "mean_vy", "vy"), ("command_yaw_rate", "mean_yaw_rate", "yaw")):
            command.append(case[key])
            response.append(case[mean_key])
            axis_name.append(name)
    command = np.asarray(command)
    response = np.asarray(response)
    for name, color, marker in (("vx", BLUE, "o"), ("vy", GREEN, "s"), ("yaw", ORANGE, "^")):
        mask = np.asarray(axis_name) == name
        ax.scatter(command[mask], response[mask], color=color, marker=marker, s=48, label=name, alpha=0.9)
    limit = 0.52
    ax.plot([-limit, limit], [-limit, limit], color=GRAY, ls="--", lw=1.2, label="Ideal")
    ax.axhline(0, color=LIGHT_GRAY, lw=0.8)
    ax.axvline(0, color=LIGHT_GRAY, lw=0.8)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_xlabel("Command")
    ax.set_ylabel("Measured mean")
    ax.set_title("Frozen Locomotion Response (6 cases)")
    ax.legend(frameon=False, ncol=2, fontsize=8.5)
    style_axis(ax, "both")

    fig.suptitle("Minimal Navigation Baseline: 40/40 Strict Success with Simultaneous 3-D Commands", fontsize=15, weight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "minimal_navigation_results.png")


def depth_perception_figure():
    # These values are recorded in docs/depth_passability_progress.md because
    # the early DepthRayNet trainer did not persist a standalone history file.
    models = ["clean", "clean weighted", "balanced weighted"]
    overall_mae = [0.4750, 0.5022, 0.2917]
    near_names = ["clean", "clean weighted"]
    near_mae = [0.2516, 0.2125]
    near_recall = [0.8498, 0.8764]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    ax = axes[0]
    bars = ax.bar(models, overall_mae, color=[BLUE, ORANGE, GREEN], width=0.62)
    ax.set_title("DepthRayNet group-disjoint test MAE")
    ax.set_ylabel("MAE (m, lower is better)")
    ax.set_ylim(0, 0.58)
    for bar, value in zip(bars, overall_mae):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.4f}", ha="center", fontsize=9)
    style_axis(ax)

    ax = axes[1]
    x = np.arange(len(near_names))
    bars = ax.bar(x, near_mae, color=[BLUE, ORANGE], width=0.52, label="Near-obstacle MAE")
    ax.set_xticks(x, near_names)
    ax.set_ylabel("<1.0 m near-obstacle MAE (m)")
    ax.set_ylim(0, 0.30)
    ax.set_title("Weighted Loss Improves Near-obstacle Recall")
    for bar, value in zip(bars, near_mae):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.008, f"{value:.4f}", ha="center", fontsize=9)
    ax2 = ax.twinx()
    ax2.plot(x, np.asarray(near_recall) * 100, color=GREEN, marker="o", lw=2.3, label="Near-obstacle recall")
    ax2.set_ylabel("Near-obstacle recall (%)", color=GREEN)
    ax2.tick_params(axis="y", colors=GREEN)
    ax2.set_ylim(82, 90)
    for px, value in zip(x, near_recall):
        ax2.text(px + 0.04, value * 100 + 0.58, f"{value * 100:.2f}%", ha="center", fontsize=8.5, color=GREEN)
    style_axis(ax)
    fig.suptitle("Depth Dataset and Perception Models: 5,056 Base Samples plus Direction-balanced Data", fontsize=15, weight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "depth_perception_results.png")


def depth_policy_figure():
    paths = [
        ("Oracle-100", "training/legged_gym/logs/Go2_pos_depth_stairs_passability/08_01_07-22-09_/eval_oracle_fixed.json"),
        ("Depth-10", "training/legged_gym/logs/Go2_pos_depth_stairs_passability/08_01_08-17-00_/eval_predicted_fixed.json"),
        ("D20@2Hz", "training/legged_gym/logs/Go2_pos_depth_stairs_passability/08_01_08-47-43_/eval_model20_2hz.json"),
        ("D100@2Hz", "training/legged_gym/logs/Go2_pos_depth_stairs_passability/08_01_08-53-24_/eval_model100_2hz.json"),
        ("D100@10Hz", "training/legged_gym/logs/Go2_pos_depth_stairs_passability/08_01_08-53-24_/eval_model100_10hz.json"),
    ]
    names = [name for name, _ in paths]
    data = [load_json(path) for _, path in paths]
    cleared = np.asarray([item["fully_cleared_rate"] for item in data]) * 100
    low_collision = np.asarray([item["avg_low_obstacle_collision_count"] for item in data])
    total_collision = np.asarray([item["avg_total_collision_count"] for item in data])
    mae = np.asarray([np.nan if item["perception_mode"] == "oracle" else item["mean_depth_ray_mae"] for item in data])

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7))
    x = np.arange(len(names))
    ax = axes[0]
    bars = ax.bar(x, cleared, color=[GRAY, BLUE, TEAL, ORANGE, GREEN], width=0.65)
    ax.set_xticks(x, names)
    ax.set_ylabel("Fully-cleared rate (%)")
    ax.set_ylim(0, 62)
    ax.set_title("Bidirectional Smoke: Full Stair Clearance")
    for bar, value in zip(bars, cleared):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.8, f"{value:.0f}%", ha="center", fontsize=8.5)
    style_axis(ax)

    ax = axes[1]
    width = 0.35
    ax.bar(x - width / 2, total_collision, width, color=RED, label="Total collisions")
    ax.bar(x + width / 2, low_collision, width, color=ORANGE, label="Low-obstacle collisions")
    ax.set_xticks(x, names)
    ax.set_title("Collision Reduction During Adaptation")
    ax.set_ylabel("Mean collisions per episode")
    ax.legend(frameon=False, fontsize=8.5)
    style_axis(ax)

    ax = axes[2]
    valid = ~np.isnan(mae)
    bars = ax.bar(x[valid], mae[valid], color=[BLUE, TEAL, ORANGE, GREEN], width=0.65)
    ax.set_xticks(x, names)
    ax.set_title("Online 21-ray Prediction Error")
    ax.set_ylabel("Mean ray MAE (m)")
    ax.set_ylim(0, 1.02)
    ax.text(0, 0.04, "Oracle\nN/A", ha="center", va="bottom", fontsize=8.5, color=GRAY)
    for bar, value in zip(bars, mae[valid]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.3f}", ha="center", fontsize=8.2)
    style_axis(ax)

    fig.suptitle("Depth-predicted Adaptation: Rendering Rate, Perception Error and Control Diagnostics", fontsize=15, weight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "depth_policy_adaptation.png")


def teacher_evolution_figure():
    stages = [
        (
            "Original geometry",
            "training/legged_gym/logs/Go2_pos_depth_stairs_passability/teacher_gate_up_50.json",
            "training/legged_gym/logs/Go2_pos_depth_stairs_passability/teacher_gate_down_50.json",
        ),
        (
            "Local planner v2",
            "training/legged_gym/logs/Go2_pos_depth_stairs_passability/teacher_gate_up_50_v2.json",
            "training/legged_gym/logs/Go2_pos_depth_stairs_passability/teacher_gate_down_50_v2.json",
        ),
        (
            ".90 + wall start",
            "artifacts/depth_passability/teacher_gate/wall09_goal_mid_up_50.json",
            "artifacts/depth_passability/teacher_gate/wall09_goal_mid_down_50.json",
        ),
        (
            "Redistributed",
            "artifacts/depth_passability/teacher_gate/redistributed2_up_50.json",
            "artifacts/depth_passability/teacher_gate/redistributed2_down_50.json",
        ),
    ]

    def teacher_payload(path):
        return load_json(path)["modes"]["teacher"]

    up = [teacher_payload(stage[1]) for stage in stages]
    down = [teacher_payload(stage[2]) for stage in stages]
    names = [stage[0] for stage in stages]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.1), gridspec_kw={"width_ratios": [1.08, 1.25]})
    ax = axes[0]
    x = np.arange(len(stages))
    width = 0.34
    up_rate = [item["safe_success_rate"] * 100 for item in up]
    down_rate = [item["safe_success_rate"] * 100 for item in down]
    ax.bar(x - width / 2, up_rate, width, color=BLUE, label="Up")
    ax.bar(x + width / 2, down_rate, width, color=GREEN, label="Down")
    ax.set_xticks(x, names)
    ax.set_ylabel("Safe success rate (%)")
    ax.set_ylim(0, 112)
    ax.set_title("Major Teacher Gates (50 trials/direction)")
    ax.legend(frameon=False)
    for px, value in zip(x - width / 2, up_rate):
        ax.text(px, value + 2, f"{value:.0f}", ha="center", fontsize=8.5)
    for px, value in zip(x + width / 2, down_rate):
        ax.text(px, value + 2, f"{value:.0f}", ha="center", fontsize=8.5)
    style_axis(ax)

    ax = axes[1]
    failure_names = ["Orig-Up", "Orig-Down", "v2-Up", "v2-Down"]
    payloads = [up[0], down[0], up[1], down[1]]
    keys = ["success", "stair_stuck", "fall_or_contact", "timeout", "other_stuck"]
    colors = [GREEN, ORANGE, RED, GRAY, PURPLE]
    labels = ["success", "stair stuck", "fall/contact", "timeout", "other stuck"]
    bottom = np.zeros(len(payloads))
    for key, color, label in zip(keys, colors, labels):
        values = np.asarray([item["terminal_reason_counts"].get(key, 0) for item in payloads])
        ax.bar(failure_names, values, bottom=bottom, color=color, label=label, width=0.62)
        bottom += values
    ax.set_ylabel("Episode count")
    ax.set_ylim(0, 55)
    ax.set_title("Failure Attribution Guided Geometry and Control Fixes")
    ax.legend(frameon=False, fontsize=7.7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    style_axis(ax)
    fig.suptitle("Teacher Evolution: From Local Geometry Failures to 100/100 Safe Episodes", fontsize=15, weight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "teacher_gate_evolution.png")


def curriculum_figure():
    stages = ["A", "B", "C", "D", "E", "F"]
    summaries = [
        load_json(f"artifacts/depth_passability/curriculum/stage_{stage}.json" if stage != "F" else "artifacts/depth_passability/curriculum/stage_F_oracle.json")
        for stage in stages
    ]
    losses = [item["teacher_loss_mean"] for item in summaries]
    teacher_steps = [item["teacher_steps"] for item in summaries]
    ppo_iterations = [item["ppo_iterations"] for item in summaries]
    rows = load_csv("training/legged_gym/logs/Go2_pos_depth_stairs_passability/08_01_15-11-38_passability_curriculum_F_oracle/train_metrics.csv")
    iterations = np.asarray([float(row["iteration"]) for row in rows])
    success = np.asarray([float(row["success"]) for row in rows]) * 100
    stair = np.asarray([float(row["stair_pass_rate"]) for row in rows]) * 100
    cleared = np.asarray([float(row["fully_cleared"]) for row in rows]) * 100

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.8))
    ax = axes[0]
    bars = ax.bar(stages, losses, color=[BLUE, TEAL, GREEN, ORANGE, RED, PURPLE], width=0.62)
    ax.set_yscale("log")
    ax.set_ylim(0.003, 0.30)
    ax.set_ylabel("Teacher loss mean (log scale)")
    ax.set_xlabel("Curriculum stage")
    ax.set_title("BC Fitting Difficulty Across Stages A-F")
    for index, (bar, loss, steps, ppo) in enumerate(zip(bars, losses, teacher_steps, ppo_iterations)):
        ax.text(bar.get_x() + bar.get_width() / 2, loss * 1.16, f"{loss:.3f}\n{steps}/{ppo}", ha="center", fontsize=7.9)
    style_axis(ax)

    ax = axes[1]
    ax.plot(iterations, stair, color=BLUE, marker="o", lw=2.1, label="stair pass")
    ax.plot(iterations, cleared, color=GREEN, marker="s", lw=2.1, label="fully cleared")
    ax.plot(iterations, success, color=ORANGE, marker="^", lw=2.1, label="aggregated success")
    ax.set_xlabel("Stage F PPO iteration")
    ax.set_ylabel("Aggregated training metric (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Full-room Oracle Training Diagnostics")
    ax.legend(frameon=False, fontsize=8.5)
    style_axis(ax)
    fig.suptitle("From Decomposed Skills to Full Room: Curriculum A-G Implemented and Executed", fontsize=15, weight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "curriculum_training.png")


def v2_trajectory_figure():
    rows = load_csv("artifacts/depth_passability/videos_teacher_preset_down_y600_630_v2/teacher_bypass_down.csv")
    path = np.asarray([[float(row["x"]), float(row["y"])] for row in rows])
    obstacles = [
        (1.45, 2.20, 0.405, 0.315), (1.45, 3.45, 0.45, 0.315), (1.45, 6.10, 0.405, 0.27),
        (2.05, 6.80, 0.36, 0.315), (2.05, 4.10, 0.405, 0.315), (2.05, 8.05, 0.45, 0.315),
        (2.65, 7.20, 0.45, 0.27), (2.65, 3.45, 0.36, 0.27), (2.65, 6.10, 0.45, 0.315),
        (3.25, 5.60, 0.405, 0.27), (3.25, 2.80, 0.36, 0.315), (3.25, 4.10, 0.45, 0.27),
        (3.85, 3.00, 0.36, 0.315), (3.85, 5.85, 0.36, 0.27), (3.85, 7.15, 0.45, 0.315),
    ]

    fig, ax = plt.subplots(figsize=(12.5, 7.2))
    ax.add_patch(Rectangle((0.35, 0.35), 9.3, 9.3, fill=False, edgecolor="#30363d", lw=2.0, label="room boundary"))
    ax.add_patch(Rectangle((4.8, 0.35), 1.5, 9.3, facecolor="#9ecae1", edgecolor=BLUE, alpha=0.34, lw=1.2, label="5-step stair run"))
    ax.add_patch(Rectangle((6.3, 0.35), 3.35, 9.3, facecolor="#d8dee4", edgecolor="none", alpha=0.28, label="high platform"))
    for index, (cx, cy, sx, sy) in enumerate(obstacles):
        ax.add_patch(Rectangle((cx - sx / 2, cy - sy / 2), sx, sy, facecolor="#7b8794", edgecolor="#30363d", alpha=0.82, lw=0.6, label="low obstacles" if index == 0 else None))

    points = path.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    collection = LineCollection(segments, cmap="viridis", norm=plt.Normalize(0, len(path) - 1), linewidth=3.0)
    collection.set_array(np.arange(len(path) - 1))
    ax.add_collection(collection)
    ax.scatter(path[0, 0], path[0, 1], s=95, color=ORANGE, edgecolor="white", linewidth=1.2, zorder=5, label="start (8.60, 6.00)")
    ax.scatter(0.70, 6.30, s=150, marker="*", color=RED, edgecolor="white", linewidth=1.0, zorder=5, label="goal (0.70, 6.30)")
    ax.scatter(path[-1, 0], path[-1, 1], s=75, color=GREEN, edgecolor="white", linewidth=1.0, zorder=5, label=f"terminal ({path[-1,0]:.2f}, {path[-1,1]:.2f})")
    ax.annotate("First small lateral bypass", xy=(4.22, 5.55), xytext=(5.0, 4.45), arrowprops={"arrowstyle": "->", "color": ORANGE}, fontsize=9, color="#424a53")
    ax.annotate("Second obstacle bypass", xy=(3.55, 5.18), xytext=(4.15, 3.8), arrowprops={"arrowstyle": "->", "color": ORANGE}, fontsize=9, color="#424a53")
    ax.annotate("Return to goal y after obstacle columns", xy=(0.70, 5.55), xytext=(1.15, 7.25), arrowprops={"arrowstyle": "->", "color": GREEN}, fontsize=9, color="#424a53")
    cbar = fig.colorbar(collection, ax=ax, fraction=0.026, pad=0.025)
    cbar.set_label("Control step")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.set_xlabel("local x (m)")
    ax.set_ylabel("local y (m)")
    ax.set_title("V2 Downward Demo: Map-aligned Two-stage Bypass with Preset Speeds", fontsize=14, weight="bold")
    ax.legend(frameon=True, framealpha=0.94, fontsize=8.5, loc="lower right", ncol=2)
    style_axis(ax, "both")
    save(fig, "v2_down_trajectory.png")


def main():
    architecture_figure()
    locomotion_gate_figure()
    minimal_navigation_figure()
    depth_perception_figure()
    depth_policy_figure()
    teacher_evolution_figure()
    curriculum_figure()
    v2_trajectory_figure()
    print(f"Generated 8 figures in {OUT}")


if __name__ == "__main__":
    main()
