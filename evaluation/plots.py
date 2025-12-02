from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Brand palette (NMBU colors)
BRAND_COLORS = [
    "#008571",  # 0 teal
    "#53143A",  # 1 plum
    "#025C4F",  # 2 dark aqua
    "#E7E3D1",  # 3 warm grey
    "#F6F6EE",  # 4 cool grey
    "#5BBEAF",  # 5 light teal
    "#0D3B34",  # 6 deep green
    "#EFC3CD",  # 7 pink
    "#D4DDEF",  # 8 light blue
    "#5967A6",  # 9 indigo
    "#FFD757",  # 10 yellow
]


def _apply_brand_theme():
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": BRAND_COLORS[4],
            "axes.edgecolor": "#222222",
            "axes.labelcolor": "#222222",
            "axes.grid": True,
            "grid.color": "#D0D0D0",
            "grid.linestyle": "--",
            "grid.linewidth": 0.7,
            "text.color": "#222222",
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "lines.linewidth": 2.5,
            "legend.frameon": False,
        },
    )
    sns.set_palette(BRAND_COLORS)


# Apply theme globally for all plots
_apply_brand_theme()


def _save(fig, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_trajectory_xy(gt_xyz: np.ndarray, est_xyz: np.ndarray, out_path: Path, title: str = "Trajectory XY"):
    fig, ax = plt.subplots(figsize=(6, 5))
    if gt_xyz.size:
        ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], label="GT", linewidth=2.5, color=BRAND_COLORS[9])
    if est_xyz.size:
        ax.plot(
            est_xyz[:, 0],
            est_xyz[:, 1],
            label="SLAM",
            linewidth=2.0,
            linestyle="--",
            color=BRAND_COLORS[0],
        )
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(title)
    ax.axis("equal")
    ax.legend()
    fig.tight_layout()
    _save(fig, out_path)


def plot_point_clouds(gt_xyz: np.ndarray, rec_xyz: np.ndarray, out_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax_gt_xy, ax_gt_xz, ax_map_xy, ax_map_xz = axes.flatten()

    if gt_xyz.size:
        ax_gt_xy.scatter(gt_xyz[:, 0], gt_xyz[:, 1], s=1, c=BRAND_COLORS[9])
        ax_gt_xy.set_title("GT Top-down")
        ax_gt_xy.set_xlabel("X [m]")
        ax_gt_xy.set_ylabel("Y [m]")

        ax_gt_xz.scatter(gt_xyz[:, 0], gt_xyz[:, 2], s=1, c=BRAND_COLORS[9])
        ax_gt_xz.set_title("GT Cross-section")
        ax_gt_xz.set_xlabel("X [m]")
        ax_gt_xz.set_ylabel("Z [m]")
    else:
        ax_gt_xy.set_visible(False)
        ax_gt_xz.set_visible(False)

    if rec_xyz.size:
        ax_map_xy.scatter(rec_xyz[:, 0], rec_xyz[:, 1], s=1, c=BRAND_COLORS[0])
        ax_map_xy.set_title("Map Top-down")
        ax_map_xy.set_xlabel("X [m]")
        ax_map_xy.set_ylabel("Y [m]")

        ax_map_xz.scatter(rec_xyz[:, 0], rec_xyz[:, 2], s=1, c=BRAND_COLORS[0])
        ax_map_xz.set_title("Map Cross-section")
        ax_map_xz.set_xlabel("X [m]")
        ax_map_xz.set_ylabel("Z [m]")
    else:
        ax_map_xy.set_visible(False)
        ax_map_xz.set_visible(False)

    fig.tight_layout()
    _save(fig, out_path)


def plot_depth_histogram(depth_errors: np.ndarray, out_path: Path, bins: int = 60):
    if depth_errors.size == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(depth_errors, bins=bins, alpha=0.9, color=BRAND_COLORS[0])
    ax.set_xlabel("Depth error Δz [m]")
    ax.set_ylabel("Count")
    ax.set_title("Depth error histogram")
    fig.tight_layout()
    _save(fig, out_path)


def plot_trajectory_xyz(gt_xyz: np.ndarray, est_xyz: np.ndarray, out_path: Path, title: str = "Trajectory XYZ"):
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    if gt_xyz.size:
        ax.plot3D(
            gt_xyz[:, 0],
            gt_xyz[:, 1],
            gt_xyz[:, 2],
            label="GT",
            linewidth=2.5,
            color=BRAND_COLORS[9],
        )
    if est_xyz.size:
        ax.plot3D(
            est_xyz[:, 0],
            est_xyz[:, 1],
            est_xyz[:, 2],
            label="SLAM",
            linewidth=2.0,
            linestyle="--",
            color=BRAND_COLORS[0],
        )
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title(title)
    ax.legend()
    _save(fig, out_path)
