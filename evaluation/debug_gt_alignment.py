from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from . import plots


def debug_alignment(run_dir: Path, out_prefix: Optional[str] = None) -> None:
    """
    Load mapping point clouds for a run and generate simple alignment plots.
    """
    run_dir = Path(run_dir)
    npz_path = run_dir / "metrics" / "mapping" / "point_clouds.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Mapping point_clouds.npz not found in {npz_path}")
    data = np.load(npz_path)
    gt = data["ground_truth"]
    rec = data["reconstructed"]

    out_prefix = out_prefix or "debug_gt_alignment"
    plots.plot_point_clouds(gt, rec, run_dir / "plots" / f"{out_prefix}.png")

    print(
        f"[debug_gt_alignment] reconstructed: {rec.shape[0]:,} pts, "
        f"gt: {gt.shape[0]:,} pts."
    )
    if gt.size and rec.size:
        mean_rec_z = float(rec[:, 2].mean())
        mean_gt_z = float(gt[:, 2].mean())
        dz_mean = mean_rec_z - mean_gt_z
        print(
            f"[debug_gt_alignment] mean rec z={mean_rec_z:.3f} m, "
            f"mean gt z={mean_gt_z:.3f} m, Δz_mean={dz_mean:.3f} m"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Debug GT alignment for a single run_dir.")
    parser.add_argument("run_dir", type=Path, help="Path to evaluation run directory (e.g., .../map_voxel_050).")
    parser.add_argument("--prefix", type=str, default=None, help="Output filename prefix for plots.")
    args = parser.parse_args()
    debug_alignment(args.run_dir, args.prefix)
