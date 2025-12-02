from __future__ import annotations

import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Callable
import zipfile

import numpy as np
from scipy.spatial import cKDTree
from multiprocessing import get_context
from tqdm import tqdm

_WORKER_TREE: Optional[cKDTree] = None
_WORKER_POINTS: Optional[np.ndarray] = None


def _worker_query_chunk(chunk: tuple[int, int]) -> tuple[int, int, np.ndarray, np.ndarray]:
    start, end = chunk
    pts = _WORKER_POINTS[start:end]
    dists, idxs = _WORKER_TREE.query(pts, k=1)
    return start, end, dists.astype(np.float32, copy=False), idxs


def _run_command(cmd, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    with log_path.open("w") as log_file:
        try:
            subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, check=True, env=env)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Command {cmd[0]} failed (see {log_path})") from exc


def _load_stats_from_zip(zip_path: Path) -> Dict:
    if not zip_path.exists():
        raise FileNotFoundError(f"evo results not found: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open("stats.json") as f:
            return json.load(f)


def _load_alignment_from_zip(zip_path: Path) -> Optional[np.ndarray]:
    if not zip_path.exists():
        return None
    with zipfile.ZipFile(zip_path, "r") as zf:
        if "alignment_transformation_sim3.npy" in zf.namelist():
            with zf.open("alignment_transformation_sim3.npy") as f:
                return np.load(f)
    return None


def run_evo_ape(
    gt_tum: Path,
    est_tum: Path,
    out_dir: Path,
    plots_path: Optional[Path] = None,
) -> Tuple[Dict, Optional[np.ndarray]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "ate_results.zip"
    log_path = out_dir / "ate.log"
    cmd = [
        "evo_ape",
        "tum",
        str(gt_tum),
        str(est_tum),
        "--align",
        "--correct_scale",
        "--save_results",
        str(results_path),
        "--no_warnings",
    ]
    if plots_path is not None:
        plots_path.mkdir(parents=True, exist_ok=True)
        cmd.extend(["--save_plot", str(plots_path / "trajectory_xy.png")])
    _run_command(cmd, log_path)
    stats = _load_stats_from_zip(results_path)
    alignment = _load_alignment_from_zip(results_path)
    return stats, alignment


def run_evo_rpe(gt_tum: Path, est_tum: Path, out_dir: Path, delta_m: float = 1.0) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "rpe_results.zip"
    log_path = out_dir / "rpe.log"
    cmd = [
        "evo_rpe",
        "tum",
        str(gt_tum),
        str(est_tum),
        "--align",
        "--delta",
        str(delta_m),
        "--delta_unit",
        "m",
        "--save_results",
        str(results_path),
        "--no_warnings",
    ]
    _run_command(cmd, log_path)
    return _load_stats_from_zip(results_path)


def _grid_cells(points: np.ndarray, xy_resolution: float) -> set[tuple[int, int]]:
    if points.size == 0 or xy_resolution <= 0:
        return set()
    cells = np.floor(points[:, :2] / xy_resolution).astype(np.int64)
    return set(map(tuple, cells))


def compute_mapping_metrics(
    reconstructed: np.ndarray,
    ground_truth: np.ndarray,
    xy_resolution: float = 0.5,
    false_hit_threshold_m: float = 1.0,
    return_errors: bool = False,
    gt_tree: Optional[cKDTree] = None,
    query_chunk_size: int = 200000,
    progress_callback: Optional[Callable[[str, int], None]] = None,
) -> tuple[Dict[str, Optional[float]], Optional[np.ndarray]]:
    metrics: Dict[str, Optional[float]] = {
        "num_points": int(reconstructed.shape[0]),
        "depth_rmse": None,
        "depth_mean": None,
        "depth_std": None,
        "false_hit_pct": None,
        "nn_rmse": None,
        "chamfer": None,
        "completeness_pct": None,
    }

    if reconstructed.size == 0 or ground_truth.size == 0:
        return metrics, None

    tree_gt = gt_tree if gt_tree is not None else cKDTree(ground_truth)
    nn_dist, nn_idx = _query_tree_in_chunks(
        tree_gt, reconstructed, chunk_size=query_chunk_size, stage="rec->gt", cb=progress_callback
    )
    dz = reconstructed[:, 2] - ground_truth[nn_idx, 2]
    metrics["depth_rmse"] = float(np.sqrt(np.mean(dz ** 2)))
    metrics["depth_mean"] = float(np.mean(dz))
    metrics["depth_std"] = float(np.std(dz))
    metrics["false_hit_pct"] = float(np.mean(np.abs(dz) > false_hit_threshold_m) * 100.0)
    metrics["nn_rmse"] = float(np.sqrt(np.mean(nn_dist ** 2)))

    tree_rec = cKDTree(reconstructed)
    gt_to_rec, _ = _query_tree_in_chunks(
        tree_rec, ground_truth, chunk_size=query_chunk_size, stage="gt->rec", cb=progress_callback
    )
    chamfer = 0.5 * (np.mean(nn_dist ** 2) + np.mean(gt_to_rec ** 2))
    metrics["chamfer"] = float(chamfer)

    rec_cells = _grid_cells(reconstructed, xy_resolution)
    gt_cells = _grid_cells(ground_truth, xy_resolution)
    if gt_cells:
        metrics["completeness_pct"] = float(len(rec_cells & gt_cells) / len(gt_cells) * 100.0)

    return metrics, dz if return_errors else None


def _query_tree_in_chunks(
    tree: cKDTree,
    points: np.ndarray,
    chunk_size: int,
    stage: str,
    cb: Optional[Callable[[str, int], None]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    if points.size == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int64)
    n = points.shape[0]
    if chunk_size is None or chunk_size <= 0 or n <= chunk_size:
        print(f"[metrics] Stage {stage}: single blocking query for {n:,} points.", flush=True)
        dists, idxs = tree.query(points, k=1)
        if cb is not None:
            cb(stage, len(points))
        return dists.astype(np.float32, copy=False), idxs

    chunk_size = max(1, chunk_size)
    ranges = [(start, min(n, start + chunk_size)) for start in range(0, n, chunk_size)]
    dists = np.empty(n, dtype=np.float32)
    idxs = np.empty(n, dtype=np.int64)

    global _WORKER_TREE, _WORKER_POINTS
    _WORKER_TREE = tree
    _WORKER_POINTS = points
    ctx = get_context("fork")
    print(
        f"[metrics] Stage {stage}: dispatching {len(ranges)} chunks (size {chunk_size}) "
        f"to 6 forked workers.",
        flush=True,
    )
    if len(ranges) == 1:
        print(f"[metrics] Stage {stage}: warning – only one chunk; expect longer runtime.", flush=True)
    start_ts = time.time()
    with ctx.Pool(processes=6) as pool:
        with tqdm(total=len(ranges), desc=f"NN {stage}", leave=False) as pbar:
            for start, end, chunk_d, chunk_i in pool.imap_unordered(_worker_query_chunk, ranges):
                dists[start:end] = chunk_d
                idxs[start:end] = chunk_i
                if cb is not None:
                    cb(stage, end - start)
                pbar.update(1)
    print(
        f"[metrics] Stage {stage}: completed {len(ranges)} chunks in {time.time() - start_ts:.2f}s.",
        flush=True,
    )
    _WORKER_TREE = None
    _WORKER_POINTS = None
    return dists, idxs


def detection_rate(num_points: int, ping_count: int, beam_count: int) -> float:
    denom = max(1, ping_count * beam_count)
    return float(num_points) / float(denom) * 100.0
