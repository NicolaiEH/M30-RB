from __future__ import annotations

"""Run evaluation experiments, replay bags, and compute nav and mapping metrics."""

import json
import re
import math
import os
import pickle
import signal
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

import typer
import yaml

try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pd = None

from . import bag_utils, ground_truth, metrics, plots

app = typer.Typer(add_completion=False, help="AUV-NAV evaluation toolkit.")


@dataclass
class ExperimentRun:
    group: str
    run_id: str
    label: str
    overrides: Dict[str, object] = field(default_factory=dict)
    metrics: Sequence[str] = field(default_factory=list)
    notes: Optional[str] = None


@dataclass
class GTContext:
    points: np.ndarray
    bounds: Dict[str, tuple[float, float]]
    tree: Optional[cKDTree]
    cache_file: Optional[Path]
    metadata: Dict[str, object] = field(default_factory=dict)
    origin: Optional[np.ndarray] = None
    dir_vec: Optional[np.ndarray] = None
    perp_vec: Optional[np.ndarray] = None
    s_range: Optional[tuple[float, float]] = None
    t_half: Optional[float] = None


@dataclass
class ManagedProcess:
    name: str
    cmd: List[str]
    log_path: Path
    proc: subprocess.Popen
    log_handle: object


def _load_config(path: Path) -> Dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def _resolve_path(base: Path, maybe_relative: str) -> Path:
    candidate = Path(maybe_relative)
    if not candidate.is_absolute():
        candidate = (base / candidate).resolve()
    return candidate


def _load_scenario_translation(config_dir: Path, gt_cfg: Dict) -> np.ndarray:
    scenario_rel = gt_cfg.get(
        "scenario_config",
        "../ros2_ws/src/holoocean_bridge/config/custom_scenario.json",
    )
    repo_root = config_dir.parent
    scenario_path = _resolve_path(repo_root, scenario_rel)
    if not scenario_path.exists():
        raise FileNotFoundError(f"Scenario config not found: {scenario_path}")
    data = json.loads(scenario_path.read_text())
    agents = data.get("agents", [])
    auv = None
    for agent in agents:
        if agent.get("agent_name") == "auv":
            auv = agent
            break
    if auv is None and agents:
        auv = agents[0]
    if auv is None:
        raise RuntimeError(f"No agents defined in scenario config {scenario_path}")
    loc = auv.get("location", [0.0, 0.0, 0.0])
    if len(loc) != 3:
        raise RuntimeError(f"Unexpected location format in {scenario_path}")
    return np.array(loc, dtype=float)


def _select_runs(config: Dict, groups: Sequence[str] | None, run_ids: Sequence[str] | None) -> List[ExperimentRun]:
    experiments = []
    run_filter = set(run_ids or [])
    group_filter = set(groups or [])
    seen_ids = set()
    for group_name, data in config.get("experiments", {}).items():
        if group_filter and group_name not in group_filter:
            continue
        metrics_cfg = data.get("metrics", [])
        for run in data.get("runs", []):
            if run.get("disabled"):
                continue
            run_id = run["id"]
            if run_filter and run_id not in run_filter:
                continue
            overrides = run.get("overrides", {})
            experiments.append(
                ExperimentRun(
                    group=group_name,
                    run_id=run_id,
                    label=run.get("label", run_id),
                    overrides=overrides,
                    metrics=metrics_cfg,
                    notes=run.get("notes"),
                )
            )
            seen_ids.add(run_id)
    if run_filter and run_filter - seen_ids:
        missing = ", ".join(sorted(run_filter - seen_ids))
        raise ValueError(f"Unknown run ids requested: {missing}")
    return experiments


def _format_launch_args(args: Dict[str, object]) -> List[str]:
    formatted = []
    for key, value in args.items():
        if isinstance(value, bool):
            val = "true" if value else "false"
        else:
            val = str(value)
        formatted.append(f"{key}:={val}")
    return formatted


def _clean_label(label: str) -> str:
    cleaned = re.sub(r"\([^)]*\)", "", label)
    cleaned = cleaned.replace(" m", " ")
    return " ".join(cleaned.split()).strip()


def _format_plot_title(base: str, label: str) -> str:
    pretty_label = _clean_label(label)
    if not pretty_label:
        return base
    return f"{base} {pretty_label}"


def _start_process(name: str, cmd: List[str], log_path: Path, env: Optional[Dict[str, str]] = None) -> ManagedProcess:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w")
    proc = subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT, env=env)
    return ManagedProcess(name=name, cmd=cmd, log_path=log_path, proc=proc, log_handle=log_handle)


def _stop_process(mp: ManagedProcess, timeout: float = 10.0):
    if mp.proc.poll() is None:
        mp.proc.send_signal(signal.SIGINT)
        try:
            mp.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            mp.proc.terminate()
            try:
                mp.proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                mp.proc.kill()
                mp.proc.wait()
    mp.log_handle.close()


GT_TREE_BYTES_PER_POINT = 512


def _tree_cache_path(cache_file: Optional[Path]) -> Optional[Path]:
    if cache_file is None:
        return None
    return cache_file.with_suffix(cache_file.suffix + ".tree.pkl")


def _estimate_tree_bytes(num_points: int) -> int:
    return int(num_points) * GT_TREE_BYTES_PER_POINT


def _random_subsample(points: np.ndarray, target_count: int, seed: int) -> np.ndarray:
    if points.shape[0] <= target_count or target_count <= 0:
        return points
    rng = np.random.default_rng(seed)
    keep_prob = target_count / float(points.shape[0])
    mask = rng.random(points.shape[0]) < keep_prob
    reduced = points[mask]
    if reduced.shape[0] > target_count:
        reduced = reduced[:target_count]
    return reduced


def _load_or_build_kdtree(points: np.ndarray, cache_file: Optional[Path]) -> tuple[Optional[cKDTree], bool]:
    if points.size == 0:
        return None, False
    tree_path = _tree_cache_path(cache_file)
    if tree_path and tree_path.exists():
        with tree_path.open("rb") as f:
            state = pickle.load(f)
        placeholder = cKDTree(np.zeros((1, 3)))
        placeholder.__setstate__(state)
        return placeholder, True
    tree = cKDTree(points, balanced_tree=False, compact_nodes=True)
    if tree_path:
        with tree_path.open("wb") as f:
            pickle.dump(tree.__getstate__(), f, protocol=4)
    return tree, False


def _postprocess_nav_metrics(
    run_dir: Path,
    recorded_bag: Path,
    config_defaults: Dict,
    evo_required: bool,
    run_label: str,
) -> tuple[Dict, Optional[np.ndarray]]:
    metrics_dir = run_dir / "metrics" / "nav"
    traj_dir = run_dir / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    gt_topic = config_defaults["sensor_topics"]["gt_odom"]
    slam_topic = "/slam/odom"
    gt_tum = traj_dir / "gt.tum"
    est_tum = traj_dir / "slam.tum"
    gt_xyz = bag_utils.export_odometry_to_tum(recorded_bag, gt_topic, gt_tum)
    est_xyz = bag_utils.export_odometry_to_tum(recorded_bag, slam_topic, est_tum)
    est_xyz_raw = est_xyz.copy()

    nav_metrics = {}
    alignment = None
    if evo_required:
        try:
            ate_stats, alignment = metrics.run_evo_ape(gt_tum, est_tum, metrics_dir, plots_path=run_dir / "plots")
            nav_metrics["ate"] = ate_stats
        except RuntimeError as exc:
            typer.echo(f"[{run_dir.name}] Warning: evo_ape failed ({exc}); skipping ATE stats.")
        try:
            nav_metrics["rpe"] = metrics.run_evo_rpe(gt_tum, est_tum, metrics_dir)
        except RuntimeError as exc:
            typer.echo(f"[{run_dir.name}] Warning: evo_rpe failed ({exc}); skipping RPE stats.")
    est_aligned = est_xyz.copy()
    if alignment is not None and est_xyz.size:
        est_aligned = _apply_sim3(est_xyz, alignment)
    plots_dir = run_dir / "plots"
    label_suffix = run_label.strip() or run_dir.name
    plots.plot_trajectory_xy(
        gt_xyz,
        est_aligned,
        plots_dir / "trajectory_xy.png",
        title=_format_plot_title("Trajectory XY", label_suffix),
    )
    plots.plot_trajectory_xyz(
        gt_xyz,
        est_aligned,
        plots_dir / "trajectory_xyz.png",
        title=_format_plot_title("Trajectory XYZ", label_suffix),
    )
    nav_metrics["trajectory_points"] = {
        "gt_count": int(gt_xyz.shape[0]),
        "est_count": int(est_xyz.shape[0]),
    }
    nav_metrics["gt_path_bounds"] = {
        "x": [float(gt_xyz[:, 0].min()), float(gt_xyz[:, 0].max())] if gt_xyz.size else [0.0, 0.0],
        "y": [float(gt_xyz[:, 1].min()), float(gt_xyz[:, 1].max())] if gt_xyz.size else [0.0, 0.0],
    }
    return nav_metrics, alignment


def _apply_sim3(points: np.ndarray, sim3: np.ndarray) -> np.ndarray:
    if points.size == 0 or sim3.shape != (4, 4):
        return points
    homo = np.ones((points.shape[0], 4), dtype=float)
    homo[:, :3] = points
    transformed = (sim3 @ homo.T).T
    return transformed[:, :3]


def _crop_points_xy(points: np.ndarray, bounds: Dict[str, tuple[float, float]]) -> np.ndarray:
    if points.size == 0:
        return points
    mask = (
        (points[:, 0] >= bounds["x"][0])
        & (points[:, 0] <= bounds["x"][1])
        & (points[:, 1] >= bounds["y"][0])
        & (points[:, 1] <= bounds["y"][1])
    )
    return points[mask]


def _apply_corridor_window(points: np.ndarray, ctx: GTContext) -> np.ndarray:
    if (
        points.size == 0
        or ctx.origin is None
        or ctx.dir_vec is None
        or ctx.perp_vec is None
        or ctx.s_range is None
        or ctx.t_half is None
    ):
        return points
    origin = ctx.origin
    dir_vec = ctx.dir_vec
    perp_vec = ctx.perp_vec
    s_min, s_max = ctx.s_range
    t_half = ctx.t_half
    rel = points[:, :2] - origin
    s = rel @ dir_vec
    t = rel @ perp_vec
    mask = (s >= s_min) & (s <= s_max) & (np.abs(t) <= t_half)
    return points[mask]


def _project_s(points_xy: np.ndarray, ctx: GTContext) -> np.ndarray:
    rel = points_xy - ctx.origin
    return rel @ ctx.dir_vec


def _subset_gt_for_rec(
    ctx: GTContext,
    rec_points: np.ndarray,
    s_margin: float,
) -> tuple[np.ndarray, cKDTree, tuple[float, float]]:
    if rec_points.size == 0 or ctx.points.size == 0 or ctx.origin is None or ctx.dir_vec is None:
        if ctx.tree is None:
            tree = cKDTree(ctx.points, balanced_tree=False, compact_nodes=True) if ctx.points.size else None
        else:
            tree = ctx.tree
        return ctx.points, tree, ctx.s_range or (-np.inf, np.inf)

    rec_s = _project_s(rec_points[:, :2], ctx)
    s_min = float(rec_s.min() - s_margin)
    s_max = float(rec_s.max() + s_margin)
    gt_s = _project_s(ctx.points[:, :2], ctx)
    mask = (gt_s >= s_min) & (gt_s <= s_max)
    subset = ctx.points[mask].copy()
    if subset.size == 0:
        subset = ctx.points.copy()
        s_min, s_max = ctx.s_range if ctx.s_range else (s_min, s_max)
    tree = cKDTree(subset, balanced_tree=False, compact_nodes=True) if subset.size else None
    return subset, tree, (s_min, s_max)


def _postprocess_mapping_metrics(
    run_dir: Path,
    recorded_bag: Path,
    gt_context: GTContext,
    config_defaults: Dict,
    ping_stats: Dict[str, int],
    alignment: Optional[np.ndarray],
) -> Dict:
    points_topic = config_defaults["sensor_topics"]["profiling_points"]
    label = run_dir.name
    chunk_size = int(config_defaults["ground_truth"].get("query_chunk_size", 200000))
    rec_margin = float(config_defaults["ground_truth"].get("corridor_rec_margin_m", 5.0))
    with typer.progressbar(length=4, label=f"[{label}] Mapping metrics") as progress:
        rec_points = bag_utils.collect_point_cloud_points(recorded_bag, points_topic)
        progress.update(1)
        if alignment is not None and rec_points.size:
            rec_points = _apply_sim3(rec_points, alignment)
        rec_points = _crop_points_xy(rec_points, gt_context.bounds)
        rec_points = _apply_corridor_window(rec_points, gt_context)
        gt_points = gt_context.points
        gt_tree = gt_context.tree
        if rec_points.size:
            gt_points, gt_tree, rec_s_range = _subset_gt_for_rec(gt_context, rec_points, rec_margin)
            typer.echo(
                f"[{label}] GT subset for rec span {rec_s_range[0]:.1f}->{rec_s_range[1]:.1f} m: {gt_points.shape[0]:,} pts."
            )
        progress.update(1)
        grid = config_defaults["ground_truth"]["completeness_grid_m"]
        false_thr = config_defaults["ground_truth"]["false_hit_threshold_m"]
        nn_batches = 0
        if chunk_size > 0:
            nn_batches = math.ceil(max(1, rec_points.shape[0]) / chunk_size)
            nn_batches += math.ceil(max(1, gt_points.shape[0]) / chunk_size)
        typer.echo(
            f"[{label}] NN setup -> rec: {rec_points.shape[0]:,}, gt: {gt_points.shape[0]:,}, "
            f"chunk={chunk_size}, batches~{nn_batches}."
        )
        progress.update(1)
        if nn_batches > 0:
            with typer.progressbar(length=nn_batches, label=f"[{label}] NN queries") as nn_bar:
                def _update(_stage: str, _count: int):
                    nn_bar.update(1)

                mapping_metrics, depth_errors = metrics.compute_mapping_metrics(
                    rec_points,
                    gt_points,
                    grid,
                    false_thr,
                    return_errors=True,
                    gt_tree=gt_tree,
                    query_chunk_size=chunk_size,
                    progress_callback=_update,
                )
        else:
            mapping_metrics, depth_errors = metrics.compute_mapping_metrics(
                rec_points,
                gt_points,
                grid,
                false_thr,
                return_errors=True,
                gt_tree=gt_tree,
                query_chunk_size=chunk_size,
            )
        mapping_metrics["gt_points_used"] = int(gt_points.shape[0])
        mapping_dir = run_dir / "metrics" / "mapping"
        mapping_dir.mkdir(parents=True, exist_ok=True)
        mapping_metrics["detection_rate"] = metrics.detection_rate(
            mapping_metrics["num_points"], ping_stats["count"], ping_stats["width"]
        )
        np.savez_compressed(mapping_dir / "point_clouds.npz", reconstructed=rec_points, ground_truth=gt_points)
        plots.plot_point_clouds(gt_points, rec_points, run_dir / "plots" / "map_cloud.png")
        if depth_errors is not None:
            plots.plot_depth_histogram(depth_errors, run_dir / "plots" / "depth_hist.png")
        progress.update(1)
    typer.echo(
        f"[{label}] Mapping stats computed for {rec_points.shape[0]:,} rec pts vs "
        f"{gt_context.points.shape[0]:,} GT pts."
    )
    return mapping_metrics


def _measure_ping_stats(bag_path: Path, sensor_topic: str) -> Dict[str, int]:
    stats = bag_utils.count_image_dimensions(bag_path, sensor_topic)
    if stats["count"] == 0:
        raise RuntimeError(f"No profiling images found on topic {sensor_topic}")
    return stats


def _prepare_gt_points(
    config_defaults: Dict,
    bag_path: Path,
    results_root: Path,
    config_dir: Path,
) -> GTContext:
    gt_cfg = config_defaults["ground_truth"]
    unit_scale = float(gt_cfg.get("unit_scale", 1.0))
    if unit_scale <= 0:
        raise ValueError("ground_truth.unit_scale must be > 0")
    max_memory_gb = float(gt_cfg.get("max_memory_gb", 48.0))
    downsample_seed = int(gt_cfg.get("downsample_seed", 0))
    xy_margin = float(gt_cfg.get("xy_margin_m", 20.0))
    z_margin = float(gt_cfg.get("z_margin_m", 10.0))
    long_margin = float(gt_cfg.get("corridor_long_margin_m", xy_margin))
    half_width = float(gt_cfg.get("corridor_half_width_m", xy_margin))
    scenario_translation = _load_scenario_translation(config_dir, gt_cfg)
    gt_topic = config_defaults["sensor_topics"]["gt_odom"]
    traj_xyz = bag_utils.extract_odometry_positions(bag_path, gt_topic)
    if traj_xyz.size == 0:
        raise RuntimeError(f"No poses found on {gt_topic} in {bag_path}")
    xy = traj_xyz[:, :2]
    origin = xy.mean(axis=0)
    centered = xy - origin
    if centered.shape[0] >= 2 and np.any(centered):
        cov = np.cov(centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        dir_vec = eigvecs[:, int(np.argmax(eigvals))]
        dir_vec = dir_vec / np.linalg.norm(dir_vec)
    else:
        dir_vec = np.array([1.0, 0.0], dtype=float)
    perp_vec = np.array([-dir_vec[1], dir_vec[0]], dtype=float)
    s_vals = centered @ dir_vec
    s_range = (
        float(s_vals.min()) - long_margin,
        float(s_vals.max()) + long_margin,
    )
    t_half = half_width
    corners = []
    for s in (s_range[0], s_range[1]):
        for t in (-t_half, t_half):
            corner = origin + dir_vec * s + perp_vec * t
            corners.append(corner)
    corners = np.asarray(corners)
    x_bounds = (float(corners[:, 0].min()), float(corners[:, 0].max()))
    y_bounds = (float(corners[:, 1].min()), float(corners[:, 1].max()))
    z_bounds = (
        float(traj_xyz[:, 2].min() - z_margin),
        float(traj_xyz[:, 2].max() + z_margin),
    )
    gt_bounds = {"x": x_bounds, "y": y_bounds, "z": z_bounds}
    filter_params = {
        "origin": origin.tolist(),
        "dir": dir_vec.tolist(),
        "perp": perp_vec.tolist(),
        "s_min": s_range[0],
        "s_max": s_range[1],
        "t_half": t_half,
    }
    cache_dir = results_root / "_gt_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    use_ply = bool(gt_cfg.get("use_ply_gt", False))
    gt_points: np.ndarray
    cache_file: Optional[Path] = None
    z_offset_m: float = 0.0
    anchor_meta: Dict[str, object] = {}

    if use_ply:
        ply_path_cfg = gt_cfg.get("ply_path", "../octree/pierharbor/min10cm_max640/pointcloud.ply")
        ply_path = _resolve_path(config_dir, ply_path_cfg)
        typer.echo(f"[GT] Using PLY ground truth from {ply_path}.")
        load_start = time.time()
        world_pts = ground_truth.load_ply_point_cloud(ply_path)
        typer.echo(
            f"[GT] Loaded PLY cloud with {world_pts.shape[0]:,} points in {time.time() - load_start:.2f}s."
        )
        xw = world_pts[:, 0]
        yw = world_pts[:, 1]
        zw = world_pts[:, 2]
        del world_pts
        anchor_cfg = gt_cfg.get("start_anchor", {})
        anchor_applied = False
        anchor_meta: Dict[str, object] = {}
        if anchor_cfg.get("enabled"):
            anchor_xy = anchor_cfg.get("xy", [float(scenario_translation[0]), float(scenario_translation[1])])
            radius = float(anchor_cfg.get("radius_m", 5.0))
            vehicle_z = float(anchor_cfg.get("vehicle_z", float(scenario_translation[2])))
            clearance = anchor_cfg.get("vehicle_clearance_m")
            if clearance is None:
                clearance = float(anchor_cfg.get("expected_vehicle_height_m", 0.0))
            desired_seafloor = vehicle_z - clearance
            mask_anchor = ((xw - anchor_xy[0]) ** 2 + (yw - anchor_xy[1]) ** 2) <= radius ** 2
            if np.any(mask_anchor):
                raw_anchor = float(np.median(zw[mask_anchor]))
                z_offset_m = desired_seafloor - raw_anchor
                zw = zw + z_offset_m
                anchor_applied = True
                anchor_meta = {
                    "xy": anchor_xy,
                    "radius_m": radius,
                    "vehicle_z": vehicle_z,
                    "clearance_m": clearance,
                    "desired_seafloor_z": desired_seafloor,
                    "raw_anchor_z": raw_anchor,
                    "z_offset": z_offset_m,
                }
                typer.echo(
                    f"[GT] Anchor seafloor raw {raw_anchor:.2f} -> target {desired_seafloor:.2f} "
                    f"(dz={z_offset_m:.2f})."
                )
            else:
                typer.echo(
                    f"[GT] Warning: no GT anchor points within radius {radius} m of {anchor_xy}; "
                    "falling back to mean alignment."
                )

        corridor_mask = ground_truth._apply_corridor_filter(  # type: ignore[attr-defined]
            xw.astype(np.float32),
            yw.astype(np.float32),
            {
                "origin": origin.tolist(),
                "dir": dir_vec.tolist(),
                "perp": perp_vec.tolist(),
                "s_min": s_range[0],
                "s_max": s_range[1],
                "t_half": t_half,
            },
        )
        if not np.any(corridor_mask):
            typer.echo("Ground-truth PLY crop returned 0 points inside corridor; mapping metrics will be skipped.")
            gt_points = np.zeros((0, 3), dtype=np.float32)
        else:
            x_c = xw[corridor_mask]
            y_c = yw[corridor_mask]
            z_c = zw[corridor_mask]
            del xw, yw, zw
            if not anchor_applied:
                z_gt_mean = float(traj_xyz[:, 2].mean())
                z_oct_mean = float(z_c.mean())
                z_offset_m = z_gt_mean - z_oct_mean
                typer.echo(
                    f"[GT] Estimated vertical offset dz={z_offset_m:.2f} m "
                    f"(mean gt z={z_gt_mean:.2f}, mean octree z={z_oct_mean:.2f})."
                )
                z_aligned = (z_c + z_offset_m).astype(np.float32)
            else:
                z_aligned = z_c.astype(np.float32)
            gt_points = np.stack(
                [x_c.astype(np.float32), y_c.astype(np.float32), z_aligned],
                axis=1,
            )
            del x_c, y_c, z_c, z_aligned
        cache_state = "ply"
        original_count = int(gt_points.shape[0])
        typer.echo(f"GT PLY corridor crop contains {original_count:,} points before pruning.")
    else:
        translation = np.array(gt_cfg.get("translation_m", [0.0, 0.0, 0.0]), dtype=float)
        octree_cache = _resolve_path(config_dir, gt_cfg["cache_dir"])
        if not octree_cache.exists():
            repo_root = config_dir.parent
            alt_path = _resolve_path(repo_root, gt_cfg["cache_dir"])
            if alt_path.exists():
                octree_cache = alt_path
        if not octree_cache.exists():
            raise FileNotFoundError(f"Ground-truth octree cache not found: {gt_cfg['cache_dir']}")
        cache_bounds = {
            "x": ((gt_bounds["x"][0] - translation[0]) / unit_scale, (gt_bounds["x"][1] - translation[0]) / unit_scale),
            "y": ((gt_bounds["y"][0] - translation[1]) / unit_scale, (gt_bounds["y"][1] - translation[1]) / unit_scale),
            "z": ((gt_bounds["z"][0] - translation[2]) / unit_scale, (gt_bounds["z"][1] - translation[2]) / unit_scale),
        }
        load_start = time.time()
        gt_points, cache_file = ground_truth.load_octree_crop(
            octree_cache,
            cache_bounds,
            cache_store=cache_dir,
            unit_scale=unit_scale,
            filter_params=filter_params,
            translation=tuple(translation.tolist()),
        )
        cache_state = "hit" if cache_file and cache_file.exists() else "miss"
        typer.echo(
            f"[GT] AOI crop loaded {gt_points.shape[0]:,} points in {time.time() - load_start:.2f}s "
            f"(cache={cache_state})."
        )
        original_count = int(gt_points.shape[0])
        typer.echo(f"GT crop contains {original_count:,} points before pruning.")
    metadata: Dict[str, object] = {
        "unit_scale": unit_scale,
        "original_count": original_count,
        "max_memory_gb": max_memory_gb,
        "corridor_origin": origin.tolist(),
        "corridor_dir": dir_vec.tolist(),
        "corridor_perp": perp_vec.tolist(),
        "corridor_s_range": list(s_range),
        "corridor_length_m": float(s_range[1] - s_range[0]),
        "corridor_half_width": t_half,
        "z_offset_m": z_offset_m,
        "cache_state": cache_state,
    }
    if anchor_meta:
        metadata["start_anchor"] = anchor_meta
    if cache_file:
        metadata["cache_file"] = str(cache_file)
    if gt_points.size == 0:
        typer.echo("Ground-truth crop returned 0 points; mapping metrics will be skipped.")
        return GTContext(
            gt_points,
            gt_bounds,
            None,
            cache_file,
            metadata,
            origin=origin,
            dir_vec=dir_vec,
            perp_vec=perp_vec,
            s_range=s_range,
            t_half=t_half,
        )
    budget_bytes = int(max_memory_gb * (1024 ** 3))
    estimated_bytes = _estimate_tree_bytes(gt_points.shape[0])
    if estimated_bytes > budget_bytes:
        target = max(1, budget_bytes // GT_TREE_BYTES_PER_POINT)
        typer.echo(
            f"GT crop has {original_count:,} pts (~{estimated_bytes / 1e9:.1f} GB); "
            f"downsampling to {target:,} pts to respect {max_memory_gb:g} GB budget."
        )
        gt_points = _random_subsample(gt_points, target, downsample_seed)
        metadata["downsampled_from"] = original_count
        metadata["downsampled_to"] = int(gt_points.shape[0])
    tree_start = time.time()
    tree, reused = _load_or_build_kdtree(gt_points, cache_file)
    typer.echo(
        f"[GT] {'Reused' if reused else 'Built'} KD-tree in {time.time() - tree_start:.2f}s "
        f"({gt_points.shape[0]:,} pts)."
    )
    return GTContext(
        gt_points,
        gt_bounds,
        tree,
        cache_file,
        metadata,
        origin=origin,
        dir_vec=dir_vec,
        perp_vec=perp_vec,
        s_range=s_range,
        t_half=t_half,
    )


def _run_ros_pipeline(
    run: ExperimentRun,
    bag_path: Path,
    run_dir: Path,
    config_defaults: Dict,
    launch_args: Dict[str, object],
    bag_play_args: Sequence[str],
    record_topics: Sequence[str],
    startup_delay: float,
    play_rate: float,
    play_duration: Optional[float],
) -> Tuple[float, Path]:
    logs_dir = run_dir / "logs"
    ros_env = os.environ.copy()
    ros_env.setdefault("RCUTILS_LOGGING_BUFFERED_STREAM", "0")

    launch_cmd = ["ros2", "launch", "holoocean_bridge", "slam_and_mapping.launch.py"] + _format_launch_args(launch_args)
    recorder_cmd = ["ros2", "bag", "record", "-o", str(run_dir / "rosbag2")] + list(record_topics)
    play_cmd = ["ros2", "bag", "play", str(bag_path)] + list(bag_play_args)

    typer.echo(f"[{run.run_id}] Launching SLAM pipeline...")
    launch_proc = _start_process("slam_launch", launch_cmd, logs_dir / "slam_launch.log", env=ros_env)
    time.sleep(startup_delay)

    typer.echo(f"[{run.run_id}] Recording topics...")
    record_proc = _start_process("bag_record", recorder_cmd, logs_dir / "record.log", env=ros_env)
    time.sleep(2.0)

    typer.echo(f"[{run.run_id}] Replaying bag {bag_path} ...")
    start = time.time()
    play_proc = _start_process("bag_play", play_cmd, logs_dir / "bag_play.log", env=ros_env)
    wait_timeout = None
    timeout_reason = None
    if play_duration is not None and play_duration > 0:
        expected_wall_time = play_duration
        if play_rate and play_rate > 0:
            expected_wall_time = play_duration / play_rate
        wait_timeout = max(expected_wall_time + 3.0, 3.0)
        timeout_reason = f"{play_duration:.1f}s request (~{expected_wall_time:.1f}s wall time)"

    try:
        play_proc.proc.wait(timeout=wait_timeout)
    except subprocess.TimeoutExpired:
        if timeout_reason is None:
            typer.echo(f"[{run.run_id}] Bag playback is still running; sending SIGINT.")
        else:
            typer.echo(f"[{run.run_id}] Bag playback exceeded {timeout_reason}; sending SIGINT.")
        play_proc.proc.send_signal(signal.SIGINT)
        try:
            play_proc.proc.wait(timeout=wait_timeout)
        except subprocess.TimeoutExpired:
            typer.echo(f"[{run.run_id}] ros2 bag play ignored SIGINT; forcing termination.")
            play_proc.proc.terminate()
    wall_time = time.time() - start

    _stop_process(record_proc)
    _stop_process(launch_proc)
    _stop_process(play_proc)

    recorded_bag = bag_utils.ensure_bag_path(run_dir / "rosbag2")
    return wall_time, recorded_bag


def _flatten_metrics(data: Dict) -> Dict:
    flat = {}
    for key, value in data.items():
        if isinstance(value, dict):
            nested = _flatten_metrics(value)
            for sub_key, sub_val in nested.items():
                flat[f"{key}.{sub_key}"] = sub_val
        else:
            flat[key] = value
    return flat


def _inject_play_rate(args: List[str], rate: float):
    if rate <= 0:
        return
    has_rate = False
    for idx, val in enumerate(args):
        if val in ("-r", "--rate"):
            has_rate = True
            if idx + 1 < len(args):
                args[idx + 1] = str(rate)
            else:
                args.append(str(rate))
            break
        if val.startswith("-r") and len(val) > 2 and val[2:].replace(".", "", 1).isdigit():
            args[idx] = f"-r{rate}"
            has_rate = True
            break
    if not has_rate:
        args.extend(["-r", str(rate)])


def _set_or_append_arg(args: List[str], flag: str, value: float):
    for idx, val in enumerate(args):
        if val == flag:
            if idx + 1 < len(args):
                args[idx + 1] = str(value)
            else:
                args.append(str(value))
            return
    args.extend([flag, str(value)])


def _inject_play_duration(args: List[str], duration: float, start: float = 0.0):
    if duration <= 0:
        return
    _set_or_append_arg(args, "--start-offset", start)
    _set_or_append_arg(args, "--playback-duration", duration)


def _write_summary(rows: List[Dict], out_path: Path):
    if not rows:
        return
    if pd is None:
        typer.echo("pandas not installed; skipping summary.csv (pip install pandas to enable).")
        return
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    typer.echo(f"Summary saved to {out_path}")


@app.command("list")
def list_runs(config: Path = typer.Option(Path(__file__).with_name("experiments.yaml"), help="Experiment config YAML.")):
    cfg = _load_config(config)
    rows = []
    for group, data in cfg.get("experiments", {}).items():
        for run in data.get("runs", []):
            status = "disabled" if run.get("disabled") else "active"
            rows.append((group, run["id"], run.get("label", run["id"]), status))
    rows.sort()
    for group, run_id, label, status in rows:
        typer.echo(f"{group:18s} {run_id:15s} [{status}] - {label}")


@app.command()
def run(
    bag: Path = typer.Option(..., exists=True, help="Recorded Run A bag (directory with metadata.yaml)."),
    config: Path = typer.Option(Path(__file__).with_name("experiments.yaml"), help="Experiment config YAML."),
    output: Optional[Path] = typer.Option(None, help="Results directory (default evaluation/results/run_<timestamp>)."),
    groups: Optional[List[str]] = typer.Option(None, help="Run only these experiment groups."),
    runs: Optional[List[str]] = typer.Option(None, help="Run only specific run ids."),
    startup_delay: float = typer.Option(6.0, help="Seconds to wait after launching SLAM before playback."),
    skip_ros: bool = typer.Option(False, help="Skip ros2 launch/playback and only recompute metrics."),
    play_rate: float = typer.Option(5.0, min=0.1, help="ros2 bag play rate multiplier (default 5x)."),
    play_duration: Optional[float] = typer.Option(None, help="Seconds of bag to play (omit for full bag)."),
    launch_rviz: bool = typer.Option(False, help="Force RViz on/off for these runs (default false)."),
):
    cfg = _load_config(config)
    config_dir = config.parent.resolve()
    selected = _select_runs(cfg, groups, runs)
    if not selected:
        typer.echo("No experiments selected.")
        raise typer.Exit(1)

    results_root = output or Path("evaluation") / "results" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results_root.mkdir(parents=True, exist_ok=True)

    defaults = cfg["defaults"]
    bag_play_args = list(defaults.get("bag_play_args", ["--clock"]))
    _inject_play_rate(bag_play_args, play_rate)
    if play_duration is not None:
        _inject_play_duration(bag_play_args, play_duration)
    record_topics = defaults.get("record_topics", [])
    launch_base = dict(defaults.get("launch_args", {}))
    launch_base["launch_rviz"] = launch_rviz
    ping_stats = _measure_ping_stats(bag, defaults["sensor_topics"]["profiling_image"])
    gt_context = _prepare_gt_points(defaults, bag, results_root, config_dir)
    bag_duration = bag_utils.bag_duration_seconds(bag)

    typer.echo(f"Bag duration: {bag_duration:.1f} s, sonar pings: {ping_stats['count']} (width={ping_stats['width']})")
    typer.echo(f"Ground-truth points loaded: {gt_context.points.shape[0]}")

    summary_rows = []
    with typer.progressbar(length=len(selected), label="Running experiments") as progress:
        for run_cfg in selected:
            run_dir = results_root / run_cfg.run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            launch_args = {**launch_base, **run_cfg.overrides}
            run_meta = {
                "group": run_cfg.group,
                "label": run_cfg.label,
                "overrides": run_cfg.overrides,
                "metrics": run_cfg.metrics,
            }
            (run_dir / "run_config.json").write_text(json.dumps(run_meta, indent=2))

            wall_time = None
            recorded_bag = None
            if skip_ros:
                typer.echo(f"[{run_cfg.run_id}] skip_ros=True -> expecting existing bag in {run_dir}")
                recorded_bag = bag_utils.ensure_bag_path(run_dir / "rosbag2")
            else:
                wall_time, recorded_bag = _run_ros_pipeline(
                    run_cfg,
                    bag,
                    run_dir,
                    defaults,
                    launch_args,
                    bag_play_args,
                    record_topics,
                    startup_delay,
                    play_rate,
                    play_duration,
                )

            summary = {
                "run_id": run_cfg.run_id,
                "group": run_cfg.group,
                "label": run_cfg.label,
                "bag_duration_sec": bag_duration,
                "wall_time_sec": wall_time,
                "real_time_factor": bag_duration / wall_time if wall_time else None,
                "ping_processing_rate_hz": ping_stats["count"] / wall_time if wall_time else None,
            }
            summary["ground_truth"] = gt_context.metadata

            nav_required = "nav" in run_cfg.metrics
            mapping_required = "mapping" in run_cfg.metrics
            run_metrics = {}
            nav_metrics = None
            alignment = None
            if nav_required or mapping_required:
                nav_metrics, alignment = _postprocess_nav_metrics(
                    run_dir,
                    recorded_bag,
                    defaults,
                    evo_required=True,
                    run_label=run_cfg.label,
                )
            if nav_required and nav_metrics is not None:
                run_metrics["nav"] = nav_metrics
            if mapping_required:
                run_metrics["mapping"] = _postprocess_mapping_metrics(
                    run_dir,
                    recorded_bag,
                    gt_context,
                    defaults,
                    ping_stats,
                    alignment,
                )
            summary["metrics"] = run_metrics
            (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
            summary_flat = {"run_dir": str(run_dir)}
            summary_flat.update(summary)
            summary_flat.update(_flatten_metrics(run_metrics))
            summary_rows.append(summary_flat)
            progress.update(1)

    _write_summary(summary_rows, results_root / "summary.csv")


if __name__ == "__main__":
    app()
