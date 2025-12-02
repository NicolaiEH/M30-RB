from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from . import bag_utils


def compute_bounds_from_gt(
    bag_path: Path,
    topic: str = "/gt/odom",
    xy_margin_m: float = 20.0,
    z_margin_m: float = 10.0,
) -> Dict[str, tuple[float, float]]:
    """
    Estimate a bounding box around the lane based on /gt/odom.
    """
    xyz = bag_utils.extract_odometry_positions(bag_path, topic)
    if xyz.size == 0:
        raise RuntimeError(f"No poses found on {topic} in {bag_path}")
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    bounds = {
        "x": (float(mins[0] - xy_margin_m), float(maxs[0] + xy_margin_m)),
        "y": (float(mins[1] - xy_margin_m), float(maxs[1] + xy_margin_m)),
        "z": (float(mins[2] - z_margin_m), float(maxs[2] + z_margin_m)),
    }
    return bounds


def load_ply_point_cloud(ply_path: Path) -> np.ndarray:
    """
    Load a binary_little_endian PLY file containing only vertex x/y/z as doubles.

    Returns an (N, 3) float32 array in the source frame (typically world coordinates, metres).
    """
    ply_path = Path(ply_path)
    if not ply_path.exists():
        raise FileNotFoundError(f"PLY ground-truth file not found: {ply_path}")

    with ply_path.open("rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError(f"Unexpected EOF while reading PLY header from {ply_path}")
            try:
                decoded = line.decode("ascii")
            except UnicodeDecodeError:
                decoded = ""
            header_lines.append(decoded.strip())
            if decoded.startswith("end_header"):
                break

        vertex_count = None
        for line in header_lines:
            if line.startswith("element vertex"):
                parts = line.split()
                if len(parts) >= 3:
                    vertex_count = int(parts[2])
                break
        if vertex_count is None:
            raise RuntimeError(f"PLY header in {ply_path} is missing an 'element vertex' line")

        arr = np.fromfile(f, dtype="<f8", count=vertex_count * 3)
        if arr.size != vertex_count * 3:
            raise RuntimeError(
                f"PLY payload in {ply_path} has {arr.size} doubles, expected {vertex_count * 3}"
            )
        pts = arr.reshape(-1, 3).astype(np.float32)
    return pts


def load_octree_crop(
    cache_dir: Path,
    bounds: Dict[str, tuple[float, float]],
    cache_store: Optional[Path] = None,
    unit_scale: float = 1.0,
    filter_params: Optional[Dict[str, object]] = None,
    translation: Optional[tuple[float, float, float]] = None,
) -> tuple[np.ndarray, Optional[Path]]:
    """
    Load cached octree tiles within the provided bounds (inclusive) and optionally cache the crop.
    Additional filtering (e.g., corridor slicing) can be provided via ``filter_params``.
    """
    cache_dir = Path(cache_dir)
    cache_store = Path(cache_store) if cache_store else None
    cache_file = None
    if cache_store:
        cache_store.mkdir(parents=True, exist_ok=True)
        cache_key = {
            "bounds": bounds,
            "unit_scale": unit_scale,
            "filter": filter_params,
            "translation": translation,
        }
        key = json.dumps(cache_key, sort_keys=True).encode("utf-8")
        digest = hashlib.sha1(key).hexdigest()[:12]
        cache_file = cache_store / f"gt_crop_{digest}.npz"
        if cache_file.exists():
            data = np.load(cache_file)
            return data["points"], cache_file

    npz_candidates = _candidate_npz_files(cache_dir, bounds)
    pts = []
    for npz_path in npz_candidates:
        data = np.load(npz_path)
        x = data["x"]
        y = data["y"]
        z = data["z"]
        mask = (
            (x >= bounds["x"][0]) & (x <= bounds["x"][1]) &
            (y >= bounds["y"][0]) & (y <= bounds["y"][1]) &
            (z >= bounds["z"][0]) & (z <= bounds["z"][1])
        )
        if np.any(mask):
            x_sel = x[mask].astype(np.float32)
            y_sel = y[mask].astype(np.float32)
            z_sel = z[mask].astype(np.float32)
            if unit_scale != 1.0:
                x_sel *= unit_scale
                y_sel *= unit_scale
                z_sel *= unit_scale
            if translation is not None:
                x_sel += float(translation[0])
                y_sel += float(translation[1])
                z_sel += float(translation[2])
            if filter_params is not None:
                mask_corridor = _apply_corridor_filter(x_sel, y_sel, filter_params)
                if not np.any(mask_corridor):
                    continue
                x_sel = x_sel[mask_corridor]
                y_sel = y_sel[mask_corridor]
                z_sel = z_sel[mask_corridor]
            sel = np.stack([x_sel, y_sel, z_sel], axis=1)
            pts.append(sel)
    if pts:
        pts_arr = np.vstack(pts).astype(np.float32)
    else:
        pts_arr = np.zeros((0, 3), dtype=np.float32)

    if cache_file:
        np.savez_compressed(cache_file, points=pts_arr)

    return pts_arr, cache_file


def _candidate_npz_files(cache_dir: Path, bounds: Dict[str, tuple[float, float]]) -> list[Path]:
    index_path = cache_dir / "index.jsonl"
    if not index_path.exists():
        return sorted(cache_dir.glob("*.npz"))
    candidates: list[Path] = []
    with index_path.open("r") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            bbox = rec.get("bbox")
            if not bbox or not _bbox_intersects(bounds, bbox):
                continue
            zmin = float(rec.get("zmin", bounds["z"][0]))
            zmax = float(rec.get("zmax", bounds["z"][1]))
            if zmax < bounds["z"][0] or zmin > bounds["z"][1]:
                continue
            npz_rel = rec.get("npz_rel")
            if not npz_rel:
                continue
            candidates.append(cache_dir / npz_rel)
    if not candidates:
        return sorted(cache_dir.glob("*.npz"))
    return sorted(candidates)


def _bbox_intersects(bounds: Dict[str, tuple[float, float]], bbox: list[float]) -> bool:
    bx0, by0, bx1, by1 = bbox
    x0, x1 = bounds["x"]
    y0, y1 = bounds["y"]
    if bx1 < x0 or bx0 > x1:
        return False
    if by1 < y0 or by0 > y1:
        return False
    return True


def _apply_corridor_filter(x: np.ndarray, y: np.ndarray, params: Dict[str, object]) -> np.ndarray:
    origin = np.asarray(params.get("origin", [0.0, 0.0]), dtype=np.float32)
    dir_vec = np.asarray(params.get("dir", [1.0, 0.0]), dtype=np.float32)
    perp_vec = np.asarray(params.get("perp", [0.0, 1.0]), dtype=np.float32)
    norm_dir = np.linalg.norm(dir_vec)
    if norm_dir > 0:
        dir_vec = dir_vec / norm_dir
    norm_perp = np.linalg.norm(perp_vec)
    if norm_perp > 0:
        perp_vec = perp_vec / norm_perp
    s_min = float(params.get("s_min", -np.inf))
    s_max = float(params.get("s_max", np.inf))
    t_half = float(params.get("t_half", np.inf))
    rel_x = x - origin[0]
    rel_y = y - origin[1]
    s = rel_x * dir_vec[0] + rel_y * dir_vec[1]
    t = rel_x * perp_vec[0] + rel_y * perp_vec[1]
    return (s >= s_min) & (s <= s_max) & (np.abs(t) <= t_half)
