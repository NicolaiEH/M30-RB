from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

CloudChoice = Literal["reconstructed", "ground_truth", "both"]


def _default_out_path(npz_path: Path, key: str) -> Path:
    return npz_path.with_name(f"{npz_path.stem}_{key}.ply")


def _write_ply(points: np.ndarray, out_path: Path) -> None:
    """
    Write an (N, 3) float32 array to a binary little-endian PLY file.
    """
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {points.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(header.encode("ascii"))
        f.write(points.astype("<f4", copy=False).tobytes())


def export_npz_to_ply(npz_path: Path, output: Path | None = None, cloud: CloudChoice = "reconstructed") -> list[Path]:
    """
    Export arrays from evaluation mapping point_clouds.npz files to PLY.

    Args:
        npz_path: Path to point_clouds.npz produced by run_experiments.
        output: Optional explicit output path (only valid when exporting a single cloud).
        cloud: Which array to export. Use "both" to write reconstructed and ground_truth side-by-side.

    Returns:
        List of written PLY paths.
    """
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    keys: Sequence[str]
    if cloud == "both":
        if output is not None:
            raise ValueError("--output cannot be used with --cloud both; outputs are derived from the NPZ filename.")
        keys = ("reconstructed", "ground_truth")
    else:
        keys = (cloud,)

    outputs: list[Path] = []
    with np.load(npz_path, allow_pickle=False) as data:
        available = set(data.files)
        for key in keys:
            if key not in available:
                readable = ", ".join(sorted(available)) or "none"
                raise ValueError(f"Key '{key}' not found in {npz_path}. Available arrays: {readable}.")

        for key in keys:
            arr = np.asarray(data[key])
            if arr.ndim != 2 or arr.shape[1] < 3:
                raise ValueError(f"Array '{key}' expected shape (N, 3+) but got {arr.shape}")
            xyz = arr[:, :3].astype("<f4", copy=False)
            out_path = Path(output) if output is not None else _default_out_path(npz_path, key)
            _write_ply(xyz, out_path)
            outputs.append(out_path)
    return outputs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert evaluation mapping point_clouds.npz files to PLY for CloudCompare."
    )
    parser.add_argument(
        "npz_path",
        type=Path,
        help="Path to metrics/mapping/point_clouds.npz produced by run_experiments.",
    )
    parser.add_argument(
        "--cloud",
        choices=["reconstructed", "ground_truth", "both"],
        default="reconstructed",
        help='Which array to export. Use "both" to write two PLYs next to the NPZ.',
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PLY path (only when exporting a single cloud). Defaults to <npz>_<cloud>.ply.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    outputs = export_npz_to_ply(args.npz_path, output=args.output, cloud=args.cloud)
    for out_path in outputs:
        print(f"Wrote {out_path} ({out_path.stat().st_size / 1_000_000:.2f} MB)")


if __name__ == "__main__":
    main()
