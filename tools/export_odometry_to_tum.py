#!/usr/bin/env python3
"""Export nav_msgs/Odometry messages from a rosbag2 to a TUM trajectory file."""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import shutil

import yaml

try:
    from nav_msgs.msg import Odometry
    from rclpy.serialization import deserialize_message
    from rosbag2_py import (
        CompressionOptions,
        CompressionMode,
        ConverterOptions,
        SequentialReader,
        StorageOptions,
    )
except ImportError as exc:  # pragma: no cover - tooling only
    print(f"[export_odometry_to_tum] Failed to import ROS 2 bag dependencies: {exc}", file=sys.stderr)
    sys.exit(1)


@dataclass
class TUMEntry:
    stamp: float
    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]


@dataclass
class BagMetadata:
    storage_id: str
    compression_format: str
    compression_mode: str


def _read_metadata(metadata_path: Path) -> BagMetadata:
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.yaml not found under {metadata_path.parent}")

    with metadata_path.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    try:
        info = data['rosbag2_bagfile_information']
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"Failed to read rosbag metadata from {metadata_path}") from exc

    try:
        storage_id = info['storage_identifier']
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"Failed to read storage identifier from {metadata_path}") from exc

    compression_format = info.get('compression_format', '') or ''
    compression_mode = info.get('compression_mode', '') or ''

    return BagMetadata(
        storage_id=storage_id,
        compression_format=compression_format,
        compression_mode=compression_mode,
    )


def _decompress_bag_to_directory(source: Path, destination: Path) -> None:
    """Materialize an uncompressed copy of `source` under `destination`."""
    destination.mkdir(parents=True, exist_ok=True)

    metadata_path = source / 'metadata.yaml'
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.yaml not found under {source}")

    with metadata_path.open('r', encoding='utf-8') as f:
        metadata = yaml.safe_load(f)

    info = metadata.get('rosbag2_bagfile_information')
    if info is None:
        raise KeyError(f"Failed to read rosbag metadata from {metadata_path}")

    relative_paths = info.get('relative_file_paths', [])
    new_relative_paths: list[str] = []

    zstd_exe = shutil.which('zstd')
    if zstd_exe is None:
        raise RuntimeError(
            "Cannot decompress rosbag: 'zstd' executable not found in PATH. "
            "Please install zstd or upgrade rosbag2_py with compression support."
        )

    for rel_path_str in relative_paths:
        rel_path = Path(rel_path_str)
        source_file = source / rel_path
        if not source_file.exists():
            raise FileNotFoundError(f"Bag chunk '{source_file}' not found while decompressing {source}")

        if rel_path.suffix == '.zstd':
            dest_rel_path = rel_path.with_suffix('')
            dest_file = destination / dest_rel_path
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(
                [zstd_exe, '-d', '--force', '--no-progress', '-o', str(dest_file), str(source_file)],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                stderr = result.stderr.strip() or result.stdout.strip()
                raise RuntimeError(f"Failed to decompress '{source_file}': {stderr}")
        else:
            dest_rel_path = rel_path
            dest_file = destination / dest_rel_path
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, dest_file)

        new_relative_paths.append(str(dest_rel_path))

    info['relative_file_paths'] = new_relative_paths
    info['compression_format'] = ''
    info['compression_mode'] = ''

    with (destination / 'metadata.yaml').open('w', encoding='utf-8') as f:
        yaml.safe_dump(metadata, f)


def _iter_odometry_messages(bag_path: Path, topic: str) -> Iterable[Odometry]:
    metadata = _read_metadata(bag_path / 'metadata.yaml')

    storage_options = StorageOptions(uri=str(bag_path), storage_id=metadata.storage_id)
    converter_options = ConverterOptions(input_serialization_format='', output_serialization_format='')

    compression_options: Optional[CompressionOptions] = None
    if metadata.compression_format:
        compression_options = CompressionOptions()
        compression_options.compression_format = metadata.compression_format
        mode_value = metadata.compression_mode
        if isinstance(mode_value, CompressionMode):
            compression_options.compression_mode = mode_value
        elif isinstance(mode_value, str):
            mode_key = mode_value.strip().upper()
            if mode_key:
                if hasattr(CompressionMode, mode_key):
                    compression_options.compression_mode = getattr(CompressionMode, mode_key)
                else:
                    raise ValueError(
                        f"Unsupported compression mode '{mode_value}' in metadata for bag '{bag_path}'"
                    )
        elif mode_value:
            raise TypeError(
                f"Unexpected compression mode type '{type(mode_value).__name__}' in metadata for bag '{bag_path}'"
            )

    reader = SequentialReader()
    temp_dir: Optional[tempfile.TemporaryDirectory] = None
    try:
        if compression_options is None:
            reader.open(storage_options, converter_options)
        else:
            reader.open(storage_options, converter_options, compression_options)
    except TypeError:
        # Older versions of rosbag2_py do not accept compression options. Attempt to
        # materialize an uncompressed copy of the bag and reopen from there.
        if compression_options is None:
            reader.open(storage_options, converter_options)
        else:
            temp_dir = tempfile.TemporaryDirectory(prefix='export_odometry_tum_')
            temp_bag_path = Path(temp_dir.name)
            try:
                _decompress_bag_to_directory(bag_path, temp_bag_path)
            except Exception:
                temp_dir.cleanup()
                raise
            storage_options = StorageOptions(uri=str(temp_bag_path), storage_id=metadata.storage_id)
            reader.open(storage_options, converter_options)
    except RuntimeError as exc:
        raise RuntimeError(f"Failed to open bag '{bag_path}' with storage '{metadata.storage_id}': {exc}") from exc

    try:
        while reader.has_next():
            topic_name, data, _ = reader.read_next()
            if topic_name != topic:
                continue

            try:
                yield deserialize_message(data, Odometry)
            except Exception as exc:  # pragma: no cover - corrupted bag
                raise RuntimeError(f"Failed to deserialize Odometry on {topic_name}: {exc}") from exc
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def _stamp_to_float(msg: Odometry) -> Optional[float]:
    stamp = msg.header.stamp
    if stamp is None:
        return None
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def export_to_tum(bag_path: Path, topic: str, output: Path) -> None:
    entries: list[TUMEntry] = []
    for msg in _iter_odometry_messages(bag_path, topic):
        stamp = _stamp_to_float(msg)
        if stamp is None:
            continue
        pose = msg.pose.pose
        entries.append(
            TUMEntry(
                stamp=stamp,
                position=(pose.position.x, pose.position.y, pose.position.z),
                orientation=(
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                ),
            )
        )

    if not entries:
        raise RuntimeError(f"No Odometry messages found for topic '{topic}' in {bag_path}")

    entries.sort(key=lambda e: e.stamp)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open('w', encoding='utf-8') as f:
        for entry in entries:
            px, py, pz = entry.position
            qx, qy, qz, qw = entry.orientation
            f.write(f"{entry.stamp:.9f} {px:.9f} {py:.9f} {pz:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag', type=Path, help='Path to the rosbag2 directory')
    parser.add_argument('topic', type=str, help='Odometry topic to export (e.g. /slam/odom)')
    parser.add_argument('output', type=Path, help='Destination TUM file')
    args = parser.parse_args(argv)

    bag_path = args.bag.expanduser().resolve()
    if not bag_path.exists() or not bag_path.is_dir():
        parser.error(f"Bag directory '{bag_path}' does not exist or is not a directory")

    try:
        export_to_tum(bag_path, args.topic, args.output.expanduser().resolve())
    except Exception as exc:
        print(f"[export_odometry_to_tum] {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
