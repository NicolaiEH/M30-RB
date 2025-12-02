from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import yaml
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rosidl_runtime_py.utilities import get_message
from sensor_msgs_py import point_cloud2 as pc2


@dataclass
class BagMessage:
    topic: str
    msg: object
    timestamp_ns: int


class BagReader:
    """Context manager around rosbag2_py.SequentialReader with topic filtering support."""

    def __init__(self, bag_path: Path | str):
        self.bag_path = Path(bag_path)
        self._reader: Optional[SequentialReader] = None
        self._type_map: Dict[str, str] = {}
        self._class_cache: Dict[str, object] = {}

    def _detect_storage_id(self) -> str:
        try:
            meta = bag_metadata(self.bag_path)
            info = meta.get("rosbag2_bagfile_information", {})
            storage = info.get("storage_identifier") or meta.get("storage_identifier")
            if storage:
                return str(storage)
        except Exception:
            pass
        return "sqlite3"

    def __enter__(self) -> "BagReader":
        storage = StorageOptions(uri=str(self.bag_path), storage_id=self._detect_storage_id())
        converter = ConverterOptions("", "")
        self._reader = SequentialReader()
        self._reader.open(storage, converter)
        self._type_map = {topic.name: topic.type for topic in self._reader.get_all_topics_and_types()}
        self._class_cache = {}
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._reader and hasattr(self._reader, "close"):
            self._reader.close()
        self._reader = None
        self._type_map = {}
        self._class_cache = {}

    def messages(self, topics: Optional[Sequence[str]] = None) -> Iterator[BagMessage]:
        if self._reader is None:
            raise RuntimeError("BagReader must be used as a context manager.")
        topics_set = set(topics) if topics else None
        while self._reader.has_next():
            raw = self._reader.read_next()
            topic_name: Optional[str] = None
            data = None
            stamp_ns: Optional[int] = None

            if isinstance(raw, tuple):
                if len(raw) == 3:
                    topic_name, data, stamp_ns = raw
                else:
                    raise RuntimeError(f"Unexpected tuple shape from rosbag2 reader: {len(raw)}")
            else:
                topic_name = getattr(raw, "topic_name", None)
                data = getattr(raw, "serialized_data", None)
                stamp_ns = getattr(raw, "time_stamp", None)

            if topic_name is None or data is None or stamp_ns is None:
                raise RuntimeError("Failed to parse rosbag2 message record.")

            if topics_set and topic_name not in topics_set:
                continue
            msg_type = self._type_map.get(topic_name)
            if msg_type is None:
                continue
            if topic_name not in self._class_cache:
                self._class_cache[topic_name] = get_message(msg_type)
            ros_msg = deserialize_message(data, self._class_cache[topic_name])
            yield BagMessage(topic=topic_name, msg=ros_msg, timestamp_ns=stamp_ns)


def export_odometry_to_tum(bag_path: Path, topic: str, output_path: Path) -> np.ndarray:
    """
    Export nav_msgs/Odometry to TUM format and return Nx3 array of XYZ positions.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[str] = []
    xyz: List[Tuple[float, float, float]] = []
    with BagReader(bag_path) as reader:
        for entry in reader.messages([topic]):
            msg = entry.msg
            stamp = msg.header.stamp
            t = float(stamp.sec) + float(stamp.nanosec) * 1e-9
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            rows.append(
                f"{t:.9f} {p.x:.6f} {p.y:.6f} {p.z:.6f} "
                f"{q.x:.6f} {q.y:.6f} {q.z:.6f} {q.w:.6f}"
            )
            xyz.append((p.x, p.y, p.z))
    output_path.write_text("\n".join(rows))
    return np.array(xyz, dtype=float) if xyz else np.zeros((0, 3), dtype=float)


def collect_point_cloud_points(bag_path: Path, topic: str) -> np.ndarray:
    """
    Aggregate PointCloud2 messages (x,y,z only) into a single Nx3 numpy array.
    """
    pts: List[np.ndarray] = []
    with BagReader(bag_path) as reader:
        for entry in reader.messages([topic]):
            msg = entry.msg
            data = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
            if not data:
                continue
            arr = np.fromiter((coord for point in data for coord in point), dtype=np.float32)
            arr = arr.reshape(-1, 3)
            pts.append(arr)
    if not pts:
        return np.zeros((0, 3), dtype=np.float32)
    return np.vstack(pts)


def count_image_dimensions(bag_path: Path, topic: str) -> Dict[str, int]:
    """
    Count number of messages on an Image topic and return width/height from the first.
    """
    stats = {"count": 0, "width": 0, "height": 0}
    with BagReader(bag_path) as reader:
        for entry in reader.messages([topic]):
            msg = entry.msg
            stats["count"] += 1
            if stats["width"] == 0:
                stats["width"] = int(msg.width)
                stats["height"] = int(msg.height)
    return stats


def extract_odometry_positions(bag_path: Path, topic: str) -> np.ndarray:
    """
    Return Nx3 numpy array of XYZ from nav_msgs/Odometry.
    """
    positions: List[Tuple[float, float, float]] = []
    with BagReader(bag_path) as reader:
        for entry in reader.messages([topic]):
            p = entry.msg.pose.pose.position
            positions.append((p.x, p.y, p.z))
    if not positions:
        return np.zeros((0, 3), dtype=float)
    return np.array(positions, dtype=float)


def bag_metadata(bag_path: Path) -> Dict:
    meta_path = Path(bag_path) / "metadata.yaml"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.yaml not found in {bag_path}")
    with meta_path.open("r") as f:
        return yaml.safe_load(f)


def bag_duration_seconds(bag_path: Path) -> float:
    meta = bag_metadata(bag_path)
    duration = meta.get("duration")
    if isinstance(duration, dict):
        dur_ns = duration.get("nanoseconds")
        if dur_ns is not None:
            return float(dur_ns) * 1e-9
    if isinstance(duration, (int, float)):
        return float(duration)
    info = meta.get("rosbag2_bagfile_information", {})
    dur_ns = None
    if isinstance(info, dict):
        dur = info.get("duration")
        if isinstance(dur, dict):
            dur_ns = dur.get("nanoseconds")
        elif isinstance(dur, (int, float)):
            return float(dur) * 1e-9
    return float(dur_ns) * 1e-9 if dur_ns is not None else 0.0


def ensure_bag_path(record_root: Path) -> Path:
    """
    ros2 bag record -o foo will create either foo or foo_0 ...; find the most recent match.
    """
    if record_root.exists() and (record_root / "metadata.yaml").exists():
        return record_root
    candidates = sorted(record_root.parent.glob(f"{record_root.name}*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for cand in candidates:
        if (cand / "metadata.yaml").exists():
            return cand
    raise FileNotFoundError(f"Could not locate a recorded bag under {record_root.parent} (prefix={record_root.name}).")
