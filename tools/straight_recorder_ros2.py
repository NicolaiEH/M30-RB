#!/usr/bin/env python3
"""Start the HoloOcean bridge in thruster mode, launch WASD teleop, and record all ROS 2 topics until interrupted."""

import argparse
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence


DEFAULT_SCENARIO = Path(__file__).resolve().parents[1] / "src/holoocean_bridge/config/custom_scenario.json"
DEFAULT_BAG_ROOT = Path("~/rosbag")
DEFAULT_BAG_PREFIX = "teleop_thrusters"
DEFAULT_TELEOP_TOPIC = "/cmd_vel"


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _bag_directory(root: Path, prefix: str) -> Path:
    root = root.expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{prefix}_{_timestamp()}"


@dataclass
class ProcHandle:
    name: str
    popen: Optional[subprocess.Popen] = None

    def stop(self, sig: int = signal.SIGINT, timeout: float = 10.0) -> None:
        if not self.popen:
            return
        if self.popen.poll() is not None:
            return

        try:
            os.killpg(os.getpgid(self.popen.pid), sig)
        except ProcessLookupError:
            return
        except Exception:
            self.popen.terminate()

        deadline = time.monotonic() + max(0.0, float(timeout))
        while self.popen.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)

        if self.popen.poll() is None:
            self.popen.kill()


def _start_process(cmd: Sequence[str], name: str, *, stdin=None) -> ProcHandle:
    print(f"[launcher] starting {name}: {' '.join(cmd)}")
    popen = subprocess.Popen(
        cmd,
        preexec_fn=os.setsid,
        stdin=stdin,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    return ProcHandle(name=name, popen=popen)


def _launch_bridge(scenario: Path, *, enable_profiling_sonar: bool, use_sim_time: bool) -> ProcHandle:
    cmd = [
        "ros2",
        "run",
        "holoocean_bridge",
        "node",
        "--ros-args",
        "-p",
        f"scenario_path:={str(scenario)}",
        "-p",
        "control_mode:=thrusters",
        "-p",
        f"enable_profiling_sonar:={'True' if enable_profiling_sonar else 'False'}",
    ]
    if use_sim_time:
        cmd.extend(["--use-sim-time"])
    return _start_process(cmd, "holoocean_bridge")


def _launch_teleop(cmd_topic: str, *, use_sim_time: bool) -> ProcHandle:
    cmd = ["ros2", "run", "wasd_teleop", "teleop_wasd"]
    ros_args = []
    if use_sim_time:
        ros_args.append("--use-sim-time")
    if cmd_topic != DEFAULT_TELEOP_TOPIC:
        ros_args.extend(["--remap", f"{DEFAULT_TELEOP_TOPIC}:={cmd_topic}"])
    if ros_args:
        cmd.extend(["--ros-args", *ros_args])
    return _start_process(cmd, "wasd_teleop")


def _start_bag(bag_dir: Path, *, include_hidden: bool) -> ProcHandle:
    cmd = ["ros2", "bag", "record", "-a"]
    if include_hidden:
        cmd.append("--include-hidden-topics")
    cmd.extend(["-o", str(bag_dir)])
    return _start_process(cmd, "ros2_bag_record", stdin=subprocess.DEVNULL)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch holoocean_bridge (thrusters), WASD teleop, and record a rosbag."
    )
    parser.add_argument(
        "--scenario",
        type=Path,
        default=DEFAULT_SCENARIO,
        help="Scenario JSON to load for holoocean_bridge (default: custom_scenario.json).",
    )
    parser.add_argument(
        "--bag-root",
        type=Path,
        default=DEFAULT_BAG_ROOT,
        help="Directory to store bag folders (default: ~/rosbag).",
    )
    parser.add_argument(
        "--bag-prefix",
        default=DEFAULT_BAG_PREFIX,
        help="Prefix for generated rosbag folder names (default: teleop_thrusters).",
    )
    parser.add_argument(
        "--teleop-topic",
        default=DEFAULT_TELEOP_TOPIC,
        help="Twist topic for teleop output (default: /cmd_vel).",
    )
    parser.add_argument(
        "--wait-bridge",
        type=float,
        default=5.0,
        help="Seconds to wait after starting holoocean_bridge before launching teleop (default: 5).",
    )
    parser.add_argument(
        "--wait-bag",
        type=float,
        default=1.0,
        help="Seconds to wait after starting rosbag record before launching teleop (default: 1).",
    )
    parser.add_argument(
        "--disable-profiling-sonar",
        action="store_true",
        help="Do not enable the profiling sonar parameter on the bridge.",
    )
    parser.add_argument(
        "--use-sim-time",
        action="store_true",
        help="Pass --use-sim-time to holoocean_bridge and teleop.",
    )
    parser.add_argument(
        "--exclude-hidden-topics",
        action="store_true",
        help="Skip --include-hidden-topics when recording the bag.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    scenario_path = Path(args.scenario).expanduser().resolve()
    if not scenario_path.exists():
        print(f"[error] Scenario file not found: {scenario_path}", file=sys.stderr)
        sys.exit(1)

    bag_dir = _bag_directory(Path(args.bag_root), args.bag_prefix)
    include_hidden = not args.exclude_hidden_topics
    enable_profiling_sonar = not args.disable_profiling_sonar

    print(f"[mission] scenario={scenario_path}")
    print(f"[mission] bag directory -> {bag_dir}")

    handles = []
    try:
        bridge = _launch_bridge(
            scenario_path,
            enable_profiling_sonar=enable_profiling_sonar,
            use_sim_time=args.use_sim_time,
        )
        handles.append(bridge)

        wait_bridge = max(0.0, float(args.wait_bridge))
        if wait_bridge > 0.0:
            print(f"[mission] waiting {wait_bridge:.1f}s for holoocean_bridge startup...")
            time.sleep(wait_bridge)

        bag = _start_bag(bag_dir, include_hidden=include_hidden)
        handles.append(bag)

        wait_bag = max(0.0, float(args.wait_bag))
        if wait_bag > 0.0:
            print(f"[mission] waiting {wait_bag:.1f}s before launching teleop...")
            time.sleep(wait_bag)

        teleop = _launch_teleop(args.teleop_topic, use_sim_time=args.use_sim_time)
        handles.append(teleop)

        if args.use_sim_time:
            print("[mission] teleop will follow /clock (use_sim_time enabled).")

        print("[mission] Manual control ready. Use WASD (and Q/E, space) in the teleop terminal.")
        print("[mission] Press Ctrl-C in this window to stop all processes and finalize the bag.")

        while True:
            for handle in handles:
                if handle.popen and handle.popen.poll() is not None:
                    code = handle.popen.returncode
                    print(f"[mission] {handle.name} exited with code {code}. Stopping remaining processes.")
                    raise RuntimeError(f"{handle.name} exited unexpectedly with code {code}")
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[mission] Ctrl-C received. Shutting down processes...")
    except RuntimeError as exc:
        print(f"[mission] {exc}", file=sys.stderr)
    finally:
        for handle in reversed(handles):
            try:
                handle.stop()
            except Exception as exc:
                print(f"[mission] Failed to stop {handle.name}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
