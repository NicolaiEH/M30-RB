#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <rosbag2_directory> [result_dir]" >&2
  exit 1
fi

BAG_PATH="$1"
if [[ ! -d "$BAG_PATH" ]]; then
  echo "Error: bag directory '$BAG_PATH' not found" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT_BASE="${2:-${SCRIPT_DIR}/../results}"
mkdir -p "$RESULT_BASE"

BAG_NAME="$(basename "$BAG_PATH")"
STAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="${RESULT_BASE}/${STAMP}_${BAG_NAME}"
mkdir -p "$RESULT_DIR"

echo "[evaluate_slam] Saving outputs to $RESULT_DIR"

GT_TUM="${RESULT_DIR}/gt.tum"
SLAM_TUM="${RESULT_DIR}/slam.tum"
EXPORTER="${SCRIPT_DIR}/export_odometry_to_tum.py"
export RESULT_DIR

set -x
python3 "$EXPORTER" "$BAG_PATH" /gt/odom "$GT_TUM"
python3 "$EXPORTER" "$BAG_PATH" /slam/odom "$SLAM_TUM"

evo_ape tum "$GT_TUM" "$SLAM_TUM" -a -s --plot --plot_mode xy --save_plot "${RESULT_DIR}/ape_xy.png" --save_results "${RESULT_DIR}/ape_results.zip"
evo_rpe tum "$GT_TUM" "$SLAM_TUM" --delta 1 --plot --save_plot "${RESULT_DIR}/rpe.png" --save_results "${RESULT_DIR}/rpe_results.zip"
set +x

python3 - <<'PY'
import csv
import os
import sys
from pathlib import Path

try:
    from evo.tools import file_interface
except ImportError as exc:  # pragma: no cover - dependency provided by evo
    print(f"[evaluate_slam] Failed to import evo: {exc}", file=sys.stderr)
    sys.exit(1)

result_dir = Path(os.environ['RESULT_DIR']).resolve()
ape_file = result_dir / 'ape_results.zip'
rpe_file = result_dir / 'rpe_results.zip'

if not ape_file.exists() or not rpe_file.exists():
    print('[evaluate_slam] Missing evo result archives; check evo_ape/evo_rpe output.', file=sys.stderr)
    sys.exit(1)

ape_res = file_interface.load_res_file(str(ape_file))
rpe_res = file_interface.load_res_file(str(rpe_file))

def write_stats(filename, stats):
    with open(result_dir / filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        for key in ['rmse', 'mean', 'median', 'std', 'min', 'max']:
            value = getattr(stats, key, None)
            if value is not None:
                writer.writerow([key, f"{value:.6f}"])

write_stats('ape_metrics.csv', ape_res.stats)
write_stats('rpe_metrics.csv', rpe_res.stats)
PY

echo "[evaluate_slam] Done. Plots and metrics saved to $RESULT_DIR"
