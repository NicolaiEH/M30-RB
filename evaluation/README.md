## Evaluation Toolkit

Run recorded PierHarbor bags through the SLAM + mapping stack and collect metrics/plots.

### Install
Activate your ROS 2 Python env, then:
```bash
pip install -r evaluation/requirements.txt
pip install evo
```
ROS 2 Python bindings (`rclpy`, `rosbag2_py`, `sensor_msgs_py`) must already be available.

### Steps
1. Record a bag with profiling sonar enabled.
2. Configure experiments in `experiments.yaml` (groups, launch overrides, ground-truth paths).
3. Run sweeps:
```bash
python -m evaluation.run_experiments run \
  --bag <bag_dir> \
  --groups navigation_accuracy voxel_downsample \
  --output evaluation/results/<run_id>
```
4. Results per run live under `evaluation/results/<run_id>/<run>` with logs, trajectories, metrics, plots, and `run_summary.json`. A CSV summary is written at `evaluation/results/<run_id>/summary.csv`.

### Utilities
- `debug_gt_alignment.py`: regenerate GT vs map overlay for a run.
- `npz_to_ply.py`: export cached point clouds to PLY for CloudCompare.
