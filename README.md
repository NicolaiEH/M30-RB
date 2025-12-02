# ProfilingSonar iSAM2

ROS 2 nodes and evaluation tools for the HoloOcean bridge, profiling sonar mapping frontend, and iSAM2 backend used in the thesis.

## Components
- `ros2_ws/src/holoocean_bridge`: HoloOcean -> ROS 2 bridge, lawnmower helper, launch files.
- `ros2_ws/src/profiling_frontend`: Profiling sonar back-projection node and launch.
- `ros2_ws/src/slam_backend`: Minimal iSAM2 backend node and launch.
- `ros2_ws/src/wasd_teleop`: Keyboard teleop for thruster control.
- `tools`: Bag recording and utility scripts.
- `evaluation`: Experiment CLI, metrics, and plotting utilities.

## Setup
1. Install ROS 2 (Humble or similar) and HoloOcean using their official guides.
2. From this repo: `pip install -r requirements.txt` in a Python env that matches your ROS 2 install.
3. Build: `colcon build --symlink-install` inside `ros2_ws`.

## Typical Workflow
1. Simulate and record: `ros2 launch holoocean_bridge holoocean_sim.launch.py enable_profiling_sonar:=true` then `ros2 bag record -a -o <bag_dir>`.
2. Play back with SLAM + mapping: `ros2 launch holoocean_bridge slam_and_mapping.launch.py bag:=<bag_dir>` (adjust launch args as needed).
3. Evaluate: `python -m evaluation.run_experiments run --bag <bag_dir> --output evaluation/results/<run_id>`.

## Notes
- Launch files assume NWU frames and `use_sim_time=true`.
- Experiment definitions live in `evaluation/experiments.yaml`.
- Ground-truth octree/PLY data and recorded bags are not included.
