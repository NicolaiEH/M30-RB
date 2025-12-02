#!/usr/bin/env python3

"""
Bridge HoloOcean BlueROV2 scenarios to ROS 2, publishing NWU-aligned sensor topics for SLAM and mapping.
"""


from __future__ import annotations
import json, math, os, sys, time, traceback
from typing import Dict, Optional, Tuple, List
from pynput import keyboard

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.clock import Clock as RclpyClock, ClockType
from rclpy.logging import LoggingSeverity
from geometry_msgs.msg import PoseStamped

import holoocean as ho

from builtin_interfaces.msg import Time
from rosgraph_msgs.msg import Clock as ClockMsg
from std_msgs.msg import Header, Float32MultiArray
from geometry_msgs.msg import Twist, TwistStamped, TransformStamped, Quaternion, PointStamped, Vector3Stamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, Image
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf2_ros import TransformBroadcaster


def _rad(deg: float) -> float:  return float(math.radians(deg))
def _wrap_pi(a: float) -> float: return math.atan2(math.sin(a), math.cos(a))

def _flat(x) -> np.ndarray:
    try:
        a = np.asarray(x, dtype=np.float32)
        return a.reshape(-1)
    except Exception:
        return np.zeros((0,), dtype=np.float32)

def _diag(vals: List[float]) -> list:
    """Return 3x3 or 6x6 diagonal (row-major) for ROS covariance fields."""
    if len(vals) == 3:
        out = [0.0]*9
        out[0] = vals[0]; out[4] = vals[1]; out[8] = vals[2]
        return out
    if len(vals) == 6:
        out = [0.0]*36
        for i, v in enumerate(vals):
            out[i*6 + i] = v
        return out
    return []

def _quat_from_rpy(roll: float, pitch: float, yaw: float) -> Tuple[float,float,float,float]:
    # ROS xyzw
    cr = math.cos(roll*0.5);  sr = math.sin(roll*0.5)
    cp = math.cos(pitch*0.5); sp = math.sin(pitch*0.5)
    cy = math.cos(yaw*0.5);   sy = math.sin(yaw*0.5)
    qw = cr*cp*cy + sr*sp*sy
    qx = sr*cp*cy - cr*sp*sy
    qy = cr*sp*cy + sr*cp*sy
    qz = cr*cp*sy - sr*sp*cy
    return qx, qy, qz, qw

def _quat_mul(q1, q2):
    """Quaternion multiply for (x,y,z,w)."""
    x1,y1,z1,w1 = q1
    x2,y2,z2,w2 = q2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    return (x,y,z,w)

Q_RX_PI = (1.0, 0.0, 0.0, 0.0)  # xyzw unit quaternion for 180 deg about +x

def _ned_to_nwu_vec(v3):
    v = np.asarray(v3, dtype=float).copy()
    v[1:] *= -1.0
    return v

def _ned_to_nwu_quat_xyzw(q_xyzw):
    return np.array(_quat_mul(Q_RX_PI, tuple(q_xyzw)), dtype=float)

def _ned_to_nwu_T(T44):
     """Rotate an IMUSocket pose from NED body frame to NWU without moving the position."""
     T = np.asarray(T44, dtype=float).copy()
     R_x_pi = np.diag([1.0, -1.0, -1.0])
     T[:3, :3] = T[:3, :3] @ R_x_pi
     return T

def _load_profiling_from_cfg(cfg: dict) -> dict:
    """Extract ProfilingSonar configuration from a single-agent scenario."""
    main = cfg["main_agent"]
    agent = next(a for a in cfg["agents"] if a["agent_name"] == main)

    prof = next(s for s in agent["sensors"]
                if s["sensor_type"].lower() == "profilingsonar")

    conf = prof.get("configuration", {})

    rpy_deg = prof.get("rotation", [0.0, 0.0, 0.0])
    xyz     = prof.get("location", [0.0, 0.0, 0.0])

    az_fov_deg    = float(conf.get("Azimuth", 120.0))
    elevation_deg = float(conf.get("Elevation", 1.0))
    range_min_m   = float(conf.get("RangeMin", 0.5))
    range_max_m   = float(conf.get("RangeMax", 75.0))

    range_bins    = conf.get("RangeBins")
    range_res     = conf.get("RangeRes")

    if range_bins is None:
        if range_res is not None:
            range_bins = max(1, int(round((range_max_m - range_min_m) / float(range_res))))
        else:
            range_bins = 750  # HoloOcean default
    else:
        range_bins = int(range_bins)

    az_bins = conf.get("AzimuthBins")
    az_res  = conf.get("AzimuthRes")
    if az_bins is None:
        if az_res is not None:
            az_bins = max(1, int(round(az_fov_deg / float(az_res))))
        else:
            az_bins = 480  # HoloOcean default
    else:
        az_bins = int(az_bins)

    az_start_deg = -az_fov_deg / 2.0

    return {
        "xyz": xyz,
        "rpy_deg": rpy_deg,                # NWU: +pitch = nose down
        "az_start_deg": az_start_deg,
        "az_fov_deg": az_fov_deg,
        "range_start_m": range_min_m,
        "range_max_m": range_max_m,
        "azimuth_bins": az_bins,
        "range_bins": range_bins,
        "elevation_deg": elevation_deg,
    }


class Bridge(Node):
    """Step a HoloOcean scenario and publish ROS topics in NWU frames."""

    def _init_pd_setpoint_from_reset(self, sensors):
        pack = sensors.get("auv", sensors) if isinstance(sensors, dict) else sensors
        for key, value in pack.items():
            if "pose" not in key.lower():
                continue
            arr = np.asarray(value, dtype=float).reshape(-1)
            if arr.size < 16:
                break
            T = arr.reshape(4, 4)
            if self.imusocket_is_ned:
                T = _ned_to_nwu_T(T)
            x, y, z = T[:3, 3]
            yaw = math.atan2(float(T[1, 0]), float(T[0, 0]))
            self.last_pose = (x, y, z, yaw)
            self.pd_setpoint[:] = [x, y, z, 0.0, 0.0, yaw]
            self.have_pd = True
            break

    def _on_pose_cmd(self, msg: PoseStamped):
        x = float(msg.pose.position.x)
        y = float(msg.pose.position.y)
        z = float(msg.pose.position.z)
        q = msg.pose.orientation
        # yaw from quaternion (xyzw):
        siny_cosp = 2.0*(q.w*q.z + q.x*q.y)
        cosy_cosp = 1.0 - 2.0*(q.y*q.y + q.z*q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        self.pd_setpoint[:] = [x, y, z, 0.0, 0.0, yaw]
        self.have_pd = True


    def _thrusters_from_keys(self) -> np.ndarray:
        scale = float(self.thruster_scale)
      
        # Newton
        key_force = {
            "a": 0, "d": 0,                   # yaw left/right
            "j": 1.0, "l": 1.0,               # strafe
            "w": 1500.0, "s": 150.0,           # surge
            "q": 250.0, "e": 250.0            # vertical
        }

        cmd = np.zeros(8, dtype=float)
        for k in self.pressed_keys:
            v = self.thruster_keymap.get(k)
            if v is not None:
                cmd += v * scale * key_force.get(k, 1.0)

        return cmd


    def __init__(self) -> None:
        super().__init__("holoocean_bridge", automatically_declare_parameters_from_overrides=False)

        self.pressed_keys = set()

        def _on_press(key):
            if hasattr(key, "char") and key.char:
                self.pressed_keys.add(key.char)

        def _on_release(key):
            if hasattr(key, "char") and key.char and key.char in self.pressed_keys:
                self.pressed_keys.remove(key.char)

        self._kb_listener = keyboard.Listener(on_press=_on_press, on_release=_on_release)
        self._kb_listener.daemon = True
        self._kb_listener.start()

        self.declare_parameter("scenario_path", "")
        self.declare_parameter("control_mode", "thrusters")      # "pose" or "thrusters"
        self.declare_parameter("rate_hz", 0.0)                   # 0 => use ticks_per_sec from scenario
        self.declare_parameter("render_quality", 3)
        self.declare_parameter("run_headless", False)

        self.declare_parameter("map_frame_id", "map")
        self.declare_parameter("odom_frame_id", "odom")
        self.declare_parameter("base_link_frame_id", "base_link")
        self.declare_parameter("imu_frame_id", "imu_link")
        self.declare_parameter("dvl_frame_id", "dvl_link")
        self.declare_parameter("depth_frame_id", "depth_link")
        self.declare_parameter("profiling_frame_id", "profiling_link")
        self.declare_parameter("imaging_frame_id", "imaging_link")
        self.declare_parameter("publish_map_to_odom", False)
        self.declare_parameter("publish_gt_tf", False)
        self.declare_parameter("cmd_vel_topic", "cmd_vel")       
        self.declare_parameter("cmd_pose_topic", "cmd/pose")
        self.cmd_pose_topic = str(self.get_parameter("cmd_pose_topic").value)
        self.pd_setpoint = np.zeros(6, dtype=float)  # [x,y,z,roll,pitch,yaw]
        self.have_pd = False
        self.sub_pose = self.create_subscription(PoseStamped, self.cmd_pose_topic, self._on_pose_cmd, 10)


        self.declare_parameter("enable_imaging_sonar", False)
        self.declare_parameter("enable_profiling_sonar", False)
   
        self.declare_parameter("profiling_parent_frame", "base_link")
     
        self.declare_parameter("thruster_force", 10)  # Scalar    

        self.thruster_keymap = {
            'e': np.array([+1, +1, +1, +1, 0, 0, 0, 0]),   # up
            'q': np.array([-1, -1, -1, -1, 0, 0, 0, 0]),   # down
            'w': np.array([0, 0, 0, 0, +1, +1, +1, +1]),   # forward
            's': np.array([0, 0, 0, 0, -1, -1, -1, -1]),   # backward
            'a': np.array([0, 0, 0, 0, +1, -1, -1, +1]),   # yaw left
            'd': np.array([0, 0, 0, 0, -1, +1, +1, -1]),   # yaw right
            'j': np.array([0, 0, 0, 0, +1, -1, +1, -1]),   # strafe left
            'l': np.array([0, 0, 0, 0, -1, +1, -1, +1]),   # strafe right
        }

        # Covariances (SLAM-ready; diagonals only here)
        self.declare_parameter("odom_pose_cov_diag",  [1e-3, 1e-3, 1e-3, 1e-2, 1e-2, 1e-2])
        self.declare_parameter("odom_twist_cov_diag", [1e-2, 1e-2, 1e-2, 1e-1, 1e-1, 1e-1])
        self.declare_parameter("imu_ang_vel_cov_diag", [1e-4, 1e-4, 1e-4])
        self.declare_parameter("imu_lin_acc_cov_diag", [1e-2, 1e-2, 1e-2])

        # Depth reference (surface level)
        self.declare_parameter("surface_z", 0.0)

        # Prefer sim time and publish /clock ourselves
        self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])

        scenario_path = str(self.get_parameter("scenario_path").value)
        cfg: Dict = {}
        if scenario_path and os.path.exists(scenario_path):
            with open(scenario_path, "r") as f:
                cfg = json.load(f)
            self.get_logger().info(f"Loading HoloOcean scenario: {scenario_path}")
        else:
            self.get_logger().warn("No scenario_path provided; using SimpleUnderwater-AUV")
            cfg = {}

        self.mode = str(self.get_parameter("control_mode").value).strip().lower()
        self.render_qual = int(self.get_parameter("render_quality").value)
        self.headless = bool(self.get_parameter("run_headless").value)
        
        self.map_frame       = str(self.get_parameter("map_frame_id").value)
        self.odom_frame      = str(self.get_parameter("odom_frame_id").value)
        self.base_link_frame = str(self.get_parameter("base_link_frame_id").value)
        self.imu_frame       = str(self.get_parameter("imu_frame_id").value)
        self.dvl_frame       = str(self.get_parameter("dvl_frame_id").value)
        self.depth_frame     = str(self.get_parameter("depth_frame_id").value)
        self.profiling_frame = str(self.get_parameter("profiling_frame_id").value)
        self.imaging_frame   = str(self.get_parameter("imaging_frame_id").value)

        self.publish_map_to_odom = bool(self.get_parameter("publish_map_to_odom").value)
        self.publish_gt_tf       = bool(self.get_parameter("publish_gt_tf").value)
        self.cmd_topic           = str(self.get_parameter("cmd_vel_topic").value)

        self.enable_imaging   = bool(self.get_parameter("enable_imaging_sonar").value)
        if self.enable_imaging:
            self.get_logger().warn(f"Not configured for imaging sonar yet")
            self.enable_imaging = False

        self.enable_profiling = bool(self.get_parameter("enable_profiling_sonar").value)

        if self.enable_profiling:
            try:
                prof = _load_profiling_from_cfg(cfg)
                # Parent frame of SonarSocket (profiling_frame_id)
                self.prof_parent = str(self.get_parameter("profiling_parent_frame").value) or self.base_link_frame 

                self.prof_xyz         = prof["xyz"]
                self.prof_rpy         = prof["rpy_deg"]   
                self.prof_az_start_deg= prof["az_start_deg"]
                self.prof_az_fov_deg  = prof["az_fov_deg"]
                self.prof_range_start = prof["range_start_m"]
                self.prof_range_max   = prof["range_max_m"]

            except Exception as e:
                self.get_logger().fatal(f"ProfilingSonar config not found in scenario. Cannot continue. ({e})")
                rclpy.shutdown()
                return

        self.cov_pose  = [float(x) for x in self.get_parameter("odom_pose_cov_diag").value]
        self.cov_twist = [float(x) for x in self.get_parameter("odom_twist_cov_diag").value]
        self.cov_imu_w = [float(x) for x in self.get_parameter("imu_ang_vel_cov_diag").value]
        self.cov_imu_a = [float(x) for x in self.get_parameter("imu_lin_acc_cov_diag").value]

        self.surface_z = float(self.get_parameter("surface_z").value)

        self.imusocket_is_ned = True  # HoloOcean IMUSocket definition (NED)
        self.thruster_scale = float(self.get_parameter("thruster_force").value)

        if cfg.get("agents"):
            for ag in cfg["agents"]:
                filtered = []
                for s in ag.get("sensors", []):
                    t = s.get("sensor_type", "")
                    if t == "ImagingSonar"  and not self.enable_imaging:   continue
                    if t == "ProfilingSonar" and not self.enable_profiling: continue
                    filtered.append(s)
                ag["sensors"] = filtered

        self.viewport = not self.headless
        self.env = ho.make(scenario_cfg=cfg, show_viewport=self.viewport) if cfg else ho.make(scenario_name="SimpleUnderwater-AUV")
        info_str = self.env.info()  # lists agents + their sensors
        self.env.set_render_quality(self.render_qual)
        self.get_logger().info("HoloOcean environment:\n" + info_str)
        
        sensors0 = self.env.reset()
        if self.mode == "pose":
            self._init_pd_setpoint_from_reset(sensors0)

        # Choose agent
        if isinstance(self.env.agents, dict):
            self.agent = self.env.agents.get("auv", next(iter(self.env.agents.values())))
        else:
            self.agent = self.env.agents[0]
        agent_name = getattr(self.agent, "name", "auv")
        self.get_logger().info(f"Using agent: {agent_name}")

        # Activate control scheme
        scheme = 1 if self.mode == "pose" else 0
        self.agent.set_control_scheme(scheme)
        self.get_logger().info(f"Using control scheme {scheme} ({'PD Controller' if scheme==1 else 'Thruster Forces'})")

        # Timing
        hz_param = float(self.get_parameter("rate_hz").value)
        self.rate_hz = hz_param if hz_param > 0 else float(cfg.get("ticks_per_sec", 30.0))
        self.dt_nominal = 1.0 / max(1.0, self.rate_hz)
        self.dt = self.dt_nominal
        self.sim_time = 0.0
        self.last_now = None

        self.pub_clock  = self.create_publisher(ClockMsg, "/clock", 10)
        self.pub_odom   = self.create_publisher(Odometry, "/gt/odom", 10)
        self.pub_imu    = self.create_publisher(Imu, "/imu/data", 10)
        self.pub_imu_acc_bias = self.create_publisher(Vector3Stamped, "/imu/bias/accel", 10)
        self.pub_imu_gyro_bias = self.create_publisher(Vector3Stamped, "/imu/bias/gyro", 10)
        self.pub_dvl    = self.create_publisher(TwistStamped, "/dvl/twist", 10)
        self.pub_ranges = self.create_publisher(Float32MultiArray, "/dvl/ranges", 10)
        self.pub_depth  = self.create_publisher(PointStamped, "/depth", 10)  # stamped; frame = depth_frame

        self.pub_img    = self.create_publisher(Image, "/imaging/image", 10)    if self.enable_imaging   else None
        self.pub_prof   = self.create_publisher(Image, "/profiling/image", 10)  if self.enable_profiling else None
        self.pub_prof_info = self.create_publisher(Float32MultiArray, "/profiling/info", 10) if self.enable_profiling else None

        self._tf_static = StaticTransformBroadcaster(self)
        self._tf_dyn    = TransformBroadcaster(self)

        if self.publish_map_to_odom and self.map_frame != self.odom_frame:
            self._send_static_tf(self.map_frame, self.odom_frame, [0.0,0.0,0.0], [0.0,0.0,0.0])
        if self.enable_imaging:
            self._send_static_tf(self.img_parent or self.base_link_frame, self.imaging_frame, self.img_xyz, self.img_rpy)
        if self.enable_profiling:
            self._send_static_tf(self.prof_parent or self.base_link_frame, self.profiling_frame, self.prof_xyz, self.prof_rpy)

        self.sub_cmd = self.create_subscription(Twist, self.cmd_topic, self._on_cmd, 10)
        self.last_cmd = Twist()
        self.last_cmd_time = self.get_clock().now()

        self.last_pose: Optional[Tuple[float,float,float,float]] = None  # (x,y,z,yaw)

        self.get_logger().info(f"Bridge running at {self.rate_hz:.1f} Hz")
        self.wall_boot = self.create_timer(
            self.dt_nominal, self._boot_step,
            clock=RclpyClock(clock_type=ClockType.SYSTEM_TIME)
        )


        self.get_logger().info(f"Bridge ready!")

    def _boot_step(self):
        self._step()
        if self.sim_time >= 2*self.dt_nominal:
            self.wall_boot.cancel()
            self.timer = self.create_timer(self.dt_nominal, self._step)

    def _publish_clock_and_dt(self):
        """Advance and publish /clock; compute dt from measured wall/sim time."""
        self.sim_time += self.dt_nominal
        secs = int(self.sim_time)
        nsecs = int((self.sim_time - secs) * 1e9)
        clk = ClockMsg()
        clk.clock = Time(sec=secs, nanosec=nsecs)
        self.pub_clock.publish(clk)

        now = self.get_clock().now()
        if self.last_now is not None:
            dt_meas = (now - self.last_now).nanoseconds * 1e-9
            self.dt = dt_meas if dt_meas > 1e-6 else self.dt_nominal
        else:
            self.dt = self.dt_nominal
        self.last_now = now

    def _stamp(self) -> Time:
        secs = int(self.sim_time)
        nsecs = int((self.sim_time - secs) * 1e9)
        return Time(sec=secs, nanosec=nsecs)

    def _send_static_tf(self, parent: str, child: str, t_xyz, r_rpy_deg):
        tx,ty,tz = [float(v) for v in t_xyz]
        rr, rp, ry = [_rad(float(v)) for v in r_rpy_deg]
        qx,qy,qz,qw = _quat_from_rpy(rr, rp, ry)

        tf = TransformStamped()
        tf.header.stamp = self._stamp()
        tf.header.frame_id = parent
        tf.child_frame_id = child
        tf.transform.translation.x = tx
        tf.transform.translation.y = ty
        tf.transform.translation.z = tz
        tf.transform.rotation.x = qx
        tf.transform.rotation.y = qy
        tf.transform.rotation.z = qz
        tf.transform.rotation.w = qw
        self._tf_static.sendTransform(tf)
     
        self.get_logger().info(
            f"Static TF {parent} -> {child}: t=({tx:.3f},{ty:.3f},{tz:.3f})  r=({rr:.3f},{rp:.3f},{ry:.3f}) rad"
        )

    def _on_cmd(self, msg: Twist) -> None:
        self.last_cmd = msg
        self.last_cmd_time = self.get_clock().now()

    def _step(self) -> None:
        try:
            self._step_impl()
        except Exception as exc:
            self.get_logger().error(f"Exception in _step: {exc}\n{traceback.format_exc()}")

    def _step_impl(self) -> None:
        # sim clock & dt
        self._publish_clock_and_dt()
        stamp = self._stamp()

        # Build action
        action = None
        if self.mode == "thrusters":
            action = self._thrusters_from_keys()
        else:  # pose
            action = self.pd_setpoint if self.have_pd else None

        # Step the sim
        try:
            sensors = self.env.step(action)
        except TypeError:
            sensors = self.env.step({"auv": action})

        pack = sensors["auv"] if isinstance(sensors, dict) and "auv" in sensors else sensors

        # IMU data (NED -> NWU).
        # HoloOcean IMUSensor rows: [acc; gyro; accel_bias; angvel_bias] (3-vectors).
        for k in pack.keys():
            if "imu" in k.lower():
                imu_arr = np.asarray(pack[k], dtype=float).reshape(-1)
                if imu_arr.size % 3 != 0:
                    self.get_logger().warn(f"IMU payload size {imu_arr.size} not multiple of 3; skipping")
                    break

                rows = imu_arr.size // 3  # 2 (acc,gyro) or 4 (acc,gyro,acc_bias,gyro_bias)
                if rows not in (2, 4):
                    self.get_logger().warn(f"Unexpected IMU rows={rows}; expected 2 or 4; skipping")
                    break

                imu_mat = imu_arr.reshape((rows, 3))
                acc, gyro = imu_mat[0], imu_mat[1]

                if self.imusocket_is_ned:
                    acc  = _ned_to_nwu_vec(acc)
                    gyro = _ned_to_nwu_vec(gyro)

                # Publish main IMU
                msg = Imu()
                msg.header = Header(stamp=stamp, frame_id=self.imu_frame)
                msg.orientation_covariance = [ -1.0,0,0, 0,0,0, 0,0,0 ]   # orientation not provided
                msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = map(float, gyro)
                msg.angular_velocity_covariance   = _diag(self.cov_imu_w)
                self.meas_r = float(gyro[2])
                msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z = map(float, acc)
                msg.linear_acceleration_covariance = _diag(self.cov_imu_a)
                self.pub_imu.publish(msg)

                # Optional biases if present (ReturnBias=true -> 4x3)
                if rows == 4:
                    accel_bias, gyro_bias = imu_mat[2], imu_mat[3]
                    if self.imusocket_is_ned:
                        accel_bias = _ned_to_nwu_vec(accel_bias)
                        gyro_bias  = _ned_to_nwu_vec(gyro_bias)

                    b1 = Vector3Stamped()
                    b1.header.stamp = stamp; b1.header.frame_id = self.imu_frame
                    b1.vector.x, b1.vector.y, b1.vector.z = map(float, accel_bias)
                    self.pub_imu_acc_bias.publish(b1)

                    b2 = Vector3Stamped()
                    b2.header.stamp = stamp; b2.header.frame_id = self.imu_frame
                    b2.vector.x, b2.vector.y, b2.vector.z = map(float, gyro_bias)
                    self.pub_imu_gyro_bias.publish(b2)

                break  # only one IMU


        for k in pack.keys():
            if "dvl" in k.lower():
                raw = np.asarray(pack[k], dtype=float).reshape(-1)
                n = raw.size
                if n < 3:
                    self.get_logger().warn(f"DVL payload too small (n={n}); expected 3 or 7; skipping")
                    break

                vx, vy, vz = raw[0], raw[1], raw[2]
                if not np.all(np.isfinite([vx, vy, vz])):
                    self.get_logger().warn("DVL velocities non-finite; skipping sample")
                    break

                # Publish raw velocities as-is (no transforms/guessing)
                ts = TwistStamped()
                ts.header = Header(stamp=stamp, frame_id=self.dvl_frame)
                ts.twist.linear.x = float(vx)
                ts.twist.linear.y = float(vy)
                ts.twist.linear.z = float(vz)
                self.pub_dvl.publish(ts)

                # Update internal measurements for PID (no inference)
                self.meas_vx = float(vx)
                self.meas_vz = float(vz)

                # Optional ranges (ReturnRange=true -> 4 beams)
                if n >= 7:
                    r = raw[3:7]
                    if np.all(np.isfinite(r)) and r.size == 4:
                        ma = Float32MultiArray()
                        ma.data = [float(x) for x in r]
                        self.pub_ranges.publish(ma)
                    else:
                        self.get_logger().warn(f"DVL ranges non-finite or wrong size (n={n}); skipping ranges")

                elif n not in (3, 7):
                    # Accept larger vendor/debug payloads but flag it
                    self.get_logger().debug(f"DVL payload length {n} not in {{3,7}}; published first 3 only")

                break  # only one DVL

        pose_T = None
        for k in pack.keys():
            if "pose" in k.lower():
                arr = np.asarray(pack[k], dtype=float).reshape(-1)
                if arr.size >= 16:
                    T = arr.reshape(4,4)
                    if self.imusocket_is_ned:
                        T = _ned_to_nwu_T(T)      # NED -> NWU once
                    pose_T = T
                break

        if pose_T is not None:
            R = pose_T[:3,:3]
            p = pose_T[:3, 3]
            x, y, z = float(p[0]), float(p[1]), float(p[2])
            roll  = math.atan2(float(R[2,1]), float(R[2,2]))
            pitch = math.asin(max(-1.0, min(1.0, -float(R[2,0]))))
            yaw   = math.atan2(float(R[1,0]), float(R[0,0]))

            # /gt/odom
            od = Odometry()
            od.header = Header(stamp=stamp, frame_id=self.odom_frame)
            od.child_frame_id = self.base_link_frame
            qx,qy,qz,qw = _quat_from_rpy(roll, pitch, yaw)
            od.pose.pose.position.x, od.pose.pose.position.y, od.pose.pose.position.z = x,y,z
            od.pose.pose.orientation.x, od.pose.pose.orientation.y, od.pose.pose.orientation.z, od.pose.pose.orientation.w = qx,qy,qz,qw
            od.pose.covariance = _diag(self.cov_pose)

            if self.last_pose is not None:
                dt = self.dt_nominal 
                od.twist.twist.linear.x = (x - self.last_pose[0]) / dt
                od.twist.twist.linear.y = (y - self.last_pose[1]) / dt
                od.twist.twist.linear.z = (z - self.last_pose[2]) / dt
            od.twist.covariance = _diag(self.cov_twist)
            self.pub_odom.publish(od)

            # TF: odom -> base_link (optional; disabled when SLAM should own the TF tree)
            if self.publish_gt_tf:
                tf = TransformStamped()
                tf.header = od.header
                tf.child_frame_id = self.base_link_frame
                tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = x,y,z
                tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w = qx,qy,qz,qw
                self._tf_dyn.sendTransform(tf)

            # /depth (positive down)
            dmsg = PointStamped()
            dmsg.header = Header(stamp=stamp, frame_id=self.depth_frame)
            dmsg.point.z = float(self.surface_z - z)
            self.pub_depth.publish(dmsg)

            self.last_pose = (x, y, z, yaw)


        if self.enable_imaging:
            for k in pack.keys():
                if "imaging" in k.lower():
                    img = np.asarray(pack[k], dtype=np.float32)
                    if img.ndim == 2:
                        # simple log compression to mono8
                        eps = 1e-6
                        img_db = 20.0 * np.log10(np.maximum(img, eps))
                        db_min, db_max = -80.0, -20.0
                        img_norm = (img_db - db_min) / (db_max - db_min)
                        img_u8 = np.clip(img_norm * 255.0, 0, 255).astype(np.uint8)

                        msg = Image()
                        msg.header = Header(stamp=stamp, frame_id=self.imaging_frame)
                        h, w = img.shape
                        msg.height, msg.width = int(h), int(w)
                        msg.encoding = "mono8"
                        msg.step = w
                        msg.data = img_u8.tobytes()
                        self.pub_img.publish(msg)
                    break

        if self.enable_profiling:
            for k in pack.keys():
                if "profiling" in k.lower():
                    arr = np.asarray(pack[k], dtype=np.float32)
                    if arr.ndim >= 2:
                        if arr.ndim == 3 and arr.shape[-1] >= 2:
                            arr = arr[..., 1]  # intensity
                        msg = Image()
                        msg.header = Header(stamp=stamp, frame_id=self.profiling_frame)
                        h, w = arr.shape[:2]
                        msg.height, msg.width = int(h), int(w)
                        msg.encoding = "32FC1"
                        msg.step = w * 4
                        msg.data = np.ascontiguousarray(arr, dtype=np.float32).tobytes()
                        self.pub_prof.publish(msg)

                        # Minimal metadata for downstream consumers
                        az_bins = int(max(1, w))
                        rng_bins = int(max(1, h))
                        az_fov = _rad(self.prof_az_fov_deg)
                        az_start = _rad(self.prof_az_start_deg)
                        az_step = az_fov / float(max(1, az_bins - 1))
                        range_res = (self.prof_range_max - self.prof_range_start) / float(max(1, rng_bins - 1))
                        info = Float32MultiArray()
                        info.data = [float(az_start), float(az_step),
                                     float(self.prof_range_start), float(range_res),
                                     float(self.rate_hz)]
                        self.pub_prof_info.publish(info)
                    break


def main():
    rclpy.init()
    node = Bridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.env.close()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
