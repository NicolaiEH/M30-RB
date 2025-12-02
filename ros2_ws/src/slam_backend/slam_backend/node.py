#!/usr/bin/env python3
"""iSAM2 backend for NWU navigation using IMU, DVL, depth, and optional GT factors."""

from collections import deque
import math
from typing import Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import (
    PoseWithCovarianceStamped, PoseStamped, TwistStamped, PointStamped,
    Vector3Stamped, TransformStamped
)
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Imu
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from tf2_ros import TransformBroadcaster

import gtsam
from gtsam import symbol


def t2sec(t: Time) -> float:
    return float(t.sec) + float(t.nanosec) * 1e-9



class ISAM2Backend(Node):
    def __init__(self):
        super().__init__("isam2_backend")

        from rclpy.parameter import Parameter
        self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])

        p = self.declare_parameter
        self.key_period = float(p("key_period", 0.25).value)

        # IMU preintegration (continuous-time sigmas)
        self.accel_sigma = float(p("imu_accel_sigma", 0.02).value)     # m/s^2
        self.gyro_sigma  = float(p("imu_gyro_sigma",  0.002).value)    # rad/s
        self.a_bias_rw   = float(p("accel_bias_rw_sigma", 1e-4).value)
        self.g_bias_rw   = float(p("gyro_bias_rw_sigma",  1e-5).value)

        # Measurement models / gating
        self.dvl_sigma        = float(p("dvl_vel_sigma", 0.10).value)   # m/s per axis
        self.depth_sigma      = float(p("depth_sigma",   0.10).value)   # m on z
        self.dvl_staleness    = float(p("dvl_staleness_sec", 0.6).value)
        self.depth_staleness  = float(p("depth_staleness_sec", 0.6).value)
        self.use_dvl_factor   = bool(p("use_dvl_factor", True).value)
        self.dvl_use_huber    = bool(p("dvl_use_huber", True).value)
        self.use_gt_factor    = bool(p("use_gt_factor", False).value)
        self.gt_sigma         = float(p("gt_pose_sigma", 3.0).value)

        # Gentle priors
        self.prior_pose_sigma = float(p("prior_pose_sigma", 1.0).value)
        self.prior_vel_sigma  = float(p("prior_vel_sigma",  0.3).value)
        self.prior_bias_sigma = float(p("prior_bias_sigma", 0.01).value)

        # Frames
        self.map_frame, self.odom_frame, self.base_frame = "map", "odom", "base_link"

        # Publishers
        self.pub_pose = self.create_publisher(PoseWithCovarianceStamped, "/slam/pose", 10)
        self.pub_odom = self.create_publisher(Odometry, "/slam/odom", 10)
        self.pub_path = self.create_publisher(Path, "/slam/path", 10)
        self.pub_ba   = self.create_publisher(Vector3Stamped, "/slam/imu_bias/accel", 10)
        self.pub_bg   = self.create_publisher(Vector3Stamped, "/slam/imu_bias/gyro", 10)
        self.pub_diag = self.create_publisher(DiagnosticArray, "/slam/diagnostics", 10)
        self.tfb      = TransformBroadcaster(self)
        self.publish_map_to_odom_tf = bool(p("publish_map_to_odom_tf", False).value)

        # Subscribers
        self.create_subscription(Imu,          "/imu/data",       self.on_imu,   100)
        self.create_subscription(Vector3Stamped, "/imu/bias/accel", self.on_ba,    10)
        self.create_subscription(Vector3Stamped, "/imu/bias/gyro",  self.on_bg,    10)
        self.create_subscription(TwistStamped, "/dvl/twist",      self.on_dvl,    30)
        self.create_subscription(PointStamped, "/depth",          self.on_depth,  30)
        self.create_subscription(Odometry,     "/gt/odom",        self.on_gt,     10)

        # IMU preintegration params (gravity + noise)
        g = 9.81
        pim_params = gtsam.PreintegrationParams.MakeSharedU(g)  # +Z up (NWU world)
        pim_params.setAccelerometerCovariance(np.eye(3) * (self.accel_sigma ** 2))
        pim_params.setGyroscopeCovariance(np.eye(3) * (self.gyro_sigma  ** 2))
        pim_params.setIntegrationCovariance(np.eye(3) * 1e-8)
        self.noise_bias = gtsam.noiseModel.Isotropic.Sigma(6, self.prior_bias_sigma)

        # Factor graph state
        self.graph  = gtsam.NonlinearFactorGraph()
        self.values = gtsam.Values()
        self.isam   = gtsam.ISAM2()

        # Keys
        self.k = 0
        self.X = lambda i: symbol('x', i)   # Pose3
        self.V = lambda i: symbol('v', i)   # Vector3
        self.B = lambda i: symbol('b', i)   # imuBias::ConstantBias

        # Initial state (identity in map)
        self.bias0 = gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3))
        self.values.insert(self.X(self.k), gtsam.Pose3())
        self.values.insert(self.V(self.k), np.zeros(3))
        self.values.insert(self.B(self.k), self.bias0)
        self.graph.add(gtsam.PriorFactorPose3(self.X(self.k), gtsam.Pose3(),
                                              gtsam.noiseModel.Isotropic.Sigma(6, self.prior_pose_sigma)))
        self.graph.add(gtsam.PriorFactorVector(self.V(self.k), np.zeros(3),
                                               gtsam.noiseModel.Isotropic.Sigma(3, self.prior_vel_sigma)))
        self.graph.add(gtsam.PriorFactorConstantBias(self.B(self.k), self.bias0, self.noise_bias))
        self.isam.update(self.graph, self.values)  # register X0,V0,B0
        self.graph.resize(0)

        # Preintegrator
        self.pim_params = pim_params
        self.pim = gtsam.PreintegratedImuMeasurements(self.pim_params, self.bias0)
        self.t_last_imu: Optional[float] = None

        # Caches
        self.dvl_buf: deque[Tuple[float, np.ndarray]] = deque(maxlen=2)  # (t, v_body)
        self.last_depth: Optional[Tuple[float, float]] = None            # (t, z_down)
        self.last_gt: Optional[Odometry] = None
        self.path = Path(); self.path.header.frame_id = self.map_frame

        self._last_ros_now = 0.0
        self.create_timer(self.key_period, self.on_keyframe)

        self.get_logger().info("iSAM2 backend ready (IMU preintegration + DVL vel + depth z).")

    def on_ba(self, msg: Vector3Stamped):
        if self.k == 0:
            self.bias0 = gtsam.imuBias.ConstantBias(
                np.array([msg.vector.x, msg.vector.y, msg.vector.z]),
                self.bias0.gyroscope()
            )
            self.values.update(self.B(0), self.bias0)
            self.pim = gtsam.PreintegratedImuMeasurements(self.pim_params, self.bias0)

    def on_bg(self, msg: Vector3Stamped):
        if self.k == 0:
            self.bias0 = gtsam.imuBias.ConstantBias(
                self.bias0.accelerometer(),
                np.array([msg.vector.x, msg.vector.y, msg.vector.z])
            )
            self.values.update(self.B(0), self.bias0)
            self.pim = gtsam.PreintegratedImuMeasurements(self.pim_params, self.bias0)

    def on_imu(self, msg: Imu):
        t = t2sec(msg.header.stamp)
        acc = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z], float)
        gyr = np.array([msg.angular_velocity.x,    msg.angular_velocity.y,    msg.angular_velocity.z],    float)
        if self.t_last_imu is None:
            self.t_last_imu = t
            return
        dt = max(1e-6, t - self.t_last_imu)
        self.t_last_imu = t
        self.pim.integrateMeasurement(acc, gyr, dt)

    def on_dvl(self, msg: TwistStamped):
        v_body = np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z], float)
        self.dvl_buf.append((t2sec(msg.header.stamp), v_body))

    def on_depth(self, msg: PointStamped):
        self.last_depth = (t2sec(msg.header.stamp), float(msg.point.z))  # positive-down

    def on_gt(self, msg: Odometry):
        self.last_gt = msg

    def _interp_dvl_body(self, t_query: float) -> Optional[np.ndarray]:
        """2-point linear interpolation of body-frame DVL at t_query."""
        if not self.dvl_buf:
            return None
        if len(self.dvl_buf) == 1:
            return self.dvl_buf[0][1]
        (t0, v0), (t1, v1) = self.dvl_buf[0], self.dvl_buf[1]
        if t1 <= t0:
            return v1
        if t_query <= t0:
            return v0
        if t_query >= t1:
            return v1
        a = (t_query - t0) / (t1 - t0)
        return (1.0 - a) * v0 + a * v1

    def on_keyframe(self):
        # Reset caches if /clock jumps back.
        now_ros = float(self.get_clock().now().nanoseconds) * 1e-9
        if now_ros + 1e-6 < self._last_ros_now:
            self.path = Path(); self.path.header.frame_id = self.map_frame
            self.dvl_buf.clear(); self.last_depth = None; self.last_gt = None
            self.pim.resetIntegration()
            self._last_ros_now = now_ros
            return
        self._last_ros_now = now_ros

        if self.pim.deltaTij() <= 0.0:
            return

        k0, k1 = self.k, self.k + 1

        # IMU factor + bias random-walk
        self.graph.add(gtsam.ImuFactor(self.X(k0), self.V(k0), self.X(k1), self.V(k1), self.B(k0), self.pim))
        bias_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.hstack([np.ones(3) * self.a_bias_rw, np.ones(3) * self.g_bias_rw]))
        self.graph.add(gtsam.BetweenFactorConstantBias(self.B(k0), self.B(k1),
                                                       gtsam.imuBias.ConstantBias(), bias_noise))

        # Predict initial for new state
        nav0 = gtsam.NavState(self.values.atPose3(self.X(k0)), self.values.atVector(self.V(k0)))
        prop = self.pim.predict(nav0, self.values.atConstantBias(self.B(k0)))
        new_vals = gtsam.Values()
        new_vals.insert(self.X(k1), prop.pose())
        new_vals.insert(self.V(k1), prop.velocity())
        new_vals.insert(self.B(k1), self.values.atConstantBias(self.B(k0)))

        # DVL factor (interpolate to keyframe time; warm-up grace N)
        t_now = self.t_last_imu if self.t_last_imu is not None else now_ros
        warmup_N, gate_sigma = 6, 3.0
        if self.k >= warmup_N and len(self.dvl_buf) and self.use_dvl_factor:
            t_newest, _ = self.dvl_buf[-1]
            age = max(0.0, t_now - t_newest)
            if age <= self.dvl_staleness:
                v_body = self._interp_dvl_body(t_now)
                if v_body is not None:
                    Rwb = prop.pose().rotation().matrix()
                    v_world_meas = Rwb @ v_body
                    res = v_world_meas - prop.velocity()
                    gate = gate_sigma * self.dvl_sigma * math.sqrt(3.0)
                    if np.linalg.norm(res) < gate:
                        base  = gtsam.noiseModel.Isotropic.Sigma(3, self.dvl_sigma)
                        if self.dvl_use_huber:
                            huber = gtsam.noiseModel.mEstimator.Huber(1.345)
                            noise = gtsam.noiseModel.Robust.Create(huber, base)
                        else:
                            noise = base
                        self.graph.add(gtsam.PriorFactorVector(self.V(k1), v_world_meas, noise))
                    else:
                        self.get_logger().warn(
                            f"DVL residual gate FAIL: ||res||={np.linalg.norm(res):.3f} > {gate:.3f}")
            else:
                self.get_logger().warn(f"DVL stale: age={age:.3f}s > {self.dvl_staleness:.3f}s")

        # Depth factor (z only; positive-down -> map z = -depth)
        if self.last_depth is not None:
            t_dep, z_down = self.last_depth
            age = max(0.0, t_now - t_dep)
            if age <= self.depth_staleness:
                z_map = -float(z_down)  # NWU
                meas = gtsam.Pose3(gtsam.Rot3(), np.array([0.0, 0.0, z_map]))
                sig = np.array([1e3, 1e3, 1e3, 1e3, 1e3, self.depth_sigma])  # rpy,xyz
                self.graph.add(gtsam.PriorFactorPose3(self.X(k1), meas,
                                 gtsam.noiseModel.Diagonal.Sigmas(sig)))
            else:
                self.get_logger().warn(f"Depth stale: age={age:.3f}s > {self.depth_staleness:.3f}s")

        # Optional soft GT pose factor
        if self.use_gt_factor and self.last_gt is not None:
            gt = self.last_gt
            px, py, pz = gt.pose.pose.position.x, gt.pose.pose.position.y, gt.pose.pose.position.z
            qx, qy, qz, qw = (gt.pose.pose.orientation.x, gt.pose.pose.orientation.y,
                              gt.pose.pose.orientation.z, gt.pose.pose.orientation.w)
            self.graph.add(gtsam.PriorFactorPose3(
                self.X(k1),
                gtsam.Pose3(gtsam.Rot3.Quaternion(qw, qx, qy, qz), np.array([px, py, pz])),
                gtsam.noiseModel.Isotropic.Sigma(6, self.gt_sigma)
            ))

        self.isam.update(self.graph, new_vals)
        result = self.isam.calculateEstimate()
        self.values.clear(); self.graph.resize(0)
        self.values.insert(self.X(k1), result.atPose3(self.X(k1)))
        self.values.insert(self.V(k1), result.atVector(self.V(k1)))
        self.values.insert(self.B(k1), result.atConstantBias(self.B(k1)))
        self.k = k1
        self.pim.resetIntegration()

        # Publish
        self.publish_outputs(result)

    def publish_outputs(self, est: gtsam.Values):
        pose: gtsam.Pose3 = est.atPose3(self.X(self.k))
        vel  = est.atVector(self.V(self.k))
        bias: gtsam.imuBias.ConstantBias = est.atConstantBias(self.B(self.k))

        p = pose.translation(); R = pose.rotation(); q = R.toQuaternion()  # (w,x,y,z)

        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = rclpy.time.Time(seconds=self.t_last_imu or 0.0).to_msg()
        msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z = p[0], p[1], p[2]
        msg.pose.pose.orientation.w, msg.pose.pose.orientation.x = q.w(), q.x()
        msg.pose.pose.orientation.y, msg.pose.pose.orientation.z = q.y(), q.z()

        try:
            marg = gtsam.Marginals(self.isam.getFactorsUnsafe(), self.isam.calculateEstimate())
            cov = marg.marginalCovariance(self.X(self.k))  # 6x6 (rpy,xyz)
            ros = np.zeros((6, 6))
            ros[0:3, 0:3] = cov[3:6, 3:6]  # xyz
            ros[3:6, 3:6] = cov[0:3, 0:3]  # rpy
            msg.pose.covariance = list(ros.reshape(-1))
        except Exception:
            msg.pose.covariance = [0.0] * 36
        self.pub_pose.publish(msg)

        od = Odometry()
        od.header = msg.header
        od.child_frame_id = self.base_frame
        od.pose = msg.pose
        od.twist.twist.linear.x, od.twist.twist.linear.y, od.twist.twist.linear.z = vel[0], vel[1], vel[2]
        self.pub_odom.publish(od)

        self.path.header.stamp = msg.header.stamp
        ps = PoseStamped(); ps.header = msg.header; ps.pose = msg.pose.pose
        self.path.poses.append(ps)
        self.pub_path.publish(self.path)

        ba, bg = bias.accelerometer(), bias.gyroscope()
        b1 = Vector3Stamped(); b1.header = msg.header; b1.vector.x, b1.vector.y, b1.vector.z = ba[0], ba[1], ba[2]
        b2 = Vector3Stamped(); b2.header = msg.header; b2.vector.x, b2.vector.y, b2.vector.z = bg[0], bg[1], bg[2]
        self.pub_ba.publish(b1); self.pub_bg.publish(b2)

        if self.publish_map_to_odom_tf:
            tf_msgs = []

            if self.map_frame != self.odom_frame:
                tf_map_odom = TransformStamped()
                tf_map_odom.header.stamp = od.header.stamp
                tf_map_odom.header.frame_id = self.map_frame
                tf_map_odom.child_frame_id = self.odom_frame
                tf_map_odom.transform.translation.x = 0.0
                tf_map_odom.transform.translation.y = 0.0
                tf_map_odom.transform.translation.z = 0.0
                tf_map_odom.transform.rotation.x = 0.0
                tf_map_odom.transform.rotation.y = 0.0
                tf_map_odom.transform.rotation.z = 0.0
                tf_map_odom.transform.rotation.w = 1.0
                tf_msgs.append(tf_map_odom)

            tf_odom_base = TransformStamped()
            tf_odom_base.header.stamp = od.header.stamp
            tf_odom_base.header.frame_id = self.odom_frame
            tf_odom_base.child_frame_id = self.base_frame
            tf_odom_base.transform.translation.x = p[0]
            tf_odom_base.transform.translation.y = p[1]
            tf_odom_base.transform.translation.z = p[2]
            tf_odom_base.transform.rotation.w = q.w()
            tf_odom_base.transform.rotation.x = q.x()
            tf_odom_base.transform.rotation.y = q.y()
            tf_odom_base.transform.rotation.z = q.z()
            tf_msgs.append(tf_odom_base)

            self.tfb.sendTransform(tf_msgs)

        # Diagnostics
        d = DiagnosticArray(); d.header = msg.header
        st = DiagnosticStatus(level=DiagnosticStatus.OK, name="isam2_backend", message="OK",
                              values=[KeyValue(key="k", value=str(self.k)),
                                      KeyValue(key="factors", value=str(self.isam.getFactorsUnsafe().size()))])
        d.status = [st]; self.pub_diag.publish(d)


def main():
    rclpy.init()
    rclpy.spin(ISAM2Backend())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
