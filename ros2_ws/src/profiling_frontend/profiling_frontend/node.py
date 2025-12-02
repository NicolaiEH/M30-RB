#!/usr/bin/env python3
"""Back-project profiling sonar pings into map-frame point clouds."""

from collections import deque
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.parameter import Parameter

from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from tf2_ros import Buffer, TransformListener


# ---------- helpers ----------
def quaternion_matrix(q_xyzw):
    """4x4 rotation from quaternion [x,y,z,w] (no translation)."""
    x, y, z, w = q_xyzw
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    T = np.eye(4, dtype=float)
    T[:3, :3] = np.array([
        [1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy)],
        [2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx)],
        [2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)]
    ], dtype=float)
    return T


def tf_to_R_p(transform):
    """geometry_msgs/Transform -> (R(3x3), p(3,))."""
    q = transform.rotation
    T = quaternion_matrix([q.x, q.y, q.z, q.w])
    t = transform.translation
    p = np.array([t.x, t.y, t.z], dtype=float)
    return T[:3, :3], p


class ProfilingFrontend(Node):
    def __init__(self):
        super().__init__("profiling_frontend")

        self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])

        self.amp_min = float(self.declare_parameter("amp_min", 0.1).value)
        self.bottom_gate_m = float(self.declare_parameter("bottom_gate_m", 2.0).value)
        self.voxel = float(self.declare_parameter("voxel_downsample", 0.0).value)
        self.points_topic = str(self.declare_parameter("points_topic", "/profiling/points").value)
        self.map_frame = str(self.declare_parameter("map_frame", "map").value)
        self.base_link_frame = str(self.declare_parameter("base_link_frame", "base_link").value)
        self.profiling_frame = str(self.declare_parameter("profiling_frame", "profiling_link").value)

        self.tf_timeout_sec = float(self.declare_parameter("tf_timeout_sec", 2.0).value)
        self.tf_backoff_sec = float(self.declare_parameter("tf_backoff_sec", 0.10).value)
        self.filter_hz = float(self.declare_parameter("filter_hz", 50.0).value)

        self.tvg_enable = bool(self.declare_parameter("tvg_enable", True).value)
        self.tvg_alpha_db_per_m = float(self.declare_parameter("tvg_alpha_db_per_m", 0.035).value)
        self.cfar_win = int(self.declare_parameter("cfar_win", 12).value)
        self.cfar_guard = int(self.declare_parameter("cfar_guard", 2).value)
        self.cfar_k = float(self.declare_parameter("cfar_k", 1.5).value)
        self.smooth_med_win = int(self.declare_parameter("smooth_med_win", 5).value)

        self.tf_timeout = Duration(seconds=max(0.0, self.tf_timeout_sec))

        self.tfbuf = Buffer(cache_time=Duration(seconds=30.0))
        self.tfl = TransformListener(self.tfbuf, self)

        self.create_subscription(Float32MultiArray, "/profiling/info", self.on_info, 10)
        self.create_subscription(Image, "/profiling/image", self._enqueue_image, 10)
        self.pub_pts = self.create_publisher(PointCloud2, self.points_topic, 10)

        try:
            if not self.tfbuf.can_transform(self.map_frame, self.base_link_frame,
                                            rclpy.time.Time(), Duration(seconds=2.0)):
                self.get_logger().warn("TF tree not ready at startup (map<->base_link); continuing anyway.")
        except Exception:
            pass

        self.a0 = self.da = self.r0 = self.dr = None
        self.rate_hz = None
        self.az_axis = None
        self.r_axis = None
        self.Tbp = None
        self._img_q = deque()
        self.create_timer(1.0 / max(1.0, self.filter_hz), self._drain_queue)

        self._last_now = 0.0
        self._dbg_last_log_ns = 0

        self.get_logger().info("profiling_frontend: ready.")

    # -------- info / metadata --------
    def on_info(self, msg: Float32MultiArray):
        if len(msg.data) >= 5:
            self.a0 = float(msg.data[0]); self.da = float(msg.data[1])
            self.r0 = float(msg.data[2]); self.dr = float(msg.data[3])
            self.rate_hz = float(msg.data[4])

    def _ensure_T_base_profiling(self):
        """Fetch static base_link<-profiling_link once (time=0 -> latest TF)."""
        if self.Tbp is not None:
            return
        try:
            tf = self.tfbuf.lookup_transform(self.base_link_frame, self.profiling_frame,
                                             rclpy.time.Time(), Duration(seconds=1.0))
            t = tf.transform.translation; q = tf.transform.rotation
            T = quaternion_matrix([q.x, q.y, q.z, q.w]); T[:3, 3] = [t.x, t.y, t.z]
            self.Tbp = T
        except Exception:
            pass  # try again later

    # -------- TF-aware queue (MessageFilter-like) --------
    def _enqueue_image(self, msg: Image):
        # bag loop guard (when /clock jumps back)
        now = float(self.get_clock().now().nanoseconds) * 1e-9
        if now + 1e-6 < self._last_now:
            self.Tbp = None
            self._img_q.clear()
        self._last_now = now
        self._img_q.append(msg)

    def _drain_queue(self):
        while self._img_q:
            msg = self._img_q[0]
            if None in (self.a0, self.da, self.r0, self.dr):
                return
            self._ensure_T_base_profiling()
            if self.Tbp is None:
                return

            h, w = int(msg.height), int(msg.width)
            if h == 0 or w == 0 or self.dr is None or self.dr <= 0.0:
                self._img_q.popleft()
                continue

            # Rebuild axes if dimensions change.
            if (self.az_axis is None) or (self.az_axis.size != w):
                self.az_axis = self.a0 + self.da * np.arange(w, dtype=np.float32)
            if (self.r_axis is None) or (self.r_axis.size != h):
                self.r_axis = self.r0 + self.dr * np.arange(h, dtype=np.float32)

            # Prefer ping_time - backoff, clamp to time 0.
            stamp = Time.from_msg(msg.header.stamp)
            try_time = stamp - Duration(seconds=max(0.0, self.tf_backoff_sec))
            if try_time.nanoseconds < 0:
                try_time = Time(seconds=0.0)

            tf = None
            if self.tfbuf.can_transform(self.map_frame, self.base_link_frame,
                                        try_time, Duration(seconds=0.0)):
                tf = self.tfbuf.lookup_transform(self.map_frame, self.base_link_frame,
                                                 try_time, Duration(seconds=0.0))
            if tf is None and self.tfbuf.can_transform(self.map_frame, self.base_link_frame,
                                                       rclpy.time.Time(), Duration(seconds=0.0)):
                tf = self.tfbuf.lookup_transform(self.map_frame, self.base_link_frame,
                                                 rclpy.time.Time(), Duration(seconds=0.0))
            if tf is None:
                break

            self._process_image(msg, tf)
            self._img_q.popleft()

    def _process_image(self, msg: Image, tf):
        h, w = int(msg.height), int(msg.width)
        A = np.frombuffer(msg.data, dtype=np.float32, count=h*w).reshape((h, w))

        # Log dynamic range once per second.
        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self._dbg_last_log_ns >= int(1e9):
            self._dbg_last_log_ns = now_ns
            self.get_logger().info(
                f"[profiling_frontend] image {w}x{h}  min={float(np.nanmin(A)):.3g}  "
                f"max={float(np.nanmax(A)):.3g}"
            )

        # Map<-base (at ping - backoff or latest)
        Rmb, pmb = tf_to_R_p(tf.transform)

        # Axes (cached)
        az = self.az_axis
        ranges = self.r_axis

        # TVG equalization (optional)
        if self.tvg_enable:
            eps = 1e-6
            A_db = 20.0 * np.log10(np.maximum(A, eps))
            r = np.maximum(ranges.reshape(-1, 1), 1e-3)
            tvg = 20.0 * np.log10(r) + 2.0 * self.tvg_alpha_db_per_m * r
            A_eq = 10.0 ** ((A_db + tvg) / 20.0)
        else:
            A_eq = A

        # CA-CFAR first-hit per column
        gate_idx = int(max(0, np.floor((self.bottom_gate_m - self.r0) / self.dr)))
        gate_idx = min(gate_idx, h - 1)
        g = max(0, self.cfar_guard)
        w_win = max(0, self.cfar_win)
        k = max(0.0, self.cfar_k)
        picks = np.full(w, gate_idx, dtype=np.int32)

        for j in range(w):
            col = A_eq[:, j]
            prefix = np.zeros(h + 1, dtype=np.float64)
            np.cumsum(col, dtype=np.float64, out=prefix[1:])
            first_hit = None
            for i in range(gate_idx, h):
                lo = max(0, i - (g + w_win))
                hi = min(h, i + (g + w_win) + 1)
                if hi - lo <= 1:
                    continue
                guard_lo = max(lo, i - g)
                guard_hi = min(hi, i + g + 1)
                sum_total = prefix[hi] - prefix[lo]
                cnt_total = hi - lo
                sum_guard = prefix[guard_hi] - prefix[guard_lo]
                cnt_guard = guard_hi - guard_lo
                cnt_ref = cnt_total - cnt_guard
                if cnt_ref <= 0:
                    continue
                ref_mean = (sum_total - sum_guard) / cnt_ref
                thr = k * ref_mean
                if not np.isfinite(thr):
                    continue
                amp = col[i]
                if amp >= max(thr, self.amp_min):
                    first_hit = i
                    break
            if first_hit is not None:
                picks[j] = first_hit

        # continuity smoothing (odd-length median)
        m = max(1, self.smooth_med_win)
        if m > 1:
            if m % 2 == 0:
                m += 1
            pad = m // 2
            padded = np.pad(picks, (pad, pad), mode="edge")
            picks = np.array([int(np.median(padded[j:j+m])) for j in range(w)], dtype=np.int32)

        # Fallback if CFAR yields nothing useful.
        valid = (picks >= 0) & np.isfinite(picks)
        if not np.any(valid):
            tail = A_eq[gate_idx:, :]
            if tail.size > 0:
                picks = gate_idx + np.argmax(tail, axis=0).astype(np.int32)
                valid = (picks >= 0)
        if not np.any(valid):
            picks = np.full(w, gate_idx, dtype=np.int32)
            valid = np.ones(w, dtype=bool)

        # Back-project.
        ranges_sel = ranges[picks]
        c = np.cos(az); s = np.sin(az)
        Pp = np.stack([ranges_sel * c,
                       ranges_sel * s,
                       np.zeros_like(ranges_sel),
                       np.ones_like(ranges_sel)], axis=0)   # 4xW

        Pb = self.Tbp @ Pp
        Pm = (Rmb @ Pb[:3, :]) + pmb.reshape(3, 1)

        pts = Pm.T
        if self.voxel > 0.0:
            vox = np.round(pts / self.voxel).astype(np.int64)
            _, uniq = np.unique(vox, axis=0, return_index=True)
            pts = pts[uniq]

        header = msg.header
        header.frame_id = self.map_frame
        cloud = pc2.create_cloud_xyz32(header, pts.astype(np.float32).tolist())
        self.pub_pts.publish(cloud)


def main():
    rclpy.init()
    rclpy.spin(ProfilingFrontend())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
