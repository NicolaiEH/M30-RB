#!/usr/bin/env python3
"""Keyboard teleop node publishing normalized Twist commands."""

import sys, termios, tty, select, time, atexit
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

HELP = """
WASD + Q/E teleop  (normalized thrusters)
-----------------------------------------
  w/s : surge forward/back   -> /cmd_vel.linear.x  in [-1..+1]
  a/d : yaw left/right       -> /cmd_vel.angular.z in [-1..+1]
  q/e : dive/surface (heave) -> /cmd_vel.linear.z  in [-1..+1]
  space: stop all (zeros)
  h: show this help

Notes:
- Keys are independent per-axis. You can hold W and tap/hold D (and Q/E) simultaneously.
- Each axis "holds" its last pressed value for a short timeout; tap again to refresh.
- Ctrl-C to quit.
""".strip()


def getch(timeout: float = 0.0) -> str:
    """Non-blocking get one char from stdin ('' if none)."""
    fd = sys.stdin.fileno()
    r, _, _ = select.select([fd], [], [], timeout)
    if not r:
        return ''
    return sys.stdin.read(1)


class TeleopNode(Node):
    def __init__(self):
        super().__init__('teleop_wasd')

        self._orig_stdin = sys.stdin
        self._tty_file = None
        if not sys.stdin.isatty():
            try:
                self._tty_file = open('/dev/tty', 'r')
                sys.stdin = self._tty_file
                self.get_logger().warn('stdin is not a TTY; using /dev/tty for keyboard input.')
            except OSError as exc:
                self.get_logger().error(f'stdin is not a TTY and /dev/tty could not be opened: {exc}')
                raise

        # Params
        self.declare_parameter('rate_hz', 200.0)           # publish rate
        self.declare_parameter('key_hold_timeout', 0.40)   # per-axis hold before auto-zero (seconds)
        self.rate_hz = float(self.get_parameter('rate_hz').value)
        self.key_hold_timeout = float(self.get_parameter('key_hold_timeout').value)

        # Publisher
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Axis state (normalized command in [-1..+1]) with independent expiries
        now = time.monotonic()
        self._axis = {
            'x':   {'val': 0.0, 'exp': now},  # linear.x (surge)
            'yaw': {'val': 0.0, 'exp': now},  # angular.z (yaw)
            'z':   {'val': 0.0, 'exp': now},  # linear.z (heave)
        }

        # Key -> (axis, value). Pressing a key only updates that axis.
        self._keymap = {
            'w': ('x',   +1.0),
            's': ('x',   -1.0),
            'a': ('yaw', +1.0),
            'd': ('yaw', -1.0),
            'q': ('z',   +1.0),  # dive (down)
            'e': ('z',   -1.0),  # surface (up)
        }

        # TTY setup (keep signals so Ctrl-C still works)
        self._fd = sys.stdin.fileno()
        self._old_tty = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        atexit.register(self._restore_tty)

        self.get_logger().info(HELP)
        self.get_logger().info(f"Publishing normalized /cmd_vel at {self.rate_hz:.1f} Hz")

        # Timer loop
        self.timer = self.create_timer(1.0 / max(1.0, self.rate_hz), self._tick)

    def _restore_tty(self):
        try:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_tty)
        except Exception:
            pass
        finally:
            if self._tty_file is not None:
                try:
                    self._tty_file.close()
                except Exception:
                    pass
                sys.stdin = self._orig_stdin
            sys.stdout.write('\n')
            sys.stdout.flush()

    def _zero_all(self):
        for a in self._axis.values():
            a['val'] = 0.0
            a['exp'] = time.monotonic()

    def _tick(self):
        # Read all pending keys this tick (allows quick combos)
        ch = getch(0.0)
        now = time.monotonic()

        while ch:
            k = ch.lower()

            if k in self._keymap:
                axis, val = self._keymap[k]
                self._axis[axis]['val'] = val
                self._axis[axis]['exp'] = now + self.key_hold_timeout

            elif k == ' ':
                self._zero_all()

            elif k == 'h':
                self.get_logger().info('\n' + HELP)

            elif ord(k) == 3:  # Ctrl-C (ETX)
                raise KeyboardInterrupt

            ch = getch(0.0)

        # Expire axes independently (so W can keep going while you tap D)
        now = time.monotonic()
        for a in self._axis.values():
            if a['exp'] <= now:
                a['val'] = 0.0

        # Publish combined command
        msg = Twist()
        msg.linear.x  = self._axis['x']['val']
        msg.angular.z = self._axis['yaw']['val']
        msg.linear.z  = self._axis['z']['val']
        self.pub.publish(msg)

        # Lightweight status line
        sys.stdout.write(
            '\rCmd (norm): surge={:+.1f}  yaw={:+.1f}  heave={:+.1f}   '.format(
                msg.linear.x, msg.angular.z, msg.linear.z
            )
        )
        sys.stdout.flush()


def main():
    rclpy.init()
    node = TeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.pub.publish(Twist())  # send stop on exit
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
