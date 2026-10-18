"""ROS 2 bridge manager for connecting Gymnasium CarGazeboEnv to live Gazebo simulation.

Communicates with ros_gz_bridge via standard ROS 2 topics:
- Subscribes to dual 3D directional LiDAR PointCloud2 (/tractor/lidar/{front,rear}/points)
- Subscribes to Wheel Odometry (/model/tractor/wheel_odometry)
- Publishes Ackermann steering angle and drive velocity (/cmd_vel, /model/tractor/steering_angle, /model/tractor/rear_axle/cmd_vel)
"""

from __future__ import annotations

import math
import threading
import time


import numpy as np

from src.translate.config import (
    FRONT_LIDAR_PITCH_RAD,
    FRONT_LIDAR_X,
    FRONT_LIDAR_Z,
    REAR_LIDAR_PITCH_RAD,
    REAR_LIDAR_X,
    REAR_LIDAR_Z,
    WHEELBASE,
)


import sys
from pathlib import Path

# Auto-discover ROS 2 site-packages if running inside container or host system with ROS
_ros_pypath = Path("/opt/ros/jazzy/lib/python3.12/site-packages")
if _ros_pypath.exists() and str(_ros_pypath) not in sys.path:
    sys.path.append(str(_ros_pypath))


class Ros2GazeboBridge:
    """Manages ROS 2 node lifecycle, topic subscriptions, and command publishing."""

    def __init__(self, node_name: str = "car_gazebo_env_bridge") -> None:
        try:
            import rclpy
            from geometry_msgs.msg import Twist
            from nav_msgs.msg import Odometry
            from sensor_msgs.msg import Imu, JointState, PointCloud2
            from std_msgs.msg import Float64
        except ImportError as exc:
            raise RuntimeError(
                "ROS 2 packages (rclpy, sensor_msgs, etc.) are not available in this environment.\n"
                "To run live Gazebo simulation, execute inside the distrobox container:\n"
                "  distrobox enter gazebo-harmonic\n"
            ) from exc

        self.rclpy = rclpy
        self.Twist = Twist
        self.Odometry = Odometry
        self.PointCloud2 = PointCloud2
        self.Float64 = Float64
        self.JointState = JointState
        self.Imu = Imu

        try:
            from ros_gz_interfaces.msg import Contacts

            self.Contacts: Any | None = Contacts
        except ImportError:
            self.Contacts = None

        if not self.rclpy.ok():
            self.rclpy.init()

        self.node = self.rclpy.create_node(node_name)
        self.lock = threading.Lock()

        # Cached state from Gazebo
        self.latest_point_clouds: dict[str, np.ndarray] = {}
        self.latest_odom: dict[str, float] | None = None
        self.latest_steering_angle: float = 0.0
        self.latest_imu: dict[str, float] | None = None
        self.latest_collision: bool = False
        self._last_vx: float = 0.0
        self._last_vx_time: float = 0.0
        self._ax: float = 0.0

        # Subscriptions: 3D Directional LiDARs (PointCloud2)
        self.sub_cloud_front = self.node.create_subscription(
            PointCloud2,
            "/tractor/lidar/front/points",
            lambda m: self._on_point_cloud("front", m),
            10,
        )
        self.sub_cloud_rear = self.node.create_subscription(
            PointCloud2,
            "/tractor/lidar/rear/points",
            lambda m: self._on_point_cloud("rear", m),
            10,
        )

        # Subscription: World Odometry & Wheel Odometry
        self.sub_odom = self.node.create_subscription(
            Odometry, "/model/tractor/odometry", self._on_odom, 10
        )
        self.sub_wheel_odom = self.node.create_subscription(
            Odometry, "/model/tractor/wheel_odometry", self._on_wheel_odom, 10
        )

        # Subscription: Tractor Contacts (Gazebo Harmonic Ground Truth)
        if self.Contacts is not None:
            self.sub_contacts = self.node.create_subscription(
                self.Contacts, "/model/tractor/contacts", self._on_contacts, 10
            )

        # Subscription: Articulated Joint States (Physical Steering Knuckles)
        self.sub_joint_state = self.node.create_subscription(
            JointState, "/model/tractor/joint_state", self._on_joint_state, 10
        )

        # Subscription: Chassis IMU (Linear Acceleration & Angular Velocity)
        self.sub_imu = self.node.create_subscription(
            Imu, "/model/tractor/imu", self._on_imu, 10
        )

        # Publishers: Commands
        self.pub_cmd_vel = self.node.create_publisher(Twist, "/cmd_vel", 10)
        self.pub_steer = self.node.create_publisher(
            Float64, "/model/tractor/steering_angle", 10
        )
        self.pub_drive = self.node.create_publisher(
            Twist, "/model/tractor/rear_axle/cmd_vel", 10
        )

        self.has_world_odom: bool = False

        # Non-blocking background executor thread for ROS 2 events
        self._executor = self.rclpy.executors.SingleThreadedExecutor()
        self._executor.add_node(self.node)
        self._spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self._spin_thread.start()

    def _spin_loop(self) -> None:
        """Continuously process ROS 2 callbacks in background daemon thread."""
        try:
            self._executor.spin()
        except Exception:
            pass

    @staticmethod
    def _parse_point_cloud2(msg: Any) -> np.ndarray:
        if not hasattr(msg, "data") or len(msg.data) == 0:
            return np.empty((0, 3), dtype=np.float32)

        offsets = {f.name: f.offset for f in getattr(msg, "fields", [])}
        if "x" in offsets and "y" in offsets and "z" in offsets:
            x_off = offsets["x"]
            y_off = offsets["y"]
            z_off = offsets["z"]
            step = getattr(msg, "point_step", 16)
            raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, step)
            xs = np.frombuffer(
                np.ascontiguousarray(raw[:, x_off : x_off + 4]), dtype=np.float32
            )
            ys = np.frombuffer(
                np.ascontiguousarray(raw[:, y_off : y_off + 4]), dtype=np.float32
            )
            zs = np.frombuffer(
                np.ascontiguousarray(raw[:, z_off : z_off + 4]), dtype=np.float32
            )
            valid = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(zs)
            if np.any(valid):
                return np.column_stack([xs[valid], ys[valid], zs[valid]]).astype(
                    np.float32
                )
        return np.empty((0, 3), dtype=np.float32)

    def _on_point_cloud(self, pod_name: str, msg: Any) -> None:
        pts = self._parse_point_cloud2(msg)
        if pts.shape[0] == 0:
            return

        # Transform sensor local coordinates to vehicle base_link ego frame
        if pod_name == "front":
            c_p = math.cos(FRONT_LIDAR_PITCH_RAD)
            s_p = math.sin(FRONT_LIDAR_PITCH_RAD)
            xe = pts[:, 0] * c_p + pts[:, 2] * s_p + FRONT_LIDAR_X
            ye = pts[:, 1]
            ze = -pts[:, 0] * s_p + pts[:, 2] * c_p + FRONT_LIDAR_Z
        else:
            c_p = math.cos(REAR_LIDAR_PITCH_RAD)
            s_p = math.sin(REAR_LIDAR_PITCH_RAD)
            xe = -pts[:, 0] * c_p - pts[:, 2] * s_p + REAR_LIDAR_X
            ye = -pts[:, 1]
            ze = -pts[:, 0] * s_p + pts[:, 2] * c_p + REAR_LIDAR_Z

        pts_ego = np.column_stack([xe, ye, ze]).astype(np.float32)
        with self.lock:
            self.latest_point_clouds[pod_name] = pts_ego

    def get_point_clouds(self) -> dict[str, np.ndarray]:
        """Thread-safe snapshot of latest 3D point clouds in vehicle ego coordinates."""
        with self.lock:
            return dict(self.latest_point_clouds)

    def _parse_odom(self, msg: Any) -> None:
        pos = msg.pose.pose.position
        orient = msg.pose.pose.orientation
        # Convert quaternion to yaw angle
        siny_cosp = 2.0 * (orient.w * orient.z + orient.x * orient.y)
        cosy_cosp = 1.0 - 2.0 * (orient.y * orient.y + orient.z * orient.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        lin_vx = float(msg.twist.twist.linear.x)
        lin_vy = float(msg.twist.twist.linear.y)
        ang_yaw_rate = float(msg.twist.twist.angular.z)

        now = time.monotonic()
        with self.lock:
            if self._last_vx_time > 0.0 and now > self._last_vx_time:
                dt = now - self._last_vx_time
                self._ax = (lin_vx - self._last_vx) / max(dt, 1e-4)
            self._last_vx = lin_vx
            self._last_vx_time = now

            self.latest_odom = {
                "x": float(pos.x),
                "y": float(pos.y),
                "yaw": yaw,
                "vx": lin_vx,
                "vy": lin_vy,
                "yaw_rate": ang_yaw_rate,
            }

    def _on_odom(self, msg: Any) -> None:
        self.has_world_odom = True
        self._parse_odom(msg)

    def _on_wheel_odom(self, msg: Any) -> None:
        if not self.has_world_odom:
            self._parse_odom(msg)

    def _on_joint_state(self, msg: Any) -> None:
        """Read true articulated joint positions (steering knuckles)."""
        steer_positions: list[float] = []
        with self.lock:
            for name, pos in zip(msg.name, msg.position, strict=False):
                if "steering" in name:
                    steer_positions.append(float(pos))
            if steer_positions:
                self.latest_steering_angle = float(np.mean(steer_positions))

    def _on_imu(self, msg: Any) -> None:
        """Read true linear acceleration and angular velocity from chassis IMU."""
        with self.lock:
            self.latest_imu = {
                "ax": float(msg.linear_acceleration.x),
                "ay": float(msg.linear_acceleration.y),
                "yaw_rate": float(msg.angular_velocity.z),
            }

    def publish_action(self, target_steer_rad: float, target_speed_mps: float) -> None:
        """Publish coordinated steering angle and rear differential twist commands."""
        # 1. Steering knuckle position command
        steer_msg = self.Float64()
        steer_msg.data = float(target_steer_rad)
        self.pub_steer.publish(steer_msg)

        # 2. Ackermann yaw rate: omega = (v / L) * tan(delta)
        yaw_rate = (target_speed_mps / WHEELBASE) * math.tan(target_steer_rad)

        twist_msg = self.Twist()
        twist_msg.linear.x = float(target_speed_mps)
        twist_msg.angular.z = float(yaw_rate)

        # Publish to both rear axle drive and unified /cmd_vel
        self.pub_drive.publish(twist_msg)
        self.pub_cmd_vel.publish(twist_msg)

    def spin_once(self, timeout_sec: float = 0.0) -> None:
        """Process incoming ROS 2 events (handled asynchronously by background executor)."""
        # Kept for interface compatibility; spinning is handled continuously by self._executor
        pass

    def _on_contacts(self, msg: Any) -> None:
        """Handle live contact collision reports from Gazebo Harmonic Contact system."""
        is_colliding = False
        if hasattr(msg, "contacts") and len(msg.contacts) > 0:
            for c in msg.contacts:
                c1 = getattr(c.collision1, "name", "").lower()
                c2 = getattr(c.collision2, "name", "").lower()
                if "ground_plane" not in c1 and "ground_plane" not in c2:
                    is_colliding = True
                    break
        with self.lock:
            self.latest_collision = is_colliding

    def reset_vehicle(self, sx: float, sy: float, syaw: float) -> None:
        """Reset commands, state caches, and reposition vehicle in Gazebo."""
        with self.lock:
            self.latest_collision = False
            self.latest_odom = None
            self.latest_steering_angle = 0.0
            self.latest_imu = None
            self._last_vx = 0.0
            self._last_vx_time = 0.0
            self._ax = 0.0

        try:
            stop_twist = self.Twist()
            self.pub_cmd_vel.publish(stop_twist)
            self.pub_drive.publish(stop_twist)
            zero_steer = self.Float64()
            zero_steer.data = 0.0
            self.pub_steer.publish(zero_steer)
        except Exception:
            pass

        # Reset pose in Gazebo via gz service call if available
        import subprocess
        qz = math.sin(syaw / 2.0)
        qw = math.cos(syaw / 2.0)
        req_str = f'name: "tractor", position: {{x: {sx:.3f}, y: {sy:.3f}, z: 0.50}}, orientation: {{x: 0.0, y: 0.0, z: {qz:.6f}, w: {qw:.6f}}}'
        gz_cmd = Path(__file__).resolve().parent.parent.parent / "scripts" / "gz"
        if gz_cmd.exists():
            try:
                subprocess.run(
                    [
                        str(gz_cmd),
                        "service",
                        "-s",
                        "/world/procedural_car_track/set_pose",
                        "--reqtype",
                        "gz.msgs.Pose",
                        "--reptype",
                        "gz.msgs.Boolean",
                        "--timeout",
                        "500",
                        "--req",
                        req_str,
                    ],
                    capture_output=True,
                    timeout=1.5,
                    check=False,
                )
            except Exception:
                pass

    def get_collision(self) -> bool:
        """Thread-safe query of Gazebo Harmonic collision state."""
        with self.lock:
            return self.latest_collision

    def get_state(self) -> tuple[dict[str, float] | None, dict[str, np.ndarray]]:
        """Retrieve latest thread-safe snapshot of odometry and 3D point clouds."""
        with self.lock:
            odom = dict(self.latest_odom) if self.latest_odom is not None else None
            if odom is not None:
                odom["steering_angle"] = self.latest_steering_angle
                if self.latest_imu is not None:
                    odom["ax"] = self.latest_imu["ax"]
                    odom["ay"] = self.latest_imu["ay"]
                else:
                    odom["ax"] = self._ax
                    odom["ay"] = odom["vx"] * odom["yaw_rate"]
            clouds = dict(self.latest_point_clouds)
        return odom, clouds

    def close(self) -> None:
        """Clean shutdown of node and executor."""
        try:
            # Send stop command before destroying
            stop_twist = self.Twist()
            self.pub_cmd_vel.publish(stop_twist)
            self.pub_drive.publish(stop_twist)
        except Exception:
            pass

        try:
            if hasattr(self, "_executor") and self._executor is not None:
                self._executor.shutdown()
        except Exception:
            pass

        try:
            if hasattr(self, "node") and self.node is not None:
                self.node.destroy_node()
        except Exception:
            pass

        try:
            if hasattr(self, "rclpy") and self.rclpy.ok():
                self.rclpy.shutdown()
        except Exception:
            pass
