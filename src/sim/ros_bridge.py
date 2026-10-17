"""ROS 2 bridge manager for connecting Gymnasium CarGazeboEnv to live Gazebo simulation.

Communicates with ros_gz_bridge via standard ROS 2 topics:
- Subscribes to 4-pod LaserScans (/tractor/lidar/*)
- Subscribes to Wheel Odometry (/model/tractor/wheel_odometry)
- Publishes Ackermann steering angle and drive velocity (/cmd_vel, /model/tractor/steering_angle, /model/tractor/rear_axle/cmd_vel)
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any

import numpy as np

from src.sim.spatial_map import LaserScanData
from src.translate.config import WHEELBASE


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
            from sensor_msgs.msg import LaserScan
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
        self.LaserScan = LaserScan
        self.Float64 = Float64

        if not self.rclpy.ok():
            self.rclpy.init()

        self.node = self.rclpy.create_node(node_name)
        self.lock = threading.Lock()

        # Cached state from Gazebo
        self.latest_scans: dict[str, LaserScanData] = {}
        self.latest_odom: dict[str, float] | None = None
        self.last_scan_time: float = 0.0

        # Subscriptions: 4 LiDAR Pods
        self.sub_front = self.node.create_subscription(
            LaserScan, "/tractor/lidar/front", lambda m: self._on_scan("front", m), 10
        )
        self.sub_rear = self.node.create_subscription(
            LaserScan, "/tractor/lidar/rear", lambda m: self._on_scan("rear", m), 10
        )
        self.sub_right = self.node.create_subscription(
            LaserScan, "/tractor/lidar/right", lambda m: self._on_scan("right", m), 10
        )
        self.sub_left = self.node.create_subscription(
            LaserScan, "/tractor/lidar/left", lambda m: self._on_scan("left", m), 10
        )

        # Subscription: World Odometry & Wheel Odometry
        self.sub_odom = self.node.create_subscription(
            Odometry, "/model/tractor/odometry", self._on_odom, 10
        )
        self.sub_wheel_odom = self.node.create_subscription(
            Odometry, "/model/tractor/wheel_odometry", self._on_wheel_odom, 10
        )

        # Publishers: Commands
        self.pub_cmd_vel = self.node.create_publisher(Twist, "/cmd_vel", 10)
        self.pub_steer = self.node.create_publisher(Float64, "/model/tractor/steering_angle", 10)
        self.pub_drive = self.node.create_publisher(Twist, "/model/tractor/rear_axle/cmd_vel", 10)

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

    def _on_scan(self, pod_name: str, msg: Any) -> None:
        scan_data = LaserScanData(
            ranges=np.array(msg.ranges, dtype=np.float32),
            angle_min=float(msg.angle_min),
            angle_max=float(msg.angle_max),
            angle_increment=float(msg.angle_increment),
            range_min=float(msg.range_min),
            range_max=float(msg.range_max),
        )
        with self.lock:
            self.latest_scans[pod_name] = scan_data
            self.last_scan_time = time.monotonic()

    def _parse_odom(self, msg: Any) -> None:
        pos = msg.pose.pose.position
        orient = msg.pose.pose.orientation
        # Convert quaternion to yaw angle
        siny_cosp = 2.0 * (orient.w * orient.z + orient.x * orient.y)
        cosy_cosp = 1.0 - 2.0 * (orient.y * orient.y + orient.z * orient.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        lin_vx = float(msg.twist.twist.linear.x)
        ang_yaw_rate = float(msg.twist.twist.angular.z)

        with self.lock:
            self.latest_odom = {
                "x": float(pos.x),
                "y": float(pos.y),
                "yaw": yaw,
                "vx": lin_vx,
                "yaw_rate": ang_yaw_rate,
            }

    def _on_odom(self, msg: Any) -> None:
        self.has_world_odom = True
        self._parse_odom(msg)

    def _on_wheel_odom(self, msg: Any) -> None:
        if not self.has_world_odom:
            self._parse_odom(msg)

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

    def get_state(self) -> tuple[dict[str, float] | None, dict[str, LaserScanData]]:
        """Retrieve latest thread-safe snapshot of odometry and 4-pod scans."""
        with self.lock:
            odom = dict(self.latest_odom) if self.latest_odom is not None else None
            scans = dict(self.latest_scans)
        return odom, scans

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
