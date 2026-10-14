#!/usr/bin/env python

"""
NeupanCore is the main ROS2 node for the NeuPAN navigation algorithm.

This node subscribes to laser scan and localization data, executes the NeuPAN
planning algorithm, and publishes velocity commands to control the robot.

Developer: Han Ruihua <hanrh@connect.hku.hk>  Li Chengyang <kevinladlee@gmail.com>
Date: 2025.04.08
"""
import os
import threading
import traceback
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import numpy.typing as npt
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from ament_index_python.packages import get_package_share_directory
import tf2_ros

from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import LaserScan

try:
    from neupan import neupan
    from neupan.util import get_transform
except ImportError as e:
    raise ImportError(
        f"Failed to import 'neupan' package: {e}. "
        "Please install NeuPAN first."
    ) from e

# Import local modules
from neupan_ros2.visualization_manager import VisualizationManager
from neupan_ros2.utils import yaw_to_quat, quat_to_yaw


class NeupanCore(Node):
    """ROS2 node for NeuPAN navigation algorithm.

    This node integrates the NeuPAN planner with ROS2, handling sensor data,
    executing planning, and publishing control commands.
    """

    def __init__(self) -> None:
        super().__init__("neupan_node")

        # Thread lock protecting shared state: robot_state, obstacle_points, stop, arrive
        # These are accessed by both control thread (run) and callback thread (scan/path/goal)
        self._state_lock = threading.Lock()

        # Callback groups for multi-threaded execution
        # Control group: MutuallyExclusive for timer (run) - ensures run() executes alone
        self.control_group = MutuallyExclusiveCallbackGroup()
        # Callback group: Reentrant for all subscriptions - allows concurrent execution
        self.callback_group = ReentrantCallbackGroup()

        # Package directory for accessing config files and models
        self.pkg_dir = get_package_share_directory("neupan_ros2")

        # Robot identification and configuration directory
        self.declare_parameter("robot_type", "")
        self.declare_parameter("robot_description", "")
        self.declare_parameter("robot_config_dir", "")  # Set by launch file
        self.declare_parameter("planner_config_file", "planner.yaml")
        self.declare_parameter("dune_checkpoint_file", "models/dune_model_5000.pth")

        # Legacy parameter name (for backward compatibility)
        self.declare_parameter("neupan_config_file", "NOT SET")

        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("lidar_frame", "laser_link")
        self.declare_parameter("marker_size", 0.05)
        self.declare_parameter("marker_z", 1.0)
        self.declare_parameter("scan_angle_max", 3.14)
        self.declare_parameter("scan_angle_min", -3.14)
        self.declare_parameter("scan_downsample", 1)
        self.declare_parameter("scan_range_min", 0.1)
        self.declare_parameter("scan_range_max", 5.0)
        self.declare_parameter("refresh_initial_path", False)
        self.declare_parameter("flip_angle", False)
        self.declare_parameter("include_initial_path_direction", False)
        self.declare_parameter("control_frequency", 50.0)  # Control loop frequency in Hz

        # Visualization control parameters
        self.declare_parameter("enable_visualization", True)
        self.declare_parameter("enable_dune_markers", True)
        self.declare_parameter("enable_nrmp_markers", True)
        self.declare_parameter("enable_robot_marker", True)

        # Topic names (configurable for flexibility)
        self.declare_parameter("cmd_vel_topic", "/neupan_cmd_vel")
        self.declare_parameter("plan_output_topic", "/neupan_plan")
        self.declare_parameter("ref_state_topic", "/neupan_ref_state")
        self.declare_parameter("initial_path_topic", "/neupan_initial_path")
        self.declare_parameter("dune_markers_topic", "/dune_point_markers")
        self.declare_parameter("robot_marker_topic", "/robot_marker")
        self.declare_parameter("nrmp_markers_topic", "/nrmp_point_markers")
        self.declare_parameter("scan_topic", "/scan")
        # Multiple scan sources — each entry is "topic:frame_id"
        # NeuPAN merges all into a single obstacle point cloud
        self.declare_parameter("extra_scan_topics", [""])
        self.declare_parameter("plan_input_topic", "/plan")
        self.declare_parameter("goal_topic", "/goal_pose")

        # === Configuration Loading ===
        # Get robot configuration directory (set by launch file)
        robot_config_dir = (
            self.get_parameter("robot_config_dir")
            .get_parameter_value().string_value
        )

        # Validate robot config directory exists
        if not robot_config_dir or not os.path.isdir(robot_config_dir):
            raise ValueError(
                f"Invalid robot_config_dir: '{robot_config_dir}'. "
                "Must be set by launch file to a valid robot config directory."
            )

        # Get robot type for logging
        robot_type = (
            self.get_parameter("robot_type")
            .get_parameter_value().string_value
        )
        robot_description = (
            self.get_parameter("robot_description")
            .get_parameter_value().string_value
        )

        self.get_logger().info(f"Loading robot configuration: {robot_type}")
        self.get_logger().info(f"Description: {robot_description}")
        self.get_logger().info(f"Config directory: {robot_config_dir}")

        # Load planner configuration (relative to robot config dir)
        planner_config_file = (
            self.get_parameter("planner_config_file")
            .get_parameter_value().string_value
        )
        self.planner_config_file = os.path.join(robot_config_dir, planner_config_file)

        # Load DUNE checkpoint (relative to robot config dir)
        dune_checkpoint_file = (
            self.get_parameter("dune_checkpoint_file")
            .get_parameter_value().string_value
        )
        self.dune_checkpoint = os.path.join(robot_config_dir, dune_checkpoint_file)

        # Validate configuration files exist
        if not os.path.isfile(self.planner_config_file):
            raise FileNotFoundError(
                f"Planner config not found: {self.planner_config_file}"
            )
        if not os.path.isfile(self.dune_checkpoint):
            raise FileNotFoundError(
                f"DUNE checkpoint not found: {self.dune_checkpoint}"
            )

        self.get_logger().info(f"Planner config: {self.planner_config_file}")
        self.get_logger().info(f"DUNE checkpoint: {self.dune_checkpoint}")

        # Load other parameters
        self.map_frame = self.get_parameter("map_frame").get_parameter_value().string_value
        self.base_frame = self.get_parameter("base_frame").get_parameter_value().string_value
        self.lidar_frame = self.get_parameter("lidar_frame").get_parameter_value().string_value
        self.marker_size = self.get_parameter("marker_size").get_parameter_value().double_value
        self.marker_z = self.get_parameter("marker_z").get_parameter_value().double_value

        self.scan_range = np.array([
            self.get_parameter("scan_range_min").get_parameter_value().double_value,
            self.get_parameter("scan_range_max").get_parameter_value().double_value
        ])

        self.scan_angle_range = np.array([
            self.get_parameter("scan_angle_min").get_parameter_value().double_value,
            self.get_parameter("scan_angle_max").get_parameter_value().double_value
        ])

        self.scan_downsample = (
            self.get_parameter("scan_downsample")
            .get_parameter_value().integer_value
        )

        self.refresh_initial_path = (
            self.get_parameter("refresh_initial_path")
            .get_parameter_value().bool_value
        )
        self.flip_angle = (
            self.get_parameter("flip_angle")
            .get_parameter_value().bool_value
        )
        self.include_initial_path_direction = (
            self.get_parameter("include_initial_path_direction")
            .get_parameter_value().bool_value
        )

        self.enable_visualization = (
            self.get_parameter("enable_visualization")
            .get_parameter_value().bool_value
        )
        self.enable_dune_markers = (
            self.get_parameter("enable_dune_markers")
            .get_parameter_value().bool_value
        )
        self.enable_nrmp_markers = (
            self.get_parameter("enable_nrmp_markers")
            .get_parameter_value().bool_value
        )
        self.enable_robot_marker = (
            self.get_parameter("enable_robot_marker")
            .get_parameter_value().bool_value
        )

        if self.refresh_initial_path:
            self.get_logger().info("Refresh initial path is enabled")

        if not self.planner_config_file:
            raise ValueError(
                "No planner config file provided! "
                "Please set the parameter 'config_file'"
            )

        # Load YAML and merge dune_checkpoint into pan section
        # (passing pan= to init_from_yaml replaces the entire section,
        #  losing iter_num, dune_max_num, nrmp_max_num)
        import yaml as _yaml
        with open(self.planner_config_file) as _f:
            _config = _yaml.safe_load(_f)
        _config.setdefault('pan', {})['dune_checkpoint'] = self.dune_checkpoint
        _config["robot_kwargs"] = _config.pop("robot", dict())
        _config["ipath_kwargs"] = _config.pop("ipath", dict())
        _config["pan_kwargs"] = _config.pop("pan", dict())
        _config["adjust_kwargs"] = _config.pop("adjust", dict())
        _config["train_kwargs"] = _config.pop("train", dict())
        self.neupan_planner = neupan(**_config)

        # Log PAN solver parameters for verification
        self.get_logger().info(
            f"PAN config - iter_num: {self.neupan_planner.pan.iter_num}, "
            f"dune_max_num: {self.neupan_planner.pan.dune_max_num}, "
            f"nrmp_max_num: {self.neupan_planner.pan.nrmp_max_num}"
        )
        # Log robot dimensions for verification
        self.get_logger().info(
            f"Robot dimensions - Length: {self.neupan_planner.robot.length:.3f}m, "
            f"Width: {self.neupan_planner.robot.width:.3f}m"
        )
        if hasattr(self.neupan_planner.robot, 'wheelbase') and self.neupan_planner.robot.wheelbase is not None:
            self.get_logger().info(
                f"Robot wheelbase: {self.neupan_planner.robot.wheelbase:.3f}m"
            )
        self.get_logger().info(f"Robot kinematics: {self.neupan_planner.robot.kinematics}")
        self.get_logger().info("NeuPAN planner initialized successfully")

        # Shared state protected by _state_lock (accessed by multiple threads)
        # Write access: scan_callback (obstacle_points), _get_robot_transform (robot_state)
        # Read access: _execute_planning (all), generate_twist_msg (stop, arrive)
        # Planning copies data before execution to minimize lock holding time
        self.obstacle_points: Optional[npt.NDArray] = None  # (2, n) merged obstacle points in map frame
        self._scan_points: Dict[str, Optional[npt.NDArray]] = {}  # per-sensor obstacle points
        self.robot_state: Optional[npt.NDArray] = None  # (3, 1) [x, y, theta] in map frame
        self.stop: bool = False  # Emergency stop flag from collision detection
        self.arrive: bool = False  # Goal reached flag
        self.goal: Optional[npt.NDArray] = None  # (3, 1) target goal [x, y, theta]
        self.goal_received: bool = False  # Gate: ignore paths until first goal

        self.vel_pub = self.create_publisher(
            Twist,
            self.get_parameter("cmd_vel_topic").get_parameter_value().string_value,
            10
        )
        self.plan_pub = self.create_publisher(
            Path,
            self.get_parameter("plan_output_topic").get_parameter_value().string_value,
            10
        )
        self.ref_state_pub = self.create_publisher(
            Path,
            self.get_parameter("ref_state_topic").get_parameter_value().string_value,
            10
        )
        self.ref_path_pub = self.create_publisher(
            Path,
            self.get_parameter("initial_path_topic").get_parameter_value().string_value,
            10
        )

        # Initialize visualization manager (handles all visualization independently)
        viz_config = {
            'enable_visualization': self.enable_visualization,
            'enable_dune_markers': self.enable_dune_markers,
            'enable_nrmp_markers': self.enable_nrmp_markers,
            'enable_robot_marker': self.enable_robot_marker,
            'map_frame': self.map_frame,
            'marker_size': self.marker_size,
            'marker_z': self.marker_z,
            'dune_markers_topic': (
                self.get_parameter("dune_markers_topic")
                .get_parameter_value().string_value
            ),
            'nrmp_markers_topic': (
                self.get_parameter("nrmp_markers_topic")
                .get_parameter_value().string_value
            ),
            'robot_marker_topic': (
                self.get_parameter("robot_marker_topic")
                .get_parameter_value().string_value
            ),
            'state_lock': self._state_lock
        }
        self.viz_manager = VisualizationManager(self, viz_config)

        # TF listener for coordinate transformations (default 10s buffer)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        scan_qos_profile = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)

        # Primary scan topic (uses lidar_frame for TF)
        primary_scan = self.get_parameter("scan_topic").get_parameter_value().string_value
        self._scan_points[primary_scan] = None
        self.create_subscription(
            LaserScan, primary_scan,
            lambda msg, t=primary_scan, f=self.lidar_frame: self._multi_scan_callback(msg, t, f),
            scan_qos_profile, callback_group=self.callback_group
        )
        self.get_logger().info(f"Scan source: {primary_scan} (frame: {self.lidar_frame})")

        # Extra scan topics — each entry is "topic:frame_id"
        extra_scans = (
            self.get_parameter("extra_scan_topics")
            .get_parameter_value().string_array_value
        )
        for entry in extra_scans:
            if not entry or ':' not in entry:
                continue
            topic, frame_id = entry.split(':', 1)
            self._scan_points[topic] = None
            self.create_subscription(
                LaserScan, topic,
                lambda msg, t=topic, f=frame_id: self._multi_scan_callback(msg, t, f),
                scan_qos_profile, callback_group=self.callback_group
            )
            self.get_logger().info(f"Scan source: {topic} (frame: {frame_id})")

        self.create_subscription(
            Path,
            self.get_parameter("plan_input_topic").get_parameter_value().string_value,
            self.path_callback,
            10,
            callback_group=self.callback_group
        )
        self.create_subscription(
            PoseStamped,
            self.get_parameter("goal_topic").get_parameter_value().string_value,
            self.goal_callback,
            10,
            callback_group=self.callback_group
        )

        # Control loop timer: frequency configurable via parameter
        self.control_frequency = (
            self.get_parameter("control_frequency")
            .get_parameter_value().double_value
        )
        if self.control_frequency <= 0:
            raise ValueError(
                f"Invalid control_frequency: {self.control_frequency}. "
                "Must be > 0 Hz"
            )

        time_period = 1.0 / self.control_frequency
        self.get_logger().info(
            f"Control loop frequency: {self.control_frequency} Hz "
            f"({time_period*1000:.1f} ms period)"
        )
        self.create_timer(time_period, self.run, callback_group=self.control_group)

    def _get_robot_transform(self) -> bool:
        """Get robot transform from TF and update robot_state.

        Returns:
            bool: True if transform successfully obtained, False otherwise

        """
        try:
            # TF query is thread-safe, no lock needed
            trans = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time()
            )

            yaw = quat_to_yaw(trans.transform.rotation)
            x = trans.transform.translation.x
            y = trans.transform.translation.y
            new_state = np.array([x, y, yaw]).reshape(3, 1)

            # Lock only for writing shared state
            with self._state_lock:
                self.robot_state = new_state

            self.get_logger().info(
                f"Robot state initialized - x: {new_state[0,0]:.2f}m, "
                f"y: {new_state[1,0]:.2f}m, yaw: {new_state[2,0]:.2f}rad",
                once=True
            )
            return True

        except tf2_ros.LookupException:
            self.get_logger().debug(
                f"Waiting for transform from {self.base_frame} to {self.map_frame}",
                throttle_duration_sec=1.0,
            )
            return False
        except tf2_ros.ConnectivityException:
            self.get_logger().warn(
                "ConnectivityException: Transform not available, waiting for connection",
                throttle_duration_sec=1.0
            )
            return False
        except tf2_ros.ExtrapolationException as e:
            self.get_logger().warn(
                f"TF extrapolation error: {e}. Check TF timestamps and buffer size.",
                throttle_duration_sec=1.0
            )
            return False

    def _validate_planning_prerequisites(self) -> bool:
        """Validate all prerequisites for planning are met.

        Returns:
            bool: True if all prerequisites are met, False otherwise

        """
        with self._state_lock:
            if self.robot_state is None:
                self.get_logger().debug("Waiting for robot state", throttle_duration_sec=1.0)
                return False

            # Initialize path from waypoints on first run if no path received
            if (len(self.neupan_planner.waypoints) >= 1
                    and self.neupan_planner.initial_path is None):
                self.neupan_planner.set_initial_path_from_state(
                    self.robot_state
                )
                self.get_logger().info(
                    f'Initialized path with '
                    f'{len(self.neupan_planner.waypoints)} waypoints'
                )

            if self.neupan_planner.initial_path is None:
                self.get_logger().debug("Waiting for initial path", throttle_duration_sec=1.0)
                return False

        return True

    def _execute_planning(self) -> Tuple[Optional[npt.NDArray], Dict[str, Any]]:
        """Execute planning and update state.

        Returns:
            tuple: (action, info) from neupan planner, or (None, None) on failure

        """
        # Publish reference path (generate message needs to read planner state)
        with self._state_lock:
            initial_path = self.neupan_planner.initial_path

        # Publishing is thread-safe, do outside lock
        self.ref_path_pub.publish(self.generate_path_msg(initial_path))

        # Step 1: Fast data copy inside lock to minimize lock holding time
        with self._state_lock:
            # Copy state data for planning
            # (allows other threads to access shared state)
            robot_state_copy = (
                self.robot_state.copy()
                if self.robot_state is not None else None
            )
            obstacle_points_copy = (
                self.obstacle_points.copy()
                if self.obstacle_points is not None else None
            )

            # Check for obstacles
            has_obstacles = obstacle_points_copy is not None

        # Step 2: Execute planning OUTSIDE lock (10-50ms)
        # (allows other threads to access shared state)
        action, info = self.neupan_planner(
            robot_state_copy, obstacle_points_copy
        )

        # Step 3: Write back results inside lock (< 0.1 μs)
        with self._state_lock:
            self.stop = info["stop"]
            self.arrive = info["arrive"]

        # Logging outside lock
        if not has_obstacles:
            self.get_logger().info(
                "No obstacle points detected, performing path tracking only",
                throttle_duration_sec=1.0,
            )

        # Log arrival
        if info["arrive"]:
            self.get_logger().info("Arrived at target", once=True)

        # Log stop condition
        if info["stop"]:
            # Read min_distance and threshold outside lock
            # (assume read-only access is safe)
            self.get_logger().warn(
                f"Collision risk detected - "
                f"min distance: {self.neupan_planner.min_distance:.2f}m, "
                f"threshold: {self.neupan_planner.collision_threshold:.2f}m",
                throttle_duration_sec=1.0,
            )

        return action, info

    def _publish_planning_results(
            self, action: Optional[npt.NDArray], info: Dict[str, Any]
    ) -> None:
        """Publish planning results and visualization markers.

        Args:
            action: Control action from planner
            info: Planning info dictionary

        """
        # Publish path messages (info is local, thread-safe)
        self.plan_pub.publish(self.generate_path_msg(info["opt_state_list"]))
        self.ref_state_pub.publish(self.generate_path_msg(info["ref_state_list"]))

        # Generate twist message using info dict (avoid reading shared state)
        vel_msg = self.generate_twist_msg(action, info["stop"], info["arrive"])
        self.vel_pub.publish(vel_msg)

        # Debug velocity output (throttled to 1Hz)
        if action is not None:
            self.get_logger().info(
                f"cmd_vel: v={vel_msg.linear.x:.4f} w={vel_msg.angular.z:.4f} "
                f"raw=[{float(action[0,0]):.4f},{float(action[1,0]):.4f}] "
                f"stop={info['stop']} arrive={info['arrive']}",
                throttle_duration_sec=1.0,
            )

        # Visualization (delegated to visualization manager)
        self.viz_manager.publish_visualization(
            self.neupan_planner, self.robot_state
        )

    def run(self) -> None:
        """Execute main control loop at fixed frequency.

        Note: Fine-grained locking is handled within each helper method.
        """
        # Step 1: Get robot transform (locks internally for robot_state write)
        if not self._get_robot_transform():
            return

        # Step 2: Validate planning prerequisites (locks internally for state read)
        if not self._validate_planning_prerequisites():
            return

        # Step 3: Execute planning (locks internally for planning execution)
        action, info = self._execute_planning()

        # Step 4: Publish results (locks internally for marker generation)
        self._publish_planning_results(action, info)

    def _multi_scan_callback(
            self, scan_msg: LaserScan, topic: str, frame_id: str
    ) -> None:
        """Process laser scan from any sensor and merge into obstacle points.

        Each sensor's points are stored separately, then merged into
        self.obstacle_points so the solver sees all obstacles at once.

        Args:
            scan_msg: LaserScan message
            topic: Source topic name (for per-sensor storage)
            frame_id: TF frame for this sensor
        """
        with self._state_lock:
            if self.robot_state is None:
                return

        ranges = np.array(scan_msg.ranges)
        angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, len(ranges))

        if self.flip_angle:
            angles = np.flip(angles)

        # Filter: downsample + range + angle
        indices = np.arange(len(ranges))
        valid_mask = (
            ((indices % self.scan_downsample) == 0)
            & (ranges >= self.scan_range[0])
            & (ranges <= self.scan_range[1])
            & (angles > self.scan_angle_range[0])
            & (angles < self.scan_angle_range[1])
        )
        valid_ranges = ranges[valid_mask]
        valid_angles = angles[valid_mask]

        if len(valid_ranges) == 0:
            with self._state_lock:
                self._scan_points[topic] = None
                self._merge_obstacle_points()
            return

        # Polar to cartesian in sensor frame
        x_coords = valid_ranges * np.cos(valid_angles)
        y_coords = valid_ranges * np.sin(valid_angles)
        point_array = np.vstack([x_coords, y_coords])

        try:
            trans = self.tf_buffer.lookup_transform(
                self.map_frame, frame_id, rclpy.time.Time()
            )
            yaw = quat_to_yaw(trans.transform.rotation)
            x = trans.transform.translation.x
            y = trans.transform.translation.y

            trans_matrix, rot_matrix = get_transform(
                np.c_[x, y, yaw].reshape(3, 1)
            )
            transformed = rot_matrix @ point_array + trans_matrix

            with self._state_lock:
                self._scan_points[topic] = transformed
                self._merge_obstacle_points()

            self.get_logger().info(
                f"Scan [{topic}] initialized: {transformed.shape[1]} pts "
                f"(frame: {frame_id})", once=True
            )

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            self.get_logger().debug(
                f"Waiting for TF {frame_id} -> {self.map_frame}",
                throttle_duration_sec=1.0
            )

    def _merge_obstacle_points(self) -> None:
        """Merge per-sensor obstacle points into self.obstacle_points.

        Each sensor is capped to prevent depth cameras from dominating
        the DUNE downsampling. Budget: primary lidar gets 70%, extras
        share the remaining 30% equally.

        Must be called with _state_lock held.
        """
        # dune_max_num is the total budget DUNE will downsample to
        total_budget = self.neupan_planner.pan.dune_max_num  # e.g. 200

        primary_topic = (
            self.get_parameter("scan_topic")
            .get_parameter_value().string_value
        )
        primary = self._scan_points.get(primary_topic)
        extras = {k: v for k, v in self._scan_points.items()
                  if k != primary_topic and v is not None}

        arrays = []

        if primary is not None:
            # Primary lidar: keep all (DUNE expects lidar-like ring)
            arrays.append(primary)

        if extras:
            # Each extra sensor gets a capped share
            # Cap so extras are at most 30% of total budget
            extra_budget = int(total_budget * 0.3)
            per_extra = max(10, extra_budget // len(extras))
            for pts in extras.values():
                n = pts.shape[1]
                if n > per_extra:
                    # Uniform downsample
                    idx = np.linspace(0, n - 1, per_extra, dtype=int)
                    pts = pts[:, idx]
                arrays.append(pts)

        if arrays:
            self.obstacle_points = np.hstack(arrays)
        else:
            self.obstacle_points = None

    def path_callback(self, path: Path) -> None:
        """Update initial path from received path message.

        Args:
            path: Path message containing waypoints

        """
        n_poses = len(path.poses)
        if n_poses == 0:
            return

        self.get_logger().info(f"Received new path with {n_poses} waypoints")

        # Optimized: single-pass extraction with transpose
        if self.include_initial_path_direction:
            # Extract x, y, and orientation in one pass
            data = [
                (p.pose.position.x, p.pose.position.y,
                 quat_to_yaw(p.pose.orientation))
                for p in path.poses
            ]
            xs, ys, thetas = np.array(data).T
        else:
            self.get_logger().debug(
                "Using path gradient for direction "
                "(include_initial_path_direction=False)", once=True
            )

            # Extract x, y in one pass
            coords = [
                (p.pose.position.x, p.pose.position.y) for p in path.poses
            ]
            xs, ys = np.array(coords).T

            # Vectorized gradient computation using np.diff
            dx = np.diff(xs, append=xs[-1])
            dy = np.diff(ys, append=ys[-1])
            thetas = np.arctan2(dy, dx)

            # For the last point, use direction from second-to-last point
            if n_poses > 1:
                thetas[-1] = thetas[-2]

        # Vectorized array construction for better performance
        ones = np.ones(n_poses)
        # Shape: (4, n_poses)
        initial_point_array = np.vstack([xs, ys, thetas, ones])

        # Convert to list of column vectors for planner API compatibility
        initial_point_list = [
            initial_point_array[:, i:i + 1] for i in range(n_poses)
        ]

        with self._state_lock:
            # Don't accept paths until we have a goal
            # (prevents tracking stale paths from previous Nav2 sessions)
            if not self.goal_received:
                self.get_logger().info(
                    "Ignoring path (no goal received yet)", once=True
                )
                return

            # Accept new path when:
            # 1. No path yet (first path)
            # 2. refresh_initial_path is enabled (continuous replanning)
            # 3. Planner has arrived (new goal via Nav2 action server)
            no_path = self.neupan_planner.initial_path is None
            arrived = self.neupan_planner.ipath.arrive_flag
            should_accept = (
                no_path
                or self.refresh_initial_path
                or arrived
            )
            if should_accept:
                self.get_logger().info(
                    f"Accepting path ({n_poses} pts) — "
                    f"no_path={no_path} refresh={self.refresh_initial_path} arrived={arrived}"
                )
                self.neupan_planner.set_initial_path(initial_point_list)
                # Full reset when transitioning out of arrive state
                if self.neupan_planner.ipath.arrive_flag:
                    self.neupan_planner.reset()
                    self.arrive = False
                    self.get_logger().info(
                        f"Planner reset on new path ({n_poses} pts, was arrived)"
                    )

    def goal_callback(self, goal: PoseStamped) -> None:
        """Update goal and regenerate initial path.

        Args:
            goal: Goal pose message

        """
        # Extract goal from message (no lock needed)
        x = goal.pose.position.x
        y = goal.pose.position.y
        theta = quat_to_yaw(goal.pose.orientation)

        new_goal = np.array([[x], [y], [theta]])
        self.goal_received = True

        self.get_logger().info(
            f"New goal set - x: {x:.2f}m, y: {y:.2f}m, "
            f"theta: {theta:.2f}rad"
        )

        # Check if robot state is ready
        if self.robot_state is None:
            self.get_logger().warn(
                "Goal received but robot state not yet available. "
                "Path planning will start once robot state is received."
            )
            self.goal = new_goal
            return

        # Lock only when accessing shared state and modifying planner
        with self._state_lock:
            self.goal = new_goal

            self.get_logger().debug(
                f"Current state: {self.robot_state.tolist()}"
            )
            self.get_logger().debug(f"Target goal: {self.goal.tolist()}")

            self.neupan_planner.update_initial_path_from_goal(
                self.robot_state, self.goal
            )
            self.neupan_planner.reset()

    def generate_path_msg(self, path_list: List[npt.NDArray]) -> Path:
        """Generate ROS Path message from list of poses.

        Args:
            path_list: List of pose arrays (3, 1) or (4, 1)
                       containing [x, y, theta, ...]

        Returns:
            Path message with poses

        """
        path = Path()
        path.header.frame_id = self.map_frame
        path.header.stamp = self.get_clock().now().to_msg()

        if len(path_list) == 0:
            return path

        # Vectorized approach: normalize all points and stack into matrix
        normalized_points = []
        for point in path_list:
            point_arr = np.array(point)
            if point_arr.ndim == 1:
                point_arr = point_arr.reshape(-1, 1)
            # Extract only first 3 elements (x, y, theta) to ensure consistent dimensions
            point_arr = point_arr[:3, :]
            normalized_points.append(point_arr)

        # Stack all points horizontally -> shape: (3, n_poses)
        points_matrix = np.hstack(normalized_points)

        # Vectorized extraction (single op instead of 3 list comps)
        xs = points_matrix[0, :].tolist()
        ys = points_matrix[1, :].tolist()
        yaws = points_matrix[2, :].tolist()

        # Create path message
        for x, y, yaw in zip(xs, ys, yaws):
            ps = PoseStamped()
            ps.header.frame_id = self.map_frame
            ps.pose.position.x = x
            ps.pose.position.y = y
            ps.pose.orientation = yaw_to_quat(yaw)
            path.poses.append(ps)

        return path

    def generate_twist_msg(
            self, vel: Optional[npt.NDArray], stop: bool, arrive: bool
    ) -> Twist:
        """Generate ROS Twist message from velocity command.

        Args:
            vel: Velocity array (2, 1)
                 containing [linear_speed, angular_speed], or None
            stop: Whether the robot should stop (collision risk)
            arrive: Whether the robot has arrived at goal

        Returns:
            Twist message (zero velocity if stopped/arrived or vel is None)

        """
        if vel is None:
            return Twist()

        speed = float(vel[0, 0])
        steer = float(vel[1, 0])

        if stop or arrive:
            return Twist()
        else:
            action = Twist()
            action.linear.x = speed
            action.angular.z = steer
            return action


def main(args=None):
    """Main entry point for NeuPAN node.

    Args:
        args: Command-line arguments (optional)

    """
    rclpy.init(args=args)

    neupan_node = None
    executor = None
    try:
        neupan_node = NeupanCore()

        # Use MultiThreadedExecutor with 2 threads for concurrent execution
        # Thread 1: run() timer (control loop at configurable frequency)
        # Thread 2: scan/path/goal callbacks (sensor and planning updates)
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(neupan_node)

        viz_status = (
            'enabled' if neupan_node.enable_visualization else 'disabled'
        )
        neupan_node.get_logger().info(
            f"NeuPAN node started - "
            f"Control: {neupan_node.control_frequency}Hz, Threads: 2, "
            f"Visualization: {viz_status}"
        )
        executor.spin()

    except KeyboardInterrupt:
        if neupan_node:
            neupan_node.get_logger().info(
                "NeuPAN node shutting down due to "
                "KeyboardInterrupt (Ctrl+C)."
            )
        pass
    except Exception as e:
        if neupan_node:
            neupan_node.get_logger().error(
                f'Unhandled exception: {e}\n{traceback.format_exc()}'
            )
        raise
    finally:
        if executor:
            executor.shutdown()
        if neupan_node:
            neupan_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
