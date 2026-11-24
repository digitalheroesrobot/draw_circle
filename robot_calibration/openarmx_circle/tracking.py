
#!/usr/bin/env python3

import copy
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.executors import MultiThreadedExecutor

from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, Point
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.srv import GetPositionIK
from visualization_msgs.msg import Marker
from moveit_msgs.msg import RobotState, MoveItErrorCodes
import tf2_ros


class CartesianTracking(Node):

    def __init__(self):
        super().__init__('cartesian_tracking')

        # --------------------------------------------------
        # Parameters
        # --------------------------------------------------

        self.declare_parameter('mode', 'sim')
        self.declare_parameter('arm', 'left')
        self.declare_parameter('base_frame', 'world')

        # 100 Hz control
        self.declare_parameter('control_frequency', 100.0)
        self.declare_parameter('command_duration', 0.15)
        self.declare_parameter('command_lead_time', 0.02)

        # IK timeout: 8 ms
        self.declare_parameter('ik_timeout', 0.008)
        self.declare_parameter('target_timeout', 0.5)

        self.declare_parameter('velocity_scale', 0.1)
        self.declare_parameter('max_rate_dt', 0.15)

        # IK statistics logging interval
        self.declare_parameter('ik_stats_interval', 1)

        # Visualization parameters
        self.declare_parameter('max_path_points', 10000)
        self.declare_parameter('path_line_width', 0.005)
        self.declare_parameter('path_min_distance', 0.001)

        self.mode = self.get_parameter('mode').value
        self.arm = self.get_parameter('arm').value
        self.base_frame = self.get_parameter('base_frame').value

        self.control_frequency = float(
            self.get_parameter('control_frequency').value
        )

        self.control_period = 1.0 / self.control_frequency

        self.command_duration = float(
            self.get_parameter('command_duration').value
        )

        self.command_lead_time = float(
            self.get_parameter('command_lead_time').value
        )

        self.ik_timeout = float(
            self.get_parameter('ik_timeout').value
        )

        self.target_timeout = float(
            self.get_parameter('target_timeout').value
        )

        self.velocity_scale = float(
            self.get_parameter('velocity_scale').value
        )

        self.max_rate_dt = float(
            self.get_parameter('max_rate_dt').value
        )

        self.ik_stats_interval = int(
            self.get_parameter('ik_stats_interval').value
        )

        self.max_path_points = int(
            self.get_parameter('max_path_points').value
        )

        self.path_line_width = float(
            self.get_parameter('path_line_width').value
        )

        self.path_min_distance = float(
            self.get_parameter('path_min_distance').value
        )

        if self.arm not in ('left', 'right'):
            raise ValueError("arm must be 'left' or 'right'")

        if self.control_frequency <= 0.0:
            raise ValueError('control_frequency must be positive')

        if not 0.0 < self.velocity_scale <= 1.0:
            raise ValueError('velocity_scale must be in (0, 1]')

        if self.ik_timeout <= 0.0:
            raise ValueError('ik_timeout must be positive')

        # --------------------------------------------------
        # IK timing statistics
        # --------------------------------------------------

        self.ik_request_start_time = None

        self.ik_count = 0
        self.ik_total_time_ms = 0.0
        self.ik_max_time_ms = 0.0
        self.ik_overrun_count = 0

        # --------------------------------------------------
        # Robot configuration
        # --------------------------------------------------

        self.group_name = f'{self.arm}_arm'

        self.ik_link_name = f'openarmx_{self.arm}_hand'

        self.joint_names = [
            f'openarmx_{self.arm}_joint{i}'
            for i in range(1, 8)
        ]

        self.controller_topic = (
            f'/{self.arm}_joint_trajectory_controller/joint_trajectory'
        )

        self.joint_lower_limits = [
            -1.25, -1.70, -1.57, 0.0,
            -1.50, -0.75, -1.40
        ]

        self.joint_upper_limits = [
            3.0, 1.70, 1.57, 1.8,
            1.50, 0.75, 1.40
        ]

        self.joint_max_velocities = [10.47] * 7

        self.joint_max_velocities = [
            v * self.velocity_scale
            for v in self.joint_max_velocities
        ]

        # --------------------------------------------------
        # Internal states
        # --------------------------------------------------

        self.latest_joint_state = None
        self.latest_target = None

        self.target_received_time = None

        self.target_sequence = 0
        self.processed_sequence = 0

        self.ik_pending = False

        self.last_commanded_positions = None
        self.last_command_time = None

        self.target_path_points = []
        self.actual_path_points = []

        # --------------------------------------------------
        # ROS subscriptions
        # --------------------------------------------------

        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        self.target_sub = self.create_subscription(
            PoseStamped,
            '/tcp_target',
            self.target_callback,
            10
        )

        # --------------------------------------------------
        # MoveIt IK service
        # --------------------------------------------------

        self.ik_client = self.create_client(
            GetPositionIK,
            '/compute_ik'
        )

        # --------------------------------------------------
        # Joint trajectory publisher
        # --------------------------------------------------

        self.trajectory_pub = self.create_publisher(
            JointTrajectory,
            self.controller_topic,
            10
        )

        # --------------------------------------------------
        # RViz marker publisher
        # --------------------------------------------------

        self.marker_pub = self.create_publisher(
            Marker,
            '/visualization_marker',
            10
        )

        # --------------------------------------------------
        # TF
        # --------------------------------------------------

        self.tf_buffer = tf2_ros.Buffer()

        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer,
            self
        )

        # --------------------------------------------------
        # Timers
        # --------------------------------------------------

        period = self.control_period

        self.control_timer = self.create_timer(
            period,
            self.control_timer_callback
        )

        self.actual_path_timer = self.create_timer(
            period,
            self.update_actual_path
        )

        # --------------------------------------------------
        # Startup information
        # --------------------------------------------------

        self.get_logger().info('------------------------------------')
        self.get_logger().info('Cartesian tracking started')
        self.get_logger().info(f'Mode: {self.mode}')
        self.get_logger().info(f'Arm: {self.arm}')
        self.get_logger().info(f'IK group: {self.group_name}')
        self.get_logger().info(f'IK link: {self.ik_link_name}')
        self.get_logger().info(f'Base frame: {self.base_frame}')

        self.get_logger().info(
            f'Control frequency: {self.control_frequency:.1f} Hz'
        )

        self.get_logger().info(
            f'Control period: {self.control_period * 1000:.2f} ms'
        )

        self.get_logger().info(
            f'IK timeout: {self.ik_timeout * 1000:.2f} ms'
        )

        self.get_logger().info(
            f'Velocity scale: {self.velocity_scale}'
        )

        self.get_logger().info(
            f'Joint velocity limit: '
            f'{self.joint_max_velocities[0]:.3f} rad/s'
        )

        self.get_logger().info(
            f'Controller topic: {self.controller_topic}'
        )

        self.get_logger().info(
            'Target path: orange / Actual path: green'
        )

        self.get_logger().info('------------------------------------')

        self.last_diag_time = time.monotonic()

        self.ik_request_id = 0
        self.active_ik_request_id = None
    # ======================================================
    # IK timing statistics
    # ======================================================

    def record_ik_timing(self):

        if self.ik_request_start_time is None:
            return

        elapsed = (
            time.perf_counter() - self.ik_request_start_time
        )

        elapsed_ms = elapsed * 1000.0

        self.ik_count += 1
        self.ik_total_time_ms += elapsed_ms

        self.ik_max_time_ms = max(
            self.ik_max_time_ms,
            elapsed_ms
        )

        # Check whether IK response exceeded control period
        if elapsed > self.control_period:
            self.ik_overrun_count += 1

        average_ms = (
            self.ik_total_time_ms / self.ik_count
        )

        self.get_logger().info(
            f'IK response: {elapsed_ms:.3f} ms | '
            f'Average: {average_ms:.3f} ms | '
            f'Max: {self.ik_max_time_ms:.3f} ms | '
            f'10ms overrun: {self.ik_overrun_count}'
        )

        self.ik_request_start_time = None
        self.last_diag_time = time.monotonic()

    # ======================================================
    # Joint state callback
    # ======================================================

    def joint_state_callback(self, msg):
        self.latest_joint_state = copy.deepcopy(msg)

    # ======================================================
    # Target pose callback
    # ======================================================

    def target_callback(self, msg):

        if msg.header.frame_id != self.base_frame:
            self.get_logger().warning(
                f'Ignoring target frame {msg.header.frame_id}. '
                f'Expected {self.base_frame}.'
            )
            return

        self.latest_target = copy.deepcopy(msg)
        self.target_received_time = time.monotonic()

        self.target_sequence += 1

        self.append_path_point(
            self.target_path_points,
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        )

        self.publish_path_marker(
            self.target_path_points,
            namespace='target_path',
            marker_id=0,
            color=(1.0, 0.45, 0.0)
        )

    # ======================================================
    # Path point management
    # ======================================================

    def append_path_point(self, path, x, y, z):

        point = Point()
        point.x = float(x)
        point.y = float(y)
        point.z = float(z)

        if len(path) > 0:

            previous = path[-1]

            distance = math.sqrt(
                (point.x - previous.x) ** 2
                + (point.y - previous.y) ** 2
                + (point.z - previous.z) ** 2
            )

            if distance < self.path_min_distance:
                return

        path.append(point)

        if len(path) > self.max_path_points:
            del path[0]

    # ======================================================
    # Publish RViz path marker
    # ======================================================

    def publish_path_marker(
        self,
        points,
        namespace,
        marker_id,
        color
    ):

        marker = Marker()

        marker.header.frame_id = self.base_frame
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.ns = namespace
        marker.id = marker_id

        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        marker.pose.orientation.w = 1.0

        marker.scale.x = self.path_line_width

        marker.color.r = float(color[0])
        marker.color.g = float(color[1])
        marker.color.b = float(color[2])
        marker.color.a = 1.0

        marker.points = list(points)

        marker.lifetime = Duration(seconds=0.0).to_msg()

        self.marker_pub.publish(marker)

    # ======================================================
    # Read actual TCP pose from TF
    # ======================================================

    def update_actual_path(self):

        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.ik_link_name,
                Time()
            )

        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException
        ):
            return

        translation = transform.transform.translation

        self.append_path_point(
            self.actual_path_points,
            translation.x,
            translation.y,
            translation.z
        )

        self.publish_path_marker(
            self.actual_path_points,
            namespace='actual_path',
            marker_id=1,
            color=(0.0, 1.0, 0.0)
        )

    # ======================================================
    # Main control timer
    # ======================================================

    def control_timer_callback(self):

        now = time.monotonic()

        # Print diagnostic information every 1 second
        if now - self.last_diag_time >= 1.0:

            service_ready = self.ik_client.service_is_ready()

            target_ready = self.latest_target is not None
            joint_ready = self.latest_joint_state is not None

            self.get_logger().info(
                f'[CONTROL STATUS] '
                f'target={target_ready}, '
                f'joint_state={joint_ready}, '
                f'ik_service={service_ready}, '
                f'ik_pending={self.ik_pending}, '
                f'target_seq={self.target_sequence}, '
                f'processed_seq={self.processed_sequence}'
            )

            self.last_diag_time = now

        if self.latest_target is None:
            return

        if self.latest_joint_state is None:
            return

        if not self.ik_client.service_is_ready():
            return

        if self.ik_pending:
            return

        if self.target_sequence == self.processed_sequence:
            return

        elapsed = time.monotonic() - self.target_received_time

        if elapsed > self.target_timeout:
            self.get_logger().warning(
                'Target pose timeout. No new command sent.'
            )
            return

        self.send_ik_request()

    # ======================================================
    # Send IK request
    # ======================================================

    def send_ik_request(self):

        if self.ik_pending:
            return

        request = GetPositionIK.Request()

        request.ik_request.group_name = self.group_name
        request.ik_request.ik_link_name = self.ik_link_name

        request.ik_request.pose_stamped = copy.deepcopy(
            self.latest_target
        )

        request.ik_request.timeout = Duration(
            seconds=self.ik_timeout
        ).to_msg()

        request.ik_request.avoid_collisions = True

        robot_state = RobotState()

        robot_state.joint_state = copy.deepcopy(
            self.latest_joint_state
        )

        robot_state.is_diff = True
        request.ik_request.robot_state = robot_state

        # Assign unique request ID
        self.ik_request_id += 1
        request_id = self.ik_request_id

        self.active_ik_request_id = request_id
        self.ik_pending = True

        request_sequence = self.target_sequence

        self.ik_request_start_time = time.perf_counter()

        self.get_logger().info(
            f'[IK REQUEST] id={request_id}, '
            f'sequence={request_sequence}'
        )

        future = self.ik_client.call_async(request)

        future.add_done_callback(
            lambda f: self.ik_response_callback(
                f,
                request_sequence,
                request_id
            )
        )

    # ======================================================
    # IK response callback
    # ======================================================

    def ik_response_callback(
        self,
        future,
        request_sequence,
        request_id
    ):

        self.get_logger().info(
            f'[IK CALLBACK] id={request_id}, '
            f'sequence={request_sequence}'
        )

        # Ignore callbacks that do not match the active request
        if request_id != self.active_ik_request_id:
            self.get_logger().warning(
                f'Unexpected IK callback: id={request_id}'
            )
            return

        self.ik_pending = False
        self.active_ik_request_id = None

        self.record_ik_timing()

        try:
            response = future.result()

        except Exception as error:
            self.get_logger().error(
                f'IK service call failed: {error}'
            )
            return

        # Discard outdated IK solution
        if request_sequence != self.target_sequence:
            return

        self.processed_sequence = request_sequence

        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().warning(
                f'IK failed. MoveIt error code: '
                f'{response.error_code.val}'
            )
            return

        solution_state = response.solution.joint_state

        solution_map = dict(
            zip(
                solution_state.name,
                solution_state.position
            )
        )

        if not all(
            name in solution_map
            for name in self.joint_names
        ):
            self.get_logger().error(
                'IK result does not contain all 7 arm joints.'
            )
            return

        target_positions = [
            solution_map[name]
            for name in self.joint_names
        ]

        if not self.check_joint_limits(target_positions):
            self.get_logger().warning(
                'IK solution is outside joint position limits.'
            )
            return

        limited_positions, dt = self.apply_velocity_limit(
            target_positions
        )

        self.publish_joint_trajectory(
            limited_positions,
            dt
        )

    # ======================================================
    # Joint position limit check
    # ======================================================

    def check_joint_limits(self, positions):

        for i, position in enumerate(positions):

            if position < self.joint_lower_limits[i]:
                self.get_logger().warning(
                    f'{self.joint_names[i]} below lower limit: '
                    f'{position:.4f}'
                )
                return False

            if position > self.joint_upper_limits[i]:
                self.get_logger().warning(
                    f'{self.joint_names[i]} above upper limit: '
                    f'{position:.4f}'
                )
                return False

        return True

    # ======================================================
    # Joint velocity limiter
    # ======================================================

    def apply_velocity_limit(self, target_positions):

        now = time.monotonic()

        if self.last_command_time is None:
            dt = self.control_period

        else:
            dt = now - self.last_command_time

        dt = max(0.001, min(dt, self.max_rate_dt))

        if self.last_commanded_positions is None:

            joint_state_map = dict(
                zip(
                    self.latest_joint_state.name,
                    self.latest_joint_state.position
                )
            )

            current_positions = [
                joint_state_map[name]
                for name in self.joint_names
            ]

        else:
            current_positions = self.last_commanded_positions

        limited_positions = []

        for i in range(7):

            current = current_positions[i]
            target = target_positions[i]

            max_delta = (
                self.joint_max_velocities[i] * dt
            )

            delta = target - current

            limited_delta = max(
                -max_delta,
                min(delta, max_delta)
            )

            limited_positions.append(
                current + limited_delta
            )

        self.last_commanded_positions = list(
            limited_positions
        )

        self.last_command_time = now

        return limited_positions, dt

    # ======================================================
    # Publish joint trajectory
    # ======================================================

    def publish_joint_trajectory(
        self,
        positions,
        dt
    ):

        trajectory = JointTrajectory()

        trajectory.joint_names = list(
            self.joint_names
        )

        trajectory.header.stamp = (
            self.get_clock().now()
            + Duration(seconds=self.command_lead_time)
        ).to_msg()

        point = JointTrajectoryPoint()

        point.positions = list(positions)

        execution_duration = max(
            self.command_duration,
            dt
        )

        point.time_from_start = Duration(
            seconds=execution_duration
        ).to_msg()

        trajectory.points.append(point)

        self.trajectory_pub.publish(trajectory)


def main(args=None):

    rclpy.init(args=args)

    node = CartesianTracking()

    executor = MultiThreadedExecutor(
        num_threads=4
    )

    executor.add_node(node)

    try:
        executor.spin()

    except KeyboardInterrupt:
        node.get_logger().info(
            'Cartesian tracking stopped.'
        )

    finally:
        executor.remove_node(node)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
