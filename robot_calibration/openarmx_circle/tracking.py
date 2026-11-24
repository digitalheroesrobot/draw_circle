
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
from moveit_msgs.msg import RobotState, MoveItErrorCodes
from visualization_msgs.msg import Marker

import tf2_ros


class CartesianTracking(Node):

    def __init__(self):
        super().__init__('cartesian_tracking')

        # --------------------------------------------------
        # Parameters
        # --------------------------------------------------

        self.declare_parameter('arm', 'left')
        self.declare_parameter('base_frame', 'world')

        self.declare_parameter('control_frequency', 100.0)

        # Streaming command horizon.
        # Verify against the active controller before changing.
        self.declare_parameter('command_duration', 0.06)
        self.declare_parameter('command_lead_time', 0.02)

        # MoveIt IK solver timeout.
        self.declare_parameter('ik_timeout', 0.03)
        self.declare_parameter('target_timeout', 0.5)

        self.declare_parameter('velocity_scale', 0.1)
        self.declare_parameter('max_rate_dt', 0.15)

        self.declare_parameter('ik_stats_interval', 50)

        # SAFETY:
        # No joint commands are published unless explicitly enabled.
        self.declare_parameter(
            'enable_command_publishing',
            False
        )

        self.declare_parameter('max_path_points', 10000)
        self.declare_parameter('path_line_width', 0.005)
        self.declare_parameter('path_min_distance', 0.001)

        self.arm = str(self.get_parameter('arm').value)
        self.base_frame = str(
            self.get_parameter('base_frame').value
        )

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

        self.ik_stats_interval = max(
            1,
            int(self.get_parameter('ik_stats_interval').value)
        )

        self.enable_command_publishing = bool(
            self.get_parameter(
                'enable_command_publishing'
            ).value
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

        if self.command_duration <= 0.0:
            raise ValueError('command_duration must be positive')

        if self.command_lead_time < 0.0:
            raise ValueError('command_lead_time cannot be negative')

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
            f'/{self.arm}_joint_trajectory_controller/'
            'joint_trajectory'
        )

        # Position limits copied from the active OpenArmX V10
        # bimanual URDF (robot_bimanual.urdf). The mirrored arms
        # have different joint1 and joint2 ranges.
        joint_limits_by_arm = {
            'left': {
                'lower': [
                    -3.344396, -3.270796, -1.570000, 0.000000,
                    -1.500000, -0.750000, -1.400000
                ],
                'upper': [
                    0.905604, 0.129204, 1.570000, 1.800000,
                    1.500000, 0.750000, 1.400000
                ],
            },
            'right': {
                'lower': [
                    -1.250000, -0.129204, -1.570000, 0.000000,
                    -1.500000, -0.750000, -1.400000
                ],
                'upper': [
                    3.000000, 3.270796, 1.570000, 1.800000,
                    1.500000, 0.750000, 1.400000
                ],
            },
        }

        self.joint_lower_limits = list(
            joint_limits_by_arm[self.arm]['lower']
        )
        self.joint_upper_limits = list(
            joint_limits_by_arm[self.arm]['upper']
        )

        self.joint_max_velocities = [
            10.47 * self.velocity_scale
            for _ in range(7)
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
        self.ik_request_id = 0
        self.active_ik_request_id = None
        self.ik_request_start_time = None

        self.last_commanded_positions = None
        self.last_command_time = None

        # Statistics
        self.ik_count = 0
        self.ik_total_time_ms = 0.0
        self.ik_max_time_ms = 0.0
        self.ik_overrun_count = 0

        self.ik_success_count = 0
        self.ik_failure_count = 0
        self.ik_stale_count = 0
        self.command_count = 0
        self.velocity_limited_count = 0

        self.target_path_points = []
        self.actual_path_points = []

        self.last_diag_time = time.monotonic()

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

        self.control_timer = self.create_timer(
            self.control_period,
            self.control_timer_callback
        )

        self.actual_path_timer = self.create_timer(
            self.control_period,
            self.update_actual_path
        )

        # --------------------------------------------------
        # Startup information
        # --------------------------------------------------

        self.get_logger().info('======================================')
        self.get_logger().info('Cartesian Tracking - Improved')
        self.get_logger().info(f'Arm: {self.arm}')
        self.get_logger().info(f'Base frame: {self.base_frame}')
        self.get_logger().info(f'IK group: {self.group_name}')
        self.get_logger().info(f'IK link: {self.ik_link_name}')

        self.get_logger().info(
            f'Control frequency: {self.control_frequency:.1f} Hz'
        )

        self.get_logger().info(
            f'Command duration: {self.command_duration:.3f} s'
        )

        self.get_logger().info(
            f'Command lead time: {self.command_lead_time:.3f} s'
        )

        self.get_logger().info(
            f'IK timeout: {self.ik_timeout:.3f} s'
        )

        self.get_logger().info(
            f'Joint velocity limit: '
            f'{self.joint_max_velocities[0]:.3f} rad/s'
        )

        self.get_logger().info(
            f'Command publishing enabled: '
            f'{self.enable_command_publishing}'
        )

        self.get_logger().info(
            f'Controller topic: {self.controller_topic}'
        )

        if not self.enable_command_publishing:
            self.get_logger().warning(
                'COMMAND OUTPUT DISABLED. '
                'No joint trajectory commands will be published.'
            )

        self.get_logger().info('======================================')

    # ======================================================
    # Joint state callback
    # ======================================================

    def joint_state_callback(self, msg):
        self.latest_joint_state = copy.deepcopy(msg)

    # ======================================================
    # Target callback
    # ======================================================

    def target_callback(self, msg):

        if msg.header.frame_id != self.base_frame:
            self.get_logger().warning(
                f'Ignoring target frame "{msg.header.frame_id}". '
                f'Expected "{self.base_frame}".'
            )
            return

        p = msg.pose.position
        q = msg.pose.orientation

        values = [
            p.x, p.y, p.z,
            q.x, q.y, q.z, q.w
        ]

        if not all(math.isfinite(v) for v in values):
            self.get_logger().warning(
                'Ignoring target containing NaN or infinity.'
            )
            return

        q_norm = math.sqrt(
            q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w
        )

        if q_norm < 1e-8:
            self.get_logger().warning(
                'Ignoring target with invalid zero quaternion.'
            )
            return

        self.latest_target = copy.deepcopy(msg)
        self.target_received_time = time.monotonic()
        self.target_sequence += 1

        self.append_path_point(
            self.target_path_points,
            p.x, p.y, p.z
        )

        self.publish_path_marker(
            self.target_path_points,
            'target_path',
            0,
            (1.0, 0.45, 0.0)
        )

    # ======================================================
    # Path point management
    # ======================================================

    def append_path_point(self, path, x, y, z):

        point = Point()
        point.x = float(x)
        point.y = float(y)
        point.z = float(z)

        if path:
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
    # RViz path marker
    # ======================================================

    def publish_path_marker(
        self, points, namespace, marker_id, color
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
    # Actual TCP path
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

        t = transform.transform.translation

        self.append_path_point(
            self.actual_path_points,
            t.x, t.y, t.z
        )

        self.publish_path_marker(
            self.actual_path_points,
            'actual_path',
            1,
            (0.0, 1.0, 0.0)
        )

    # ======================================================
    # Control timer
    # ======================================================

    def control_timer_callback(self):

        self.publish_diagnostics()

        if self.ik_pending:
            return

        self.dispatch_latest_target()

    # ======================================================
    # Diagnostics
    # ======================================================

    def publish_diagnostics(self):

        now = time.monotonic()

        if now - self.last_diag_time < 1.0:
            return

        target_ready = self.latest_target is not None
        joint_ready = self.latest_joint_state is not None

        service_ready = self.ik_client.service_is_ready()

        target_age = -1.0

        if self.target_received_time is not None:
            target_age = now - self.target_received_time

        self.get_logger().info(
            '[CONTROL STATUS] '
            f'target={target_ready}, '
            f'joint_state={joint_ready}, '
            f'ik_service={service_ready}, '
            f'ik_pending={self.ik_pending}, '
            f'target_seq={self.target_sequence}, '
            f'processed_seq={self.processed_sequence}, '
            f'target_age={target_age:.3f}s, '
            f'IK_success={self.ik_success_count}, '
            f'IK_failed={self.ik_failure_count}, '
            f'IK_stale={self.ik_stale_count}, '
            f'commands={self.command_count}, '
            f'velocity_limited={self.velocity_limited_count}'
        )

        self.last_diag_time = now

    # ======================================================
    # Latest-target dispatcher
    # ======================================================

    def dispatch_latest_target(self):

        if self.latest_target is None:
            return

        if self.latest_joint_state is None:
            return

        if not self.ik_client.service_is_ready():
            return

        if self.ik_pending:
            return

        if self.target_received_time is None:
            return

        target_age = (
            time.monotonic() - self.target_received_time
        )

        if target_age > self.target_timeout:
            return

        if self.target_sequence == self.processed_sequence:
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

        request_sequence = self.target_sequence

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

        self.ik_request_id += 1
        request_id = self.ik_request_id

        self.active_ik_request_id = request_id
        self.ik_pending = True

        self.ik_request_start_time = time.perf_counter()

        try:
            future = self.ik_client.call_async(request)

            future.add_done_callback(
                lambda f: self.ik_response_callback(
                    f,
                    request_sequence,
                    request_id
                )
            )

        except Exception as error:
            self.ik_pending = False
            self.active_ik_request_id = None

            self.get_logger().error(
                f'Could not send IK request: {error}'
            )

    # ======================================================
    # IK timing
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

        if elapsed > self.control_period:
            self.ik_overrun_count += 1

        average_ms = (
            self.ik_total_time_ms / self.ik_count
        )

        if self.ik_count % self.ik_stats_interval == 0:
            self.get_logger().info(
                f'[IK TIMING] '
                f'current={elapsed_ms:.2f}ms, '
                f'average={average_ms:.2f}ms, '
                f'max={self.ik_max_time_ms:.2f}ms, '
                f'overrun={self.ik_overrun_count}/{self.ik_count}'
            )

        self.ik_request_start_time = None

    # ======================================================
    # IK response callback
    # ======================================================

    def ik_response_callback(
        self, future, request_sequence, request_id
    ):

        if request_id != self.active_ik_request_id:
            self.get_logger().warning(
                f'Unexpected IK callback: {request_id}'
            )
            return

        self.ik_pending = False
        self.active_ik_request_id = None

        self.record_ik_timing()

        try:
            response = future.result()

        except Exception as error:
            self.ik_failure_count += 1

            self.get_logger().error(
                f'IK service call failed: {error}'
            )

            self.dispatch_latest_target()
            return

        # A newer Cartesian target arrived while IK was running.
        # Do not send an obsolete joint command.
        if request_sequence != self.target_sequence:

            self.ik_stale_count += 1

            self.get_logger().warning(
                f'[IK STALE] request_seq={request_sequence}, '
                f'latest_seq={self.target_sequence}. '
                f'Old solution discarded.'
            )

            # Immediately process the newest target.
            self.dispatch_latest_target()
            return

        self.processed_sequence = request_sequence

        if response.error_code.val != MoveItErrorCodes.SUCCESS:

            self.ik_failure_count += 1

            self.get_logger().warning(
                f'IK failed: MoveIt error '
                f'{response.error_code.val}'
            )

            self.dispatch_latest_target()
            return

        target_positions = self.extract_arm_positions(
            response.solution.joint_state
        )

        if target_positions is None:
            self.ik_failure_count += 1
            self.dispatch_latest_target()
            return

        if not self.check_joint_limits(target_positions):
            self.ik_failure_count += 1
            self.dispatch_latest_target()
            return

        self.ik_success_count += 1

        limited_positions, limited = self.apply_velocity_limit(
            target_positions
        )

        if limited:
            self.velocity_limited_count += 1

        if self.enable_command_publishing:

            self.publish_joint_trajectory(
                limited_positions
            )

        # Do not wait for the next timer tick.
        # Process the newest target immediately if one arrived.
        self.dispatch_latest_target()

    # ======================================================
    # Extract arm joints from IK result
    # ======================================================

    def extract_arm_positions(self, joint_state):

        solution_map = dict(
            zip(
                joint_state.name,
                joint_state.position
            )
        )

        if not all(
            name in solution_map
            for name in self.joint_names
        ):
            self.get_logger().error(
                'IK result does not contain all 7 arm joints.'
            )
            return None

        positions = [
            float(solution_map[name])
            for name in self.joint_names
        ]

        if not all(math.isfinite(p) for p in positions):
            self.get_logger().error(
                'IK result contains NaN or infinity.'
            )
            return None

        return positions

    # ======================================================
    # Joint position limits
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

        dt = max(
            0.001,
            min(dt, self.max_rate_dt)
        )

        if self.last_commanded_positions is None:

            joint_state_map = dict(
                zip(
                    self.latest_joint_state.name,
                    self.latest_joint_state.position
                )
            )

            if not all(
                name in joint_state_map
                for name in self.joint_names
            ):
                raise RuntimeError(
                    'Current joint state is missing arm joints.'
                )

            current_positions = [
                float(joint_state_map[name])
                for name in self.joint_names
            ]

        else:
            current_positions = list(
                self.last_commanded_positions
            )

        limited_positions = []
        was_limited = False

        for i in range(7):

            current = current_positions[i]
            target = target_positions[i]

            max_delta = (
                self.joint_max_velocities[i] * dt
            )

            delta = target - current

            if abs(delta) > max_delta:
                was_limited = True

            limited_delta = max(
                -max_delta,
                min(delta, max_delta)
            )

            limited_positions.append(
                current + limited_delta
            )

        return limited_positions, was_limited

    # ======================================================
    # Publish joint trajectory
    # ======================================================

    def publish_joint_trajectory(self, positions):

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

        point.time_from_start = Duration(
            seconds=self.command_duration
        ).to_msg()

        trajectory.points.append(point)

        self.trajectory_pub.publish(trajectory)

        self.last_commanded_positions = list(positions)
        self.last_command_time = time.monotonic()

        self.command_count += 1

        if self.command_count % 50 == 0:
            self.get_logger().info(
                f'[COMMAND] count={self.command_count}, '
                f'duration={self.command_duration:.3f}s'
            )


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
