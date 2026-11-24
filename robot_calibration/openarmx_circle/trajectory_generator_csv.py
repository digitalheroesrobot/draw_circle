#!/usr/bin/env python3

import csv
import math
import os

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool

import tf2_ros


class TrajectoryGenerator(Node):

    def __init__(self):
        super().__init__('trajectory_generator')

        # --------------------------------------------------
        # Parameters
        # --------------------------------------------------
        self.declare_parameter('arm', 'left')

        self.declare_parameter(
            'csv_file',
            '~/openarmx_ws/src/openarmx_circle/trajectory.csv'
        )

        self.declare_parameter('base_frame', 'world')

        self.declare_parameter('publish_frequency', 50.0)

        # 초기 TCP 위치에서 CSV 첫 번째 점까지 이동하는 시간
        self.declare_parameter('transition_duration', 3.0)

        self.arm = self.get_parameter('arm').value

        self.csv_file = os.path.expanduser(
            self.get_parameter('csv_file').value
        )

        self.base_frame = self.get_parameter('base_frame').value

        self.publish_frequency = float(
            self.get_parameter('publish_frequency').value
        )

        self.transition_duration = float(
            self.get_parameter('transition_duration').value
        )

        if self.arm not in ('left', 'right'):
            raise ValueError("arm must be 'left' or 'right'")

        if self.publish_frequency <= 0.0:
            raise ValueError('publish_frequency must be positive')

        if self.transition_duration < 0.0:
            raise ValueError('transition_duration cannot be negative')

        # IK에 사용하는 TCP link
        self.tcp_link = f'openarmx_{self.arm}_hand'

        # --------------------------------------------------
        # TF
        # --------------------------------------------------
        self.tf_buffer = tf2_ros.Buffer()

        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer,
            self
        )

        # --------------------------------------------------
        # Publisher / Subscriber
        # --------------------------------------------------
        self.publisher = self.create_publisher(
            PoseStamped,
            '/tcp_target',
            10
        )

        self.start_sub = self.create_subscription(
            Bool,
            '/trajectory_start',
            self.start_callback,
            10
        )
        self.state_publisher = self.create_publisher(
            Bool,
            '/trajectory_state',
            10
        )
        # --------------------------------------------------
        # Load CSV
        # --------------------------------------------------
        self.trajectory = self.load_csv(self.csv_file)

        # --------------------------------------------------
        # Execution state
        # --------------------------------------------------
        self.running = False
        self.start_time = None

        self.initial_pose = None
        self.target_start_pose = None

        self.timer = self.create_timer(
            1.0 / self.publish_frequency,
            self.publish_callback
        )

        self.get_logger().info(
            f'Loaded {len(self.trajectory)} CSV points'
        )

        self.get_logger().info(
            f'Arm: {self.arm}, TCP: {self.tcp_link}'
        )

        self.get_logger().info(
            f'Transition duration: {self.transition_duration:.2f} s'
        )

        self.get_logger().info(
            'Waiting for /trajectory_start = true'
        )

    # ======================================================
    # CSV loading
    # ======================================================
    def load_csv(self, filename):

        trajectory = []

        required_columns = [
            't',
            'position.x',
            'position.y',
            'position.z',
            'q.x',
            'q.y',
            'q.z',
            'q.w',
            'on_off'
        ]

        with open(filename, 'r', newline='') as file:

            reader = csv.DictReader(file)

            if not reader.fieldnames:
                raise ValueError('CSV header is missing')

            missing = [
                name for name in required_columns
                if name not in reader.fieldnames
            ]

            if missing:
                raise ValueError(
                    f'Missing CSV columns: {missing}'
                )

            for row in reader:

                q = self.normalize_quaternion([
                    float(row['q.x']),
                    float(row['q.y']),
                    float(row['q.z']),
                    float(row['q.w'])
                ])
                on_off = int(row['on_off'])

                if on_off not in (0, 1):
                    raise ValueError(
                        f'Invalid on_off value: {on_off}'
                    )
                trajectory.append({
                    't': float(row['t']),

                    'x': float(row['position.x']),
                    'y': float(row['position.y']),
                    'z': float(row['position.z']),

                    'qx': q[0],
                    'qy': q[1],
                    'qz': q[2],
                    'qw': q[3],

                    'on_off': on_off

                })

        if not trajectory:
            raise ValueError('CSV trajectory is empty')

        # 시간은 반드시 증가해야 함
        for i in range(1, len(trajectory)):

            if trajectory[i]['t'] <= trajectory[i - 1]['t']:
                raise ValueError(
                    'CSV timestamps must be strictly increasing'
                )

        # CSV 궤적은 t=0부터 시작하도록 제한
        if abs(trajectory[0]['t']) > 1e-6:
            raise ValueError(
                'The first CSV timestamp must be 0.0 seconds'
            )

        return trajectory

    # ======================================================
    # Quaternion utilities
    # Quaternion order: x, y, z, w
    # ======================================================
    def normalize_quaternion(self, q):

        norm = math.sqrt(sum(v * v for v in q))

        if norm < 1e-12:
            raise ValueError('Invalid zero quaternion')

        return [v / norm for v in q]

    def slerp(self, q1, q2, alpha):

        q1 = self.normalize_quaternion(q1)
        q2 = self.normalize_quaternion(q2)

        dot = sum(a * b for a, b in zip(q1, q2))

        # 가장 짧은 회전 방향 선택
        if dot < 0.0:
            q2 = [-v for v in q2]
            dot = -dot

        dot = max(-1.0, min(1.0, dot))

        # 두 quaternion이 거의 같을 때 선형 보간
        if dot > 0.9995:

            result = [
                a + alpha * (b - a)
                for a, b in zip(q1, q2)
            ]

            return self.normalize_quaternion(result)

        theta_0 = math.acos(dot)
        sin_theta_0 = math.sin(theta_0)

        theta = theta_0 * alpha

        s1 = math.sin(theta_0 - theta) / sin_theta_0
        s2 = math.sin(theta) / sin_theta_0

        result = [
            s1 * a + s2 * b
            for a, b in zip(q1, q2)
        ]

        return self.normalize_quaternion(result)

    # ======================================================
    # Pose utilities
    # ======================================================
    def pose_to_dict(self, pose):

        return {
            'x': pose.position.x,
            'y': pose.position.y,
            'z': pose.position.z,

            'qx': pose.orientation.x,
            'qy': pose.orientation.y,
            'qz': pose.orientation.z,
            'qw': pose.orientation.w
        }

    def get_current_tcp_pose(self):

        transform = self.tf_buffer.lookup_transform(
            self.base_frame,
            self.tcp_link,
            Time(),
            timeout=Duration(seconds=1.0)
        )

        t = transform.transform.translation
        q = transform.transform.rotation

        quaternion = self.normalize_quaternion([
            q.x, q.y, q.z, q.w
        ])

        return {
            'x': t.x,
            'y': t.y,
            'z': t.z,

            'qx': quaternion[0],
            'qy': quaternion[1],
            'qz': quaternion[2],
            'qw': quaternion[3]
        }

    # ======================================================
    # Start / Stop
    # ======================================================
    def start_callback(self, msg):

        if msg.data:

            if self.running:
                self.get_logger().warn(
                    'Trajectory is already running'
                )
                return

            try:
                # 현재 로봇 TCP 위치 읽기
                self.initial_pose = self.get_current_tcp_pose()

            except Exception as e:
                self.get_logger().error(
                    f'Cannot read current TCP pose: {e}'
                )
                return

            # CSV의 첫 번째 Pose
            self.target_start_pose = self.trajectory[0].copy()

            # 궤적 시작 시각
            self.start_time = self.get_clock().now()

            self.running = True

            self.get_logger().info(
                'Trajectory started'
            )

            self.get_logger().info(
                'Transitioning from current TCP pose '
                'to CSV start pose'
            )

        else:

            self.running = False

            # Stop 시 OFF 상태 발행
            self.publish_state(0)

            self.get_logger().info(
                'Trajectory publishing stopped'
            )

    # ======================================================
    # Smoothstep
    # ======================================================
    def smoothstep(self, alpha):

        alpha = max(0.0, min(1.0, alpha))

        return alpha * alpha * (3.0 - 2.0 * alpha)

    # ======================================================
    # Pose interpolation
    # ======================================================
    def interpolate_pose(self, pose1, pose2, alpha):

        # 위치: Linear interpolation에 Smoothstep 적용
        s = self.smoothstep(alpha)

        result = {}

        result['x'] = pose1['x'] + s * (
            pose2['x'] - pose1['x']
        )

        result['y'] = pose1['y'] + s * (
            pose2['y'] - pose1['y']
        )

        result['z'] = pose1['z'] + s * (
            pose2['z'] - pose1['z']
        )

        # 자세: Quaternion SLERP
        q1 = [
            pose1['qx'], pose1['qy'],
            pose1['qz'], pose1['qw']
        ]

        q2 = [
            pose2['qx'], pose2['qy'],
            pose2['qz'], pose2['qw']
        ]

        q = self.slerp(q1, q2, s)

        result['qx'] = q[0]
        result['qy'] = q[1]
        result['qz'] = q[2]
        result['qw'] = q[3]

        return result

    # ======================================================
    # CSV trajectory interpolation
    # ======================================================
    def get_csv_pose(self, elapsed):

        # 마지막 시간 이후에는 마지막 Pose 유지
        if elapsed >= self.trajectory[-1]['t']:
            return self.trajectory[-1].copy()

        # 해당 시간에 포함되는 두 행 찾기
        for i in range(len(self.trajectory) - 1):

            p1 = self.trajectory[i]
            p2 = self.trajectory[i + 1]

            if p1['t'] <= elapsed <= p2['t']:

                dt = p2['t'] - p1['t']

                alpha = (elapsed - p1['t']) / dt

                # CSV 구간 내부는 시간 기준 보간
                result = {}

                for key in ('x', 'y', 'z'):

                    result[key] = (
                        p1[key] + alpha * (p2[key] - p1[key])
                    )

                q1 = [
                    p1['qx'], p1['qy'],
                    p1['qz'], p1['qw']
                ]

                q2 = [
                    p2['qx'], p2['qy'],
                    p2['qz'], p2['qw']
                ]

                q = self.slerp(q1, q2, alpha)

                result['qx'] = q[0]
                result['qy'] = q[1]
                result['qz'] = q[2]
                result['qw'] = q[3]

                return result

        return self.trajectory[-1].copy()

    # ======================================================
    # Publish PoseStamped
    # ======================================================
    def publish_pose(self, pose):

        msg = PoseStamped()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.base_frame

        msg.pose.position.x = pose['x']
        msg.pose.position.y = pose['y']
        msg.pose.position.z = pose['z']

        msg.pose.orientation.x = pose['qx']
        msg.pose.orientation.y = pose['qy']
        msg.pose.orientation.z = pose['qz']
        msg.pose.orientation.w = pose['qw']

        self.publisher.publish(msg)

    # ======================================================
    # Main timer callback
    # ======================================================
    # ======================================================
    # Main timer callback
    # ======================================================

    def publish_callback(self):

        if not self.running:
            return

        elapsed = (
            self.get_clock().now() - self.start_time
        ).nanoseconds / 1e9

        # --------------------------------------------------
        # Phase 1:
        # Initial TCP pose -> CSV first pose
        # --------------------------------------------------
        if elapsed < self.transition_duration:

            if self.transition_duration > 0.0:
                alpha = elapsed / self.transition_duration
            else:
                alpha = 1.0

            target_pose = self.interpolate_pose(
                self.initial_pose,
                self.target_start_pose,
                alpha
            )

            self.publish_pose(target_pose)

            # Transition 구간에서는 OFF
            self.publish_state(0)

            return

        # --------------------------------------------------
        # Phase 2:
        # Execute CSV trajectory
        # --------------------------------------------------

        csv_elapsed = elapsed - self.transition_duration

        target_pose = self.get_csv_pose(csv_elapsed)

        self.publish_pose(target_pose)

        # CSV 시간에 대응하는 ON/OFF 상태
        target_state = self.get_csv_state(csv_elapsed)

        self.publish_state(target_state)

        # --------------------------------------------------
        # Phase 3:
        # Finish
        # --------------------------------------------------

        if csv_elapsed >= self.trajectory[-1]['t']:

            self.running = False

            self.get_logger().info(
                'CSV trajectory completed'
            )

            self.get_logger().info(
                f'Final ON/OFF state: {target_state}'
            )

    # ======================================================
    # ON/OFF state publisher
    # ======================================================

    def publish_state(self, state):

        msg = Bool()
        msg.data = bool(state)

        self.state_publisher.publish(msg)

    # ======================================================
    # CSV ON/OFF interpolation (Zero-order hold)
    # ======================================================

    def get_csv_state(self, elapsed):

        if elapsed <= 0.0:
            return self.trajectory[0]['on_off']

        if elapsed >= self.trajectory[-1]['t']:
            return self.trajectory[-1]['on_off']

        for i in range(len(self.trajectory) - 1):

            p1 = self.trajectory[i]
            p2 = self.trajectory[i + 1]

            if p1['t'] <= elapsed < p2['t']:

                return p1['on_off']

        return self.trajectory[-1]['on_off']

def main(args=None):

    rclpy.init(args=args)

    node = TrajectoryGenerator()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
