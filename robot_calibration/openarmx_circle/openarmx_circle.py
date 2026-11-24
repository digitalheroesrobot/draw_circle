
#!/usr/bin/env python3

import math
import sys

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.action import ActionClient

from geometry_msgs.msg import Pose
from moveit_msgs.srv import GetCartesianPath
from moveit_msgs.action import ExecuteTrajectory

import tf2_ros


class OpenArmXCircle(Node):

    def __init__(self):
        super().__init__('openarmx_circle')

        # =====================================
        # Robot configuration
        # =====================================

        self.base_frame = 'world'
        self.tcp_link = 'openarmx_left_hand'
        self.group_name = 'left_arm'

        # Circle parameters
        self.radius = 0.02
        self.num_points = 100

        # Lift TCP by 10 cm
        self.lift_height = 0.10

        # Cartesian planning parameters
        self.max_step = 0.005
        self.jump_threshold = 0.0

        self.velocity_scale = 0.10
        self.acceleration_scale = 0.10

        # =====================================
        # TF2
        # =====================================

        self.tf_buffer = tf2_ros.Buffer()

        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer,
            self
        )

        # =====================================
        # MoveIt2 service
        # =====================================

        self.cartesian_client = self.create_client(
            GetCartesianPath,
            '/compute_cartesian_path'
        )

        # =====================================
        # Execute trajectory action
        # =====================================

        self.execute_client = ActionClient(
            self,
            ExecuteTrajectory,
            '/execute_trajectory'
        )

        self.get_logger().info(
            'OpenArmX Circle node initialized.'
        )

    # =========================================
    # Get current TCP pose
    # =========================================

    
    def get_current_tcp_pose(self):

        self.get_logger().info(
            'Waiting for TCP transform...'
        )

        start_time = self.get_clock().now()

        timeout_sec = 20.0

        while rclpy.ok():

            # TF subscription callbacks must be processed
            rclpy.spin_once(
                self,
                timeout_sec=0.1
            )

            try:

                transform = self.tf_buffer.lookup_transform(
                    self.base_frame,
                    self.tcp_link,
                    Time(),
                    timeout=Duration(seconds=0.2)
                )

                self.get_logger().info(
                    'TCP transform received successfully.'
                )

                return transform.transform

            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException
            ) as e:

                elapsed = (
                    self.get_clock().now() - start_time
                ).nanoseconds / 1e9

                self.get_logger().warn(
                    f'Waiting for TF: {e}'
                )

                if elapsed > timeout_sec:

                    self.get_logger().error(
                        'TF lookup timeout.'
                    )

                    return None

        return None


    # =========================================
    # Convert TF transform to Pose
    # =========================================

    def transform_to_pose(self, transform):

        pose = Pose()

        pose.position.x = transform.translation.x
        pose.position.y = transform.translation.y
        pose.position.z = transform.translation.z

        pose.orientation.x = transform.rotation.x
        pose.orientation.y = transform.rotation.y
        pose.orientation.z = transform.rotation.z
        pose.orientation.w = transform.rotation.w

        return pose

    # =========================================
    # Compute Cartesian path
    # =========================================

    def compute_cartesian_path(self, waypoints):

        if not self.cartesian_client.wait_for_service(
            timeout_sec=10.0
        ):

            self.get_logger().error(
                'Cartesian path service unavailable.'
            )

            return None

        request = GetCartesianPath.Request()

        request.header.frame_id = self.base_frame
        request.header.stamp = self.get_clock().now().to_msg()

        request.start_state.is_diff = True

        request.group_name = self.group_name
        request.link_name = self.tcp_link

        request.waypoints = waypoints

        request.max_step = self.max_step
        request.jump_threshold = self.jump_threshold

        request.avoid_collisions = True

        request.max_velocity_scaling_factor = (
            self.velocity_scale
        )

        request.max_acceleration_scaling_factor = (
            self.acceleration_scale
        )

        self.get_logger().info(
            'Computing Cartesian path...'
        )

        future = self.cartesian_client.call_async(request)

        rclpy.spin_until_future_complete(
            self,
            future,
            timeout_sec=60.0
        )

        if not future.done():

            self.get_logger().error(
                'Cartesian planning timeout.'
            )

            return None

        response = future.result()

        if response is None:

            self.get_logger().error(
                'Cartesian planning failed.'
            )

            return None

        self.get_logger().info(
            f'Cartesian fraction: {response.fraction:.4f}'
        )

        self.get_logger().info(
            f'MoveIt error code: {response.error_code.val}'
        )

        if response.fraction < 0.999:

            self.get_logger().error(
                'Incomplete Cartesian path. '
                'Trajectory will NOT be executed.'
            )

            return None

        if response.error_code.val != 1:

            self.get_logger().error(
                'MoveIt returned an unsuccessful error code.'
            )

            return None

        return response.solution

    # =========================================
    # Execute trajectory
    # =========================================

    def execute_trajectory(self, trajectory):

        if not self.execute_client.wait_for_server(
            timeout_sec=10.0
        ):

            self.get_logger().error(
                'Execute trajectory action unavailable.'
            )

            return False

        print()
        print('===================================')
        print('Trajectory is ready.')
        print('Check the robot workspace.')
        print('Type YES to execute.')
        print('===================================')

        answer = input('Execute? ')

        if answer.strip() != 'YES':

            self.get_logger().warn(
                'Execution cancelled by user.'
            )

            return False

        goal = ExecuteTrajectory.Goal()

        goal.trajectory = trajectory

        goal_future = self.execute_client.send_goal_async(
            goal
        )

        rclpy.spin_until_future_complete(
            self,
            goal_future,
            timeout_sec=30.0
        )

        if not goal_future.done():

            self.get_logger().error(
                'Action goal response timeout.'
            )

            return False

        goal_handle = goal_future.result()

        if goal_handle is None or not goal_handle.accepted:

            self.get_logger().error(
                'Trajectory goal rejected.'
            )

            return False

        self.get_logger().info(
            'Trajectory execution started.'
        )

        result_future = goal_handle.get_result_async()

        rclpy.spin_until_future_complete(
            self,
            result_future,
            timeout_sec=120.0
        )

        if not result_future.done():

            self.get_logger().error(
                'Trajectory execution timeout.'
            )

            return False

        result = result_future.result()

        if result is None:

            return False

        if result.result.error_code.val != 1:

            self.get_logger().error(
                f'Execution error: '
                f'{result.result.error_code.val}'
            )

            return False

        self.get_logger().info(
            'Trajectory execution completed.'
        )

        return True

    # =========================================
    # Create lifted pose
    # =========================================

    def make_lift_waypoints(self, transform):

        current_pose = self.transform_to_pose(transform)

        lifted_pose = Pose()

        lifted_pose.position.x = current_pose.position.x
        lifted_pose.position.y = current_pose.position.y

        lifted_pose.position.z = (
            current_pose.position.z + self.lift_height
        )

        lifted_pose.orientation = current_pose.orientation

        self.get_logger().info(
            'Creating lift trajectory...'
        )

        self.get_logger().info(
            f'Current Z: {current_pose.position.z:.6f} m'
        )

        self.get_logger().info(
            f'Lifted Z: {lifted_pose.position.z:.6f} m'
        )

        return [lifted_pose]

    # =========================================
    # Create circle around lifted TCP position
    # =========================================

    def make_circle_waypoints(self, transform):

        p = transform.translation
        q = transform.rotation

        x0 = p.x
        y0 = p.y
        z0 = p.z

        # Circle starts at the current lifted TCP position.
        # The center is shifted to the left by the radius.

        center_x = x0 - self.radius
        center_y = y0
        center_z = z0

        waypoints = []

        for i in range(self.num_points):

            theta = (
                2.0 * math.pi * i /
                (self.num_points - 1)
            )

            pose = Pose()

            pose.position.x = (
                center_x + self.radius * math.cos(theta)
            )

            pose.position.y = (
                center_y + self.radius * math.sin(theta)
            )

            pose.position.z = center_z

            # Keep TCP orientation constant

            pose.orientation.x = q.x
            pose.orientation.y = q.y
            pose.orientation.z = q.z
            pose.orientation.w = q.w

            waypoints.append(pose)

        self.get_logger().info(
            f'Circle start: '
            f'x={x0:.6f}, y={y0:.6f}, z={z0:.6f}'
        )

        self.get_logger().info(
            f'Circle radius: {self.radius:.3f} m'
        )

        self.get_logger().info(
            f'Circle waypoints: {len(waypoints)}'
        )

        return waypoints

    # =========================================
    # Main sequence
    # =========================================

    def run(self):

        # -------------------------------------
        # Step 1: Get initial TCP position
        # -------------------------------------

        self.get_logger().info(
            'Waiting for initial TCP transform...'
        )

        initial_transform = self.get_current_tcp_pose()

        if initial_transform is None:

            return

        self.get_logger().info(
            'Initial TCP position received.'
        )

        # -------------------------------------
        # Step 2: Move arm upward
        # -------------------------------------

        lift_waypoints = self.make_lift_waypoints(
            initial_transform
        )

        lift_trajectory = self.compute_cartesian_path(
            lift_waypoints
        )

        if lift_trajectory is None:

            self.get_logger().error(
                'Lift path planning failed.'
            )

            return

        self.get_logger().info(
            'Lift trajectory planned successfully.'
        )

        if not self.execute_trajectory(lift_trajectory):

            self.get_logger().error(
                'Arm lifting was not completed.'
            )

            return

        # -------------------------------------
        # Step 3: Read new lifted TCP position
        # -------------------------------------

        self.get_logger().info(
            'Waiting for updated lifted TCP transform...'
        )

        lifted_transform = self.get_current_tcp_pose()

        if lifted_transform is None:

            return

        self.get_logger().info(
            f'Lifted TCP: '
            f'x={lifted_transform.translation.x:.6f}, '
            f'y={lifted_transform.translation.y:.6f}, '
            f'z={lifted_transform.translation.z:.6f}'
        )

        # -------------------------------------
        # Step 4: Generate circle at lifted pose
        # -------------------------------------

        circle_waypoints = self.make_circle_waypoints(
            lifted_transform
        )

        # -------------------------------------
        # Step 5: Plan circle
        # -------------------------------------

        circle_trajectory = self.compute_cartesian_path(
            circle_waypoints
        )

        if circle_trajectory is None:

            self.get_logger().error(
                'Circle path planning failed.'
            )

            return

        self.get_logger().info(
            'Complete circle trajectory planned.'
        )

        # -------------------------------------
        # Step 6: Execute circle
        # -------------------------------------

        self.execute_trajectory(circle_trajectory)

    # =========================================
    # End class


def main(args=None):

    rclpy.init(args=args)

    node = OpenArmXCircle()

    try:

        node.run()

    except KeyboardInterrupt:

        node.get_logger().warn(
            'Program interrupted.'
        )

    except Exception as e:

        node.get_logger().error(
            f'Unexpected error: {e}'
        )

    finally:

        node.destroy_node()

        rclpy.shutdown()


if __name__ == '__main__':

    main()
