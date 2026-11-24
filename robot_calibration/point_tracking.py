
# link to Rviz for point tracking test

#!/usr/bin/env python3
# OpenArmX V10 Bimanual MoveIt Dashboard
# ROS 2 Humble / Python 3.10
#
# Run MoveIt simulation OR real robot MoveIt first.
# Then:
#   conda deactivate
#   source /opt/ros/humble/setup.bash
#   source ~/openarmx_ws/install/local_setup.bash
#   /usr/bin/python3 script.py
#
# This file does not launch MoveIt or modify other files.
# PLAN only plans. EXECUTE moves the robot.

import math
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import xml.etree.ElementTree as ET

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
)

from std_msgs.msg import String
from sensor_msgs.msg import JointState
from control_msgs.msg import DynamicJointState
from geometry_msgs.msg import Pose, PoseStamped
from shape_msgs.msg import SolidPrimitive
from visualization_msgs.msg import InteractiveMarkerFeedback

from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    PositionConstraint,
    OrientationConstraint,
    BoundingVolume,
    RobotState,
    DisplayTrajectory,
)

from tf2_ros import Buffer, TransformListener


# ============================================================
# ROBOT CONFIGURATION
# ============================================================

WORLD_FRAME = "world"

LEFT_GROUP = "left_arm"
RIGHT_GROUP = "right_arm"

LEFT_GRIPPER_GROUP = "left_gripper"
RIGHT_GRIPPER_GROUP = "right_gripper"

LEFT_TCP = "openarmx_left_hand_tcp"
RIGHT_TCP = "openarmx_right_hand_tcp"

# RViz goal marker is associated with the hand parent link.
LEFT_GOAL_LINK = "openarmx_left_hand"
RIGHT_GOAL_LINK = "openarmx_right_hand"

LEFT_JOINTS = [
    f"openarmx_left_joint{i}" for i in range(1, 8)
]

RIGHT_JOINTS = [
    f"openarmx_right_joint{i}" for i in range(1, 8)
]

LEFT_FINGER = "openarmx_left_finger_joint1"
RIGHT_FINGER = "openarmx_right_finger_joint1"

ALL_JOINTS = (
    LEFT_JOINTS
    + RIGHT_JOINTS
    + [LEFT_FINGER, RIGHT_FINGER]
)

GROUPS = {
    "Left": (LEFT_GROUP, LEFT_GOAL_LINK),
    "Right": (RIGHT_GROUP, RIGHT_GOAL_LINK),
}

TCP_LINKS = {
    "Left": LEFT_TCP,
    "Right": RIGHT_TCP,
}

GRIPPER_GROUPS = {
    "Left": LEFT_GRIPPER_GROUP,
    "Right": RIGHT_GRIPPER_GROUP,
}

FINGERS = {
    "Left": LEFT_FINGER,
    "Right": RIGHT_FINGER,
}

MARKER_TOPICS = {
    "Left": (
        "/rviz/moveit/move_marker/goal_openarmx_left_hand"
    ),
    "Right": (
        "/rviz/moveit/move_marker/goal_openarmx_right_hand"
    ),
}

FEEDBACK_TOPIC = (
    "/rviz_moveit_motion_planning_display/"
    "robot_interaction_interactive_marker_topic/feedback"
)

ZERO_PRESET = {
    name: 0.0 for name in ALL_JOINTS
}

HAND_UP_PRESET = {
    name: 0.0 for name in ALL_JOINTS
}

HAND_UP_PRESET["openarmx_left_joint4"] = 0.2
HAND_UP_PRESET["openarmx_right_joint4"] = 0.2

PLANNING_TIME = 5.0
POSITION_TOLERANCE = 0.01
ORIENTATION_TOLERANCE = 0.08

VELOCITY_SCALE = 0.15
ACCELERATION_SCALE = 0.10


# ============================================================
# MATH
# ============================================================

def quaternion_from_rpy(roll, pitch, yaw):
    cr = math.cos(roll / 2)
    sr = math.sin(roll / 2)

    cp = math.cos(pitch / 2)
    sp = math.sin(pitch / 2)

    cy = math.cos(yaw / 2)
    sy = math.sin(yaw / 2)

    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    w = cr * cp * cy + sr * sp * sy

    return x, y, z, w


def quaternion_to_rpy(x, y, z, w):
    sinr = 2 * (w * x + y * z)
    cosr = 1 - 2 * (x * x + y * y)

    roll = math.atan2(sinr, cosr)

    sinp = 2 * (w * y - z * x)

    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    siny = 2 * (w * z + x * y)
    cosy = 1 - 2 * (y * y + z * z)

    yaw = math.atan2(siny, cosy)

    return roll, pitch, yaw


# ============================================================
# ROS NODE
# ============================================================

class OpenArmNode(Node):

    def __init__(self):
        super().__init__("openarmx_tk_dashboard")

        self.lock = threading.RLock()

        self.joints = {}
        self.dynamic_joints = {}
        self.limits = {}
        self.poses = {}

        self.marker_feedback = {}
        self.marker_versions = {
            "Left": 0,
            "Right": 0,
        }

        self.planned_trajectories = []

        # MoveIt actions
        self.move_client = ActionClient(
            self, MoveGroup, "/move_action"
        )

        self.execute_client = ActionClient(
            self,
            ExecuteTrajectory,
            "/execute_trajectory"
        )

        # IK service
        self.ik_client = self.create_client(
            GetPositionIK,
            "/compute_ik"
        )

        # Joint states
        self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_callback,
            20
        )

        self.create_subscription(
            DynamicJointState,
            "/dynamic_joint_states",
            self.dynamic_callback,
            20
        )

        # URDF subscription: transient-local QoS
        urdf_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.create_subscription(
            String,
            "/robot_description",
            self.robot_description_callback,
            urdf_qos
        )

        # RViz goal marker movement feedback
        self.create_subscription(
            InteractiveMarkerFeedback,
            FEEDBACK_TOPIC,
            self.marker_feedback_callback,
            20
        )

        # RViz goal marker publishers
        self.marker_publishers = {}

        for side, topic in MARKER_TOPICS.items():
            self.marker_publishers[side] = self.create_publisher(
                PoseStamped,
                topic,
                10
            )

        # RViz custom goal state publisher
        self.goal_state_publisher = self.create_publisher(
            RobotState,
            "/rviz/moveit/update_custom_goal_state",
            10
        )

        # RViz planning group selection publisher
        self.group_publisher = self.create_publisher(
            String,
            "/rviz/moveit/select_planning_group",
            10
        )

        # Publish planned trajectory for RViz visualization
        self.display_publisher = self.create_publisher(
            DisplayTrajectory,
            "/display_planned_path",
            10
        )

        # TF
        self.tf_buffer = Buffer()

        self.tf_listener = TransformListener(
            self.tf_buffer,
            self
        )

        self.create_timer(0.25, self.update_tf)

    # --------------------------------------------------------
    # JOINT STATES
    # --------------------------------------------------------

    def joint_callback(self, msg):
        with self.lock:
            for i, name in enumerate(msg.name):
                item = self.joints.setdefault(name, {})

                if i < len(msg.position):
                    item["position"] = msg.position[i]

                if i < len(msg.velocity):
                    item["velocity"] = msg.velocity[i]

                if i < len(msg.effort):
                    item["effort"] = msg.effort[i]

    def dynamic_callback(self, msg):
        with self.lock:
            for i, name in enumerate(msg.joint_names):
                if i >= len(msg.interface_values):
                    continue

                data = msg.interface_values[i]

                self.dynamic_joints[name] = dict(
                    zip(data.interface_names, data.values)
                )

    # --------------------------------------------------------
    # URDF LIMITS
    # --------------------------------------------------------

    def robot_description_callback(self, msg):
        try:
            root = ET.fromstring(msg.data)
            parsed = {}

            for joint in root.findall("joint"):
                name = joint.get("name")
                joint_type = joint.get("type", "")

                if not name:
                    continue

                limit = joint.find("limit")

                if joint_type == "continuous":
                    parsed[name] = {
                        "lower": None,
                        "upper": None,
                        "velocity": (
                            float(limit.get("velocity"))
                            if limit is not None
                            and limit.get("velocity") is not None
                            else None
                        ),
                        "effort": (
                            float(limit.get("effort"))
                            if limit is not None
                            and limit.get("effort") is not None
                            else None
                        ),
                        "continuous": True,
                    }
                    continue

                if limit is None:
                    continue

                def number(attribute):
                    value = limit.get(attribute)
                    return float(value) if value is not None else None

                parsed[name] = {
                    "lower": number("lower"),
                    "upper": number("upper"),
                    "velocity": number("velocity"),
                    "effort": number("effort"),
                    "continuous": False,
                }

            with self.lock:
                self.limits = parsed

            self.get_logger().info(
                f"Loaded URDF limits for {len(parsed)} joints"
            )

        except Exception as exc:
            self.get_logger().error(
                f"URDF parsing failed: {exc}"
            )

    # --------------------------------------------------------
    # RVIZ MARKER FEEDBACK
    # --------------------------------------------------------

    def marker_feedback_callback(self, msg):
        name = msg.marker_name.lower()

        if "left_hand" in name:
            side = "Left"
        elif "right_hand" in name:
            side = "Right"
        else:
            return

        p = msg.pose.position
        q = msg.pose.orientation

        rpy = quaternion_to_rpy(
            q.x, q.y, q.z, q.w
        )

        pose = (
            p.x, p.y, p.z,
            rpy[0], rpy[1], rpy[2]
        )

        with self.lock:
            self.marker_feedback[side] = pose
            self.marker_versions[side] += 1

    # --------------------------------------------------------
    # TF TCP POSITION
    # --------------------------------------------------------

    def update_tf(self):
        for side, link in [
            ("Left", LEFT_TCP),
            ("Right", RIGHT_TCP),
        ]:
            try:
                transform = self.tf_buffer.lookup_transform(
                    WORLD_FRAME,
                    link,
                    rclpy.time.Time()
                )

                p = transform.transform.translation
                q = transform.transform.rotation

                rpy = quaternion_to_rpy(
                    q.x, q.y, q.z, q.w
                )

                with self.lock:
                    self.poses[side] = (
                        p.x, p.y, p.z,
                        rpy[0], rpy[1], rpy[2]
                    )

            except Exception:
                pass

    # --------------------------------------------------------
    # FUTURE HELPER
    # --------------------------------------------------------

    @staticmethod
    def wait_future(future, timeout=60):
        event = threading.Event()
        result = {}

        def callback(f):
            try:
                result["value"] = f.result()
            except Exception as exc:
                result["error"] = exc

            event.set()

        future.add_done_callback(callback)

        if not event.wait(timeout):
            raise TimeoutError("ROS operation timeout")

        if "error" in result:
            raise result["error"]

        return result["value"]

    # --------------------------------------------------------
    # JOINT CONSTRAINTS
    # --------------------------------------------------------

    def joint_constraints(self, values):
        constraints = Constraints()

        for name, value in values.items():
            jc = JointConstraint()

            jc.joint_name = name
            jc.position = float(value)

            jc.tolerance_above = 0.005
            jc.tolerance_below = 0.005
            jc.weight = 1.0

            constraints.joint_constraints.append(jc)

        return constraints

    # --------------------------------------------------------
    # CARTESIAN CONSTRAINTS
    # --------------------------------------------------------

    def pose_constraints(
        self,
        link,
        x, y, z,
        roll, pitch, yaw
    ):
        constraints = Constraints()

        pc = PositionConstraint()

        pc.header.frame_id = WORLD_FRAME
        pc.link_name = link
        pc.weight = 1.0

        region = BoundingVolume()

        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [POSITION_TOLERANCE]

        center = Pose()
        center.position.x = x
        center.position.y = y
        center.position.z = z
        center.orientation.w = 1.0

        region.primitives.append(sphere)
        region.primitive_poses.append(center)

        pc.constraint_region = region
        constraints.position_constraints.append(pc)

        qx, qy, qz, qw = quaternion_from_rpy(
            roll, pitch, yaw
        )

        oc = OrientationConstraint()

        oc.header.frame_id = WORLD_FRAME
        oc.link_name = link

        oc.orientation.x = qx
        oc.orientation.y = qy
        oc.orientation.z = qz
        oc.orientation.w = qw

        oc.absolute_x_axis_tolerance = ORIENTATION_TOLERANCE
        oc.absolute_y_axis_tolerance = ORIENTATION_TOLERANCE
        oc.absolute_z_axis_tolerance = ORIENTATION_TOLERANCE

        oc.weight = 1.0

        constraints.orientation_constraints.append(oc)

        return constraints

    # --------------------------------------------------------
    # MOVEIT PLAN
    # --------------------------------------------------------

    def plan(self, group, constraints):
        if not self.move_client.wait_for_server(
            timeout_sec=5
        ):
            raise RuntimeError(
                "/move_action server unavailable"
            )

        goal = MoveGroup.Goal()
        request = goal.request

        request.group_name = group

        request.num_planning_attempts = 5
        request.allowed_planning_time = PLANNING_TIME

        request.max_velocity_scaling_factor = VELOCITY_SCALE
        request.max_acceleration_scaling_factor = ACCELERATION_SCALE

        request.goal_constraints = [constraints]
        request.start_state.is_diff = True

        goal.planning_options.plan_only = True
        goal.planning_options.replan = False
        goal.planning_options.look_around = False

        handle = self.wait_future(
            self.move_client.send_goal_async(goal),
            10
        )

        if not handle.accepted:
            raise RuntimeError("MoveIt rejected planning request")

        response = self.wait_future(
            handle.get_result_async(),
            60
        )

        result = response.result

        if result.error_code.val != 1:
            raise RuntimeError(
                f"MoveIt planning failed: "
                f"error code {result.error_code.val}"
            )

        trajectory = result.planned_trajectory

        if not trajectory.joint_trajectory.points:
            raise RuntimeError("Empty trajectory returned")

        return {
            "trajectory": trajectory,
            "trajectory_start": result.trajectory_start,
        }

    # --------------------------------------------------------
    # PUBLISH PLANS TO RVIZ
    # --------------------------------------------------------

    def publish_display_trajectory(self, plans):
        if not plans:
            return

        display = DisplayTrajectory()

        display.model_id = "openarmx"

        display.trajectory_start = (
            plans[0]["trajectory_start"]
        )

        for plan in plans:
            display.trajectory.append(
                plan["trajectory"]
            )

        self.display_publisher.publish(display)

        self.get_logger().info(
            f"Published {len(plans)} planned trajectory(s) to RViz"
        )

    # --------------------------------------------------------
    # EXECUTE
    # --------------------------------------------------------

    def execute(self, plan):
        if not self.execute_client.wait_for_server(
            timeout_sec=5
        ):
            raise RuntimeError(
                "/execute_trajectory server unavailable"
            )

        goal = ExecuteTrajectory.Goal()

        goal.trajectory = plan["trajectory"]
        goal.trajectory_start = plan["trajectory_start"]

        handle = self.wait_future(
            self.execute_client.send_goal_async(goal),
            10
        )

        if not handle.accepted:
            raise RuntimeError("Execution rejected")

        response = self.wait_future(
            handle.get_result_async(),
            120
        )

        if response.result.error_code.val != 1:
            raise RuntimeError(
                f"Execution failed: "
                f"{response.result.error_code.val}"
            )

    @staticmethod
    def final_positions(plan):
        jt = plan["trajectory"].joint_trajectory
        point = jt.points[-1]

        return list(zip(jt.joint_names, point.positions))

    # --------------------------------------------------------
    # CURRENT ROBOT STATE FOR IK
    # --------------------------------------------------------

    def current_robot_state(self):
        state = RobotState()

        with self.lock:
            joint_data = dict(self.joints)

        names = []
        positions = []

        for name in ALL_JOINTS:
            if name not in joint_data:
                continue

            if "position" not in joint_data[name]:
                continue

            names.append(name)
            positions.append(
                float(joint_data[name]["position"])
            )

        if not names:
            raise RuntimeError(
                "No current joint states available for IK"
            )

        state.joint_state.name = names
        state.joint_state.position = positions
        state.is_diff = False

        return state

    # --------------------------------------------------------
    # IK FOR GUI TARGET -> RVIZ GOAL STATE
    # --------------------------------------------------------

    def solve_ik(self, side, pose, seed_state):
        if not self.ik_client.wait_for_service(
            timeout_sec=5
        ):
            raise RuntimeError(
                "/compute_ik service unavailable"
            )

        group, link = GROUPS[side]

        x, y, z, roll, pitch, yaw = pose

        qx, qy, qz, qw = quaternion_from_rpy(
            roll, pitch, yaw
        )

        request = GetPositionIK.Request()

        ik = request.ik_request

        ik.group_name = group
        ik.ik_link_name = link

        ik.pose_stamped.header.frame_id = WORLD_FRAME
        ik.pose_stamped.header.stamp = self.get_clock().now().to_msg()

        ik.pose_stamped.pose.position.x = x
        ik.pose_stamped.pose.position.y = y
        ik.pose_stamped.pose.position.z = z

        ik.pose_stamped.pose.orientation.x = qx
        ik.pose_stamped.pose.orientation.y = qy
        ik.pose_stamped.pose.orientation.z = qz
        ik.pose_stamped.pose.orientation.w = qw

        ik.robot_state = seed_state
        ik.avoid_collisions = True
        ik.attempts = 5

        ik.timeout.sec = 1
        ik.timeout.nanosec = 0

        response = self.wait_future(
            self.ik_client.call_async(request),
            15
        )

        if response.error_code.val != 1:
            raise RuntimeError(
                f"{side} IK failed: "
                f"error code {response.error_code.val}"
            )

        return response.solution

    # --------------------------------------------------------
    # GUI TARGET -> RVIZ
    # --------------------------------------------------------

    def publish_gui_target(self, sides, pose):
        seed = self.current_robot_state()

        merged = {}

        for side in sides:
            solution = self.solve_ik(
                side,
                pose,
                seed
            )

            for name, value in zip(
                solution.joint_state.name,
                solution.joint_state.position
            ):
                merged[name] = value

            # Use the previous IK result as the next seed.
            for name, value in merged.items():
                if name not in seed.joint_state.name:
                    seed.joint_state.name.append(name)
                    seed.joint_state.position.append(value)
                else:
                    index = seed.joint_state.name.index(name)
                    seed.joint_state.position[index] = value

        # Publish full merged state.
        state = self.current_robot_state()

        state_values = dict(
            zip(
                state.joint_state.name,
                state.joint_state.position
            )
        )

        state_values.update(merged)

        state.joint_state.name = list(state_values.keys())
        state.joint_state.position = list(state_values.values())
        state.is_diff = False

        self.goal_state_publisher.publish(state)

        # Update the visible RViz interactive marker.
        x, y, z, roll, pitch, yaw = pose

        qx, qy, qz, qw = quaternion_from_rpy(
            roll, pitch, yaw
        )

        for side in sides:
            group_message = String()
            group_message.data = GROUPS[side][0]
            self.group_publisher.publish(group_message)

            marker = PoseStamped()

            marker.header.frame_id = WORLD_FRAME
            marker.header.stamp = self.get_clock().now().to_msg()

            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = z

            marker.pose.orientation.x = qx
            marker.pose.orientation.y = qy
            marker.pose.orientation.z = qz
            marker.pose.orientation.w = qw

            self.marker_publishers[side].publish(marker)

    # --------------------------------------------------------
    # CLOSE
    # --------------------------------------------------------

    def close(self):
        self.destroy_node()


# ============================================================
# TKINTER DASHBOARD
# ============================================================

class Dashboard:

    def __init__(self, root, node):
        self.root = root
        self.node = node

        self.root.title(
            "OpenArmX V10 Bimanual MoveIt Dashboard"
        )

        self.root.geometry("1400x900")

        self.side = tk.StringVar(value="Both")

        self.target_vars = {
            "X": tk.StringVar(value="0.30"),
            "Y": tk.StringVar(value="0.20"),
            "Z": tk.StringVar(value="0.90"),
            "R": tk.StringVar(value="0.0"),
            "P": tk.StringVar(value="0.0"),
            "Yaw": tk.StringVar(value="0.0"),
        }

        self.busy = False
        self.suppress_target_trace = False

        self.sync_after_id = None
        self.last_marker_versions = {
            "Left": 0,
            "Right": 0,
        }

        self.build_ui()

        for var in self.target_vars.values():
            var.trace_add(
                "write",
                self.target_changed
            )

        self.root.after(200, self.refresh)

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.close
        )

    # --------------------------------------------------------
    # UI
    # --------------------------------------------------------

    def build_ui(self):
        header = ttk.Frame(self.root, padding=10)
        header.pack(fill="x")

        ttk.Label(
            header,
            text="OpenArmX V10 | Motion Dashboard",
            font=("Arial", 17, "bold")
        ).pack(side="left")

        self.status = tk.StringVar(
            value="Connecting to MoveIt..."
        )

        ttk.Label(
            header,
            textvariable=self.status
        ).pack(side="right")

        main = ttk.Panedwindow(
            self.root,
            orient="horizontal"
        )

        main.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=5
        )

        left = ttk.Frame(main, padding=5)
        right = ttk.Frame(main, padding=5)

        main.add(left, weight=2)
        main.add(right, weight=3)

        # Current TCP pose
        pose_frame = ttk.LabelFrame(
            left,
            text="Current Gripper TCP Pose (world)",
            padding=8
        )

        pose_frame.pack(fill="x", pady=5)

        self.pose_labels = {}

        for side in ["Left", "Right"]:
            var = tk.StringVar(
                value="X -- Y -- Z -- R -- P -- Y --"
            )

            self.pose_labels[side] = var

            row = ttk.Frame(pose_frame)
            row.pack(fill="x", pady=3)

            ttk.Label(
                row,
                text=side,
                width=8
            ).pack(side="left")

            ttk.Label(
                row,
                textvariable=var
            ).pack(side="left")

        # Target controls
        target = ttk.LabelFrame(
            left,
            text="Target Position / Orientation",
            padding=8
        )

        target.pack(fill="x", pady=5)

        row = ttk.Frame(target)
        row.pack(fill="x")

        ttk.Label(row, text="Arm:").pack(side="left")

        for side in ["Left", "Right", "Both"]:
            ttk.Radiobutton(
                row,
                text=side,
                value=side,
                variable=self.side,
                command=self.side_changed
            ).pack(side="left", padx=5)

        fields = [
            ("X (m)", "X"),
            ("Y (m)", "Y"),
            ("Z (m)", "Z"),
            ("Roll (rad)", "R"),
            ("Pitch (rad)", "P"),
            ("Yaw (rad)", "Yaw"),
        ]

        grid = ttk.Frame(target)
        grid.pack(fill="x", pady=5)

        for i, (label, key) in enumerate(fields):
            r = i // 2
            c = (i % 2) * 2

            ttk.Label(
                grid,
                text=label
            ).grid(
                row=r,
                column=c,
                sticky="w",
                padx=3,
                pady=3
            )

            ttk.Entry(
                grid,
                textvariable=self.target_vars[key],
                width=12
            ).grid(
                row=r,
                column=c + 1,
                sticky="ew",
                padx=3,
                pady=3
            )

        buttons = ttk.Frame(target)
        buttons.pack(fill="x", pady=5)

        ttk.Button(
            buttons,
            text="PLAN",
            command=self.plan_cartesian
        ).pack(side="left", padx=3)

        self.execute_button = ttk.Button(
            buttons,
            text="EXECUTE",
            command=self.execute_plan,
            state="disabled"
        )

        self.execute_button.pack(side="left", padx=3)

        ttk.Button(
            buttons,
            text="HOME",
            command=self.home
        ).pack(side="left", padx=3)

        # Presets
        preset = ttk.LabelFrame(
            left,
            text="Joint Presets",
            padding=8
        )

        preset.pack(fill="x", pady=5)

        ttk.Button(
            preset,
            text="ZERO POSITION | All 16 joints = 0 rad",
            command=lambda: self.preset(
                ZERO_PRESET,
                "Zero Position"
            )
        ).pack(fill="x", pady=4)

        ttk.Button(
            preset,
            text="HAND UP | Joint4 = 0.2 rad, others = 0",
            command=lambda: self.preset(
                HAND_UP_PRESET,
                "Hand Up"
            )
        ).pack(fill="x", pady=4)

        ttk.Label(
            preset,
            text="Preset buttons PLAN only. Press EXECUTE to move."
        ).pack(anchor="w")

        # Planning result
        result_frame = ttk.LabelFrame(
            left,
            text="Planning Result",
            padding=8
        )

        result_frame.pack(fill="x", pady=5)

        self.signal = tk.StringVar(
            value="NOT PLANNED"
        )

        self.lamp = tk.Canvas(
            result_frame,
            width=30,
            height=30,
            highlightthickness=0
        )

        self.lamp.pack(side="left")

        self.lamp_circle = self.lamp.create_oval(
            4, 4, 26, 26,
            fill="gray"
        )

        ttk.Label(
            result_frame,
            textvariable=self.signal,
            font=("Arial", 11, "bold")
        ).pack(side="left", padx=5)

        # Motor monitor
        monitor = ttk.LabelFrame(
            right,
            text="16 Motor States / URDF Limits",
            padding=5
        )

        monitor.pack(
            fill="both",
            expand=True,
            pady=5
        )

        columns = (
            "joint",
            "position",
            "velocity",
            "effort",
            "lower",
            "upper",
            "max_velocity",
            "max_effort",
            "limit",
        )

        self.joint_tree = ttk.Treeview(
            monitor,
            columns=columns,
            show="headings",
            height=17
        )

        headings = {
            "joint": ("Joint", 230),
            "position": ("Actual rad", 80),
            "velocity": ("Velocity", 75),
            "effort": ("Effort", 70),
            "lower": ("Min", 60),
            "upper": ("Max", 60),
            "max_velocity": ("Vel Max", 70),
            "max_effort": ("Effort Max", 75),
            "limit": ("Limit", 70),
        }

        for col, (heading, width) in headings.items():
            self.joint_tree.heading(
                col,
                text=heading
            )

            self.joint_tree.column(
                col,
                width=width,
                anchor="center"
            )

        self.joint_tree.pack(
            fill="both",
            expand=True,
            side="left"
        )

        scroll = ttk.Scrollbar(
            monitor,
            orient="vertical",
            command=self.joint_tree.yview
        )

        scroll.pack(side="right", fill="y")

        self.joint_tree.configure(
            yscrollcommand=scroll.set
        )

        self.joint_tree.tag_configure(
            "violation",
            background="#ffcccc"
        )

        # Planned joint angles
        plan_frame = ttk.LabelFrame(
            right,
            text="Planned Joint Angles (rad)",
            padding=5
        )

        plan_frame.pack(
            fill="both",
            expand=True,
            pady=5
        )

        self.plan_tree = ttk.Treeview(
            plan_frame,
            columns=("joint", "goal"),
            show="headings",
            height=10
        )

        self.plan_tree.heading(
            "joint",
            text="Joint"
        )

        self.plan_tree.heading(
            "goal",
            text="Planned Goal (rad)"
        )

        self.plan_tree.column("joint", width=250)
        self.plan_tree.column(
            "goal",
            width=150,
            anchor="center"
        )

        self.plan_tree.pack(fill="both", expand=True)

    # --------------------------------------------------------
    # TARGET CHANGE / INVALIDATION
    # --------------------------------------------------------

    def target_changed(self, *_):
        if self.suppress_target_trace:
            return

        self.invalidate()

        if self.sync_after_id is not None:
            self.root.after_cancel(self.sync_after_id)

        self.sync_after_id = self.root.after(
            800,
            self.sync_gui_target_to_rviz
        )

    def side_changed(self):
        self.invalidate()

        if self.sync_after_id is not None:
            self.root.after_cancel(self.sync_after_id)

        self.sync_after_id = self.root.after(
            300,
            self.sync_gui_target_to_rviz
        )

    def read_pose(self):
        try:
            return [
                float(self.target_vars[k].get())
                for k in ["X", "Y", "Z", "R", "P", "Yaw"]
            ]
        except ValueError:
            raise ValueError(
                "XYZ and RPY must contain numeric values."
            )

    def selected_sides(self):
        if self.side.get() == "Both":
            return ["Left", "Right"]

        return [self.side.get()]

    def sync_gui_target_to_rviz(self):
        self.sync_after_id = None

        try:
            pose = self.read_pose()
        except ValueError:
            return

        sides = self.selected_sides()

        def work():
            self.node.publish_gui_target(
                sides,
                pose
            )
            return "GUI target synchronized to RViz"

        self.start_worker(
            "Updating RViz Goal State...",
            work,
            is_plan=False
        )

    # --------------------------------------------------------
    # RVIZ MARKER -> GUI
    # --------------------------------------------------------

    def apply_marker_feedback(self):
        with self.node.lock:
            feedback = dict(self.node.marker_feedback)
            versions = dict(self.node.marker_versions)

        for side in ["Left", "Right"]:
            if versions[side] == self.last_marker_versions[side]:
                continue

            self.last_marker_versions[side] = versions[side]

            if side not in feedback:
                continue

            pose = feedback[side]

            self.suppress_target_trace = True

            self.side.set(side)

            for key, value in zip(
                ["X", "Y", "Z", "R", "P", "Yaw"],
                pose
            ):
                self.target_vars[key].set(
                    f"{value:.6f}"
                )

            self.suppress_target_trace = False

            self.invalidate()

            self.status.set(
                f"RViz {side} Goal Marker -> GUI synchronized"
            )

    # --------------------------------------------------------
    # PLANNING DISPLAY
    # --------------------------------------------------------

    def set_lamp(self, color, text):
        self.lamp.itemconfigure(
            self.lamp_circle,
            fill=color
        )

        self.signal.set(text)

    def invalidate(self):
        self.node.planned_trajectories = []

        for item in self.plan_tree.get_children():
            self.plan_tree.delete(item)

        self.execute_button.configure(
            state="disabled"
        )

        self.set_lamp("gray", "NOT PLANNED")

    # --------------------------------------------------------
    # BACKGROUND WORKER
    # --------------------------------------------------------

    def start_worker(
        self,
        title,
        function,
        is_plan=False
    ):
        if self.busy:
            return

        self.busy = True

        if is_plan:
            self.execute_button.configure(
                state="disabled"
            )

        self.status.set(title)

        self.set_lamp("#e0a000", "PROCESSING")

        def worker():
            try:
                result = function()

                self.root.after(
                    0,
                    lambda: self.worker_success(
                        result,
                        title,
                        is_plan
                    )
                )

            except Exception as exc:
                self.root.after(
                    0,
                    lambda error=str(exc): self.worker_failure(error)
                )

        threading.Thread(
            target=worker,
            daemon=True
        ).start()

    def worker_success(self, result, title, is_plan):
        self.busy = False

        if is_plan:
            self.node.planned_trajectories = result

            self.show_plan(result)

            self.execute_button.configure(
                state="normal"
            )

            self.set_lamp("#20a34a", "PLAN SUCCESS")

            self.status.set(
                title + " — Press EXECUTE to move"
            )

        else:
            self.status.set(result)

    def worker_failure(self, error):
        self.busy = False

        self.node.planned_trajectories = []

        self.execute_button.configure(
            state="disabled"
        )

        self.set_lamp("#d62828", "ACTION FAILED")

        self.status.set(error)

        messagebox.showerror(
            "OpenArmX Error",
            error
        )

    # --------------------------------------------------------
    # CARTESIAN PLAN
    # --------------------------------------------------------

    def plan_cartesian(self):
        self.invalidate()

        try:
            pose = self.read_pose()
        except ValueError as exc:
            messagebox.showerror(
                "Target Error",
                str(exc)
            )
            return

        sides = self.selected_sides()

        def work():
            plans = []

            x, y, z, r, p, yaw = pose

            for side in sides:
                group, link = GROUPS[side]

                constraints = self.node.pose_constraints(
                    link,
                    x, y, z,
                    r, p, yaw
                )

                plans.append(
                    self.node.plan(
                        group,
                        constraints
                    )
                )

            # This is the important RViz visualization step.
            self.node.publish_display_trajectory(plans)

            return plans

        self.start_worker(
            "Planning Cartesian target...",
            work,
            is_plan=True
        )

    # --------------------------------------------------------
    # PRESET PLAN
    # --------------------------------------------------------

    def preset(self, values, name):
        self.invalidate()

        sides = self.selected_sides()

        def work():
            plans = []

            for side in sides:
                arm_joints = (
                    LEFT_JOINTS
                    if side == "Left"
                    else RIGHT_JOINTS
                )

                arm_values = {
                    joint: values[joint]
                    for joint in arm_joints
                }

                plans.append(
                    self.node.plan(
                        GROUPS[side][0],
                        self.node.joint_constraints(arm_values)
                    )
                )

                finger = FINGERS[side]

                plans.append(
                    self.node.plan(
                        GRIPPER_GROUPS[side],
                        self.node.joint_constraints({
                            finger: values[finger]
                        })
                    )
                )

            self.node.publish_display_trajectory(plans)

            return plans

        self.start_worker(
            "Planning " + name + "...",
            work,
            is_plan=True
        )

    # --------------------------------------------------------
    # HOME
    # --------------------------------------------------------

    def home(self):
        self.invalidate()

        sides = self.selected_sides()

        def work():
            plans = []

            for side in sides:
                joints = (
                    LEFT_JOINTS
                    if side == "Left"
                    else RIGHT_JOINTS
                )

                values = {
                    joint: 0.0 for joint in joints
                }

                plans.append(
                    self.node.plan(
                        GROUPS[side][0],
                        self.node.joint_constraints(values)
                    )
                )

            self.node.publish_display_trajectory(plans)

            return plans

        self.start_worker(
            "Planning HOME...",
            work,
            is_plan=True
        )

    # --------------------------------------------------------
    # DISPLAY PLANNED ANGLES
    # --------------------------------------------------------

    def show_plan(self, plans):
        for item in self.plan_tree.get_children():
            self.plan_tree.delete(item)

        shown = set()

        for plan in plans:
            for name, value in self.node.final_positions(plan):
                if name in shown:
                    continue

                shown.add(name)

                self.plan_tree.insert(
                    "",
                    "end",
                    values=(name, f"{value:.5f}")
                )

    # --------------------------------------------------------
    # EXECUTE
    # --------------------------------------------------------

    def execute_plan(self):
        plans = list(
            self.node.planned_trajectories
        )

        if not plans:
            messagebox.showwarning(
                "No Plan",
                "Please PLAN successfully before EXECUTE."
            )
            return

        self.node.planned_trajectories = []

        self.execute_button.configure(
            state="disabled"
        )

        def work():
            for plan in plans:
                self.node.execute(plan)

            return "Execution completed"

        self.start_worker(
            "Executing trajectory...",
            work,
            is_plan=False
        )

    # --------------------------------------------------------
    # REFRESH
    # --------------------------------------------------------

    def refresh(self):
        with self.node.lock:
            joints = {
                k: dict(v)
                for k, v in self.node.joints.items()
            }

            dynamic = {
                k: dict(v)
                for k, v in self.node.dynamic_joints.items()
            }

            poses = dict(self.node.poses)
            limits = dict(self.node.limits)

        # TCP display
        for side in ["Left", "Right"]:
            if side in poses:
                x, y, z, r, p, yaw = poses[side]

                self.pose_labels[side].set(
                    f"X {x:+.3f}  "
                    f"Y {y:+.3f}  "
                    f"Z {z:+.3f} m   "
                    f"R {r:+.2f}  "
                    f"P {p:+.2f}  "
                    f"Y {yaw:+.2f} rad"
                )

        # Motor states
        for name in ALL_JOINTS:
            js = joints.get(name, {})
            ds = dynamic.get(name, {})

            position = ds.get(
                "position",
                js.get("position")
            )

            velocity = ds.get(
                "velocity",
                js.get("velocity")
            )

            effort = ds.get(
                "effort",
                js.get("effort")
            )

            limit = limits.get(name, {})

            lower = limit.get("lower")
            upper = limit.get("upper")
            max_velocity = limit.get("velocity")
            max_effort = limit.get("effort")

            continuous = limit.get("continuous", False)

            violation = (
                position is not None
                and (
                    (lower is not None and position < lower - 1e-5)
                    or
                    (upper is not None and position > upper + 1e-5)
                )
            )

            def fmt(value):
                if value is None:
                    return "--"
                return f"{value:.4f}"

            if violation:
                limit_text = "OUT"
            elif continuous:
                limit_text = "CONT"
            elif lower is not None or upper is not None:
                limit_text = "OK"
            else:
                limit_text = "N/A"

            values = (
                name,
                fmt(position),
                fmt(velocity),
                fmt(effort),
                fmt(lower),
                fmt(upper),
                fmt(max_velocity),
                fmt(max_effort),
                limit_text,
            )

            if self.joint_tree.exists(name):
                self.joint_tree.item(
                    name,
                    values=values,
                    tags=("violation",) if violation else ()
                )
            else:
                self.joint_tree.insert(
                    "",
                    "end",
                    iid=name,
                    values=values,
                    tags=("violation",) if violation else ()
                )

        # Apply RViz marker changes to GUI fields.
        self.apply_marker_feedback()

        self.root.after(
            200,
            self.refresh
        )

    # --------------------------------------------------------
    # CLOSE
    # --------------------------------------------------------

    def close(self):
        self.root.destroy()


# ============================================================
# MAIN
# ============================================================

def main():
    rclpy.init()

    node = OpenArmNode()

    executor = MultiThreadedExecutor(
        num_threads=4
    )

    executor.add_node(node)

    spin_thread = threading.Thread(
        target=executor.spin,
        daemon=True
    )

    spin_thread.start()

    root = tk.Tk()

    Dashboard(root, node)

    try:
        root.mainloop()

    finally:
        executor.shutdown()

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

