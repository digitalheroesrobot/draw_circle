
#!/usr/bin/env python3

import math
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox

import rclpy
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import JointTrajectoryControllerState
from control_msgs.action import FollowJointTrajectory, GripperCommand


# ---------------- CONFIGURATION ----------------

LEFT_JOINTS = [
    f"openarmx_left_joint{i}" for i in range(1, 8)
]
RIGHT_JOINTS = [
    f"openarmx_right_joint{i}" for i in range(1, 8)
]

LEFT_GRIPPER = "openarmx_left_finger_joint1"
RIGHT_GRIPPER = "openarmx_right_finger_joint1"

ALL_JOINTS = LEFT_JOINTS + RIGHT_JOINTS + [
    LEFT_GRIPPER, RIGHT_GRIPPER
]

# Position limits from the generated bimanual V10 URDF.
LIMITS = {
    "openarmx_left_joint1": (-3.344396, 0.905604),
    "openarmx_left_joint2": (-3.2707963268, 0.1292036732),
    "openarmx_left_joint3": (-1.57, 1.57),
    "openarmx_left_joint4": (0.0, 1.8),
    "openarmx_left_joint5": (-1.5, 1.5),
    "openarmx_left_joint6": (-0.75, 0.75),
    "openarmx_left_joint7": (-1.4, 1.4),

    "openarmx_right_joint1": (-1.25, 3.0),
    "openarmx_right_joint2": (-0.1292036732, 3.2707963268),
    "openarmx_right_joint3": (-1.57, 1.57),
    "openarmx_right_joint4": (0.0, 1.8),
    "openarmx_right_joint5": (-1.5, 1.5),
    "openarmx_right_joint6": (-0.75, 0.75),
    "openarmx_right_joint7": (-1.4, 1.4),

    LEFT_GRIPPER: (0.0, 0.044),
    RIGHT_GRIPPER: (0.0, 0.044),
}

# Default trajectory duration. Increase this for slower movement.
DEFAULT_DURATION = 5.0

# Reject commands if joint feedback is older than this.
MAX_FEEDBACK_AGE = 0.5

REFRESH_MS = 200
GRIPPER_MAX_EFFORT = 14.0


# ---------------- ROS 2 NODE ----------------

class OpenArmNode:

    def __init__(self):
        self.node = rclpy.create_node("openarmx_motor_dashboard")

        self.lock = threading.Lock()

        self.actual = {}
        self.velocity = {}

        # Controller에서 받은 보간 위치
        self.arm_commands = {}

        # 마지막으로 전송한 목표 위치
        self.command_targets = {}

        # 마지막으로 전송한 그리퍼 목표 위치
        self.gripper_commands = {}

        self.last_feedback_time = None
        self.events = queue.Queue()

        # Joint feedback
        self.joint_sub = self.node.create_subscription(
            JointState,
            "/joint_states",
            self.joint_callback,
            20
        )

        # Arm controller command feedback
        self.left_state_sub = self.node.create_subscription(
            JointTrajectoryControllerState,
            "/left_joint_trajectory_controller/state",
            lambda msg: self.controller_callback(msg),
            10
        )

        self.right_state_sub = self.node.create_subscription(
            JointTrajectoryControllerState,
            "/right_joint_trajectory_controller/state",
            lambda msg: self.controller_callback(msg),
            10
        )

        # Arm trajectory actions
        self.left_arm_client = ActionClient(
            self.node,
            FollowJointTrajectory,
            "/left_joint_trajectory_controller/follow_joint_trajectory"
        )

        self.right_arm_client = ActionClient(
            self.node,
            FollowJointTrajectory,
            "/right_joint_trajectory_controller/follow_joint_trajectory"
        )

        # Gripper actions
        self.left_gripper_client = ActionClient(
            self.node,
            GripperCommand,
            "/left_gripper_controller/gripper_cmd"
        )

        self.right_gripper_client = ActionClient(
            self.node,
            GripperCommand,
            "/right_gripper_controller/gripper_cmd"
        )

        # ARM action 상태
        self.arm_busy = {
            "left": False,
            "right": False
        }

        # 현재 활성 Action handle
        self.active_arm_handles = {
            "left": None,
            "right": None
        }

        # Goal 전송 요청이 아직 응답되지 않았는지 여부
        self.arm_request_in_flight = {
            "left": False,
            "right": False
        }

        # 전송 대기 중인 최신 Goal
        # 빠르게 여러 번 SET하면 가장 마지막 요청으로 갱신
        self.pending_arm_goals = {
            "left": None,
            "right": None
        }

        # 오래된 Action callback 구분용
        self.arm_generation = {
            "left": 0,
            "right": 0
        }
        self.requested_arm_targets = {
            "left": {},
            "right": {}
        }

        self.accepted_arm_targets = {
            "left": {},
            "right": {}
        }

        self.arm_target_valid = {
            "left": False,
            "right": False
        }

        self.gripper_target_valid = {
            LEFT_GRIPPER: False,
            RIGHT_GRIPPER: False
        }

        self.gripper_generation = {
            LEFT_GRIPPER: 0,
            RIGHT_GRIPPER: 0
        }
        
    def joint_callback(self, msg):
        positions = dict(zip(msg.name, msg.position))
        velocities = dict(zip(msg.name, msg.velocity)) if msg.velocity else {}

        with self.lock:
            self.actual.update({
                name: float(value)
                for name, value in positions.items()
            })

            self.velocity.update({
                name: float(value)
                for name, value in velocities.items()
            })

            self.last_feedback_time = time.monotonic()

            # Initialize gripper command display from measured position.
            for name in (LEFT_GRIPPER, RIGHT_GRIPPER):
                if name in positions and name not in self.gripper_commands:
                    self.gripper_commands[name] = float(positions[name])

    def controller_callback(self, msg):
        desired = dict(zip(msg.joint_names, msg.desired.positions))

        with self.lock:
            self.arm_commands.update({
                name: float(value)
                for name, value in desired.items()
            })

    def snapshot(self):
        with self.lock:
            return (
                dict(self.actual),
                dict(self.velocity),
                dict(self.arm_commands),
                dict(self.command_targets),
                dict(self.gripper_commands),
                self.last_feedback_time
            )
    # ---------- ARM COMMAND ----------


    def send_arm(self, side, joints, targets, duration):

        targets = {
            name: float(targets[name])
            for name in joints
        }

        with self.lock:

            self.arm_generation[side] += 1
            generation = self.arm_generation[side]

            # 가장 최근 요청 목표를 보관
            self.requested_arm_targets[side] = dict(targets)

            # 전송 대기 Goal은 최신 요청으로 교체
            self.pending_arm_goals[side] = (
                list(joints),
                dict(targets),
                float(duration),
                generation
            )

            self.arm_busy[side] = True

        # 이미 Goal 요청이 전송 중이면 응답 후 최신 Goal 전송
        self._dispatch_pending_arm_goal(side)


    def _dispatch_pending_arm_goal(self, side):

        with self.lock:

            if self.arm_request_in_flight[side]:
                return

            pending = self.pending_arm_goals[side]

            if pending is None:
                return

            self.pending_arm_goals[side] = None
            self.arm_request_in_flight[side] = True

        joints, targets, duration, generation = pending

        client = (
            self.left_arm_client if side == "left"
            else self.right_arm_client
        )

        if not client.server_is_ready():

            self.events.put(
                f"ERROR: {side} arm action server is not ready."
            )

            with self.lock:
                self.arm_request_in_flight[side] = False

                if (
                    self.arm_generation[side] == generation
                    and self.pending_arm_goals[side] is None
                ):
                    self.requested_arm_targets[side] = dict(
                        self.accepted_arm_targets[side]
                    )
                    self.arm_busy[side] = False

            # 그 사이 새 명령이 들어왔으면 최신 요청 처리
            self._dispatch_pending_arm_goal(side)
            return

        trajectory = JointTrajectory()
        trajectory.joint_names = list(joints)

        point = JointTrajectoryPoint()
        point.positions = [
            targets[name] for name in joints
        ]

        point.time_from_start.sec = int(duration)
        point.time_from_start.nanosec = int(
            (duration - int(duration)) * 1_000_000_000
        )

        trajectory.points = [point]

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory

        try:

            future = client.send_goal_async(goal)

            future.add_done_callback(
                lambda f, arm=side, gen=generation,
                    t=dict(targets):
                    self.arm_goal_response(f, arm, gen, t)
            )

        except Exception as exc:

            self.events.put(
                f"ERROR: sending {side} arm goal: {exc}"
            )

            with self.lock:
                self.arm_request_in_flight[side] = False

                if (
                    self.arm_generation[side] == generation
                    and self.pending_arm_goals[side] is None
                ):
                    self.requested_arm_targets[side] = dict(
                        self.accepted_arm_targets[side]
                    )
                    self.arm_busy[side] = False

            self._dispatch_pending_arm_goal(side)


    def arm_goal_response(self, future, side, generation, targets):

        accepted = False

        try:
            handle = future.result()
            accepted = handle.accepted

            if not accepted:
                self.events.put(
                    f"ERROR: {side} arm goal rejected."
                )

            else:
                with self.lock:
                    # 승인된 목표값은 세대 번호와 무관하게 기록
                    self.arm_target_valid[side] = True
                    self.accepted_arm_targets[side] = dict(targets)
                    self.command_targets.update(targets)

                    if self.arm_generation[side] == generation:
                        self.active_arm_handles[side] = handle

                self.events.put(
                    f"{side.capitalize()} arm goal accepted."
                )

                result_future = handle.get_result_async()

                result_future.add_done_callback(
                    lambda f, arm=side, gen=generation:
                        self.arm_result(f, arm, gen)
                )

        except Exception as exc:
            self.events.put(
                f"ERROR: {side} arm goal response: {exc}"
            )

        finally:
            with self.lock:
                self.arm_request_in_flight[side] = False

                # 최신 요청이 실패했고 대기 Goal도 없다면
                # 마지막 승인 목표로 요청값 복구
                if (
                    self.arm_generation[side] == generation
                    and self.pending_arm_goals[side] is None
                    and not accepted
                ):
                    self.requested_arm_targets[side] = dict(
                        self.accepted_arm_targets[side]
                    )
                    self.arm_busy[side] = False

            self._dispatch_pending_arm_goal(side)

    def arm_result(self, future, side, generation):

        success = False

        try:
            wrapped = future.result()
            result = wrapped.result

            if (
                result.error_code
                == FollowJointTrajectory.Result.SUCCESSFUL
            ):
                success = True

                self.events.put(
                    f"{side.capitalize()} arm movement completed."
                )

            else:
                self.events.put(
                    f"WARNING: {side} arm result: "
                    f"error_code={result.error_code}, "
                    f"{result.error_string}"
                )

        except Exception as exc:
            self.events.put(
                f"ERROR: {side} arm result: {exc}"
            )

        finally:
            with self.lock:

                if self.arm_generation[side] == generation:

                    self.active_arm_handles[side] = None

                    if not success:
                        self.arm_target_valid[side] = False

                        self.requested_arm_targets[side] = {}

                        for name in (
                            LEFT_JOINTS if side == "left"
                            else RIGHT_JOINTS
                        ):
                            self.command_targets.pop(name, None)

                    if (
                        self.pending_arm_goals[side] is None
                        and not self.arm_request_in_flight[side]
                    ):
                        self.arm_busy[side] = False

    
    # ---------- GRIPPER COMMAND ----------

    def send_gripper(self, side, joint_name, position):
        with self.lock:
            self.gripper_generation[joint_name] += 1
            generation = self.gripper_generation[joint_name]

        client = (
            self.left_gripper_client if side == "left"
            else self.right_gripper_client
        )

        if not client.server_is_ready():
            self.events.put(
                f"ERROR: {side} gripper action server is not ready."
            )
            return

        goal = GripperCommand.Goal()
        goal.command.position = float(position)
        goal.command.max_effort = GRIPPER_MAX_EFFORT

        try:
            future = client.send_goal_async(goal)

            future.add_done_callback(
                lambda f, arm=side, joint=joint_name,
                    pos=float(position), gen=generation:
                    self.gripper_goal_response(f, arm, joint, pos, gen)
            )

            self.events.put(
                f"{side.capitalize()} gripper goal submitted."
            )

        except Exception as exc:
            self.events.put(
                f"ERROR: sending {side} gripper goal: {exc}"
            )

    def gripper_goal_response(
        self, future, side, joint_name, position, generation
    ):
        try:
            handle = future.result()

            if not handle.accepted:
                self.events.put(
                    f"ERROR: {side} gripper goal rejected."
                )
                return

            with self.lock:
                self.gripper_commands[joint_name] = position
                self.gripper_target_valid[joint_name] = True

            self.events.put(
                f"{side.capitalize()} gripper goal accepted."
            )

            result_future = handle.get_result_async()

            result_future.add_done_callback(
                lambda f, arm=side, joint=joint_name:
                self.gripper_result(f, arm, joint)
            )

        except Exception as exc:
            self.events.put(
                f"ERROR: {side} gripper action: {exc}"
            )

    def gripper_result(self, future, side, joint_name):
        try:
            wrapped = future.result()
            result = wrapped.result

            if not result.reached_goal:
                with self.lock:
                    self.gripper_target_valid[joint_name] = False

            self.events.put(
                f"{side.capitalize()} gripper: "
                f"position={result.position:.5f}, "
                f"reached_goal={result.reached_goal}, "
                f"stalled={result.stalled}"
            )

        except Exception as exc:
            with self.lock:
                self.gripper_target_valid[joint_name] = False

            self.events.put(f"ERROR: gripper result: {exc}")


# ---------------- TKINTER DASHBOARD ----------------

class Dashboard:

    def __init__(self, root, ros):
        self.root = root
        self.ros = ros

        self.root.title("OpenArmX V10 Motor Dashboard")
        self.root.geometry("1000x760")

        self.actual_vars = {}
        self.command_vars = {}
        self.target_vars = {}

        self.initialized_targets = False

        self.build_interface()

        self.root.after(REFRESH_MS, self.refresh)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def build_interface(self):
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill="both", expand=True)

        ttk.Label(
            main,
            text="OpenArmX V10 — Live Motor Dashboard",
            font=("Arial", 16, "bold")
        ).pack(anchor="w", pady=(0, 5))

        ttk.Label(
            main,
            text=(
                "All positions are in radians. Actual = measured position. "
                "Command = arm controller desired position; "
                "gripper = last accepted target. "
                "Target = editable destination."
            ),
            wraplength=950
        ).pack(anchor="w", pady=(0, 10))

        table = ttk.Frame(main)
        table.pack(fill="both", expand=True)

        headers = [
            ("Joint", 34),
            ("Actual (rad)", 16),
            ("Current command (rad)", 23),
            ("Target (rad)", 16)
        ]

        header = ttk.Frame(table)
        header.pack(fill="x")

        for text, width in headers:
            ttk.Label(
                header,
                text=text,
                width=width,
                font=("Arial", 10, "bold")
            ).pack(side="left", padx=3)

        # Scrollable joint list
        canvas = tk.Canvas(table, highlightthickness=0)
        scrollbar = ttk.Scrollbar(
            table, orient="vertical", command=canvas.yview
        )

        self.rows = ttk.Frame(canvas)

        self.rows.bind(
            "<Configure>",
            lambda event: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas_window = canvas.create_window(
            (0, 0), window=self.rows, anchor="nw"
        )

        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.bind(
            "<Configure>",
            lambda event: canvas.itemconfigure(
                canvas_window, width=event.width
            )
        )

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.add_group("LEFT ARM", LEFT_JOINTS)
        self.add_group("RIGHT ARM", RIGHT_JOINTS)
        self.add_group("GRIPPERS", [LEFT_GRIPPER, RIGHT_GRIPPER])

        # Trajectory duration
        duration_row = ttk.Frame(main)
        duration_row.pack(fill="x", pady=(10, 5))

        ttk.Label(
            duration_row, text="Movement duration (seconds):"
        ).pack(side="left")

        self.duration_var = tk.StringVar(
            value=str(DEFAULT_DURATION)
        )

        ttk.Entry(
            duration_row,
            textvariable=self.duration_var,
            width=8
        ).pack(side="left", padx=6)

        # Buttons
        buttons = ttk.Frame(main)
        buttons.pack(fill="x", pady=5)

        ttk.Button(
            buttons,
            text="SET",
            command=self.set_positions
        ).pack(side="left", padx=(0, 8))

        ttk.Button(
            buttons,
            text="ZERO POSITION",
            command=self.zero_position
        ).pack(side="left", padx=8)

        ttk.Button(
            buttons,
            text="HANDS UP",
            command=self.hands_up
        ).pack(side="left", padx=8)
        
        ttk.Button(
            buttons,
            text="Copy actual → targets",
            command=self.copy_actual
        ).pack(side="left", padx=8)

        # Status and messages
        self.status_var = tk.StringVar(
            value="Waiting for joint states..."
        )

        ttk.Label(
            main,
            textvariable=self.status_var,
            foreground="blue",
            wraplength=950
        ).pack(anchor="w", pady=(8, 2))

        self.event_var = tk.StringVar(value="Ready.")

        ttk.Label(
            main,
            textvariable=self.event_var,
            wraplength=950
        ).pack(anchor="w")

        ttk.Label(
            main,
            text=(
                "SAFETY: Check the robot workspace before moving. "
                "The zero button commands all arm joints to 0 rad and "
                "both grippers to 0 rad (closed). Keep the physical "
                "emergency stop accessible. URDF limits do not guarantee "
                "collision-free motion."
            ),
            foreground="#9b3500",
            wraplength=950
        ).pack(anchor="w", pady=(8, 0))

    def add_group(self, title, joints):
        ttk.Label(
            self.rows,
            text=title,
            font=("Arial", 10, "bold")
        ).pack(anchor="w", pady=(9, 3))

        for name in joints:
            row = ttk.Frame(self.rows)
            row.pack(fill="x", pady=1)

            short_name = name.replace("openarmx_", "")

            ttk.Label(
                row, text=short_name, width=34
            ).pack(side="left", padx=3)

            actual = tk.StringVar(value="—")
            command = tk.StringVar(value="—")
            target = tk.StringVar(value="0.00000")

            ttk.Label(
                row, textvariable=actual, width=16
            ).pack(side="left", padx=3)

            ttk.Label(
                row, textvariable=command, width=23
            ).pack(side="left", padx=3)

            ttk.Entry(
                row, textvariable=target, width=16
            ).pack(side="left", padx=3)

            self.actual_vars[name] = actual
            self.command_vars[name] = command
            self.target_vars[name] = target

    # ---------- LIVE REFRESH ----------

    def refresh(self):
        (
            actual,
            velocity,
            arm_commands,
            command_targets,
            gripper_commands,
            last_time
        ) = self.ros.snapshot()

        now = time.monotonic()

        if last_time is None:
            self.status_var.set("Waiting for /joint_states...")
        else:
            age = now - last_time
            missing = [n for n in ALL_JOINTS if n not in actual]

            if age > MAX_FEEDBACK_AGE:
                self.status_var.set(
                    f"WARNING: feedback is stale ({age:.2f} seconds). "
                    "Commands are disabled."
                )
            elif missing:
                self.status_var.set(
                    "Missing feedback: " + ", ".join(missing)
                )
            else:
                self.status_var.set(
                    f"Feedback OK — {len(ALL_JOINTS)} joints. "
                    f"Last update {age:.2f} seconds ago."
                )

        # Initialize target fields from actual positions on startup.
        if (
            not self.initialized_targets
            and all(name in actual for name in ALL_JOINTS)
        ):
            for name in ALL_JOINTS:
                self.target_vars[name].set(
                    f"{actual[name]:.5f}"
                )
            self.initialized_targets = True

        for name in ALL_JOINTS:
            if name in actual:
                self.actual_vars[name].set(
                    f"{actual[name]: .5f}"
                )
            else:
                self.actual_vars[name].set("—")

            # Command 표시
            if name in (LEFT_GRIPPER, RIGHT_GRIPPER):

                # 마지막으로 전송한 그리퍼 목표값
                command = gripper_commands.get(name, actual.get(name))

            else:

                # 마지막으로 전송한 팔의 목표값
                # 아직 명령하지 않은 경우 실제 위치를 표시
                command = arm_commands.get(
                    name,
                    actual.get(name)
                )

            self.command_vars[name].set(
                f"{command: .5f}" if command is not None else "—"
            )

        # Display ROS action feedback in the GUI thread.
        while True:
            try:
                event = self.ros.events.get_nowait()
                self.event_var.set(event)
            except queue.Empty:
                break

        self.root.after(REFRESH_MS, self.refresh)

    # ---------- VALIDATION ----------

    def get_fresh_actual(self):
        actual, _, _, _, _, last_time = self.ros.snapshot()

        if (
            last_time is None
            or time.monotonic() - last_time > MAX_FEEDBACK_AGE
        ):
            messagebox.showerror(
                "Feedback error",
                "Joint feedback is missing or stale. No command sent."
            )
            return None

        missing = [name for name in ALL_JOINTS if name not in actual]

        if missing:
            messagebox.showerror(
                "Missing joint feedback",
                "\n".join(missing)
            )
            return None

        return actual

    def read_targets(self):
        targets = {}
        errors = []

        for name in ALL_JOINTS:
            try:
                value = float(self.target_vars[name].get())

                if not math.isfinite(value):
                    raise ValueError()

                low, high = LIMITS[name]

                if value < low or value > high:
                    errors.append(
                        f"{name}: {value:.5f} outside "
                        f"[{low:.5f}, {high:.5f}]"
                    )

                targets[name] = value

            except ValueError:
                errors.append(f"{name}: invalid number")

        if errors:
            messagebox.showerror(
                "Invalid target values",
                "\n".join(errors)
            )
            return None

        try:
            duration = float(self.duration_var.get())

            if not math.isfinite(duration) or duration < 1.0:
                raise ValueError()

        except ValueError:
            messagebox.showerror(
                "Invalid duration",
                "Enter a movement duration of at least 1 second."
            )
            return None

        return targets, duration

    # ---------- SEND COMMANDS ----------

    
    def send_targets(self, targets, actual, duration):

        # 마지막으로 요청한 팔 목표값과 그리퍼 명령값 가져오기
        with self.ros.lock:

            left_previous = dict(
                self.ros.requested_arm_targets["left"]
            )

            right_previous = dict(
                self.ros.requested_arm_targets["right"]
            )

            command_targets = dict(self.ros.command_targets)
            gripper_commands = dict(self.ros.gripper_commands)
            left_valid = self.ros.arm_target_valid["left"]
            right_valid = self.ros.arm_target_valid["right"]
            left_gripper_valid = self.ros.gripper_target_valid[LEFT_GRIPPER]
            right_gripper_valid = self.ros.gripper_target_valid[RIGHT_GRIPPER]

        # LEFT ARM
        left_targets = {}
        left_changed = False

        for name in LEFT_JOINTS:

            previous = left_previous.get(
                name,
                command_targets.get(name, actual[name])
            )

            if (
                not left_valid
                or abs(targets[name] - previous) > 0.001
            ):
                left_changed = True

            left_targets[name] = targets[name]

        if left_changed:

            self.ros.send_arm(
                "left",
                LEFT_JOINTS,
                left_targets,
                duration
            )

        # RIGHT ARM
        right_targets = {}
        right_changed = False

        for name in RIGHT_JOINTS:

            previous = right_previous.get(
                name,
                command_targets.get(name, actual[name])
            )

            if (
                not right_valid
                or abs(targets[name] - previous) > 0.001
            ):
                right_changed = True

            right_targets[name] = targets[name]

        if right_changed:

            self.ros.send_arm(
                "right",
                RIGHT_JOINTS,
                right_targets,
                duration
            )

        # LEFT GRIPPER
        previous = gripper_commands.get(
            LEFT_GRIPPER,
            actual[LEFT_GRIPPER]
        )

        if (
            not left_gripper_valid
            or abs(targets[LEFT_GRIPPER] - previous) > 0.001
        ):
            self.ros.send_gripper(
                "left",
                LEFT_GRIPPER,
                targets[LEFT_GRIPPER]
            )

        # RIGHT GRIPPER
        previous = gripper_commands.get(
            RIGHT_GRIPPER,
            actual[RIGHT_GRIPPER]
        )

        if (
            not right_gripper_valid
            or abs(targets[RIGHT_GRIPPER] - previous) > 0.001
        ):
            self.ros.send_gripper(
                "right",
                RIGHT_GRIPPER,
                targets[RIGHT_GRIPPER]
            )


    def set_positions(self):
        actual = self.get_fresh_actual()

        if actual is None:
            return

        result = self.read_targets()

        if result is None:
            return

        targets, duration = result

        if not messagebox.askyesno(
            "Confirm SET",
            "Send all target positions shown in the table?\n\n"
            "An arm receives a complete 7-joint trajectory if any "
            "target differs from its last requested target. "
            "Grippers move only when their target differs."
        ):
            return

        self.send_targets(targets, actual, duration)

    def zero_position(self):
        actual = self.get_fresh_actual()

        if actual is None:
            return

        try:
            duration = float(self.duration_var.get())

            if not math.isfinite(duration) or duration < 1.0:
                raise ValueError()

        except ValueError:
            messagebox.showerror(
                "Invalid duration",
                "Enter a movement duration of at least 1 second."
            )
            return

        if not messagebox.askyesno(
            "Confirm ZERO POSITION",
            "Move all 14 arm joints to 0 radians and both grippers "
            "to 0 radians (closed)?\n\n"
            "Check the robot workspace before continuing."
        ):
            return

        targets = {name: 0.0 for name in ALL_JOINTS}

        for name in ALL_JOINTS:
            self.target_vars[name].set("0.00000")

        self.send_targets(targets, actual, duration)


    def hands_up(self):
        actual = self.get_fresh_actual()

        if actual is None:
            return

        try:
            duration = float(self.duration_var.get())

            if not math.isfinite(duration) or duration < 1.0:
                raise ValueError()

        except ValueError:
            messagebox.showerror(
                "Invalid duration",
                "Enter a movement duration of at least 1 second."
            )
            return

        if not messagebox.askyesno(
            "Confirm HANDS UP",
            "Move both arms to HANDS UP position?\n\n"
            "Left joint 4  = 0.2 rad\n"
            "Right joint 4 = 0.2 rad\n"
            "All other arm joints = 0 rad\n"
            "Both grippers = 0 rad (closed)"
        ):
            return

        targets = {name: 0.0 for name in ALL_JOINTS}

        targets["openarmx_left_joint4"] = 0.2
        targets["openarmx_right_joint4"] = 0.2

        for name in ALL_JOINTS:
            self.target_vars[name].set(f"{targets[name]:.5f}")

        self.send_targets(targets, actual, duration)


    def copy_actual(self):
        actual = self.get_fresh_actual()

        if actual is None:
            return

        for name in ALL_JOINTS:
            self.target_vars[name].set(
                f"{actual[name]:.5f}"
            )

        self.event_var.set(
            "Measured positions copied into target fields."
        )

    def close(self):
        if messagebox.askokcancel(
            "Exit",
            "Close the dashboard? This is not an emergency stop."
        ):
            self.root.destroy()


# ---------------- MAIN ----------------

def main():
    rclpy.init()

    ros = OpenArmNode()

    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(ros.node)

    spin_thread = threading.Thread(
        target=executor.spin,
        daemon=True
    )
    spin_thread.start()

    root = tk.Tk()
    Dashboard(root, ros)

    try:
        root.mainloop()
    finally:
        executor.shutdown()
        ros.node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()

# motor test program pid gain correct