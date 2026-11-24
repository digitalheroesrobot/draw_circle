# can setting
lsusb
ip -details link show

sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up

sudo ip link set can1 down
sudo ip link set can1 type can bitrate 1000000
sudo ip link set can1 up

ip -details link show can0
ip -details link show can1

# run script.py(motor_test)
clear
source /opt/ros/humble/setup.bash
source ~/openarmx_ws/install/setup.bash
/usr/bin/python3 ~/openarmx_ws/src/script.py

# run point_tracking.py(point tracking test)
source /opt/ros/humble/setup.bash
source ~/openarmx_ws/install/local_setup.bash

cd ~/openarmx_ws/src/openarmx_ros2/openarmx_bimanual_moveit_config

/usr/bin/python3 point_tracking.py


# In simulation Mode
source ~/openarmx_ws/src/openarmx_tools/install/setup.bash
# Terminal 1	
source /opt/ros/humble/setup.bash
source ~/openarmx_ws/install/setup.bash
source ~/openarmx_ws/src/openarmx_tools/install/setup.bash
cd ~/openarmx_ws/src/openarmx_ros2/openarmx_bimanual_moveit_config
unset GTK_PATH
bash run_bimanual_moveit_sim.sh

bash /home/ubuntu/openarmx_ws/src/openarmx_ros2/openarmx_bimanual_moveit_config/run_bimanual_moveit_with_can2.0.sh
# Terminal 2:
cd ~/openarmx_ws/src/robot_calibration/openarmx_circle

source /opt/ros/humble/setup.bash
source ~/openarmx_ws/install/setup.bash

python3 tracking.py --ros-args \
  -p mode:=real \
  -p arm:=left \
  -p control_frequency:=100.0 \
  -p ik_timeout:=0.008 \
  -p command_duration:=0.01 \
  -p velocity_scale:=0.1

python3 tracking.py --ros-args \
  -p mode:=sim \
  -p arm:=left \
  -p base_frame:=world \
  -p control_frequency:=100.0 \
  -p ik_timeout:=0.03 \
  -p target_timeout:=0.5 \
  -p command_duration:=0.06 \
  -p command_lead_time:=0.02 \
  -p velocity_scale:=0.1 \
  -p max_rate_dt:=0.15 \
  -p ik_stats_interval:=1 \
  -p enable_command_publishing:=false
  
# Terminal 3:

# trajectory_generator_csv.py
cd ~/openarmx_ws/src/robot_calibration/openarmx_circle

source /opt/ros/$ROS_DISTRO/setup.bash
source ~/openarmx_ws/install/setup.bash

python3 trajectory_generator_csv.py --ros-args \
  -p arm:=left \
  -p csv_file:=~/openarmx_ws/src/robot_calibration/openarmx_circle/data.csv \
  -p publish_frequency:=50.0 \
  -p transition_duration:=5.0
  
# trajectory_generator_make_csv  
cd ~/openarmx_ws/src/robot_calibration/openarmx_circle

source /opt/ros/$ROS_DISTRO/setup.bash
source ~/openarmx_ws/install/setup.bash
python3 trajectory_generator.py --ros-args \
  -p arm:=left \
  -p base_frame:=world \
  -p lift_distance:=0.1 \
  -p lift_duration:=5.0 \
  -p circle_radius:=0.05 \
  -p circle_z_amplitude:=0.001 \
  -p circle_frequency:=0.05 \
  -p roll_angle_deg:=10.0 \
  -p pitch_angle_deg:=10.0 \
  -p yaw_angle_deg:=10.0 \
  -p orientation_frequency:=0.01
  -p publish_frequency:=50.0

# Terminal 4:
  ros2 topic pub --once \
  /trajectory_start \
  std_msgs/msg/Bool \
  "{data: true}"

# In Real robot

# Terminal 1
source /opt/ros/humble/setup.bash
source ~/openarmx_ws/install/setup.bash
source ~/openarmx_ws/src/openarmx_tools/install/setup.bash
cd ~/openarmx_ws/src/openarmx_ros2/openarmx_bimanual_moveit_config
unset GTK_PATH
bash run_bimanual_moveit_with_can2.0.sh

# Terminal 2
python3 tracking.py --ros-args \
  -p mode:=real \
  -p arm:=left

# Terminal 3
python3 trajectory_generator.py --ros-args \
  -p arm:=left \
  -p base_frame:=world \
  -p lift_distance:=0.1 \
  -p lift_duration:=5.0 \
  -p circle_radius:=0.05 \
  -p circle_z_amplitude:=0.001 \
  -p circle_frequency:=0.05 \
  -p publish_frequency:=10.0
  
# Terminal 4
ros2 topic pub --once \
  /trajectory_start \
  std_msgs/msg/Bool \
  "{data: true}"
  
  
  
## BUILD Command
cd ~/openarmx_ws

source /opt/ros/humble/setup.bash

colcon build \
  --packages-select openarmx_bimanual_moveit_config \
  --symlink-install \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3

=================================================================
echo "========== RUNNING MOVE_GROUP PROCESS =========="
ps -eo pid,args | grep '[m]ove_group'

echo
echo "========== ACTIVE MOVE_GROUP PARAMETER FILE =========="
PID=$(pgrep -f '/moveit_ros_move_group/move_group' | head -n 1)

if [ -n "$PID" ]; then
    ARGS=$(ps -p "$PID" -o args=)
    echo "$ARGS"

    PARAM_FILE=$(echo "$ARGS" | grep -oE '/tmp/launch_params_[^ ]+' | head -n 1)

    if [ -n "$PARAM_FILE" ] && [ -f "$PARAM_FILE" ]; then
        echo
        echo "PARAM FILE: $PARAM_FILE"
        grep -n -B 5 -A 8 'kinematics_solver_timeout' "$PARAM_FILE"
    else
        echo "Active launch parameter file not found."
    fi
fi