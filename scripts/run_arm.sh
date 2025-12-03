#!/bin/bash
# ROS2 Launch script for PX150 Arm Control
# This script launches the interbotix_xsarm_control node

# Source ROS setup if available
if [ -f /opt/ros/humble/setup.bash ]; then
    source /opt/ros/humble/setup.bash
fi

# Source Interbotix workspace if available
if [ -f ~/interbotix_ws/install/setup.bash ]; then
    source ~/interbotix_ws/install/setup.bash
fi

# Launch the robot arm control
ros2 launch interbotix_xsarm_control xsarm_control.launch.py robot_model:=px150


