#!/bin/bash
# ROS2 Launch script for PX150 Arm Control
# This script launches the interbotix_xsarm_control node

source /opt/ros/humble/setup.bash
source ~/interbotix_ws/install/setup.bash

ros2 launch interbotix_xsarm_control xsarm_control.launch.py robot_model:=px150


