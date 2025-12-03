#!/bin/bash
# Python Game Script Runner - Pygame Version
# This script runs the color picking game with Pygame UI

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

# Source ROS setup if available
if [ -f /opt/ros/humble/setup.bash ]; then
    source /opt/ros/humble/setup.bash
fi

# Source Interbotix workspace if available
if [ -f ~/interbotix_ws/install/setup.bash ]; then
    source ~/interbotix_ws/install/setup.bash
fi

# Change to project directory
cd "$PROJECT_DIR"

# Run the game
python3 src/color_picking_game_pygame.py


