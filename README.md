# PX150 Game

**CS468 Project**

This project implements an interactive color block picking game using the Interbotix PX150 robotic arm, an Intel RealSense depth camera, and a custom Pygame interface. The system performs real-time color detection using OpenCV, displays live camera output through Pygame, and controls the robotic arm based on user key takes.

---

## Project Description

The user places colored objects (blocks in this case) within the camera's field of view. The game selects a random target color. The user should the use the arm to move the color into the detection region.When the correct color appears inside the Region of Interest (ROI), the application:

- Confirms the color using HSV segmentation and depth validation
- Triggers an animation in the Pygame interface
- Commands the PX150 robotic arm to home position
- Increments the game score and assigns a new target color

The system includes manual keyboard control for the robot arm, as well as real-time calibration through tunable parameters.

---

## Hardware and Software Requirements

### Hardware

- Interbotix PX150 Manipulator
- Intel RealSense RGB-D camera (D415 or D435 recommended)
- Linux machine with USB 3.0
- Robot power supply and U2D2 communication module

### Software

- Ubuntu 20.04 or 22.04
- Python 3.8+
- RealSense SDK (librealsense2)
- Interbotix SDK (XS series)
- Pygame
- OpenCV
- NumPy

---

## Running the Application

### Terminal 1: Start the Robot Arm
```
cd /home/roboarm/interbotix_ws/src/px150_game
source /opt/ros/humble/setup.bash
source ~/interbotix_ws/install/setup.bash
./run_arm.sh
```

### Terminal 2: Launch the Game Interface
```
cd /home/roboarm/interbotix_ws/src/px150_game
source /opt/ros/humble/setup.bash
source ~/interbotix_ws/install/setup.bash
./run_game_pygame.sh
```
After launch, the following will appear in a single Pygame window:

Live RGB stream from RealSense
Depth-filtered ROI
Status display (target color, score, instructions)

## Controls

### Robot Joint Controls

| Key | Action |
|-----|--------|
| Q/A | Shoulder rotation ± |
| W/S | Shoulder pitch ± |
| E/D | Elbow ± |
| R/F | Wrist angle ± |
| T/G | Wrist rotation ± |

### Gripper Controls

| Key | Action |
|-----|--------|
| Z | Open gripper |
| X | Close gripper |

### Game Controls

| Key | Action |
|-----|--------|
| Space | Reset target color |
| Esc | Quit application |

---

## Key Tunable Parameters

Inside the script:

```python
JOINT_STEP = 0.12
MOVING_TIME = 0.12
ROI_PIXEL_MIN = 500
DEPTH_THRESHOLD_M = 0.7
SAMPLES_FOR_CONFIRM = 5
SAMPLES_MAJORITY = 3
```

These values govern robot responsiveness, detection stability, and game pacing.

---

## File Structure

```
project/
│
├── color_game.py        # Main Pygame + RealSense + PX150 control script
├── requirements.txt     
├── README.md            # Project documentation
This section needs to be updated
```

## Acknowledgments

- Interbotix Robotics for the PX150 SDK
- Intel RealSense for depth camera support
- AI tools (ChatGPT & Cursor) for code development assistance

  
