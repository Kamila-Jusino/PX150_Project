# PX150 Game

**CS468 Project**

This project implements an interactive object-picking game using the Interbotix PX150 robotic arm, an Intel RealSense depth camera, and a custom Pygame interface. The system performs real-time object detection using YOLOv8 nano (AI-driven), displays live camera output through Pygame, and controls the robotic arm based on user keyboard inputs.

---

## Project Description

The user places colored objects within the camera's field of view. The game selects a random target color. The user uses the arm to move the target object into the detection region. When the correct object appears inside the Region of Interest (ROI), the application:

- Detects the object using YOLOv8 nano object detection model (AI-driven)
- Classifies the color using YOLOv8 nano classification model (with HSV fallback for display)
- Validates depth and object type
- Triggers an animation in the Pygame interface
- Commands the PX150 robotic arm to home position
- Increments the game score and assigns a new target object

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
- PyTorch (for YOLO models)
- Ultralytics (YOLO library)
- pyrealsense2

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

In `ai_vision.py`:

```python
DEPTH_THRESHOLD_M = 0.7        # Only detect objects within this depth (meters)
MIN_CONFIDENCE = 0.2            # Minimum confidence for object detection
YOLO_CONF_THRESHOLD = 0.1       # YOLO confidence threshold (very low for all objects)
YOLO_IMAGE_SIZE = 640          # YOLO input image size (optimized for speed)
YOLO_IOU_THRESHOLD = 0.5       # IoU threshold for non-maximum suppression
YOLO_MAX_DETECTIONS = 500      # Maximum detections per image
```

These values govern AI detection sensitivity, depth filtering, and performance.

---

## File Structure

```
project/
├── ai_vision.py                    # AI Vision System (YOLOv8 nano object/color detection)
├── ai_game_state.py               # Game state management and AI integration
├── color_picking_game_pygame.py   # Main Pygame + RealSense + PX150 control script
├── test_ai_system.py              # AI system testing
├── test_dependencies.py           # Dependency verification
├── yolov8n.pt                     # YOLOv8 nano object detection model
├── yolov8n-cls.pt                 # YOLOv8 nano color classification model
├── yolo_colors.pt                 # Fine-tuned YOLOv8 color classifier (optional)
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
└── run_game_pygame.sh            # Launch script
```

## Acknowledgments

- Interbotix Robotics for the PX150 SDK
- Intel RealSense for depth camera support
- AI tools (ChatGPT & Cursor) for code development assistance

  
