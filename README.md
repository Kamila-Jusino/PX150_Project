# PX150 Game

**CS468 Project**

This project implements an interactive object-picking game using the Interbotix PX150 robotic arm, an Intel RealSense depth camera, and a custom Pygame interface. The system performs real-time object detection using YOLOv8 nano (AI-driven), displays live camera output through Pygame, and controls the robotic arm based on user keyboard inputs.

---

## Project Description

The user places colored objects within the camera's field of view. The game uses AI to scan the scene and selects a random target object type. The user uses the arm to move the target object into the detection region (ROI). When the correct object appears inside the Region of Interest (ROI) and the user presses 'C' to verify, the application:

- Detects the object using YOLOv8 nano object detection model (AI-driven)
- Classifies the object type and color using YOLOv8 models (with RGB template matching fallback for color classification)
- Uses HSV for display/info purposes in scan_scene(), not for game logic
- Validates depth and object type using majority voting (5 samples, 3/5 agreement)
- Triggers visual feedback (green flash for correct, red flash for wrong) in the Pygame interface
- Increments the game score and assigns a new target object
- User can press 'H' to move the robot to home position manually

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
./scripts/run_game_pygame.sh
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
| Z | Close gripper |
| X | Open gripper |

### Game Controls

| Key | Action |
|-----|--------|
| N | Start new round (AI scan) |
| C | Check object in ROI (verification) |
| H | Move robot to home position |
| Esc | Quit application |

---

## Key Tunable Parameters

### In `src/ai_vision.py`:

```python
DEPTH_THRESHOLD_M = 0.7        # Only detect objects within this depth (meters)
MIN_CONFIDENCE = 0.2            # Minimum confidence for object detection
YOLO_CONF_THRESHOLD = 0.1       # YOLO confidence threshold (very low for all objects)
YOLO_IMAGE_SIZE = 640          # YOLO input image size (optimized for speed)
YOLO_IOU_THRESHOLD = 0.5       # IoU threshold for non-maximum suppression
YOLO_MAX_DETECTIONS = 500      # Maximum detections per image
```

### In `src/color_picking_game_pygame.py`:

```python
JOINT_STEP = 0.12              # Radians per keypress (robot movement increment)
MOVING_TIME = 0.12             # Moving time for joint movements (seconds)
ROI_PIXEL_MIN = 500            # Minimum color pixels for detection
SAMPLES_FOR_CONFIRM = 5        # Number of frames to sample for verification
SAMPLES_MAJORITY = 3            # Majority votes needed (3 out of 5 samples)
```

**Note**: AI scan uses 3 frames (optimized for speed), verification uses 5 frames with 3/5 majority voting.

These values govern AI detection sensitivity, depth filtering, robot movement, and performance.

---

## File Structure

```
PX150_Project/
├── README.md                      # Main project documentation
├── requirements.txt               # Python dependencies
├── DESIGN.md                     # Design documentation
│
├── src/                          # Main source code
│   ├── color_picking_game_pygame.py   # Main game controller
│   ├── ai_vision.py              # AI vision system (YOLOv8)
│   └── ai_game_state.py          # Game state management
│
├── models/                       # AI model files
│   ├── yolov8n.pt               # YOLOv8 nano object detection model
│   ├── yolov8n-cls.pt           # YOLOv8 nano classification model
│   └── yolov8s.pt                # YOLOv8 small model (optional)
│
├── tests/                        # Test files and results
│   ├── test_ai_system.py         # AI system tests
│   ├── test_dependencies.py      # Dependency verification
│   └── test_results_*.log         # Test result logs
│
├── docs/                         # Documentation
│   ├── AI_INTEGRATION.md         # AI integration guide
│   ├── TEST_SUMMARY.md           # Test results summary
│   ├── TESTING_DOCUMENTATION.md  # Comprehensive testing docs
│   ├── INSTALLATION_STATUS.md    # Installation guide
│   └── YOLO_CLASSIFICATION_INTEGRATION.md
│
└── scripts/                      # Shell scripts
    ├── run_arm.sh                # Launch robot arm
    └── run_game_pygame.sh        # Launch game interface
```



  
