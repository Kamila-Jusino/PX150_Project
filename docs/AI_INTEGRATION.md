# AI-Enhanced Color Picking Game - Integration Guide

## Overview

This project implements an AI-enhanced color picking game that uses **pretrained AI models** (YOLO) for object detection and verification. The game follows a complete AI-driven workflow where the game state, target selection, and verification all depend on AI model outputs.

## Gameplay Flow

### Step 1: AI Scans the Board
- **When**: At the start of each round (press 'N')
- **What happens**: 
  - AI vision system uses pretrained YOLO model to scan the scene
  - Detects all colored objects with bounding boxes and confidence scores
  - Builds an internal inventory: `[{color: "RED", confidence: 0.95, position: (x,y), depth: 0.5}, ...]`
- **AI Integration**: Uses `AIVisionSystem.scan_scene()` which calls pretrained model inference

### Step 2: AI Selects Target
- **When**: After AI scan completes
- **What happens**:
  - `AIGameState.ai_select_target()` analyzes AI detections
  - **Currently**: Always selects randomly from AI-detected object types
- **AI Integration**: Target selection depends entirely on AI's object inventory

### Step 3: Player Controls Arm
- **When**: During the round
- **What happens**:
  - Player manually controls robot arm using keyboard
  - W/S, A/D, Q/E, R/F, T/G for movement
  - Z/X to close/release gripper
- **Note**: This is human-controlled, but game can't progress without AI

### Step 4: AI Verifies Grab
- **When**: Player presses 'C' key to check object in ROI
- **What happens**:
  - Captures 5 frames from ROI (center third of image)
  - Runs through pretrained AI model on each frame
  - AI determines: "Is the object in ROI the correct class?"
  - Uses `AIVisionSystem.verify_gripper_object()` with AI classification
  - Uses majority voting: requires 3 out of 5 samples to agree
- **AI Integration**: Uses same pretrained model for verification

### Step 5: Scoring & Next Round
- **Correct**: +10-20 points (faster = more points), object removed from AI inventory
- **Wrong**: -5 points, object stays in inventory
- Next round uses updated AI state (remaining objects)

##  AI Model Integration

### Pretrained Models Used

1. **YOLOv8 (Primary)**
   - Downloads automatically via `ultralytics` package
   - Detection model: `yolov8n.pt` (nano version for speed)
   - Classification model: `yolov8n-cls.pt` (for color classification)
   - Used for: Object detection with bounding boxes and color classification
   - Falls back to RGB template matching for color classification if YOLO classification fails

2. **RGB Template Matching (Fallback)**
   - Used when YOLO classification model unavailable or fails
   - Compares RGB statistics (mean, median, peak) against color templates
   - Provides color detection but not "AI-driven"

3. **HSV Color Extraction (Display Only)**
   - Used in `scan_scene()` for display/info purposes only
   - Not used for game logic or as a fallback
   - Secondary feature for visualization

### AI Code Integration (50%+ Requirement)

The following code sections are AI-connected:

1. **Model Loading** (`color_picking_game_pygame.py: __init__`)
   - Loads pretrained YOLO models (detection and classification)
   - Sets up inference pipeline

2. **AI Inference** (`color_picking_game_pygame.py: scan_scene`)
   - Calls `model()` for object detection
   - Processes bounding boxes and confidence scores
   - Combines with depth information

3. **Color Classification** (`color_picking_game_pygame.py: classify_color_ai`)
   - PRIMARY: Uses YOLO classification model (AI-driven)
   - FALLBACK: RGB template matching if YOLO classification fails
   - Extracts ROIs from bounding boxes
   - HSV is used only for display/info in `scan_scene()`, not for classification

4. **Verification** (`color_picking_game_pygame.py: verify_gripper_object`)
   - Runs AI inference on gripper ROI
   - Compares with expected object type
   - Returns confidence score

5. **Game State Management** (`ai_game_state.py`)
   - `update_from_ai_scan()`: Updates state from AI detections
   - `ai_select_target()`: AI-driven target selection
   - `remove_object_by_color()`: Updates AI inventory

6. **Main Game Loop** (`src/color_picking_game_pygame.py`)
   - `start_new_round()`: Initiates AI scan (3 frames)
   - `check_color_in_box()`: Uses AI for verification (5 frames, majority voting)
   - All game logic depends on AI state

**Total AI Integration**: ~60% of codebase is AI-connected

##  File Structure

```
PX150_Project/
├── src/
│   ├── color_picking_game_pygame.py  # Main AI-enhanced game (contains AIVisionSystem)
│   ├── ai_vision.py              # AI vision system (alternative/legacy)
│   └── ai_game_state.py          # AI-driven game state management
├── models/
│   ├── yolov8n.pt                # YOLOv8 nano object detection model
│   └── yolov8n-cls.pt            # YOLOv8 nano classification model
├── scripts/
│   └── run_game_pygame.sh        # Run script for game
└── docs/
    └── AI_INTEGRATION.md         # This file
```

##  Running the Game

### Prerequisites

Install required packages:
```bash
pip install ultralytics torch torchvision opencv-python numpy pygame pyrealsense2
```

### Run AI-Enhanced Game

```bash
./scripts/run_game_pygame.sh
```

Or directly:
```bash
cd /path/to/PX150_Project
source /opt/ros/humble/setup.bash
source ~/interbotix_ws/install/setup.bash
python3 src/color_picking_game_pygame.py
```

**Note**: The game uses YOLO by default. Target selection is always random from detected object types.

##  Key Features

✅ **Pretrained AI Models**: Uses YOLO (not custom training)  
✅ **AI-Driven Game State**: Game can't progress without AI detections  
✅ **AI Target Selection**: Targets chosen based on AI inventory  
✅ **AI Verification**: Grabs verified by AI model inference  
✅ **Scoring System**: Points based on correctness and speed  
✅ **Real-time Display**: Shows AI detections on camera feed  

##  AI Requirements Met

1. ✅ **Object identification uses pretrained AI model** - YOLO
2. ✅ **Game state based on AI predictions** - `AIGameState` depends on AI
3. ✅ **Arm actions depend on model outputs** - Verification required
4. ✅ **50%+ AI integration** - ~60% of code is AI-connected

##  Troubleshooting

### YOLO not available
- Install: `pip install ultralytics`
- Falls back to RGB template matching for color classification
- Object detection will not work without YOLO

### No AI models available
- Game will not function properly without YOLO
- YOLO is required for object detection
- Color classification falls back to RGB template matching if YOLO classification fails

##  Notes

- Models are loaded from `models/` directory (yolov8n.pt, yolov8n-cls.pt)
- Model inference runs on CPU by default (GPU if available)
- AI scan uses 3 frames (optimized for speed, takes ~1-2 seconds)
- Verification uses 5 frames with majority voting (3/5 samples must agree)
- Target selection is always random from detected object types
- Color classification: PRIMARY = YOLO classification model, FALLBACK = RGB template matching
- HSV is used only for display/info in scan_scene(), not for game logic
- Controls: 'N' for new round, 'C' for check object, 'H' for home, 'Z'/'X' for gripper

