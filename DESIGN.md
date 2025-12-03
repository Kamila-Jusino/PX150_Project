# PX150 Game - Design Documentation

## Table of Contents
1. [AI Tool Mapping](#ai-tool-mapping)
2. [Architecture Diagram](#architecture-diagram)
3. [Component Descriptions](#component-descriptions)
4. [Data Flow](#data-flow)

---

## AI Tool Mapping

The following table documents the specific AI tools used throughout the development lifecycle and their contributions to the project.

| AI Tool | Purpose/Contribution | Location/Usage | Justification |
|---------|---------------------|----------------|---------------|
| **Cursor (AI-Powered IDE)** | Code generation, refactoring, debugging, and project restructuring | Throughout entire codebase | Primary development assistant for: <br>- Generating initial game loop structure<br>- Refactoring code for better organization<br>- Fixing import paths after restructuring<br>- Debugging integration issues between components<br>- Writing documentation and comments |
| **ChatGPT** | Design consultation, algorithm design, and code review | Architecture design, AI integration logic | Used for: <br>- Initial project architecture planning<br>- AI model integration strategy (YOLO vs MobileNet)<br>- Game state management design patterns<br>- RealSense camera integration guidance<br>- ROS2/Interbotix SDK integration help |
| **YOLOv8 (Ultralytics)** | Object detection and classification model | `src/ai_vision.py` - Model inference | Pretrained AI model for: <br>- Real-time object detection (yolov8n.pt)<br>- Object classification (yolov8n-cls.pt)<br>- Bounding box generation with confidence scores<br>- Primary AI component meeting 50%+ AI integration requirement |
| **Hugging Face (Indirect)** | Model architecture reference | Model selection research | Used for: <br>- Researching pretrained model options<br>- Understanding YOLO model variants<br>- Performance comparison between models |

### AI Tool Contribution Summary

- **Development Tools (Cursor, ChatGPT)**: ~40% of code structure, integration logic, and documentation
- **AI Models (YOLOv8)**: ~60% of runtime AI functionality (object detection, classification, verification)
- **Total AI Integration**: Exceeds 50% requirement through combined use of AI development tools and AI runtime models

---

## Architecture Diagram

The following diagram illustrates the system architecture, component interactions, and data flow between hardware and software components.

```mermaid
graph TB
    subgraph Hardware["Hardware Layer"]
        CAM[Intel RealSense Camera<br/>RGB-D Stream]
        ROBOT[Interbotix PX150<br/>Robotic Arm]
    end
    
    subgraph Software["Software Layer"]
        subgraph Main["Main Game Controller<br/>(color_picking_game_pygame.py)"]
            GAME[ColorPickingGame Class<br/>- Event handling<br/>- Game loop<br/>- UI rendering]
        end
        
        subgraph AI["AI Vision System<br/>(ai_vision.py)"]
            YOLO[YOLOv8 Model<br/>- Object Detection<br/>- Classification]
            VISION[AIVisionSystem<br/>- scan_scene()<br/>- verify_gripper_object()<br/>- classify_color_ai()]
        end
        
        subgraph State["Game State Manager<br/>(ai_game_state.py)"]
            STATE[AIGameState<br/>- update_from_ai_scan()<br/>- ai_select_target()<br/>- remove_object_by_type()]
        end
        
        subgraph UI["User Interface<br/>(Pygame)"]
            DISPLAY[Pygame Window<br/>- Camera feed<br/>- Status display<br/>- Controls overlay]
        end
        
        subgraph Control["Robot Controller<br/>(Interbotix SDK)"]
            ARM[InterbotixManipulatorXS<br/>- Joint control<br/>- Gripper control<br/>- Home position]
        end
    end
    
    %% Data Flow
    CAM -->|RGB + Depth Frames| GAME
    GAME -->|Frames| VISION
    VISION -->|Model Inference| YOLO
    YOLO -->|Detections + Bounding Boxes| VISION
    VISION -->|Object List| STATE
    STATE -->|Target Selection| GAME
    GAME -->|Display Data| DISPLAY
    GAME -->|Control Commands| ARM
    ARM -->|Joint Positions| ROBOT
    GAME -->|Keyboard Events| DISPLAY
    
    %% User Interaction
    DISPLAY -.->|User Input| GAME
    
    style CAM fill:#e1f5ff
    style ROBOT fill:#e1f5ff
    style YOLO fill:#ffeb3b
    style VISION fill:#c8e6c9
    style STATE fill:#c8e6c9
    style GAME fill:#fff9c4
    style DISPLAY fill:#f3e5f5
    style ARM fill:#ffccbc
```

### Alternative Text-Based Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        HARDWARE LAYER                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────┐              ┌──────────────────┐          │
│  │ RealSense Camera │              │  PX150 Robot Arm  │          │
│  │  (RGB-D Stream)  │              │  (6-DOF + Gripper)│          │
│  └────────┬─────────┘              └────────┬─────────┘          │
│           │                                   │                    │
└───────────┼───────────────────────────────────┼──────────────────┘
            │                                   │
            ▼                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                        SOFTWARE LAYER                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │         Main Game Controller                              │   │
│  │  (color_picking_game_pygame.py)                           │   │
│  │  • Event loop management                                  │   │
│  │  • Coordinate all subsystems                              │   │
│  │  • Game state transitions                                 │   │
│  └───────┬───────────────────────────────────────┬───────────┘   │
│           │                                       │               │
│           │                                       │               │
│  ┌────────▼────────┐                    ┌────────▼────────┐      │
│  │  AI Vision      │                    │  Game State     │      │
│  │  System         │                    │  Manager        │      │
│  │                 │                    │                 │      │
│  │  ┌──────────┐   │                    │  • Object       │      │
│  │  │ YOLOv8   │   │                    │    inventory   │      │
│  │  │ Model    │◄──┼─────────────────────│  • Target      │      │
│  │  └──────────┘   │                    │    selection    │      │
│  │                 │                    │  • Score        │      │
│  │  • scan_scene() │                    │    tracking      │      │
│  │  • verify_      │                    └─────────────────┘      │
│  │    gripper()    │                                               │
│  └─────────────────┘                                               │
│           │                                                         │
│           │                                                         │
│  ┌────────▼────────┐                    ┌────────▼────────┐      │
│  │  Pygame UI      │                    │  Robot          │      │
│  │                 │                    │  Controller     │      │
│  │  • Camera feed  │                    │                 │      │
│  │  • Status       │                    │  • Joint        │      │
│  │  • Controls     │                    │    control     │      │
│  │  • Visual       │                    │  • Gripper      │      │
│  │    effects      │                    │    control     │      │
│  └─────────────────┘                    └─────────────────┘      │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

---

## Component Descriptions

### 1. Hardware Layer

#### Intel RealSense Camera
- **Type**: RGB-D (RGB + Depth) camera (D415/D435)
- **Function**: Provides real-time color and depth frames
- **Data**: 
  - RGB frames: 640x480 @ 30fps
  - Depth frames: 640x480 @ 30fps (16-bit depth values)
- **Integration**: Via `pyrealsense2` Python SDK

#### Interbotix PX150 Robotic Arm
- **Type**: 6-DOF manipulator with gripper
- **Function**: Executes physical movements and object manipulation
- **Control**: Via ROS2/Interbotix SDK (`interbotix_xs_modules`)
- **Joints**: Shoulder rotation, shoulder pitch, elbow, wrist angle, wrist rotation, gripper

### 2. Software Components

#### Main Game Controller (`ColorPickingGame`)
- **File**: `src/color_picking_game_pygame.py`
- **Responsibilities**:
  - Initialize all subsystems (camera, robot, AI, UI)
  - Manage main game loop
  - Handle keyboard events (N: new round, C: check object, H: home, Z/X: gripper, W/S/A/D/Q/E/R/F/T/G: joints)
  - Coordinate between AI vision, game state, and robot control
  - Render UI updates
- **Key Methods**:
  - `start_new_round()`: Initiates AI scan (3 frames) and target selection
  - `check_color_in_box()`: Verifies object in ROI using AI (5 frames, majority voting)
  - `run()`: Main event loop (60 FPS display, 30 FPS camera)

#### AI Vision System (`AIVisionSystem`)
- **File**: `src/ai_vision.py`
- **Responsibilities**:
  - Load and manage YOLOv8 models
  - Perform object detection on camera frames
  - Classify object types and colors
  - Verify objects in gripper ROI
- **Key Methods**:
  - `scan_scene()`: Scans entire scene for objects
  - `verify_gripper_object()`: Verifies object in ROI
  - `classify_color_ai()`: AI-based color classification
- **Models Used**:
  - `yolov8n.pt`: Object detection (primary)
  - `yolov8n-cls.pt`: Color classification (fallback)

#### Game State Manager (`AIGameState`)
- **File**: `src/ai_game_state.py`
- **Responsibilities**:
  - Maintain AI-detected object inventory
  - Select target objects based on AI detections
  - Track removed objects
  - Provide detection summaries
- **Key Methods**:
  - `update_from_ai_scan()`: Updates state from AI detections
  - `ai_select_target()`: AI-driven target selection
  - `remove_object_by_type()`: Updates inventory after successful grab

#### Pygame User Interface
- **File**: `src/color_picking_game_pygame.py` (UI methods)
- **Responsibilities**:
  - Display live camera feed
  - Show game status and instructions
  - Render visual effects (flashes, confetti)
  - Display AI detection overlays
- **Components**:
  - Camera viewport
  - Status panel
  - Statistics panel
  - Control instructions

#### Robot Controller
- **File**: `src/color_picking_game_pygame.py` (robot methods)
- **SDK**: `InterbotixManipulatorXS` from Interbotix SDK
- **Responsibilities**:
  - Execute joint movements
  - Control gripper open/close
  - Move to home position
  - Handle movement commands from keyboard

---

## Data Flow

### 1. Initialization Flow
```
Start Application
    ↓
Initialize RealSense Camera → RGB + Depth Streams
    ↓
Initialize PX150 Robot → ROS2 Connection
    ↓
Load YOLOv8 Models → AI Vision System Ready
    ↓
Initialize Pygame UI → Display Window
    ↓
Game Ready
```

### 2. Game Round Flow
```
User Presses 'N' (New Round)
    ↓
AI Vision System: scan_scene()
    ├─→ Capture 3 camera frames (optimized for speed, reduced from 5 to avoid hanging)
    ├─→ Run YOLO inference on each frame
    ├─→ Extract bounding boxes + confidence scores
    ├─→ Combine with depth data
    └─→ Return: [{class, color, position, depth, confidence}, ...]
    ↓
Game State Manager: update_from_ai_scan()
    ├─→ Update detected_objects list
    ├─→ Extract available_object_types
    └─→ Extract available_colors
    ↓
Game State Manager: ai_select_target()
    └─→ Randomly select from available_object_types
    ↓
Main Controller: Display target to user
    ↓
User Controls Robot (Keyboard Input)
    ├─→ Joint movements (W/S, A/D, Q/E, R/F, T/G)
    └─→ Gripper control (Z/X)
    ↓
User Presses 'C' (Check Object in ROI)
    ↓
AI Vision System: verify_gripper_object()
    ├─→ Capture 5 frames from ROI
    ├─→ Run YOLO inference on each
    ├─→ Compare detected class with target
    └─→ Return: (is_correct, confidence)
    ↓
Main Controller: Process Result
    ├─→ If correct: Score points, remove object, flash green
    └─→ If wrong: Penalty, flash red
    ↓
Update UI and Game State
```

### 3. Real-Time Display Flow
```
Camera Frame (30 FPS)
    ↓
Main Loop: get_camera_frame()
    ├─→ Get RGB frame from RealSense
    ├─→ Get depth frame from RealSense
    └─→ Convert to Pygame surface
    ↓
Main Loop: draw_ui()
    ├─→ Draw camera feed
    ├─→ Draw status panel (target, score, instructions)
    ├─→ Draw statistics panel (AI detections summary)
    └─→ Draw visual effects (if any)
    ↓
Pygame: display.flip()
    └─→ Update screen (30 FPS)
```

### 4. Robot Control Flow
```
Keyboard Event Detected
    ↓
Main Controller: Event Handler
    ├─→ Joint Movement Keys (W/S, A/D, etc.)
    │   └─→ move_joint_delta(joint_id, delta)
    │       └─→ robot.arm.set_joint_positions()
    │
    ├─→ Gripper Keys (Z/X)
    │   ├─→ Z: robot.gripper.grasp()
    │   └─→ X: robot.gripper.release()
    │
    └─→ Special Keys (N, C, H, ESC)
        ├─→ N: start_new_round() (initiates AI scan)
        ├─→ C: check_color_in_box() (verifies object in ROI)
        ├─→ H: go_to_home() (moves robot to home position)
        └─→ ESC: quit()
    ↓
Robot Executes Command
    ↓
Feedback to UI (Status Message)
```

---

## Key Design Decisions

### 1. AI-First Architecture
- **Decision**: Game state and target selection depend entirely on AI model outputs
- **Rationale**: Meets project requirement of 50%+ AI integration
- **Implementation**: `AIGameState` maintains object inventory from AI detections only

### 2. Dual Model System
- **Decision**: Use YOLOv8 for object detection, separate model for color classification
- **Rationale**: YOLO excels at object detection, classification model provides color info
- **Fallback**: HSV color detection if AI models unavailable

### 3. Majority Voting for Verification
- **Decision**: Sample 5 frames and require 3/5 agreement for verification
- **Rationale**: Reduces false positives from single-frame noise
- **Implementation**: `SAMPLES_FOR_CONFIRM = 5`, `SAMPLES_MAJORITY = 3`
- **Note**: AI scan uses 3 frames (optimized for speed), verification uses 5 frames (higher confidence needed)

### 4. Modular Component Design
- **Decision**: Separate AI vision, game state, and main controller into distinct classes
- **Rationale**: Enables independent testing and easier maintenance
- **Structure**: `AIVisionSystem`, `AIGameState`, `ColorPickingGame`

### 5. Real-Time Performance Optimization
- **Decision**: Use YOLOv8 nano (smallest model), reduce image size to 640px, limit scan to 3 frames
- **Rationale**: Balance between accuracy and real-time performance (30 FPS camera)
- **Trade-off**: Slightly lower accuracy for faster inference
- **Implementation**: 
  - `YOLO_IMAGE_SIZE = 640` (optimized for speed)
  - `scan_scene(num_samples=3)` (reduced from 5 to avoid hanging)
  - Verification still uses 5 samples for higher confidence

---

## Integration Points

### Hardware-Software Integration
1. **RealSense → Python**: `pyrealsense2` SDK provides frame access
2. **PX150 → Python**: `interbotix_xs_modules` provides ROS2-based control
3. **Camera-Robot Calibration**: Manual alignment (no automatic calibration)

### AI Model Integration
1. **YOLO Loading**: Models loaded from `models/` directory at startup (relative path: `../models/`)
2. **Inference Pipeline**: Frames → Preprocessing → YOLO inference → Post-processing
3. **Result Format**: Standardized dictionary format for detections: `{class, color, position, depth, confidence, bbox}`
4. **Model Files**: 
   - `yolov8n.pt`: Primary object detection model (loaded from `models/yolov8n.pt`)
   - `yolov8n-cls.pt`: Color classification fallback (loaded from `models/yolov8n-cls.pt`)

### UI Integration
1. **Pygame Rendering**: Camera frames converted from NumPy arrays to Pygame surfaces
2. **Event Handling**: Keyboard events captured in Pygame event loop
3. **Visual Feedback**: Status messages, color flashes, confetti effects

---

## Future Enhancements

1. **Automatic Calibration**: Camera-robot coordinate system alignment
2. **Custom Model Training**: Fine-tune YOLO on specific object types
3. **Multi-Object Tracking**: Track objects across frames for better state management
4. **Autonomous Mode**: AI-driven robot control (currently manual)
5. **Performance Metrics**: FPS tracking, inference time monitoring

---

## Key Tunable Parameters

The following parameters can be adjusted to fine-tune system behavior:

### AI Vision Parameters (`src/ai_vision.py`)
```python
DEPTH_THRESHOLD_M = 0.7        # Only detect objects within this depth (meters)
MIN_CONFIDENCE = 0.2            # Minimum confidence for object detection
YOLO_CONF_THRESHOLD = 0.1       # YOLO confidence threshold (very low for all objects)
YOLO_IMAGE_SIZE = 640          # YOLO input image size (optimized for speed)
YOLO_IOU_THRESHOLD = 0.5       # IoU threshold for non-maximum suppression
YOLO_MAX_DETECTIONS = 500      # Maximum detections per image
```

### Game Control Parameters (`src/color_picking_game_pygame.py`)
```python
JOINT_STEP = 0.12              # Radians per keypress (robot movement increment)
MOVING_TIME = 0.12             # Moving time for joint movements (seconds)
ROI_PIXEL_MIN = 500            # Minimum color pixels for detection
SAMPLES_FOR_CONFIRM = 5        # Number of frames to sample for verification
SAMPLES_MAJORITY = 3            # Majority votes needed (3 out of 5 samples)
```

### Performance Parameters
- **AI Scan**: 3 frames (optimized for speed, reduced from 5 to avoid hanging)
- **Verification**: 5 frames with 3/5 majority voting (higher confidence needed)
- **Camera**: 640x480 @ 30 FPS
- **Display**: 60 FPS (Pygame refresh rate)

These values govern AI detection sensitivity, depth filtering, robot movement speed, and overall system performance.

---

## File Structure

The project follows this organized structure:

```
PX150_Project/
├── README.md                      # Main project documentation
├── requirements.txt               # Python dependencies
├── DESIGN.md                     # This file - design documentation
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

## Controls Reference

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

## References

- **YOLOv8 Documentation**: https://docs.ultralytics.com/
- **Interbotix SDK**: https://github.com/Interbotix/interbotix_ros_toolboxes
- **RealSense SDK**: https://github.com/IntelRealSense/librealsense
- **Pygame Documentation**: https://www.pygame.org/docs/

