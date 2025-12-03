#!/usr/bin/env python3

"""
PX150 Color Picking Game - Pygame Version
All UI, camera, and controls in one Pygame window
"""

import time
import random
import cv2
import numpy as np
import pygame
import sys
import logging
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
import pyrealsense2 as rs

# AI Integration imports
from ai_vision import AIVisionSystem
from ai_game_state import AIGameState

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('px150_game.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ----- TUNABLE PARAMETERS -----
JOINT_STEP = 0.12 # radians per keypress
MOVING_TIME = 0.12 # moving_time passed to Interbotix set_joint_positions
ROI_PIXEL_MIN = 500 # min color pixels for detection
DEPTH_THRESHOLD_M = 0.7 # meters, only detect close objects
SAMPLES_FOR_CONFIRM = 5 # color samples for closing gripper
SAMPLES_MAJORITY = 3 # majority votes needed
# Pygame window settings - FULLSCREEN
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
# -------------------------------

class ColorPickingGame:
    def __init__(self):
        # --- Robot init ---
        logger.info("Initializing Interbotix PX150...")
        try:
            self.robot = InterbotixManipulatorXS(
                robot_model="px150",
                group_name="arm",
                gripper_name="gripper"
            )
        except Exception as e:
            logger.error(f"Failed to initialize robot: {e}")
            raise

        # --- Camera init ---
        logger.info("Initializing RealSense camera...")
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.z16, 30)
        self.profile = self.pipeline.start(cfg)
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale() if depth_sensor else 0.001

        # warm-up frames
        for _ in range(20):
            self.pipeline.wait_for_frames()

        # --- AI Systems ---
        logger.info("Initializing AI Vision System...")
        try:
            self.ai_vision = AIVisionSystem(self.pipeline, model_type='yolo')
        except Exception as e:
            logger.error(f"Failed to initialize AI Vision System: {e}")
            raise
        
        logger.info("Initializing AI Game State...")
        self.ai_game_state = AIGameState()

        # Movement tuning
        self.joint_step = JOINT_STEP
        self.moving_time = MOVING_TIME

        # Color HSV ranges (used as fallback when YOLO fails)
        self.color_ranges = {
            'RED': [[(0, 120, 70), (8, 255, 255)], [(170, 120, 70), (180, 255, 255)]],
            'GREEN': [[(36, 50, 50), (85, 255, 255)]],
            'BLUE': [[(95, 80, 80), (125, 255, 255)]],
            'YELLOW': [[(15, 100, 100), (35, 255, 255)]],
            'ORANGE': [[(5, 100, 100), (15, 255, 255)]],
            'PURPLE': [[(125, 50, 50), (155, 255, 255)]],
        }

        self.colors = list(self.color_ranges.keys())
        self.current_color = None  # Legacy, kept for compatibility
        self.current_target = None  # NEW: Object type target (PRIMARY - AI-driven)
        self.start_time = None
        self.round_times = []
        self.gripper_open = True
        self.detected_color = None
        self.status_message = "Press 'N' to start a new round"
        self.last_status_time = time.time()
        self.score = 0
        
        # Visual effects
        self.flash_color = None  # For red/green screen flash
        self.flash_end_time = 0
        self.confetti_particles = []  # For confetti effect
        self.confetti_active = False
        self.confetti_end_time = 0

        # --- Pygame init ---
        pygame.init()
        # Fullscreen mode
        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        self.screen_width, self.screen_height = self.screen.get_size()
        pygame.display.set_caption("PX150 Game")
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
        self.running = True

        # Move to a safe starting pose
        print("Going to home pose...")
        try:
            self.robot.arm.go_to_home_pose()
            self.robot.gripper.release()
            time.sleep(1.0)
        except Exception as e:
            logger.warning(f"Couldn't go to home pose: {e}")
        logger.info("Game initialized! Pygame window should open.")

    # ----------------- Camera & Vision -----------------
    def get_camera_frame(self):
        """Get latest camera frame and convert to Pygame surface"""
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=100)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                return None, None
            color_image = np.asanyarray(color_frame.get_data())
            h, w = color_image.shape[:2]
            
            # Draw ROI rectangle
            cv2.rectangle(color_image, (w//3, h//3), (2*w//3, 2*h//3), (0, 255, 0), 2)
            # Get detected color - Use HSV for camera display (faster, YOLO still used for scanning/verification)
            roi = color_image[h//3:2*h//3, w//3:2*w//3]
            # Use HSV for real-time display (faster than YOLO per frame)
            detected = self.get_dominant_color_with_depth(roi, depth_frame, w, h)
            self.detected_color = detected
            # Add text overlay on camera image
            if self.current_target:
                display_text = f"TARGET: {self.current_target}"
                if self.current_color:
                    display_text += f" ({self.current_color})"
                cv2.putText(color_image, display_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                if self.start_time:
                    elapsed = time.time() - self.start_time
                    cv2.putText(color_image, f"Time: {elapsed:.1f}s", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            else:
                cv2.putText(color_image, "Press 'N' to start", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            if detected:
                cv2.putText(color_image, f"Sees: {detected}", (10, h-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            # Convert BGR to RGB for Pygame
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            
            # Convert to Pygame surface
            # pygame.surfarray expects (width, height) format
            surface = pygame.surfarray.make_surface(np.swapaxes(color_image, 0, 1))
            return surface, color_image
        except Exception as e:
            return None, None

    def get_dominant_color_with_depth(self, roi_color, depth_frame, full_w, full_h):
        """HSV-based color detection - used as fallback when YOLO fails"""
        try:
            hsv = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)
            best_color = None
            best_count = 0
            for cname, ranges in self.color_ranges.items():
                mask_total = np.zeros(hsv.shape[:2], dtype=np.uint8)
                for r in ranges:
                    mask_total = cv2.bitwise_or(mask_total, cv2.inRange(hsv, np.array(r[0]), np.array(r[1])))
                count = int(cv2.countNonZero(mask_total))
                if count > best_count:
                    best_count = count
                    best_color = cname

            if best_count < ROI_PIXEL_MIN:
                return None

            depth_image = np.asanyarray(depth_frame.get_data())
            h_r, w_r = roi_color.shape[:2]
            x0 = full_w // 3
            y0 = full_h // 3
            depth_roi = depth_image[y0:y0 + h_r, x0:x0 + w_r]
            valid = depth_roi[depth_roi > 0]
            if valid.size == 0:
                return None
            mean_depth_m = float(valid.mean()) * float(self.depth_scale)
            if mean_depth_m > DEPTH_THRESHOLD_M:
                return None
            return best_color
        except (cv2.error, ValueError, AttributeError) as e:
            logger.debug(f"Color detection error: {e}")
            return None
        except Exception as e:
            logger.warning(f"Unexpected error in color detection: {e}")
            return None

    # ----------------- Game logic -----------------
    def start_new_round(self):
        if not self.gripper_open:
            self._release_gripper()
        
        # AI Integration: YOLO scans table and identifies objects + colors
        print("\n" + "="*50)
        print("YOLO SCANNING TABLE...")
        print("="*50)
        # Reduce samples to avoid hanging (3 samples is enough)
        try:
            detections = self.ai_vision.scan_scene(num_samples=3)  # Optimized for speed
            num_objects = self.ai_game_state.update_from_ai_scan(detections)
        except (RuntimeError, rs.error) as e:
            logger.error(f"Camera error during YOLO scan: {e}")
            num_objects = 0
            detections = []
        except Exception as e:
            logger.error(f"Error during YOLO scan: {e}", exc_info=True)
            num_objects = 0
            detections = []
        
        print(f"\n✓ YOLO scan complete: {num_objects} objects detected")
        
        # Show what colors YOLO found
        if num_objects > 0:
            summary = self.ai_game_state.get_detection_summary()
            print(f"  Object types detected: {', '.join(summary['available_object_types'])}")
            for obj_type, count in summary['object_type_counts'].items():
                print(f"    - {obj_type}: {count} object(s)")
            print(f"  Colors detected (for display): {', '.join(summary['available_colors'])}")
        
        # AI Integration: AI picks target OBJECT TYPE (random selection)
        if num_objects > 0:
            # Target selection: randomly selects from detected object types
           
            ai_target_object = self.ai_game_state.ai_select_target()
            if ai_target_object:
                self.current_target = ai_target_object  # PRIMARY: Object type (AI-driven)
                self.start_time = time.time()
                
                # Get color for display (SECONDARY - HSV extracted)
                target_colors = [obj['color'] for obj in self.ai_game_state.detected_objects 
                               if obj.get('class') == ai_target_object and obj.get('color')]
                display_color = target_colors[0] if target_colors else "unknown"
                
                print(f"\n✓ AI selected target: {ai_target_object} ({display_color}) (from {num_objects} detected objects)")
                self.set_status(f"ROUND {len(self.round_times) + 1} - Pick up the {ai_target_object}! ({display_color})")
            else:
                # Fallback to random if AI can't select
                print(f"\n⚠ AI couldn't select target, using random")
                self.current_target = 'block'  # Default fallback
                self.current_color = random.choice(self.colors)
                self.start_time = time.time()
                self.set_status(f"ROUND {len(self.round_times) + 1} - Pick up the {self.current_target}! ({self.current_color})")
        else:
            # Fallback to random if no AI detections
            print(f"\n⚠ No objects detected by YOLO, using random")
            self.current_target = 'block'  # Default fallback
            self.current_color = random.choice(self.colors)
            self.start_time = time.time()
            self.set_status(f"ROUND {len(self.round_times) + 1} - Pick up the {self.current_target}! ({self.current_color})")
        print("="*50)

    def check_color_in_box(self):
        """Check if the target object type is within the ROI square - PRIMARY: YOLO object detection, SECONDARY: HSV color (display)"""
        if not self.current_target:
            self.set_status("Start a round first by pressing 'N'.")
            return
        self.set_status("AI verifying object in box...")
        
        # PRIMARY: Use AI to verify grabbed object TYPE (YOLO)
        ai_verifications = []
        color_samples = []  # SECONDARY: HSV color samples (for display/info)
        for _ in range(SAMPLES_FOR_CONFIRM):
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=2000)
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue
                color_image = np.asanyarray(color_frame.get_data())
                h, w = color_image.shape[:2]
                roi = color_image[h//3:2*h//3, w//3:2*w//3]
                
                # PRIMARY: Verify object TYPE using AI model (YOLO)
                is_correct, ai_confidence = self.ai_vision.verify_gripper_object(roi, self.current_target)
                if ai_confidence > 0.3:
                    ai_verifications.append((is_correct, ai_confidence))
                
                # SECONDARY: Extract color using HSV (for display/info)
                detected_color = self.get_dominant_color_with_depth(roi, depth_frame, w, h)
                if detected_color:
                    color_samples.append(detected_color)
            except (RuntimeError, rs.error) as e:
                logger.debug(f"Frame capture error: {e}")
            except Exception as e:
                logger.warning(f"Unexpected error in verification loop: {e}")
            time.sleep(0.1)
        
        # PRIMARY: Use AI verification (object type) if available
        if len(ai_verifications) >= SAMPLES_MAJORITY:
            correct_votes = sum(1 for is_correct, _ in ai_verifications if is_correct)
            avg_conf = sum(conf for _, conf in ai_verifications) / len(ai_verifications)
            
            if correct_votes >= SAMPLES_MAJORITY:
                # Correct object type detected - award points and complete round
                self.score += 1
                elapsed = time.time() - self.start_time if self.start_time else 0
                
                # Get color for display (SECONDARY)
                display_color = max(set(color_samples), key=color_samples.count) if color_samples else "unknown"
                
                self.set_status(f"CORRECT! You grabbed a {self.current_target} ({display_color})! +1 point (Score: {self.score}) | Time: {elapsed:.2f}s")
                self.round_times.append(elapsed)
                # Visual effects: Green flash and confetti
                self.flash_color = (0, 255, 0)  # Green
                self.flash_end_time = time.time() + 0.5
                self._start_confetti()
                # AI Integration: Remove object from AI inventory
                self.ai_game_state.remove_object_by_type(self.current_target)
                self.current_target = None
                self.current_color = None  # Clear legacy color too
                self.start_time = None
            else:
                # Wrong object type detected
                detected_types = []
                for _ in range(3):  # Quick check for detected type
                    try:
                        frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                        color_frame = frames.get_color_frame()
                        if color_frame:
                            color_image = np.asanyarray(color_frame.get_data())
                            h, w = color_image.shape[:2]
                            roi = color_image[h//3:2*h//3, w//3:2*w//3]
                            detected_type = self.ai_vision.detect_object_type(roi)
                            if detected_type:
                                detected_types.append(detected_type)
                    except (RuntimeError, rs.error) as e:
                        logger.debug(f"Frame capture error in wrong object detection: {e}")
                    except Exception as e:
                        logger.warning(f"Error detecting wrong object type: {e}")
                
                wrong_type = max(set(detected_types), key=detected_types.count) if detected_types else "unknown"
                display_color = max(set(color_samples), key=color_samples.count) if color_samples else "unknown"
                self.set_status(f"Wrong object in box: {wrong_type} ({display_color}) (need {self.current_target})")
                # Visual effects: Red flash
                self.flash_color = (255, 0, 0)  # Red
                self.flash_end_time = time.time() + 0.5
        elif not color_samples:
            # No detections from either method
            self.set_status("No object detected in the box.")
            return
        else:
            # FALLBACK: If AI didn't have enough samples, try HSV color as last resort
            most_common = max(set(color_samples), key=color_samples.count)
            count = color_samples.count(most_common)
            
            # Check if color matches any object of target type
            target_colors = [obj['color'] for obj in self.ai_game_state.detected_objects 
                           if obj.get('class') == self.current_target and obj.get('color')]
            
            if most_common in target_colors and count >= SAMPLES_MAJORITY:
                # Correct color (and likely correct object type) - award points
                self.score += 1
                elapsed = time.time() - self.start_time if self.start_time else 0
                self.set_status(f"CORRECT! +1 point (Score: {self.score}) | Time: {elapsed:.2f}s")
                self.round_times.append(elapsed)
                # Visual effects: Green flash and confetti
                self.flash_color = (0, 255, 0)  # Green
                self.flash_end_time = time.time() + 0.5
                self._start_confetti()
                # AI Integration: Remove object from AI inventory
                self.ai_game_state.remove_object_by_type(self.current_target)
                self.current_target = None
                self.current_color = None
                self.start_time = None
            else:
                self.set_status(f"Wrong color in box: {most_common} (need {self.current_target})")
                # Visual effect: Red flash
                self.flash_color = (255, 0, 0)  # Red
                self.flash_end_time = time.time() + 0.5

    def _release_gripper(self):
        try:
            self.robot.gripper.release()
            self.gripper_open = True
        except (AttributeError, RuntimeError) as e:
            logger.error(f"Gripper release error: {e}")
            self.gripper_open = True  # Assume open on error
        except Exception as e:
            logger.error(f"Unexpected gripper error: {e}")
            self.gripper_open = True
        time.sleep(0.2)

    def set_status(self, message):
        self.status_message = message
        self.last_status_time = time.time()

    def go_to_home(self):
        """Move arm to home position and drop anything being held"""
        self.set_status("Moving to home position...")
        try:
            # Release gripper first to drop anything
            if not self.gripper_open:
                self._release_gripper()
            # Move to home
            self.robot.arm.go_to_home_pose(moving_time=2.0)
            self.set_status("Arm moved to home position.")
        except (AttributeError, RuntimeError) as e:
            logger.error(f"Error moving to home: {e}")
            self.set_status(f"Error moving to home: {e}")
        except Exception as e:
            logger.error(f"Unexpected error moving to home: {e}", exc_info=True)
            self.set_status(f"Error moving to home: {e}")

    # ----------------- Movement -----------------
    def move_joint_delta(self, joint_idx, delta):
        """Move a joint by delta amount. Validates input before execution."""
        if not isinstance(joint_idx, int) or not isinstance(delta, (int, float)):
            self.set_status(f"Invalid input: joint_idx and delta must be numbers")
            return
        
        try:
            current_joints = list(self.robot.arm.get_joint_commands())
            if joint_idx < 0 or joint_idx >= len(current_joints):
                self.set_status(f"Invalid joint index: {joint_idx} (valid range: 0-{len(current_joints)-1})")
                return
            current_joints[joint_idx] += delta
            self.robot.arm.set_joint_positions(current_joints, moving_time=MOVING_TIME)
        except (ValueError, IndexError, AttributeError) as e:
            logger.error(f"Movement error: {e}")
            self.set_status(f"Movement error: {e}")
        except Exception as e:
            logger.error(f"Unexpected movement error: {e}", exc_info=True)
            self.set_status(f"Movement error: {e}")

    # ----------------- Visual Effects -----------------
    def _start_confetti(self):
        """Start confetti effect"""
        self.confetti_active = True
        self.confetti_end_time = time.time() + 2.0  # Confetti for 2 seconds
        self.confetti_particles = []
        # Create confetti particles
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        for _ in range(50):
            self.confetti_particles.append({
                'x': random.randint(0, self.screen_width),
                'y': random.randint(-100, 0),
                'vx': random.uniform(-3, 3),
                'vy': random.uniform(2, 6),
                'color': random.choice(colors),
                'size': random.randint(5, 15)
            })
    
    def _update_confetti(self):
        """Update confetti particles"""
        if not self.confetti_active:
            return
        
        if time.time() > self.confetti_end_time:
            self.confetti_active = False
            self.confetti_particles = []
            return
        
        for particle in self.confetti_particles:
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']
            particle['vy'] += 0.2  # Gravity
            
            # Reset if off screen
            if particle['y'] > self.screen_height:
                particle['y'] = random.randint(-100, 0)
                particle['x'] = random.randint(0, self.screen_width)
    
    def _draw_confetti(self):
        """Draw confetti particles"""
        if not self.confetti_active:
            return
        
        for particle in self.confetti_particles:
            pygame.draw.rect(
                self.screen,
                particle['color'],
                (int(particle['x']), int(particle['y']), particle['size'], particle['size'])
            )
    
    # ----------------- Rendering -----------------
    def _wrap_text(self, text, font, max_width):
        """Wrap text to fit within max_width, returning list of lines"""
        words = text.split(' ')
        lines = []
        current_line = []
        
        for word in words:
            # Test if adding this word would exceed width
            test_line = ' '.join(current_line + [word])
            test_surface = font.render(test_line, True, (255, 255, 255))
            if test_surface.get_width() <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines if lines else [text]
    
    def draw_ui(self):
        """Draw UI overlay on the screen"""
        # Handle screen flash effect
        if self.flash_color and time.time() < self.flash_end_time:
            # Flash the screen with red or green
            alpha = int(128 * (1.0 - (self.flash_end_time - time.time()) / 0.5))
            flash_surface = pygame.Surface((self.screen_width, self.screen_height))
            flash_surface.set_alpha(alpha)
            flash_surface.fill(self.flash_color)
            self.screen.blit(flash_surface, (0, 0))
        elif self.flash_color:
            self.flash_color = None
        
        # Clear screen with dark background (if no flash)
        if not self.flash_color or time.time() >= self.flash_end_time:
            self.screen.fill((20, 20, 30))
        # Get and draw camera frame
        camera_surface, _ = self.get_camera_frame()
        if camera_surface:
            # Scale camera to fit left side (maintain aspect ratio)
            scale_factor = min((CAMERA_HEIGHT) / CAMERA_WIDTH, (self.screen_height - 40) / CAMERA_HEIGHT)
            scaled_width = int(CAMERA_WIDTH * scale_factor)
            scaled_height = int(CAMERA_HEIGHT * scale_factor)
            camera_scaled = pygame.transform.scale(camera_surface, (scaled_width, scaled_height))
            self.screen.blit(camera_scaled, (20, 20))

        # Draw UI panel in the middle
        panel_x = CAMERA_HEIGHT + 40
        panel_y = 20
        panel_width = 350 # Increased width for better text display
        panel_height = self.screen_height - 40
        

        # Statistics panel on the far right (increased width for better display)
        stats_panel_x = panel_x + panel_width + 20
        stats_panel_width = max(300, self.screen_width - stats_panel_x - 20)  # Minimum 300px width
        # Draw panel background
        pygame.draw.rect(self.screen, (40, 40, 50), (panel_x, panel_y, panel_width, panel_height))
        pygame.draw.rect(self.screen, (60, 60, 70), (panel_x, panel_y, panel_width, panel_height), 2)
        y_offset = panel_y + 20
        # Title
        title = self.font_large.render("PX150 Color Picking Game", True, (255, 255, 255))
        self.screen.blit(title, (panel_x + 20, y_offset))
        y_offset += 60
        # Game status
        if self.current_target:
            target_text = f"Target: {self.current_target}"
            if self.current_color:
                target_text += f" ({self.current_color})"
            color_text = self.font_medium.render(target_text, True, (255, 255, 0))
            self.screen.blit(color_text, (panel_x + 20, y_offset))
            y_offset += 40
            if self.start_time:
                elapsed = time.time() - self.start_time
                time_text = self.font_small.render(f"Time: {elapsed:.1f}s", True, (200, 200, 200))
                self.screen.blit(time_text, (panel_x + 20, y_offset))
                y_offset += 30
        # Detected color
        if self.detected_color:
            detected_text = self.font_small.render(f"Camera sees: {self.detected_color}", True, (0, 255, 255))
            self.screen.blit(detected_text, (panel_x + 20, y_offset))
            y_offset += 30
        # Status message (with text wrapping)
        status_lines = self._wrap_text(self.status_message, self.font_small, panel_width - 40)
        for line in status_lines:
            status_text = self.font_small.render(line, True, (150, 255, 150))
            self.screen.blit(status_text, (panel_x + 20, y_offset))
            y_offset += 25
        y_offset += 25  # Extra spacing after status
        # Controls section
        controls_title = self.font_medium.render("Controls:", True, (255, 255, 255))
        self.screen.blit(controls_title, (panel_x + 20, y_offset))
        y_offset += 35
        controls = [
            "W/S: Forward/Backward",
            "A/D: Left/Right",
            "Q/E: Up/Down",
            "R/F: Wrist pitch up/down",
            "T/G: Wrist roll left/right",
            "C: Check color in box",
            "Z: Close gripper manually",
            "X: Release gripper manually",
            "H: Go to home (drops object)",
            "N: New round",
            "ESC: Quit"
        ]
        for control in controls:
            # Wrap long control text
            control_lines = self._wrap_text(control, self.font_small, panel_width - 50)
            for line in control_lines:
                control_text = self.font_small.render(line, True, (200, 200, 200))
                self.screen.blit(control_text, (panel_x + 30, y_offset))
                y_offset += 25
        # Gripper status
        gripper_status = "OPEN" if self.gripper_open else "CLOSED"
        gripper_color = (0, 255, 0) if self.gripper_open else (255, 0, 0)
        gripper_text = self.font_small.render(f"Gripper: {gripper_status}", True, gripper_color)
        self.screen.blit(gripper_text, (panel_x + 20, panel_y + panel_height - 30))

        # Draw statistics panel on the right
        pygame.draw.rect(self.screen, (40, 40, 50), (stats_panel_x, panel_y, stats_panel_width, panel_height))
        pygame.draw.rect(self.screen, (60, 60, 70), (stats_panel_x, panel_y, stats_panel_width, panel_height), 2)
        
        stats_y_offset = panel_y + 20
        
        # Stats title
        stats_title = self.font_medium.render("Statistics", True, (255, 255, 255))
        self.screen.blit(stats_title, (stats_panel_x + 20, stats_y_offset))
        stats_y_offset += 40
        # Always show score prominently
        score_label = self.font_small.render("Score:", True, (200, 200, 200))
        self.screen.blit(score_label, (stats_panel_x + 20, stats_y_offset))
        score_value = self.font_large.render(f"{self.score}", True, (255, 215, 0))
        self.screen.blit(score_value, (stats_panel_x + 20, stats_y_offset + 25))
        stats_y_offset += 70
        # Round statistics
        if self.round_times:
            rounds_text = self.font_small.render(f"Rounds: {len(self.round_times)}", True, (200, 200, 200))
            self.screen.blit(rounds_text, (stats_panel_x + 20, stats_y_offset))
            stats_y_offset += 30
            best_text = self.font_small.render(f"Best time: {min(self.round_times):.2f}s", True, (200, 200, 200))
            self.screen.blit(best_text, (stats_panel_x + 20, stats_y_offset))
            stats_y_offset += 30
            avg_text = self.font_small.render(f"Average: {sum(self.round_times)/len(self.round_times):.2f}s", True, (200, 200, 200))
            self.screen.blit(avg_text, (stats_panel_x + 20, stats_y_offset))
        else:
            no_stats_text = self.font_small.render("No rounds completed yet", True, (150, 150, 150))
            self.screen.blit(no_stats_text, (stats_panel_x + 20, stats_y_offset))

    # ----------------- Main loop -----------------
    def run(self):
        logger.info("Game running! Use the Pygame window for controls.")
        
        while self.running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_n:
                        if not self.gripper_open:
                            self._release_gripper()
                        self.start_new_round()
                    elif event.key == pygame.K_c:
                        self.check_color_in_box()
                    elif event.key == pygame.K_h:
                        self.go_to_home()
                    elif event.key == pygame.K_z:
                        if self.gripper_open:
                            try:
                                self.robot.gripper.grasp()
                                self.gripper_open = False
                                self.set_status("Gripper manually closed.")
                            except (AttributeError, RuntimeError) as e:
                                logger.error(f"Gripper error: {e}")
                                self.set_status(f"Gripper error: {e}")
                            except Exception as e:
                                logger.error(f"Unexpected gripper error: {e}")
                                self.set_status(f"Gripper error: {e}")
                    elif event.key == pygame.K_x:
                        if not self.gripper_open:
                            self._release_gripper()
                            self.set_status("Gripper manually released.")
                    elif event.key == pygame.K_w:
                        self.move_joint_delta(1, -JOINT_STEP)
                    elif event.key == pygame.K_s:
                        self.move_joint_delta(1, JOINT_STEP)
                    elif event.key == pygame.K_a:
                        self.move_joint_delta(0, JOINT_STEP)
                    elif event.key == pygame.K_d:
                        self.move_joint_delta(0, -JOINT_STEP)
                    elif event.key == pygame.K_q:
                        self.move_joint_delta(2, -JOINT_STEP)
                    elif event.key == pygame.K_e:
                        self.move_joint_delta(2, JOINT_STEP)
                    elif event.key == pygame.K_r:
                        self.move_joint_delta(3, -JOINT_STEP)
                    elif event.key == pygame.K_f:
                        self.move_joint_delta(3, JOINT_STEP)
                    elif event.key == pygame.K_t:
                        self.move_joint_delta(4, JOINT_STEP)
                    elif event.key == pygame.K_g:
                        self.move_joint_delta(4, -JOINT_STEP)

            # Update visual effects
            self._update_confetti()
            
            # Draw everything
            self.draw_ui()
            
            # Draw confetti on top
            self._draw_confetti()
            
            pygame.display.flip()
            self.clock.tick(30) # 30 FPS
        self.cleanup()

    def cleanup(self):
        logger.info("Shutting down: moving to safe pose & stopping camera.")
        try:
            if not self.gripper_open:
                self._release_gripper()
            self.robot.arm.go_to_home_pose()
            time.sleep(0.6)
            self.robot.arm.go_to_sleep_pose()
        except Exception as e:
            logger.error(f"Cleanup robot error: {e}")
        try:
            self.ai_vision.cleanup()
        except Exception as e:
            logger.warning(f"Error cleaning up AI vision: {e}")
        try:
            self.pipeline.stop()
        except Exception as e:
            logger.warning(f"Error stopping camera pipeline: {e}")
        pygame.quit()
        if self.round_times:
            summary = f"Rounds: {len(self.round_times)} | Best: {min(self.round_times):.2f}s | Avg: {sum(self.round_times)/len(self.round_times):.2f}s"
            logger.info(summary)
            print(summary)
        else:
            logger.info("No rounds completed.")
            print("No rounds completed.")
        logger.info("Goodbye.")
        print("Goodbye.")

if __name__ == "__main__":
    try:
        game = ColorPickingGame()
        game.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        print("\n\nInterrupted by user")
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        import traceback
        print(f"\nError: {e}")
        traceback.print_exc()
    finally:
        logger.info("Exiting...")
        print("Exiting...")
