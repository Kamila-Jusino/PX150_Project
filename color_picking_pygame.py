#!/usr/bin/env python3
"""
PX150 Color Picking Game - Pygame Version
All UI, camera, and controls in one Pygame window
"""

import math
import time
import random
import cv2
import numpy as np
import pygame
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
import pyrealsense2 as rs
import sys

# ----- TUNABLE PARAMETERS -----
JOINT_STEP = 0.12         # radians per keypress
MOVING_TIME = 0.12        # moving_time passed to Interbotix set_joint_positions
ROI_PIXEL_MIN = 500       # min color pixels for detection
DEPTH_THRESHOLD_M = 0.7   # meters, only detect close objects
SAMPLES_FOR_CONFIRM = 5   # color samples for closing gripper
SAMPLES_MAJORITY = 3      # majority votes needed

# Pygame window settings
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
# -------------------------------

class Confetti:
    def __init__(self, x, y, num_particles=100):
        self.particles = []
        self.colors = [
            (255, 215, 0), (255, 255, 255), (0, 255, 255),
            (255, 0, 255), (0, 255, 0), (255, 105, 180)
        ]
        for _ in range(num_particles):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 8)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            self.particles.append([
                [x, y],  # position
                [vx, vy], # velocity
                random.choice(self.colors),
                random.randint(60, 100) # lifetime
            ])

    def update(self):
        for p in self.particles:
            p[0][0] += p[1][0]
            p[0][1] += p[1][1]
            p[1][1] += 0.1  # Gravity
            p[3] -= 1
        self.particles = [p for p in self.particles if p[3] > 0]

class ColorPickingGame:
    def __init__(self):
        # --- Robot init ---
        print("Initializing Interbotix PX150...")
        self.robot = InterbotixManipulatorXS(
            robot_model="px150",
            group_name="arm",
            gripper_name="gripper"
        )

        # --- Camera init ---
        print("Initializing RealSense camera...")
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

        # Movement tuning
        self.joint_step = JOINT_STEP
        self.moving_time = MOVING_TIME

        # Color HSV ranges
        self.color_ranges = {
            'RED': [[(0, 120, 70), (8, 255, 255)], [(170, 120, 70), (180, 255, 255)]],
            'GREEN': [[(36, 50, 50), (85, 255, 255)]],
            'BLUE': [[(95, 80, 80), (125, 255, 255)]],
            'YELLOW': [[(15, 100, 100), (35, 255, 255)]],
            'ORANGE': [[(5, 100, 100), (15, 255, 255)]],
            'PURPLE': [[(125, 50, 50), (155, 255, 255)]],
        }

        self.colors = list(self.color_ranges.keys())
        self.current_color = None
        self.start_time = None
        self.round_times = []
        self.gripper_open = True
        self.detected_color = None
        self.status_message = "Press 'N' to start a new round"
        self.last_status_time = time.time()
        self.score = 0
        self.pulse_timer = 0
        self.screen_flash_color = None
        self.screen_flash_alpha = 0
        self.confetti_bursts = []

        # --- Pygame init ---
        pygame.init()
        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        # Get actual screen dimensions for dynamic UI layout
        self.WINDOW_WIDTH = self.screen.get_width()
        self.WINDOW_HEIGHT = self.screen.get_height()
        pygame.display.set_caption("PX150 Color Picking Game")
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
            print(f"Warning: couldn't go home: {e}")

        print("Game initialized! Pygame window should open.")

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

            # Get detected color
            roi = color_image[h//3:2*h//3, w//3:2*w//3]
            detected = self.get_dominant_color_with_depth(roi, depth_frame, w, h)
            self.detected_color = detected

            # Add text overlay on camera image
            if self.current_color:
                cv2.putText(color_image, f"TARGET: {self.current_color}", (10, 30),
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
        except Exception:
            return None

    # ----------------- Game logic -----------------
    def start_new_round(self):
        if not self.gripper_open:
            self._release_gripper()
        self.current_color = random.choice(self.colors)
        self.start_time = time.time()
        self.set_status(f"ROUND {len(self.round_times) + 1} - Pick up the {self.current_color} block!")

    def check_color_in_box(self):
        """Check if the target color is within the ROI square"""
        if not self.current_color:
            self.set_status("Start a round first by pressing 'N'.")
            return

        self.set_status("Checking color in box...")
        
        # Take multiple samples from the ROI
        samples = []
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
                detected = self.get_dominant_color_with_depth(roi, depth_frame, w, h)
                if detected:
                    samples.append(detected)
            except Exception:
                pass
            time.sleep(0.1)

        if not samples:
            self.set_status("No color detected in the box.")
            return

        most_common = max(set(samples), key=samples.count)
        count = samples.count(most_common)

        if most_common == self.current_color and count >= SAMPLES_MAJORITY:
            # Correct color detected - award points and complete round
            self.score += 1
            elapsed = time.time
