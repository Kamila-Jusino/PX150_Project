#!/usr/bin/env python3
"""
AI Vision System
Provides AI-driven object detection and classification using pretrained models.
Uses YOLO (ultralytics) or MobileNet for object detection and color classification.
"""

import cv2
import numpy as np
import pyrealsense2 as rs
from typing import List, Dict, Optional, Tuple

# Try to import YOLO (ultralytics) - primary choice
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Fallback to MobileNet if YOLO not available
MOBILENET_AVAILABLE = False
try:
    import torch
    import torchvision.transforms as transforms
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
    MOBILENET_AVAILABLE = True
except ImportError:
    pass

# ----- TUNABLE PARAMETERS -----
DEPTH_THRESHOLD_M = 0.7   # meters, only detect close objects
MIN_CONFIDENCE = 0.2       # minimum confidence for detection (lowered further for better detection)
YOLO_CONF_THRESHOLD = 0.1  # YOLO confidence threshold (very low to catch all objects)
YOLO_IMAGE_SIZE = 640     # Optimized for speed (was 1280, faster processing)
YOLO_IOU_THRESHOLD = 0.5   # IoU threshold for NMS (higher = keep more overlapping detections)
YOLO_MAX_DETECTIONS = 500  # Maximum detections per image (optimized for speed)
# Color mapping for classification
COLOR_NAMES = ['RED', 'GREEN', 'BLUE', 'YELLOW', 'ORANGE', 'PURPLE']
# Block-like classes that YOLO might detect (will be mapped to 'block')
BLOCK_LIKE_CLASSES = ['bottle', 'cup', 'book', 'remote', 'mouse', 'keyboard', 'cell phone', 'laptop']
# -------------------------------


class AIVisionSystem:
    """
    AI-driven vision system using pretrained models for object detection.
    Uses YOLO or MobileNet for detection and classification.
    """
    
    def __init__(self, pipeline: Optional[rs.pipeline] = None, model_type: str = 'yolo'):
        """
        Initialize AI vision system with pretrained model.
        
        Args:
            pipeline: Optional RealSense pipeline (if None, creates new one)
            model_type: 'yolo' or 'mobilenet'
        """
        self.pipeline = pipeline
        self.depth_scale = 0.001
        self.model = None
        self.model_type = None
        if MOBILENET_AVAILABLE:
            try:
                import torch
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            except ImportError:
                self.device = None
        else:
            self.device = None
        
        # Initialize AI model
        if model_type == 'yolo' and YOLO_AVAILABLE:
            print("Loading YOLOv8 pretrained model (optimized for speed)...")
            try:
                # Use YOLOv8n (nano) for faster processing
                self.model = YOLO('yolov8n.pt')  # Nano model for speed (object detection)
                self.model_type = 'yolo'
                print("✓ YOLO detection model loaded successfully (yolov8n for fast detection)")
            except Exception as e:
                print(f"Failed to load YOLO: {e}")
                self.model = None
            
            # Initialize color classification model (AI-driven color classification)
            if self.model:
                try:
                    # Try to load fine-tuned model first (if available)
                    try:
                        self.color_model = YOLO('yolo_colors.pt')
                        self.color_class_names = ['RED', 'GREEN', 'BLUE', 'YELLOW', 'ORANGE', 'PURPLE']
                        print("✓ Fine-tuned YOLO color classifier loaded")
                    except:
                        # Fallback to pretrained classification model
                        self.color_model = YOLO('yolov8n-cls.pt')
                        self.color_class_names = None  # Will use ImageNet classes (not ideal, but works)
                        print("✓ Pretrained YOLO color classifier loaded (yolov8n-cls)")
                        print("  Note: Consider fine-tuning for better color accuracy")
                except Exception as e:
                    print(f"Warning: Could not load color classifier: {e}")
                    self.color_model = None
            else:
                self.color_model = None
        
        if self.model is None and MOBILENET_AVAILABLE:
            print("Loading MobileNetV2 pretrained model...")
            try:
                self.model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
                self.model.eval()
                if self.device:
                    self.model = self.model.to(self.device)
                self.model_type = 'mobilenet'
                # Preprocessing for MobileNet
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                      std=[0.229, 0.224, 0.225])
                ])
                print("✓ MobileNet model loaded successfully")
            except Exception as e:
                print(f"Failed to load MobileNet: {e}")
                self.model = None
        
        if self.model is None:
            print("WARNING: No AI model available. YOLO-only mode requires YOLO.")
            self.model_type = None
        
        # Note: Color classifier is now handled by YOLO classification model above
        # Keep old color classifier initialization commented out for reference
        # if self.model_type == 'yolo' and self.model:
        #     self._init_color_classifier()
        
        # Reference color templates for AI-based classification (RGB values)
        # Updated to better match real-world colored blocks
        self.color_templates = {
            'RED': np.array([220, 30, 30]),      # Bright red
            'GREEN': np.array([30, 180, 30]),    # Bright green
            'BLUE': np.array([30, 60, 220]),     # Bright blue (more blue, less red/green)
            'YELLOW': np.array([255, 240, 0]),   # Bright yellow
            'ORANGE': np.array([255, 140, 0]),   # Bright orange
            'PURPLE': np.array([150, 0, 150]),   # Purple (balanced red/blue, no green)
        }
        
        # HSV color ranges for color extraction (display feature)
        self.color_ranges_hsv = {
            'RED': [[(0, 120, 70), (8, 255, 255)], [(170, 120, 70), (180, 255, 255)]],
            'GREEN': [[(36, 50, 50), (85, 255, 255)]],
            'BLUE': [[(95, 80, 80), (125, 255, 255)]],
            'YELLOW': [[(15, 100, 100), (35, 255, 255)]],
            'ORANGE': [[(5, 100, 100), (15, 255, 255)]],
            'PURPLE': [[(125, 50, 50), (155, 255, 255)]],
        }
        
        # Initialize pipeline if not provided
        if self.pipeline is None:
            self.pipeline = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.profile = self.pipeline.start(cfg)
            depth_sensor = self.profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale() if depth_sensor else 0.001
            
            # Warm-up frames
            for _ in range(20):
                self.pipeline.wait_for_frames()
        else:
            # Get depth scale from existing pipeline
            try:
                profile = self.pipeline.get_active_profile()
                depth_sensor = profile.get_device().first_depth_sensor()
                self.depth_scale = depth_sensor.get_depth_scale() if depth_sensor else 0.001
            except Exception:
                self.depth_scale = 0.001
    
    def _init_color_classifier(self):
        """Initialize color classifier using YOLO features."""
        # This will be used to extract features from YOLO's backbone
        try:
            import torch
            import torch.nn as nn
            
            # Create a simple color classifier head
            # Input: YOLO features (we'll extract from intermediate layers) + color stats
            # Output: 6 color classes
            self.color_classifier = None  # Will be created on-demand
            self.use_yolo_features = True
            print("✓ Color classifier initialized (using YOLO features)")
        except Exception as e:
            print(f"Warning: Could not initialize color classifier: {e}")
            self.use_yolo_features = False
    
    def _extract_yolo_features(self, roi: np.ndarray) -> Optional[np.ndarray]:
        """Extract features from YOLO's backbone/neck layers.
        Simplified to avoid hanging - uses YOLO's inference output.
        """
        if self.model_type != 'yolo' or not self.model:
            return None
        
        try:
            # Simplified approach: Use YOLO's inference result
            # This is faster and more reliable than accessing internal model structure
            # Run YOLO inference (this is fast)
            results = self.model(roi, conf=0.1, verbose=False, imgsz=640)
            
            # Extract features from the results
            # YOLO processes the image through its backbone, we can use the detection confidence
            # and bounding box features as a proxy for learned features
            if len(results) > 0 and len(results[0].boxes) > 0:
                # Use detection features as proxy
                boxes = results[0].boxes
                # Extract features from detections: confidence, box dimensions, position
                features = []
                for box in boxes[:3]:  # Use top 3 detections
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    w_box = x2 - x1
                    h_box = y2 - y1
                    features.extend([conf, w_box/640.0, h_box/640.0, (x1+x2)/1280.0, (y1+y2)/1280.0])
                
                # Pad or truncate to 64 features
                while len(features) < 64:
                    features.append(0.0)
                return np.array(features[:64])
            
            # If no detections, return None (will use color stats only)
            return None
                    
        except Exception as e:
            # Feature extraction failed, will use color stats only
            return None
    
    def _extract_color_statistics(self, roi: np.ndarray) -> np.ndarray:
        """Extract statistical color features from ROI for AI classification."""
        try:
            # Convert to RGB
            rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            pixels = rgb.reshape(-1, 3)
            
            # Mean RGB values (primary feature)
            mean_rgb = np.mean(pixels, axis=0)
            
            # Standard deviation (texture/variation)
            std_rgb = np.std(pixels, axis=0)
            
            # Median RGB (robust to outliers)
            median_rgb = np.median(pixels, axis=0)
            
            # Dominant color (histogram peak) - more robust
            hist_r = np.histogram(rgb[:,:,0].flatten(), bins=32, range=(0, 256))[0]
            hist_g = np.histogram(rgb[:,:,1].flatten(), bins=32, range=(0, 256))[0]
            hist_b = np.histogram(rgb[:,:,2].flatten(), bins=32, range=(0, 256))[0]
            peak_r = np.argmax(hist_r) * 8
            peak_g = np.argmax(hist_g) * 8
            peak_b = np.argmax(hist_b) * 8
            
            # Color moments (higher order statistics)
            # Skewness approximation
            skew_r = np.mean(((pixels[:, 0] - mean_rgb[0]) / (std_rgb[0] + 1e-6)) ** 3)
            skew_g = np.mean(((pixels[:, 1] - mean_rgb[1]) / (std_rgb[1] + 1e-6)) ** 3)
            skew_b = np.mean(((pixels[:, 2] - mean_rgb[2]) / (std_rgb[2] + 1e-6)) ** 3)
            
            # Combine features (15 total features)
            features = np.concatenate([
                mean_rgb / 255.0,        # Normalized mean (3 values)
                std_rgb / 255.0,        # Normalized std (3 values)
                median_rgb / 255.0,     # Normalized median (3 values)
                [peak_r / 255.0, peak_g / 255.0, peak_b / 255.0],  # Normalized peaks (3 values)
                [skew_r, skew_g, skew_b]  # Skewness (3 values)
            ])
            
            return features
        except Exception:
            return np.zeros(15)
    
    def _classify_color_rgb_fallback(self, roi: np.ndarray) -> Optional[Tuple[str, float]]:
        """
        Fallback method: Classify color using RGB template matching.
        Used when YOLO classification model is not available.
        """
        # Extract color statistics
        color_stats = self._extract_color_statistics(roi)
        # Use existing RGB template matching logic
        return self._classify_color_from_features(None, color_stats)
    
    def _classify_color_from_features(self, yolo_features: Optional[np.ndarray], 
                                      color_stats: np.ndarray) -> Optional[Tuple[str, float]]:
        """Classify color using RGB template matching (fallback method)."""
        try:
            # PRIMARY METHOD: Use color statistics (enhanced with YOLO if available)
            # Extract mean RGB (first 3 values of color_stats)
            mean_rgb = color_stats[:3] * 255.0  # Denormalize to 0-255 range
            
            # VALIDATION: Check if color is too desaturated (too gray/neutral)
            # Calculate saturation: how far from gray scale
            max_channel = np.max(mean_rgb)
            min_channel = np.min(mean_rgb)
            saturation = (max_channel - min_channel) / (max_channel + 1e-6)  # Avoid division by zero
            
            # VALIDATION: Reject gray/neutral colors
            # Calculate saturation: how far from gray scale
            max_channel = np.max(mean_rgb)
            min_channel = np.min(mean_rgb)
            saturation = (max_channel - min_channel) / (max_channel + 1e-6)  # Avoid division by zero
            channel_diff = max_channel - min_channel
            
            # Reject if too desaturated (less than 20% saturation = too gray)
            if saturation < 0.20:
                return None  # Too gray/neutral to classify
            
            # Also reject if all channels are very similar (gray)
            if channel_diff < 25:  # Less than 25 RGB units difference = too gray
                return None
            
            # Reject if brightness is too low (dark gray/black)
            if max_channel < 40:  # Too dark to reliably classify
                return None
            
            # Method: Multi-feature distance-based classification
            # Use peak (dominant) color as primary - more reliable than mean
            best_color = None
            best_score = float('inf')
            scores = {}
            
            # Get peak RGB (dominant color) - most reliable indicator
            if len(color_stats) >= 9:
                peak_rgb = color_stats[6:9] * 255.0
            else:
                peak_rgb = mean_rgb
            
            # Get median RGB (robust to outliers)
            if len(color_stats) >= 6:
                median_rgb = color_stats[3:6] * 255.0
            else:
                median_rgb = mean_rgb
            
            for color_name, template in self.color_templates.items():
                # Calculate multiple distance metrics
                
                # 1. Peak color distance (PRIMARY - most reliable for solid colored objects)
                peak_distance = np.linalg.norm(peak_rgb - template)
                
                # 2. Mean RGB distance (secondary)
                rgb_distance = np.linalg.norm(mean_rgb - template)
                
                # 3. Median RGB distance (robust to outliers)
                median_distance = np.linalg.norm(median_rgb - template)
                
                # 4. Channel-specific distance for colors that are sensitive to specific channels
                # For BLUE: emphasize blue channel, de-emphasize red/green
                # For PURPLE: emphasize red and blue, de-emphasize green
                if color_name == 'BLUE':
                    # Blue should have high blue, low red/green
                    channel_weights = np.array([2.0, 2.0, 1.0])  # Penalize red/green more
                elif color_name == 'PURPLE':
                    # Purple should have balanced red/blue, very low green
                    channel_weights = np.array([1.0, 3.0, 1.0])  # Heavily penalize green
                elif color_name == 'RED':
                    # Red should have high red, low green/blue
                    channel_weights = np.array([1.0, 2.0, 2.0])
                elif color_name == 'GREEN':
                    # Green should have high green, low red/blue
                    channel_weights = np.array([2.0, 1.0, 2.0])
                else:
                    channel_weights = np.array([1.0, 1.0, 1.0])
                
                weighted_distance = np.linalg.norm((peak_rgb - template) * channel_weights)
                
                # Combined score (peak color weighted most heavily)
                combined_score = (peak_distance * 0.5 +      # Primary: peak color
                                rgb_distance * 0.2 +         # Secondary: mean
                                median_distance * 0.15 +     # Tertiary: median
                                weighted_distance * 0.15)    # Channel-specific weighting
                
                scores[color_name] = combined_score
                
                if combined_score < best_score:
                    best_score = combined_score
                    best_color = color_name
            
            # Convert distance to confidence (inverse relationship)
            # Lower distance = higher confidence
            max_distance = 250.0  # Maximum expected distance in RGB space
            confidence = max(0.0, 1.0 - (best_score / max_distance))
            
            # Boost confidence if YOLO features are available (AI is helping)
            if yolo_features is not None and len(yolo_features) > 0:
                confidence *= 1.15  # 15% boost for using YOLO features
            
            confidence = min(1.0, confidence)
            
            # Additional validation: check if the best color is significantly better
            if len(scores) > 1:
                sorted_scores = sorted(scores.values())
                if len(sorted_scores) >= 2:
                    # If second best is very close, reduce confidence
                    score_diff = sorted_scores[1] - sorted_scores[0]
                    if score_diff < 20:  # Less than 20 RGB units difference
                        confidence *= 0.8  # Reduce confidence
            
            # Additional validation: check if best match is actually close enough
            # If the best score (distance) is too high, the color doesn't match well
            if best_score > 150:  # If distance > 150 RGB units, color doesn't match well
                return None  # Reject classification - color doesn't match any template well
            
            # Threshold: need reasonable confidence
            if confidence < 0.25:  # Increased threshold to require better matches
                return None
            
            return (best_color, confidence)
            
        except Exception as e:
            return None
    
    def _extract_color_hsv(self, roi: np.ndarray) -> Optional[str]:
        """
        Extract color using HSV (for display/info purposes).
        This is a secondary feature, not used for game logic.
        """
        try:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            best_color = None
            best_count = 0
            
            for cname, ranges in self.color_ranges_hsv.items():
                mask_total = np.zeros(hsv.shape[:2], dtype=np.uint8)
                for r in ranges:
                    mask_total = cv2.bitwise_or(mask_total, cv2.inRange(hsv, np.array(r[0]), np.array(r[1])))
                count = int(cv2.countNonZero(mask_total))
                if count > best_count:
                    best_count = count
                    best_color = cname
            
            # Minimum threshold for color detection
            if best_count < 100:  # Lower threshold for smaller ROIs
                return None
            
            return best_color
        except Exception:
            return None
    
    def detect_object_type(self, roi: np.ndarray) -> Optional[str]:
        """
        Detect object type using YOLO (AI-driven).
        This is the PRIMARY method for game logic.
        
        Args:
            roi: Region of interest (BGR image)
            
        Returns:
            Object class name (e.g., 'bottle', 'cup', 'block') or None
        """
        if self.model_type == 'yolo' and self.model:
            try:
                results = self.model(roi, conf=YOLO_CONF_THRESHOLD, verbose=False, imgsz=YOLO_IMAGE_SIZE)
                
                if results and len(results) > 0 and len(results[0].boxes) > 0:
                    boxes = results[0].boxes
                    best_box = boxes[0]  # Most confident detection
                    obj_conf = float(best_box.conf[0])
                    
                    if obj_conf < MIN_CONFIDENCE:
                        return None
                    
                    # Get YOLO's class prediction
                    try:
                        cls_id = int(best_box.cls[0].cpu().numpy())
                        cls_name = self.model.names[cls_id] if hasattr(self.model, 'names') else f"object_{cls_id}"
                    except:
                        cls_name = "object"
                    
                    # Get bounding box for block detection
                    x1, y1, x2, y2 = map(int, best_box.xyxy[0].cpu().numpy())
                    bbox = [x1, y1, x2, y2]
                    
                    # Map to 'block' if it's block-like
                    if self._is_likely_block(cls_name, bbox):
                        return 'block'
                    
                    return cls_name
            except Exception as e:
                print(f"Object type detection error: {e}")
        
        return None
    
    def _is_likely_block(self, cls_name: str, bbox: List[int]) -> bool:
        """
        Determine if detected object is likely a block.
        More aggressive: Map most small/medium objects to blocks.
        """
        # Check aspect ratio and size (blocks are roughly cubic)
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        if h == 0:
            return False
        
        aspect_ratio = w / h
        area = w * h
        
        # Very lenient criteria: accept almost any reasonable-sized object as a block
        # This will catch blocks, pens, cans, etc. - even small ones
        if 0.3 <= aspect_ratio <= 3.0 and 200 < area < 100000:  # Lowered min area, increased max
            return True
        
        return False
    
    def classify_color_ai(self, roi: np.ndarray) -> Optional[Tuple[str, float]]:
        """
        Classify color using YOLOv8 classification model (AI-driven).
        This is the PRIMARY method - uses neural network for color classification.
        
        Args:
            roi: Region of interest (BGR image)
            
        Returns:
            Tuple of (color_name, confidence) or None if classification fails
        """
        # PRIMARY: Use YOLO classification model (AI-driven)
        if hasattr(self, 'color_model') and self.color_model is not None:
            try:
                # Run YOLO classification inference
                # Note: YOLO classification models expect RGB images, but we pass BGR
                # YOLO will handle the conversion internally
                results = self.color_model(roi, verbose=False, imgsz=224)
                
                if results and len(results) > 0:
                    # Get top prediction
                    top1_idx = results[0].probs.top1
                    confidence = float(results[0].probs.top1conf.item())
                    
                    # If fine-tuned model with color classes
                    if hasattr(self, 'color_class_names') and self.color_class_names:
                        if top1_idx < len(self.color_class_names):
                            color_name = self.color_class_names[top1_idx]
                            # Only return if confidence is reasonable
                            if confidence >= 0.3:  # Minimum confidence threshold
                                return (color_name, confidence)
                    
                    # If using pretrained ImageNet model (not fine-tuned)
                    # This won't work well for colors, so fall through to RGB fallback
                    # In practice, you should fine-tune the model
                    else:
                        # Pretrained model doesn't know our colors
                        # Fall through to RGB fallback
                        pass
                        
            except Exception as e:
                print(f"YOLO color classification error: {e}")
                # Fall through to RGB fallback
        
        # FALLBACK: Use RGB template matching if classification model unavailable or fails
        return self._classify_color_rgb_fallback(roi)
    
    
    def scan_scene(self, num_samples: int = 5) -> List[Dict]:
        """
        Perform AI scan of the scene using pretrained model.
        This is the main AI vision function that returns detections
        in the format expected by AIGameState.update_from_ai_scan().
        
        Args:
            num_samples: Number of frames to sample for robust detection
            
        Returns:
            List of detected objects, each with:
                - 'color': Color name (str)
                - 'confidence': Detection confidence (float, 0-1)
                - 'position': Approximate position (x, y) in image
                - 'depth': Distance in meters (float)
                - 'bbox': Bounding box [x1, y1, x2, y2] (optional)
        """
        all_detections = []
        
        try:
            # Sample multiple frames for robustness (optimized for speed)
            max_samples = min(num_samples, 3)  # Reduced to 3 samples for faster scanning
            for sample_idx in range(max_samples):
                try:
                    frames = self.pipeline.wait_for_frames(timeout_ms=1000)  # Reduced timeout
                    color_frame = frames.get_color_frame()
                    depth_frame = frames.get_depth_frame()
                    
                    if not color_frame or not depth_frame:
                        continue
                    
                    color_image = np.asanyarray(color_frame.get_data())
                    depth_image = np.asanyarray(depth_frame.get_data())
                    h, w = color_image.shape[:2]
                    
                    detections = []
                    yolo_detections_count = 0
                    
                    if self.model_type == 'yolo' and self.model:
                        # PRIMARY: Use YOLO to scan table and identify objects + colors
                        # Balanced option: yolov8s, larger image size, lower confidence
                        try:
                            results = self.model(
                                color_image, 
                                conf=YOLO_CONF_THRESHOLD, 
                                verbose=False, 
                                imgsz=YOLO_IMAGE_SIZE,
                                iou=YOLO_IOU_THRESHOLD,
                                max_det=YOLO_MAX_DETECTIONS
                            )
                        except Exception as e:
                            print(f"YOLO inference error: {e}")
                            results = None
                        
                        if results and len(results) > 0 and len(results[0].boxes) > 0:
                            boxes = results[0].boxes
                            
                            for idx, box in enumerate(boxes):
                                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                                obj_conf = float(box.conf[0])
                                
                                # Get YOLO's class prediction (what object type YOLO thinks it is)
                                try:
                                    cls_id = int(box.cls[0].cpu().numpy())
                                    cls_name = self.model.names[cls_id] if hasattr(self.model, 'names') else f"object_{cls_id}"
                                except:
                                    cls_name = "object"
                                
                                if obj_conf < MIN_CONFIDENCE:
                                    continue
                                
                                # Extract object ROI
                                x1, y1 = max(0, x1), max(0, y1)
                                x2, y2 = min(w, x2), min(h, y2)
                                
                                if x2 > x1 and y2 > y1:
                                    object_roi = color_image[y1:y2, x1:x2]
                                    
                                    # Skip if ROI is too small (lowered threshold)
                                    if object_roi.size < 50:  # Lowered from 100 to catch smaller objects
                                        continue
                                    
                                    # PRIMARY: Object type from YOLO (AI-driven, for game logic)
                                    # Map to 'block' if it's block-like
                                    if self._is_likely_block(cls_name, [x1, y1, x2, y2]):
                                        cls_name = 'block'
                                    # Also map if it's a small/medium object regardless of class
                                    w, h = x2 - x1, y2 - y1
                                    area = w * h
                                    if 200 < area < 100000:  # More lenient: catch smaller and larger objects
                                        cls_name = 'block'
                                    
                                    # SECONDARY: Extract color using HSV (for display/info)
                                    color_name = self._extract_color_hsv(object_roi)
                                    
                                    # Check depth
                                    center_x = (x1 + x2) // 2
                                    center_y = (y1 + y2) // 2
                                    depth_value = depth_image[center_y, center_x]
                                    
                                    if depth_value > 0:
                                        depth_m = float(depth_value) * float(self.depth_scale)
                                        if depth_m <= DEPTH_THRESHOLD_M:
                                            # Create object with unique ID
                                            # PRIMARY: object class (AI-driven), SECONDARY: color (HSV, display)
                                            object_id = len(all_detections) + idx
                                            
                                            detections.append({
                                                'object_id': object_id,
                                                'class': cls_name,  # PRIMARY: Object type (AI-driven)
                                                'color': color_name,  # SECONDARY: Color (HSV, for display)
                                                'confidence': float(obj_conf),  # YOLO confidence
                                                'position': (center_x, center_y),
                                                'depth': float(depth_m),
                                                'bbox': [x1, y1, x2, y2]
                                            })
                        
                        all_detections.extend(detections)
                    
                    # YOLO-only mode: no other detection methods
                    # If YOLO is not available, return empty detections
                
                except Exception as e:
                    print(f"Error processing frame {sample_idx + 1}: {e}")
                    continue
            
            # Remove duplicates - keep highest confidence for similar objects
            final_detections = []
            
            # Sort by confidence (highest first)
            all_detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            for det in all_detections:
                obj_class = det.get('class', 'unknown')
                pos = det['position']
                
                # Check if we have this object already at a similar position
                found_similar = False
                for existing in final_detections:
                    # Check distance between positions and same object type
                    dist = np.sqrt((existing['position'][0] - pos[0])**2 + 
                                 (existing['position'][1] - pos[1])**2)
                    existing_class = existing.get('class', 'unknown')
                    if dist < 50 and obj_class == existing_class:  # Same position and same type = duplicate
                        found_similar = True
                        # Update if this one has higher confidence
                        if det['confidence'] > existing['confidence']:
                            final_detections.remove(existing)
                            final_detections.append(det)
                        break
                
                if not found_similar:
                    final_detections.append(det)
            
            # Print what YOLO detected (for debugging/feedback)
            if final_detections:
                print(f"  YOLO scan found {len(final_detections)} objects:")
                for i, det in enumerate(final_detections, 1):
                    # Print object type (PRIMARY) and color (SECONDARY, for display)
                    obj_type = det.get('class', 'unknown')
                    color_info = f" ({det['color']})" if det.get('color') else ""
                    print(f"    {i}. {obj_type}{color_info} (conf: {det['confidence']:.2f}, pos: {det['position']})")
            
            return final_detections
            
        except Exception as e:
            print(f"AI Vision scan error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def verify_gripper_object(self, roi: np.ndarray, expected_object_type: str) -> Tuple[bool, float]:
        """
        Verify if object in gripper matches expected object type using AI.
        This is called when player attempts to grab an object.
        
        Args:
            roi: Region of interest showing object in gripper
            expected_object_type: Expected object type (e.g., 'bottle', 'cup', 'block')
            
        Returns:
            Tuple of (is_correct, confidence)
        """
        if roi is None or roi.size == 0:
            return (False, 0.0)
        
        try:
            # PRIMARY: Use YOLO to detect object type (AI-driven)
            detected_type = self.detect_object_type(roi)
            
            if detected_type:
                is_correct = (detected_type.lower() == expected_object_type.lower())
                # Get confidence from YOLO detection
                results = self.model(roi, conf=YOLO_CONF_THRESHOLD, verbose=False, imgsz=YOLO_IMAGE_SIZE)
                if results and len(results) > 0 and len(results[0].boxes) > 0:
                    confidence = float(results[0].boxes[0].conf[0])
                    return (is_correct, confidence)
                return (is_correct, 0.5)  # Default confidence if can't get from YOLO
            
            return (False, 0.0)
        except Exception as e:
            print(f"Verification error: {e}")
            return (False, 0.0)
    
    def cleanup(self):
        """Cleanup resources."""
        if self.pipeline:
            try:
                self.pipeline.stop()
            except Exception:
                pass


def create_ai_vision_system(pipeline: Optional[rs.pipeline] = None, 
                            model_type: str = 'yolo') -> AIVisionSystem:
    """
    Factory function to create an AI vision system.
    
    Args:
        pipeline: Optional RealSense pipeline
        model_type: 'yolo' or 'mobilenet'
        
    Returns:
        AIVisionSystem instance
    """
    return AIVisionSystem(pipeline, model_type)


if __name__ == '__main__':
    # Test the AI vision system
    print("Testing AI Vision System...")
    vision = AIVisionSystem()
    
    try:
        print("Scanning scene with AI model...")
        detections = vision.scan_scene(num_samples=3)
        
        print(f"\nDetected {len(detections)} objects:")
        for i, det in enumerate(detections, 1):
            print(f"  {i}. {det['color']} - Confidence: {det['confidence']:.2f} - "
                  f"Position: {det['position']} - Depth: {det['depth']:.3f}m")
        
    finally:
        vision.cleanup()
        print("\nTest complete.")
