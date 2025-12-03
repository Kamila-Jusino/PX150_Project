#!/usr/bin/env python3
"""
AI Game State Management
Manages game state based on AI model detections and drives game logic.
"""

import random
from collections import Counter

class AIGameState:
    """
    AI-driven game state management.
    Game logic depends on AI's object detections and classifications.
    """
    
    def __init__(self):
        """Initialize AI game state."""
        self.detected_objects = []  # AI-maintained object list
        self.available_object_types = set()  # Object types available based on AI detections (PRIMARY)
        self.available_colors = set()  # Colors available based on AI detections (SECONDARY, for display)
        self.removed_objects = []  # Track removed objects
        # self.difficulty = 'normal'  # DISABLED: Difficulty feature removed
    
    def update_from_ai_scan(self, ai_detections):
        """
        Update game state from AI scan results.
        
        This is AI-driven: game state depends entirely on AI model outputs.
        
        Args:
            ai_detections: List of detected objects from AI vision system
        """
        self.detected_objects = ai_detections.copy()
        
        # AI-driven: Extract available object types (PRIMARY - for game logic)
        self.available_object_types = {obj['class'] for obj in self.detected_objects if obj.get('class')}
        
        # Extract available colors (SECONDARY - for display/info)
        self.available_colors = {obj['color'] for obj in self.detected_objects if obj.get('color')}
        
        return len(self.detected_objects)
    
    def ai_select_target(self, difficulty='normal'):
        """
        AI-driven target selection based on object types (not colors).
        
        Game logic depends on AI's object inventory to select targets.
        This function makes decisions based on AI-maintained state.
        
        Args:
            difficulty: 'easy', 'normal', 'hard' (IGNORED - always uses random selection)
            
        Returns:
            Selected object type (e.g., 'bottle', 'cup', 'block') or None
        """
        # self.difficulty = difficulty  # DISABLED: Difficulty feature removed
        
        if not self.available_object_types:
            return None
        
        # AI-driven selection logic: Always random selection (difficulty disabled)
        # object_counts = Counter(obj['class'] for obj in self.detected_objects if obj.get('class'))
        
        # if difficulty == 'hard':
        #     # Select least common object type (more challenging)
        #     if object_counts:
        #         return min(object_counts, key=object_counts.get)
        # elif difficulty == 'easy':
        #     # Select most common object type (easier)
        #     if object_counts:
        #         return max(object_counts, key=object_counts.get)
        # else:  # normal
        # Random selection from AI-detected object types (always used now)
        return random.choice(list(self.available_object_types))
    
    def remove_object_by_type(self, object_type):
        """
        Remove object from AI-maintained inventory after successful grab.
        
        This updates the AI-driven game state.
        
        Args:
            object_type: Object type to remove (e.g., 'bottle', 'cup', 'block')
        """
        # Find and remove first object of this type
        for i, obj in enumerate(self.detected_objects):
            if obj.get('class') == object_type:
                removed = self.detected_objects.pop(i)
                self.removed_objects.append(removed)
                break
        
        # Update available object types based on remaining AI detections
        self.available_object_types = {obj['class'] for obj in self.detected_objects if obj.get('class')}
        # Also update colors (for display)
        self.available_colors = {obj['color'] for obj in self.detected_objects if obj.get('color')}
    
    def remove_object_by_color(self, color):
        """
        Remove object by color (legacy method, kept for compatibility).
        Prefer remove_object_by_type() for object-based game logic.
        """
        # Find and remove first object of this color
        for i, obj in enumerate(self.detected_objects):
            if obj.get('color') == color:
                removed = self.detected_objects.pop(i)
                self.removed_objects.append(removed)
                break
        
        # Update available colors and object types
        self.available_colors = {obj['color'] for obj in self.detected_objects if obj.get('color')}
        self.available_object_types = {obj['class'] for obj in self.detected_objects if obj.get('class')}
    
    def get_object_count_by_color(self, color):
        """
        Get count of objects of a specific color from AI detections.
        
        Args:
            color: Color to count
            
        Returns:
            Number of objects of that color
        """
        return sum(1 for obj in self.detected_objects if obj.get('color') == color)
    
    def has_color_available(self, color):
        """
        Check if color is available based on AI detections.
        
        Args:
            color: Color to check
            
        Returns:
            True if color is available
        """
        return color in self.available_colors
    
    def get_detection_summary(self):
        """
        Get summary of AI detections for display.
        
        Returns:
            Dictionary with detection statistics
        """
        object_counts = Counter(obj['class'] for obj in self.detected_objects if obj.get('class'))
        color_counts = Counter(obj['color'] for obj in self.detected_objects if obj.get('color'))
        return {
            'total_objects': len(self.detected_objects),
            'available_object_types': list(self.available_object_types),  # PRIMARY
            'available_colors': list(self.available_colors),  # SECONDARY (for display)
            'object_type_counts': dict(object_counts),  # PRIMARY
            'color_counts': dict(color_counts),  # SECONDARY (for display)
            'avg_confidence': sum(obj.get('confidence', 0) for obj in self.detected_objects) / max(len(self.detected_objects), 1)
        }

