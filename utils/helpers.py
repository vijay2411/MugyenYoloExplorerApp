"""
Utility Helper Functions
"""

import cv2
import numpy as np
from typing import Tuple, List
import time

class FPSCounter:
    """Calculate FPS"""
    
    def __init__(self, buffer_size=30):
        self.buffer_size = buffer_size
        self.timestamps = []
    
    def update(self):
        """Update FPS counter"""
        current_time = time.time()
        self.timestamps.append(current_time)
        
        if len(self.timestamps) > self.buffer_size:
            self.timestamps.pop(0)
    
    def get_fps(self):
        """Get current FPS"""
        if len(self.timestamps) < 2:
            return 0.0
        
        time_diff = self.timestamps[-1] - self.timestamps[0]
        if time_diff > 0:
            return len(self.timestamps) / time_diff
        return 0.0


class ColorPalette:
    """Color palette for visualization"""
    
    def __init__(self, n_colors=80):
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(n_colors, 3), dtype=np.uint8)
    
    def get_color(self, class_id):
        """Get color for class ID"""
        return tuple(map(int, self.colors[class_id % len(self.colors)]))


def resize_frame(frame, max_width=1280, max_height=720):
    """
    Resize frame to fit within max dimensions while maintaining aspect ratio
    
    Args:
        frame: Input frame
        max_width: Maximum width
        max_height: Maximum height
        
    Returns:
        Resized frame
    """
    h, w = frame.shape[:2]
    
    # Calculate scaling factor
    scale = min(max_width / w, max_height / h, 1.0)
    
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return frame


def draw_fps(frame, fps, position=(10, 30)):
    """
    Draw FPS on frame
    
    Args:
        frame: Input frame
        fps: FPS value
        position: Text position (x, y)
        
    Returns:
        Frame with FPS drawn
    """
    text = f"FPS: {fps:.1f}"
    
    # Draw background
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, 
                  (position[0] - 5, position[1] - text_h - 5),
                  (position[0] + text_w + 5, position[1] + 5),
                  (0, 0, 0), -1)
    
    # Draw text
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)
    
    return frame


def get_video_properties(video_path):
    """
    Get video properties
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video properties
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None
    
    properties = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))
    }
    
    cap.release()
    return properties


def create_blank_frame(width=640, height=480, text="No Video Feed"):
    """
    Create blank frame with text
    
    Args:
        width: Frame width
        height: Frame height
        text: Text to display
        
    Returns:
        Blank frame with text
    """
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add text
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    x = (width - text_w) // 2
    y = (height + text_h) // 2
    
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2)
    
    return frame


def format_time(seconds):
    """Format seconds to HH:MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"
