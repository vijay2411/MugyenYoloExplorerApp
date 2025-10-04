"""
Video Processor Module
Handles video/webcam/image input processing
"""

import cv2
import numpy as np

class VideoProcessor:
    def __init__(self, source=0, is_image=False):
        """
        Initialize video processor
        
        Args:
            source: Video source (0 for webcam, path for video/image, or RTSP URL)
            is_image: Whether source is a static image
        """
        self.source = source
        self.is_image = is_image
        self.cap = None
        self.image = None
        self.total_frames = 0
        self.current_frame = 0
        
        self._initialize_source()
    
    def _initialize_source(self):
        """Initialize video source"""
        if self.is_image:
            # Load static image
            self.image = cv2.imread(self.source)
            if self.image is None:
                raise ValueError(f"Could not load image: {self.source}")
            self.total_frames = 1
        else:
            # Open video capture
            self.cap = cv2.VideoCapture(self.source)
            
            if not self.cap.isOpened():
                raise ValueError(f"Could not open video source: {self.source}")
            
            # Get video properties
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # For webcam, set some default properties if they're not available
            if self.total_frames <= 0:  # Live camera
                self.total_frames = 0
                self.fps = 30.0 if self.fps <= 0 else self.fps
    
    def read_frame(self):
        """
        Read next frame from source
        
        Returns:
            frame: Next frame or None if no more frames
        """
        if self.is_image:
            # Return static image
            return self.image.copy()
        
        if self.cap is None or not self.cap.isOpened():
            return None
        
        ret, frame = self.cap.read()
        
        if ret:
            self.current_frame += 1
            return frame
        else:
            return None
    
    def release(self):
        """Release video capture resources"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def get_progress(self):
        """Get current progress percentage"""
        if self.total_frames > 0:
            return (self.current_frame / self.total_frames) * 100
        return 0
    
    def get_info(self):
        """Get video information"""
        if self.is_image:
            h, w = self.image.shape[:2]
            return {
                'type': 'image',
                'width': w,
                'height': h,
                'total_frames': 1
            }
        else:
            return {
                'type': 'video',
                'width': self.width,
                'height': self.height,
                'fps': self.fps,
                'total_frames': self.total_frames
            }
    
    def seek_frame(self, frame_number):
        """Seek to specific frame"""
        if not self.is_image and self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            self.current_frame = frame_number
    
    def restart(self):
        """Restart video from beginning"""
        if not self.is_image and self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame = 0
