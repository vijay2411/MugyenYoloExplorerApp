"""Core functionality module"""

from .yolo_detector import YOLODetector
from .video_processor import VideoProcessor

__all__ = [
    'YOLODetector',
    'VideoProcessor'
]