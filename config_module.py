"""
Configuration Settings for YOLO Explorer
"""

import os

# Application Settings
APP_NAME = "YOLO Explorer"
APP_VERSION = "1.0.0"

# Default Model Settings
DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_CONFIDENCE = 0.5
DEFAULT_IOU = 0.45
DEFAULT_DEVICE = "cpu"

# Video Settings
DEFAULT_FPS = 30
MAX_FRAME_WIDTH = 1920
MAX_FRAME_HEIGHT = 1080

# UI Settings
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 900
VIDEO_DISPLAY_WIDTH = 800
VIDEO_DISPLAY_HEIGHT = 600

# Available Models
AVAILABLE_MODELS = {
    'YOLOv8': [
        ('YOLOv8 Nano', 'yolov8n.pt'),
        ('YOLOv8 Small', 'yolov8s.pt'),
        ('YOLOv8 Medium', 'yolov8m.pt'),
        ('YOLOv8 Large', 'yolov8l.pt'),
        ('YOLOv8 XLarge', 'yolov8x.pt'),
    ],
    'YOLOv11': [
        ('YOLOv11 Nano', 'yolo11n.pt'),
        ('YOLOv11 Small', 'yolo11s.pt'),
        ('YOLOv11 Medium', 'yolo11m.pt'),
        ('YOLOv11 Large', 'yolo11l.pt'),
    ]
}

# Task Types
TASK_TYPES = ['Detection', 'Segmentation', 'Pose', 'Tracking', 'Classification']

# Input Source Types
SOURCE_TYPES = ['Webcam', 'Video File', 'Image', 'RTSP Stream', 'YouTube URL']

# Supported Video Formats
VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']

# Supported Image Formats
IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']

# Color Scheme
COLORS = {
    'primary': '#4CAF50',
    'secondary': '#2196F3',
    'danger': '#f44336',
    'warning': '#FF9800',
    'dark': '#212121',
    'light': '#F5F5F5'
}

# Paths
MODELS_DIR = os.path.join(os.path.expanduser('~'), '.yolo_explorer', 'models')
CACHE_DIR = os.path.join(os.path.expanduser('~'), '.yolo_explorer', 'cache')
OUTPUT_DIR = os.path.join(os.path.expanduser('~'), 'YOLO_Output')

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
