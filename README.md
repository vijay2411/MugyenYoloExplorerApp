# YOLO Explorer 🚀

A comprehensive Python-based object detection application using YOLO models with a modern PyQt6 GUI interface.

![YOLO Explorer](https://img.shields.io/badge/YOLO-Explorer-green)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyQt6](https://img.shields.io/badge/PyQt6-Latest-orange)

## Features ✨

### Core Features
- **Multiple YOLO Models Support**: YOLOv8, YOLOv11 (Nano, Small, Medium, Large, XLarge)
- **Multiple Input Sources**: 
  - Webcam (USB/Integrated)
  - Video Files (MP4, AVI, MOV, MKV, etc.)
  - Static Images (JPG, PNG, BMP, etc.)
  - RTSP Streams (IP Cameras)
- **Real-time Detection**: Live object detection with bounding boxes and confidence scores
- **Task Selection**: Detection, Segmentation, Pose Estimation, Tracking
- **Adjustable Confidence Threshold**: Fine-tune detection sensitivity
- **Performance Statistics**: 
  - FPS (Frames Per Second)
  - Inference Time
  - Object Count per Class
  - Average Confidence per Class

### GUI Features
- **Modern Interface**: Clean, intuitive PyQt6-based GUI
- **Live Video Feed**: Real-time video display with annotations
- **Control Panel**: Easy-to-use controls for all settings
- **Statistics Dashboard**: Live performance metrics
- **Model Configuration**: Simple model selection and device configuration
- **Progress Tracking**: Visual feedback for video processing

## Installation 🛠️

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) NVIDIA GPU with CUDA support for faster inference

### Step 1: Clone or Download

```bash
# If using git
git clone <repository-url>
cd yolo-explorer

# Or simply create a project directory and add the files
mkdir yolo-explorer
cd yolo-explorer
```

### Step 2: Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt
```

### Step 3: Install PyTorch (GPU Support - Optional)

For NVIDIA GPU support (recommended for better performance):

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU only (if no GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Project Structure 📁

```
yolo-explorer/
│
├── main.py                 # Main application entry point
├── requirements.txt        # Python dependencies
├── README.md              # This file
│
├── gui/                   # GUI components
│   ├── __init__.py
│   ├── main_window.py     # Main application window
│   ├── widgets.py         # Custom widgets (StatsWidget, etc.)
│   ├── model_config_dialog.py  # Model configuration dialog
│   └── rtsp_dialog.py     # RTSP stream configuration
│
├── core/                  # Core functionality
│   ├── __init__.py
│   ├── yolo_detector.py   # YOLO detection engine
│   └── video_processor.py # Video/image processing
│
├── utils/                 # Utility functions
│   ├── __init__.py
│   └── helpers.py         # Helper functions
│
└── config/                # Configuration
    ├── __init__.py
    └── settings.py        # Application settings
```

## Usage 🎯

### Running the Application

```bash
python main.py
```

### Quick Start Guide

1. **Load a Model**
   - Click "Configure Model..." button
   - Select your desired YOLO model (start with YOLOv8 Nano for quick testing)
   - Choose CPU or GPU (CUDA)
   - Click OK

2. **Select Input Source**
   - Choose source type from dropdown (Webcam, Video File, Image, RTSP)
   - Click "Select Source..." to configure
   - For Webcam: Default camera will be used
   - For Video/Image: Browse and select file
   - For RTSP: Enter stream URL

3. **Adjust Settings**
   - Use the Confidence slider to adjust detection threshold (0.0 - 1.0)
   - Higher values = fewer but more confident detections
   - Lower values = more detections but may include false positives

4. **Start Detection**
   - Click "Start Detection" button
   - Watch real-time detections in the video feed
   - Monitor statistics in the left panel
   - Click "Stop Detection" to pause

### Advanced Features

#### RTSP Stream Configuration
```
Format: rtsp://username:password@ip:port/stream
Example: rtsp://admin:password@192.168.1.64:554/stream1
```

#### Custom Model Loading
- Click "Load Custom Model..." to load your own trained YOLO models (.pt or .onnx files)

## Configuration ⚙️

### Model Settings
- **Confidence Threshold**: Minimum confidence for detections (default: 0.5)
- **IOU Threshold**: Intersection over Union for NMS (default: 0.45)
- **Device**: CPU or CUDA (GPU)

### Video Settings
- **Max FPS**: 30 (adjustable in settings.py)
- **Max Resolution**: 1920x1080 (adjustable in settings.py)

## Keyboard Shortcuts ⌨️

- `Ctrl+Q` - Quit application
- `Space` - Start/Stop detection
- `R` - Restart video

## Troubleshooting 🔧

### Common Issues

**1. Model Download Error**
```
Solution: YOLO models will be automatically downloaded on first use.
Ensure you have internet connection for first run.
```

**2. Webcam Not Detected**
```
Solution: Check camera permissions and ensure no other application is using it.
Try changing camera index in source selection.
```

**3. Low FPS / Slow Performance**
```
Solution:
- Use GPU if available (CUDA)
- Try smaller model (e.g., YOLOv8 Nano instead of Large)
- Reduce input resolution
- Close other applications
```

**4. CUDA Out of Memory**
```
Solution:
- Use smaller model
- Reduce batch size
- Switch to CPU mode
```

## Model Performance Comparison

| Model | Size | Speed (CPU) | Speed (GPU) | mAP |
|-------|------|-------------|-------------|-----|
| YOLOv8n | 3.2MB | ~45ms | ~1.5ms | 37.3 |
| YOLOv8s | 11.2MB | ~80ms | ~2.5ms | 44.9 |
| YOLOv8m | 25.9MB | ~150ms | ~4ms | 50.2 |
| YOLOv8l | 43.7MB | ~230ms | ~6ms | 52.9 |
| YOLOv8x | 68.2MB | ~380ms | ~9ms | 53.9 |

## Features Roadmap 🗺️

- [ ] Export detection results to JSON/CSV
- [ ] Video recording with annotations
- [ ] Custom class filtering
- [ ] Multi-camera support
- [ ] Batch processing mode
- [ ] Detection zones/ROI selection
- [ ] Alert system for specific detections
- [ ] Cloud model storage
- [ ] Performance profiling
- [ ] Dark mode theme

## Dependencies 📦

- **ultralytics**: YOLO implementation
- **opencv-python**: Video/image processing
- **PyQt6**: GUI framework
- **numpy**: Numerical operations
- **torch**: Deep learning framework
- **Pillow**: Image processing
- **psutil**: System monitoring (optional)

## Credits 👏

- YOLO models by [Ultralytics](https://github.com/ultralytics/ultralytics)
- GUI framework: PyQt6
- Computer Vision: OpenCV

## License 📄

This project is open source and available for educational and research purposes.

## Support 💬

For issues, questions, or contributions, please visit the project repository or contact the maintainer.

## Changelog 📝

### Version 1.0.0 (Initial Release)
- Complete YOLO detection application
- Multiple model support (YOLOv8, YOLOv11)
- Multiple input sources (Webcam, Video, Image, RTSP)
- Real-time statistics dashboard
- Modern PyQt6 GUI
- CPU and GPU support
- Adjustable confidence threshold
- Model configuration dialog

---

**Happy Detecting! 🎉**
