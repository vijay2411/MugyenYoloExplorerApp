# YOLO Explorer 🚀

A comprehensive Python-based object detection application using YOLO models with a modern PyQt6 GUI interface. Features real-time detection, multiple input sources, responsive design, and cross-platform GPU support.

![YOLO Explorer](https://img.shields.io/badge/YOLO-Explorer-green)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyQt6](https://img.shields.io/badge/PyQt6-Latest-orange)
![Platform](https://img.shields.io/badge/Platform-Cross--Platform-lightgrey)

## ✨ Features

### 🎯 Core Detection Features
- **Multiple YOLO Models**: YOLOv8, YOLOv11 (Nano, Small, Medium, Large, XLarge)
- **Task Selection**: Detection, Segmentation, Pose Estimation, Tracking
- **Real-time Inference**: Live object detection with bounding boxes and confidence scores
- **Adjustable Confidence**: Fine-tune detection sensitivity (0.0 - 1.0)
- **Performance Statistics**: 
  - FPS (Frames Per Second)
  - Inference Time
  - Object Count per Class
  - Average Confidence per Class

### 📹 Input Sources
- **Webcam Support**: Multiple camera detection (USB/Integrated)
- **Video Files**: MP4, AVI, MOV, MKV, etc.
- **Static Images**: JPG, PNG, BMP, etc.
- **RTSP Streams**: IP Cameras with authentication
- **Auto Camera Detection**: Automatically finds and lists all available cameras

### 🖥️ GUI Features
- **Responsive Design**: Automatically adapts to screen size
- **Modern Interface**: Clean, intuitive PyQt6-based GUI
- **Live Video Feed**: Real-time video display with annotations
- **Control Panel**: Easy-to-use controls for all settings
- **Statistics Dashboard**: Live performance metrics
- **System Monitor**: CPU, Memory, GPU usage monitoring
- **Model Configuration**: Simple model selection and device configuration

### 🔧 Advanced Features
- **Cross-Platform GPU Support**: 
  - CUDA (NVIDIA GPUs)
  - MPS (Apple Silicon Macs)
  - NPU (NVIDIA Jetson)
- **Dynamic System Monitoring**: Real-time CPU, Memory, GPU usage
- **Multi-Camera Support**: Detect and switch between multiple cameras
- **Auto Model Loading**: Default YOLOv8n model preloaded on startup
- **Auto Video Start**: Video feed starts immediately when source is selected

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) GPU support for faster inference

### Installation & Running

**One-Command Setup:**
```bash
bash quick_start_script.sh
```

This script will:
- Create and activate virtual environment
- Install all dependencies
- Clear Python cache
- Set up macOS environment variables
- Launch the application

### Manual Installation (Alternative)

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run application
python main.py
```

## 📁 Project Structure

```
YoloExplorer/
│
├── main.py                    # Main application entry point
├── requirements.txt           # Python dependencies
├── quick_start_script.sh     # One-command setup script
├── quick_start_windows.txt   # Windows setup instructions
├── installation_guide.md     # Detailed installation guide
├── setup_script.py           # Setup automation script
├── advanced_features.py      # Advanced features (tracking, zones, etc.)
├── config_module.py          # Configuration module
│
├── core/                     # Core functionality
│   ├── __init__.py
│   ├── yolo_detector.py      # YOLO detection engine
│   └── video_processor.py    # Video/image processing
│
├── gui/                      # GUI components
│   ├── __init__.py
│   ├── main_window.py        # Main application window
│   ├── widgets.py            # Custom widgets (StatsWidget, SystemMonitor)
│   ├── model_config_dialog.py # Model configuration dialog
│   └── rtsp_dialog.py        # RTSP stream configuration
│
├── utils/                    # Utility functions
│   ├── __init__.py
│   └── helpers.py            # Helper functions
│
├── config/                   # Configuration
│   └── __init__.py
│
└── YoloExplorer_venv/        # Virtual environment (created by script)
```

## 🎮 Usage Guide

### 1. **Start the Application**
```bash
bash quick_start_script.sh
```

### 2. **Select Input Source**
- **Webcam**: Choose from detected cameras (Camera 0, Camera 1, etc.)
- **Video File**: Browse and select video file
- **Image**: Select static image for detection
- **RTSP Stream**: Enter stream URL (e.g., `rtsp://admin:password@192.168.1.64:554/stream1`)

### 3. **Configure Model (Optional)**
- Default YOLOv8n model is preloaded
- Click "Configure Model..." to change model or device
- Available devices: CPU, CUDA, MPS (Mac), NPU (Jetson)

### 4. **Start Detection**
- Video feed starts automatically when source is selected
- Click "Start YOLO Detection" to begin inference
- Adjust confidence slider for detection sensitivity
- Monitor real-time statistics in the left panel

### 5. **Monitor Performance**
- **FPS**: Current frames per second
- **Inference Time**: Time per detection
- **Object Count**: Number of detected objects per class
- **System Stats**: CPU, Memory, GPU usage

## ⚙️ Configuration

### Model Settings
- **Confidence Threshold**: Minimum confidence for detections (default: 0.5)
- **IOU Threshold**: Intersection over Union for NMS (default: 0.45)
- **Device**: CPU, CUDA, MPS, or NPU

### System Requirements
- **Minimum**: 4GB RAM, Python 3.8+
- **Recommended**: 8GB RAM, GPU support
- **GPU Support**: CUDA 11.8+, MPS (Apple Silicon), NPU (Jetson)

## 🔧 Troubleshooting

### Common Issues

**1. PyQt6 Platform Plugin Error (macOS)**
```bash
# Solution: Use the quick_start_script.sh which sets proper environment variables
bash quick_start_script.sh
```

**2. Camera Not Detected**
```bash
# Solution: Check camera permissions and refresh
# Click "Refresh Cameras" button in the GUI
```

**3. Model Loading Issues**
```bash
# Solution: Ensure internet connection for first-time model download
# Models are cached locally after first download
```

**4. Low Performance**
```bash
# Solutions:
# - Use GPU if available (CUDA/MPS/NPU)
# - Try smaller model (YOLOv8n instead of YOLOv8x)
# - Reduce input resolution
# - Close other applications
```

**5. CUDA Out of Memory**
```bash
# Solutions:
# - Use smaller model
# - Switch to CPU mode
# - Reduce batch size
```

## 🎯 Advanced Features

### RTSP Stream Configuration
```
Format: rtsp://username:password@ip:port/stream
Example: rtsp://admin:password@192.168.1.64:554/stream1
```

### GPU Support
- **CUDA**: NVIDIA GPUs with CUDA 11.8+
- **MPS**: Apple Silicon Macs (M1/M2/M3)
- **NPU**: NVIDIA Jetson (Orin, Xavier, Nano)

### System Monitoring
- Real-time CPU usage
- Memory consumption
- GPU utilization (CUDA/MPS/NPU)
- Process monitoring

## 📊 Model Performance

| Model | Size | Speed (CPU) | Speed (GPU) | mAP | Use Case |
|-------|------|-------------|-------------|-----|----------|
| YOLOv8n | 3.2MB | ~45ms | ~1.5ms | 37.3 | Fast, lightweight |
| YOLOv8s | 11.2MB | ~80ms | ~2.5ms | 44.9 | Balanced |
| YOLOv8m | 25.9MB | ~150ms | ~4ms | 50.2 | Good accuracy |
| YOLOv8l | 43.7MB | ~230ms | ~6ms | 52.9 | High accuracy |
| YOLOv8x | 68.2MB | ~380ms | ~9ms | 53.9 | Best accuracy |

## 🛠️ Development

### Dependencies
- **ultralytics**: YOLO implementation
- **opencv-python**: Video/image processing
- **PyQt6**: GUI framework
- **numpy**: Numerical operations
- **torch**: Deep learning framework
- **Pillow**: Image processing
- **psutil**: System monitoring

### Keyboard Shortcuts
- `Ctrl+Q` - Quit application
- `Space` - Start/Stop detection
- `R` - Restart video

## 📝 Changelog

### Version 2.0.0 (Current)
- ✅ Responsive GUI design
- ✅ Multi-camera detection
- ✅ Cross-platform GPU support (CUDA/MPS/NPU)
- ✅ Real-time system monitoring
- ✅ Auto video start
- ✅ Default model preloading
- ✅ Dynamic GPU usage monitoring
- ✅ Fixed PyQt6 warnings
- ✅ Enhanced error handling

### Version 1.0.0 (Initial)
- ✅ Basic YOLO detection
- ✅ Multiple input sources
- ✅ PyQt6 GUI
- ✅ CPU/GPU support

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available for educational and research purposes.

## 🆘 Support

For issues, questions, or contributions:
- Check the troubleshooting section above
- Review the installation guide
- Ensure all dependencies are properly installed
- Use the quick_start_script.sh for automatic setup

---

**Happy Detecting! 🎉**

*YOLO Explorer - Making Computer Vision Accessible*