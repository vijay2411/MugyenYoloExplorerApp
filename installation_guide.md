# YOLO Explorer - Detailed Installation Guide

This guide provides step-by-step instructions for installing YOLO Explorer on different operating systems.

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Windows Installation](#windows-installation)
3. [macOS Installation](#macos-installation)
4. [Linux Installation](#linux-installation)
5. [GPU Setup (Optional)](#gpu-setup-optional)
6. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, macOS 10.14+, Ubuntu 18.04+
- **Python**: 3.8 or higher
- **RAM**: 4GB (8GB recommended)
- **Storage**: 2GB free space
- **Display**: 1280x720 minimum resolution

### Recommended Requirements
- **RAM**: 16GB
- **GPU**: NVIDIA GPU with 4GB+ VRAM (for GPU acceleration)
- **Storage**: 5GB free space (for models and cache)

---

## Windows Installation

### Method 1: Using Quick Start Script (Recommended)

1. **Download or Clone the Project**
   ```cmd
   # Create project directory
   mkdir C:\YOLOExplorer
   cd C:\YOLOExplorer
   ```

2. **Place all project files in the directory**

3. **Run Quick Start Script**
   ```cmd
   quick_start.bat
   ```
   
   This will:
   - Create virtual environment
   - Install all dependencies
   - Launch the application

### Method 2: Manual Installation

1. **Install Python**
   - Download from [python.org](https://www.python.org/downloads/)
   - **Important**: Check "Add Python to PATH" during installation
   - Verify installation:
     ```cmd
     python --version
     ```

2. **Create Project Directory**
   ```cmd
   mkdir C:\YOLOExplorer
   cd C:\YOLOExplorer
   ```

3. **Create Virtual Environment**
   ```cmd
   python -m venv venv
   ```

4. **Activate Virtual Environment**
   ```cmd
   venv\Scripts\activate
   ```

5. **Upgrade pip**
   ```cmd
   python -m pip install --upgrade pip
   ```

6. **Install Dependencies**
   ```cmd
   pip install -r requirements.txt
   ```

7. **Create Directory Structure**
   ```cmd
   mkdir gui core utils config
   ```

8. **Run Application**
   ```cmd
   python main.py
   ```

### Windows-Specific Notes
- If you encounter SSL certificate errors, use:
  ```cmd
  pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
  ```
- For PyQt6 installation issues, you may need Microsoft Visual C++ Redistributable

---

## macOS Installation

### Method 1: Using Quick Start Script (Recommended)

1. **Open Terminal**

2. **Navigate to Project Directory**
   ```bash
   cd ~/Downloads/yolo-explorer
   ```

3. **Make Script Executable**
   ```bash
   chmod +x quick_start.sh
   ```

4. **Run Script**
   ```bash
   ./quick_start.sh
   ```

### Method 2: Manual Installation

1. **Install Python**
   - macOS usually comes with Python, but install latest version:
   ```bash
   # Using Homebrew
   brew install python@3.11
   ```

2. **Verify Python Installation**
   ```bash
   python3 --version
   ```

3. **Create and Navigate to Project Directory**
   ```bash
   mkdir ~/YOLOExplorer
   cd ~/YOLOExplorer
   ```

4. **Create Virtual Environment**
   ```bash
   python3 -m venv venv
   ```

5. **Activate Virtual Environment**
   ```bash
   source venv/bin/activate
   ```

6. **Upgrade pip**
   ```bash
   pip install --upgrade pip
   ```

7. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

8. **Create Directory Structure**
   ```bash
   mkdir -p gui core utils config
   ```

9. **Run Application**
   ```bash
   python main.py
   ```

### macOS-Specific Notes
- You may need to install Xcode Command Line Tools:
  ```bash
  xcode-select --install
  ```
- For camera permissions, grant access in System Preferences > Security & Privacy > Camera

---

## Linux Installation

### Ubuntu/Debian

1. **Update System**
   ```bash
   sudo apt update
   sudo apt upgrade
   ```

2. **Install Python and Dependencies**
   ```bash
   sudo apt install python3 python3-pip python3-venv
   sudo apt install python3-pyqt6 libgl1-mesa-glx
   ```

3. **Create Project Directory**
   ```bash
   mkdir ~/yolo-explorer
   cd ~/yolo-explorer
   ```

4. **Create Virtual Environment**
   ```bash
   python3 -m venv venv
   ```

5. **Activate Virtual Environment**
   ```bash
   source venv/bin/activate
   ```

6. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

7. **Create Directory Structure**
   ```bash
   mkdir -p gui core utils config
   ```

8. **Run Application**
   ```bash
   python main.py
   ```

### Fedora/RHEL/CentOS

1. **Install Python**
   ```bash
   sudo dnf install python3 python3-pip python3-virtualenv
   ```

2. **Follow steps 3-8 from Ubuntu instructions**

### Arch Linux

1. **Install Python**
   ```bash
   sudo pacman -S python python-pip python-virtualenv
   ```

2. **Follow steps 3-8 from Ubuntu instructions**

---

## GPU Setup (Optional)

GPU acceleration significantly improves performance. Follow these steps to enable CUDA support.

### Prerequisites
- NVIDIA GPU (GTX 900 series or newer)
- NVIDIA Drivers installed

### Windows

1. **Install CUDA Toolkit**
   - Download from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
   - Install CUDA 11.8 or 12.1

2. **Install cuDNN**
   - Download from [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)
   - Extract and copy files to CUDA installation directory

3. **Install PyTorch with CUDA**
   ```cmd
   # For CUDA 11.8
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   
   # For CUDA 12.1
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Verify GPU Setup**
   ```python
   python -c "import torch; print(torch.cuda.is_available())"
   ```
   Should print `True`

### Linux

1. **Install NVIDIA Drivers**
   ```bash
   # Ubuntu
   sudo ubuntu-drivers autoinstall
   
   # Or manually
   sudo apt install nvidia-driver-535
   ```

2. **Install CUDA**
   ```bash
   # Add NVIDIA package repositories
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
   sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
   sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
   sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
   sudo apt update
   sudo apt install cuda
   ```

3. **Install PyTorch with CUDA**
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Verify**
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

### macOS
Note: CUDA is not available on macOS. Use CPU or MPS (Metal Performance Shaders) for M1/M2 Macs.

For M1/M2 Macs:
```bash
pip install torch torchvision
```

Verify MPS:
```python
python -c "import torch; print(torch.backends.mps.is_available())"
```

---

## Troubleshooting

### Common Issues

#### 1. "Python is not recognized"
**Windows:**
- Add Python to PATH: System Properties > Environment Variables > PATH
- Or reinstall Python with "Add to PATH" checked

**Linux/macOS:**
- Use `python3` instead of `python`

#### 2. "pip is not recognized"
```bash
# Windows
python -m pip install --upgrade pip

# Linux/macOS
python3 -m pip install --upgrade pip
```

#### 3. PyQt6 Installation Fails
**Windows:**
- Install Visual C++ Redistributable
- Download from [Microsoft](https://aka.ms/vs/17/release/vc_redist.x64.exe)

**Linux:**
```bash
sudo apt install python3-pyqt6
```

#### 4. OpenCV Import Error
```bash
# Install system dependencies (Linux)
sudo apt install libgl1-mesa-glx libglib2.0-0

# Reinstall opencv
pip uninstall opencv-python opencv-python-headless
pip install opencv-python
```

#### 5. CUDA Out of Memory
- Use smaller YOLO model (yolov8n instead of yolov8l)
- Reduce input resolution
- Switch to CPU mode

#### 6. Webcam Not Working
**Windows:**
- Check camera permissions in Windows Settings
- Close other applications using camera

**Linux:**
- Add user to video group:
  ```bash
  sudo usermod -a -G video $USER
  ```
- Reboot

**macOS:**
- Grant camera permission in System Preferences

#### 7. Slow Performance
- Enable GPU if available
- Use smaller model (yolov8n)
- Reduce input resolution
- Close background applications

#### 8. Model Download Fails
- Check internet connection
- Models are downloaded to `~/.cache/ultralytics/`
- Manually download from [Ultralytics Models](https://github.com/ultralytics/assets/releases)

---

## Verification

After installation, verify everything works:

```python
# test_installation.py
import sys
print("Python:", sys.version)

try:
    import cv2
    print("✓ OpenCV:", cv2.__version__)
except ImportError:
    print("✗ OpenCV not installed")

try:
    from PyQt6.QtWidgets import QApplication
    print("✓ PyQt6 installed")
except ImportError:
    print("✗ PyQt6 not installed")

try:
    import torch
    print("✓ PyTorch:", torch.__version__)
    print("  CUDA available:", torch.cuda.is_available())
except ImportError:
    print("✗ PyTorch not installed")

try:
    from ultralytics import YOLO
    print("✓ Ultralytics installed")
except ImportError:
    print("✗ Ultralytics not installed")
```

Run with:
```bash
python test_installation.py
```

---

## Getting Help

If you encounter issues not covered here:

1. Check the [GitHub Issues](https://github.com/your-repo/issues)
2. Review [Ultralytics Documentation](https://docs.ultralytics.com)
3. Check [PyQt6 Documentation](https://www.riverbankcomputing.com/static/Docs/PyQt6/)

---

**Installation complete! Run `python main.py` to start YOLO Explorer.**
