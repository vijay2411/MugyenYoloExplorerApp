#!/bin/bash

# YOLO Explorer Quick Start Script
# This script sets up the environment and runs the application

echo "========================================="
echo "YOLO Explorer - Quick Start"
echo "========================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed!"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

echo "✓ Python found: $(python3 --version)"
echo ""

# Check if virtual environment exists
if [ ! -d "../YoloExplorer_venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv ../YoloExplorer_venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source ../YoloExplorer_venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Install/upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo "✓ Pip upgraded"
echo ""

# Install requirements
echo "Installing dependencies..."
echo "This may take a few minutes..."

# First install PyQt6 with proper backend support
echo "Installing PyQt6 with macOS support..."
pip install --upgrade PyQt6 PyQt6-Qt6 PyQt6-sip
echo "✓ PyQt6 installed"

# Then install other requirements
pip install -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Create necessary directories
echo "Creating project structure..."
mkdir -p gui core utils config
echo "✓ Directories created"
echo ""

# Clear Python cache
echo "Clearing Python cache..."
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
echo "✓ Python cache cleared"
echo ""

# Set environment variables for PyQt6 on macOS
export QT_MAC_WANTS_LAYER=1
export QT_AUTO_SCREEN_SCALE_FACTOR=1
export QT_LOGGING_RULES="qt.qpa.drawing.debug=false"
export QT_LOGGING_RULES="qt.qpa.plugin.debug=false"

# Run the application
echo "========================================="
echo "Starting YOLO Explorer..."
echo "========================================="
echo ""

python main.py

# Deactivate virtual environment on exit
deactivate