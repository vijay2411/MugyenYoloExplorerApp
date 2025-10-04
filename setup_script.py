"""
Setup script for YOLO Explorer
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="yolo-explorer",
    version="1.0.0",
    author="YOLO Explorer Team",
    description="A comprehensive object detection application using YOLO models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=[
        "ultralytics>=8.0.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "PyQt6>=6.5.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "Pillow>=10.0.0",
        "scipy>=1.11.0",
        "imageio>=2.31.0",
        "imageio-ffmpeg>=0.4.9",
        "psutil>=5.9.0",
    ],
    entry_points={
        "console_scripts": [
            "yolo-explorer=main:main",
        ],
    },
)
