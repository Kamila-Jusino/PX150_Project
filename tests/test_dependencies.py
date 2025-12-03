#!/usr/bin/env python3
"""
Test script to verify all dependencies are installed correctly.
Run this before starting the AI-enhanced game.
"""

import sys

def test_import(module_name, package_name=None):
    """Test if a module can be imported."""
    try:
        if package_name:
            module = __import__(package_name)
        else:
            module = __import__(module_name)
        print(f"✓ {module_name} - OK")
        if hasattr(module, '__version__'):
            print(f"  Version: {module.__version__}")
        return True
    except ImportError as e:
        print(f"✗ {module_name} - FAILED")
        print(f"  Error: {e}")
        return False

def test_yolo():
    """Test YOLO model loading."""
    try:
        from ultralytics import YOLO
        print("Loading YOLO model (this may take a moment on first run)...")
        model = YOLO('yolov8n.pt')  # This will download if not present
        print("✓ YOLO model - OK (model loaded successfully)")
        return True
    except Exception as e:
        print(f"✗ YOLO model - FAILED")
        print(f"  Error: {e}")
        return False

def main():
    print("=" * 60)
    print("Testing AI-Enhanced Game Dependencies")
    print("=" * 60)
    print()
    
    all_ok = True
    
    # Core dependencies
    print("Core Dependencies:")
    print("-" * 60)
    all_ok &= test_import("cv2", "cv2")
    all_ok &= test_import("numpy", "numpy")
    all_ok &= test_import("pygame", "pygame")
    all_ok &= test_import("pyrealsense2", "pyrealsense2")
    print()
    
    # AI/ML dependencies
    print("AI/ML Dependencies:")
    print("-" * 60)
    all_ok &= test_import("torch", "torch")
    all_ok &= test_import("torchvision", "torchvision")
    all_ok &= test_import("ultralytics", "ultralytics")
    print()
    
    # Test YOLO model
    print("YOLO Model Test:")
    print("-" * 60)
    all_ok &= test_yolo()
    print()
    
    # Summary
    print("=" * 60)
    if all_ok:
        print("✓ ALL DEPENDENCIES INSTALLED CORRECTLY!")
        print("You can now run the AI-enhanced game with:")
        print("  ./run_game_ai.sh")
        print("  or")
        print("  python3 color_picking_game_ai.py")
    else:
        print("✗ SOME DEPENDENCIES ARE MISSING")
        print("Please install missing packages:")
        print("  pip3 install -r requirements.txt")
    print("=" * 60)
    
    return 0 if all_ok else 1

if __name__ == '__main__':
    sys.exit(main())

