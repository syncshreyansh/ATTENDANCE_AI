#!/usr/bin/env python3
"""
System Health Check - Diagnose issues before running
"""
import os
import sys

def check(name, condition, fix_message=""):
    """Check a condition and print result"""
    status = "✓" if condition else "✗"
    color = "\033[92m" if condition else "\033[91m"
    reset = "\033[0m"
    
    print(f"{color}{status}{reset} {name}")
    if not condition and fix_message:
        print(f"  → Fix: {fix_message}")
    
    return condition

def main():
    print("=" * 60)
    print("SMART ATTENDANCE SYSTEM - HEALTH CHECK")
    print("=" * 60)
    
    all_ok = True
    
    # 1. Check Python version
    print("\n1. Python Environment:")
    python_ok = sys.version_info >= (3, 8)
    all_ok &= check("Python 3.8+", python_ok, "Install Python 3.8 or higher")
    
    # 2. Check required files
    print("\n2. Required Files:")
    predictor_ok = os.path.exists("shape_predictor_68_face_landmarks.dat")
    all_ok &= check("Facial landmark predictor", predictor_ok, 
                    "Run: python setup.py")
    
    yolo_ok = os.path.exists("models/yolov5n.pt")
    check("YOLO phone detector", yolo_ok, 
          "Optional but recommended. Run: python setup.py")
    
    # 3. Check Python dependencies
    print("\n3. Python Dependencies:")
    
    try:
        import flask
        all_ok &= check("Flask", True)
    except ImportError:
        all_ok &= check("Flask", False, "pip install Flask")
    
    try:
        import flask_socketio
        all_ok &= check("Flask-SocketIO", True)
    except ImportError:
        all_ok &= check("Flask-SocketIO", False, "pip install Flask-SocketIO")
    
    try:
        import cv2
        all_ok &= check("OpenCV", True)
    except ImportError:
        all_ok &= check("OpenCV", False, "pip install opencv-python")
    
    try:
        import dlib
        all_ok &= check("dlib", True)
    except ImportError:
        all_ok &= check("dlib", False, 
                       "pip install dlib (requires cmake)")
    
    try:
        import face_recognition
        all_ok &= check("face_recognition", True)
    except ImportError:
        all_ok &= check("face_recognition", False, 
                       "pip install face-recognition")
    
    try:
        import numpy
        all_ok &= check("numpy", True)
    except ImportError:
        all_ok &= check("numpy", False, "pip install numpy")
    
    # 4. Check database
    print("\n4. Database:")
    db_exists = os.path.exists("attendance.db")
    check("Database file", db_exists, 
          "Will be created on first run")
    
    # 5. Check configuration
    print("\n5. Configuration:")
    env_exists = os.path.exists(".env")
    check(".env file", env_exists, 
          "Optional. Create for WhatsApp integration")
    
    # 6. Test imports
    print("\n6. Module Imports:")
    
    try:
        from liveness_detection import LivenessDetector
        all_ok &= check("Liveness detection module", True)
    except Exception as e:
        all_ok &= check("Liveness detection module", False, str(e))
    
    try:
        from spoof_detection.ensemble_spoof import check as spoof_check
        all_ok &= check("Spoof detection module", True)
    except Exception as e:
        all_ok &= check("Spoof detection module", False, str(e))
    
    try:
        from face_recognition_service import FaceRecognitionService
        all_ok &= check("Face recognition service", True)
    except Exception as e:
        all_ok &= check("Face recognition service", False, str(e))
    
    # 7. Test camera (optional)
    print("\n7. Camera Test (optional):")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            check("Camera accessible", True)
        else:
            check("Camera accessible", False, 
                  "Camera might be in use or not connected")
    except Exception as e:
        check("Camera accessible", False, str(e))
    
    # Summary
    print("\n" + "=" * 60)
    if all_ok:
        print("✓✓✓ ALL CRITICAL CHECKS PASSED ✓✓✓")
        print("\nYou're ready to run the system!")
        print("\nNext steps:")
        print("1. python main.py")
        print("2. Open http://localhost:5000/")
        print("3. Login with admin/admin123")
    else:
        print("✗✗✗ SOME CHECKS FAILED ✗✗✗")
        print("\nPlease fix the issues above before running.")
        print("\nQuick fixes:")
        print("1. Run: pip install -r requirements.txt")
        print("2. Run: python setup.py")
        print("3. Run this script again: python health_check.py")
    print("=" * 60)
    
    return 0 if all_ok else 1

if __name__ == '__main__':
    sys.exit(main())