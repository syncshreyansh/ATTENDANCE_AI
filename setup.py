#!/usr/bin/env python3
"""
Setup script to download required model files
"""
import os
import urllib.request
import bz2
import shutil

def download_file(url, filename):
    """Download file with progress"""
    print(f"Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"✓ Downloaded {filename}")
        return True
    except Exception as e:
        print(f"✗ Failed to download {filename}: {e}")
        return False

def extract_bz2(source, dest):
    """Extract bz2 file"""
    print(f"Extracting {source}...")
    try:
        with bz2.BZ2File(source, 'rb') as f_in:
            with open(dest, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(source)
        print(f"✓ Extracted to {dest}")
        return True
    except Exception as e:
        print(f"✗ Failed to extract: {e}")
        return False

def main():
    print("=" * 60)
    print("SMART ATTENDANCE SYSTEM - SETUP")
    print("=" * 60)
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Download shape predictor for facial landmarks
    shape_predictor = "shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(shape_predictor):
        print("\n1. Downloading facial landmark predictor...")
        url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        compressed = shape_predictor + ".bz2"
        
        if download_file(url, compressed):
            extract_bz2(compressed, shape_predictor)
    else:
        print(f"✓ {shape_predictor} already exists")
    
    # Download YOLO model for phone detection (optional but recommended)
    yolo_model = "models/yolov5n.pt"
    if not os.path.exists(yolo_model):
        print("\n2. Downloading YOLO model for phone detection...")
        print("   (This is optional but highly recommended for spoof detection)")
        url = "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt"
        
        download_file(url, yolo_model)
    else:
        print(f"✓ {yolo_model} already exists")
    
    print("\n" + "=" * 60)
    print("SETUP COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Install Python dependencies:")
    print("   pip install -r requirements.txt")
    print("\n2. Configure WhatsApp (optional):")
    print("   Create .env file with:")
    print("   WHATSAPP_TOKEN='your_token'")
    print("   WHATSAPP_PHONE_ID='your_phone_id'")
    print("   WHATSAPP_DRY_RUN=1  # Set to 0 to enable actual sending")
    print("\n3. Run the system:")
    print("   python main.py")
    print("\n4. Access dashboards:")
    print("   Admin: http://localhost:5000/")
    print("   Student: http://localhost:5000/student-dashboard")
    print("\nDefault login:")
    print("   Username: admin")
    print("   Password: admin123")
    print("=" * 60)

if __name__ == '__main__':
    main()