# 🚀 Quick Start Installation Guide

## Prerequisites

### Windows
1. **Python 3.8+** - [Download](https://www.python.org/downloads/)
2. **Microsoft C++ Build Tools** - [Download](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
3. **CMake** - [Download](https://cmake.org/download/)

### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake libopenblas-dev liblapack-dev libjpeg-dev python3-dev
```

## Installation Steps

### 1. Clone/Navigate to Project
```bash
cd d:\FINAL_ATTENDANCE_SYSTEM_AI\ATTENDANCE_SYS_AI
```

### 2. Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Run Automated Setup
```bash
python setup.py
```

This will:
- ✅ Check Python version
- ✅ Install all dependencies from requirements.txt
- ✅ Download facial landmarks model (100MB)
- ✅ Create required directories
- ✅ Verify YOLO models (already present)
- ✅ Create .env.example file

### 4. Configure Environment Variables
```bash
# Copy example to actual .env file
cp .env.example .env

# Edit .env with your credentials
# For WhatsApp integration (optional for testing)
```

**Minimal .env for testing:**
```env
SECRET_KEY=my-secret-key-change-in-production
WHATSAPP_DRY_RUN=1
COORDINATOR_PHONE=+919876543210
```

### 5. Download Facial Landmarks Model (if not auto-downloaded)
```bash
# Windows PowerShell
Invoke-WebRequest -Uri "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2" -OutFile "shape_predictor_68_face_landmarks.dat.bz2"

# Then extract the .bz2 file
```

OR download manually:
- URL: https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2
- Extract to project root

### 6. Start the Application
```bash
python main.py
```

### 7. Access the System
- Open browser: **http://127.0.0.1:5000**
- Default admin credentials:
  - **Username:** `admin`
  - **Password:** `admin123`

---

## Manual Installation (If Automated Setup Fails)

### Install Dependencies Step-by-Step
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install core dependencies first
pip install numpy==1.24.3
pip install opencv-python==4.8.1.78
pip install Pillow==10.1.0

# Install dlib (takes longest - 5-10 minutes)
pip install dlib==19.24.2

# Install face recognition
pip install face-recognition==1.3.0

# Install remaining packages
pip install -r requirements.txt
```

---

## Troubleshooting

### dlib Installation Fails
**Windows:**
1. Install Visual Studio Build Tools with C++ support
2. Install CMake
3. Restart terminal and try again

**Linux:**
```bash
sudo apt-get install build-essential cmake
pip install dlib
```

### CMake Not Found
Download and install from: https://cmake.org/download/
Ensure CMake is added to PATH

### Face Recognition Import Error
```bash
pip install cmake
pip install dlib
pip install face-recognition
```

### YOLO Model Missing
The models are already in `models/` folder. If missing:
```bash
# Download YOLOv5 nano
mkdir models
cd models
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt
```

---

## Verify Installation

Run verification script:
```python
python -c "import cv2, dlib, face_recognition, flask, torch; print('All OK!')"
```

---

## Anti-Spoofing Features (Already Configured)

Your system has **MAXIMUM security** to prevent proxy attendance:

### 🛡️ Multi-Layer Protection
1. **Liveness Detection** - Requires real blink (not photos)
2. **Gaze Tracking** - Must look at camera (not videos)
3. **Texture Analysis** - Detects photo/screen quality
4. **Phone Detection** - YOLO AI blocks phones showing photos
5. **Anti-Spoofing Threshold** - Set to 0.50 (aggressive blocking)

### 🚨 What Gets Blocked
- ❌ Printed photos
- ❌ Photos on phone screens
- ❌ Photos on computer monitors
- ❌ Recorded videos
- ❌ Deep fakes
- ❌ Any non-live face

### ✅ What Works
- ✅ Real person in front of webcam
- ✅ Blinks naturally
- ✅ Looks at camera
- ✅ Proper lighting and distance

---

## Next Steps After Installation

1. **Enroll Students**
   - Login as admin
   - Click "Enroll Student"
   - Fill details and capture face
   - System will verify it's a real person

2. **Start Attendance System**
   - Click "Start System"
   - Students stand in front of camera
   - System automatically recognizes and marks attendance

3. **Monitor Dashboard**
   - View live recognition events
   - Check attendance statistics
   - Review suspicious activity logs

4. **Configure WhatsApp Alerts** (Optional)
   - Sign up for Twilio: https://www.twilio.com/
   - Get WhatsApp sandbox credentials
   - Update .env file with Twilio credentials
   - System will auto-send absence alerts

---

## Support

If you encounter issues:
1. Check this guide's Troubleshooting section
2. Verify all prerequisites are installed
3. Check Python version: `python --version` (must be 3.8+)
4. Check CMake: `cmake --version`
5. Run: `python setup.py` again

---

## Security Note

⚠️ **DO NOT use default admin password in production!**

Change the admin password after first login.
