# 🔮 Smart Attendance System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3670A0?logo=python&logoColor=ffdd54&style=for-the-badge)
![Flask](https://img.shields.io/badge/Flask-3.0-000000?logo=flask&logoColor=white&style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8-5C3EE8?logo=opencv&logoColor=white&style=for-the-badge)
![SQLite](https://img.shields.io/badge/SQLite-003B57?logo=sqlite&logoColor=white&style=for-the-badge)

**An AI-Powered Face Recognition Attendance System with Advanced Anti-Spoofing, Real-Time Dashboards, and Automated WhatsApp Alerts**

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [Architecture](#-architecture) • [API Documentation](#-api-documentation)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Technology Stack](#-technology-stack)
- [System Architecture](#-system-architecture)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Running the Application](#-running-the-application)
- [Usage Guide](#-usage-guide)
- [API Documentation](#-api-documentation)
- [Project Structure](#-project-structure)
- [Security Features](#-security-features)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

The **Smart Attendance System** is a production-ready, full-stack application that revolutionizes attendance tracking using AI-powered facial recognition. Built with a focus on security and user experience, it prevents proxy attendance through advanced anti-spoofing techniques and provides real-time insights through interactive dashboards.

### Why This System?

- ✅ **Eliminate Proxy Attendance**: Multi-layer verification with liveness detection
- ✅ **Save Time**: Automated attendance marking in seconds
- ✅ **Real-Time Insights**: Live dashboards with instant statistics
- ✅ **Parental Engagement**: Automated WhatsApp alerts for chronic absenteeism
- ✅ **Gamification**: Point system and leaderboards to motivate students

---

## ✨ Key Features

### 🔐 Advanced Security & Anti-Spoofing

- **Multi-Layer Liveness Detection**
  - Eye Aspect Ratio (EAR) for blink detection
  - Head pose estimation for gaze tracking
  - Texture analysis for photo detection
  - FFT-based moiré pattern detection for screen attacks

- **Phone/Screen Detection**
  - YOLOv5-based object detection for devices
  - Edge detection fallback mechanism
  - Real-time blocking of spoofing attempts

- **Duplicate Face Prevention**
  - SHA-256 hash-based face verification
  - Prevents multiple enrollments of same person

### 📊 Dual Dashboard System

#### Admin Dashboard (`/admin-dashboard`)
- **Real-Time Camera Feed**: Live face recognition with status indicators
- **Instant Statistics**: Present/Absent counts, attendance rate, total students
- **Activity Logs**: Chronological event tracking with timestamps
- **Student Management**: Enrollment, search, and detailed student stats
- **Multi-Shot Enrollment**: Capture 7 frames for better accuracy
- **Quality Assessment**: Real-time feedback on image quality during enrollment

#### Student Dashboard (`/student-dashboard`)
- **Personal Profile**: Photo, class info, and points earned
- **Attendance Statistics**: Days present/absent, attendance rate, current streak
- **Trend Visualization**: 30-day attendance rate chart
- **Monthly Calendar**: Visual representation with present/absent dates
- **Class Leaderboard**: Rankings with profile pictures and medals

### 📱 WhatsApp Integration

- **Automated Absence Alerts**: Notifications after 3 consecutive absences
- **Dual Notifications**: Sent to both parents and class coordinators
- **OTP-Based Password Reset**: Secure password recovery via WhatsApp
- **Dry Run Mode**: Test notifications without actual sending

### 🎮 Gamification Features

- **Points System**: Earn points for on-time attendance and liveness checks
- **Class Leaderboard**: Compete with classmates
- **Achievement Badges**: Gold/Silver/Bronze medals for top performers
- **Streak Tracking**: Monitor consecutive attendance days

---

## 🛠️ Technology Stack

### Backend
- **Framework**: Flask 3.0 (Python web framework)
- **Real-Time Communication**: Flask-SocketIO 5.x (WebSocket support)
- **Database**: SQLAlchemy with SQLite (easily upgradeable to PostgreSQL)
- **Authentication**: JWT (JSON Web Tokens) with bcrypt password hashing
- **Task Scheduling**: APScheduler (for daily automated tasks)

### AI & Computer Vision
- **Face Recognition**: `face_recognition` library (built on dlib)
- **Face Detection**: dlib's HOG-based detector + CNN fallback
- **Image Processing**: OpenCV 4.8 (cv2)
- **Liveness Detection**: Custom algorithms using dlib's 68-point facial landmarks
- **Anti-Spoofing**: 
  - YOLOv5 nano for phone/screen detection
  - Custom texture and FFT analysis
  - Optional: ONNX CNN model for advanced spoof detection

### Frontend
- **HTML5/CSS3**: Modern, responsive UI
- **JavaScript (Vanilla)**: Client-side logic and real-time updates
- **Chart.js**: Interactive attendance trend charts
- **Font Awesome 6**: Icon library
- **Google Fonts**: Montserrat & JetBrains Mono

### External Services
- **WhatsApp Business API**: Automated notifications
- **Camera API**: WebRTC for live video streaming

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Client Browser                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Admin        │  │ Student      │  │ Login        │     │
│  │ Dashboard    │  │ Dashboard    │  │ Page         │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
│         │                 │                  │              │
└─────────┼─────────────────┼──────────────────┼──────────────┘
          │                 │                  │
          │   WebSocket     │   HTTP/REST      │   JWT Auth
          │   (Socket.IO)   │                  │
          ▼                 ▼                  ▼
┌─────────────────────────────────────────────────────────────┐
│                    Flask Application                         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Routes & API Endpoints                               │  │
│  │  • routes.py (Admin)                                  │  │
│  │  • student_routes.py (Student)                        │  │
│  │  • auth_routes.py (Authentication)                    │  │
│  └────────────────┬─────────────────────────────────────┘  │
│                   │                                          │
│  ┌────────────────▼─────────────────────────────────────┐  │
│  │  Core Services                                        │  │
│  │  • face_recognition_service.py (AI Recognition)      │  │
│  │  • liveness_detection.py (Anti-Spoofing)             │  │
│  │  • attendance_service.py (Business Logic)            │  │
│  │  • whatsapp_service.py (Notifications)               │  │
│  │  • auth_service.py (JWT & Security)                  │  │
│  └────────────────┬─────────────────────────────────────┘  │
│                   │                                          │
│  ┌────────────────▼─────────────────────────────────────┐  │
│  │  Data Layer (SQLAlchemy ORM)                         │  │
│  │  • models.py (Student, Attendance, User, etc.)       │  │
│  └────────────────┬─────────────────────────────────────┘  │
└───────────────────┼──────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│                  SQLite Database                             │
│  • attendance.db (Can be upgraded to PostgreSQL)            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              External Services & Models                      │
│  • WhatsApp Business API (Notifications)                    │
│  • YOLOv5 nano (Phone Detection)                            │
│  • dlib 68-point facial landmarks (Liveness)                │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Prerequisites

### System Requirements
- **Operating System**: Linux, macOS, or Windows
- **Python**: 3.8 or higher (3.10+ recommended)
- **RAM**: Minimum 4GB (8GB+ recommended)
- **Webcam**: Required for face recognition
- **Internet**: Required for initial setup and WhatsApp features

### Required Tools
- **Python 3.10+**: [Download](https://www.python.org/downloads/)
- **pip**: Python package installer (included with Python)
- **CMake**: Required for building dlib
  - **Ubuntu/Debian**: `sudo apt-get install cmake`
  - **macOS**: `brew install cmake`
  - **Windows**: [Download CMake](https://cmake.org/download/)
- **Git**: For cloning the repository

### Build Tools (for dlib compilation)
- **Linux**: `sudo apt-get install build-essential libopenblas-dev liblapack-dev libjpeg-dev`
- **macOS**: `xcode-select --install`
- **Windows**: [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

---

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/smart-attendance-system.git
cd smart-attendance-system
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate

# On Windows:
.\venv\Scripts\activate
```

### Step 3: Install Python Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

**Note**: If `dlib` installation fails, ensure CMake and build tools are installed (see Prerequisites).

### Step 4: Download Required Models

Run the setup script to automatically download facial landmark predictor and YOLO model:

```bash
python setup.py
```

This script downloads:
- `shape_predictor_68_face_landmarks.dat` (99.7 MB) - for facial landmark detection
- `yolov5n.pt` (3.9 MB) - for phone/screen detection (optional but recommended)

**Manual Download** (if script fails):
```bash
# Facial landmarks
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2

# YOLO model
mkdir -p models
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt -O models/yolov5n.pt
```

### Step 5: Verify Installation

```bash
python health_check.py
```

This script validates:
- ✅ Python version
- ✅ Required files (models)
- ✅ Python dependencies
- ✅ Database setup
- ✅ Module imports
- ✅ Camera access

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
# Flask Configuration
SECRET_KEY='your-super-secret-key-change-in-production'
FLASK_PORT=5000
DEBUG=False

# WhatsApp Business API (Optional)
WHATSAPP_TOKEN='YOUR_WHATSAPP_API_TOKEN'
WHATSAPP_PHONE_ID='YOUR_WHATSAPP_SENDER_PHONE_ID'
WHATSAPP_DRY_RUN=1  # Set to 0 to enable actual sending

# Class Coordinator Contact
COORDINATOR_PHONE='+919876543210'

# OTP Configuration
OTP_EXP_MINUTES=10
OTP_RESEND_COOLDOWN_SEC=60

# Timezone
TIMEZONE='Asia/Kolkata'
```

### WhatsApp Setup (Optional)

To enable WhatsApp notifications:

1. **Create a Meta (Facebook) Developer Account**: [developers.facebook.com](https://developers.facebook.com)
2. **Set up WhatsApp Business API**:
   - Create a new app
   - Add WhatsApp product
   - Get your `Phone Number ID` and `Access Token`
3. **Update `.env`**:
   ```env
   WHATSAPP_TOKEN='your_token_here'
   WHATSAPP_PHONE_ID='your_phone_id_here'
   WHATSAPP_DRY_RUN=0  # Enable actual sending
   ```
4. **Test**: Use the admin dashboard's "Test WhatsApp" feature

**Dry Run Mode**: Keep `WHATSAPP_DRY_RUN=1` to test without sending actual messages (messages are logged only).

### Configuration File (`config.py`)

Key configurable parameters:

```python
# Face Recognition
FACE_MATCH_THRESHOLD = 0.55  # Lower = stricter matching
RECOGNITION_CONFIDENCE_MIN = 0.45
MIN_FACE_SIZE_PIXELS = 70

# Liveness Detection
EAR_THRESHOLD = 0.18  # Eye aspect ratio for blink
BLINK_CONSECUTIVE_FRAMES = 1
REQUIRE_EYE_CONTACT = False

# Anti-Spoofing
AUTO_BLOCK_SPOOF = True
SPOOF_CONFIDENCE_THRESHOLD_BLOCK = 0.50

# Attendance
ABSENCE_THRESHOLD = 3  # Days before alert
```

---

## 🎬 Running the Application

### Start the Server

```bash
python main.py
```

You should see:
```
✓ Configuration validated successfully
✓ Landmark predictor loaded
✓ Liveness detector initialized
✓ Database tables created
Starting server on port 5000
Timezone: Asia/Kolkata
```

### Access the Application

Open your browser and navigate to:

- **Login Page**: [http://localhost:5000/login](http://localhost:5000/login)
- **Admin Dashboard**: [http://localhost:5000/admin-dashboard](http://localhost:5000/admin-dashboard)
- **Student Dashboard**: [http://localhost:5000/student-dashboard](http://localhost:5000/student-dashboard)

### Default Credentials

**Admin Account**:
- Username: `admin`
- Password: `admin123`

⚠️ **Security Warning**: Change the default admin password immediately after first login!

---

## 📖 Usage Guide

### For Administrators

#### 1. Enroll a Student

1. Click **"Enroll Student"** button
2. Fill in student details:
   - Full Name (alphabets only)
   - Student ID (alphanumeric, minimum 3 characters)
   - Class and Section
   - Parent Phone Number (10-15 digits)
3. **Capture Photo**:
   - Position face in camera frame
   - Wait for quality feedback (green = excellent)
   - Click **"Capture Photo"** - system captures 7 frames automatically
4. Click **"Submit"** to enroll

**Tips**:
- Ensure good lighting
- Look directly at camera
- Remove glasses/hats for better accuracy
- System automatically creates student login credentials

#### 2. Start Attendance System

1. Click **"Start"** button in camera feed
2. System activates live recognition
3. Students stand in front of camera
4. System prompts: **"Please BLINK"**
5. After successful verification:
   - ✅ Green notification: "Student Name Marked!"
   - 🎵 Success sound plays
   - Points awarded
   - Stats update instantly

#### 3. Search Students

- Use search bar in header
- Search by name or student ID
- Click student to view detailed stats:
  - Total days, present/absent
  - Attendance rate, current streak
  - First/last seen dates

#### 4. Monitor Activity

- **Recent Events**: Live log of all recognition events
- **Stats Cards**: Real-time present/absent counts and attendance rate
- **Spoof Alerts**: Flagged spoofing attempts appear in activity log

### For Students

#### 1. Login

- Navigate to [http://localhost:5000/login](http://localhost:5000/login)
- Username: Your Student ID
- Default Password: `student123` (can be changed)

#### 2. View Dashboard

- **Profile**: See your photo, class, and points
- **Statistics**: Days present/absent, attendance rate, streak
- **Trend Chart**: Visualize your 30-day attendance pattern
- **Calendar**: See which days you were present/absent
- **Leaderboard**: Compare with classmates

#### 3. Change Password

- Click profile icon → Settings
- Enter old and new password
- Submit to update

---

## 🔌 API Documentation

### Authentication Endpoints

#### Register User
```http
POST /api/auth/register
Content-Type: application/json

{
  "username": "student123",
  "email": "student@school.com",
  "password": "password123",
  "role": "student",
  "student_id": 1
}
```

#### Login
```http
POST /api/auth/login
Content-Type: application/json

{
  "username": "student123",
  "password": "password123"
}

Response:
{
  "success": true,
  "token": "eyJhbGc...",
  "user": {
    "id": 2,
    "username": "student123",
    "role": "student",
    "student_id": 1
  }
}
```

### Student Endpoints (Require JWT)

#### Get Student Profile
```http
GET /api/student-profile/{student_id}
Authorization: Bearer {token}
```

#### Get Attendance Stats
```http
GET /api/student-stats/{student_id}
Authorization: Bearer {token}
```

#### Get Attendance Records
```http
GET /api/student-attendance/{student_id}?month=11&year=2025
Authorization: Bearer {token}
```

### Admin Endpoints (Admin Only)

#### Enroll Student Face
```http
POST /api/enroll-multishot
Authorization: Bearer {token}
Content-Type: application/json

{
  "student_id": "STU123",
  "frames": ["base64_frame1", "base64_frame2", ...]
}
```

#### Get All Students
```http
GET /api/students
Authorization: Bearer {token}
```

#### Get Attendance Statistics
```http
GET /api/stats
Authorization: Bearer {token}
```

---

## 📁 Project Structure

```
smart-attendance-system/
│
├── main.py                      # Application entry point
├── config.py                    # Configuration settings
├── requirements.txt             # Python dependencies
├── setup.py                     # Model download script
├── health_check.py              # System health validator
├── .env                         # Environment variables (create this)
│
├── models.py                    # Database models (SQLAlchemy)
├── auth_service.py              # JWT authentication
├── auth_routes.py               # Auth API routes
├── routes.py                    # Admin routes
├── student_routes.py            # Student routes
│
├── face_recognition_service.py  # Face recognition core
├── liveness_detection.py        # Liveness verification
├── attendance_service.py        # Business logic
├── whatsapp_service.py          # WhatsApp integration
│
├── spoof_detection/
│   ├── __init__.py
│   ├── ensemble_spoof.py        # Anti-spoofing algorithms
│   ├── phone_in_frame.py        # Device detection
│   └── metadata_checks.py       # Image metadata analysis
│
├── templates/                   # HTML templates
│   ├── login.html
│   ├── dashboard.html           # Admin dashboard
│   └── student_dashboard.html   # Student dashboard
│
├── static/                      # Static assets
│   ├── css/
│   │   ├── dashboard.css
│   │   └── dashboard_fixes.css
│   ├── js/
│   │   └── dashboard.js
│   └── enrollments/             # Stored enrollment photos
│
├── models/                      # AI models
│   └── yolov5n.pt              # YOLO phone detector
│
├── test_images/                 # Test dataset for evaluation
│   ├── student_1/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── student_2/
│       └── img1.jpg
│
├── evaluate_recognition.py      # Accuracy evaluation
├── evaluate_spoofing.py         # Anti-spoof testing
├── train_antispoofing.py        # Train custom CNN model
├── migrate_database.py          # Database migration script
│
├── shape_predictor_68_face_landmarks.dat  # dlib model (download)
└── attendance.db                # SQLite database (auto-created)
```

---

## 🔒 Security Features

### Authentication & Authorization
- **JWT-based authentication** with secure token generation
- **Role-based access control** (Admin, Student, Coordinator)
- **Password hashing** using bcrypt (not stored in plain text)
- **Session management** with token expiration

### Face Recognition Security
- **Face hash verification** prevents duplicate enrollments
- **Minimum face size** validation (70x70 pixels)
- **Brightness & sharpness** checks for quality
- **Confidence thresholding** (55% match threshold)

### Anti-Spoofing Measures
- **Blink detection** using Eye Aspect Ratio (EAR < 0.18)
- **Head pose estimation** ensures user is facing camera
- **Texture analysis** detects flat surfaces (photos)
- **Phone/screen detection** using YOLOv5 object detection
- **FFT moiré pattern analysis** for screen attacks
- **Multi-frame verification** (requires 2-3 consecutive frames)

### Data Protection
- **Encrypted passwords** (bcrypt with salt)
- **Secure JWT tokens** (HS256 algorithm)
- **SQL injection prevention** (SQLAlchemy ORM)
- **XSS protection** (Content Security Policy headers)
- **CORS configuration** for API security

---

## 🐛 Troubleshooting

### Common Issues

#### 1. `dlib` Installation Fails

**Problem**: `pip install dlib` fails with compilation errors

**Solutions**:
- **Install CMake**: 
  - Ubuntu: `sudo apt-get install cmake`
  - macOS: `brew install cmake`
  - Windows: Download from [cmake.org](https://cmake.org)
- **Install build tools**:
  - Ubuntu: `sudo apt-get install build-essential`
  - Windows: Install Visual Studio Build Tools
- **Use pre-built wheel** (Windows):
  ```bash
  pip install https://github.com/jloh02/dlib/releases/download/v19.22/dlib-19.22.99-cp310-cp310-win_amd64.whl
  ```

#### 2. Camera Not Detected

**Problem**: "Camera access denied" or camera not working

**Solutions**:
- Grant browser camera permissions
- Check if camera is being used by another application
- On Linux, add user to video group: `sudo usermod -a -G video $USER`
- Restart browser after granting permissions

#### 3. Face Not Recognized

**Problem**: System says "Face not recognized" for enrolled student

**Solutions**:
- Ensure good lighting conditions
- Student should face camera directly
- Remove glasses/hats during recognition
- Re-enroll with better quality images
- Lower `FACE_MATCH_THRESHOLD` in `config.py` (e.g., from 0.55 to 0.50)

#### 4. WhatsApp Notifications Not Working

**Problem**: No WhatsApp messages received

**Solutions**:
- Verify `WHATSAPP_TOKEN` and `WHATSAPP_PHONE_ID` in `.env`
- Check if `WHATSAPP_DRY_RUN=0` (must be 0 to send actual messages)
- Test using admin dashboard "Test WhatsApp" button
- Verify phone number format: `+<country_code><number>` (e.g., `+919876543210`)
- Check WhatsApp Business API quota and limits

#### 5. Database Errors

**Problem**: "OperationalError: no such table" or similar

**Solutions**:
```bash
# Delete existing database
rm attendance.db

# Restart application (creates fresh database)
python main.py
```

For schema updates:
```bash
python migrate_database.py
```

#### 6. Model Files Missing

**Problem**: "FileNotFoundError: shape_predictor_68_face_landmarks.dat"

**Solution**:
```bash
python setup.py
# Or download manually as shown in Installation section
```

---

## 🧪 Testing

### Run System Health Check
```bash
python health_check.py
```

### Test Face Recognition Accuracy
```bash
# Prepare test dataset (see project structure)
python evaluate_recognition.py
```

### Test Anti-Spoofing System
```bash
# Prepare test dataset with live/ and spoof/ folders
python evaluate_spoofing.py --spoof-test-dir ./spoof_testset --recog-test-dir ./test_images
```

### Manual Testing Checklist

- [ ] Admin can login successfully
- [ ] Student enrollment works
- [ ] Live recognition identifies enrolled students
- [ ] Blink detection prompts appear
- [ ] Spoof attempts are blocked (test with photo)
- [ ] Student can login and view dashboard
- [ ] Stats update in real-time
- [ ] WhatsApp notifications send (if enabled)
- [ ] Leaderboard displays correctly
- [ ] Search functionality works

---

## 🚀 Deployment

### Production Recommendations

1. **Change Secret Key**:
   ```env
   SECRET_KEY='use-strong-random-secret-key-here'
   ```

2. **Use PostgreSQL** (instead of SQLite):
   ```python
   # config.py
   SQLALCHEMY_DATABASE_URI = 'postgresql://user:password@localhost/attendance_db'
   ```

3. **Enable HTTPS**: Use reverse proxy (Nginx) with SSL certificate

4. **Set Debug to False**:
   ```env
   DEBUG=False
   ```

5. **Use Production WSGI Server**:
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 --worker-class eventlet main:app
   ```

6. **Environment Variables**: Use proper secret management (e.g., AWS Secrets Manager)

### Deployment Platforms

- **Render**: Easy deployment with free tier
- **Heroku**: Platform-as-a-Service with PostgreSQL addon
- **AWS EC2**: Full control with Elastic Beanstalk
- **DigitalOcean**: App Platform or Droplet
- **Railway**: Modern deployment with GitHub integration

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/AmazingFeature`
3. **Commit changes**: `git commit -m 'Add some AmazingFeature'`
4. **Push to branch**: `git push origin feature/AmazingFeature`
5. **Open a Pull Request**

### Development Guidelines

- Follow PEP 8 style guide for Python code
- Write meaningful commit messages
- Add docstrings to functions and classes
- Test thoroughly before submitting PR
- Update README if adding new features

---


## 🙏 Acknowledgments

- **dlib**: Face detection and landmarks
- **face_recognition**: Simplified face recognition API
- **OpenCV**: Computer vision operations
- **Flask**: Web framework
- **YOLOv5**: Object detection for anti-spoofing
- **Chart.js**: Beautiful charts for student dashboard

---

## 🗺️ Roadmap

- [ ] Mobile app (React Native/Flutter)
- [ ] Multi-camera support
- [ ] Cloud storage integration (AWS S3)
- [ ] Advanced analytics dashboard
- [ ] Email notification support
- [ ] Fingerprint authentication backup
- [ ] REST API documentation (Swagger)
- [ ] Docker containerization
- [ ] Kubernetes deployment configs

---

