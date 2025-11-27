#!/usr/bin/env python3
"""
Setup Script for AI Attendance System
Automates dependency installation, model downloads, and database initialization
"""

import os
import sys
import subprocess
import urllib.request
import shutil
from pathlib import Path

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(message):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{message.center(60)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")

def print_success(message):
    print(f"{Colors.GREEN}✓ {message}{Colors.END}")

def print_warning(message):
    print(f"{Colors.YELLOW}⚠ {message}{Colors.END}")

def print_error(message):
    print(f"{Colors.RED}✗ {message}{Colors.END}")

def print_info(message):
    print(f"{Colors.BLUE}ℹ {message}{Colors.END}")

def check_python_version():
    """Ensure Python 3.8+"""
    print_header("Checking Python Version")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print_error(f"Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    print_success(f"Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_build_tools():
    """Check for C++ build tools (required for dlib)"""
    print_header("Checking Build Tools")
    
    # Check CMake
    try:
        result = subprocess.run(['cmake', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print_success("CMake is installed")
        else:
            print_warning("CMake not found - required for dlib")
            return False
    except FileNotFoundError:
        print_warning("CMake not found - required for dlib")
        print_info("Install CMake: https://cmake.org/download/")
        return False
    
    return True

def install_requirements():
    """Install Python dependencies"""
    print_header("Installing Python Dependencies")
    
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print_error("requirements.txt not found!")
        return False
    
    print_info("Installing packages (this may take 10-20 minutes)...")
    print_info("Installing dlib may take the longest time...")
    
    try:
        # Upgrade pip first
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                      check=True)
        
        # Install requirements
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True)
        
        print_success("All dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install dependencies: {e}")
        print_info("Try installing problematic packages manually")
        return False

def download_dlib_model():
    """Download dlib facial landmarks model"""
    print_header("Downloading Facial Landmarks Model")
    
    model_file = "shape_predictor_68_face_landmarks.dat"
    
    if os.path.exists(model_file):
        file_size = os.path.getsize(model_file) / (1024 * 1024)
        print_success(f"{model_file} already exists ({file_size:.1f} MB)")
        return True
    
    print_info("Downloading dlib 68 face landmarks model (~100 MB)...")
    url = "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2"
    
    try:
        # Download compressed file
        print_info("Downloading...")
        urllib.request.urlretrieve(url, f"{model_file}.bz2")
        
        # Decompress
        print_info("Decompressing...")
        import bz2
        with bz2.open(f"{model_file}.bz2", 'rb') as source:
            with open(model_file, 'wb') as dest:
                dest.write(source.read())
        
        # Remove compressed file
        os.remove(f"{model_file}.bz2")
        
        print_success(f"{model_file} downloaded and extracted")
        return True
    except Exception as e:
        print_error(f"Failed to download model: {e}")
        print_info(f"Manual download: {url}")
        return False

def check_yolo_models():
    """Check for YOLO models (for phone detection)"""
    print_header("Checking YOLO Models")
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    yolo_model = models_dir / "yolov5n.pt"
    
    if yolo_model.exists():
        file_size = os.path.getsize(yolo_model) / (1024 * 1024)
        print_success(f"YOLOv5 nano model found ({file_size:.1f} MB)")
        return True
    
    print_warning("YOLOv5 model not found (required for phone detection)")
    print_info("The system will use fallback edge detection (less accurate)")
    print_info("Download manually:")
    print_info("  https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt")
    print_info(f"  Save to: {yolo_model.absolute()}")
    
    return False

def create_env_file():
    """Create example .env file if it doesn't exist"""
    print_header("Setting Up Environment Variables")
    
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_file.exists():
        print_success(".env file already exists")
        return True
    
    # Create example .env
    env_content = """# Flask Configuration
SECRET_KEY=change-this-to-a-random-secret-key-in-production

# WhatsApp Business API (Twilio)
WHATSAPP_TOKEN=your_whatsapp_api_token_here
WHATSAPP_PHONE_ID=your_whatsapp_phone_id_here
WHATSAPP_DRY_RUN=1

# Twilio Configuration
TWILIO_ACCOUNT_SID=your_twilio_account_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_WHATSAPP_NUMBER=+14155238886

# Coordinator Contact
COORDINATOR_PHONE=+919876543210

# OTP Settings
OTP_EXP_MINUTES=10
OTP_RESEND_COOLDOWN_SEC=60

# Debug Mode (set to True for development)
SHOW_DEBUG=False
"""
    
    with open(env_example, 'w') as f:
        f.write(env_content)
    
    print_success("Created .env.example file")
    print_warning("Copy .env.example to .env and configure your credentials")
    print_info("  cp .env.example .env")
    print_info("  Edit .env with your WhatsApp API credentials")
    
    return True

def create_directories():
    """Create required directories"""
    print_header("Creating Required Directories")
    
    directories = [
        "models",
        "instance",
        "static/uploads",
        "debug_images",
        "test_images"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print_success(f"Created/verified: {directory}/")
    
    return True

def initialize_database():
    """Initialize SQLite database"""
    print_header("Initializing Database")
    
    db_file = Path("instance/attendance.db")
    
    if db_file.exists():
        print_warning("Database already exists - skipping initialization")
        print_info("To reset database, delete: instance/attendance.db")
        return True
    
    try:
        print_info("Creating database tables...")
        # Import and create tables
        from models import db
        from main import app
        
        with app.app_context():
            db.create_all()
            print_success("Database tables created")
        
        return True
    except Exception as e:
        print_error(f"Failed to initialize database: {e}")
        print_info("Run 'python main.py' manually to initialize")
        return False

def verify_installation():
    """Verify all key components are working"""
    print_header("Verifying Installation")
    
    checks = {
        "OpenCV": "import cv2; print(cv2.__version__)",
        "face_recognition": "import face_recognition; print('OK')",
        "dlib": "import dlib; print('OK')",
        "Flask": "import flask; print(flask.__version__)",
        "torch": "import torch; print(torch.__version__)",
        "ultralytics": "import ultralytics; print('OK')",
    }
    
    all_ok = True
    for package, check_code in checks.items():
        try:
            result = subprocess.run(
                [sys.executable, "-c", check_code],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                version = result.stdout.strip()
                print_success(f"{package}: {version}")
            else:
                print_error(f"{package}: Failed to import")
                all_ok = False
        except Exception as e:
            print_error(f"{package}: {e}")
            all_ok = False
    
    return all_ok

def print_next_steps():
    """Print instructions for next steps"""
    print_header("Setup Complete!")
    
    print(f"\n{Colors.BOLD}Next Steps:{Colors.END}\n")
    print("1. Configure environment variables:")
    print(f"   {Colors.BLUE}Edit .env file with your WhatsApp API credentials{Colors.END}\n")
    
    print("2. Download YOLO model (if not already present):")
    print(f"   {Colors.BLUE}Download from: https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt{Colors.END}")
    print(f"   {Colors.BLUE}Save to: models/yolov5n.pt{Colors.END}\n")
    
    print("3. Start the application:")
    print(f"   {Colors.GREEN}python main.py{Colors.END}\n")
    
    print("4. Access the system:")
    print(f"   {Colors.BLUE}Open browser: http://127.0.0.1:5000{Colors.END}")
    print(f"   {Colors.BLUE}Default admin: username=admin, password=admin123{Colors.END}\n")
    
    print(f"\n{Colors.BOLD}Security Features Enabled:{Colors.END}")
    print(f"  {Colors.GREEN}✓ Blink detection (liveness){Colors.END}")
    print(f"  {Colors.GREEN}✓ Gaze tracking (head pose){Colors.END}")
    print(f"  {Colors.GREEN}✓ Texture analysis (photo detection){Colors.END}")
    print(f"  {Colors.GREEN}✓ Phone/screen detection (YOLO){Colors.END}")
    print(f"  {Colors.GREEN}✓ Anti-spoofing threshold: 0.50 (blocks proxy attempts){Colors.END}")
    print(f"\n{Colors.RED}⚠ NO ONE CAN MARK PROXY ATTENDANCE WITH PHOTOS/VIDEOS! ⚠{Colors.END}\n")

def main():
    """Main setup workflow"""
    print_header("AI Attendance System - Automated Setup")
    
    steps = [
        ("Python Version", check_python_version),
        ("Build Tools", check_build_tools),
        ("Directories", create_directories),
        ("Python Dependencies", install_requirements),
        ("Facial Landmarks Model", download_dlib_model),
        ("YOLO Models", check_yolo_models),
        ("Environment File", create_env_file),
        ("Installation Verification", verify_installation),
    ]
    
    results = []
    for step_name, step_func in steps:
        try:
            result = step_func()
            results.append((step_name, result))
        except Exception as e:
            print_error(f"{step_name} failed: {e}")
            results.append((step_name, False))
    
    # Summary
    print_header("Setup Summary")
    for step_name, result in results:
        if result:
            print_success(step_name)
        else:
            print_error(step_name)
    
    # Next steps
    if all(result for _, result in results if _ != "YOLO Models"):
        print_next_steps()
    else:
        print_error("\nSetup incomplete. Please fix errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()