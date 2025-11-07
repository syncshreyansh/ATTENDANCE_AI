# config.py - ENHANCED VERSION with optimized thresholds
import os
from datetime import timedelta

class Config:
    # Database
    DATABASE_PATH = 'attendance_system.db'
    SQLALCHEMY_DATABASE_URI = 'sqlite:///attendance.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Flask
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production-v2'
    FLASK_PORT = 5000
    DEBUG = False
    
    # Timezone Settings
    TIMEZONE = 'Asia/Kolkata'
    
    # Camera Settings
    CAMERA_INDEX = 0
    FRAME_WIDTH = 640  # Reduced for better performance
    FRAME_HEIGHT = 480
    FRAME_SKIP = 2  # Process every 2nd frame
    
    # === ENHANCED Face Recognition Thresholds ===
    # Lower = stricter (more accurate but may reject valid faces)
    # Higher = lenient (accepts more faces but may have false positives)
    FACE_MATCH_THRESHOLD = 0.5  # Stricter (was 0.6) - RECOMMENDED: 0.45-0.55
    RECOGNITION_CONFIDENCE_MIN = 0.5  # Minimum confidence to accept match
    
    # Face Quality Requirements
    MIN_FACE_SIZE_PIXELS = 100  # Minimum face dimensions
    MIN_FACE_BRIGHTNESS = 30  # Minimum average brightness
    MAX_FACE_BRIGHTNESS = 240  # Maximum (to detect overexposure)
    MIN_IMAGE_SHARPNESS = 50  # Laplacian variance threshold
    
    # === Enrollment Settings (Higher Quality) ===
    ENROLLMENT_NUM_JITTERS = 10  # More jitters = better encoding (default: 1)
    ENROLLMENT_MODEL = 'large'  # 'small' or 'large' (large is more accurate)
    DUPLICATE_FACE_THRESHOLD = 0.35  # Stricter than recognition
    
    # === Liveness Detection (Blink, Gaze) ===
    EAR_THRESHOLD = 0.25  # Eye Aspect Ratio for blink
    BLINK_CONSECUTIVE_FRAMES = 3
    REQUIRED_BLINKS = 1  # Minimum blinks required
    
    # Head Pose (degrees allowed for "looking at camera")
    EYE_CONTACT_THRESHOLD = 35  # Relaxed (was 15)
    REQUIRE_EYE_CONTACT = False  # Set True for stricter liveness
    
    # Texture Analysis
    TEXTURE_QUALITY_THRESHOLD = 50  # For photo/screen detection
    LIVENESS_CONFIDENCE_THRESHOLD = 0.6
    
    # === Anti-Spoofing Settings ===
    # AUTO_BLOCK_SPOOF: If True, blocks high-confidence spoofs automatically
    # If False, only flags them for review
    AUTO_BLOCK_SPOOF = os.environ.get('AUTO_BLOCK_SPOOF', 'False').lower() == 'true'
    
    # Confidence thresholds for spoofing
    SPOOF_CONFIDENCE_THRESHOLD_FLAG = 0.7  # Flag for review
    SPOOF_CONFIDENCE_THRESHOLD_BLOCK = 0.85  # Auto-block (if enabled)
    
    # Spoof detection weights (must sum to ~1.0)
    SPOOF_WEIGHT_CNN = 0.25  # Deep learning model
    SPOOF_WEIGHT_TEXTURE = 0.20  # Texture analysis
    SPOOF_WEIGHT_PHONE = 0.20  # Phone/device detection
    SPOOF_WEIGHT_MOIRE = 0.15  # Screen pattern detection
    SPOOF_WEIGHT_REFLECTION = 0.10  # Eye reflection
    SPOOF_WEIGHT_BLINK = 0.10  # Blink verification
    
    # Model paths
    ANTI_SPOOF_CNN_MODEL = 'models/anti_spoof_resnet18.onnx'
    PHONE_DETECTOR_MODEL = 'models/yolov5n.pt'
    LANDMARK_PREDICTOR = 'shape_predictor_68_face_landmarks.dat'
    
    # === WhatsApp API ===
    WHATSAPP_TOKEN = os.environ.get('WHATSAPP_TOKEN')
    WHATSAPP_PHONE_ID = os.environ.get('WHATSAPP_PHONE_ID')
    WHATSAPP_WEBHOOK_VERIFY_TOKEN = os.environ.get('WHATSAPP_WEBHOOK_VERIFY_TOKEN')
    
    # Coordinator Contact
    CLASS_COORDINATOR_PHONE = os.environ.get('COORDINATOR_PHONE') or '+919876543210'
    
    # === Alert Settings ===
    ABSENCE_THRESHOLD = 3  # Days before alert
    LATE_THRESHOLD_MINUTES = 10
    TAMPER_SENSITIVITY = 0.8
    
    # Daily Reset Time (IST)
    RESET_TIME_HOUR = 0  # 12:00 AM
    RESET_TIME_MINUTE = 0
    
    # === Performance Settings ===
    # Recognition cooldown (seconds between recognitions for same student)
    RECOGNITION_COOLDOWN_SECONDS = 5
    
    # Frame processing rate (FPS)
    TARGET_FPS = 3  # Process 3 frames per second
    
    # Thread pool workers
    MAX_WORKERS = 2
    
    # === Security Settings ===
    # Maximum recognition attempts before temporary lockout
    MAX_RECOGNITION_ATTEMPTS = 5
    LOCKOUT_DURATION_SECONDS = 30
    
    # Log suspicious activity
    LOG_UNKNOWN_FACES = True
    LOG_FAILED_ENROLLMENTS = True
    LOG_SPOOF_ATTEMPTS = True
    
    # === Development/Debug Settings ===
    SHOW_DEBUG_OVERLAY = os.environ.get('SHOW_DEBUG', 'False').lower() == 'true'
    SAVE_DEBUG_IMAGES = False  # Save frames with detection issues
    DEBUG_IMAGE_DIR = 'debug_images'
    
    @classmethod
    def validate(cls):
        """Validate configuration on startup"""
        errors = []
        
        # Check model files
        if not os.path.exists(cls.LANDMARK_PREDICTOR):
            errors.append(f"Missing landmark predictor: {cls.LANDMARK_PREDICTOR}")
        
        # Validate thresholds
        if not 0 < cls.FACE_MATCH_THRESHOLD < 1:
            errors.append("FACE_MATCH_THRESHOLD must be between 0 and 1")
        
        if not 0 < cls.RECOGNITION_CONFIDENCE_MIN < 1:
            errors.append("RECOGNITION_CONFIDENCE_MIN must be between 0 and 1")
        
        # Validate spoof weights
        total_weight = (cls.SPOOF_WEIGHT_CNN + cls.SPOOF_WEIGHT_TEXTURE + 
                       cls.SPOOF_WEIGHT_PHONE + cls.SPOOF_WEIGHT_MOIRE + 
                       cls.SPOOF_WEIGHT_REFLECTION + cls.SPOOF_WEIGHT_BLINK)
        
        if not 0.95 <= total_weight <= 1.05:
            errors.append(f"Spoof weights must sum to ~1.0 (currently: {total_weight})")
        
        return errors
    
    @classmethod
    def get_summary(cls):
        """Get configuration summary for logging"""
        return {
            'face_threshold': cls.FACE_MATCH_THRESHOLD,
            'confidence_min': cls.RECOGNITION_CONFIDENCE_MIN,
            'min_face_size': cls.MIN_FACE_SIZE_PIXELS,
            'spoof_auto_block': cls.AUTO_BLOCK_SPOOF,
            'spoof_flag_threshold': cls.SPOOF_CONFIDENCE_THRESHOLD_FLAG,
            'enrollment_quality': cls.ENROLLMENT_NUM_JITTERS,
            'target_fps': cls.TARGET_FPS
        }

# Validate on import
validation_errors = Config.validate()
if validation_errors:
    print("⚠️ Configuration Errors:")
    for error in validation_errors:
        print(f"  - {error}")
else:
    print("✓ Configuration validated successfully")
    print(f"✓ Settings: {Config.get_summary()}")