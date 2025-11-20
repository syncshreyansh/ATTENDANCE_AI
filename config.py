# config.py - OPTIMIZED VERSION with balanced speed and security
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
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    FRAME_SKIP = 3  # OPTIMIZED: Skip more frames for speed
    
    # OPTIMIZED: Face Recognition Thresholds - Balanced for speed
    FACE_MATCH_THRESHOLD = 0.55  # Slightly more lenient
    RECOGNITION_CONFIDENCE_MIN = 0.45  # Lower for faster matching
    
    # OPTIMIZED: Face Quality Requirements - Less strict for speed
    MIN_FACE_SIZE_PIXELS = 70  # Smaller minimum
    MIN_FACE_BRIGHTNESS = 25   # More lenient
    MAX_FACE_BRIGHTNESS = 245
    MIN_IMAGE_SHARPNESS = 40   # Lower threshold
    
    # Enrollment Settings
    ENROLLMENT_NUM_JITTERS = 10
    ENROLLMENT_MODEL = 'large'
    DUPLICATE_FACE_THRESHOLD = 0.35
    
    # OPTIMIZED: Liveness Detection - Faster but secure
    EAR_THRESHOLD = 0.18  # Lower = easier blink detection
    BLINK_CONSECUTIVE_FRAMES = 1  # Just 1 frame needed
    REQUIRED_BLINKS = 1
    EYE_CONTACT_THRESHOLD = 50  # More tolerance
    REQUIRE_EYE_CONTACT = False
    TEXTURE_QUALITY_THRESHOLD = 30  # Lower for speed
    LIVENESS_CONFIDENCE_THRESHOLD = 0.4  # Lower pass threshold
    
    # CRITICAL: Anti-Spoofing Settings - ENHANCED SECURITY
    AUTO_BLOCK_SPOOF = True
    SPOOF_CONFIDENCE_THRESHOLD_FLAG = 0.50  # LOWERED for better blocking
    SPOOF_CONFIDENCE_THRESHOLD_BLOCK = 0.50  # Unified threshold
    
    # CRITICAL: Weighted scoring emphasizing phone detection
    SPOOF_WEIGHT_CNN = 0.15  # Reduced (optional CNN)
    SPOOF_WEIGHT_TEXTURE = 0.30  # Texture matters
    SPOOF_WEIGHT_PHONE = 0.55  # CRITICAL: Phone detection is key
    SPOOF_WEIGHT_MOIRE = 0.00  # Removed for speed (included in texture)
    SPOOF_WEIGHT_REFLECTION = 0.00  # Removed for speed
    SPOOF_WEIGHT_BLINK = 0.00  # Moved to liveness check
    
    # Model paths
    ANTI_SPOOF_CNN_MODEL = 'models/anti_spoof_resnet18.onnx'
    PHONE_DETECTOR_MODEL = 'models/yolov5n.pt'
    LANDMARK_PREDICTOR = 'shape_predictor_68_face_landmarks.dat'
    
    # WhatsApp API with DRY_RUN
    WHATSAPP_TOKEN = os.environ.get("WHATSAPP_TOKEN") or ""
    WHATSAPP_PHONE_ID = os.environ.get("WHATSAPP_PHONE_ID") or ""
    WHATSAPP_DRY_RUN = bool(int(os.environ.get("WHATSAPP_DRY_RUN", "1")))
    WHATSAPP_WEBHOOK_VERIFY_TOKEN = os.environ.get('WHATSAPP_WEBHOOK_VERIFY_TOKEN')
    
    if not os.path.exists(PHONE_DETECTOR_MODEL):
        print(f"⚠️  CRITICAL: YOLO model missing at {PHONE_DETECTOR_MODEL}")
        print("   Download: wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt -O models/yolov5n.pt")
        print("   OR: pip install gdown && gdown 1Drs_Aiu7xx6S-ix95f9kNsA6ueKRpN2b -O models/yolov5n.pt")
    
    # OTP Settings
    OTP_EXP_MINUTES = int(os.environ.get("OTP_EXP_MINUTES", "10"))
    OTP_RESEND_COOLDOWN_SEC = int(os.environ.get("OTP_RESEND_COOLDOWN_SEC", "60"))
    
    # Coordinator Contact
    CLASS_COORDINATOR_PHONE = os.environ.get('COORDINATOR_PHONE') or '+919876543210'
    
    # Alert Settings
    ABSENCE_THRESHOLD = 3
    LATE_THRESHOLD_MINUTES = 10
    TAMPER_SENSITIVITY = 0.8
    
    # Daily Reset Time (IST)
    RESET_TIME_HOUR = 0
    RESET_TIME_MINUTE = 0
    
    # OPTIMIZED: Performance Settings for Speed
    RECOGNITION_COOLDOWN_SECONDS = 5
    TARGET_FPS = 5  # INCREASED: Process more frames
    MAX_WORKERS = 2
    
    # OPTIMIZED: Processing timeouts
    BLINK_WAIT_TIMEOUT = 5  # REDUCED: Faster timeout
    REQUIRED_CONSECUTIVE_FRAMES = 2  # REDUCED: Fewer frames
    
    # Security Settings
    MAX_RECOGNITION_ATTEMPTS = 5
    LOCKOUT_DURATION_SECONDS = 30
    LOG_UNKNOWN_FACES = True
    LOG_FAILED_ENROLLMENTS = True
    LOG_SPOOF_ATTEMPTS = True
    
    # OPTIMIZED: Caching for performance
    ENABLE_SPOOF_CACHE = True
    SPOOF_CACHE_TIMEOUT = 3  # seconds
    ENABLE_LANDMARK_CACHE = True
    LANDMARK_CACHE_TIMEOUT = 0.5  # seconds
    
    # Development/Debug Settings
    SHOW_DEBUG_OVERLAY = os.environ.get('SHOW_DEBUG', 'False').lower() == 'true'
    SAVE_DEBUG_IMAGES = False
    DEBUG_IMAGE_DIR = 'debug_images'
    
    @classmethod
    def validate(cls):
        """Validate configuration on startup"""
        errors = []
        
        if not os.path.exists(cls.LANDMARK_PREDICTOR):
            errors.append(f"Missing landmark predictor: {cls.LANDMARK_PREDICTOR}")
        
        if not 0 < cls.FACE_MATCH_THRESHOLD < 1:
            errors.append("FACE_MATCH_THRESHOLD must be between 0 and 1")
        
        if not 0 < cls.RECOGNITION_CONFIDENCE_MIN < 1:
            errors.append("RECOGNITION_CONFIDENCE_MIN must be between 0 and 1")
        
        # Validate spoof weights
        total_weight = (cls.SPOOF_WEIGHT_CNN + cls.SPOOF_WEIGHT_TEXTURE + 
                       cls.SPOOF_WEIGHT_PHONE)
        
        if not 0.95 <= total_weight <= 1.05:
            errors.append(f"Spoof weights must sum to ~1.0 (currently: {total_weight})")
        
        # Check YOLO model
        if not os.path.exists(cls.PHONE_DETECTOR_MODEL):
            print(f"⚠️  Warning: YOLO phone detector not found")
            print("   Phone detection is CRITICAL for security!")
            print("   System will use fallback edge detection (less accurate)")
        
        return errors
    
    @classmethod
    def get_summary(cls):
        """Get configuration summary for logging"""
        return {
            'face_threshold': cls.FACE_MATCH_THRESHOLD,
            'confidence_min': cls.RECOGNITION_CONFIDENCE_MIN,
            'min_face_size': cls.MIN_FACE_SIZE_PIXELS,
            'spoof_auto_block': cls.AUTO_BLOCK_SPOOF,
            'spoof_threshold': cls.SPOOF_CONFIDENCE_THRESHOLD_BLOCK,
            'liveness_threshold': cls.LIVENESS_CONFIDENCE_THRESHOLD,
            'texture_threshold': cls.TEXTURE_QUALITY_THRESHOLD,
            'blink_timeout': cls.BLINK_WAIT_TIMEOUT,
            'required_frames': cls.REQUIRED_CONSECUTIVE_FRAMES,
            'target_fps': cls.TARGET_FPS,
            'spoof_cache_enabled': cls.ENABLE_SPOOF_CACHE,
            'whatsapp_dry_run': cls.WHATSAPP_DRY_RUN
        }

# Validate on import
validation_errors = Config.validate()
if validation_errors:
    print("⚠️  Configuration Errors:")
    for error in validation_errors:
        print(f"  - {error}")
else:
    print("✓ Configuration validated successfully")
    print("=" * 60)
    print("OPTIMIZED SETTINGS:")
    print("  • Faster processing with aggressive frame skipping")
    print("  • Lenient liveness thresholds for quick approval")
    print("  • ENHANCED phone/photo blocking (0.50 threshold)")
    print("  • Caching enabled for performance")
    print("=" * 60)
    summary = Config.get_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print("=" * 60)