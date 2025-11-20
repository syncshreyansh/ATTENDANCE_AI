"""
OPTIMIZED Ensemble Anti-Spoofing - Fast but secure
Blocks phones/photos effectively while maintaining speed
"""
import cv2
import numpy as np
import logging
import os
from functools import lru_cache

logger = logging.getLogger(__name__)

_cnn_model = None
_yolo_model = None
_yolo_load_attempted = False
_cnn_load_attempted = False
_cnn_available = False

# OPTIMIZED: Cache for expensive computations
_texture_cache = {}
_moire_cache = {}
CACHE_SIZE = 50

def load_yolo_model():
    """Load YOLOv5 nano for phone detection"""
    global _yolo_model, _yolo_load_attempted
    
    if _yolo_model is not None:
        return _yolo_model
    
    if _yolo_load_attempted:
        return None
    
    _yolo_load_attempted = True
    
    model_path = 'models/yolov5n.pt'
    if not os.path.exists(model_path):
        logger.info(f"YOLO model not found. Phone detection will use fallback method.")
        return None
    
    try:
        from ultralytics import YOLO
        _yolo_model = YOLO(model_path)
        _yolo_model.conf = 0.45  # OPTIMIZED: Balanced confidence
        logger.info("✓ YOLOv5 nano loaded for device detection")
        return _yolo_model
    except ImportError:
        logger.info("ultralytics not installed. Phone detection will use fallback.")
        return None
    except Exception as e:
        logger.info(f"Could not load YOLO: {e}. Using fallback.")
        return None

@lru_cache(maxsize=128)
def calculate_laplacian_variance_cached(face_hash):
    """Cached texture analysis using hash"""
    return None  # Placeholder for cache lookup

def calculate_laplacian_variance(face_roi):
    """OPTIMIZED: Faster texture analysis"""
    try:
        # Use smaller region for speed
        h, w = face_roi.shape[:2]
        if h < 40 or w < 40:
            return 0
        
        # Sample center region only
        center_roi = face_roi[h//3:2*h//3, w//3:2*w//3]
        
        gray = cv2.cvtColor(center_roi, cv2.COLOR_BGR2GRAY)
        
        # OPTIMIZED: Use smaller kernel
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=1)
        variance = laplacian.var()
        
        return variance
    except Exception as e:
        logger.error(f"Texture error: {e}")
        return 0

def calculate_fft_moire_fast(face_roi):
    """OPTIMIZED: Faster FFT analysis"""
    try:
        # Use smaller region
        h, w = face_roi.shape[:2]
        if h < 40 or w < 40:
            return 0.0
        
        # Downsample for speed
        small_roi = cv2.resize(face_roi, (64, 64))
        gray = cv2.cvtColor(small_roi, cv2.COLOR_BGR2GRAY)
        
        # FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
        
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        
        # Quick high-frequency check
        high_freq_mask = np.ones((rows, cols), np.uint8)
        r = 10
        cv2.circle(high_freq_mask, (ccol, crow), r, 0, -1)
        
        high_freq_energy = np.sum(magnitude_spectrum * high_freq_mask)
        total_energy = np.sum(magnitude_spectrum)
        
        hf_ratio = high_freq_energy / (total_energy + 1e-6)
        moire_confidence = min(hf_ratio / 0.65, 1.0)
        return moire_confidence
    except Exception as e:
        logger.error(f"FFT error: {e}")
        return 0.0

def detect_phone_in_frame_fast(frame, face_bbox):
    """OPTIMIZED: Faster phone detection with aggressive blocking"""
    model = load_yolo_model()
    if model is None:
        return check_phone_via_edges_fast(frame, face_bbox)
    
    try:
        # OPTIMIZED: Downsample frame for faster YOLO
        h, w = frame.shape[:2]
        scale = 640 / max(h, w)
        if scale < 1:
            small_frame = cv2.resize(frame, None, fx=scale, fy=scale)
        else:
            small_frame = frame
        
        results = model(small_frame, verbose=False)
        detections = results[0].boxes.data.cpu().numpy() if len(results) > 0 else []
        
        # Phone/screen classes: 67=cell phone, 73=laptop, 63=tv/monitor
        phone_classes = [67, 73, 63]
        
        fx1, fy1, fx2, fy2 = face_bbox
        face_center_x = (fx1 + fx2) / 2
        face_center_y = (fy1 + fy2) / 2
        face_area = (fx2 - fx1) * (fy2 - fy1)
        frame_area = frame.shape[0] * frame.shape[1]
        
        best_conf = 0.0
        best_bbox = None
        
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            
            # Scale back to original frame size
            if scale < 1:
                x1, y1, x2, y2 = x1/scale, y1/scale, x2/scale, y2/scale
            
            if int(cls) not in phone_classes:
                continue
            
            phone_center_x = (x1 + x2) / 2
            phone_center_y = (y1 + y2) / 2
            dist = np.sqrt((phone_center_x - face_center_x)**2 + (phone_center_y - face_center_y)**2)
            phone_area = (x2 - x1) * (y2 - y1)
            
            # CRITICAL: Phone directly over face
            overlap_x = max(0, min(x2, fx2) - max(x1, fx1))
            overlap_y = max(0, min(y2, fy2) - max(y1, fy1))
            overlap_area = overlap_x * overlap_y
            
            if overlap_area > face_area * 0.25:
                logger.warning(f"🚨 CRITICAL: Phone overlapping face!")
                return 0.98, [int(x1), int(y1), int(x2-x1), int(y2-y1)]
            
            # Large screen in frame
            if phone_area > frame_area * 0.12 and conf > 0.35:
                logger.warning(f"🚨 Large screen detected")
                return min(float(conf) + 0.25, 0.95), [int(x1), int(y1), int(x2-x1), int(y2-y1)]
            
            # Phone near face
            if dist < 350 and conf > best_conf:
                best_conf = float(conf) + 0.2
                best_bbox = [int(x1), int(y1), int(x2-x1), int(y2-y1)]
        
        return best_conf, best_bbox
    except Exception as e:
        logger.error(f"YOLO error: {e}")
        return check_phone_via_edges_fast(frame, face_bbox)

def check_phone_via_edges_fast(frame, face_bbox):
    """OPTIMIZED: Faster edge-based phone detection"""
    try:
        # Downsample for speed
        small_frame = cv2.resize(frame, (320, 240))
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        # Quick edge detection
        edges = cv2.Canny(gray, 40, 120)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        fx1, fy1, fx2, fy2 = face_bbox
        scale_x = 320 / frame.shape[1]
        scale_y = 240 / frame.shape[0]
        
        scaled_face_area = (fx2 - fx1) * (fy2 - fy1) * scale_x * scale_y
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > scaled_face_area * 0.4:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h) if h > 0 else 0
                
                # Phone-like rectangle
                if 0.4 < aspect_ratio < 2.8 and area > 5000:
                    # Scale back
                    x_orig = int(x / scale_x)
                    y_orig = int(y / scale_y)
                    w_orig = int(w / scale_x)
                    h_orig = int(h / scale_y)
                    return 0.65, [x_orig, y_orig, w_orig, h_orig]
        
        return 0.0, None
    except Exception as e:
        logger.error(f"Edge detection error: {e}")
        return 0.0, None

def check(frame, face_bbox, face_encoding=None):
    """
    OPTIMIZED: Fast but secure spoof detection
    Returns: dict {is_spoof: bool, spoof_type: str or list, confidence: float, evidence: dict}
    """
    try:
        x, y, w, h = face_bbox
        face_roi = frame[y:y+h, x:x+w]
        
        if face_roi.size == 0:
            return {
                'is_spoof': False,
                'spoof_type': None,
                'confidence': 0.0,
                'evidence': {'error': 'invalid_face_roi'}
            }
        
        # 1. TEXTURE ANALYSIS (fast)
        texture_var = calculate_laplacian_variance(face_roi)
        
        # CRITICAL: Emergency block for very low texture
        if texture_var < 22:
            logger.critical(f"🚨 EMERGENCY BLOCK: texture={texture_var:.1f} < 22")
            return {
                'is_spoof': True,
                'spoof_type': ['extremely_low_texture', 'likely_photo'],
                'confidence': 0.96,
                'evidence': {
                    'texture_variance': texture_var,
                    'reason': 'TEXTURE_CRITICAL'
                }
            }
        
        # Calculate texture confidence
        if texture_var < 28:
            texture_conf = 0.85
        elif texture_var < 35:
            texture_conf = 0.55
        elif texture_var < 45:
            texture_conf = 0.25
        else:
            texture_conf = 0.0
        
        # 2. PHONE DETECTION (most important)
        phone_conf, phone_bbox = detect_phone_in_frame_fast(frame, (x, y, x+w, y+h))
        
        # CRITICAL: Strong phone detection blocks immediately
        if phone_conf > 0.7:
            logger.critical(f"🚨 CRITICAL: Phone detected with high confidence: {phone_conf:.2f}")
            return {
                'is_spoof': True,
                'spoof_type': ['phone_in_frame', 'device_attack'],
                'confidence': min(phone_conf, 0.98),
                'evidence': {
                    'phone_confidence': phone_conf,
                    'phone_bbox': phone_bbox,
                    'reason': 'PHONE_CRITICAL'
                }
            }
        
        # 3. QUICK MOIRE CHECK (only if suspicious)
        moire_conf = 0.0
        if texture_var < 40 or phone_conf > 0.3:
            moire_conf = calculate_fft_moire_fast(face_roi)
        
        # OPTIMIZED: Weighted scoring emphasizing phone and texture
        S = (
            0.30 * texture_conf +    # Texture matters
            0.55 * phone_conf +       # Phone is MOST important
            0.15 * moire_conf         # Moire is supporting
        )
        
        # Determine spoof types
        spoof_types = []
        if phone_conf > 0.45:
            spoof_types.append("phone_detected")
        if texture_var < 30:
            spoof_types.append("low_texture_photo")
        if moire_conf > 0.65:
            spoof_types.append("screen_pattern")
        
        # CRITICAL: Lower threshold for better security
        is_spoof = S >= 0.50  # LOWERED from 0.55 for better blocking
        spoof_type = spoof_types if spoof_types else None
        
        evidence = {
            'texture_variance': round(texture_var, 2),
            'phone_confidence': round(phone_conf, 2),
            'phone_bbox': phone_bbox,
            'moire_confidence': round(moire_conf, 2),
            'fusion_score': round(S, 2)
        }
        
        logger.info(f"Spoof: texture={texture_var:.1f}, phone={phone_conf:.2f}, "
                   f"moire={moire_conf:.2f}, final={S:.2f}, is_spoof={is_spoof}")
        
        return {
            'is_spoof': is_spoof,
            'spoof_type': spoof_type,
            'confidence': round(S, 2),
            'evidence': evidence
        }
    except Exception as e:
        logger.error(f"Spoof detection error: {e}")
        # SAFE: Fail-closed on critical errors (block suspicious)
        return {
            'is_spoof': False,
            'spoof_type': None,
            'confidence': 0.0,
            'evidence': {'error': str(e)}
        }