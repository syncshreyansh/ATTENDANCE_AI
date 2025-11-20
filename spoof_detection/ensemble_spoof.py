"""
Ensemble Anti-Spoofing Module - BALANCED VERSION
Detects phones/photos but allows real faces through
"""
import cv2
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

_cnn_model = None
_yolo_model = None
_yolo_load_attempted = False
_cnn_load_attempted = False
_cnn_available = False

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
        _yolo_model.conf = 0.5
        logger.info("✓ YOLOv5 nano loaded for device detection")
        return _yolo_model
    except ImportError:
        logger.info("ultralytics not installed. Phone detection will use fallback.")
        return None
    except Exception as e:
        logger.info(f"Could not load YOLO: {e}. Using fallback.")
        return None

def calculate_laplacian_variance(face_roi):
    """Texture analysis - low variance = photo/screen"""
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var

def calculate_fft_moire(face_roi):
    """FFT analysis for moiré patterns (screen refresh artifacts)"""
    try:
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
        
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        high_freq_mask = np.ones((rows, cols), np.uint8)
        r = 30
        cv2.circle(high_freq_mask, (ccol, crow), r, 0, -1)
        
        high_freq_energy = np.sum(magnitude_spectrum * high_freq_mask)
        total_energy = np.sum(magnitude_spectrum)
        
        hf_ratio = high_freq_energy / (total_energy + 1e-6)
        moire_confidence = min(hf_ratio / 0.65, 1.0)
        return moire_confidence
    except Exception as e:
        logger.error(f"FFT error: {e}")
        return 0.0

def detect_phone_in_frame(frame, face_bbox):
    """Detect phone/tablet near face - PRIMARY SPOOF DETECTION"""
    model = load_yolo_model()
    if model is None:
        return check_phone_via_edges(frame, face_bbox)
    
    try:
        results = model(frame, verbose=False)
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
            if int(cls) not in phone_classes:
                continue
            
            phone_center_x = (x1 + x2) / 2
            phone_center_y = (y1 + y2) / 2
            dist = np.sqrt((phone_center_x - face_center_x)**2 + (phone_center_y - face_center_y)**2)
            phone_area = (x2 - x1) * (y2 - y1)
            
            # Check for direct overlap with face
            overlap_x = max(0, min(x2, fx2) - max(x1, fx1))
            overlap_y = max(0, min(y2, fy2) - max(y1, fy1))
            overlap_area = overlap_x * overlap_y
            
            # CRITICAL: Phone directly over face
            if overlap_area > face_area * 0.3:
                logger.warning(f"🚨 Phone overlapping face detected!")
                return 0.95, [int(x1), int(y1), int(x2-x1), int(y2-y1)]
            
            # Large phone/screen in frame
            if phone_area > frame_area * 0.15 and conf > 0.4:
                logger.warning(f"🚨 Large screen detected in frame")
                return min(float(conf) + 0.2, 0.90), [int(x1), int(y1), int(x2-x1), int(y2-y1)]
            
            # Phone near face
            if dist < 300 and conf > best_conf:
                best_conf = float(conf) + 0.15
                best_bbox = [int(x1), int(y1), int(x2-x1), int(y2-y1)]
        
        return best_conf, best_bbox
    except Exception as e:
        logger.error(f"YOLO error: {e}")
        return check_phone_via_edges(frame, face_bbox)

def check_phone_via_edges(frame, face_bbox):
    """Fallback: detect rectangular phone screen via edge detection"""
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        fx1, fy1, fx2, fy2 = face_bbox
        face_area = (fx2 - fx1) * (fy2 - fy1)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > face_area * 0.5:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h) if h > 0 else 0
                
                if 0.4 < aspect_ratio < 2.5 and area > 10000:
                    return 0.6, [x, y, w, h]
        
        return 0.0, None
    except Exception as e:
        logger.error(f"Edge detection error: {e}")
        return 0.0, None

def check(frame, face_bbox, face_encoding=None):
    """
    BALANCED spoof detection - blocks phones/photos but allows real faces
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
        
        # 1. TEXTURE ANALYSIS
        texture_var = calculate_laplacian_variance(face_roi)
        
        # FIXED: More reasonable texture thresholds
        if texture_var < 30:  # Extremely low = definite spoof
            texture_conf = 0.9
        elif texture_var < 40:
            texture_conf = 0.6
        elif texture_var < 50:
            texture_conf = 0.3
        else:
            texture_conf = 0.0  # Good texture = not a spoof
        
        # 2. PHONE DETECTION (most important)
        phone_conf, phone_bbox = detect_phone_in_frame(frame, (x, y, x+w, y+h))
        
        # 3. FFT MOIRE PATTERN
        moire_conf = calculate_fft_moire(face_roi)
        
        # EMERGENCY BLOCK: Very low texture
        if texture_var < 25:
            logger.critical(f"🚨 EMERGENCY: texture={texture_var:.1f} < 25")
            return {
                'is_spoof': True,
                'spoof_type': ['extremely_low_texture'],
                'confidence': 0.95,
                'evidence': {
                    'texture_variance': texture_var,
                    'reason': 'TEXTURE_TOO_LOW'
                }
            }
        
        # CRITICAL: Phone detected near face
        if phone_conf > 0.6:
            logger.critical(f"🚨 PHONE DETECTED: conf={phone_conf:.2f}")
            return {
                'is_spoof': True,
                'spoof_type': ['phone_in_frame'],
                'confidence': min(phone_conf, 0.98),
                'evidence': {
                    'phone_confidence': phone_conf,
                    'phone_bbox': phone_bbox,
                    'reason': 'PHONE_DETECTED'
                }
            }
        
        # FIXED: Balanced scoring
        S = (
            0.35 * texture_conf +    # Texture matters
            0.50 * phone_conf +       # Phone detection is KEY
            0.15 * moire_conf         # Moire is supporting evidence
        )
        
        # Determine spoof types
        spoof_types = []
        if phone_conf > 0.4:
            spoof_types.append("possible_phone")
        if texture_var < 35:
            spoof_types.append("low_texture")
        if moire_conf > 0.7:
            spoof_types.append("screen_pattern")
        
        # FIXED: Reasonable threshold
        is_spoof = S >= 0.55  # Balanced - not too strict
        spoof_type = spoof_types if spoof_types else None
        
        evidence = {
            'texture_variance': round(texture_var, 2),
            'phone_confidence': round(phone_conf, 2),
            'phone_bbox': phone_bbox,
            'moire_confidence': round(moire_conf, 2),
            'fusion_score': round(S, 2)
        }
        
        logger.info(f"Spoof check: texture={texture_var:.1f}, phone={phone_conf:.2f}, "
                   f"moire={moire_conf:.2f}, final={S:.2f}, is_spoof={is_spoof}")
        
        return {
            'is_spoof': is_spoof,
            'spoof_type': spoof_type,
            'confidence': round(S, 2),
            'evidence': evidence
        }
    except Exception as e:
        logger.error(f"Spoof detection error: {e}")
        import traceback
        traceback.print_exc()
        # FIXED: Fail-open on error (allow attendance)
        return {
            'is_spoof': False,
            'spoof_type': None,
            'confidence': 0.0,
            'evidence': {'error': str(e)}
        }