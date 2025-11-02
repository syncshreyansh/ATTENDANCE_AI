"""
Ensemble Anti-Spoofing Module
Combines texture analysis, CNN classification, object detection, FFT/moiré, and reflection checks
"""
import cv2
import numpy as np
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)

# Lazy-load heavy models
_cnn_model = None
_yolo_model = None
_yolo_load_attempted = False

def load_cnn_model():
    """Load ONNX anti-spoof CNN model (ResNet18 or MobileNetV2)"""
    global _cnn_model
    if _cnn_model is not None:
        return _cnn_model
    
    model_path = 'models/anti_spoof_resnet18.onnx'
    if not os.path.exists(model_path):
        logger.warning(f"CNN model not found at {model_path}, skipping CNN check")
        return None
    
    try:
        import onnxruntime as ort
        _cnn_model = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        logger.info("CNN anti-spoof model loaded (ONNX)")
        return _cnn_model
    except Exception as e:
        logger.error(f"Failed to load CNN model: {e}")
        return None

def load_yolo_model():
    """Load YOLOv5 nano for phone/tablet/screen detection"""
    global _yolo_model, _yolo_load_attempted
    
    if _yolo_model is not None:
        return _yolo_model
    
    # Only attempt to load once to avoid repeated errors
    if _yolo_load_attempted:
        return None
    
    _yolo_load_attempted = True
    
    model_path = 'models/yolov5n.pt'
    if not os.path.exists(model_path):
        logger.warning(f"YOLO model not found at {model_path}, skipping object detection")
        logger.info("Phone detection will be disabled. To enable, download: wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt -O models/yolov5n.pt")
        return None
    
    try:
        # Use ultralytics library directly instead of torch.hub
        from ultralytics import YOLO
        _yolo_model = YOLO(model_path)
        _yolo_model.conf = 0.5
        logger.info("YOLOv5 nano loaded for device detection")
        return _yolo_model
    except ImportError:
        logger.warning("ultralytics library not installed. Install with: pip install ultralytics")
        return None
    except Exception as e:
        logger.warning(f"Could not load YOLO model: {e}")
        logger.info("Phone detection will be disabled. System will work with other spoof checks.")
        return None

def calculate_laplacian_variance(face_roi):
    """Texture analysis: low variance indicates printed photo or screen"""
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var

def calculate_fft_moire(face_roi):
    """FFT analysis for moiré patterns and refresh banding (screen artifacts)"""
    try:
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
        
        # High-frequency energy ratio indicates moiré/screen patterns
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        high_freq_mask = np.ones((rows, cols), np.uint8)
        r = 30  # radius of low-freq region to exclude
        cv2.circle(high_freq_mask, (ccol, crow), r, 0, -1)
        
        high_freq_energy = np.sum(magnitude_spectrum * high_freq_mask)
        total_energy = np.sum(magnitude_spectrum)
        
        hf_ratio = high_freq_energy / (total_energy + 1e-6)
        
        # Threshold: ratio > 0.65 suggests screen artifacts
        moire_confidence = min(hf_ratio / 0.65, 1.0)
        return moire_confidence
    except Exception as e:
        logger.error(f"FFT moire calculation error: {e}")
        return 0.0

def reflection_in_eyes_score(face_roi):
    """Detect specular highlights in eye regions (real faces have natural reflections)"""
    try:
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Simple heuristic: detect bright spots in upper half (eye region)
        h, w = gray.shape
        eye_region = gray[int(h*0.2):int(h*0.5), :]
        
        _, bright_mask = cv2.threshold(eye_region, 220, 255, cv2.THRESH_BINARY)
        bright_ratio = np.sum(bright_mask) / (255.0 * eye_region.size + 1e-6)
        
        # Real faces: 0.01-0.05, photos: <0.005
        reflection_conf = min(bright_ratio / 0.02, 1.0)
        return reflection_conf
    except Exception as e:
        logger.error(f"Reflection calculation error: {e}")
        return 0.0

def run_cnn_classifier(face_roi):
    """Run CNN to classify live vs photo vs screen"""
    model = load_cnn_model()
    if model is None:
        return 0.0, "cnn_model_unavailable"
    
    try:
        # Preprocess: resize to 224x224, normalize
        input_img = cv2.resize(face_roi, (224, 224))
        input_img = input_img.astype(np.float32) / 255.0
        input_img = np.transpose(input_img, (2, 0, 1))  # CHW
        input_img = np.expand_dims(input_img, 0)  # NCHW
        
        # Run inference
        input_name = model.get_inputs()[0].name
        outputs = model.run(None, {input_name: input_img})
        
        # Softmax output: [live, printed_photo, phone_screen]
        probabilities = outputs[0][0]
        spoof_conf = max(probabilities[1], probabilities[2])  # max of photo or screen
        spoof_type = "printed_photo" if probabilities[1] > probabilities[2] else "phone_screen"
        
        return spoof_conf, spoof_type
    except Exception as e:
        logger.error(f"CNN inference error: {e}")
        return 0.0, "cnn_error"

def detect_phone_in_frame(frame, face_bbox):
    """Detect phone/tablet near face using YOLO"""
    model = load_yolo_model()
    if model is None:
        return 0.0, None
    
    try:
        # Run YOLO inference
        results = model(frame, verbose=False)
        
        # Extract detections
        detections = results[0].boxes.data.cpu().numpy() if len(results) > 0 else []
        
        # Classes: 67=cell phone, 73=laptop (COCO dataset)
        phone_classes = [67, 73]
        
        fx1, fy1, fx2, fy2 = face_bbox
        face_center_x = (fx1 + fx2) / 2
        face_center_y = (fy1 + fy2) / 2
        
        best_conf = 0.0
        best_bbox = None
        
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            if int(cls) in phone_classes:
                # Check proximity to face
                phone_center_x = (x1 + x2) / 2
                phone_center_y = (y1 + y2) / 2
                dist = np.sqrt((phone_center_x - face_center_x)**2 + (phone_center_y - face_center_y)**2)
                
                # If phone within 300px of face center, flag
                if dist < 300 and conf > best_conf:
                    best_conf = float(conf)
                    best_bbox = [int(x1), int(y1), int(x2-x1), int(y2-y1)]  # [x, y, w, h]
        
        return best_conf, best_bbox
    except Exception as e:
        logger.error(f"YOLO detection error: {e}")
        return 0.0, None

def check(frame, face_bbox, face_encoding=None):
    """
    Main ensemble spoof detection
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
        
        # Run all checks
        texture_var = calculate_laplacian_variance(face_roi)
        texture_conf = 1.0 if texture_var < 50 else 0.0  # Low variance = spoof
        
        moire_conf = calculate_fft_moire(face_roi)
        
        reflection_conf = reflection_in_eyes_score(face_roi)
        # Invert: low reflection = spoof
        reflection_spoof_conf = 1.0 - reflection_conf if reflection_conf < 0.3 else 0.0
        
        cnn_conf, cnn_type = run_cnn_classifier(face_roi)
        
        phone_conf, phone_bbox = detect_phone_in_frame(frame, (x, y, x+w, y+h))
        
        # Blink check is handled externally by liveness_detection.py
        blink_conf = 0.0
        
        # Weighted fusion score
        # S = 0.25*cnn + 0.2*texture + 0.2*phone + 0.15*moire + 0.1*reflection + 0.1*(1-blink)
        S = (0.25 * cnn_conf +
             0.2 * texture_conf +
             0.2 * phone_conf +
             0.15 * moire_conf +
             0.1 * reflection_spoof_conf +
             0.1 * (1 - blink_conf))
        
        # Calculate reliability based on image quality
        reliability = 1.0
        avg_brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if avg_brightness < 30:
            reliability *= 0.5
            logger.warning("Low-light detected, spoof detection reliability reduced")
        
        if face_roi.size < 10000:  # Very small face
            reliability *= 0.7
            logger.warning("Small/distant face, reliability reduced")
        
        # Determine spoof types
        spoof_types = []
        if cnn_conf > 0.6:
            spoof_types.append(cnn_type)
        if phone_conf > 0.5:
            spoof_types.append("phone_in_frame")
        if moire_conf > 0.7:
            spoof_types.append("screen_refresh_banding_detected")
        if texture_var < 30:
            spoof_types.append("low_texture_printed_photo")
        
        # Decision thresholds
        is_spoof = S >= 0.7
        spoof_type = spoof_types if spoof_types else None
        
        evidence = {
            'texture_variance': round(texture_var, 2),
            'moire_confidence': round(moire_conf, 2),
            'reflection_confidence': round(reflection_conf, 2),
            'cnn_confidence': round(cnn_conf, 2),
            'cnn_type': cnn_type,
            'phone_detector_confidence': round(phone_conf, 2),
            'phone_bbox': phone_bbox,
            'fusion_score': round(S, 2),
            'reliability_score': round(reliability, 2)
        }
        
        # If unreliable, downweight confidence
        if reliability < 0.6 and is_spoof:
            logger.info("Low reliability - downweighting confidence")
            S = S * reliability
            spoof_types = ['low_reliability_check'] + (spoof_types if spoof_types else [])
        
        return {
            'is_spoof': is_spoof,
            'spoof_type': spoof_type,
            'confidence': round(S, 2),
            'evidence': evidence
        }
    except Exception as e:
        logger.error(f"Error in spoof detection: {e}")
        return {
            'is_spoof': False,
            'spoof_type': None,
            'confidence': 0.0,
            'evidence': {'error': str(e)}
        }