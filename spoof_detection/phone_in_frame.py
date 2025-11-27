"""
Phone-in-frame detection wrapper
"""
from spoof_detection.ensemble_spoof import detect_phone_in_frame

def check_phone_near_face(frame, face_bbox):
    """
    Wrapper for YOLO phone detection
    Returns: (phone_detected: bool, confidence: float, bbox: dict)
    """
    conf, bbox = detect_phone_in_frame(frame, face_bbox)
    return conf > 0.5, conf, bbox