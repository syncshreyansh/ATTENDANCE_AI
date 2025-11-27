"""
Improved Blink Detection using MediaPipe Face Mesh
More reliable than dlib EAR method
"""
import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
import logging
import time

logger = logging.getLogger(__name__)

class MediaPipeBlinkDetector:
    def __init__(self):
        """Initialize MediaPipe Face Mesh for blink detection"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # MediaPipe eye landmark indices
        # Left eye: [362, 385, 387, 263, 373, 380]
        # Right eye: [33, 160, 158, 133, 153, 144]
        self.LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        
        # Blink detection parameters
        self.EAR_THRESHOLD = 0.22  # More sensitive than dlib's 0.20
        self.CONSECUTIVE_FRAMES = 2  # Frames with closed eyes to confirm blink
        
        # State tracking
        self.blink_counter = 0
        self.total_blinks = 0
        self.eyes_closed_frames = 0
        self.blink_in_progress = False
        self.last_blink_time = 0
        
        logger.info("✓ MediaPipe Blink Detector initialized")
    
    def calculate_ear(self, eye_landmarks):
        """
        Calculate Eye Aspect Ratio
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        """
        try:
            # Vertical eye distances
            A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
            B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
            
            # Horizontal eye distance
            C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
            
            # Calculate EAR
            ear = (A + B) / (2.0 * C + 1e-6)
            return ear
        except Exception as e:
            logger.error(f"Error calculating EAR: {e}")
            return 0.3  # Default value
    
    def detect_blink(self, frame):
        """
        Detect blink in frame using MediaPipe
        Returns: (blink_detected, ear_value, details)
        """
        try:
            # Convert to RGB (MediaPipe requires RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)
            
            if not results.multi_face_landmarks:
                return False, 0.0, {'error': 'No face detected'}
            
            # Get face landmarks
            face_landmarks = results.multi_face_landmarks[0]
            h, w, _ = frame.shape
            
            # Extract eye landmark coordinates
            landmarks_np = np.array([
                [landmark.x * w, landmark.y * h]
                for landmark in face_landmarks.landmark
            ])
            
            # Get left and right eye landmarks
            left_eye = landmarks_np[self.LEFT_EYE_INDICES]
            right_eye = landmarks_np[self.RIGHT_EYE_INDICES]
            
            # Calculate EAR for both eyes
            left_ear = self.calculate_ear(left_eye)
            right_ear = self.calculate_ear(right_eye)
            ear = (left_ear + right_ear) / 2.0
            
            # Detect blink
            blink_detected = False
            
            if ear < self.EAR_THRESHOLD:
                # Eyes are closing/closed
                self.eyes_closed_frames += 1
                if not self.blink_in_progress:
                    self.blink_in_progress = True
                    logger.info(f"👁️ Eyes closing... EAR={ear:.3f}")
            else:
                # Eyes are open
                if self.blink_in_progress and self.eyes_closed_frames >= self.CONSECUTIVE_FRAMES:
                    # Blink completed!
                    self.total_blinks += 1
                    self.last_blink_time = time.time()
                    blink_detected = True
                    logger.info(f"✅ BLINK #{self.total_blinks} DETECTED! EAR={ear:.3f}")
                
                # Reset
                self.eyes_closed_frames = 0
                self.blink_in_progress = False
            
            details = {
                'ear': ear,
                'left_ear': left_ear,
                'right_ear': right_ear,
                'eyes_closed_frames': self.eyes_closed_frames,
                'total_blinks': self.total_blinks,
                'threshold': self.EAR_THRESHOLD,
                'blink_in_progress': self.blink_in_progress
            }
            
            return blink_detected, ear, details
            
        except Exception as e:
            logger.error(f"Error in blink detection: {e}")
            return False, 0.0, {'error': str(e)}
    
    def has_blinked(self):
        """Check if user has blinked at least once"""
        return self.total_blinks > 0
    
    def reset(self):
        """Reset blink counter for new user"""
        self.blink_counter = 0
        self.total_blinks = 0
        self.eyes_closed_frames = 0
        self.blink_in_progress = False
        self.last_blink_time = 0
        logger.info("🔄 Blink detector reset")
    
    def get_stats(self):
        """Get blink statistics"""
        return {
            'total_blinks': self.total_blinks,
            'last_blink_seconds_ago': time.time() - self.last_blink_time if self.last_blink_time > 0 else None
        }

# Test function
def test_blink_detector():
    """Test blink detection with webcam"""
    detector = MediaPipeBlinkDetector()
    cap = cv2.VideoCapture(0)
    
    print("=" * 60)
    print("BLINK DETECTION TEST")
    print("=" * 60)
    print("Instructions:")
    print("1. Look at camera")
    print("2. Blink slowly and deliberately")
    print("3. Press 'q' to quit")
    print("=" * 60)
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect blink every frame
        blink_detected, ear, details = detector.detect_blink(frame)
        
        # Display feedback
        status_text = f"EAR: {ear:.3f} | Blinks: {detector.total_blinks}"
        if detector.blink_in_progress:
            status_text += " | CLOSING..."
        if blink_detected:
            status_text += " | BLINK!"
        
        cv2.putText(frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Blink Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print(f"Test Complete! Total blinks detected: {detector.total_blinks}")
    print("=" * 60)

if __name__ == "__main__":
    test_blink_detector()
