"""
Enhanced Liveness Detection Service with MediaPipe Blink Detection
Combines MediaPipe (478 landmarks) for blinks + dlib for head pose
"""
import cv2
import numpy as np
import dlib
from scipy.spatial import distance as dist
import time
import logging

# MediaPipe imports
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("⚠️  MediaPipe not installed. Install with: pip install mediapipe")

logger = logging.getLogger(__name__)

class LivenessDetector:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        try:
            self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
            logger.info("✓ Liveness detector: dlib predictor loaded (for head pose)")
        except Exception as e:
            logger.error(f"✗ Failed to load dlib predictor: {e}")
            self.predictor = None
        
        # Initialize MediaPipe Face Mesh for blink detection
        if MEDIAPIPE_AVAILABLE:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            logger.info("✓ MediaPipe Face Mesh initialized for blink detection")
            
            # MediaPipe eye landmark indices (473 landmarks total)
            self.LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
            self.RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        else:
            self.face_mesh = None
            logger.warning("MediaPipe not available - falling back to dlib only")
        
        # Thresholds
        self.EAR_THRESHOLD = 0.22  # MediaPipe threshold
        self.HEAD_POSE_THRESHOLD = 45
        self.TEXTURE_THRESHOLD = 35
        
        # State tracking
        self.blink_counter = 0
        self.total_blinks = 0
        self.frame_check_counter = 0
        self.last_verification_time = 0
        self.verification_history = []
        
        # Blink detection state
        self.eyes_closed_frames = 0
        self.blink_in_progress = False
        self.last_blink_time = 0
    
    def calculate_ear_mediapipe(self, eye_landmarks):
        """
        Calculate Eye Aspect Ratio using MediaPipe landmarks
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        """
        try:
            # Vertical distances
            A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
            B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
            
            # Horizontal distance
            C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
            
            ear = (A + B) / (2.0 * C + 1e-6)
            return ear
        except Exception as e:
            logger.error(f"Error calculating EAR: {e}")
            return 0.25
    
    def detect_blink_mediapipe(self, frame):
        """
        Detect blink using MediaPipe Face Mesh
        Returns: (blink_detected, ear_value, details)
        """
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)
            
            if not results.multi_face_landmarks:
                return False, 0.0, {'error': 'No face detected'}
            
            # Get face landmarks
            face_landmarks = results.multi_face_landmarks[0]
            h, w, _ = frame.shape
            
            # Extract landmark coordinates
            landmarks_np = np.array([
                [landmark.x * w, landmark.y * h]
                for landmark in face_landmarks.landmark
            ])
            
            # Get eye landmarks
            left_eye = landmarks_np[self.LEFT_EYE_INDICES]
            right_eye = landmarks_np[self.RIGHT_EYE_INDICES]
            
            # Calculate EAR for both eyes
            left_ear = self.calculate_ear_mediapipe(left_eye)
            right_ear = self.calculate_ear_mediapipe(right_eye)
            ear = (left_ear + right_ear) / 2.0
            
            # Detect blink
            blink_detected = False
            
            if ear < self.EAR_THRESHOLD:
                # Eyes closing/closed
                self.eyes_closed_frames += 1
                if not self.blink_in_progress:
                    self.blink_in_progress = True
            else:
                # Eyes open
                if self.blink_in_progress and self.eyes_closed_frames >= 1:
                    # Blink completed!
                    self.total_blinks += 1
                    self.last_blink_time = time.time()
                    blink_detected = True
                    logger.info(f"✅ BLINK #{self.total_blinks} DETECTED! (MediaPipe, EAR={ear:.3f})")
                
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
            logger.error(f"MediaPipe blink detection error: {e}")
            return False, 0.0, {'error': str(e)}
    
    def calculate_ear_dlib(self, eye):
        """Fallback: Calculate EAR using dlib (68 landmarks)"""
        try:
            A = dist.euclidean(eye[1], eye[5])
            B = dist.euclidean(eye[2], eye[4])
            C = dist.euclidean(eye[0], eye[3])
            ear = (A + B) / (2.0 * C + 1e-6)
            return ear
        except Exception as e:
            logger.error(f"Error calculating EAR (dlib): {e}")
            return 0.25
    
    def estimate_head_pose(self, landmarks, frame_shape):
        """Estimate head pose using dlib 68 landmarks"""
        try:
            model_points = np.array([
                (0.0, 0.0, 0.0),
                (0.0, -330.0, -65.0),
                (-225.0, 170.0, -135.0),
                (225.0, 170.0, -135.0),
                (-150.0, -150.0, -125.0),
                (150.0, -150.0, -125.0)
            ], dtype=np.float64)
            
            image_points = np.array([
                landmarks[30],
                landmarks[8],
                landmarks[36],
                landmarks[45],
                landmarks[48],
                landmarks[54]
            ], dtype=np.float64)
            
            size = frame_shape
            focal_length = size[1]
            center = (size[1] / 2, size[0] / 2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)
            
            dist_coeffs = np.zeros((4, 1))
            
            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if not success:
                return 0, 0, 0
            
            rotation_mat, _ = cv2.Rodrigues(rotation_vector)
            pose_mat = cv2.hconcat((rotation_mat, translation_vector))
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
            
            pitch = float(euler_angles[0][0])
            yaw = float(euler_angles[1][0])
            roll = float(euler_angles[2][0])
            
            return pitch, yaw, roll
        except Exception as e:
            logger.error(f"Error estimating head pose: {e}")
            return 0, 0, 0
    
    def detect_texture_quality(self, face_roi):
        """Analyze texture - real faces have more detail"""
        try:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            return laplacian_var
        except Exception as e:
            logger.error(f"Error detecting texture: {e}")
            return 0
    
    def comprehensive_liveness_check(self, frame):
        """
        ENHANCED: Comprehensive liveness check with MediaPipe blink detection
        Returns: (is_live, confidence, details)
        """
        try:
            if self.predictor is None:
                logger.warning("dlib predictor not loaded - passing by default")
                return True, 0.6, {
                    'blink_detected': True,
                    'head_pose_correct': True,
                    'texture_valid': True,
                    'note': 'predictor_unavailable',
                    'scores': {'blink': 1.0, 'texture': 1.0, 'head_pose': 1.0}
                }
            
            # Initialize scores
            verification_scores = {
                'blink': 0,
                'texture': 0,
                'head_pose': 0
            }
            
            # 1. BLINK DETECTION - Try MediaPipe first, fallback to dlib
            if MEDIAPIPE_AVAILABLE and self.face_mesh:
                blink_detected, ear, blink_details = self.detect_blink_mediapipe(frame)
                
                if 'error' not in blink_details:
                    # MediaPipe succeeded
                    if blink_detected:
                        verification_scores['blink'] = 1.0
                    elif ear < self.EAR_THRESHOLD:
                        verification_scores['blink'] = 0.5  # Eyes closing
                    
                    logger.debug(f"MediaPipe blink: ear={ear:.3f}, blinks={blink_details.get('total_blinks', 0)}")
                else:
                    # MediaPipe failed, use dlib fallback
                    logger.debug("MediaPipe failed, using dlib fallback for blink")
                    verification_scores['blink'] = self._dlib_blink_fallback(frame)
            else:
                # No MediaPipe, use dlib
                verification_scores['blink'] = self._dlib_blink_fallback(frame)
            
            # 2. HEAD POSE (using dlib)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)
            
            if len(faces) > 0:
                face = faces[0]
                landmarks = self.predictor(gray, face)
                landmarks_np = np.array([(p.x, p.y) for p in landmarks.parts()])
                
                pitch, yaw, roll = self.estimate_head_pose(landmarks_np, frame.shape)
                
                if abs(pitch) < self.HEAD_POSE_THRESHOLD and abs(yaw) < self.HEAD_POSE_THRESHOLD:
                    verification_scores['head_pose'] = 1.0
                elif abs(pitch) < 60 and abs(yaw) < 60:
                    verification_scores['head_pose'] = 0.7
                else:
                    verification_scores['head_pose'] = 0.4
                
                # 3. TEXTURE ANALYSIS
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                face_roi = frame[y:y+h, x:x+w]
                
                if face_roi.size > 0:
                    texture_quality = self.detect_texture_quality(face_roi)
                    if texture_quality >= self.TEXTURE_THRESHOLD:
                        verification_scores['texture'] = 1.0
                    elif texture_quality >= 25:
                        verification_scores['texture'] = 0.6
                    else:
                        verification_scores['texture'] = 0.3
            else:
                # No face detected by dlib
                verification_scores['head_pose'] = 0.0
                verification_scores['texture'] = 0.0
            
            # Calculate confidence
            blink_score = verification_scores['blink']
            texture_score = verification_scores['texture']
            head_pose_score = verification_scores['head_pose']
            
            confidence = (
                texture_score * 0.5 +      # Texture most important
                head_pose_score * 0.3 +    # Head pose
                blink_score * 0.2          # Blink is bonus
            )
            
            is_live = confidence >= 0.5
            
            details = {
                'blink_detected': blink_score >= 0.5,
                'head_pose_correct': head_pose_score > 0,
                'texture_valid': texture_score > 0,
                'total_blinks': self.total_blinks,
                'scores': verification_scores,
                'method': 'mediapipe' if MEDIAPIPE_AVAILABLE else 'dlib'
            }
            
            logger.info(f"Liveness: conf={confidence:.2f}, texture={texture_score:.2f}, "
                       f"head={head_pose_score:.2f}, blink={blink_score:.2f}, method={details['method']}")
            
            return is_live, confidence, details
            
        except Exception as e:
            logger.error(f"Error in liveness detection: {e}")
            import traceback
            traceback.print_exc()
            return True, 0.5, {'error': str(e), 'fail_open': True}
    
    def _dlib_blink_fallback(self, frame):
        """Fallback blink detection using dlib"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)
            
            for face in faces:
                landmarks = self.predictor(gray, face)
                landmarks_np = np.array([(p.x, p.y) for p in landmarks.parts()])
                
                left_eye = landmarks_np[42:48]
                right_eye = landmarks_np[36:42]
                
                left_ear = self.calculate_ear_dlib(left_eye)
                right_ear = self.calculate_ear_dlib(right_eye)
                ear = (left_ear + right_ear) / 2.0
                
                if ear < 0.20:  # dlib threshold
                    self.eyes_closed_frames += 1
                    if not self.blink_in_progress:
                        self.blink_in_progress = True
                else:
                    if self.blink_in_progress and self.eyes_closed_frames >= 1:
                        self.total_blinks += 1
                        logger.info(f"✅ BLINK #{self.total_blinks} (dlib fallback)")
                        self.eyes_closed_frames = 0
                        self.blink_in_progress = False
                        return 1.0
                    
                    self.eyes_closed_frames = 0
                    self.blink_in_progress = False
                
                if ear < 0.20:
                    return 0.5
            
            return 0.0
        except Exception as e:
            logger.error(f"dlib fallback error: {e}")
            return 0.0
    
    def reset_session(self):
        """Reset session tracking for new user"""
        self.blink_counter = 0
        self.total_blinks = 0
        self.frame_check_counter = 0
        self.verification_history = []
        self.eyes_closed_frames = 0
        self.blink_in_progress = False
        self.last_blink_time = 0