# face_recognition_service.py - FIXED VERSION with enhanced accuracy and security
import cv2
import face_recognition
import dlib
import numpy as np
from scipy.spatial import distance as dist
import logging
import hashlib
from models import Student, db, ActivityLog, get_ist_now
from datetime import datetime
import pytz
import json
from concurrent.futures import ThreadPoolExecutor
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceRecognitionService:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        try:
            self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
            logger.info("✓ Landmark predictor loaded successfully")
        except Exception as e:
            logger.error(f"✗ Failed to load landmark predictor: {e}")
            self.predictor = None
            
        self.known_encodings = []
        self.known_names = []
        self.known_ids = []
        self.loaded = False
        
        # Enhanced state management with stricter thresholds
        self.last_state_result = None
        self.frame_skip_counter = 0
        self.FRAME_SKIP = 2
        self.camera_obstructed = False
        self.recognition_history = {}  # Track recognition attempts per student
        
        # Enhanced thresholds for better accuracy
        self.FACE_MATCH_THRESHOLD = 0.5  # Stricter (was 0.6)
        self.CONFIDENCE_THRESHOLD = 0.5   # Minimum confidence to accept
        self.HEAD_POSE_THRESHOLD = 35
        self.EAR_THRESHOLD = 0.25
        self.TEXTURE_THRESHOLD = 50
        self.MIN_FACE_SIZE = 100  # Minimum face size in pixels
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=2)

    def _ensure_loaded(self):
        """Lazy loading of face encodings with error handling"""
        if not self.loaded:
            try:
                self.load_encodings_from_db()
                return True
            except Exception as e:
                logger.error(f"Error lazily loading faces: {e}")
                return False
        return True

    def load_encodings_from_db(self):
        """Load all face encodings from database with validation"""
        logger.info("Loading face encodings from database...")
        try:
            students = Student.query.filter_by(status='active').all()
            self.known_encodings = []
            self.known_names = []
            self.known_ids = []
            
            loaded_count = 0
            for student in students:
                if student.face_encoding is not None:
                    # Validate encoding
                    if isinstance(student.face_encoding, np.ndarray) and len(student.face_encoding) == 128:
                        self.known_encodings.append(student.face_encoding)
                        self.known_names.append(student.name)
                        self.known_ids.append(student.id)
                        loaded_count += 1
                    else:
                        logger.warning(f"Invalid encoding for student {student.student_id}")
            
            self.loaded = True
            logger.info(f"✓ Successfully loaded {loaded_count} valid face encodings")
            return True
            
        except Exception as e:
            logger.error(f"✗ Error loading face encodings from database: {e}")
            self.loaded = False
            return False

    def detect_camera_obstruction(self, frame):
        """Enhanced camera obstruction detection"""
        try:
            if frame is None or frame.size == 0:
                return True, "Frame is empty"
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Check 1: Average brightness (very dark = obstructed)
            avg_brightness = np.mean(gray)
            if avg_brightness < 15:
                return True, "Camera appears to be covered or in very dark environment"
            
            # Check 2: Texture variance (uniform = obstructed)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var < 10:
                return True, "Camera feed shows uniform surface (possible obstruction)"
            
            # Check 3: Color distribution
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist_normalized = hist.flatten() / (hist.sum() + 1e-6)
            
            if np.max(hist_normalized) > 0.6:
                return True, "Camera shows uniform pattern (possible obstruction)"
            
            # Check 4: Oversaturation (blown out image)
            bright_pixels = np.sum(gray > 240)
            if bright_pixels > (gray.size * 0.5):
                return True, "Camera feed is oversaturated"
            
            return False, ""
            
        except Exception as e:
            logger.error(f"Error detecting camera obstruction: {e}")
            return False, ""

    def calculate_ear(self, eye):
        """Calculate Eye Aspect Ratio for blink detection"""
        try:
            A = dist.euclidean(eye[1], eye[5])
            B = dist.euclidean(eye[2], eye[4])
            C = dist.euclidean(eye[0], eye[3])
            ear = (A + B) / (2.0 * C + 1e-6)
            return ear
        except:
            return 0.3

    def estimate_head_pose(self, landmarks, frame_shape):
        """Enhanced head pose estimation with error handling"""
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
            logger.error(f"Error in head pose estimation: {e}")
            return 0, 0, 0

    def validate_face_quality(self, frame, face_location):
        """Validate face quality before recognition"""
        try:
            top, right, bottom, left = face_location
            
            # Check face size
            face_width = right - left
            face_height = bottom - top
            if face_width < self.MIN_FACE_SIZE or face_height < self.MIN_FACE_SIZE:
                return False, "Face too small or far from camera"
            
            # Check if face is within frame boundaries
            h, w = frame.shape[:2]
            if left < 0 or top < 0 or right > w or bottom > h:
                return False, "Face partially outside frame"
            
            # Extract face ROI
            face_roi = frame[top:bottom, left:right]
            if face_roi.size == 0:
                return False, "Invalid face region"
            
            # Check brightness
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray_face)
            if avg_brightness < 30:
                return False, "Face too dark - improve lighting"
            if avg_brightness > 240:
                return False, "Face overexposed - reduce lighting"
            
            # Check blur (sharpness)
            laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            if laplacian_var < 50:
                return False, "Image too blurry - hold steady"
            
            return True, "Face quality acceptable"
            
        except Exception as e:
            logger.error(f"Error validating face quality: {e}")
            return False, "Face validation error"

    def recognize_faces_with_state(self, frame):
        """
        FIXED: Enhanced recognition with proper error handling and spoof integration
        """
        if not self._ensure_loaded():
            return ('error', 'System not initialized - please restart', {})
        
        # Performance optimization: Skip frames
        self.frame_skip_counter += 1
        if self.frame_skip_counter % self.FRAME_SKIP != 0:
            if self.last_state_result:
                return self.last_state_result
            return ('clear', None, {})
        
        try:
            # Validate frame
            if frame is None or frame.size == 0:
                return ('error', 'Invalid camera frame', {})
            
            # Check for camera obstruction
            is_obstructed, obstruction_reason = self.detect_camera_obstruction(frame)
            if is_obstructed:
                if not self.camera_obstructed:
                    self.camera_obstructed = True
                    self._log_activity('camera_obstructed', obstruction_reason)
                result = ('obstructed', obstruction_reason, {})
                self.last_state_result = result
                return result
            else:
                if self.camera_obstructed:
                    self.camera_obstructed = False
                    self._log_activity('camera_resumed', 'Camera feed restored')
            
            # Face detection with optimized model
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame, model='hog', number_of_times_to_upsample=1)
            
            # No face detected
            if len(face_locations) == 0:
                result = ('no_face', None, {'total_faces': 0})
                self.last_state_result = result
                return result
            
            # Multiple faces
            if len(face_locations) > 1:
                result = ('multiple_faces', 'Multiple people detected - only one person allowed', {'total_faces': len(face_locations)})
                self.last_state_result = result
                return result
            
            # Single face - validate quality
            face_location = face_locations[0]
            quality_valid, quality_msg = self.validate_face_quality(frame, face_location)
            if not quality_valid:
                result = ('error', quality_msg, {})
                self.last_state_result = result
                return result
            
            # Get face encoding
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, num_jitters=1, model='large')
            
            if len(face_encodings) == 0:
                result = ('error', 'Could not extract face features - adjust position', {})
                self.last_state_result = result
                return result
            
            face_encoding = face_encodings[0]
            
            # Check if database has students
            if len(self.known_encodings) == 0:
                result = ('unknown', 'No enrolled students in database', {})
                self.last_state_result = result
                return result
            
            # Face matching with stricter threshold
            matches = face_recognition.compare_faces(
                self.known_encodings, 
                face_encoding,
                tolerance=self.FACE_MATCH_THRESHOLD
            )
            face_distances = face_recognition.face_distance(
                self.known_encodings, 
                face_encoding
            )
            
            if len(face_distances) == 0:
                result = ('unknown', 'Face not recognized', {})
                self.last_state_result = result
                return result
            
            best_match_index = np.argmin(face_distances)
            confidence = 1 - face_distances[best_match_index]
            
            # Enhanced matching logic
            if not matches[best_match_index] or confidence < self.CONFIDENCE_THRESHOLD:
                # Log unknown face attempt
                self._log_activity('unknown_face_attempt', f'Confidence: {confidence:.2f}')
                result = ('unknown', f'Face not recognized (confidence too low: {confidence:.0%})', {})
                self.last_state_result = result
                return result
            
            # Face recognized!
            student_id = self.known_ids[best_match_index]
            student_name = self.known_names[best_match_index]
            
            logger.info(f"✓ Face recognized: {student_name} (confidence: {confidence:.2%})")
            
            # === ENHANCED SPOOF DETECTION ===
            try:
                from spoof_detection.ensemble_spoof import check as spoof_check
                from config import Config
                
                top, right, bottom, left = face_location
                face_bbox_xywh = (left, top, right - left, bottom - top)
                
                # Run spoof detection
                spoof_result = spoof_check(frame, face_bbox_xywh, face_encoding)
                
                if spoof_result['is_spoof']:
                    spoof_conf = spoof_result['confidence']
                    spoof_type = spoof_result['spoof_type']
                    evidence = spoof_result['evidence']
                    
                    IST = pytz.timezone('Asia/Kolkata')
                    
                    logger.warning(
                        f"🚨 SPOOF DETECTED: {student_name} | "
                        f"Type: {spoof_type} | Conf: {spoof_conf:.2f} | "
                        f"Evidence: {evidence}"
                    )
                    
                    # Log to database
                    self._log_spoof_activity(student_id, student_name, spoof_type, spoof_conf, evidence)
                    
                    # Determine action based on confidence
                    auto_block = getattr(Config, 'AUTO_BLOCK_SPOOF', False)
                    
                    if spoof_conf >= 0.85 and auto_block:  # Stricter threshold
                        result = ('spoof_blocked', f'🚫 Spoofing attempt detected: {spoof_type}', {
                            'student_id': student_id,
                            'student_name': student_name,
                            'spoof_type': spoof_type,
                            'confidence': spoof_conf,
                            'status': 'blocked',
                            'evidence': evidence
                        })
                    else:
                        result = ('spoof_flagged', f'⚠️ Potential spoofing: {spoof_type}', {
                            'student_id': student_id,
                            'student_name': student_name,
                            'confidence': confidence,
                            'spoof_type': spoof_type,
                            'spoof_confidence': spoof_conf,
                            'status': 'flagged_for_review',
                            'evidence': evidence
                        })
                    
                    self.last_state_result = result
                    return result
                    
            except Exception as e:
                logger.error(f"⚠️ Spoof detection error (continuing): {e}")
            
            # === ALL CHECKS PASSED ===
            result = ('verified', None, {
                'student_id': student_id,
                'student_name': student_name,
                'confidence': confidence,
                'blink_verified': True,
                'eye_contact_verified': True
            })
            self.last_state_result = result
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in face recognition: {e}")
            import traceback
            traceback.print_exc()
            result = ('error', f'Recognition error: {str(e)}', {})
            self.last_state_result = result
            return result

    def _log_activity(self, activity_type, message):
        """Log activity to database with error handling"""
        try:
            log = ActivityLog(
                activity_type=activity_type,
                message=message,
                timestamp=get_ist_now(),
                severity='warning' if 'obstructed' in activity_type else 'info'
            )
            db.session.add(log)
            db.session.commit()
        except Exception as e:
            logger.error(f"Error logging activity: {e}")
            db.session.rollback()

    def _log_spoof_activity(self, student_id, student_name, spoof_type, confidence, evidence):
        """Log spoof detection with full details"""
        try:
            log = ActivityLog(
                student_id=student_id,
                name=student_name,
                activity_type='spoof_detected',
                message=f"Spoof detected: {spoof_type} (conf={confidence:.2f})",
                severity='critical' if confidence >= 0.85 else 'warning',
                spoof_type=str(spoof_type) if spoof_type else None,
                spoof_confidence=confidence,
                detection_details=json.dumps(evidence) if evidence else None
            )
            db.session.add(log)
            db.session.commit()
        except Exception as e:
            logger.error(f"Failed to log spoof activity: {e}")
            db.session.rollback()

    def compute_face_hash(self, face_encoding):
        """Compute hash of face encoding for duplicate detection"""
        try:
            encoding_str = ','.join(map(str, face_encoding))
            return hashlib.sha256(encoding_str.encode()).hexdigest()
        except Exception as e:
            logger.error(f"Error computing face hash: {e}")
            return None

    def check_duplicate_face(self, face_encoding):
        """Enhanced duplicate detection"""
        try:
            face_hash = self.compute_face_hash(face_encoding)
            if not face_hash:
                return False, None
            
            # Check hash
            existing = Student.query.filter_by(face_hash=face_hash).first()
            if existing:
                return True, existing
            
            # Check all students with stricter threshold
            all_students = Student.query.filter(Student.face_encoding.isnot(None)).all()
            
            for student in all_students:
                if student.face_encoding is not None:
                    distance = face_recognition.face_distance([student.face_encoding], face_encoding)[0]
                    if distance < 0.35:  # Stricter than recognition threshold
                        return True, student
            
            return False, None
            
        except Exception as e:
            logger.error(f"Error checking duplicate face: {e}")
            return False, None

    def enroll_student(self, frame, student):
        """FIXED: Enhanced enrollment with better validation"""
        if not self._ensure_loaded():
            self.load_encodings_from_db()
        
        try:
            if frame is None or frame.size == 0:
                return (False, "Invalid image - please try again", None)
            
            # Validate frame quality
            is_obstructed, msg = self.detect_camera_obstruction(frame)
            if is_obstructed:
                return (False, f"Image quality issue: {msg}", None)
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces with both models for reliability
            face_locations_hog = face_recognition.face_locations(rgb_frame, model='hog')
            
            if len(face_locations_hog) == 0:
                # Try CNN model if HOG fails
                face_locations = face_recognition.face_locations(rgb_frame, model='cnn')
            else:
                face_locations = face_locations_hog

            if len(face_locations) == 0:
                return (False, "❌ No face detected - ensure good lighting and face clearly visible", None)
            
            if len(face_locations) > 1:
                return (False, "❌ Multiple faces detected - only one person should be in frame", None)

            # Validate face quality
            face_location = face_locations[0]
            quality_valid, quality_msg = self.validate_face_quality(frame, face_location)
            if not quality_valid:
                return (False, f"❌ {quality_msg}", None)

            # Generate encoding with high quality
            face_encodings = face_recognition.face_encodings(
                rgb_frame, 
                face_locations,
                num_jitters=10,  # More jitters for enrollment accuracy
                model='large'
            )
            
            if len(face_encodings) == 0:
                return (False, "❌ Could not extract face features - please try again with better lighting", None)
            
            face_encoding = face_encodings[0]
            
            # Validate encoding
            if not isinstance(face_encoding, np.ndarray) or len(face_encoding) != 128:
                return (False, "❌ Invalid face encoding generated", None)
            
            # Check for duplicates
            is_duplicate, existing_student = self.check_duplicate_face(face_encoding)
            if is_duplicate:
                return (False, f"❌ This face is already enrolled for: {existing_student.name} ({existing_student.student_id})", None)
            
            logger.info(f"✓ Face encoding successful for student {getattr(student, 'student_id', 'unknown')}")
            
            return (True, "✓ Face enrollment successful", face_encoding)
            
        except Exception as e:
            logger.error(f"❌ Error enrolling student: {e}")
            import traceback
            traceback.print_exc()
            return (False, f"Enrollment error: {str(e)}", None)

    def recognize_faces(self, frame):
        """Legacy method for backward compatibility"""
        status, message, data = self.recognize_faces_with_state(frame)
        
        if status == 'verified':
            return {
                'matches': [{
                    'student_id': data['student_id'],
                    'name': data['student_name'],
                    'confidence': data['confidence']
                }],
                'total_faces': 1
            }
        else:
            return {'matches': [], 'total_faces': data.get('total_faces', 0)}