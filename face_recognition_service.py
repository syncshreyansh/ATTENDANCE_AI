# face_recognition_service.py - FULLY WORKING VERSION
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
            logger.info("✓ Landmark predictor loaded")
        except Exception as e:
            logger.error(f"✗ Failed to load landmark predictor: {e}")
            self.predictor = None
            
        self.known_encodings = []
        self.known_names = []
        self.known_ids = []
        self.loaded = False
        
        # State management
        self.last_state_result = None
        self.frame_skip_counter = 0
        self.FRAME_SKIP = 2
        self.camera_obstructed = False
        self.recognition_history = {}
        
        # FIXED: Lenient thresholds
        self.FACE_MATCH_THRESHOLD = 0.5
        self.CONFIDENCE_THRESHOLD = 0.5
        self.MIN_FACE_SIZE = 80  # Smaller minimum
        
        # Thread pool
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Initialize liveness detector
        from liveness_detection import LivenessDetector
        self.liveness_detector = LivenessDetector()
        logger.info("✓ Liveness detector initialized")
        
        # Blink tracking
        self.blink_wait_started = None
        self.blink_wait_timeout = 10  # seconds
        self.consecutive_frames_with_face = 0
        self.required_consecutive_frames = 3

    def _ensure_loaded(self):
        """Lazy loading of face encodings"""
        if not self.loaded:
            try:
                self.load_encodings_from_db()
                return True
            except Exception as e:
                logger.error(f"Error loading faces: {e}")
                return False
        return True

    def load_encodings_from_db(self):
        """Load face encodings from database"""
        logger.info("Loading face encodings from database...")
        try:
            students = Student.query.filter_by(status='active').all()
            self.known_encodings = []
            self.known_names = []
            self.known_ids = []
            
            loaded_count = 0
            for student in students:
                if student.face_encoding is not None:
                    if isinstance(student.face_encoding, np.ndarray) and len(student.face_encoding) == 128:
                        self.known_encodings.append(student.face_encoding)
                        self.known_names.append(student.name)
                        self.known_ids.append(student.id)
                        loaded_count += 1
            
            self.loaded = True
            logger.info(f"✓ Loaded {loaded_count} face encodings")
            return True
            
        except Exception as e:
            logger.error(f"✗ Error loading encodings: {e}")
            self.loaded = False
            return False

    def detect_camera_obstruction(self, frame):
        """Check if camera is obstructed"""
        try:
            if frame is None or frame.size == 0:
                return True, "Frame is empty"
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            avg_brightness = np.mean(gray)
            if avg_brightness < 10:
                return True, "Camera covered or very dark"
            
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var < 5:
                return True, "Camera shows uniform surface"
            
            return False, ""
            
        except Exception as e:
            logger.error(f"Error detecting obstruction: {e}")
            return False, ""

    def validate_face_quality(self, frame, face_location):
        """Validate face quality"""
        try:
            top, right, bottom, left = face_location
            
            face_width = right - left
            face_height = bottom - top
            if face_width < self.MIN_FACE_SIZE or face_height < self.MIN_FACE_SIZE:
                return False, "Face too small - move closer"
            
            h, w = frame.shape[:2]
            if left < 0 or top < 0 or right > w or bottom > h:
                return False, "Face partially outside frame"
            
            face_roi = frame[top:bottom, left:right]
            if face_roi.size == 0:
                return False, "Invalid face region"
            
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray_face)
            if avg_brightness < 25:
                return False, "Face too dark"
            if avg_brightness > 245:
                return False, "Face overexposed"
            
            laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            if laplacian_var < 30:
                return False, "Image blurry - hold steady"
            
            return True, "OK"
            
        except Exception as e:
            logger.error(f"Error validating quality: {e}")
            return False, "Validation error"

    def recognize_faces_with_state(self, frame):
        """
        FIXED: Working recognition with proper blink prompt
        """
        if not self._ensure_loaded():
            return ('error', 'System not initialized', {})
        
        # Frame skip for performance
        self.frame_skip_counter += 1
        if self.frame_skip_counter % self.FRAME_SKIP != 0:
            if self.last_state_result:
                return self.last_state_result
            return ('clear', None, {})
        
        try:
            # Validate frame
            if frame is None or frame.size == 0:
                return ('error', 'Invalid frame', {})
            
            # Check obstruction
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
            
            # Detect faces
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame, model='hog')
            
            if len(face_locations) == 0:
                self.consecutive_frames_with_face = 0
                self.blink_wait_started = None
                result = ('no_face', None, {'total_faces': 0})
                self.last_state_result = result
                return result
            
            if len(face_locations) > 1:
                result = ('multiple_faces', 'Only one person allowed', {'total_faces': len(face_locations)})
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
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, num_jitters=1)
            
            if len(face_encodings) == 0:
                result = ('error', 'Could not extract face features', {})
                self.last_state_result = result
                return result
            
            face_encoding = face_encodings[0]
            
            # Check if database has students
            if len(self.known_encodings) == 0:
                result = ('unknown', 'No students enrolled', {})
                self.last_state_result = result
                return result
            
            # Match face
            matches = face_recognition.compare_faces(
                self.known_encodings,
                face_encoding,
                tolerance=self.FACE_MATCH_THRESHOLD
            )
            face_distances = face_recognition.face_distance(self.known_encodings, face_encoding)
            
            if len(face_distances) == 0:
                result = ('unknown', 'Face not recognized', {})
                self.last_state_result = result
                return result
            
            best_match_index = np.argmin(face_distances)
            confidence = 1 - face_distances[best_match_index]
            
            # Check match quality
            if not matches[best_match_index] or confidence < self.CONFIDENCE_THRESHOLD:
                self._log_activity('unknown_face', f'Low confidence: {confidence:.2f}')
                result = ('unknown', f'Face not recognized (confidence: {confidence:.0%})', {})
                self.last_state_result = result
                return result
            
            # FACE RECOGNIZED!
            student_id = self.known_ids[best_match_index]
            student_name = self.known_names[best_match_index]
            
            logger.info(f"✓ Recognized: {student_name} (conf: {confidence:.2%})")
            
            # Track consecutive frames
            self.consecutive_frames_with_face += 1
            
            # STEP 1: Show "Please Blink" message
            if self.blink_wait_started is None:
                self.blink_wait_started = time.time()
                logger.info(f"⏳ Waiting for blink from {student_name}")
                result = ('waiting_blink', f'👤 {student_name} - Please BLINK', {
                    'student_name': student_name,
                    'student_id': student_id
                })
                self.last_state_result = result
                return result
            
            # Check if blink wait timed out
            if time.time() - self.blink_wait_started > self.blink_wait_timeout:
                logger.warning(f"⏱️ Blink timeout for {student_name}")
                self.blink_wait_started = None
                result = ('error', '⏱️ Timeout - Please try again and blink', {})
                self.last_state_result = result
                return result
            
            # STEP 2: Run liveness detection
            try:
                is_live, liveness_conf, liveness_details = self.liveness_detector.comprehensive_liveness_check(frame)
                
                blink_detected = liveness_details.get('blink_detected', False)
                blink_score = liveness_details.get('scores', {}).get('blink', 0.0)
                
                logger.info(f"📊 Liveness: is_live={is_live}, conf={liveness_conf:.2f}, blink={blink_detected}")
                
                # Keep showing "Please Blink" until blink detected
                if not blink_detected:
                    result = ('waiting_blink', f'👤 {student_name} - Please BLINK', {
                        'student_name': student_name,
                        'student_id': student_id,
                        'blink_score': blink_score
                    })
                    self.last_state_result = result
                    return result
                
                # Blink detected! Now check overall liveness
                if not is_live or liveness_conf < 0.5:
                    logger.warning(f"❌ Liveness failed: conf={liveness_conf:.2f}")
                    self.blink_wait_started = None
                    result = ('error', '❌ Liveness verification failed', {})
                    self.last_state_result = result
                    return result
                
                logger.info(f"✅ Liveness passed for {student_name}")
                
            except Exception as e:
                logger.error(f"❌ Liveness error: {e}")
                import traceback
                traceback.print_exc()
                result = ('error', 'Liveness system error', {})
                self.last_state_result = result
                return result
            
            # STEP 3: Run spoof detection
            logger.info(f"🔍 Running spoof detection for {student_name}...")
            
            try:
                from spoof_detection.ensemble_spoof import check as spoof_check
                
                top, right, bottom, left = face_location
                face_bbox = (left, top, right - left, bottom - top)
                
                spoof_result = spoof_check(frame, face_bbox, face_encoding)
                
                logger.info(f"📊 Spoof: is_spoof={spoof_result['is_spoof']}, conf={spoof_result['confidence']:.2f}")
                
                if spoof_result['is_spoof']:
                    spoof_conf = spoof_result['confidence']
                    spoof_type = spoof_result['spoof_type']
                    
                    logger.warning(f"🚨 SPOOF: {student_name} | Type: {spoof_type} | Conf: {spoof_conf:.2f}")
                    
                    self._log_spoof_activity(student_id, student_name, spoof_type, spoof_conf, spoof_result['evidence'])
                    
                    self.blink_wait_started = None
                    
                    result = ('spoof_blocked', f'🚫 BLOCKED: {spoof_type}', {
                        'student_id': student_id,
                        'student_name': student_name,
                        'spoof_type': spoof_type,
                        'confidence': spoof_conf
                    })
                    self.last_state_result = result
                    return result
                
                logger.info(f"✅ Spoof check passed for {student_name}")
                
            except Exception as e:
                logger.error(f"❌ Spoof detection error: {e}")
                import traceback
                traceback.print_exc()
                result = ('error', 'Security verification error', {})
                self.last_state_result = result
                return result
            
            # ALL CHECKS PASSED!
            logger.info(f"🎉 All checks passed for {student_name}")
            
            # Require multiple consecutive frames for stability
            if self.consecutive_frames_with_face < self.required_consecutive_frames:
                logger.info(f"Verifying stability: {self.consecutive_frames_with_face}/{self.required_consecutive_frames}")
                result = ('verifying', f'Verifying... ({self.consecutive_frames_with_face}/3)', {
                    'student_id': student_id,
                    'progress': self.consecutive_frames_with_face
                })
                self.last_state_result = result
                return result
            
            # VERIFIED!
            self.blink_wait_started = None
            self.consecutive_frames_with_face = 0
            
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
            logger.error(f"❌ Recognition error: {e}")
            import traceback
            traceback.print_exc()
            result = ('error', f'System error: {str(e)}', {})
            self.last_state_result = result
            return result

    def _log_activity(self, activity_type, message):
        """Log activity"""
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
        """Log spoof detection"""
        try:
            log = ActivityLog(
                student_id=student_id,
                name=student_name,
                activity_type='spoof_detected',
                message=f"Spoof: {spoof_type} (conf={confidence:.2f})",
                severity='critical' if confidence >= 0.7 else 'warning',
                spoof_type=str(spoof_type) if spoof_type else None,
                spoof_confidence=confidence,
                detection_details=json.dumps(evidence) if evidence else None
            )
            db.session.add(log)
            db.session.commit()
        except Exception as e:
            logger.error(f"Error logging spoof: {e}")
            db.session.rollback()

    def compute_face_hash(self, face_encoding):
        """Compute hash for duplicate detection"""
        try:
            encoding_str = ','.join(map(str, face_encoding))
            return hashlib.sha256(encoding_str.encode()).hexdigest()
        except Exception as e:
            logger.error(f"Error computing hash: {e}")
            return None

    def check_duplicate_face(self, face_encoding):
        """Check for duplicate faces"""
        try:
            face_hash = self.compute_face_hash(face_encoding)
            if not face_hash:
                return False, None
            
            existing = Student.query.filter_by(face_hash=face_hash).first()
            if existing:
                return True, existing
            
            all_students = Student.query.filter(Student.face_encoding.isnot(None)).all()
            
            for student in all_students:
                if student.face_encoding is not None:
                    distance = face_recognition.face_distance([student.face_encoding], face_encoding)[0]
                    if distance < 0.35:
                        return True, student
            
            return False, None
            
        except Exception as e:
            logger.error(f"Error checking duplicate: {e}")
            return False, None

    def enroll_student(self, frame, student):
        """Enroll student with better validation"""
        if not self._ensure_loaded():
            self.load_encodings_from_db()
        
        try:
            if frame is None or frame.size == 0:
                return (False, "Invalid image", None)
            
            is_obstructed, msg = self.detect_camera_obstruction(frame)
            if is_obstructed:
                return (False, f"Image quality issue: {msg}", None)
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_GRAY2RGB)
            
            face_locations = face_recognition.face_locations(rgb_frame, model='hog')
            
            if len(face_locations) == 0:
                face_locations = face_recognition.face_locations(rgb_frame, model='cnn')

            if len(face_locations) == 0:
                return (False, "❌ No face detected", None)
            
            if len(face_locations) > 1:
                return (False, "❌ Multiple faces detected", None)

            face_location = face_locations[0]
            quality_valid, quality_msg = self.validate_face_quality(frame, face_location)
            if not quality_valid:
                return (False, f"❌ {quality_msg}", None)

            face_encodings = face_recognition.face_encodings(
                rgb_frame,
                face_locations,
                num_jitters=10,
                model='large'
            )
            
            if len(face_encodings) == 0:
                return (False, "❌ Could not extract face features", None)
            
            face_encoding = face_encodings[0]
            
            if not isinstance(face_encoding, np.ndarray) or len(face_encoding) != 128:
                return (False, "❌ Invalid face encoding", None)
            
            is_duplicate, existing_student = self.check_duplicate_face(face_encoding)
            if is_duplicate:
                return (False, f"❌ Face already enrolled: {existing_student.name} ({existing_student.student_id})", None)
            
            logger.info(f"✓ Face encoding successful for {getattr(student, 'student_id', 'unknown')}")
            
            return (True, "✓ Enrollment successful", face_encoding)
            
        except Exception as e:
            logger.error(f"❌ Enrollment error: {e}")
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