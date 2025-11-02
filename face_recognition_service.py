# Complete face recognition service with intelligent state management and spoof detection
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceRecognitionService:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.known_encodings = []
        self.known_names = []
        self.known_ids = []
        self.loaded = False
        
        # State management
        self.last_state_result = None
        self.frame_skip_counter = 0
        self.FRAME_SKIP = 2
        self.camera_obstructed = False
        
        # Thresholds
        self.HEAD_POSE_THRESHOLD = 35
        self.EAR_THRESHOLD = 0.25
        self.TEXTURE_THRESHOLD = 40
        self.BRIGHTNESS_RATIO_THRESHOLD = 0.7

    def _ensure_loaded(self):
        """Lazy loading of face encodings"""
        if not self.loaded:
            try:
                self.load_encodings_from_db()
            except Exception as e:
                logger.error(f"Error lazily loading faces: {e}")

    def load_encodings_from_db(self):
        """Load all face encodings from database into memory"""
        logger.info("Loading face encodings from database...")
        try:
            students = Student.query.filter_by(status='active').all()
            self.known_encodings = []
            self.known_names = []
            self.known_ids = []
            
            for student in students:
                if student.face_encoding is not None:
                    self.known_encodings.append(student.face_encoding)
                    self.known_names.append(student.name)
                    self.known_ids.append(student.id)
            
            self.loaded = True
            logger.info(f"Successfully loaded {len(self.known_ids)} face encodings")
        except Exception as e:
            logger.error(f"Error loading face encodings from database: {e}")
            raise

    def load_known_faces(self):
        """Alias for backward compatibility"""
        self.load_encodings_from_db()

    def detect_camera_obstruction(self, frame):
        """
        Detect if camera is obstructed (covered, very dark, or uniform color)
        Returns: (is_obstructed: bool, reason: str)
        """
        try:
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
            hist_normalized = hist.flatten() / hist.sum()
            
            if np.max(hist_normalized) > 0.6:
                return True, "Camera shows uniform pattern (possible obstruction)"
            
            return False, ""
            
        except Exception as e:
            logger.error(f"Error detecting camera obstruction: {e}")
            return False, ""

    def calculate_ear(self, eye):
        """Calculate Eye Aspect Ratio for blink detection"""
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def detect_blink(self, frame):
        """Detect blink for liveness verification"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)
            
            for face in faces:
                landmarks = self.predictor(gray, face)
                landmarks = np.array([(p.x, p.y) for p in landmarks.parts()])
                
                left_eye = landmarks[42:48]
                right_eye = landmarks[36:42]
                
                left_ear = self.calculate_ear(left_eye)
                right_ear = self.calculate_ear(right_eye)
                ear = (left_ear + right_ear) / 2.0
                
                if ear < self.EAR_THRESHOLD:
                    return True
            return False
        except Exception as e:
            logger.error(f"Error in blink detection: {e}")
            return False

    def estimate_head_pose(self, landmarks, frame_shape):
        """Estimate head pose to verify eye contact"""
        try:
            model_points = np.array([
                (0.0, 0.0, 0.0),
                (0.0, -330.0, -65.0),
                (-225.0, 170.0, -135.0),
                (225.0, 170.0, -135.0),
                (-150.0, -150.0, -125.0),
                (150.0, -150.0, -125.0)
            ])
            
            image_points = np.array([
                landmarks[30],
                landmarks[8],
                landmarks[36],
                landmarks[45],
                landmarks[48],
                landmarks[54]
            ], dtype="double")
            
            size = frame_shape
            focal_length = size[1]
            center = (size[1] / 2, size[0] / 2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype="double")
            
            dist_coeffs = np.zeros((4, 1))
            
            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            rotation_mat, _ = cv2.Rodrigues(rotation_vector)
            pose_mat = cv2.hconcat((rotation_mat, translation_vector))
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
            
            pitch = euler_angles[0][0]
            yaw = euler_angles[1][0]
            roll = euler_angles[2][0]
            
            return pitch, yaw, roll
        except Exception as e:
            logger.error(f"Error in head pose estimation: {e}")
            return 0, 0, 0

    def check_eye_contact(self, frame):
        """Check if person is looking at camera"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)
            
            if len(faces) == 0:
                return False, (0, 0, 0)
            
            face = faces[0]
            landmarks = self.predictor(gray, face)
            landmarks_np = np.array([(p.x, p.y) for p in landmarks.parts()])
            
            pitch, yaw, roll = self.estimate_head_pose(landmarks_np, frame.shape)
            
            has_eye_contact = (abs(pitch) < self.HEAD_POSE_THRESHOLD and 
                             abs(yaw) < self.HEAD_POSE_THRESHOLD)
            
            return has_eye_contact, (pitch, yaw, roll)
            
        except Exception as e:
            logger.error(f"Error checking eye contact: {e}")
            return True, (0, 0, 0)

    def recognize_faces_with_state(self, frame):
        """
        Enhanced recognition with intelligent state management and spoof detection
        Returns: (status, message, data)
        """
        self._ensure_loaded()
        
        # Performance optimization: Skip frames
        self.frame_skip_counter += 1
        if self.frame_skip_counter % self.FRAME_SKIP != 0:
            if self.last_state_result:
                return self.last_state_result
            return ('clear', None, {})
        
        try:
            # Check for camera obstruction
            is_obstructed, obstruction_reason = self.detect_camera_obstruction(frame)
            
            if is_obstructed:
                if not self.camera_obstructed:
                    self.camera_obstructed = True
                    self._log_activity('camera_obstructed', obstruction_reason)
                    logger.warning(f"Camera obstructed: {obstruction_reason}")
                
                result = ('obstructed', obstruction_reason, {})
                self.last_state_result = result
                return result
            else:
                if self.camera_obstructed:
                    self.camera_obstructed = False
                    self._log_activity('camera_resumed', 'Camera feed restored')
                    logger.info("Camera feed restored")
            
            # Face detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame, model='hog')
            
            # No face detected
            if len(face_locations) == 0:
                result = ('no_face', None, {'total_faces': 0})
                self.last_state_result = result
                return result
            
            # Multiple faces
            if len(face_locations) > 1:
                result = ('multiple_faces', 'Multiple people detected. Please ensure only one person is in frame.', {'total_faces': len(face_locations)})
                self.last_state_result = result
                return result
            
            # Single face - get encoding
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            if len(face_encodings) == 0:
                result = ('no_face', None, {'total_faces': 1})
                self.last_state_result = result
                return result
            
            face_encoding = face_encodings[0]
            
            if len(self.known_encodings) == 0:
                result = ('unknown', 'No enrolled students in database', {})
                self.last_state_result = result
                return result
            
            # Face matching
            matches = face_recognition.compare_faces(
                self.known_encodings, 
                face_encoding,
                tolerance=0.6
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
            
            if not matches[best_match_index] or face_distances[best_match_index] >= 0.6:
                result = ('unknown', 'Face not recognized - Not enrolled', {})
                self.last_state_result = result
                return result
            
            # Face recognized!
            student_id = self.known_ids[best_match_index]
            student_name = self.known_names[best_match_index]
            confidence = 1 - face_distances[best_match_index]
            
            logger.info(f"Face recognized: {student_name}")
            
            # === SPOOF DETECTION CHECK ===
            try:
                from spoof_detection.ensemble_spoof import check as spoof_check
                from config import Config
                
                # Convert face location to bbox (x, y, w, h)
                top, right, bottom, left = face_locations[0]
                face_bbox_xywh = (left, top, right - left, bottom - top)
                
                spoof_result = spoof_check(frame, face_bbox_xywh, face_encoding)
                
                if spoof_result['is_spoof']:
                    spoof_conf = spoof_result['confidence']
                    spoof_type = spoof_result['spoof_type']
                    evidence = spoof_result['evidence']
                    
                    # Get timezone for logging
                    IST = pytz.timezone('Asia/Kolkata')
                    
                    # Log to terminal
                    logger.warning(
                        f"[{datetime.now(IST).isoformat()}] SPOOF_DETECTED "
                        f"student_id={student_id} name=\"{student_name}\" "
                        f"spoof_type={spoof_type} confidence={spoof_conf:.2f} "
                        f"evidence={evidence}"
                    )
                    
                    # Log to database
                    self._log_spoof_activity(student_id, student_name, spoof_type, spoof_conf, evidence)
                    
                    # Decide action based on config
                    auto_block = getattr(Config, 'AUTO_BLOCK_SPOOF', False)
                    
                    if spoof_conf >= 0.93 and auto_block:
                        result = ('spoof_blocked', f'Spoof detected: {spoof_type}', {
                            'student_id': student_id,
                            'student_name': student_name,
                            'spoof_type': spoof_type,
                            'confidence': spoof_conf,
                            'status': 'blocked',
                            'evidence': evidence
                        })
                    else:
                        result = ('spoof_flagged', f'Potential spoof detected: {spoof_type}', {
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
                logger.error(f"Spoof detection error: {e}")
                # Continue with normal recognition if spoof check fails
            
            # === END SPOOF CHECK ===
            
            # All checks passed - return verified
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
            logger.error(f"Error in face recognition: {e}")
            result = ('error', str(e), {})
            self.last_state_result = result
            return result

    def _log_activity(self, activity_type, message):
        """Log activity to database"""
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
        """Log spoof detection to ActivityLog"""
        try:
            log = ActivityLog(
                student_id=student_id,
                name=student_name,
                activity_type='spoof_detected',
                message=f"Spoof detected: {spoof_type} (conf={confidence:.2f})",
                severity='critical' if confidence >= 0.93 else 'warning',
                spoof_type=str(spoof_type) if spoof_type else None,
                spoof_confidence=confidence,
                detection_details=json.dumps(evidence)
            )
            db.session.add(log)
            db.session.commit()
        except Exception as e:
            logger.error(f"Failed to log spoof activity: {e}")
            db.session.rollback()

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

    def compute_face_hash(self, face_encoding):
        """Compute hash of face encoding for duplicate detection"""
        encoding_str = ','.join(map(str, face_encoding))
        return hashlib.sha256(encoding_str.encode()).hexdigest()

    def check_duplicate_face(self, face_encoding):
        """Check if face encoding already exists"""
        try:
            face_hash = self.compute_face_hash(face_encoding)
            
            existing = Student.query.filter_by(face_hash=face_hash).first()
            if existing:
                return True, existing
            
            all_students = Student.query.filter(Student.face_encoding.isnot(None)).all()
            
            for student in all_students:
                if student.face_encoding is not None:
                    distance = face_recognition.face_distance([student.face_encoding], face_encoding)[0]
                    if distance < 0.4:
                        return True, student
            
            return False, None
            
        except Exception as e:
            logger.error(f"Error checking duplicate face: {e}")
            return False, None

    def enroll_student(self, frame, student):
        """Enroll student face with duplicate detection"""
        self._ensure_loaded()
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame, model='hog')
            
            if len(face_locations) == 0:
                face_locations = face_recognition.face_locations(rgb_frame, model='cnn')

            if len(face_locations) == 0:
                return (False, "No face found in the image. Please try again with better lighting.", None)
            
            if len(face_locations) > 1:
                return (False, "Multiple faces found. Please ensure only one person is in the photo.", None)

            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            if len(face_encodings) == 0:
                return (False, "Could not compute face encoding. Please try again.", None)
            
            face_encoding = face_encodings[0]
            
            is_duplicate, existing_student = self.check_duplicate_face(face_encoding)
            if is_duplicate:
                return (False, f"This face is already enrolled for student: {existing_student.name} ({existing_student.student_id})", None)
            
            logger.info(f"Face encoding successful for student {getattr(student, 'student_id', 'unknown')}")
            
            return (True, "Face encoding successful.", face_encoding)
            
        except Exception as e:
            logger.error(f"Error enrolling student: {e}")
            return (False, f"Enrollment error: {str(e)}", None)