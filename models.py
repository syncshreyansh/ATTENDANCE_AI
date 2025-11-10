# models.py - ENHANCED with CoordinatorScope and OTPToken
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import pytz

db = SQLAlchemy()

IST = pytz.timezone('Asia/Kolkata')

def get_ist_now():
    """Get current time in IST"""
    return datetime.now(IST)

class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    student_id = db.Column(db.String(20), unique=True, nullable=False)
    class_name = db.Column(db.String(20), nullable=False)
    section = db.Column(db.String(10), nullable=False)
    parent_phone = db.Column(db.String(15), nullable=False)
    face_encoding = db.Column(db.PickleType, nullable=True)
    enrollment_date = db.Column(db.DateTime, default=get_ist_now)
    status = db.Column(db.String(20), default='active')
    points = db.Column(db.Integer, default=0)
    image_path = db.Column(db.String(200), nullable=True)
    face_hash = db.Column(db.String(64), unique=True, nullable=True)

class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('student.id'), nullable=False)
    date = db.Column(db.Date, nullable=False)
    time_in = db.Column(db.DateTime, nullable=True)
    time_out = db.Column(db.DateTime, nullable=True)
    status = db.Column(db.String(20), nullable=False)
    confidence = db.Column(db.Float, nullable=True)
    blink_verified = db.Column(db.Boolean, default=False)
    eye_contact_verified = db.Column(db.Boolean, default=False)
    points_earned = db.Column(db.Integer, default=0)

class ActivityLog(db.Model):
    __tablename__ = 'activity_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('student.id'), nullable=True)
    name = db.Column(db.String(100), nullable=True)
    activity_type = db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.DateTime, default=get_ist_now)
    message = db.Column(db.Text, nullable=False)
    severity = db.Column(db.String(20), default='warning')
    spoof_type = db.Column(db.String(100), nullable=True)
    spoof_confidence = db.Column(db.Float, nullable=True)
    detection_details = db.Column(db.Text, nullable=True)

class Alert(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('student.id'), nullable=False)
    alert_type = db.Column(db.String(50), nullable=False)
    message = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=get_ist_now)
    sent = db.Column(db.Boolean, default=False)
    delivered = db.Column(db.Boolean, default=False)

class SystemLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    event_type = db.Column(db.String(50), nullable=False)
    description = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=get_ist_now)
    severity = db.Column(db.String(20), default='info')

class AbsenceTracker(db.Model):
    __tablename__ = 'absence_tracker'
    
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('student.id'), nullable=False, unique=True)
    consecutive_absences = db.Column(db.Integer, default=0)
    last_present_date = db.Column(db.Date, nullable=True)
    notification_sent = db.Column(db.Boolean, default=False)
    last_notification_date = db.Column(db.Date, nullable=True)
    
    student = db.relationship('Student', backref='absence_tracker')

# === NEW: Coordinator Scope ===
class CoordinatorScope(db.Model):
    __tablename__ = "coordinator_scope"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), unique=True, nullable=False)
    class_name = db.Column(db.String(20), nullable=False)
    section = db.Column(db.String(10), nullable=True)  # None = all sections in class

# === NEW: OTP Token for Password Reset ===
class OTPToken(db.Model):
    __tablename__ = 'otp_tokens'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    code = db.Column(db.String(6), nullable=False)
    expires_at = db.Column(db.DateTime, nullable=False)
    used = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=get_ist_now)