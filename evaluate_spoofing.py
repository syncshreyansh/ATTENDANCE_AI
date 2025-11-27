#!/usr/bin/env python3
"""
Spoof Detection Evaluation Script
Tests anti-spoofing accuracy and recognition metrics simultaneously
"""
import os
import cv2
import numpy as np
from spoof_detection.ensemble_spoof import check as spoof_check
from face_recognition_service import FaceRecognitionService
from main import create_app
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpoofEvaluator:
    def __init__(self, spoof_test_dir, recog_test_dir):
        self.spoof_test_dir = spoof_test_dir
        self.recog_test_dir = recog_test_dir
        self.face_service = FaceRecognitionService()
    
    def evaluate_spoofing(self):
        """
        Evaluate spoof detection on labeled dataset
        Expected structure: spoof_test_dir/live/*.jpg, spoof_test_dir/spoof/*.jpg
        """
        logger.info("=" * 60)
        logger.info("EVALUATING SPOOF DETECTION")
        logger.info("=" * 60)
        
        live_dir = os.path.join(self.spoof_test_dir, 'live')
        spoof_dir = os.path.join(self.spoof_test_dir, 'spoof')
        
        if not os.path.exists(live_dir) or not os.path.exists(spoof_dir):
            logger.error(f"Spoof test directories not found. Please create:")
            logger.error(f"  {live_dir}/  (real face images)")
            logger.error(f"  {spoof_dir}/  (photo/screen/phone attack images)")
            return None
        
        # Test live images (should NOT be flagged)
        live_images = [os.path.join(live_dir, f) for f in os.listdir(live_dir) if f.endswith(('.jpg', '.png'))]
        spoof_images = [os.path.join(spoof_dir, f) for f in os.listdir(spoof_dir) if f.endswith(('.jpg', '.png'))]
        
        true_negatives = 0  # Live correctly identified
        false_positives = 0  # Live incorrectly flagged as spoof
        true_positives = 0  # Spoof correctly detected
        false_negatives = 0  # Spoof missed
        
        for img_path in live_images:
            frame = cv2.imread(img_path)
            # Assume face is in frame; detect face bbox
            import face_recognition
            face_locs = face_recognition.face_locations(frame)
            if len(face_locs) == 0:
                continue
            top, right, bottom, left = face_locs[0]
            face_bbox = (left, top, right-left, bottom-top)
            
            result = spoof_check(frame, face_bbox)
            if not result['is_spoof']:
                true_negatives += 1
            else:
                false_positives += 1
                logger.warning(f"FP: {os.path.basename(img_path)} flagged as spoof (conf={result['confidence']})")
        
        for img_path in spoof_images:
            frame = cv2.imread(img_path)
            import face_recognition
            face_locs = face_recognition.face_locations(frame)
            if len(face_locs) == 0:
                continue
            top, right, bottom, left = face_locs[0]
            face_bbox = (left, top, right-left, bottom-top)
            
            result = spoof_check(frame, face_bbox)
            if result['is_spoof']:
                true_positives += 1
            else:
                false_negatives += 1
                logger.warning(f"FN: {os.path.basename(img_path)} not detected as spoof (conf={result['confidence']})")
        
        total = true_positives + true_negatives + false_positives + false_negatives
        if total == 0:
            logger.error("No test images processed!")
            return None
        
        tpr = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        tnr = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
        fpr = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0
        accuracy = (true_positives + true_negatives) / total
        
        logger.info("=" * 60)
        logger.info("SPOOF DETECTION RESULTS")
        logger.info("=" * 60)
        logger.info(f"True Positives (Spoof Detected): {true_positives}")
        logger.info(f"True Negatives (Live Passed): {true_negatives}")
        logger.info(f"False Positives (Live Flagged): {false_positives}")
        logger.info(f"False Negatives (Spoof Missed): {false_negatives}")
        logger.info("-" * 60)
        logger.info(f"TPR (Sensitivity): {tpr:.2%}")
        logger.info(f"TNR (Specificity): {tnr:.2%}")
        logger.info(f"FPR (False Alarm Rate): {fpr:.2%}")
        logger.info(f"Overall Accuracy: {accuracy:.2%}")
        logger.info("=" * 60)
        
        return {
            'tpr': tpr,
            'tnr': tnr,
            'fpr': fpr,
            'accuracy': accuracy
        }
    
    def evaluate_recognition(self):
        """Run face recognition evaluation"""
        logger.info("=" * 60)
        logger.info("EVALUATING FACE RECOGNITION ACCURACY")
        logger.info("=" * 60)
        
        from evaluate_recognition import RecognitionEvaluator
        evaluator = RecognitionEvaluator()
        evaluator.face_service = self.face_service
        evaluator.test_images_dir = self.recog_test_dir
        
        test_data = evaluator.load_test_images()
        if len(test_data) == 0:
            logger.error(f"No test images found in {self.recog_test_dir}")
            return None
        
        correct = 0
        total = 0
        
        for test_item in test_data:
            frame = cv2.imread(test_item['image_path'])
            if frame is None:
                continue
            
            result = self.face_service.recognize_faces(frame)
            total += 1
            
            if len(result['matches']) > 0:
                from models import Student
                predicted_student = Student.query.get(result['matches'][0]['student_id'])
                if predicted_student and predicted_student.student_id == test_item['student_id']:
                    correct += 1
        
        accuracy = (correct / total * 100) if total > 0 else 0
        
        logger.info("=" * 60)
        logger.info("RECOGNITION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Correct: {correct}/{total}")
        logger.info(f"Accuracy: {accuracy:.2f}%")
        logger.info("=" * 60)
        
        return accuracy

def main():
    parser = argparse.ArgumentParser(description='Evaluate spoofing and recognition')
    parser.add_argument('--spoof-test-dir', default='./spoof_testset', help='Directory with live/ and spoof/ subdirs')
    parser.add_argument('--recog-test-dir', default='./test_images', help='Directory with student_id subdirs')
    args = parser.parse_args()
    
    app = create_app()
    with app.app_context():
        evaluator = SpoofEvaluator(args.spoof_test_dir, args.recog_test_dir)
        
        # Load face encodings
        evaluator.face_service.load_encodings_from_db()
        
        # Evaluate spoof detection
        spoof_metrics = evaluator.evaluate_spoofing()
        
        # Evaluate recognition
        recog_accuracy = evaluator.evaluate_recognition()
        
        # Check acceptance criteria
        logger.info("")
        logger.info("=" * 60)
        logger.info("ACCEPTANCE CRITERIA CHECK")
        logger.info("=" * 60)
        
        if spoof_metrics and recog_accuracy is not None:
            spoof_pass = spoof_metrics['tpr'] >= 0.85 and spoof_metrics['fpr'] <= 0.10
            recog_pass = recog_accuracy >= 85.0  # Adjust based on your baseline
            
            logger.info(f"Spoof TPR ≥ 85%: {'✓ PASS' if spoof_metrics['tpr'] >= 0.85 else '✗ FAIL'} ({spoof_metrics['tpr']:.1%})")
            logger.info(f"Spoof FPR ≤ 10%: {'✓ PASS' if spoof_metrics['fpr'] <= 0.10 else '✗ FAIL'} ({spoof_metrics['fpr']:.1%})")
            logger.info(f"Recognition Accuracy ≥ 85%: {'✓ PASS' if recog_accuracy >= 85.0 else '✗ FAIL'} ({recog_accuracy:.1f}%)")
            
            if spoof_pass and recog_pass:
                logger.info("=" * 60)
                logger.info("✓✓✓ ALL TESTS PASSED ✓✓✓")
                logger.info("=" * 60)
            else:
                logger.warning("=" * 60)
                logger.warning("✗✗✗ SOME TESTS FAILED - TUNING REQUIRED ✗✗✗")
                logger.warning("=" * 60)
                logger.warning("Recommended tuning:")
                if spoof_metrics['fpr'] > 0.10:
                    logger.warning("  - Increase SPOOF_CONFIDENCE_THRESHOLD_FLAG from 0.7 to 0.75")
                    logger.warning("  - Reduce texture_conf weight from 0.2 to 0.15")
                if spoof_metrics['tpr'] < 0.85:
                    logger.warning("  - Decrease SPOOF_CONFIDENCE_THRESHOLD_FLAG from 0.7 to 0.65")
                    logger.warning("  - Increase cnn_conf weight from 0.25 to 0.30")
                if recog_accuracy < 85.0:
                    logger.warning("  - Spoof detection may be too aggressive")
                    logger.warning("  - Review false positives and adjust threshold")
        
        logger.info("=" * 60)

if __name__ == '__main__':
    main()