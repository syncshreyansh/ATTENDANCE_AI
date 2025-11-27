"""
Quick Blink Test Utility
Run this to verify blink detection works properly
"""
import cv2
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from mediapipe_blink_detector import MediaPipeBlinkDetector
    print("✓ Using MediaPipe Blink Detector (ML-based)")
    detector = MediaPipeBlinkDetector()
except Exception as e:
    print(f"⚠️ MediaPipe not available: {e}")
    print("Falling back to dlib EAR method")
    from liveness_detection import LivenessDetector
    detector = LivenessDetector()

def main():
    print("\n" + "=" * 70)
    print("                    BLINK DETECTION TEST")
    print("=" * 70)
    print("\n📋 Instructions:")
    print("  1. Position yourself in front of the camera")
    print("  2. Look directly at the camera")
    print("  3. BLINK SLOWLY and DELIBERATELY")
    print("  4. Close your eyes for at least 1 second")
    print("  5. Then open them")
    print("\n⚙️  Press 'q' to quit")
    print("=" * 70 + "\n")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        return
    
    frame_count = 0
    last_blink_frame = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error reading frame")
            break
        
        frame_count += 1
        
        # Detect blinks
        if hasattr(detector, 'detect_blink'):
            # MediaPipe detector
            blink_detected, ear, details = detector.detect_blink(frame)
            total_blinks = details.get('total_blinks', 0)
            status = f"EAR: {ear:.3f}"
            
            if details.get('blink_in_progress'):
                status +=  " | 👁️ CLOSING..."
            
            if blink_detected:
                status += " | ✅ BLINK!"
                last_blink_frame = frame_count
        else:
            # dlib detector (fallback)
            is_live, conf, details = detector.comprehensive_liveness_check(frame)
            total_blinks = details.get('total_blinks', 0)
            ear = details.get('ear', 0)
            status = f"EAR: {ear:.3f}"
        
        # Display status
        color = (0, 255, 0) if frame_count - last_blink_frame < 30 else (255, 255, 255)
        cv2.putText(frame, f"Blinks Detected: {total_blinks}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, status, (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Instructions
        cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        cv2.imshow("Blink Detection Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 70)
    print(f"✅ Test Complete!")
    print(f"   Total Blinks Detected: {total_blinks}")
    if total_blinks > 0:
        print("   ✓ Blink detection is WORKING!")
    else:
        print("   ⚠️ No blinks detected. Try:")
        print("      - Better lighting")
        print("      - Slower, more deliberate blinks")
        print("      - Looking directly at camera")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
