import cv2
import numpy as np
from collections import deque

class ViewBlockageDetector:
    def __init__(self, threshold=10, min_obstruction_frames=5, history_size=30):
        self.threshold = threshold  # Grayscale STD threshold
        self.min_obstruction_frames = min_obstruction_frames  # Minimum consecutive frames to trigger alert
        self.obstruction_counter = 0
        self.normal_counter = 0
        self.status = "CLEAR"
        
        # Store historical STD values for dynamic thresholding
        self.history = deque(maxlen=history_size)
        self.dynamic_threshold = threshold
        
    def update_dynamic_threshold(self, current_std):
        """Adjust threshold based on historical scene variations"""
        self.history.append(current_std)
        if len(self.history) > 10:
            mean_std = np.mean(self.history)
            std_dev = np.std(self.history)
            self.dynamic_threshold = max(5, mean_std - 2 * std_dev)
    
    def detect_blockage(self, frame):
        # Convert to grayscale and calculate standard deviation
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_std = np.std(gray)
        
        # Update dynamic threshold
        self.update_dynamic_threshold(current_std)
        
        # Check for obstruction
        if current_std < self.dynamic_threshold:
            self.obstruction_counter += 1
            self.normal_counter = max(0, self.normal_counter - 2)
        else:
            self.normal_counter += 1
            self.obstruction_counter = max(0, self.obstruction_counter - 1)
        
        # Determine status with hysteresis
        if self.obstruction_counter >= self.min_obstruction_frames:
            self.status = "BLOCKED"
        elif self.normal_counter >= self.min_obstruction_frames:
            self.status = "CLEAR"
            
        return gray, current_std, self.status

def main():
    # Initialize detector - adjust parameters here
    detector = ViewBlockageDetector(threshold=15, min_obstruction_frames=8)
    
    # Initialize video source - 0 for webcam or path to video file
    cap = cv2.VideoCapture(0)  # Replace with CCTV feed URL if available
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Detect view blockage
        gray_frame, current_std, status = detector.detect_blockage(frame)
        
        # Display status and metrics
        color = (0, 0, 255) if status == "BLOCKED" else (0, 255, 0)
        cv2.putText(frame, f"Status: {status}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"STD: {current_std:.1f} | Thr: {detector.dynamic_threshold:.1f}", 
                   (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show frames
        cv2.imshow("Live Feed", frame)
        cv2.imshow("Grayscale Analysis", gray_frame)
        
        # Print status to console (for logging systems)
        print(f"Frame: STD={current_std:.1f}, Status={status}")
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()