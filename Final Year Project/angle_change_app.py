import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Extract features from a frame
def extract_features(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()  # Using ORB for feature detection
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors

# Monitor video feed or camera for angle deviation
def monitor_video(baseline_descriptors, baseline_keypoints, video_source):
    cap = cv2.VideoCapture(video_source)
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    st_frame = st.empty()  # Placeholder for the video feed

    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("End of video or error reading frame.")
            break

        # Extract features from the current frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = orb.detectAndCompute(gray, None)

        if descriptors is not None:
            # Match features between baseline and current frame
            matches = bf.match(baseline_descriptors, descriptors)
            matches = sorted(matches, key=lambda x: x.distance)

            # Calculate homography if enough matches are found
            if len(matches) > 10:
                src_pts = np.float32([baseline_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                if matrix is not None:
                    # Extract rotation and translation
                    angle = np.arctan2(matrix[1, 0], matrix[0, 0]) * 180 / np.pi
                    st.write(f"Detected angle deviation: {angle:.2f} degrees")

                    if abs(angle) > 5:  # Assuming threshold of 5 degrees
                        st.error("Alert: Camera angle deviation detected!")

        # Display the frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for Streamlit
        st_frame.image(frame, channels="RGB")

    cap.release()

# Main Streamlit app
def main():
    st.title("Camera Angle Deviation Detection")

    # Option to choose video source
    source_option = st.selectbox("Select Video Source", ("Upload Video", "Camera"))

    if source_option == "Upload Video":
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            video_path = "uploaded_video.mp4"
            with open(video_path, "wb") as f:
                f.write(uploaded_file.read())

            # Capture the first frame as the baseline
            cap = cv2.VideoCapture(video_path)
            ret, baseline_frame = cap.read()
            cap.release()

            if ret:
                st.image(cv2.cvtColor(baseline_frame, cv2.COLOR_BGR2RGB), caption="Baseline Frame", use_column_width=True)
                baseline_keypoints, baseline_descriptors = extract_features(baseline_frame)

                if st.button("Start Monitoring"):
                    monitor_video(baseline_descriptors, baseline_keypoints, video_path)

    elif source_option == "Camera":
        camera_index = st.number_input("Enter Camera Index", min_value=0, step=1, value=0)
        if st.button("Capture Baseline Frame"):
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                st.error("Camera failed to initialize! Check the camera index or permissions.")
            else:
                ret, baseline_frame = cap.read()
                cap.release()

                if ret:
                    st.image(cv2.cvtColor(baseline_frame, cv2.COLOR_BGR2RGB), caption="Baseline Frame", use_column_width=True)
                    baseline_keypoints, baseline_descriptors = extract_features(baseline_frame)

                    if st.button("Start Monitoring"):
                        monitor_video(baseline_descriptors, baseline_keypoints, camera_index)

if __name__ == "__main__":
    main()
