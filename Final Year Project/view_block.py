import cv2
import numpy as np
import streamlit as st

# Function to monitor view blocking
def monitor_view_blocking(baseline_frame, video_source):
    cap = cv2.VideoCapture(video_source)
    st_frame = st.empty()  
    # Preprocess the baseline frame:
    baseline_gray = cv2.cvtColor(baseline_frame, cv2.COLOR_BGR2GRAY)
    baseline_blurred = cv2.GaussianBlur(baseline_gray, (5, 5), 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("End of video or error reading frame.")
            break

        # Preprocess the current frame:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray_frame, (5, 5), 0)

        # Compute absolute difference between baseline and current frame
        diff = cv2.absdiff(baseline_blurred, gray_blurred)
        # Threshold the difference to get a binary mask
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        # Calculate the fraction of the frame that differs
        blocking_ratio = np.count_nonzero(thresh) / thresh.size
        blocking_percentage = blocking_ratio * 100

        # Alert logic based on thresholds
        if blocking_percentage >= 70:
            st.error(f"Extreme Warning: Camera view is blocked by {blocking_percentage:.2f}%!")
        elif blocking_percentage >= 50:
            st.warning(f"Warning: Camera view is blocked by {blocking_percentage:.2f}%!")
        else:
            st.write(f"Camera view is clear: blocked by {blocking_percentage:.2f}%")

        # (Optional) Display the thresholded difference image for debugging
        thresh_disp = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
        st_frame.image(thresh_disp, channels="RGB")

    cap.release()

def main():
    st.title("CCTV View Blocking Detection")
    st.write("This app detects if the camera's view is being blocked.")

    # Option to choose the video source
    source_option = st.selectbox("Select Video Source", ("Upload Video", "Camera"))

    if source_option == "Upload Video":
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            video_path = "uploaded_video.mp4"
            with open(video_path, "wb") as f:
                f.write(uploaded_file.read())

            # Capture the first frame as the baseline reference
            cap = cv2.VideoCapture(video_path)
            ret, baseline_frame = cap.read()
            cap.release()

            if ret:
                st.image(cv2.cvtColor(baseline_frame, cv2.COLOR_BGR2RGB),
                         caption="Baseline Frame (Unobstructed View)",
                         use_column_width=True)
                if st.button("Start View Blocking Monitoring"):
                    monitor_view_blocking(baseline_frame, video_path)

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
                    st.image(cv2.cvtColor(baseline_frame, cv2.COLOR_BGR2RGB),
                             caption="Baseline Frame (Unobstructed View)",
                             use_column_width=True)
                    if st.button("Start View Blocking Monitoring"):
                        monitor_view_blocking(baseline_frame, camera_index)

if __name__ == "__main__":
    main()
