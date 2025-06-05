import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Camera Angle Deviation Detection", layout="wide")

# --- Feature Extraction ---

def extract_features(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors

# --- Single‑Frame Monitoring Step ---

def monitor_step():
    cap = st.session_state.cap
    orb = st.session_state.orb
    bf = st.session_state.bf
    baseline_kp = st.session_state.baseline_kp
    baseline_desc = st.session_state.baseline_desc
    threshold_deg = st.session_state.threshold_deg

    ret, frame = cap.read()
    if not ret:
        st.warning("📹 End of video or camera disconnected.")
        st.session_state.monitoring = False
        cap.release()
        return

    # Feature detection on current frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp, desc = orb.detectAndCompute(gray, None)

    # Match & compute homography if possible
    if desc is not None and baseline_desc is not None:
        matches = bf.match(baseline_desc, desc)
        matches = sorted(matches, key=lambda x: x.distance)
        if len(matches) > 10:
            src_pts = np.float32([baseline_kp[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
            dst_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is not None:
                angle = np.degrees(np.arctan2(M[1,0], M[0,0]))
                st.write(f"Detected angle deviation: **{angle:.2f}°**")
                if abs(angle) > threshold_deg:
                    st.error(f"🚨 Alert: Deviation > {threshold_deg}°")

    # Display current frame
    st.session_state.st_frame.image(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        channels="RGB",
        caption="Live Feed"
    )

    # Schedule next iteration
    st.experimental_rerun()

# --- Main App ---

def main():
    st.title("🔍 Camera Angle Deviation Detection")

    # Sidebar: threshold for alert
    st.sidebar.header("Settings")
    st.sidebar.markdown("Set the deviation threshold (degrees)")
    st.session_state.threshold_deg = st.sidebar.slider(
        "Angle Threshold (°)",
        min_value=1,
        max_value=45,
        value=5
    )

    # Video source selection
    source_option = st.selectbox("Select Video Source", ("Upload Video", "Camera"))

    # --- Upload Video Mode ---
    if source_option == "Upload Video":
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4","avi","mov"])
        if uploaded_file:
            video_path = "uploaded_video.mp4"
            with open(video_path, "wb") as f:
                f.write(uploaded_file.read())

            cap = cv2.VideoCapture(video_path)
            ret, baseline_frame = cap.read()
            cap.release()

            if not ret:
                st.error("Could not read the uploaded video.")
                return

            st.image(
                cv2.cvtColor(baseline_frame, cv2.COLOR_BGR2RGB),
                caption="Baseline Frame",
                use_column_width=True
            )

            # Extract baseline features
            kp, desc = extract_features(baseline_frame)
            st.session_state.baseline_kp = kp
            st.session_state.baseline_desc = desc

            if st.button("Start Monitoring"):
                # Initialize monitoring state
                st.session_state.monitoring = True
                st.session_state.cap = cv2.VideoCapture(video_path)
                st.session_state.orb = cv2.ORB_create()
                st.session_state.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                st.session_state.st_frame = st.empty()

    # --- Camera Mode ---
    else:
        cam_idx = st.number_input("Enter Camera Index", min_value=0, step=1, value=0)
        if st.button("Capture Baseline Frame"):
            cap = cv2.VideoCapture(cam_idx)
            if not cap.isOpened():
                st.error("Cannot open camera. Check index or permissions.")
                return

            ret, baseline_frame = cap.read()
            cap.release()

            if not ret:
                st.error("Failed to capture from camera.")
                return

            st.image(
                cv2.cvtColor(baseline_frame, cv2.COLOR_BGR2RGB),
                caption="Baseline Frame",
                use_column_width=True
            )

            # Extract baseline features
            kp, desc = extract_features(baseline_frame)
            st.session_state.baseline_kp = kp
            st.session_state.baseline_desc = desc

            if st.button("Start Monitoring"):
                st.session_state.monitoring = True
                st.session_state.cap = cv2.VideoCapture(cam_idx)
                st.session_state.orb = cv2.ORB_create()
                st.session_state.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                st.session_state.st_frame = st.empty()

    # --- Monitoring Loop Trigger ---
    if st.session_state.get("monitoring", False):
        monitor_step()

if __name__ == "__main__":
    main()
