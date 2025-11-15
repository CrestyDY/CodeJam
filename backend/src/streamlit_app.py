import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Video Feed Capture",
    page_icon="📹",
    layout="wide"
)

# Initialize session state
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'video_capture' not in st.session_state:
    st.session_state.video_capture = None

def start_camera():
    """Start the camera capture"""
    if st.session_state.video_capture is None:
        st.session_state.video_capture = cv2.VideoCapture(0)
        if not st.session_state.video_capture.isOpened():
            st.error("Unable to access camera. Please check your camera permissions.")
            st.session_state.video_capture = None
            return False
    st.session_state.camera_active = True
    return True

def stop_camera():
    """Stop the camera capture"""
    if st.session_state.video_capture is not None:
        st.session_state.video_capture.release()
        st.session_state.video_capture = None
    st.session_state.camera_active = False

# Main layout: Camera takes most of the screen, side column for controls/info
col_main, col_side = st.columns([4, 1])

with col_main:
    st.title("📹 Video Feed Capture")
    
    # Video display area - takes most of the screen
    frame_placeholder = st.empty()
    
    # Capture and display video frames
    if st.session_state.camera_active and st.session_state.video_capture is not None:
        ret, frame = st.session_state.video_capture.read()
        
        if ret:
            # Convert BGR to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Display the frame - large and prominent
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            # Auto-refresh to create video effect
            import time
            time.sleep(0.03)  # ~30 FPS
            st.rerun()
        else:
            frame_placeholder.error("Failed to capture frame from camera.")
            stop_camera()
    else:
        frame_placeholder.info("👆 Click 'Start Camera' to begin")

with col_side:
    st.markdown("### Controls")
    
    # Control buttons
    start_btn = st.button("🎥 Start", type="primary", disabled=st.session_state.camera_active, use_container_width=True)
    stop_btn = st.button("⏹️ Stop", disabled=not st.session_state.camera_active, use_container_width=True)
    
    # Handle button clicks
    if start_btn:
        if start_camera():
            st.rerun()
    
    if stop_btn:
        stop_camera()
        st.rerun()
    
    st.markdown("---")
    
    # Status indicator
    st.markdown("### Status")
    if st.session_state.camera_active:
        st.success("🟢 Active")
    else:
        st.info("⚪ Inactive")
    
    st.markdown("---")
    
    # Information section
    st.markdown("### ℹ️ Info")
    st.markdown("""
    This application captures video feed from your webcam.
    
    **Instructions:**
    1. Click "Start" to begin
    2. Grant camera permissions if prompted
    3. View your live video feed
    4. Click "Stop" when done
    """)
    
    # Frame information (if we have a captured frame)
    if st.session_state.camera_active and st.session_state.video_capture is not None:
        ret, frame = st.session_state.video_capture.read()
        if ret:
            st.markdown("---")
            st.markdown("### Frame Info")
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.write(f"**Shape:** {frame_rgb.shape}")
            st.write(f"**Size:** {frame_rgb.size} px")
            st.write(f"**Type:** {frame_rgb.dtype}")

