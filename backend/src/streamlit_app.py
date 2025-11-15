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
            frame_placeholder.image(frame_rgb, channels="RGB", width='stretch')
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


import pickle
import mediapipe as mp
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(4)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

letters_detected = ""
previous_detected = ""
current_character = ""
character_start_time = None
HOLD_DURATION = 1.5  # seconds - adjust this value to change how long to hold

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}

def infer(frame):
    global current_character, letters_detected, previous_detected, character_start_time
    data_aux = []
    x_ = []
    y_ = []

    H, W, _ = frame.shape

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        # Only process the first hand to match training data (42 features)
        hand_landmarks = results.multi_hand_landmarks[0]

        mp_drawing.draw_landmarks(
            frame,  # image to draw
            hand_landmarks,  # model output
            mp_hands.HAND_CONNECTIONS,  # hand connections
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y

            x_.append(x)
            y_.append(y)

        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            data_aux.append(x - min(x_))
            data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        prediction = model.predict([np.asarray(data_aux)])

        predicted_character = labels_dict[int(prediction[0])]

        # Check if this is a new character or same as current
        if predicted_character != current_character:
            # New character detected, reset timer
            current_character = predicted_character
            character_start_time = time.time()
        else:
            # Same character, check if held long enough
            if character_start_time is not None:
                elapsed_time = time.time() - character_start_time
                if elapsed_time >= HOLD_DURATION and current_character != previous_detected:
                    # Character held long enough and is different from last added
                    letters_detected += current_character
                    previous_detected = current_character
                    print(f"Added '{current_character}' to detected letters: {letters_detected}")

        # Display the predicted character and progress
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

        # Show hold progress bar
        if character_start_time is not None and current_character != previous_detected:
            elapsed_time = time.time() - character_start_time
            progress = min(elapsed_time / HOLD_DURATION, 1.0)
            bar_width = int((x2 - x1) * progress)
            cv2.rectangle(frame, (x1, y2 + 5), (x1 + bar_width, y2 + 15), (0, 255, 0), -1)
            cv2.rectangle(frame, (x1, y2 + 5), (x2, y2 + 15), (0, 0, 0), 2)
    else:
        # No hand detected, reset tracking
        current_character = ""
        character_start_time = None

    # Display detected letters on screen
    cv2.putText(frame, f"Detected: {letters_detected}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                cv2.LINE_AA)


# Auto-refresh to create video effect
import time
while st.session_state is not None:
    time.sleep(0.03)  # ~30 FPS
    ret, frame = st.session_state.video_capture.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    infer(frame_rgb)
    frame_placeholder.image(frame_rgb, channels="RGB", width='stretch')