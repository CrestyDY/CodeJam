import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Video Feed Capture",
    page_icon="📹",
    layout="wide"
)

st.sidebar.markdown("# Settings")
st.sidebar.checkbox("Flip Camera", value=True, key="flip_camera")
st.sidebar.slider("Confidence Level", min_value=0.0, max_value=1.0, value=0.5, step=0.05, key="confidence_level")

# Initialize session state
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'video_capture' not in st.session_state:
    st.session_state.video_capture = None
if 'signs' not in st.session_state:
    st.session_state.signs = ""


def start_camera():
    """Start the camera capture"""
    if st.session_state.video_capture is None:
        st.session_state.video_capture = cv2.VideoCapture(st.session_state.camera_index)
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
    size = 0.50
    left, right = st.columns([size, 1 - size])
    with left:
        frame_placeholder = st.empty()
    with right:
        st.markdown("### Detected Signs")
        signs_placeholder = st.empty()
        signs_placeholder.markdown(
            f"<div style='font-size:24px; border:2px solid black; padding:10px; min-height:100px;'>{st.session_state.signs}</div>",
            unsafe_allow_html=True)

    # Capture and display video frames
    if st.session_state.camera_active and st.session_state.video_capture is not None:
        ret, frame = st.session_state.video_capture.read()

        if ret:
            # Convert BGR to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if st.session_state.flip_camera:
                frame_rgb = cv2.flip(frame_rgb, 1)  # Mirror the frame for a more natural webcam feel

            # Display the frame - large and prominent
            frame_placeholder.image(frame_rgb, channels="RGB", width='stretch')
            st.session_state.camera_error = False
        else:
            st.session_state.camera_error = True
            frame_placeholder.error("Failed to capture frame from camera.")
            stop_camera()
    else:
        frame_placeholder.info("Click 'Start Camera' to begin 👉")

with col_side:
    st.markdown("### Controls")

    # Control buttons
    start_btn = False
    stop_btn = False
    if not st.session_state.camera_active:
        st.info("⚪ Inactive")
        start_btn = st.button("🎥 Start", type="primary", disabled=st.session_state.camera_active,
                              use_container_width=True)
        st.number_input("Camera Index", min_value=0, max_value=10, disabled=st.session_state.camera_active, value=0,
                        step=1, key="camera_index")
    else:
        st.success("🟢 Active")
        stop_btn = st.button("⏹️ Stop", disabled=not st.session_state.camera_active, use_container_width=True)

    # Handle button clicks
    if start_btn:
        if start_camera():
            st.rerun()

    if stop_btn:
        stop_camera()
        st.rerun()

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


@st.cache_resource
def load_models():
    model1_dict = pickle.load(open('util/models/model_one_hand.p', 'rb'))
    model2_dict = pickle.load(open('util/models/model_two_hands.p', 'rb'))
    return model1_dict['model'], model2_dict['model']


model_one, model_two = load_models()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=st.session_state.confidence_level)

if 'letters_detected' not in st.session_state:
    st.session_state.letters_detected = ""
if 'previous_detected' not in st.session_state:
    st.session_state.previous_detected = ""
if 'current_character' not in st.session_state:
    st.session_state.current_character = ""
if 'character_start_time' not in st.session_state:
    st.session_state.character_start_time = None

HOLD_DURATION = 1.5  # seconds - adjust this value to change how long to hold
default_labels_one = {0: 'HI', 1: 'MY', 2: 'H', 3: 'E'}  # one-hand model
default_labels_two = {0: 'NAME', 1: 'interpreter', 2: 'world', 3: 'L'}  # two-hand model


def infer(frame):
    predicted_character = None
    H, W, _ = frame.shape

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 1:
        st.session_state.mode_text = "Mode: ONE-HAND"
        hand_landmarks = results.multi_hand_landmarks[0]

        data_aux = []
        x_ = []
        y_ = []

        for lm in hand_landmarks.landmark:
            x = lm.x
            y = lm.y
            x_.append(x)
            y_.append(y)

        min_x = min(x_)
        min_y = min(y_)

        for lm in hand_landmarks.landmark:
            x = lm.x
            y = lm.y
            data_aux.append(x - min_x)
            data_aux.append(y - min_y)

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) + 10
        y2 = int(max(y_) * H) + 10

        if len(data_aux) == 42:
            prediction = model_one.predict([np.asarray(data_aux)])
            predicted_character = default_labels_one[int(prediction[0])]
        else:
            st.error(f"Error: {len(data_aux)} features (need 42)")

        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )
    elif results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
        st.session_state.mode_text = "Mode: TWO-HAND"
        data_aux = []
        all_x = []
        all_y = []

        # Process both hands and combine their features (same as your 2-hand code)
        for hand_landmarks in results.multi_hand_landmarks:
            x_hand = []
            y_hand = []

            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Collect coords
            for lm in hand_landmarks.landmark:
                x = lm.x
                y = lm.y
                x_hand.append(x)
                y_hand.append(y)
                all_x.append(x)
                all_y.append(y)

            # Normalize per-hand
            min_x_hand = min(x_hand)
            min_y_hand = min(y_hand)

            for lm in hand_landmarks.landmark:
                x = lm.x
                y = lm.y
                data_aux.append(x - min_x_hand)
                data_aux.append(y - min_y_hand)

        # Combined bounding box across both hands
        x1 = int(min(all_x) * W) - 10
        y1 = int(min(all_y) * H) - 10
        x2 = int(max(all_x) * W) + 10
        y2 = int(max(all_y) * H) + 10

        if len(data_aux) == 84:
            prediction = model_two.predict([np.asarray(data_aux)])
            predicted_character = default_labels_two[int(prediction[0])]
        else:
            st.error(f"Error: {len(data_aux)} features (need 84)")
    else:
        # No hand detected, reset tracking
        st.session_state.current_character = ""
        st.session_state.character_start_time = None

    if predicted_character is not None:
        # New or existing character?
        if predicted_character != st.session_state.current_character:
            st.session_state.current_character = predicted_character
            st.session_state.character_start_time = time.time()
        else:
            if st.session_state.character_start_time is not None:
                elapsed_time = time.time() - st.session_state.character_start_time
                if elapsed_time >= HOLD_DURATION and st.session_state.current_character != st.session_state.previous_detected:
                    st.session_state.letters_detected += ' ' + st.session_state.current_character
                    st.session_state.previous_detected = st.session_state.current_character
                    print(
                        f"Added '{st.session_state.current_character}' to detected letters: {st.session_state.letters_detected}")

        # Draw bounding box and predicted char
        if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
            color = (0, 0, 0) if st.session_state.mode_text.endswith("ONE-HAND") else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3, cv2.LINE_AA)

        # Progress bar (only when different from last added)
        if st.session_state.character_start_time is not None and st.session_state.current_character != st.session_state.previous_detected and x1 is not None:
            elapsed_time = time.time() - st.session_state.character_start_time
            progress = min(elapsed_time / HOLD_DURATION, 1.0)
            bar_width = int((x2 - x1) * progress)
            cv2.rectangle(frame, (x1, y2 + 5), (x1 + bar_width, y2 + 15), (0, 255, 0), -1)
            cv2.rectangle(frame, (x1, y2 + 5), (x2, y2 + 15), (0, 0, 0), 2)

    # Display detected letters on screen
    cv2.putText(frame, f"Detected: {st.session_state.letters_detected}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255), 2,
                cv2.LINE_AA)


# Auto-refresh to create video effect
import time

while st.session_state.video_capture is not None:
    time.sleep(0.03)  # ~30 FPS
    ret, frame = st.session_state.video_capture.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if st.session_state.flip_camera:
        frame_rgb = cv2.flip(frame_rgb, 1)  # Mirror the frame for a more natural webcam feel
    infer(frame_rgb)
    signs_placeholder.markdown(
        f"<div style='font-size:24px; border:2px solid black; padding:10px; min-height:100px;'>{st.session_state.letters_detected}</div>",
        unsafe_allow_html=True)
    frame_placeholder.image(frame_rgb, channels="RGB", width='stretch')