import streamlit as st
import cv2
import numpy as np
import time
import pickle
import mediapipe as mp
from collections import deque
import os

# Try to import TensorFlow for motion detection
try:
    from tensorflow import keras
    MOTION_AVAILABLE = True
except ImportError:
    MOTION_AVAILABLE = False
    keras = None

st.set_page_config(page_title="Sign Language Recognition", layout="wide")

# Initialize session state
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'video_capture' not in st.session_state:
    st.session_state.video_capture = None
if 'signs' not in st.session_state:
    st.session_state.signs = ""
if 'camera_index' not in st.session_state:
    st.session_state.camera_index = 0
if 'detection_mode' not in st.session_state:
    st.session_state.detection_mode = "static"
if 'motion_buffer' not in st.session_state:
    st.session_state.motion_buffer = deque(maxlen=30)
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None
if 'last_prediction_time' not in st.session_state:
    st.session_state.last_prediction_time = 0

# Sidebar configuration
st.sidebar.markdown("# ⚙️ Settings")

# Detection mode selector
if MOTION_AVAILABLE:
    detection_mode = st.sidebar.radio(
        "Detection Mode",
        ["Static (Letters/Words)", "Motion (Gestures)"],
        index=0 if st.session_state.detection_mode == "static" else 1
    )
    st.session_state.detection_mode = "static" if detection_mode.startswith("Static") else "motion"
else:
    st.sidebar.info("📦 Install TensorFlow to enable Motion Detection:\npip install tensorflow")
    st.session_state.detection_mode = "static"

flip_camera = st.sidebar.checkbox("Flip Camera", value=True)
camera_index = st.sidebar.number_input("Camera Index", min_value=0, max_value=10, value=st.session_state.camera_index, step=1)
st.session_state.camera_index = int(camera_index)

# Motion detection settings
if st.session_state.detection_mode == "motion" and MOTION_AVAILABLE:
    st.sidebar.markdown("### Motion Settings")
    sequence_length = st.sidebar.slider("Sequence Length (frames)", 15, 60, 30)
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.5, 1.0, 0.7, 0.05)
    st.session_state.motion_buffer = deque(maxlen=sequence_length)

# Load static models
@st.cache_resource
def load_static_models():
    try:
        model_one_dict = pickle.load(open('util/models/model_one_hand.p', 'rb'))
        model_two_dict = pickle.load(open('util/models/model_two_hands.p', 'rb'))
        return model_one_dict['model'], model_two_dict['model']
    except FileNotFoundError:
        st.error("Static models not found. Please train them first.")
        return None, None

# Load motion model
@st.cache_resource
def load_motion_model():
    if not MOTION_AVAILABLE:
        return None, None

    try:
        model_path = 'util/models/motion_model.h5'
        metadata_path = 'util/models/motion_model_metadata.pkl'

        if not os.path.exists(model_path) or not os.path.exists(metadata_path):
            return None, None

        model = keras.models.load_model(model_path)
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)

        return model, metadata
    except Exception as e:
        st.error(f"Error loading motion model: {e}")
        return None, None

# Load appropriate models
model_one, model_two = load_static_models()
motion_model, motion_metadata = load_motion_model() if MOTION_AVAILABLE else (None, None)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Labels for static detection
labels_one = {0: 'HI', 1: 'MY', 2: 'H', 3: 'E'}
labels_two = {0: 'NAME', 1: 'interpreter', 2: 'world', 3: 'L'}

# Helper function to extract hand features (for motion detection)
def extract_hand_features(hand_landmarks):
    """Extract normalized features from a single hand."""
    data_aux = []
    x_ = []
    y_ = []

    for lm in hand_landmarks.landmark:
        x_.append(lm.x)
        y_.append(lm.y)

    min_x = min(x_)
    min_y = min(y_)

    for lm in hand_landmarks.landmark:
        data_aux.append(lm.x - min_x)
        data_aux.append(lm.y - min_y)

    return data_aux

# Initialize MediaPipe hands based on mode
def get_hands_detector(mode="static"):
    """Get MediaPipe hands detector configured for the mode."""
    if mode == "static":
        return mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, max_num_hands=2)
    else:  # motion
        return mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5,
                             min_tracking_confidence=0.5, max_num_hands=2)

# Layout
col_main, col_side = st.columns([4, 1])

with col_main:
    st.title("📹 Sign Language Recognition")

    # Display current mode
    if st.session_state.detection_mode == "static":
        st.info("🔷 Static Mode: Detecting letters and static words")
    else:
        if motion_model is None:
            st.warning("🌊 Motion Mode: Model not found. Please train it first.")
        else:
            st.info("🌊 Motion Mode: Detecting gestures that involve movement")

    left, right = st.columns([0.6, 0.4])

    with left:
        frame_placeholder = st.empty()
    with right:
        st.markdown("### 📝 Detected Signs")
        sign_placeholder = st.empty()

        if st.session_state.detection_mode == "motion":
            st.markdown("### 📊 Buffer Status")
            buffer_placeholder = st.empty()

with col_side:
    st.markdown("### Controls")

    if not st.session_state.camera_active:
        if st.button("🎥 Start", type="primary", use_container_width=True):
            st.session_state.video_capture = cv2.VideoCapture(st.session_state.camera_index)
            if st.session_state.video_capture.isOpened():
                st.session_state.camera_active = True
                st.rerun()
            else:
                st.error("Cannot open camera")
    else:
        if st.button("⏹️ Stop", use_container_width=True):
            if st.session_state.video_capture:
                st.session_state.video_capture.release()
            st.session_state.camera_active = False
            st.session_state.video_capture = None
            st.rerun()

    st.markdown("---")
    if st.session_state.camera_active:
        st.success("🟢 Active")
    else:
        st.info("⚪ Inactive")

    if st.button("🔄 Clear Signs", use_container_width=True):
        st.session_state.signs = ""
        st.session_state.motion_buffer.clear()
        st.session_state.last_prediction = None

# Video processing loop
if st.session_state.camera_active and st.session_state.video_capture:
    letters_detected = st.session_state.signs

    # Initialize MediaPipe with appropriate mode
    hands = get_hands_detector(st.session_state.detection_mode)

    # Static detection variables
    previous_detected = ""
    current_character = ""
    character_start_time = None
    HOLD_DURATION = 1.5

    # Motion detection variables
    prediction_cooldown = 2.0

    # Run video capture in a tight loop (no st.rerun() calls)
    frame_count = 0
    while st.session_state.camera_active and st.session_state.video_capture:
        ret, frame = st.session_state.video_capture.read()
        if not ret:
            st.error("Failed to read frame")
            break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if flip_camera:
            frame_rgb = cv2.flip(frame_rgb, 1)

        results = hands.process(frame_rgb)
        predicted_character = None
        num_hands = 0
        current_time = time.time()

        # ===== MODE-SPECIFIC PROCESSING =====

        if st.session_state.detection_mode == "static":
            # Static detection logic
            if results.multi_hand_landmarks:
                num_hands = len(results.multi_hand_landmarks)

                if num_hands == 1 and model_one:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    data_aux, x_, y_ = [], [], []

                    for lm in hand_landmarks.landmark:
                        x_.append(lm.x)
                        y_.append(lm.y)

                    min_x, min_y = min(x_), min(y_)
                    for lm in hand_landmarks.landmark:
                        data_aux.append(lm.x - min_x)
                        data_aux.append(lm.y - min_y)

                    if len(data_aux) == 42:
                        pred = model_one.predict([np.asarray(data_aux)])
                        predicted_character = labels_one[int(pred[0])]

                    x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
                    x2, y2 = int(max(x_) * W) + 10, int(max(y_) * H) + 10
                    cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 0, 0), 4)

                    mp_drawing.draw_landmarks(frame_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                             mp_drawing_styles.get_default_hand_landmarks_style(),
                                             mp_drawing_styles.get_default_hand_connections_style())

                elif num_hands == 2 and model_two:
                    data_aux, all_x, all_y = [], [], []

                    for hand_landmarks in results.multi_hand_landmarks:
                        x_hand, y_hand = [], []
                        for lm in hand_landmarks.landmark:
                            x_hand.append(lm.x)
                            y_hand.append(lm.y)
                            all_x.append(lm.x)
                            all_y.append(lm.y)

                        min_x_hand, min_y_hand = min(x_hand), min(y_hand)
                        for lm in hand_landmarks.landmark:
                            data_aux.append(lm.x - min_x_hand)
                            data_aux.append(lm.y - min_y_hand)

                        mp_drawing.draw_landmarks(frame_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                                 mp_drawing_styles.get_default_hand_landmarks_style(),
                                                 mp_drawing_styles.get_default_hand_connections_style())

                    if len(data_aux) == 84:
                        pred = model_two.predict([np.asarray(data_aux)])
                        predicted_character = labels_two[int(pred[0])]

                    x1, y1 = int(min(all_x) * W) - 10, int(min(all_y) * H) - 10
                    x2, y2 = int(max(all_x) * W) + 10, int(max(all_y) * H) + 10
                    cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 4)

            # Character hold logic for static mode
            if predicted_character:
                if predicted_character != current_character:
                    current_character = predicted_character
                    character_start_time = time.time()
                else:
                    if character_start_time and time.time() - character_start_time >= HOLD_DURATION:
                        if current_character != previous_detected:
                            letters_detected += current_character
                            st.session_state.signs = letters_detected
                            previous_detected = current_character
            else:
                current_character = ""
                character_start_time = None

        else:  # Motion detection mode
            if motion_model is None or motion_metadata is None:
                cv2.putText(frame_rgb, "Motion model not loaded", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                if results.multi_hand_landmarks:
                    num_hands = len(results.multi_hand_landmarks)
                    expected_hands = 1 if motion_metadata['feature_dim'] == 42 else 2

                    if num_hands == expected_hands:
                        # Extract features
                        if expected_hands == 1:
                            features = extract_hand_features(results.multi_hand_landmarks[0])
                        else:
                            features = []
                            for hand_landmarks in results.multi_hand_landmarks:
                                features.extend(extract_hand_features(hand_landmarks))

                        if len(features) == motion_metadata['feature_dim']:
                            st.session_state.motion_buffer.append(features)

                        # Draw landmarks
                        for hand_landmarks in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(frame_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                                     mp_drawing_styles.get_default_hand_landmarks_style(),
                                                     mp_drawing_styles.get_default_hand_connections_style())

                    # Make prediction if buffer is full
                    if len(st.session_state.motion_buffer) >= sequence_length:
                        sequence = np.array([list(st.session_state.motion_buffer)])
                        predictions = motion_model.predict(sequence, verbose=0)[0]

                        predicted_idx = np.argmax(predictions)
                        confidence = predictions[predicted_idx]

                        if confidence >= confidence_threshold:
                            predicted_sign = motion_metadata['classes'][predicted_idx]

                            if (st.session_state.last_prediction != predicted_sign or
                                current_time - st.session_state.last_prediction_time > prediction_cooldown):
                                letters_detected += predicted_sign + " "
                                st.session_state.signs = letters_detected
                                st.session_state.last_prediction = predicted_sign
                                st.session_state.last_prediction_time = current_time
                                st.session_state.motion_buffer.clear()

        # Draw text on frame
        if st.session_state.detection_mode == "static":
            cv2.putText(frame_rgb, f"Hands: {num_hands}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(frame_rgb, f"Signs: {letters_detected}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:  # Motion mode
            buffer_fill = len(st.session_state.motion_buffer)
            buffer_max = sequence_length if 'sequence_length' in locals() else 30
            buffer_percent = (buffer_fill / buffer_max) * 100

            cv2.putText(frame_rgb, f"Buffer: {buffer_fill}/{buffer_max} ({buffer_percent:.0f}%)",
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame_rgb, f"Signs: {letters_detected}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Progress bar
            bar_width = 300
            bar_height = 15
            bar_x, bar_y = 10, 80
            cv2.rectangle(frame_rgb, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                         (255, 255, 255), 2)
            filled_width = int((buffer_fill / buffer_max) * bar_width)
            if filled_width > 0:
                cv2.rectangle(frame_rgb, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height),
                             (0, 255, 0), -1)

        # Update placeholders (no st.rerun())
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        sign_placeholder.markdown(
            f"<div style='font-size:28px; border:2px solid black; padding:15px; min-height:120px; word-wrap:break-word;'>{st.session_state.signs}</div>",
            unsafe_allow_html=True
        )

        if st.session_state.detection_mode == "motion" and 'buffer_placeholder' in locals():
            buffer_percent = (len(st.session_state.motion_buffer) / (sequence_length if 'sequence_length' in locals() else 30)) * 100
            buffer_placeholder.progress(buffer_percent / 100)

        frame_count += 1
        time.sleep(0.01)  # ~100 FPS target