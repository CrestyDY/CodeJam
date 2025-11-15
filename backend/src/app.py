import streamlit as st
import cv2
import numpy as np
import time
import pickle
import mediapipe as mp

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

# Sidebar configuration
st.sidebar.markdown("# ⚙️ Settings")
flip_camera = st.sidebar.checkbox("Flip Camera", value=True)
camera_index = st.sidebar.number_input("Camera Index", min_value=0, max_value=10, value=st.session_state.camera_index, step=1)
st.session_state.camera_index = int(camera_index)

# Load models
@st.cache_resource
def load_models():
    model_one_dict = pickle.load(open('util/models/model_one_hand.p', 'rb'))
    model_two_dict = pickle.load(open('util/models/model_two_hands.p', 'rb'))
    return model_one_dict['model'], model_two_dict['model']

model_one, model_two = load_models()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, max_num_hands=2)

labels_one = {0: 'HI', 1: 'MY', 2: 'H', 3: 'E'}
labels_two = {0: 'NAME', 1: 'interpreter', 2: 'world', 3: 'L'}

# Layout
col_main, col_side = st.columns([4, 1])

with col_main:
    st.title("📹 Sign Language Recognition")
    left, right = st.columns([0.6, 0.4])

    with left:
        frame_placeholder = st.empty()
    with right:
        st.markdown("### 📝 Detected Signs")
        sign_placeholder = st.empty()

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

# Video processing loop
if st.session_state.camera_active and st.session_state.video_capture:
    letters_detected = st.session_state.signs
    previous_detected = ""
    current_character = ""
    character_start_time = None
    HOLD_DURATION = 1.5

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

        if results.multi_hand_landmarks:
            num_hands = len(results.multi_hand_landmarks)

            if num_hands == 1:
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

            elif num_hands == 2:
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

        # Character hold logic
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

        # Draw text on frame
        cv2.putText(frame_rgb, f"Hands: {num_hands}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame_rgb, f"Signs: {letters_detected}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Update placeholders (no st.rerun())
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        sign_placeholder.markdown(
            f"<div style='font-size:28px; border:2px solid black; padding:15px; min-height:120px; word-wrap:break-word;'>{st.session_state.signs}</div>",
            unsafe_allow_html=True
        )

        frame_count += 1
        time.sleep(0.01)  # ~100 FPS target