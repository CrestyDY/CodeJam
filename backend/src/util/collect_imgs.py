import os
import cv2
import json
import mediapipe as mp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'asl.json')
if not os.path.exists(BASE_DATA_DIR):
    os.makedirs(BASE_DATA_DIR)

with open(CONFIG_PATH, 'r') as f:
    asl_config = json.load(f)
    number_of_classes = len(asl_config)
    print(f"Loaded {number_of_classes} sign language gestures from config")

dataset_size = 100              # images per class (per model)

# change camera index if needed
cap = cv2.VideoCapture(1)

# ========== MEDIAPIPE SETUP ==========
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3,
    max_num_hands=2
)

# ========== DIRECTORY SETUP ==========

# We will store:
#  ./data/one_hand/<class_id>/
#  ./data/two_hands/<class_id>/
one_hand_root = os.path.join(BASE_DATA_DIR, "one_hand")
two_hands_root = os.path.join(BASE_DATA_DIR, "two_hands")

os.makedirs(one_hand_root, exist_ok=True)
os.makedirs(two_hands_root, exist_ok=True)

for j in range(number_of_classes):
    os.makedirs(os.path.join(one_hand_root, str(j)), exist_ok=True)
    os.makedirs(os.path.join(two_hands_root, str(j)), exist_ok=True)

# ========== DATA COLLECTION ==========

for j in range(number_of_classes):
    one_class_dir = os.path.join(one_hand_root, str(j))
    two_class_dir = os.path.join(two_hands_root, str(j))

    print(f'\n=== Collecting data for class {j} ===')
    print('Show the gesture for this class in front of the camera.')
    print('1 hand → goes to ONE-HAND dataset')
    print('2 hands → goes to TWO-HANDS dataset')
    print(f'Aim: {dataset_size} images for ONE-HAND, {dataset_size} for TWO-HANDS.')

    # -------- Ready screen --------
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera.")
            break

        display_frame = frame.copy()
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    display_frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
            )
        msg = f'Class {j} | Press "Q" to start capturing'
        cv2.putText(display_frame, msg, (40, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', display_frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # -------- Capture loop --------
    one_count = 0
    two_count = 0

    while one_count < dataset_size or two_count < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera.")
            break

        # keep a clean copy for saving
        clean_frame = frame.copy()
        display_frame = frame.copy()

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        mode_text = "No hands"
        num_hands = 0

        # ===== detect + draw landmarks on display_frame =====
        if results.multi_hand_landmarks:
            num_hands = len(results.multi_hand_landmarks)

            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    display_frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

            if num_hands == 1:
                mode_text = "ONE HAND"
            elif num_hands == 2:
                mode_text = "TWO HANDS"
            else:
                mode_text = f"{num_hands} hands (ignored)"

        status = f'Class {j} | 1H: {one_count}/{dataset_size}  2H: {two_count}/{dataset_size}'
        cv2.putText(display_frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(display_frame, f'Detected: {mode_text}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(display_frame, 'Press ESC to skip this class', (10, H - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('frame', display_frame)

        key = cv2.waitKey(25) & 0xFF
        if key == 27:  # ESC → skip rest of this class
            print("Skipping remaining images for this class.")
            break

        # ===== save based on # of hands (use clean_frame, not display_frame) =====
        if results.multi_hand_landmarks:
            if num_hands == 1 and one_count < dataset_size:
                img_path = os.path.join(one_class_dir, f'{one_count}.jpg')
                cv2.imwrite(img_path, clean_frame)
                one_count += 1

            elif num_hands == 2 and two_count < dataset_size:
                img_path = os.path.join(two_class_dir, f'{two_count}.jpg')
                cv2.imwrite(img_path, clean_frame)
                two_count += 1

    print(f'Finished class {j}: ONE-HAND={one_count}, TWO-HANDS={two_count}')

cap.release()
cv2.destroyAllWindows()
