import os
import cv2
import json
import sys
import argparse
import mediapipe as mp

# ========== ARGUMENT PARSING ==========
parser = argparse.ArgumentParser(description='Collect ASL gesture images for training')
parser.add_argument('mode', choices=['one', 'two'], 
                    help='Collection mode: "one" for one-hand gestures, "two" for two-hand gestures')
parser.add_argument('--config', type=str, default=None,
                    help='Path to custom config file (default: asl_one_hand.json for one, asl.json for two)')
parser.add_argument('--camera', type=int, default=1,
                    help='Camera index (default: 1)')
parser.add_argument('--start-at', type=int, default=0,
                    help='Class index to start at (default: 0)')
parser.add_argument('--dataset-size', type=int, default=100,
                    help='Number of images to collect per class (default: 100)')

args = parser.parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')  # Shared data directory

# Determine mode and config
MODE = args.mode
REQUIRED_HANDS = 1 if MODE == 'one' else 2
MODE_NAME = "ONE-HAND" if MODE == 'one' else "TWO-HAND"
START_AT = args.start_at

# Set default config based on mode
if args.config:
    CONFIG_PATH = args.config
else:
    if MODE == 'one':
        CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'asl_one_hand.json')
    else:
        CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'asl.json')

if not os.path.exists(BASE_DATA_DIR):
    os.makedirs(BASE_DATA_DIR)

# Load config
try:
    with open(CONFIG_PATH, 'r') as f:
        asl_config = json.load(f)
        number_of_classes = len(asl_config)
        print(f"Loaded {number_of_classes} {MODE_NAME} sign language gestures from {os.path.basename(CONFIG_PATH)}")
        print(f"Mode: {MODE_NAME} (requires exactly {REQUIRED_HANDS} hand(s))")
except FileNotFoundError:
    print(f"Error: Config file not found at {CONFIG_PATH}")
    sys.exit(1)

dataset_size = args.dataset_size

# Setup camera
cap = cv2.VideoCapture(args.camera)

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

# Determine output directory based on mode
if MODE == 'one':
    output_root = os.path.join(BASE_DATA_DIR, "one_hand")
else:
    output_root = os.path.join(BASE_DATA_DIR, "two_hands")

os.makedirs(output_root, exist_ok=True)

# Create class directories
for j in range(number_of_classes):
    os.makedirs(os.path.join(output_root, str(j)), exist_ok=True)

print(f"Saving images to: {output_root}")

# ========== DATA COLLECTION ==========

for j in range(START_AT, number_of_classes):
    class_dir = os.path.join(output_root, str(j))
    class_label = asl_config.get(str(j), f"Class {j}")

    print(f'\n=== Collecting data for class {j}: "{class_label}" ===')
    print(f'Mode: {MODE_NAME} (show exactly {REQUIRED_HANDS} hand(s))')
    print(f'Goal: {dataset_size} images for this class')

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
        msg = f'Class {j}: "{class_label}" | {MODE_NAME} MODE'
        cv2.putText(display_frame, msg, (40, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        msg2 = f'Press "Q" to start capturing'
        cv2.putText(display_frame, msg2, (40, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', display_frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # -------- Capture loop --------
    img_count = 0

    while img_count < dataset_size:
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

        # Color code based on whether correct number of hands
        hands_color = (0, 255, 0) if num_hands == REQUIRED_HANDS else (0, 0, 255)
        
        status = f'Class {j}: "{class_label}" | Captured: {img_count}/{dataset_size}'
        cv2.putText(display_frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(display_frame, f'Detected: {mode_text}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, hands_color, 2, cv2.LINE_AA)
        cv2.putText(display_frame, f'Need: {REQUIRED_HANDS} hand(s)', (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(display_frame, 'Press ESC to skip this class', (10, H - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('frame', display_frame)

        key = cv2.waitKey(25) & 0xFF
        if key == 27:  # ESC → skip rest of this class
            print("Skipping remaining images for this class.")
            break

        # ===== save image if correct number of hands detected =====
        if results.multi_hand_landmarks and num_hands == REQUIRED_HANDS:
            img_path = os.path.join(class_dir, f'{img_count}.jpg')
            cv2.imwrite(img_path, clean_frame)
            img_count += 1
            print(f"  Saved image {img_count}/{dataset_size}")

    print(f'Finished class {j}: "{class_label}" - Collected {img_count} images')

cap.release()
cv2.destroyAllWindows()

print(f"\n{'='*60}")
print(f"Data collection complete!")
print(f"Mode: {MODE_NAME}")
print(f"Output directory: {output_root}")
print(f"Total classes: {number_of_classes}")
print(f"{'='*60}")
