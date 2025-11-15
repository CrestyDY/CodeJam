import pickle
import time
import cv2
import mediapipe as mp
import numpy as np

#Load both models
# Change filenames if you save them differently
model_one_dict = pickle.load(open('models/model_one_hand.p', 'rb'))   # 42-feature model
model_two_dict = pickle.load(open('models/model_two_hands.p', 'rb'))   # 84-feature model

model_one = model_one_dict['model']
model_two = model_two_dict['model']

print("model 1 features:", model_one.n_features_in_)
print("model 2 features:", model_two.n_features_in_)


# Set up camera, use 0, 1 or 4 depending on setup(Windows, Mac or Linux)
cap = cv2.VideoCapture(0)

#setup mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=True,
    min_detection_confidence=0.3,
    max_num_hands=2   
)

letters_detected = ""
previous_detected = ""
current_character = ""
character_start_time = None
HOLD_DURATION = 1.5  # seconds to confirm a character

#training sets
default_labels_one = {0: 'HI', 1: 'MY', 2: 'H', 3: 'E'}   # one-hand model
default_labels_two = {0: 'NAME', 1: 'interpreter', 2: 'world',3:'L'}   # two-hand model

print("Starting auto-switch inference: 1 hand -> 1-hand model, 2 hands -> 2-hand model.")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from camera.")
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    predicted_character = None
    x1 = y1 = x2 = y2 = None  # for drawing bounding boxes
    status_text = ""
    mode_text = ""

    if results.multi_hand_landmarks:
        num_hands = len(results.multi_hand_landmarks)

       #exactly one hand detected
        if num_hands == 1:
            mode_text = "Mode: ONE-HAND"
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
                status_text = f"Error: {len(data_aux)} features (need 42)"

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

       #exactly 2 hands detected
        elif num_hands == 2:
            mode_text = "Mode: TWO-HAND"
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
                status_text = f"Error: {len(data_aux)} features (need 84)"

       #>=2 hands detected
        else:
            mode_text = "Mode: UNKNOWN"
            status_text = f"Unexpected number of hands: {num_hands}"
            current_character = ""
            character_start_time = None

    else:
        # No hands detected
        mode_text = "Mode: NONE"
        status_text = "No hands detected"
        current_character = ""
        character_start_time = None


    if predicted_character is not None:
        # New or existing character?
        if predicted_character != current_character:
            current_character = predicted_character
            character_start_time = time.time()
        else:
            if character_start_time is not None:
                elapsed_time = time.time() - character_start_time
                if elapsed_time >= HOLD_DURATION and current_character != previous_detected:
                    letters_detected += current_character
                    previous_detected = current_character
                    print(f"Added '{current_character}' to detected letters: {letters_detected}")

        # Draw bounding box and predicted char
        if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
            color = (0, 0, 0) if mode_text.endswith("ONE-HAND") else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3, cv2.LINE_AA)

        # Progress bar (only when different from last added)
        if character_start_time is not None and current_character != previous_detected and x1 is not None:
            elapsed_time = time.time() - character_start_time
            progress = min(elapsed_time / HOLD_DURATION, 1.0)
            bar_width = int((x2 - x1) * progress)
            cv2.rectangle(frame, (x1, y2 + 5), (x1 + bar_width, y2 + 15), (0, 255, 0), -1)
            cv2.rectangle(frame, (x1, y2 + 5), (x2, y2 + 15), (0, 0, 0), 2)
    else:
        # No valid prediction this frame → reset only current (keep accumulated letters)
        current_character = ""
        character_start_time = None


    cv2.putText(frame, mode_text, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

    cv2.putText(frame, f"Detected: {letters_detected}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

    if status_text:
        cv2.putText(frame, status_text, (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.putText(frame, "q: quit", (10, H - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('ASL Inference (Auto Switch)', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
