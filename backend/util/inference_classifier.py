import pickle
import time
import cv2
import mediapipe as mp
import numpy as np

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
while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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

    cv2.imshow('frame', frame)
    cv2.waitKey(1)


cap.release()
cv2.destroyAllWindows()
