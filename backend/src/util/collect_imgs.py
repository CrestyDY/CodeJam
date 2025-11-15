import os
import json
import cv2
import mediapipe as mp
from time import sleep

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, max_num_hands=2)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'asl.json')

with open(CONFIG_PATH, 'r') as f:
    asl_config = json.load(f)
    number_of_classes = len(asl_config)
    print(f"Loaded {number_of_classes} sign language gestures from config")


if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

dataset_size = 100

cap = cv2.VideoCapture(0)
for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for word {}'.format(asl_config[str(j)]))

    done = False
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        # Check if both hands are detected
        num_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
        
        # Draw hands if detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
        
        # Display status
        status_text = f'Ready? Press "Q" ! Hands: {num_hands}/2'
        color = (0, 255, 0) if num_hands == 2 else (0, 0, 255)
        cv2.putText(frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
        cv2.putText(frame, 'Need BOTH hands to collect data!', (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break
    sleep(1)

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        # Only capture if both hands are detected
        num_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
        
        if num_hands == 2:
            # Verify detection one more time with the actual frame we're about to save
            # This ensures consistency with static_image_mode=True
            verify_results = hands.process(frame_rgb)
            verify_num_hands = len(verify_results.multi_hand_landmarks) if verify_results.multi_hand_landmarks else 0
            
            if verify_num_hands == 2:
                # Save the original frame WITHOUT annotations
                cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)
                
                # Draw both hands for display only
                frame_display = frame.copy()
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame_display,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())
                
                # Display collection status
                status_text = f'Collecting: {counter + 1}/{dataset_size} (Both hands detected)'
                cv2.putText(frame_display, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow('frame', frame_display)
                cv2.waitKey(25)
                counter += 1
            else:
                # Detection failed on verification, skip this frame
                status_text = f'Verification failed: {verify_num_hands}/2 hands'
                cv2.putText(frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, f'Collected: {counter}/{dataset_size}', (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow('frame', frame)
                cv2.waitKey(25)
        else:
            # Show warning if not both hands
            status_text = f'Need BOTH hands! Currently: {num_hands}/2'
            cv2.putText(frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Collected: {counter}/{dataset_size}', (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('frame', frame)
            cv2.waitKey(25)

cap.release()
cv2.destroyAllWindows()
