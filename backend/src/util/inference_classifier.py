import pickle
import time
import cv2
import mediapipe as mp
import numpy as np
import threading
from src.ai.get_llm_response import get_response
import os

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'model.p')

model_dict = pickle.load(open(MODEL_PATH, 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, max_num_hands=2)

letters_detected = ""
previous_detected = ""
current_character = ""
character_start_time = None
HOLD_DURATION = 1.5  # seconds - adjust this value to change how long to hold

labels_dict = {0: 'HI ', 1: 'MOM ', 2: 'HELLO ', 3: 'WORLD ', 4: ':) '}

def run_llm_in_background(user_input):
    """Run the LLM call in a separate thread to avoid blocking the video feed"""
    def thread_target():
        try:
            # get_response is now synchronous but uses a persistent event loop internally
            response = get_response(user_input)
            if response:
                print(f"\n{'='*60}")
                print(f"📝 LLM Response for '{user_input}':")
                print(f"{'='*60}")
                print(response)
                print(f"{'='*60}\n")
        except Exception as e:
            print(f"\n❌ Error getting LLM response: {e}\n")
    
    thread = threading.Thread(target=thread_target, daemon=True)
    thread.start()

while True:
    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    
    # Only process if exactly 2 hands are detected (combined gesture)
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
        data_aux = []
        all_x = []
        all_y = []
        
        # Process both hands and combine their features
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            x_hand = []
            y_hand = []
            
            # Draw landmarks for this hand
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            
            # Extract landmarks for this hand
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_hand.append(x)
                y_hand.append(y)
                all_x.append(x)
                all_y.append(y)
            
            # Normalize coordinates relative to this hand's bounding box
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_hand))
                data_aux.append(y - min(y_hand))
        
        # Get combined bounding box for both hands
        x1 = int(min(all_x) * W) - 10
        y1 = int(min(all_y) * H) - 10
        x2 = int(max(all_x) * W) + 10
        y2 = int(max(all_y) * H) + 10
        
        # Make prediction using combined features (84 features: 42 per hand)
        if len(data_aux) == 84:
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
                        # Run the LLM call in background thread so video doesn't freeze
                        run_llm_in_background(letters_detected)
            
            # Display the predicted character
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                        cv2.LINE_AA)
            
            # Show hold progress bar
            if character_start_time is not None and current_character != previous_detected:
                elapsed_time = time.time() - character_start_time
                progress = min(elapsed_time / HOLD_DURATION, 1.0)
                bar_width = int((x2 - x1) * progress)
                cv2.rectangle(frame, (x1, y2 + 5), (x1 + bar_width, y2 + 15), (0, 255, 0), -1)
                cv2.rectangle(frame, (x1, y2 + 5), (x2, y2 + 15), (0, 0, 0), 2)
        else:
            # Invalid feature count
            cv2.putText(frame, f"Error: {len(data_aux)} features (need 84)", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        # Not both hands detected
        num_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
        
        # Draw detected hands
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
        
        # Show status message
        status_text = f"Need BOTH hands! Currently: {num_hands}/2"
        cv2.putText(frame, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Reset tracking when not both hands
        current_character = ""
        character_start_time = None

    # Display detected letters on screen
    cv2.putText(frame, f"Detected: {letters_detected}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                cv2.LINE_AA)

    cv2.imshow('frame', frame)
    cv2.waitKey(1)


cap.release()
cv2.destroyAllWindows()
