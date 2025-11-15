import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
PICKLE_PATH = os.path.join(SCRIPT_DIR, 'data.pickle')

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.2, max_num_hands=2)

data = []
labels = []
skipped_images = 0

for dir_ in os.listdir(DATA_DIR):
    if dir_ == ".gitkeep":
        continue
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        # Only process if exactly 2 hands are detected
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
            data_aux = []
            
            # Process both hands and combine their features
            for hand_landmarks in results.multi_hand_landmarks:
                x_hand = []
                y_hand = []
                
                # Extract landmarks for this hand
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_hand.append(x)
                    y_hand.append(y)
                
                # Normalize coordinates relative to this hand's bounding box
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_hand))
                    data_aux.append(y - min(y_hand))
            
            # Should have 84 features (42 per hand: 21 landmarks * 2 coordinates)
            if len(data_aux) == 84:
                data.append(data_aux)
                labels.append(dir_)
            else:
                print(f"Warning: Expected 84 features, got {len(data_aux)} for {img_path}")
                skipped_images += 1
        else:
            num_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
            print(f"Skipping {img_path}: {num_hands} hands detected (need 2)")
            skipped_images += 1

print(f"Processed {len(data)} images successfully")
print(f"Skipped {skipped_images} images")
print(f"Feature vector size: {len(data[0]) if data else 0} features (84 = 2 hands * 42 features)")

f = open(PICKLE_PATH, 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
print(f"Saved {len(data)} samples to data.pickle")
