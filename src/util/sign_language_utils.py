"""
Utility functions for sign language recognition
Extracted from inference_classifier.py for reuse in web apps
"""
import pickle
import json
import os
import time
import cv2
import mediapipe as mp
import numpy as np

def load_models():
    """Load both one-hand and two-hand trained models"""
    script_dir = os.path.dirname(__file__)
    
    # Try to load separate one-hand and two-hand models
    try:
        model_one_dict = pickle.load(open(os.path.join(script_dir, 'models', 'model_one_hand.p'), 'rb'))
        model_two_dict = pickle.load(open(os.path.join(script_dir, 'models', 'model_two_hands.p'), 'rb'))
        return model_one_dict['model'], model_two_dict['model']
    except FileNotFoundError:
        # Fall back to single model for both (compatible with original dataset)
        try:
            model_dict = pickle.load(open(os.path.join(script_dir, 'model.p'), 'rb'))
            model = model_dict['model']
            return model, model  # Use same model for both one and two hands
        except FileNotFoundError:
            raise FileNotFoundError("No trained models found. Please train models first.")

def load_labels(config_file='asl_words_two_hands.json', config_file_one_hand='asl_words_one_hand.json'):
    """Load label dictionaries from config files"""
    # Get the src directory (parent of util directory)
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_dir = os.path.join(src_dir, 'config')
    
    # Default labels matching your current 5-class dataset (0-4)
    default_labels = {0: 'CLASS_0 ', 1: 'CLASS_1 ', 2: 'CLASS_2 ', 3: 'CLASS_3 ', 4: 'CLASS_4 '}
    
    # Try to load two-hand gestures
    labels_dict_two = default_labels.copy()
    CONFIG_PATH_TWO = os.path.join(config_dir, config_file)
    try:
        with open(CONFIG_PATH_TWO, 'r') as f:
            asl_config_two = json.load(f)
            labels_dict_two = {int(k): v + ' ' for k, v in asl_config_two.items()}
    except FileNotFoundError:
        pass  # Use default
    
    # Try to load one-hand gestures
    labels_dict_one = default_labels.copy()
    CONFIG_PATH_ONE = os.path.join(config_dir, config_file_one_hand)
    try:
        with open(CONFIG_PATH_ONE, 'r') as f:
            asl_config_one = json.load(f)
            labels_dict_one = {int(k): v + ' ' for k, v in asl_config_one.items()}
    except FileNotFoundError:
        pass  # Use default
    
    return labels_dict_one, labels_dict_two

def setup_mediapipe():
    """Setup MediaPipe hands detection (optimized for speed)"""
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    # Optimized settings for better framerate
    hands = mp_hands.Hands(
        static_image_mode=False,  # False is faster for video
        model_complexity=0,  # 0 is fastest, 1 is default
        min_detection_confidence=0.5,  # Lower = faster but less accurate
        min_tracking_confidence=0.5,
        max_num_hands=2
    )
    return mp_hands, mp_drawing, mp_drawing_styles, hands

def process_hand_gesture(frame, hands, model_one, model_two, labels_dict_one, labels_dict_two, 
                         mp_hands, mp_drawing, mp_drawing_styles, flip=True):
    """
    Process a single frame for hand gesture recognition
    
    Args:
        frame: BGR frame from camera
        hands: MediaPipe hands detector
        model_one: One-hand trained model
        model_two: Two-hand trained model
        labels_dict_one: Label dictionary for one-hand gestures
        labels_dict_two: Label dictionary for two-hand gestures
        mp_hands: MediaPipe hands module
        mp_drawing: MediaPipe drawing utilities
        mp_drawing_styles: MediaPipe drawing styles
        flip: Whether to flip frame horizontally (mirror mode)
    
    Returns:
        tuple: (processed_frame, predicted_character, num_hands, mode_text, prediction_id)
    """
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    if flip:
        frame_rgb = cv2.flip(frame_rgb, 1)
    
    results = hands.process(frame_rgb)
    predicted_character = None
    prediction_id = None
    num_hands = 0
    mode_text = "No hands"
    
    if results.multi_hand_landmarks:
        num_hands = len(results.multi_hand_landmarks)
        
        # ONE HAND DETECTED
        if num_hands == 1:
            mode_text = "ONE-HAND"
            hand_landmarks = results.multi_hand_landmarks[0]
            
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
            
            if len(data_aux) == 42:
                prediction = model_one.predict([np.asarray(data_aux)])
                prediction_id = int(prediction[0])
                predicted_character = labels_dict_one[prediction_id]
            
            # Draw bounding box (black for one hand)
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10
            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 0, 0), 4)
            
            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame_rgb,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Draw predicted character
            if predicted_character:
                cv2.putText(frame_rgb, predicted_character.strip(), (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
        
        # TWO HANDS DETECTED
        elif num_hands == 2:
            mode_text = "TWO-HAND"
            data_aux = []
            all_x = []
            all_y = []
            
            for hand_landmarks in results.multi_hand_landmarks:
                x_hand = []
                y_hand = []
                
                for lm in hand_landmarks.landmark:
                    x_hand.append(lm.x)
                    y_hand.append(lm.y)
                    all_x.append(lm.x)
                    all_y.append(lm.y)
                
                min_x_hand = min(x_hand)
                min_y_hand = min(y_hand)
                
                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min_x_hand)
                    data_aux.append(lm.y - min_y_hand)
                
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    frame_rgb,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
            
            if len(data_aux) == 84:
                prediction = model_two.predict([np.asarray(data_aux)])
                prediction_id = int(prediction[0])
                predicted_character = labels_dict_two[prediction_id]
            
            # Draw bounding box (green for two hands)
            x1 = int(min(all_x) * W) - 10
            y1 = int(min(all_y) * H) - 10
            x2 = int(max(all_x) * W) + 10
            y2 = int(max(all_y) * H) + 10
            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 4)
            
            # Draw predicted character
            if predicted_character:
                cv2.putText(frame_rgb, predicted_character.strip(), (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
    
    return frame_rgb, predicted_character, num_hands, mode_text, prediction_id

def update_character_tracking(predicted_character, current_character, character_start_time, 
                              previous_detected, letters_detected, hold_duration=1.5):
    """
    Update character tracking with hold duration logic
    
    Args:
        predicted_character: Currently predicted character (or None)
        current_character: Currently tracked character
        character_start_time: Time when current character was first detected
        previous_detected: Previously added character
        letters_detected: Accumulated detected letters
        hold_duration: How long to hold before adding character
    
    Returns:
        tuple: (new_current_character, new_character_start_time, new_previous_detected, new_letters_detected)
    """
    if predicted_character:
        if predicted_character != current_character:
            # New character detected
            current_character = predicted_character
            character_start_time = time.time()
        else:
            # Same character, check hold duration
            if character_start_time:
                elapsed_time = time.time() - character_start_time
                if elapsed_time >= hold_duration:
                    if current_character != previous_detected:
                        # Add character to detected letters
                        letters_detected += current_character
                        previous_detected = current_character
    else:
        # No prediction, reset tracking
        current_character = ""
        character_start_time = None
    
    return current_character, character_start_time, previous_detected, letters_detected

