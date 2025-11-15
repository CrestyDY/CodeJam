import pickle
import json
import os
import time
import cv2
import mediapipe as mp
import numpy as np
import threading
from src.ai.get_llm_response import get_response, check_if_sentence_complete, speak_text


def run_sign_language_classifier(config_file='asl.json', config_file_one_hand='asl_one_hand.json'):
    """
    Run the sign language to text classifier with auto-switching between one-hand and two-hand models
    
    Args:
        config_file (str): Name of the JSON config file containing two-hand gesture mappings (default: 'asl.json')
        config_file_one_hand (str): Name of the JSON config file containing one-hand gesture mappings (default: 'asl_one_hand.json')
    """
    # Load both trained models
    script_dir = os.path.dirname(__file__)
    model_one_dict = pickle.load(open(os.path.join(script_dir, 'models', 'model_one_hand.p'), 'rb'))
    model_two_dict = pickle.load(open(os.path.join(script_dir, 'models', 'model_two_hands.p'), 'rb'))
    
    model_one = model_one_dict['model']  # One-hand model (42 features)
    model_two = model_two_dict['model']  # Two-hand model (84 features)
    
    # Setup camera with higher resolution
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Setup MediaPipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, max_num_hands=2)
    
    # State variables
    letters_detected = ""
    previous_detected = ""
    current_character = ""
    character_start_time = None
    HOLD_DURATION = 1.5  # seconds - adjust this value to change how long to hold
    
    # Sentence selection state
    sentence_options = []  # List of 3 sentences from LLM
    selected_sentence = ""  # The sentence chosen by the user
    selection_mode = False  # Whether we're in sentence selection mode
    
    # Sentence completion tracking
    sentence_completion_status = {}  # Initialize as empty dict
    last_checked_input = ""  # Track what we last checked
    last_spoken_input = ""  # Track what we last spoke to avoid repeating
    status_lock = threading.Lock()  # Thread lock for safe access
    
    # Load ASL word mappings from config files
    config_dir = os.path.join(os.path.dirname(__file__), '..', 'config')
    
    # Load two-hand gestures
    CONFIG_PATH_TWO = os.path.join(config_dir, config_file)
    with open(CONFIG_PATH_TWO, 'r') as f:
        asl_config_two = json.load(f)
        # Convert string keys to integers and add space after each word
        labels_dict_two = {int(k): v + ' ' for k, v in asl_config_two.items()}
        print(f"Loaded {len(labels_dict_two)} two-hand sign language gestures from {config_file}")
    
    CONFIG_PATH_ONE = os.path.join(config_dir, config_file_one_hand)
    try:
        with open(CONFIG_PATH_ONE, 'r') as f:
            asl_config_one = json.load(f)
            labels_dict_one = {int(k): v + ' ' for k, v in asl_config_one.items()}
            print(f"Loaded {len(labels_dict_one)} one-hand sign language gestures from {config_file_one_hand}")
    except FileNotFoundError:
        print(f"Warning: {config_file_one_hand} not found, using default one-hand labels")
    
    # Find indices for selection gestures dynamically (check both dictionaries)
    select_indices = {}
    for key, value in labels_dict_two.items():
        if value.strip() == "Select 1":
            select_indices[1] = key
        elif value.strip() == "Select 2":
            select_indices[2] = key
        elif value.strip() == "Select 3":
            select_indices[3] = key
    print(f"Selection gestures mapped: {select_indices}")
    
    def run_llm_in_background(user_input):
        """Run the LLM call in a separate thread to avoid blocking the video feed"""
        nonlocal sentence_completion_status, last_checked_input, last_spoken_input, letters_detected
        nonlocal sentence_options, selection_mode, selected_sentence
        
        def thread_target():
            nonlocal sentence_completion_status, last_checked_input, last_spoken_input, letters_detected
            nonlocal sentence_options, selection_mode, selected_sentence
            
            try:
                # ALWAYS get response interpretations for every word added
                response = get_response(user_input)
                if response:
                    print(f"\n{'='*60}")
                    print(f"📝 LLM Response for '{user_input}':")
                    print(f"{'='*60}")
                    print(response)
                    print(f"{'='*60}\n")
                    
                    # Parse JSON response for sentence options
                    try:
                        response_data = json.loads(response)
                        if 'sentences' in response_data and isinstance(response_data['sentences'], list):
                            sentence_options = response_data['sentences'][:3]  # Take up to 3 sentences
                            selection_mode = True  # Enable selection mode
                            selected_sentence = ""  # Reset selected sentence
                            print(f"\n✨ {len(sentence_options)} sentence options available!")
                            print("   - Use 'Select 1/2/3' gestures to choose a sentence")
                            print("   - OR add more words to make the sentence more complex\n")
                    except json.JSONDecodeError as je:
                        print(f"Warning: Could not parse sentence options from JSON response: {je}")
                        sentence_options = []
                        selection_mode = False
                
                # Also check if sentence is complete
                completion_response = check_if_sentence_complete(user_input)
                
                # Parse the completion JSON response
                try:
                    completion_data = json.loads(completion_response)
                    
                    # Thread-safe update
                    with status_lock:
                        sentence_completion_status = completion_data
                        last_checked_input = user_input
                    
                    print(f"\n{'='*60}")
                    print(f"🔍 Sentence Completion Check for '{user_input}':")
                    print(f"   Complete: {completion_data.get('is_complete', 'unknown')}")
                    print(f"   Reason: {completion_data.get('reason', 'no reason provided')}")
                    print(f"{'='*60}")
                    
                    # If sentence is complete, speak it and clear buffer
                    if completion_data.get('is_complete', False):
                        # Speak the text only if we haven't spoken this exact input before
                        if user_input != last_spoken_input:
                            # Speak in a separate thread to avoid any blocking
                            def speak_completion():
                                speak_text(user_input)
                            threading.Thread(target=speak_completion, daemon=True).start()
                            last_spoken_input = user_input
                        
                        # Clear the buffer to allow starting a new sentence
                        print("✨ Sentence complete! Clearing buffer for new sentence...")
                        letters_detected = ""
                
                except json.JSONDecodeError:
                    print(f"\n⚠️ Could not parse completion response as JSON: {completion_response}\n")
                    with status_lock:
                        sentence_completion_status = {"is_complete": False, "reason": "Parse error"}
                
            except Exception as e:
                print(f"\n❌ Error getting LLM response: {e}\n")
        
        thread = threading.Thread(target=thread_target, daemon=True)
        thread.start()
    
    # Setup OpenCV window
    window_name = 'Sign Language to Text Classifier'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    
    # Main loop
    while True:
        ret, frame = cap.read()

        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        
        predicted_character = None
        x1 = y1 = x2 = y2 = None
        mode_text = ""
        num_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
        
        # Process based on number of hands detected
        if results.multi_hand_landmarks and num_hands in [1, 2]:
            
            # ============ ONE HAND DETECTED ============
            if num_hands == 1:
                mode_text = "Mode: ONE-HAND"
                hand_landmarks = results.multi_hand_landmarks[0]
                
                data_aux = []
                x_ = []
                y_ = []
                
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                
                # Collect coordinates
                for lm in hand_landmarks.landmark:
                    x = lm.x
                    y = lm.y
                    x_.append(x)
                    y_.append(y)
                
                min_x = min(x_)
                min_y = min(y_)
                
                # Normalize coordinates
                for lm in hand_landmarks.landmark:
                    x = lm.x
                    y = lm.y
                    data_aux.append(x - min_x)
                    data_aux.append(y - min_y)
                
                # Bounding box
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) + 10
                y2 = int(max(y_) * H) + 10
                
                # Make prediction with one-hand model (42 features)
                if len(data_aux) == 42:
                    prediction = model_one.predict([np.asarray(data_aux)])
                    predicted_character = labels_dict_one[int(prediction[0])]
                else:
                    cv2.putText(frame, f"Error: {len(data_aux)} features (need 42)", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            
            # ============ TWO HANDS DETECTED ============
            elif num_hands == 2:
                mode_text = "Mode: TWO-HAND"
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
                    prediction = model_two.predict([np.asarray(data_aux)])
                    predicted_character = labels_dict_two[int(prediction[0])]
                else:
                    cv2.putText(frame, f"Error: {len(data_aux)} features (need 84)", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            
            # ============ CHARACTER TRACKING LOGIC (for both 1 and 2 hands) ============
            if predicted_character is not None:
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
                            previous_detected = current_character
                            
                            # Check if we're in selection mode
                            if selection_mode and sentence_options:
                                # Check if it's a selection gesture
                                # IMPORTANT: Only check for selection gestures in TWO-HAND mode
                                # (selection gestures are defined in two-hand config)
                                prediction_id = int(prediction[0])
                                
                                # Check if this is a selection gesture
                                is_selection = False
                                
                                # Only check selection if we're in TWO-HAND mode
                                if num_hands == 2:
                                    for select_num, select_id in select_indices.items():
                                        if prediction_id == select_id:
                                            selection_index = select_num - 1  # 1->0, 2->1, 3->2
                                            if selection_index < len(sentence_options):
                                                selected_sentence = sentence_options[selection_index]
                                                print(f"\n{'='*60}")
                                                print(f"✅ Selected option {select_num}: {selected_sentence}")
                                                print(f"{'='*60}\n")
                                                
                                                # Speak the selected sentence in background thread
                                                def speak_in_background():
                                                    speak_text(selected_sentence)
                                                threading.Thread(target=speak_in_background, daemon=True).start()
                                                
                                                # Exit selection mode and reset detected letters
                                                selection_mode = False
                                                sentence_options = []
                                                letters_detected = ""  # Reset for next gesture sequence
                                                print("📝 Detected letters reset. Ready for new gesture sequence.\n")
                                                is_selection = True
                                                break
                                
                                # If not a selection gesture, treat as normal word in selection mode
                                if not is_selection:
                                    letters_detected += current_character
                                    print(f"Added '{current_character}' to detected letters: {letters_detected}")
                                    # Exit selection mode since user is adding more words
                                    selection_mode = False
                                    sentence_options = []
                                    # Run LLM call with new accumulated text
                                    run_llm_in_background(letters_detected)
                            else:
                                # Normal mode: add character to detected letters
                                letters_detected += current_character
                                print(f"Added '{current_character}' to detected letters: {letters_detected}")
                                # Run the LLM call in background thread so video doesn't freeze
                                run_llm_in_background(letters_detected)
                
                # Display the predicted character with bounding box
                if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
                    # Use different colors for one-hand vs two-hand
                    color = (0, 0, 0) if mode_text == "Mode: ONE-HAND" else (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
                    cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3,
                                cv2.LINE_AA)
                    
                    # Show hold progress bar
                    if character_start_time is not None and current_character != previous_detected:
                        elapsed_time = time.time() - character_start_time
                        progress = min(elapsed_time / HOLD_DURATION, 1.0)
                        bar_width = int((x2 - x1) * progress)
                        cv2.rectangle(frame, (x1, y2 + 5), (x1 + bar_width, y2 + 15), (0, 255, 0), -1)
                        cv2.rectangle(frame, (x1, y2 + 5), (x2, y2 + 15), (0, 0, 0), 2)
            else:
                # No valid prediction, reset tracking
                current_character = ""
                character_start_time = None
        
        else:
            # No hands or invalid number of hands detected
            # Draw detected hands if any
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
            
            # Show status message
            if num_hands == 0:
                status_text = "No hands detected"
            else:
                status_text = f"Unsupported: {num_hands} hands (need 1 or 2)"
            cv2.putText(frame, status_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
            
            # Reset tracking
            current_character = ""
            character_start_time = None

        # Display mode (one-hand vs two-hand)
        if mode_text:
            cv2.putText(frame, mode_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
        
        # Display detected letters on screen
        cv2.putText(frame, f"Detected: {letters_detected}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                    cv2.LINE_AA)
        
        # Display sentence completion status with thread-safe access
        with status_lock:
            if sentence_completion_status:  # Check if dict is not empty
                completion_text = "Complete ✓" if sentence_completion_status.get('is_complete', False) else "Incomplete..."
                reason = sentence_completion_status.get('reason', 'N/A')
                
                # Use different colors based on completion
                color = (0, 255, 0) if sentence_completion_status.get('is_complete', False) else (255, 255, 0)
                
                cv2.putText(frame, f"Status: {completion_text}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
                            cv2.LINE_AA)
                
                # Truncate reason if too long
                if len(reason) > 50:
                    reason = reason[:47] + "..."
                cv2.putText(frame, f"{reason}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                            cv2.LINE_AA)
        
        # Display sentence options if in selection mode
        if selection_mode and sentence_options:
            # Add a semi-transparent overlay for better readability
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, H - 250), (W - 10, H - 10), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Display title
            cv2.putText(frame, "SELECT A SENTENCE (or add more words):", (20, H - 220), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            
            # Display each sentence option
            for idx, sentence in enumerate(sentence_options, 1):
                y_pos = H - 220 + (idx * 60)
                # Truncate sentence if too long
                display_text = sentence if len(sentence) <= 60 else sentence[:57] + "..."
                cv2.putText(frame, f"[{idx}] {display_text}", (20, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Display selected sentence if available
        if selected_sentence:
            cv2.putText(frame, f"Selected: {selected_sentence[:50]}", (10, H - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow(window_name, frame)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()


# Main entry point
if __name__ == "__main__":
    run_sign_language_classifier()
