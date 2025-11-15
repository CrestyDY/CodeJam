"""
Unified sign language detector that automatically switches between static and motion detection.
"""
import cv2
import numpy as np
import mediapipe as mp
import pickle
import json
import threading
import time
import os
from collections import deque

# Try importing TensorFlow for motion detection
try:
    from tensorflow import keras
    MOTION_AVAILABLE = True
except ImportError:
    MOTION_AVAILABLE = False
    print("Warning: TensorFlow not available. Motion detection disabled.")

# Try importing AI features
try:
    from src.ai.get_llm_response import get_response, check_if_sentence_complete, speak_text
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    print("Warning: AI features not available.")


class UnifiedSignDetector:
    """
    Unified detector that handles both static and motion-based sign detection.
    Automatically switches based on sign requirements.
    """
    
    def __init__(self, static_model_one_path, static_model_two_path, 
                 motion_model_path=None, motion_metadata_path=None,
                 motion_signs_config=None, sequence_length=30):
        """
        Initialize unified detector.
        
        Args:
            static_model_one_path: Path to static one-hand model
            static_model_two_path: Path to static two-hand model
            motion_model_path: Path to motion model (optional)
            motion_metadata_path: Path to motion metadata (optional)
            motion_signs_config: List of signs that require motion detection
            sequence_length: Frames to accumulate for motion detection
        """
        # Load static models
        print("Loading static models...")
        model_one_dict = pickle.load(open(static_model_one_path, 'rb'))
        model_two_dict = pickle.load(open(static_model_two_path, 'rb'))
        self.static_model_one = model_one_dict['model']
        self.static_model_two = model_two_dict['model']
        
        # Load static labels from config
        config_dir = os.path.join(os.path.dirname(__file__), '..', 'config')
        
        # One-hand static labels - load from the config file used during training
        # The model was trained on specific class indices, so we must use exact mapping
        self.static_labels_one = {}

        try:
            # Try loading the main one-hand config (used for training)
            config_path = os.path.join(config_dir, 'asl_one_hand.json')
            if not os.path.exists(config_path):
                # Fallback: try asl_words_one_hand.json
                config_path = os.path.join(config_dir, 'asl_words_one_hand.json')

            with open(config_path, 'r') as f:
                asl_config = json.load(f)
                for k, v in asl_config.items():
                    self.static_labels_one[int(k)] = v
                print(f"  Loaded {len(asl_config)} one-hand signs")
        except FileNotFoundError:
            # Try loading individual config files (letters, numbers, words)
            count = 0
            try:
                with open(os.path.join(config_dir, 'asl_letters.json'), 'r') as f:
                    asl_config = json.load(f)
                    for k, v in asl_config.items():
                        self.static_labels_one[int(k)] = v
                    count += len(asl_config)
                    print(f"  Loaded {len(asl_config)} letters")
            except FileNotFoundError:
                pass

            try:
                with open(os.path.join(config_dir, 'asl_numbers.json'), 'r') as f:
                    asl_config = json.load(f)
                    for k, v in asl_config.items():
                        self.static_labels_one[int(k)] = v
                    count += len(asl_config)
                    print(f"  Loaded {len(asl_config)} numbers")
            except FileNotFoundError:
                pass

            if count == 0:
                print(f"  Warning: No one-hand config files found")

        # Two-hand static labels
        self.static_labels_two = {}
        try:
            # Try loading the main two-hand config (used for training)
            config_path = os.path.join(config_dir, 'asl.json')
            if not os.path.exists(config_path):
                # Fallback: try asl_words_two_hands.json
                config_path = os.path.join(config_dir, 'asl_words_two_hands.json')

            with open(config_path, 'r') as f:
                asl_config = json.load(f)
                for k, v in asl_config.items():
                    self.static_labels_two[int(k)] = v
                print(f"  Loaded {len(asl_config)} two-hand signs")
        except FileNotFoundError:
            print(f"  Warning: No two-hand config files found")

        # Load motion model if available
        self.motion_enabled = False
        self.motion_model = None
        self.motion_classes = []
        
        if MOTION_AVAILABLE and motion_model_path and motion_metadata_path:
            if os.path.exists(motion_model_path) and os.path.exists(motion_metadata_path):
                print("Loading motion model...")
                self.motion_model = keras.models.load_model(motion_model_path)
                
                with open(motion_metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                
                self.motion_classes = metadata['classes']
                self.motion_feature_dim = metadata['feature_dim']
                self.motion_num_hands = 1 if self.motion_feature_dim == 42 else 2
                self.motion_enabled = True
                print(f"  Motion classes: {self.motion_classes}")
            else:
                print("Motion model files not found. Motion detection disabled.")
        
        # Motion signs configuration (which signs require motion)
        self.motion_signs = set(motion_signs_config) if motion_signs_config else set(self.motion_classes)
        
        # Sequence buffer for motion detection
        self.sequence_length = sequence_length
        self.frame_buffer = deque(maxlen=sequence_length)
        self.confidence_threshold = 0.7
        
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Use tracking mode to support both static and motion
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,  # Enable tracking
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_hands=2
        )
        
        print(f"\nUnified detector initialized:")
        print(f"  Static signs available: {len(self.static_labels_one) + len(self.static_labels_two)}")
        print(f"  Motion signs available: {len(self.motion_classes)}")
        print(f"  Total signs: {len(self.static_labels_one) + len(self.static_labels_two) + len(self.motion_classes)}")
    
    def extract_hand_features(self, hand_landmarks):
        """Extract normalized features from a single hand."""
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
        
        return data_aux, x_, y_
    
    def detect_static(self, frame):
        """
        Perform static detection on a single frame.
        
        Returns:
            (sign_name, confidence, num_hands, hand_results) or None
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        if not results.multi_hand_landmarks:
            return None
        
        num_hands = len(results.multi_hand_landmarks)
        
        # One hand detection
        if num_hands == 1:
            hand_landmarks = results.multi_hand_landmarks[0]
            features, _, _ = self.extract_hand_features(hand_landmarks)
            
            if len(features) == 42:
                pred = self.static_model_one.predict([np.asarray(features)])
                pred_idx = int(pred[0])
                
                if pred_idx in self.static_labels_one:
                    sign_name = self.static_labels_one[pred_idx]
                    # Get confidence (for RandomForest, use probability)
                    proba = self.static_model_one.predict_proba([np.asarray(features)])[0]
                    confidence = proba[pred_idx]
                    return (sign_name, confidence, num_hands, results)
        
        # Two hands detection
        elif num_hands == 2:
            features = []
            for hand_landmarks in results.multi_hand_landmarks:
                hand_features, _, _ = self.extract_hand_features(hand_landmarks)
                features.extend(hand_features)
            
            if len(features) == 84:
                pred = self.static_model_two.predict([np.asarray(features)])
                pred_idx = int(pred[0])
                
                if pred_idx in self.static_labels_two:
                    sign_name = self.static_labels_two[pred_idx]
                    proba = self.static_model_two.predict_proba([np.asarray(features)])[0]
                    confidence = proba[pred_idx]
                    return (sign_name, confidence, num_hands, results)
        
        return None
    
    def add_frame_for_motion(self, frame):
        """
        Add frame to motion buffer and check if ready for prediction.
        
        Returns:
            (sign_name, confidence, hand_results) or None
        """
        if not self.motion_enabled:
            return None
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        if not results.multi_hand_landmarks:
            return None
        
        num_hands = len(results.multi_hand_landmarks)
        
        # Check if correct number of hands
        if num_hands != self.motion_num_hands:
            return None
        
        # Extract features
        if self.motion_num_hands == 1:
            features, _, _ = self.extract_hand_features(results.multi_hand_landmarks[0])
        else:
            features = []
            for hand_landmarks in results.multi_hand_landmarks:
                hand_features, _, _ = self.extract_hand_features(hand_landmarks)
                features.extend(hand_features)
        
        if len(features) != self.motion_feature_dim:
            return None
        
        # Add to buffer
        self.frame_buffer.append(features)
        
        # Check if buffer is full
        if len(self.frame_buffer) >= self.sequence_length:
            sequence = np.array([list(self.frame_buffer)])
            predictions = self.motion_model.predict(sequence, verbose=0)[0]
            
            predicted_idx = np.argmax(predictions)
            confidence = predictions[predicted_idx]
            
            if confidence >= self.confidence_threshold:
                sign_name = self.motion_classes[predicted_idx]
                return (sign_name, confidence, results)
        
        return None
    
    def process_frame(self, frame):
        """
        Process a single frame with unified detection.
        Tries static detection first, accumulates for motion detection.
        
        Returns:
            dict with detection results and metadata
        """
        result = {
            'sign': None,
            'confidence': None,
            'type': None,  # 'static' or 'motion'
            'num_hands': 0,
            'hand_results': None,
            'buffer_fill': len(self.frame_buffer),
            'buffer_max': self.sequence_length
        }
        
        # Try static detection first
        static_result = self.detect_static(frame)
        
        if static_result:
            sign_name, confidence, num_hands, hand_results = static_result
            
            # Check if this sign requires motion detection
            if sign_name.strip() not in self.motion_signs:
                result['sign'] = sign_name
                result['confidence'] = confidence
                result['type'] = 'static'
                result['num_hands'] = num_hands
                result['hand_results'] = hand_results
                self.frame_buffer.clear()  # Clear motion buffer
                return result
        
        # Try motion detection (always accumulate frames)
        if self.motion_enabled:
            motion_result = self.add_frame_for_motion(frame)
            
            if motion_result:
                sign_name, confidence, hand_results = motion_result
                result['sign'] = sign_name
                result['confidence'] = confidence
                result['type'] = 'motion'
                result['num_hands'] = self.motion_num_hands
                result['hand_results'] = hand_results
                self.frame_buffer.clear()  # Clear buffer after detection
                return result
            else:
                # Update hand results even if no detection
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hand_results = self.hands.process(frame_rgb)
                if hand_results and hand_results.multi_hand_landmarks:
                    result['hand_results'] = hand_results
                    result['num_hands'] = len(hand_results.multi_hand_landmarks)
        
        return result
    
    def reset_buffer(self):
        """Clear the motion detection buffer."""
        self.frame_buffer.clear()
    
    def get_buffer_status(self):
        """Get motion buffer status."""
        return len(self.frame_buffer), self.sequence_length


def run_unified_detection(static_one_path, static_two_path, 
                         motion_model_path=None, motion_metadata_path=None,
                         camera_index=0, enable_ai=True):
    """
    Run unified sign detection with automatic mode switching.
    """
    # Initialize detector
    detector = UnifiedSignDetector(
        static_one_path, static_two_path,
        motion_model_path, motion_metadata_path
    )
    
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Detection state
    detected_signs = []
    last_prediction = None
    last_prediction_time = 0
    prediction_cooldown = 1.5
    current_character = ""
    character_start_time = None
    HOLD_DURATION = 1.5
    
    # AI state
    sentence_completion_status = {}
    last_checked_input = ""
    last_spoken_input = ""
    status_lock = threading.Lock()
    sentence_options = []
    selection_mode = False
    
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    
    print("\n=== Unified Sign Language Detection ===")
    print("Automatically switches between static and motion detection")
    print("Press 'q' to quit")
    print("Press 'r' to reset motion buffer")
    print("Press 'c' to clear detected signs")
    if enable_ai and AI_AVAILABLE:
        print("✨ AI features enabled")
    print("=" * 60)
    
    def run_llm_in_background(user_input):
        """Run AI processing in background thread."""
        nonlocal sentence_completion_status, last_checked_input, last_spoken_input
        nonlocal detected_signs, sentence_options, selection_mode
        
        def thread_target():
            nonlocal sentence_completion_status, last_checked_input, last_spoken_input
            nonlocal detected_signs, sentence_options, selection_mode
            
            try:
                response = get_response(user_input)
                if response:
                    print(f"\n{'='*60}")
                    print(f"📝 LLM Response: {response}")
                    print(f"{'='*60}\n")
                    
                    try:
                        response_data = json.loads(response)
                        if 'sentences' in response_data:
                            sentence_options = response_data['sentences'][:3]
                            selection_mode = True
                            print(f"✨ {len(sentence_options)} sentence options available")
                    except json.JSONDecodeError:
                        pass
                
                completion_response = check_if_sentence_complete(user_input)
                try:
                    completion_data = json.loads(completion_response)
                    
                    with status_lock:
                        sentence_completion_status = completion_data
                        last_checked_input = user_input
                    
                    print(f"🔍 Sentence: {completion_data.get('is_complete', False)} - {completion_data.get('reason', '')}")
                    
                    if completion_data.get('is_complete', False):
                        if user_input != last_spoken_input:
                            threading.Thread(target=lambda: speak_text(user_input), daemon=True).start()
                            last_spoken_input = user_input
                        detected_signs.clear()
                        print("✨ Sentence complete! Buffer cleared.")
                
                except json.JSONDecodeError:
                    with status_lock:
                        sentence_completion_status = {"is_complete": False, "reason": "Parse error"}
            
            except Exception as e:
                print(f"❌ AI Error: {e}")
        
        threading.Thread(target=thread_target, daemon=True).start()
    
    # Main loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        current_time = time.time()
        
        # Process frame with unified detector
        result = detector.process_frame(frame)
        
        # Draw hand landmarks and bounding boxes
        if result['hand_results'] and result['hand_results'].multi_hand_landmarks:
            # Calculate bounding box for visualization
            all_x, all_y = [], []
            for hand_landmarks in result['hand_results'].multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Collect coordinates for bounding box
                for lm in hand_landmarks.landmark:
                    all_x.append(lm.x)
                    all_y.append(lm.y)

            # Draw bounding box around detected hands
            if all_x and all_y:
                H, W, _ = frame.shape
                x1 = int(min(all_x) * W) - 10
                y1 = int(min(all_y) * H) - 10
                x2 = int(max(all_x) * W) + 10
                y2 = int(max(all_y) * H) + 10

                # Color based on detection type
                box_color = (0, 0, 0) if result['num_hands'] == 1 else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 4)

                # Show predicted sign near bounding box
                if result['sign']:
                    cv2.putText(frame, result['sign'].strip(), (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.3, box_color, 3, cv2.LINE_AA)

                # Show hold progress bar for static detection
                if result['type'] == 'static' and character_start_time is not None and current_character:
                    elapsed_time = current_time - character_start_time
                    progress = min(elapsed_time / HOLD_DURATION, 1.0)
                    bar_width = int((x2 - x1) * progress)

                    # Progress bar
                    cv2.rectangle(frame, (x1, y2 + 5), (x1 + bar_width, y2 + 15), (0, 255, 0), -1)
                    # Progress bar border
                    cv2.rectangle(frame, (x1, y2 + 5), (x2, y2 + 15), (0, 0, 0), 2)

                    # Show hold time
                    cv2.putText(frame, f"Hold: {elapsed_time:.1f}s / {HOLD_DURATION}s",
                               (x1, y2 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Handle detection
        if result['sign']:
            sign_name = result['sign'].strip()
            confidence = result['confidence']
            detection_type = result['type']
            
            # For static signs, use hold duration logic
            if detection_type == 'static':
                if sign_name != current_character:
                    current_character = sign_name
                    character_start_time = current_time
                elif character_start_time and current_time - character_start_time >= HOLD_DURATION:
                    if sign_name != last_prediction or current_time - last_prediction_time > prediction_cooldown:
                        detected_signs.append(sign_name)
                        last_prediction = sign_name
                        last_prediction_time = current_time
                        print(f"✓ Static: {sign_name} (conf: {confidence:.2f})")
                        
                        if enable_ai and AI_AVAILABLE:
                            current_sentence = " ".join([s.strip() for s in detected_signs])
                            if current_sentence != last_checked_input:
                                run_llm_in_background(current_sentence)
            
            # For motion signs, immediate detection
            elif detection_type == 'motion':
                if sign_name != last_prediction or current_time - last_prediction_time > prediction_cooldown:
                    detected_signs.append(sign_name)
                    last_prediction = sign_name
                    last_prediction_time = current_time
                    current_character = ""  # Reset static tracking
                    character_start_time = None
                    print(f"✓ Motion: {sign_name} (conf: {confidence:.2f})")
                    
                    if enable_ai and AI_AVAILABLE:
                        current_sentence = " ".join([s.strip() for s in detected_signs])
                        if current_sentence != last_checked_input:
                            run_llm_in_background(current_sentence)
        else:
            # Reset static tracking if no detection
            if result['type'] != 'motion':
                current_character = ""
                character_start_time = None
        
        # Display info
        buffer_fill, buffer_max = result['buffer_fill'], result['buffer_max']
        buffer_percent = (buffer_fill / buffer_max) * 100
        
        # PROMINENT: Detected text at top (like static inference)
        signs_text = " ".join([s.strip() for s in detected_signs])
        cv2.putText(frame, f"Detected: {signs_text}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

        # Mode indicator
        mode_color = (255, 255, 0) if result['type'] == 'static' else (255, 0, 255) if result['type'] == 'motion' else (200, 200, 200)
        mode_text = f"Mode: {result['type'].upper() if result['type'] else 'DETECTING'}"
        cv2.putText(frame, mode_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2, cv2.LINE_AA)

        # Buffer status (only show when accumulating for motion)
        if buffer_fill > 0:
            cv2.putText(frame, f"Motion Buffer: {buffer_fill}/{buffer_max} ({buffer_percent:.0f}%)",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # AI status
        y_pos = 120 if buffer_fill == 0 else 150
        if enable_ai and AI_AVAILABLE:
            with status_lock:
                is_complete = sentence_completion_status.get('is_complete', False)
            status_text = "Complete" if is_complete else "Incomplete"
            status_color = (0, 255, 0) if is_complete else (0, 165, 255)
            cv2.putText(frame, f"Sentence: {status_text}",
                        (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

            if selection_mode and sentence_options:
                y_offset = y_pos + 30
                cv2.putText(frame, "AI Suggestions:",
                            (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                for i, sentence in enumerate(sentence_options[:3], 1):
                    y_offset += 20
                    sentence_short = sentence[:60] + "..." if len(sentence) > 60 else sentence
                    cv2.putText(frame, f"{i}. {sentence_short}",
                                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Motion buffer progress bar at bottom (only when buffer is accumulating)
        if buffer_fill > 0:
            H, W, _ = frame.shape
            bar_y = H - 40  # 40 pixels from bottom
            bar_width = 400
            bar_height = 20
            bar_x = (W - bar_width) // 2  # Center horizontally

            # Background
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
            # Fill
            filled = int((buffer_fill / buffer_max) * bar_width)
            if filled > 0:
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_height), (0, 255, 0), -1)
            # Label
            cv2.putText(frame, f"Motion Detection: {buffer_percent:.0f}%",
                       (bar_x, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow('Unified Sign Detection', frame)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            detector.reset_buffer()
            print("Motion buffer reset")
        elif key == ord('c'):
            detected_signs.clear()
            print("Signs cleared")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n=== Session Summary ===")
    print(f"Total signs detected: {len(detected_signs)}")
    print(f"Signs: {' '.join([s.strip() for s in detected_signs])}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified sign detection with auto-switching")
    parser.add_argument("--static-one", type=str, default="../models/model_one_hand.p")
    parser.add_argument("--static-two", type=str, default="../models/model_two_hands.p")
    parser.add_argument("--motion-model", type=str, default="../models/motion_model.h5")
    parser.add_argument("--motion-metadata", type=str, default="../models/motion_model_metadata.pkl")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--disable-ai", action="store_true")
    
    args = parser.parse_args()
    
    enable_ai = not args.disable_ai and AI_AVAILABLE
    
    run_unified_detection(
        args.static_one, args.static_two,
        args.motion_model if os.path.exists(args.motion_model) else None,
        args.motion_metadata if os.path.exists(args.motion_metadata) else None,
        args.camera, enable_ai
    )

