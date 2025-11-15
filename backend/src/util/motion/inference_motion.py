"""
Real-time inference for motion-based sign language recognition.
"""
import cv2
import numpy as np
import mediapipe as mp
import pickle
import json
import threading
import time
from collections import deque
from tensorflow import keras
from src.ai.get_llm_response import get_response, check_if_sentence_complete, speak_text



class MotionSignDetector:
    """
    Real-time motion-based sign language detector.
    Accumulates frames and runs inference when enough frames are collected.
    """
    
    def __init__(self, model_path, metadata_path, sequence_length=30, confidence_threshold=0.7):
        """
        Initialize motion detector.
        
        Args:
            model_path: Path to trained Keras model (.h5)
            metadata_path: Path to metadata pickle file
            sequence_length: Number of frames to accumulate
            confidence_threshold: Minimum confidence for prediction
        """
        # Load model
        self.model = keras.models.load_model(model_path)
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.label_encoder = metadata['label_encoder']
        self.classes = metadata['classes']
        self.sequence_length = sequence_length
        self.feature_dim = metadata['feature_dim']
        self.confidence_threshold = confidence_threshold
        
        # Determine number of hands from feature dimension
        self.num_hands = 1 if self.feature_dim == 42 else 2
        
        # Frame buffer
        self.frame_buffer = deque(maxlen=sequence_length)
        
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_hands=2
        )
        
        print(f"Motion detector initialized:")
        print(f"  Model: {model_path}")
        print(f"  Classes: {self.classes}")
        print(f"  Sequence length: {self.sequence_length}")
        print(f"  Expected hands: {self.num_hands}")
        print(f"  Confidence threshold: {self.confidence_threshold}")
    
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
        
        return data_aux
    
    def process_frame(self, frame):
        """
        Process a single frame and return features if hands detected.
        
        Returns:
            features: List of features or None if hands not detected properly
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        if not results.multi_hand_landmarks:
            return None
        
        # Check if we have the expected number of hands
        if len(results.multi_hand_landmarks) != self.num_hands:
            return None
        
        # Extract features
        if self.num_hands == 1:
            features = self.extract_hand_features(results.multi_hand_landmarks[0])
        else:  # num_hands == 2
            features = []
            for hand_landmarks in results.multi_hand_landmarks:
                features.extend(self.extract_hand_features(hand_landmarks))
        
        if len(features) != self.feature_dim:
            return None
        
        return features, results
    
    def add_frame(self, frame):
        """
        Add a frame to the buffer and return prediction if buffer is full.
        
        Returns:
            prediction: (class_name, confidence) or None
        """
        result = self.process_frame(frame)
        
        if result is None:
            return None, None
        
        features, hand_results = result
        self.frame_buffer.append(features)
        
        # Only predict when buffer is full
        if len(self.frame_buffer) < self.sequence_length:
            return None, hand_results
        
        # Make prediction
        sequence = np.array([list(self.frame_buffer)])  # Shape: (1, seq_len, features)
        predictions = self.model.predict(sequence, verbose=0)[0]
        
        predicted_class_idx = np.argmax(predictions)
        confidence = predictions[predicted_class_idx]
        
        if confidence >= self.confidence_threshold:
            predicted_class = self.classes[predicted_class_idx]
            return (predicted_class, confidence), hand_results
        
        return None, hand_results
    
    def reset_buffer(self):
        """Clear the frame buffer."""
        self.frame_buffer.clear()
    
    def get_buffer_status(self):
        """Get current buffer fill status."""
        return len(self.frame_buffer), self.sequence_length


def run_motion_detection(model_path, metadata_path, camera_index=0, enable_ai=True):
    """
    Run real-time motion detection from webcam.
    
    Args:
        model_path: Path to trained model
        metadata_path: Path to metadata
        camera_index: Camera index to use
        enable_ai: Enable AI sentence completion and TTS features
    """
    detector = MotionSignDetector(model_path, metadata_path)
    
    cap = cv2.VideoCapture(camera_index)
    
    detected_signs = []
    last_prediction = None
    last_prediction_time = 0
    prediction_cooldown = 2.0  # seconds
    
    # AI and sentence completion state
    sentence_completion_status = {}
    last_checked_input = ""
    last_spoken_input = ""
    status_lock = threading.Lock()
    sentence_options = []
    selection_mode = False
    selected_sentence = ""

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    print("\n=== Motion Sign Detection ===")
    print("Press 'q' to quit")
    print("Press 'r' to reset buffer")
    print("Press 'c' to clear detected signs")
    print("=" * 50)
    
    def run_llm_in_background(user_input):
        """Run the LLM call in a separate thread to avoid blocking the video feed"""
        nonlocal sentence_completion_status, last_checked_input, last_spoken_input
        nonlocal detected_signs, sentence_options, selection_mode, selected_sentence

        def thread_target():
            nonlocal sentence_completion_status, last_checked_input, last_spoken_input
            nonlocal detected_signs, sentence_options, selection_mode, selected_sentence

            try:
                # Get response interpretations
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
                            sentence_options = response_data['sentences'][:3]
                            selection_mode = True
                            selected_sentence = ""
                            print(f"\n✨ {len(sentence_options)} sentence options available!")
                            print("   - Add more signs to refine the sentence\n")
                    except json.JSONDecodeError as je:
                        print(f"Warning: Could not parse sentence options: {je}")
                        sentence_options = []
                        selection_mode = False

                # Check if sentence is complete
                completion_response = check_if_sentence_complete(user_input)

                try:
                    completion_data = json.loads(completion_response)

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
                        if user_input != last_spoken_input:
                            def speak_completion():
                                speak_text(user_input)
                            threading.Thread(target=speak_completion, daemon=True).start()
                            last_spoken_input = user_input

                        print("✨ Sentence complete! Clearing buffer for new sentence...")
                        detected_signs.clear()

                except json.JSONDecodeError:
                    print(f"\n⚠️ Could not parse completion response as JSON\n")
                    with status_lock:
                        sentence_completion_status = {"is_complete": False, "reason": "Parse error"}

            except Exception as e:
                print(f"\n❌ Error getting LLM response: {e}\n")

        thread = threading.Thread(target=thread_target, daemon=True)
        thread.start()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        current_time = time.time()
        
        # Process frame
        prediction, hand_results = detector.add_frame(frame)
        
        # Draw hand landmarks if detected
        if hand_results and hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
        
        # Handle prediction
        if prediction is not None:
            sign_name, confidence = prediction
            
            # Check cooldown to avoid duplicate detections
            if (last_prediction != sign_name or 
                current_time - last_prediction_time > prediction_cooldown):
                detected_signs.append(sign_name)
                last_prediction = sign_name
                last_prediction_time = current_time
                print(f"✓ Detected: {sign_name} (confidence: {confidence:.2f})")
                detector.reset_buffer()  # Reset after successful detection

                if detected_signs:
                    current_sentence = " ".join(detected_signs)
                    # Only call LLM if the sentence changed significantly
                    if current_sentence != last_checked_input:
                        run_llm_in_background(current_sentence)

        # Display info
        buffer_fill, buffer_max = detector.get_buffer_status()
        buffer_percent = (buffer_fill / buffer_max) * 100
        
        # Status text
        cv2.putText(frame, f"Buffer: {buffer_fill}/{buffer_max} ({buffer_percent:.0f}%)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if last_prediction:
            cv2.putText(frame, f"Last: {last_prediction}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Detected signs
        signs_text = " ".join(detected_signs[-10:])  # Show last 10
        cv2.putText(frame, f"Signs: {signs_text}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        

        with status_lock:
            is_complete = sentence_completion_status.get('is_complete', False)

        status_text = "Complete" if is_complete else "Incomplete"
        status_color = (0, 255, 0) if is_complete else (0, 165, 255)
        cv2.putText(frame, f"Sentence: {status_text}",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        # Show sentence options if available
        if selection_mode and sentence_options:
            y_offset = 150
            cv2.putText(frame, "AI Suggestions:",
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            for i, sentence in enumerate(sentence_options[:3], 1):
                y_offset += 20
                sentence_short = sentence[:60] + "..." if len(sentence) > 60 else sentence
                cv2.putText(frame, f"{i}. {sentence_short}",
                            (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Buffer progress bar
        bar_width = 300
        bar_height = 20
        bar_x, bar_y = 10, 250 if sentence_options else 110
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                      (255, 255, 255), 2)
        filled_width = int((buffer_fill / buffer_max) * bar_width)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height),
                      (0, 255, 0), -1)
        
        cv2.imshow('Motion Sign Detection', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r'):
            detector.reset_buffer()
            print("Buffer reset")
        elif key == ord('c'):
            detected_signs.clear()
            print("Detected signs cleared")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n=== Session Summary ===")
    print(f"Total signs detected: {len(detected_signs)}")
    print(f"Signs: {' '.join(detected_signs)}")


if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Run motion-based sign detection")
    parser.add_argument(
        "--model",
        type=str,
        default="models/motion_model.h5",
        help="Path to trained model"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="models/motion_model_metadata.pkl",
        help="Path to metadata file"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index"
    )
    parser.add_argument(
        "--enable-ai",
        action="store_true",
        default=True,
        help="Enable AI sentence completion and TTS (default: True)"
    )
    parser.add_argument(
        "--disable-ai",
        action="store_true",
        help="Disable AI features"
    )

    args = parser.parse_args()

    # Determine AI enabled state
    enable_ai = args.enable_ai and not args.disable_ai

    # Check if files exist
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        exit(1)
    
    if not os.path.exists(args.metadata):
        print(f"Error: Metadata file not found: {args.metadata}")
        exit(1)
    


    run_motion_detection(args.model, args.metadata, args.camera, enable_ai=enable_ai)

