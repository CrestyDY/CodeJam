"""
Motion-based sign language dataset builder.
Captures sequences of frames to detect signs that involve motion (e.g., "thank you", "hello", "sorry").
"""
import os
import pickle
import cv2
import mediapipe as mp
import numpy as np
import argparse
from collections import deque

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, '..', 'data')  # Shared data directory

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,  # Enable tracking for motion
    min_detection_confidence=0.3,
    min_tracking_confidence=0.5,
    max_num_hands=2
)

# Configuration
SEQUENCE_LENGTH = 30  # Number of frames to capture per sign (at ~30 fps = 1 second)
FEATURE_DIM_ONE_HAND = 42  # 21 landmarks × 2 (x, y)
FEATURE_DIM_TWO_HANDS = 84  # 42 × 2 hands


def extract_hand_features(hand_landmarks, normalize=True):
    """
    Extract normalized features from a single hand.
    Returns: list of 42 features (21 landmarks × 2 coords)
    """
    data_aux = []
    x_ = []
    y_ = []
    
    for lm in hand_landmarks.landmark:
        x_.append(lm.x)
        y_.append(lm.y)
    
    if normalize:
        min_x = min(x_)
        min_y = min(y_)
        
        for lm in hand_landmarks.landmark:
            data_aux.append(lm.x - min_x)
            data_aux.append(lm.y - min_y)
    else:
        for lm in hand_landmarks.landmark:
            data_aux.append(lm.x)
            data_aux.append(lm.y)
    
    return data_aux


def calculate_motion_features(sequence):
    """
    Calculate motion features from a sequence of frames.
    
    Args:
        sequence: List of feature arrays [frame1, frame2, ..., frameN]
    
    Returns:
        Dictionary containing:
        - velocity: frame-to-frame changes
        - acceleration: second-order changes
        - trajectory: overall displacement
        - speed: magnitude of velocity
    """
    sequence = np.array(sequence)  # shape: (num_frames, num_features)
    
    if len(sequence) < 2:
        return None
    
    # Calculate velocity (first derivative)
    velocity = np.diff(sequence, axis=0)
    
    # Calculate acceleration (second derivative)
    acceleration = np.diff(velocity, axis=0) if len(velocity) > 1 else np.zeros_like(velocity)
    
    # Calculate speed (magnitude of velocity)
    speed = np.linalg.norm(velocity.reshape(len(velocity), -1, 2), axis=2)
    
    # Trajectory features
    start_frame = sequence[0]
    end_frame = sequence[-1]
    total_displacement = end_frame - start_frame
    
    return {
        'velocity': velocity,
        'acceleration': acceleration,
        'speed': speed,
        'displacement': total_displacement,
        'mean_speed': np.mean(speed),
        'max_speed': np.max(speed)
    }


def build_motion_dataset_from_videos(data_dir, num_hands=1):
    """
    Build dataset from video files.
    
    Expected structure:
      data_dir/
        class_0/
          video_0.mp4
          video_1.mp4
        class_1/
          video_0.mp4
        ...
    
    Args:
        data_dir: Directory containing class subdirectories with videos
        num_hands: Expected number of hands (1 or 2)
    
    Returns:
        sequences: List of sequences (each sequence is SEQUENCE_LENGTH frames)
        labels: List of corresponding labels
    """
    sequences = []
    labels = []
    
    if not os.path.exists(data_dir):
        print(f"[MOTION] Directory does not exist: {data_dir}")
        return sequences, labels
    
    expected_features = FEATURE_DIM_ONE_HAND if num_hands == 1 else FEATURE_DIM_TWO_HANDS
    
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        print(f"[MOTION] Processing class: {class_name}")
        
        for video_name in os.listdir(class_dir):
            video_path = os.path.join(class_dir, video_name)
            
            if not video_name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                continue
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"[MOTION] Warning: cannot open {video_path}")
                continue
            
            frame_sequence = []
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)
                
                if not results.multi_hand_landmarks:
                    continue
                
                # Check if we have the expected number of hands
                if len(results.multi_hand_landmarks) != num_hands:
                    continue
                
                # Extract features
                if num_hands == 1:
                    features = extract_hand_features(results.multi_hand_landmarks[0])
                else:  # num_hands == 2
                    features = []
                    for hand_landmarks in results.multi_hand_landmarks:
                        features.extend(extract_hand_features(hand_landmarks))
                
                if len(features) != expected_features:
                    continue
                
                frame_sequence.append(features)
                frame_count += 1
                
                # If we have enough frames, save the sequence
                if len(frame_sequence) >= SEQUENCE_LENGTH:
                    sequences.append(frame_sequence[:SEQUENCE_LENGTH])
                    labels.append(class_name)
                    frame_sequence = []  # Start new sequence
            
            cap.release()
            print(f"[MOTION]   Processed {video_name}: {frame_count} frames")
    
    print(f"[MOTION] Built {len(sequences)} sequences from {data_dir}")
    return sequences, labels


def build_motion_dataset_interactive(num_hands=1, signs_config=None):
    """
    Interactive dataset builder - captures video sequences via webcam.
    
    Args:
        num_hands: Expected number of hands (1 or 2)
        signs_config: Dict mapping sign names to class IDs
    
    Example:
        signs_config = {"thank_you": 0, "hello": 1, "sorry": 2}
    """
    if signs_config is None:
        print("No signs configuration provided!")
        return
    
    sequences = []
    labels = []
    
    cap = cv2.VideoCapture(4)
    
    print("\n=== Interactive Motion Dataset Builder ===")
    print(f"Capturing sequences of {SEQUENCE_LENGTH} frames")
    print(f"Expected hands: {num_hands}")
    print("\nSigns to capture:")
    for sign_name, class_id in signs_config.items():
        print(f"  {class_id}: {sign_name}")
    print("\nInstructions:")
    print("  - Press the number key (0-9) to start recording that sign")
    print("  - Perform the sign when recording starts")
    print("  - Press 'q' to quit and save")
    print("=" * 50)
    
    recording = False
    current_sequence = []
    current_label = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )
        
        # Display status
        if recording:
            status = f"RECORDING: {current_label} ({len(current_sequence)}/{SEQUENCE_LENGTH})"
            color = (0, 0, 255)
        else:
            status = "Press 0-9 to start recording a sign"
            color = (0, 255, 0)
        
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Sequences captured: {len(sequences)}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow('Motion Dataset Builder', frame)
        
        # Recording logic
        if recording and results.multi_hand_landmarks:
            if len(results.multi_hand_landmarks) == num_hands:
                # Extract features
                if num_hands == 1:
                    features = extract_hand_features(results.multi_hand_landmarks[0])
                else:
                    features = []
                    for hand_landmarks in results.multi_hand_landmarks:
                        features.extend(extract_hand_features(hand_landmarks))
                
                current_sequence.append(features)
                
                if len(current_sequence) >= SEQUENCE_LENGTH:
                    sequences.append(current_sequence[:SEQUENCE_LENGTH])
                    labels.append(current_label)
                    print(f"✓ Captured sequence for '{current_label}' (Total: {len(sequences)})")
                    recording = False
                    current_sequence = []
                    current_label = None
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key in [ord(str(i)) for i in range(10)]:
            digit = int(chr(key))
            # Find the sign name for this digit
            for sign_name, class_id in signs_config.items():
                if class_id == digit:
                    recording = True
                    current_sequence = []
                    current_label = sign_name
                    print(f"\nStarting recording for '{sign_name}'...")
                    break
    
    cap.release()
    cv2.destroyAllWindows()
    
    return sequences, labels


def save_motion_dataset(sequences, labels, out_path):
    """Save motion dataset to pickle file."""
    data = {
        'sequences': sequences,
        'labels': labels,
        'sequence_length': SEQUENCE_LENGTH,
        'num_sequences': len(sequences)
    }
    with open(out_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"\nSaved {len(sequences)} sequences to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build motion-based ASL dataset")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["video", "interactive"],
        default="interactive",
        help="Build from video files or capture interactively via webcam"
    )
    parser.add_argument(
        "--hands",
        type=int,
        choices=[1, 2],
        default=1,
        help="Number of hands expected (1 or 2)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Data directory for video mode"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="motion_data.pickle",
        help="Output pickle file name"
    )
    
    args = parser.parse_args()
    
    if args.mode == "video":
        if not args.data_dir:
            print("Error: --data-dir required for video mode")
            exit(1)
        
        data_dir = os.path.join(BASE_DIR, args.data_dir)
        print(f"\n=== Building MOTION dataset from videos in {data_dir} ===")
        sequences, labels = build_motion_dataset_from_videos(data_dir, args.hands)
        
        if sequences:
            save_motion_dataset(sequences, labels, args.output)
        else:
            print("[MOTION] No data collected. Check your directory and videos.")
    
    elif args.mode == "interactive":
        # Example configuration - modify as needed
        signs_config = {
            "thank_you": 0,
            "hello": 1,
            "sorry": 2,
            "please": 3,
            "help": 4
        }
        
        print(f"\n=== Interactive MOTION dataset builder ===")
        sequences, labels = build_motion_dataset_interactive(args.hands, signs_config)
        
        if sequences:
            save_motion_dataset(sequences, labels, args.output)
        else:
            print("[MOTION] No sequences captured.")
    
    print("\n✓ Done building motion dataset.")

