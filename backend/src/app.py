from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import time
import pickle
import mediapipe as mp
import os
import json
import asyncio
from threading import Thread, Lock
import base64

app = Flask(__name__)
app.config['SECRET_KEY'] = 'sign-language-interpreter-secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global state
class AppState:
    def __init__(self):
        self.camera_active = False
        self.video_capture = None
        self.detected_signs = ""
        self.suggestions = []
        self.current_mode = 'default'
        self.flip_camera = True
        self.camera_index = 0
        self.llm_processing = False
        self.lock = Lock()

state = AppState()

# Load models
def load_models():
    script_dir = os.path.dirname(__file__)
    model_letters_dict = pickle.load(open(os.path.join(script_dir, 'util', 'models', 'model_letters.p'), 'rb'))
    model_numbers_dict = pickle.load(open(os.path.join(script_dir, 'util', 'models', 'model_numbers.p'), 'rb'))
    model_words_one_dict = pickle.load(open(os.path.join(script_dir, 'util', 'models', 'model_words_one_hand.p'), 'rb'))
    model_words_two_dict = pickle.load(open(os.path.join(script_dir, 'util', 'models', 'model_words_two_hands.p'), 'rb'))
    
    return (
        model_letters_dict['model'],
        model_numbers_dict['model'],
        model_words_one_dict['model'],
        model_words_two_dict['model']
    )

def load_config():
    script_dir = os.path.dirname(__file__)
    config_dir = os.path.join(script_dir, 'config')
    
    with open(os.path.join(config_dir, 'asl_letters.json'), 'r') as f:
        labels_letters = {int(k): v + ' ' for k, v in json.load(f).items()}
    
    with open(os.path.join(config_dir, 'asl_numbers.json'), 'r') as f:
        labels_numbers = {int(k): v + ' ' for k, v in json.load(f).items()}
    
    with open(os.path.join(config_dir, 'asl_words_one_hand.json'), 'r') as f:
        labels_words_one = {int(k): v + ' ' for k, v in json.load(f).items()}
    
    with open(os.path.join(config_dir, 'asl_words_two_hands.json'), 'r') as f:
        labels_words_two = {int(k): v + ' ' for k, v in json.load(f).items()}
    
    return labels_letters, labels_numbers, labels_words_one, labels_words_two

# Load models and configs
print("Loading models...")
model_letters, model_numbers, model_words_one, model_words_two = load_models()
labels_letters, labels_numbers, labels_words_one, labels_words_two = load_config()
print("Models loaded successfully!")

# Find selection gesture indices and mode triggers
select_indices = {}
trigger_letters_idx = None
trigger_numbers_idx = None

for key, value in labels_words_two.items():
    if value.strip() == "Select 1":
        select_indices[1] = key
    elif value.strip() == "Select 2":
        select_indices[2] = key
    elif value.strip() == "Select 3":
        select_indices[3] = key
    elif value.strip() == "Letters - Mode Switch":
        trigger_letters_idx = key
    elif value.strip() == "Numbers - Mode Switch":
        trigger_numbers_idx = key

print(f"Selection gestures mapped: {select_indices}")
print(f"Letters trigger index: {trigger_letters_idx}")
print(f"Numbers trigger index: {trigger_numbers_idx}")

# Setup MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, max_num_hands=2)

# Helper function to get LLM response
def get_llm_suggestions(user_input):
    """Get suggestions from LLM in background thread"""
    if not user_input.strip():
        return
    
    with state.lock:
        state.llm_processing = True
    socketio.emit('llm_status', {'processing': True})
    
    def run_async():
        try:
            # Import with proper module path handling
            import sys
            script_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(script_dir)
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            
            from ai.get_llm_response import _get_llm_response_async
            from ai.prompts import prompt1, casual_prompt
            
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Run the async function
            prompt_text = prompt1(user_input, casual_prompt)
            response = loop.run_until_complete(_get_llm_response_async(prompt_text))
            loop.close()
            
            # Parse JSON response
            try:
                response_data = json.loads(response)
                if 'sentences' in response_data and isinstance(response_data['sentences'], list):
                    with state.lock:
                        state.suggestions = response_data['sentences'][:3]
                    socketio.emit('suggestions_update', {'suggestions': state.suggestions})
            except json.JSONDecodeError:
                with state.lock:
                    state.suggestions = ["Error parsing response"]
                socketio.emit('suggestions_update', {'suggestions': state.suggestions})
        except Exception as e:
            print(f"Error getting LLM response: {e}")
            import traceback
            traceback.print_exc()
            with state.lock:
                state.suggestions = [f"Error: {str(e)}"]
            socketio.emit('suggestions_update', {'suggestions': state.suggestions})
        finally:
            with state.lock:
                state.llm_processing = False
            socketio.emit('llm_status', {'processing': False})
    
    thread = Thread(target=run_async, daemon=True)
    thread.start()

def process_video():
    """Process video stream and emit frames via SocketIO - using exact inference_classifier logic"""
    letters_detected = state.detected_signs
    previous_detected = ""
    current_character = ""
    character_start_time = None
    HOLD_DURATION = 1.5
    last_llm_call_time = 0
    LLM_CALL_COOLDOWN = 2.0
    
    # Mode switching state
    last_mode_switch_time = 0
    last_mode_switch_gesture = None
    MODE_SWITCH_COOLDOWN = 1.0
    
    while state.camera_active and state.video_capture:
        ret, frame = state.video_capture.read()
        if not ret:
            break
        
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if state.flip_camera:
            frame_rgb = cv2.flip(frame_rgb, 1)
        
        results = hands.process(frame_rgb)
        predicted_character = None
        prediction = None
        num_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
        
        # Process hand detection (exact logic from inference_classifier)
        if results.multi_hand_landmarks and num_hands in [1, 2]:
            if num_hands == 1:
                hand_landmarks = results.multi_hand_landmarks[0]
                data_aux, x_, y_ = [], [], []
                
                mp_drawing.draw_landmarks(
                    frame_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                for lm in hand_landmarks.landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)
                
                min_x, min_y = min(x_), min(y_)
                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min_x)
                    data_aux.append(lm.y - min_y)
                
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) + 10
                y2 = int(max(y_) * H) + 10
                
                if len(data_aux) == 42:
                    try:
                        if state.current_mode == 'letters':
                            prediction = model_letters.predict([np.asarray(data_aux)])
                            pred_idx = int(prediction[0])
                            if pred_idx in labels_letters:
                                predicted_character = labels_letters[pred_idx]
                        elif state.current_mode == 'numbers':
                            prediction = model_numbers.predict([np.asarray(data_aux)])
                            pred_idx = int(prediction[0])
                            if pred_idx in labels_numbers:
                                predicted_character = labels_numbers[pred_idx]
                        else:
                            prediction = model_words_one.predict([np.asarray(data_aux)])
                            pred_idx = int(prediction[0])
                            if pred_idx in labels_words_one:
                                predicted_character = labels_words_one[pred_idx]
                    except Exception:
                        pass
                
                color = (0, 0, 0)
                cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), color, 4)
                
                if predicted_character and character_start_time and current_character != previous_detected:
                    elapsed_time = time.time() - character_start_time
                    progress = min(elapsed_time / HOLD_DURATION, 1.0)
                    bar_width = int((x2 - x1) * progress)
                    cv2.rectangle(frame_rgb, (x1, y2 + 5), (x1 + bar_width, y2 + 15), (0, 255, 0), -1)
                    cv2.rectangle(frame_rgb, (x1, y2 + 5), (x2, y2 + 15), (0, 0, 0), 2)
            
            elif num_hands == 2:
                data_aux, all_x, all_y = [], [], []
                
                for hand_landmarks in results.multi_hand_landmarks:
                    x_hand, y_hand = [], []
                    
                    mp_drawing.draw_landmarks(
                        frame_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    for lm in hand_landmarks.landmark:
                        x_hand.append(lm.x)
                        y_hand.append(lm.y)
                        all_x.append(lm.x)
                        all_y.append(lm.y)
                    
                    for lm in hand_landmarks.landmark:
                        data_aux.append(lm.x - min(x_hand))
                        data_aux.append(lm.y - min(y_hand))
                
                x1 = int(min(all_x) * W) - 10
                y1 = int(min(all_y) * H) - 10
                x2 = int(max(all_x) * W) + 10
                y2 = int(max(all_y) * H) + 10
                
                if len(data_aux) == 84:
                    try:
                        prediction = model_words_two.predict([np.asarray(data_aux)])
                        pred_idx = int(prediction[0])
                        if pred_idx in labels_words_two:
                            predicted_character = labels_words_two[pred_idx]
                    except Exception:
                        pass
                
                cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 4)
                
                if predicted_character and character_start_time and current_character != previous_detected:
                    elapsed_time = time.time() - character_start_time
                    progress = min(elapsed_time / HOLD_DURATION, 1.0)
                    bar_width = int((x2 - x1) * progress)
                    cv2.rectangle(frame_rgb, (x1, y2 + 5), (x1 + bar_width, y2 + 15), (0, 255, 0), -1)
                    cv2.rectangle(frame_rgb, (x1, y2 + 5), (x2, y2 + 15), (0, 0, 0), 2)
            
            # Character tracking logic (exact from inference_classifier)
            if predicted_character is not None:
                if predicted_character != current_character:
                    current_character = predicted_character
                    character_start_time = time.time()
                else:
                    if character_start_time is not None:
                        elapsed_time = time.time() - character_start_time
                        if elapsed_time >= HOLD_DURATION and current_character != previous_detected:
                            previous_detected = current_character
                            
                            # Check if this is a special gesture (selection or mode switch) for two-hand gestures
                            is_special_gesture = False
                            if num_hands == 2 and prediction is not None:
                                prediction_id = int(prediction[0])
                                current_time = time.time()
                                
                                # Check if it's a Select 1/2/3 gesture
                                for select_num, select_id in select_indices.items():
                                    if prediction_id == select_id:
                                        # Only process if we have suggestions available
                                        if len(state.suggestions) > 0:
                                            selection_index = select_num - 1  # 1->0, 2->1, 3->2
                                            if selection_index < len(state.suggestions):
                                                selected_sentence = state.suggestions[selection_index]
                                                print(f"\n{'='*60}")
                                                print(f"✅ Selected option {select_num}: {selected_sentence}")
                                                print(f"{'='*60}\n")
                                                
                                                # Clear detected signs and suggestions
                                                letters_detected = ""
                                                with state.lock:
                                                    state.detected_signs = ""
                                                    state.suggestions = []
                                                
                                                # Emit clear updates
                                                socketio.emit('detected_update', {'text': ''})
                                                socketio.emit('suggestions_update', {'suggestions': []})
                                                socketio.emit('selection_made', {'text': selected_sentence})
                                                
                                                print("📝 Detected letters reset. Ready for new gesture sequence.\n")
                                                is_special_gesture = True
                                                break
                                        else:
                                            print(f"⚠️ Select gesture shown but no suggestions available (ignored)")
                                            is_special_gesture = True
                                            break
                                
                                # Check if Letters trigger
                                if not is_special_gesture and prediction_id == trigger_letters_idx:
                                    # Skip if we just switched with this same gesture
                                    if last_mode_switch_gesture == prediction_id:
                                        # Check cooldown to prevent rapid switching
                                        if current_time - last_mode_switch_time < MODE_SWITCH_COOLDOWN:
                                            print(f"⏳ Mode switch cooldown: {MODE_SWITCH_COOLDOWN - (current_time - last_mode_switch_time):.1f}s remaining")
                                            is_special_gesture = True
                                    
                                    if not is_special_gesture:
                                        # Cooldown passed or different gesture, allow switch
                                        last_mode_switch_time = current_time
                                        last_mode_switch_gesture = prediction_id
                                        if state.current_mode == 'letters':
                                            state.current_mode = 'default'
                                            print(f"\n🔄 Switched to DEFAULT mode (words)\n")
                                        else:
                                            state.current_mode = 'letters'
                                            print(f"\n🔄 Switched to LETTERS mode\n")
                                        
                                        # Emit mode update
                                        socketio.emit('mode_update', {'mode': state.current_mode})
                                        
                                        previous_detected = current_character  # Prevent re-triggering this gesture
                                        is_special_gesture = True
                                
                                # Check if Numbers trigger
                                elif not is_special_gesture and prediction_id == trigger_numbers_idx:
                                    # Skip if we just switched with this same gesture
                                    if last_mode_switch_gesture == prediction_id:
                                        # Check cooldown to prevent rapid switching
                                        if current_time - last_mode_switch_time < MODE_SWITCH_COOLDOWN:
                                            print(f"⏳ Mode switch cooldown: {MODE_SWITCH_COOLDOWN - (current_time - last_mode_switch_time):.1f}s remaining")
                                            is_special_gesture = True
                                    
                                    if not is_special_gesture:
                                        # Cooldown passed or different gesture, allow switch
                                        last_mode_switch_time = current_time
                                        last_mode_switch_gesture = prediction_id
                                        if state.current_mode == 'numbers':
                                            state.current_mode = 'default'
                                            print(f"\n🔄 Switched to DEFAULT mode (words)\n")
                                        else:
                                            state.current_mode = 'numbers'
                                            print(f"\n🔄 Switched to NUMBERS mode\n")
                                        
                                        # Emit mode update
                                        socketio.emit('mode_update', {'mode': state.current_mode})
                                        
                                        previous_detected = current_character  # Prevent re-triggering this gesture
                                        is_special_gesture = True
                            
                            # Only add to detected text if not a special gesture
                            if not is_special_gesture:
                                letters_detected += current_character
                                
                                with state.lock:
                                    state.detected_signs = letters_detected
                                
                                print(f"Added '{current_character}' to detected letters: {letters_detected}")
                                
                                # Emit update to client
                                socketio.emit('detected_update', {'text': letters_detected})
                                
                                # Call LLM with cooldown
                                current_time = time.time()
                                if current_time - last_llm_call_time >= LLM_CALL_COOLDOWN:
                                    get_llm_suggestions(letters_detected)
                                    last_llm_call_time = current_time
                
                # Display character with bounding box
                if predicted_character:
                    cv2.putText(frame_rgb, predicted_character, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0) if num_hands == 1 else (0, 255, 0), 3, cv2.LINE_AA)
            else:
                current_character = ""
                character_start_time = None
        else:
            current_character = ""
            character_start_time = None
        
        # Display detected text on frame
        cv2.putText(frame_rgb, f"Detected: {letters_detected}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Convert to JPEG and emit (higher quality, faster)
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        frame_data = base64.b64encode(buffer).decode('utf-8')
        socketio.emit('video_frame', {'data': frame_data})
        
        time.sleep(0.01)  # Faster FPS

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/state', methods=['GET'])
def get_state():
    with state.lock:
        return jsonify({
            'camera_active': state.camera_active,
            'detected_signs': state.detected_signs,
            'suggestions': state.suggestions,
            'current_mode': state.current_mode,
            'llm_processing': state.llm_processing
        })

# SocketIO events
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('state_update', {
        'camera_active': state.camera_active,
        'detected_signs': state.detected_signs,
        'suggestions': state.suggestions,
        'current_mode': state.current_mode
    })

@socketio.on('start_camera')
def handle_start_camera():
    if not state.camera_active:
        state.video_capture = cv2.VideoCapture(state.camera_index)
        if state.video_capture.isOpened():
            state.camera_active = True
            emit('camera_status', {'active': True}, broadcast=True)
            # Start processing in background thread
            thread = Thread(target=process_video, daemon=True)
            thread.start()
        else:
            emit('error', {'message': 'Cannot open camera'})

@socketio.on('stop_camera')
def handle_stop_camera():
    if state.camera_active:
        state.camera_active = False
        if state.video_capture:
            state.video_capture.release()
        state.video_capture = None
        emit('camera_status', {'active': False}, broadcast=True)

@socketio.on('clear_all')
def handle_clear_all():
    with state.lock:
        state.detected_signs = ""
        state.suggestions = []
    emit('detected_update', {'text': ''}, broadcast=True)
    emit('suggestions_update', {'suggestions': []}, broadcast=True)

@socketio.on('change_mode')
def handle_change_mode(data):
    state.current_mode = data.get('mode', 'default')
    emit('mode_update', {'mode': state.current_mode}, broadcast=True)

@socketio.on('get_suggestions')
def handle_get_suggestions():
    if state.detected_signs.strip():
        get_llm_suggestions(state.detected_signs)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)
