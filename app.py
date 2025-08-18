# app.py - Main Flask application
from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import json
import threading
import time

app = Flask(__name__)

class HandGestureRecognizer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.6
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.current_gesture = "None"
        self.gesture_data = []
        self.frame_count = 0
        
        # --- Object Interaction Variables ---
        self.object_state = "idle" # 'idle', 'holding'
        self.object_position = {'x': 320, 'y': 240} # Initial position of the object (centered on 640x480)
        self.object_base_size = 50 # Base size for the object
        self.object_current_size_multiplier = 1.0 # Multiplier for current size (e.g., 1.0, 1.5)
        self.object_color_index = 0 # Index for color cycling
        self.colors = ["#ff6347", "#4a90e2", "#28a745", "#fd7e14", "#6f42c1"] # Tomato, Blue, Green, Orange, Purple
        self.held_by_hand_idx = None # Which hand is holding the object (0 for first hand, 1 for second)

        self.last_color_change_time = 0
        self.last_size_change_time = 0
        self.gesture_cooldown = 0.5 # seconds to prevent rapid changes
        # -----------------------------------
        
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def recognize_gesture(self, landmarks):
        """Recognize hand gestures based on landmark positions"""
        wrist = landmarks[0]
        thumb_cmc = landmarks[1]
        thumb_mcp = landmarks[2]
        thumb_ip = landmarks[3]
        thumb_tip = landmarks[4]

        index_mcp = landmarks[5]
        index_pip = landmarks[6]
        index_dip = landmarks[7]
        index_tip = landmarks[8]

        middle_mcp = landmarks[9]
        middle_pip = landmarks[10]
        middle_dip = landmarks[11]
        middle_tip = landmarks[12]

        ring_mcp = landmarks[13]
        ring_pip = landmarks[14]
        ring_dip = landmarks[15]
        ring_tip = landmarks[16]

        pinky_mcp = landmarks[17]
        pinky_pip = landmarks[18]
        pinky_dip = landmarks[19]
        pinky_tip = landmarks[20]

        fingers_up = []

        thumb_extended_vertically = thumb_tip[1] < thumb_ip[1] and self.calculate_distance(thumb_tip, thumb_ip) > 20
        thumb_x_offset_from_mcp = thumb_tip[0] - thumb_mcp[0]
        thumb_extended_horizontally_right = thumb_x_offset_from_mcp > 30 
        
        if thumb_extended_vertically or thumb_extended_horizontally_right:
            fingers_up.append(1)
        else:
            fingers_up.append(0)

        for tip, pip in [(index_tip, index_pip), (middle_tip, middle_pip), 
                         (ring_tip, ring_pip), (pinky_tip, pinky_pip)]:
            if tip[1] < pip[1] and self.calculate_distance(tip, pip) > 20:
                fingers_up.append(1)
            else:
                fingers_up.append(0)
        
        total_fingers = sum(fingers_up)
        
        # --- Gesture Recognition Logic ---
        if total_fingers == 0:
            if (fingers_up[0] == 1 and all(f == 0 for f in fingers_up[1:]) and self.calculate_distance(thumb_tip, index_tip) > 25): 
                return "ASL A"
            else:
                return "Fist" # Used for "dropping" the object

        elif total_fingers == 1:
            if fingers_up[0] == 1 and all(f == 0 for f in fingers_up[1:]):
                if abs(thumb_tip[0] - index_tip[0]) < 50 and abs(thumb_tip[0] - middle_tip[0]) < 50:
                     return "ASL T"
                else:
                     return "Thumbs Up" # New: Used for "bouncing" the object

            elif fingers_up[1] == 1 and all(f == 0 for f in [fingers_up[0], *fingers_up[2:]]):
                dist_thumb_to_middle_pip = self.calculate_distance(thumb_tip, middle_pip)
                if dist_thumb_to_middle_pip < 50:
                    return "ASL D"
                else:
                    return "Point" # Used for "moving" the object
            else:
                if fingers_up[4] == 1 and all(f == 0 for f in fingers_up[:4]):
                    return "ASL I"
                return "One Finger" 

        elif total_fingers == 2:
            if fingers_up[1] == 1 and fingers_up[2] == 1 and fingers_up[0] == 0 and fingers_up[3] == 0 and fingers_up[4] == 0:
                # Check for ASL H (fingers close) vs Peace/ASL V (fingers separated)
                dist_index_middle_tips = self.calculate_distance(index_tip, middle_tip)
                if dist_index_middle_tips < 40: # Threshold for 'together'
                    return "ASL H" 
                else:
                    return "Peace Sign" # New: Used for "changing color"

            elif fingers_up[0] == 1 and fingers_up[4] == 1 and fingers_up[1] == 0 and fingers_up[2] == 0 and fingers_up[3] == 0:
                return "Rock On"
            
            elif fingers_up[0] == 1 and fingers_up[1] == 1 and not fingers_up[2] and not fingers_up[3] and not fingers_up[4]:
                return "ASL L" # New: Used for "toggling size"
            
            elif fingers_up[1] == 1 and fingers_up[2] == 1 and not fingers_up[0] and not fingers_up[3] and not fingers_up[4]:
                dist_index_middle_tips = self.calculate_distance(index_tip, middle_tip)
                dist_index_middle_mcp = self.calculate_distance(index_mcp, middle_mcp)
                if dist_index_middle_tips > dist_index_middle_mcp * 0.8:
                    return "ASL V"
                else:
                    return "Peace Sign" # This condition should now fall under the earlier Peace Sign check
            
            else:
                return "Two Fingers"

        elif total_fingers == 3:
            if fingers_up[0] == 0 and fingers_up[1] == 1 and fingers_up[2] == 1 and fingers_up[3] == 1 and fingers_up[4] == 0:
                return "ASL W"
            
            dist_thumb_index_tip = self.calculate_distance(thumb_tip, index_tip)
            dist_index_pip_tip = self.calculate_distance(index_pip, index_tip)
            if dist_thumb_index_tip < dist_index_pip_tip * 0.5:
                if fingers_up[2] == 1 and fingers_up[3] == 1 and fingers_up[4] == 1:
                    return "ASL F"
                elif fingers_up[2] == 1 and fingers_up[3] == 1 and fingers_up[4] == 0:
                     return "OK Gesture"
            
            return "Three Fingers"

        elif total_fingers == 4:
            if fingers_up[0] == 0 and fingers_up[1] == 1 and fingers_up[2] == 1 and fingers_up[3] == 1 and fingers_up[4] == 1:
                return "ASL 4"
            return "Four Fingers"

        elif total_fingers == 5:
            all_fingers_extended = all(f == 1 for f in fingers_up)
            dist_thumb_pinky_mcp = self.calculate_distance(thumb_mcp, pinky_mcp)
            dist_index_ring_mcp = self.calculate_distance(index_mcp, ring_mcp)
            palm_length_approx = self.calculate_distance(wrist, index_mcp)

            openness_threshold_thumb_pinky = palm_length_approx * 0.8
            openness_threshold_index_ring = palm_length_approx * 0.6

            if all_fingers_extended and \
               dist_thumb_pinky_mcp > openness_threshold_thumb_pinky and \
               dist_index_ring_mcp > openness_threshold_index_ring:
                return "Open Palm" # Used for "picking up" the object
            
            index_bent = index_tip[1] > index_pip[1] and self.calculate_distance(index_tip, index_pip) < self.calculate_distance(index_mcp, index_tip) * 0.8
            middle_bent = middle_tip[1] > middle_pip[1] and self.calculate_distance(middle_tip, middle_pip) < self.calculate_distance(middle_mcp, middle_tip) * 0.8
            ring_bent = ring_tip[1] > ring_pip[1] and self.calculate_distance(ring_tip, ring_pip) < self.calculate_distance(ring_mcp, ring_tip) * 0.8
            pinky_bent = pinky_tip[1] > pinky_pip[1] and self.calculate_distance(pinky_tip, pinky_pip) < self.calculate_distance(pinky_mcp, pinky_tip) * 0.8

            if all_fingers_extended and \
               index_bent and middle_bent and ring_bent and pinky_bent:
                return "ASL C"
            
            return "Five Fingers"

        return "Unknown"
    
    def process_frame(self, frame):
        """Process video frame for hand detection and gesture recognition"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        h, w, _ = frame.shape
        
        gesture_info = {
            'gestures': [],
            'hand_count': 0,
            'object_state': self.object_state,
            'object_position': self.object_position,
            'object_size': {'width': int(self.object_base_size * self.object_current_size_multiplier),
                            'height': int(self.object_base_size * self.object_current_size_multiplier)},
            'object_color': self.colors[self.object_color_index % len(self.colors)],
            'object_bounce_trigger': False # Set to True for a frame if bounce should occur
        }
        
        detected_hands_detailed = []

        if results.multi_hand_landmarks:
            gesture_info['hand_count'] = len(results.multi_hand_landmarks)
            
            for hand_landmarks_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                self.mp_draw.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append([int(lm.x * w), int(lm.y * h)])
                
                gesture = self.recognize_gesture(landmarks)
                
                detected_hands_detailed.append({
                    'gesture': gesture,
                    'index_tip': landmarks[8],
                    'wrist': landmarks[0],
                    'hand_idx': hand_landmarks_idx,
                    'all_landmarks': landmarks
                })

                gesture_info['gestures'].append({
                    'gesture': gesture,
                    'landmarks': landmarks # Send all landmarks for potential frontend use (e.g., debug dot)
                })
                
                text_x = landmarks[0][0]
                text_y = landmarks[0][1] - 40 - (hand_landmarks_idx * 30)

                cv2.putText(frame, gesture, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
        
        # --- Object Interaction Logic ---
        current_time = time.time()

        if self.object_state == 'holding':
            holding_hand = None
            for hand in detected_hands_detailed:
                if hand['hand_idx'] == self.held_by_hand_idx:
                    holding_hand = hand
                    break
            
            if holding_hand:
                if holding_hand['gesture'] == 'Open Palm' or holding_hand['gesture'] == 'Point':
                    self.object_position['x'] = holding_hand['index_tip'][0]
                    self.object_position['y'] = holding_hand['index_tip'][1]

                    # New Feature: Change Color
                    if holding_hand['gesture'] == 'Peace Sign' and (current_time - self.last_color_change_time > self.gesture_cooldown):
                        self.object_color_index = (self.object_color_index + 1) % len(self.colors)
                        self.last_color_change_time = current_time
                        print(f"Object color changed to: {self.colors[self.object_color_index]}")

                    # New Feature: Toggle Size
                    elif holding_hand['gesture'] == 'ASL L' and (current_time - self.last_size_change_time > self.gesture_cooldown):
                        if self.object_current_size_multiplier == 1.0:
                            self.object_current_size_multiplier = 1.5 # Make larger
                        else:
                            self.object_current_size_multiplier = 1.0 # Back to normal
                        self.last_size_change_time = current_time
                        print(f"Object size toggled to: {self.object_current_size_multiplier}")
                    
                    # New Feature: Explicit Bounce Trigger
                    elif holding_hand['gesture'] == 'Thumbs Up' and (current_time - self.last_bounce_time > self.gesture_cooldown):
                        gesture_info['object_bounce_trigger'] = True # Signal frontend to bounce
                        self.last_bounce_time = current_time
                        print("Bounce triggered!")

                elif holding_hand['gesture'] == 'Fist' or holding_hand['gesture'] == 'Unknown' or not holding_hand:
                    self.object_state = 'idle'
                    self.held_by_hand_idx = None
                    gesture_info['object_bounce_trigger'] = True # Automatic bounce on drop
                    print("Object dropped!")
            else:
                self.object_state = 'idle'
                self.held_by_hand_idx = None
                gesture_info['object_bounce_trigger'] = True # Automatic bounce on hand disappearance
                print("Holding hand disappeared, object dropped!")

        else: # self.object_state == 'idle'
            for hand in detected_hands_detailed:
                if hand['gesture'] == 'Open Palm':
                    hand_index_tip_x = hand['index_tip'][0]
                    hand_index_tip_y = hand['index_tip'][1]
                    
                    obj_center_x = self.object_position['x']
                    obj_center_y = self.object_position['y']
                    obj_half_w = gesture_info['object_size']['width'] / 2
                    obj_half_h = gesture_info['object_size']['height'] / 2

                    is_colliding = (obj_center_x - obj_half_w < hand_index_tip_x < obj_center_x + obj_half_w and
                                    obj_center_y - obj_half_h < hand_index_tip_y < obj_center_y + obj_half_h)
                    
                    if is_colliding:
                        self.object_state = 'holding'
                        self.held_by_hand_idx = hand['hand_idx']
                        print(f"Object picked up by hand {self.held_by_hand_idx}!")
                        break
        
        # Ensure object stays within frame boundaries
        self.object_position['x'] = np.clip(self.object_position['x'], gesture_info['object_size']['width']/2, w - gesture_info['object_size']['width']/2)
        self.object_position['y'] = np.clip(self.object_position['y'], gesture_info['object_size']['height']/2, h - gesture_info['object_size']['height']/2)

        # Update gesture_info with the latest object state, position, size, and color
        gesture_info['object_state'] = self.object_state
        gesture_info['object_position'] = self.object_position
        gesture_info['object_size']['width'] = int(self.object_base_size * self.object_current_size_multiplier)
        gesture_info['object_size']['height'] = int(self.object_base_size * self.object_current_size_multiplier)
        gesture_info['object_color'] = self.colors[self.object_color_index % len(self.colors)]
        # -----------------------------------------------

        return frame, gesture_info

# Global variables
gesture_recognizer = HandGestureRecognizer()
camera = None
# Initialize current_gesture_data with default object state and position
# Ensure initial object_size and object_color match what's in the recognizer
current_gesture_data = {
    'gestures': [],
    'hand_count': 0,
    'object_state': 'idle',
    'object_position': {'x': 320, 'y': 240},
    'object_size': {'width': gesture_recognizer.object_base_size, 'height': gesture_recognizer.object_base_size},
    'object_color': gesture_recognizer.colors[0],
    'object_bounce_trigger': False
}
camera_status_message = "Camera initializing..."


def initialize_camera():
    """Initialize camera with multiple fallback options"""
    global camera, camera_status_message
    
    if camera is not None:
        camera.release()
        camera = None

    preferred_cameras = [0, 1]
    
    for camera_index in preferred_cameras:
        try:
            camera = cv2.VideoCapture(camera_index)
            if camera.isOpened():
                ret, frame = camera.read()
                if ret and frame is not None:
                    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    camera.set(cv2.CAP_PROP_FPS, 30)
                    camera_status_message = f"Camera {camera_index} initialized."
                    print(f"Camera initialized successfully with index {camera_index}")
                    return True
                else:
                    camera.release()
            else:
                if camera is not None:
                    camera.release()
        except Exception as e:
            print(f"Failed to initialize camera with index {camera_index}: {e}")
            if camera is not None:
                camera.release()
    
    if cv2.CAP_DSHOW:
        try:
            camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if camera.isOpened():
                ret, frame = camera.read()
                if ret and frame is not None:
                    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    camera_status_message = "Camera initialized with DirectShow backend."
                    print("Camera initialized with DirectShow backend")
                    return True
                else:
                    camera.release()
        except Exception as e:
            print(f"DirectShow backend failed: {e}")
    
    camera_status_message = "ERROR: Could not initialize any camera."
    print("ERROR: Could not initialize any camera!")
    camera = None
    return False

def generate_frames():
    """Generate video frames for streaming"""
    global current_gesture_data, camera_status_message
    
    while True:
        if camera is None:
            black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            text_color = (0, 0, 255)
            cv2.putText(black_frame, camera_status_message, (int(640/2) - 200, 200),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)
            cv2.putText(black_frame, "Please ensure camera is connected", (int(640/2) - 250, 230),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1, cv2.LINE_AA)
            cv2.putText(black_frame, "and try clicking 'Start Camera'", (int(640/2) - 250, 260),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1, cv2.LINE_AA)
            
            ret, buffer = cv2.imencode('.jpg', black_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(1)
            continue
            
        success, frame = camera.read()
        if not success:
            print("Failed to read from camera. Attempting to reinitialize...")
            camera_status_message = "Camera read error. Reinitializing..."
            if not initialize_camera():
                error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                text_color = (0, 0, 255)
                cv2.putText(error_frame, "Camera Disconnected!", (int(640/2) - 200, 200),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2, cv2.LINE_AA)
                cv2.putText(error_frame, "Please check connection and restart.", (int(640/2) - 250, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1, cv2.LINE_AA)
                ret, buffer = cv2.imencode('.jpg', error_frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                time.sleep(1)
                continue
            time.sleep(0.5)
            continue
        
        frame = cv2.flip(frame, 1)
        
        try:
            processed_frame, gesture_info = gesture_recognizer.process_frame(frame)
            current_gesture_data = gesture_info
        except Exception as e:
            print(f"Error processing frame: {e}")
            processed_frame = frame
            current_gesture_data = {
                'gestures': [],
                'hand_count': 0,
                'object_state': 'idle',
                'object_position': {'x': 320, 'y': 240},
                'object_size': {'width': gesture_recognizer.object_base_size, 'height': gesture_recognizer.object_base_size},
                'object_color': gesture_recognizer.colors[0],
                'object_bounce_trigger': False
            }
        
        cv2.putText(processed_frame, "AI Hand Gesture Recognizer", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(processed_frame, f"Hands: {current_gesture_data['hand_count']}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)
        
        cv2.putText(processed_frame, f"Object State: {current_gesture_data['object_state'].capitalize()}", (10, 450),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1, cv2.LINE_AA)

        if current_gesture_data['gestures']:
            y_offset = 90
            for i, gesture_entry in enumerate(current_gesture_data['gestures']):
                gesture_text = f"Gesture {i+1}: {gesture_entry['gesture']}"
                cv2.putText(processed_frame, gesture_text, (10, y_offset + (i*30)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/gesture_data')
def gesture_data():
    """API endpoint to get current gesture data and object state"""
    return jsonify(current_gesture_data)

@app.route('/start_camera')
def start_camera():
    """Start camera"""
    success = initialize_camera()
    if success:
        return jsonify({'status': 'success', 'message': 'Camera started successfully'})
    else:
        return jsonify({'status': 'error', 'message': camera_status_message}), 500

@app.route('/stop_camera')
def stop_camera():
    """Stop camera"""
    global camera, camera_status_message
    if camera is not None:
        camera.release()
        camera = None
        camera_status_message = "Camera stopped by user."
        return jsonify({'status': 'success', 'message': 'Camera stopped'})
    else:
        camera_status_message = "Camera was already off."
        return jsonify({'status': 'info', 'message': 'Camera was already off'})

if __name__ == '__main__':
    app.run(debug=True, threaded=True)