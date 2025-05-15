import cv2
import numpy as np
from collections import deque
import time
import mediapipe as mp
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pickle
import winsound
import threading

# Manually set the paths for YOLO and COCO files
YOLO_CFG_PATH = r"C:\Users\arunk\OneDrive\Desktop\fight detection\fight detection\yolov4-tiny.cfg"
YOLO_WEIGHTS_PATH = r"C:\Users\arunk\OneDrive\Desktop\fight detection\fight detection\yolov4-tiny.weights"
COCO_NAMES_PATH = r"C:\Users\arunk\OneDrive\Desktop\fight detection\fight detection\coco.names"

# Initialize MediaPipe components with optimized settings
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Performance optimization: Use lighter models and lower detection confidence thresholds
pose = mp_pose.Pose(
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5,
    model_complexity=1,  # Use medium complexity model (0=light, 1=medium, 2=heavy)
    static_image_mode=False  # Ensure video stream compatibility
)
hands = mp_hands.Hands(
    min_detection_confidence=0.6, 
    min_tracking_confidence=0.5,
    max_num_hands=2,
    model_complexity=0  # Use light complexity model
)

# Load YOLO for person detection - use smaller input size for speed
net = cv2.dnn.readNet(YOLO_WEIGHTS_PATH, YOLO_CFG_PATH)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)  # Use OpenCV backend for better CPU performance
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # Can change to DNN_TARGET_CUDA if GPU available

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
with open(COCO_NAMES_PATH, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Global variables for threading
frame_processing_active = False
processed_frame = None
people_data_global = []
fps_global = 0
frame_buffer = deque(maxlen=8)  # Reduced from 16 to 8 for better performance

# Function to extract features from a single person - optimized
def extract_features(pose_landmarks, hand_landmarks):
    features = np.zeros(67)
    if pose_landmarks and hand_landmarks:
        # Get key landmarks for faster processing
        left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

        # Calculate key distances
        shoulder_dist = np.sqrt((left_shoulder.x - right_shoulder.x) ** 2 + (left_shoulder.y - right_shoulder.y) ** 2)
        wrist_dist = np.sqrt((left_wrist.x - right_wrist.x) ** 2 + (left_wrist.y - right_wrist.y) ** 2)
        features[0] = shoulder_dist
        features[1] = wrist_dist

        # Hand landmarks
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        wrist_hand = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

        finger_dist = np.sqrt((thumb_tip.x - index_finger_tip.x) ** 2 + (thumb_tip.y - index_finger_tip.y) ** 2)
        wrist_to_thumb = np.sqrt((wrist_hand.x - thumb_tip.x) ** 2 + (wrist_hand.y - thumb_tip.y) ** 2)
        features[2] = finger_dist
        features[3] = wrist_to_thumb

        # Store all hand landmarks
        for i, landmark in enumerate(hand_landmarks.landmark):
            if i < 21:
                features[4 + i * 3] = landmark.x
                features[5 + i * 3] = landmark.y
                features[6 + i * 3] = landmark.z
    return features

# Violence detection logic - simplified for better performance
def detect_violence(pose_landmarks, hand_landmarks, frame_buffer, model=None):
    # Return immediately if landmarks are missing
    if not pose_landmarks or not hand_landmarks:
        return "Non-Violent", 0.0
        
    # Extract features for ML model
    features = extract_features(pose_landmarks, hand_landmarks)
    
    # If we have a trained model, use it
    if model is not None:
        try:
            action_id = model.predict([features])[0]
            action_map = {0: "Non-Violent", 1: "Grappling", 2: "Punching", 3: "Weapon Attack"}
            confidence = max(0.7, np.max(model.decision_function([features])) / 10) if hasattr(model, 'decision_function') else 0.8
            return action_map.get(action_id, "Non-Violent"), confidence
        except Exception as e:
            print(f"Model prediction error: {e}. Using fallback detection.")
    
    # Fallback to heuristic-based detection (simplified)
    left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

    shoulder_dist = np.sqrt((left_shoulder.x - right_shoulder.x) ** 2 + (left_shoulder.y - right_shoulder.y) ** 2)
    wrist_dist = np.sqrt((left_wrist.x - right_wrist.x) ** 2 + (left_wrist.y - right_wrist.y) ** 2)

    hand_landmark = hand_landmarks.landmark
    thumb_tip = hand_landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_finger_tip = hand_landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    wrist_hand = hand_landmark[mp_hands.HandLandmark.WRIST]

    finger_dist = np.sqrt((thumb_tip.x - index_finger_tip.x) ** 2 + (thumb_tip.y - index_finger_tip.y) ** 2)
    wrist_to_thumb = np.sqrt((wrist_hand.x - thumb_tip.x) ** 2 + (wrist_hand.y - thumb_tip.y) ** 2)

    # Calculate motion score from frame buffer (if available)
    motion_score = 0
    if len(frame_buffer) >= 2:
        prev_frame = cv2.cvtColor(frame_buffer[-2], cv2.COLOR_RGB2GRAY)
        curr_frame = cv2.cvtColor(frame_buffer[-1], cv2.COLOR_RGB2GRAY)
        # Use faster absdiff and mean calculation
        frame_diff = cv2.absdiff(curr_frame, prev_frame)
        motion_score = np.mean(frame_diff)

    # Simple heuristic rules
    if finger_dist < 0.1 and wrist_to_thumb > 0.3 and motion_score > 50:
        return "Knife Attack", 0.7
    elif wrist_dist < 0.1 and shoulder_dist > 0.5:
        return "Grappling", 0.7
    elif wrist_dist > 0.5 and shoulder_dist < 0.3:
        return "Punching", 0.65
    
    return "Non-Violent", 0.0

# Training function - optimize for speed
def train_model(X, y):
    if len(np.unique(y)) <= 1:
        raise ValueError("Dataset contains only one class. Collect data for multiple actions.")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Use a faster kernel (linear) for real-time performance
    model = SVC(kernel='linear', C=1.0, probability=True)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "live_trained_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved as {model_path}")
    return model

# Background frame processing thread to improve UI responsiveness
def process_frame_thread(frame, model=None):
    global processed_frame, people_data_global, fps_global, frame_processing_active
    
    start_time = time.time()
    
    # YOLO person detection with smaller input size (320x320 instead of 416x416)
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes, confidences = [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if class_id == 0 and confidence > 0.5:  # 0 is "person" in COCO
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                box_x = int(center_x - w / 2)
                box_y = int(center_y - h / 2)
                boxes.append([box_x, box_y, w, h])
                confidences.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    if len(indices) == 0:
        # No people detected, return quickly
        cv2.putText(frame, "No people detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        processed_frame = frame
        fps_global = 1 / (time.time() - start_time)
        frame_processing_active = False
        return
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_buffer.append(rgb_frame)
    output_frame = frame.copy()
    
    # Store people's data
    people_data = []

    # Process each detected person
    for i in range(len(indices)):
        idx = indices[i]
        idx = idx if isinstance(idx, np.int32) else idx[0]  # Handle numpy int32 or tuple
        box_x, box_y, w, h = boxes[idx]
        box_x, box_y = max(0, box_x), max(0, box_y)
        roi = rgb_frame[box_y:box_y+h, box_x:box_x+w]
        
        if roi.size == 0:
            continue

        # Apply MediaPipe to ROI
        pose_results = pose.process(roi)
        hand_results = hands.process(roi)
        
        person_data = {
            "id": i,
            "box": (box_x, box_y, w, h),
            "center": (box_x + w/2, box_y + h/2),
            "pose_landmarks": None,
            "hand_landmarks": None,
            "action": "Non-Violent",
            "confidence": 0.0
        }

        # Draw bounding box
        cv2.rectangle(output_frame, (box_x, box_y), (box_x + w, box_y + h), (255, 0, 0), 2)
        
        # Process landmarks if available
        if pose_results.pose_landmarks:
            # Adjust landmarks to global coordinates
            for landmark in pose_results.pose_landmarks.landmark:
                landmark.x = (landmark.x * w + box_x) / width
                landmark.y = (landmark.y * h + box_y) / height
                
            # Draw landmarks with thinner lines for better performance
            mp_drawing.draw_landmarks(
                output_frame, 
                pose_results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
            )
            person_data["pose_landmarks"] = pose_results.pose_landmarks
            
        if hand_results.multi_hand_landmarks:
            # Adjust and draw hand landmarks
            for hand_landmarks in hand_results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    landmark.x = (landmark.x * w + box_x) / width
                    landmark.y = (landmark.y * h + box_y) / height
                
                # Draw hand landmarks with thinner lines
                mp_drawing.draw_landmarks(
                    output_frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
                )
            person_data["hand_landmarks"] = hand_results.multi_hand_landmarks[0]

        # Violence detection if model is provided
        if (len(frame_buffer) >= 2 and pose_results.pose_landmarks and 
            hand_results.multi_hand_landmarks and model is not None):
            
            action, confidence = detect_violence(
                pose_results.pose_landmarks, 
                hand_results.multi_hand_landmarks[0], 
                frame_buffer,
                model
            )
            
            person_data["action"] = action
            person_data["confidence"] = confidence
            
            # Add detection label to the frame
            label = f"Person {i+1}: {action} ({confidence:.2f})"
            cv2.putText(output_frame, label, (box_x, box_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Highlight violent behavior
            if action != "Non-Violent" and confidence > 0.6:
                cv2.rectangle(output_frame, (box_x, box_y), (box_x + w, box_y + h), (0, 0, 255), 3)
                # Use a separate thread for beep to prevent UI lag
                threading.Thread(target=winsound.Beep, args=(1000, 100)).start()
            
        people_data.append(person_data)

    # Analyze interactions between people (only if model is available)
    if model is not None:
        interaction_threshold = 0.15  # Distance threshold
        for i, person1 in enumerate(people_data):
            for j, person2 in enumerate(people_data):
                if i >= j:  # Skip self and duplicates
                    continue
                    
                # Calculate distance between people
                distance = np.sqrt(
                    (person1["center"][0] - person2["center"][0])**2 + 
                    (person1["center"][1] - person2["center"][1])**2
                ) / max(width, height)  # Normalize by frame size
                
                # Check if people are interacting
                if distance < interaction_threshold:
                    # Draw interaction line
                    cv2.line(output_frame, 
                            (int(person1["center"][0]), int(person1["center"][1])), 
                            (int(person2["center"][0]), int(person2["center"][1])), 
                            (255, 255, 0), 2)
                    
                    # Check if any of the interacting people are violent
                    if (person1["action"] != "Non-Violent" or person2["action"] != "Non-Violent") and \
                       (person1["confidence"] > 0.6 or person2["confidence"] > 0.6):
                        
                        interaction_label = f"Interaction: {person1['action']} - {person2['action']}"
                        mid_x = int((person1["center"][0] + person2["center"][0]) / 2)
                        mid_y = int((person1["center"][1] + person2["center"][1]) / 2)
                        cv2.putText(output_frame, interaction_label, (mid_x, mid_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        # Beep in separate thread
                        threading.Thread(target=winsound.Beep, args=(2000, 200)).start()

    # Add FPS counter and person count to the frame
    fps = 1 / (time.time() - start_time)
    cv2.putText(output_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(output_frame, f"People: {len(indices)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if model is None:
        cv2.putText(output_frame, "Training required - No detection active", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Update global variables
    processed_frame = output_frame
    people_data_global = people_data
    fps_global = fps
    frame_processing_active = False

# Multi-person live training with optimizations and error handling
def multi_person_live_training(window_size=8, min_samples_per_class=40):
    global frame_processing_active, processed_frame, people_data_global, fps_global
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return [], [], None
    
    # Try to set higher resolution but prioritize FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    X, labels = [], []  # Renamed y to labels to avoid confusion
    current_label = None
    selected_person_idx = 0
    label_map = {ord('0'): 0, ord('1'): 1, ord('2'): 2, ord('3'): 3}
    action_map = {0: "Non-Violent", 1: "Grappling", 2: "Punching", 3: "Weapon Attack"}
    samples_per_class = {0: 0, 1: 0, 2: 0, 3: 0}
    
    print("\n===== MULTI-PERSON VIOLENCE DETECTION - TRAINING MODE =====")
    print("\nCommands:")
    print("  0-3: Set label (0=Non-Violent, 1=Grappling, 2=Punching, 3=Weapon Attack)")
    print("  p: Select next person")
    print("  t: Train model (requires minimum samples)")
    print("  q: Quit")
    print("\nCollect at least", min_samples_per_class, "samples per action class\n")
    
    last_capture_time = time.time()
    capture_interval = 0.1  # Seconds between sample captures
    
    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
            
        # Only process frame eyebrows not already processing one
        if not frame_processing_active:
            frame_processing_active = True
            
            # Process frame in background thread
            processing_thread = threading.Thread(
                target=process_frame_thread, 
                args=(frame.copy(), None)
            )
            processing_thread.daemon = True
            processing_thread.start()
        
        # Use processed frame if available
        if processed_frame is not None:
            display_frame = processed_frame.copy()
            people_data = people_data_global.copy()
            
            # Add training-specific UI elements
            label_text = f"Current Label: {action_map.get(current_label, 'None')}"
            cv2.putText(display_frame, label_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Selected: Person {selected_person_idx + 1}", (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Display samples collected
            sample_text = ", ".join([f"{action_map[k]}: {v}" for k, v in samples_per_class.items()])
            cv2.putText(display_frame, f"Samples: {sample_text}", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Highlight selected person
            if people_data and selected_person_idx < len(people_data):
                person = people_data[selected_person_idx]
                box_x, box_y, w, h = person["box"]
                cv2.rectangle(display_frame, (box_x, box_y), (box_x + w, box_y + h), (0, 255, 0), 3)  # Green for selected
                
                # Collect samples if label is set and enough time passed
                if (current_label is not None and 
                    person["pose_landmarks"] and 
                    person["hand_landmarks"] and
                    time.time() - last_capture_time > capture_interval):
                    
                    features = extract_features(person["pose_landmarks"], person["hand_landmarks"])
                    X.append(features)
                    # Ensure labels is a list before appending
                    if not isinstance(labels, list):
                        print(f"Error: labels is not a list, it is {type(labels)} with value {labels}")
                        labels = []  # Reinitialize as list if corrupted
                    labels.append(current_label)
                    samples_per_class[current_label] += 1
                    last_capture_time = time.time()
                    print(f"Collected sample for {action_map[current_label]}, Total: {samples_per_class[current_label]}")
            
            # Show the frame
            cv2.imshow("Violence Detection - Training Mode", display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key in label_map:
            current_label = label_map[key]
            print(f"Set label to {action_map[current_label]}")
        elif key == ord('p'):
            if people_data:
                selected_person_idx = (selected_person_idx + 1) % len(people_data)
                print(f"Selected Person {selected_person_idx + 1}")
        elif key == ord('t'):
            if all(count >= min_samples_per_class for count in samples_per_class.values()):
                print("\nTraining model... Please wait.")
                try:
                    model = train_model(np.array(X), np.array(labels))
                    print("Training complete. You can now switch to detection mode.")
                    
                    # Ask if user wants to continue to detection
                    cv2.putText(display_frame, "Training complete! Press 'd' to start detection or 'q' to quit", 
                              (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow("Violence Detection - Training Mode", display_frame)
                    
                    while True:
                        key = cv2.waitKey(0) & 0xFF
                        if key == ord('d'):
                            cap.release()
                            cv2.destroyAllWindows()
                            detect_multiple_people(source=0, use_trained_model=True, model=model)
                            return np.array(X), np.array(labels), model
                        elif key == ord('q'):
                            break
                    break
                except ValueError as e:
                    print(f"Training failed: {e}")
            else:
                print(f"Need at least {min_samples_per_class} samples per class. Current: {samples_per_class}")
        elif key == ord('q'):
            print("Quitting training mode.")
            break

    cap.release()
    cv2.destroyAllWindows()
    try:
        if 'pose' in locals() and pose is not None:
            pose.close()
        if 'hands' in locals() and hands is not None:
            hands.close()
    except Exception as e:
        print(f"Warning: Error during cleanup - {e}")
    
    return np.array(X), np.array(labels), None

# Detection function with optimizations
def detect_multiple_people(source=0, use_trained_model=False, model=None):
    global frame_processing_active, processed_frame, people_data_global, fps_global
    
    if use_trained_model and model is None:
        # Try to load model if not provided
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "live_trained_model.pkl")
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            print(f"Loaded trained model from {model_path}")
            use_trained_model = True
        except FileNotFoundError:
            print(f"Model not found at {model_path}. Please train the model first.")
            return
        except Exception as e:
            print(f"Error loading model: {e}. Please train the model first.")
            return
    
    if not use_trained_model or model is None:
        print("Detection requires a trained model. Please run training first.")
        return
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return
    
    # Try to set optimal parameters for video capture
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("\n===== MULTI-PERSON VIOLENCE DETECTION - DETECTION MODE =====")
    print("\nPress 'q' to quit")
    print("Using trained model for violence detection")
    
    # Initialize the frame processing state
    frame_processing_active = False
    
    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("End of video or error.")
            break
            
        # Only process new frame if not already processing one
        if not frame_processing_active:
            frame_processing_active = True
            
            # Process frame in background thread
            processing_thread = threading.Thread(
                target=process_frame_thread, 
                args=(frame.copy(), model)
            )
            processing_thread.daemon = True
            processing_thread.start()
        
        # Display the processed frame if available
        if processed_frame is not None:
            cv2.imshow("Multi-Person Violence Detection", processed_frame)
        
        # Check for quit command
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting detection mode.")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    try:
        if 'pose' in locals() and pose is not None:
            pose.close()
        if 'hands' in locals() and hands is not None:
            hands.close()
    except Exception as e:
        print(f"Warning: Error during cleanup - {e}")

# Main execution with clearer workflow
if __name__ == "__main__":
    print("\n===== MULTI-PERSON VIOLENCE DETECTION SYSTEM =====")
    print("\nThis system requires training before detection can be used.")
    mode = input("\nEnter mode (train/detect): ").lower()

    if mode == "train":
        X, labels, model = multi_person_live_training(window_size=8, min_samples_per_class=40)
    elif mode == "detect":
        # Try to load an existing model
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "live_trained_model.pkl")
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            print(f"Loaded trained model from {model_path}")
            detect_multiple_people(source=0, use_trained_model=True, model=model)
        except FileNotFoundError:
            print("\nNo trained model found. You must train the model first.")
            train_first = input("Would you like to train the model now? (yes/no): ").lower()
            if train_first == "yes":
                X, labels, model = multi_person_live_training(window_size=8, min_samples_per_class=40)
                if model is not None:
                    detect_multiple_people(source=0, use_trained_model=True, model=model)
        except Exception as e:
            print(f"Error loading model: {e}. Please train the model first.")
    else:
        print("Invalid mode. Use 'train' or 'detect'.")

    # Cleanup
    try:
        if 'pose' in locals() and pose is not None:
            pose.close()
        if 'hands' in locals() and hands is not None:
            hands.close()
    except Exception as e:
        print(f"Warning: Error during cleanup - {e}")
