import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
import math

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_features_and_draw(image, results, frame_index, fps):
    features = {
        'timestamp': frame_index / fps,
        'is_looking_away': False,
        'is_touching_face': False,
        'is_slouching': False
    }
    
    if not results.pose_landmarks:
        return features, image

    h, w, _ = image.shape
    landmarks = results.pose_landmarks.landmark
    
    nose = landmarks[mp_holistic.PoseLandmark.NOSE.value]
    left_ear = landmarks[mp_holistic.PoseLandmark.LEFT_EAR.value]
    right_ear = landmarks[mp_holistic.PoseLandmark.RIGHT_EAR.value]
    
    # 1. Gaze Deviation (Head Turn)
    dist_l = abs(nose.x - left_ear.x)
    dist_r = abs(nose.x - right_ear.x)
    ratio = max(dist_l, dist_r) / (min(dist_l, dist_r) + 1e-5)
    
    if ratio > 2.5: # Head turned significantly
        features['is_looking_away'] = True
        cv2.putText(image, "Looking Away!", (int(nose.x * w) - 100, int(nose.y * h) - 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # 2. Hand to Face
    left_wrist = landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value]
    right_wrist = landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value]
    left_index = landmarks[mp_holistic.PoseLandmark.LEFT_INDEX.value]
    right_index = landmarks[mp_holistic.PoseLandmark.RIGHT_INDEX.value]
    
    def dist(p1, p2):
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
        
    dists = []
    if left_wrist.visibility > 0.5: dists.append(dist(nose, left_wrist))
    if right_wrist.visibility > 0.5: dists.append(dist(nose, right_wrist))
    if left_index.visibility > 0.5: dists.append(dist(nose, left_index))
    if right_index.visibility > 0.5: dists.append(dist(nose, right_index))
    
    min_dist = min(dists) if dists else 1.0
    
    if min_dist < 0.15: # Hand very close to nose
        features['is_touching_face'] = True
        cv2.putText(image, "Touching Face!", (int(nose.x * w) + 20, int(nose.y * h) + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2, cv2.LINE_AA)
                    
    # 3. Slouching (Shoulder tilt)
    left_shoulder = landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value]
    dy = left_shoulder.y - right_shoulder.y
    dx = left_shoulder.x - right_shoulder.x
    angle = abs(math.atan2(dy, dx))
    
    if angle > 0.2: # Shoulders tilted
        features['is_slouching'] = True
        cv2.putText(image, "Slouching!", (int((left_shoulder.x + right_shoulder.x)/2 * w) - 50, int(left_shoulder.y * h) - 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    return features, image

def record_and_analyze(duration_sec=30, output_filename="recorded_video.mp4", st_placeholder=None):
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps > 60: fps = 30 # default
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
    
    features_list = []
    start_time = time.time()
    frame_idx = 0
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if time.time() - start_time > duration_sec:
                break
                
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Draw landmarks faintly
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                      mp_drawing.DrawingSpec(color=(245,117,66), thickness=1, circle_radius=1),
                                      mp_drawing.DrawingSpec(color=(245,66,230), thickness=1, circle_radius=1))
            
            feats, annotated_image = extract_features_and_draw(image, results, frame_idx, fps)
            features_list.append(feats)
            
            out.write(annotated_image)
            
            if st_placeholder:
                st_placeholder.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), channels="RGB")
                
            frame_idx += 1
            
    cap.release()
    out.release()
    
    return pd.DataFrame(features_list)
