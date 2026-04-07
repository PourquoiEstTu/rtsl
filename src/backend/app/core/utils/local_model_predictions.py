import json 
import cv2
import mediapipe as mp
import numpy as np
import torch

# import from local files
from tgcn_model import GCN_muti_att
from configs import Config
from pose_extractor import PoseExtractor
from pathlib import Path

import csv

DIR = Path(__file__).parent
NUM_CLASSES = 10
INPUT_SIZE = 55
WINDOW_SIZE = 50
MOVEMENT_THRESHOLD = 1.2
PAUSE_THRESHOLD = 20 # number of frames

def get_model():
    checkpoint_path = DIR / "94_0.8346.pth"
    config_path = DIR / "config2.ini"
    config = Config(config_path)
    
    # initalize model
    model = GCN_muti_att(
        input_feature=config.num_samples * 2,  # 50 * 2 = 100
        hidden_feature=config.hidden_size,      # 256
        num_class= NUM_CLASSES,
        p_dropout=config.drop_p,               # 0.3
        num_stage=config.num_stages            # 24
    )

    # load weights
    checkpoint = torch.load(checkpoint_path, map_location='cpu', )
    state_dict = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval() # put in evaluation mode

    return model

def generate_class_integer_mappings(json_path, max_classes=None):
    with open(json_path, "r") as f:
        data = json.load(f)

    # extract gloss names
    all_glosses = sorted([entry["gloss"] for entry in data])

    if max_classes is not None:
        glosses = all_glosses[:max_classes]
    else:
        glosses = all_glosses

    class_to_idx = {g: i for i, g in enumerate(glosses)}
    idx_to_class = {i: g for i, g in enumerate(glosses)}

    return idx_to_class, class_to_idx

def get_labels():
    return sorted(["YES", "NO", "HELP", "EAT", "DRINK", "WANT", "FINISH", "GO", "WHAT", "WHO"])

    dataset_json = "asl_citizen/asl_citizens100.json"   # or whatever your dataset file is
    idx_to_class, class_to_idx = generate_class_integer_mappings(
        dataset_json,
        max_classes=NUM_CLASSES
    )
    labels = [idx_to_class[i] for i in range(len(idx_to_class))]
    return labels
     
def predict(model, labels, seq):
    logits = model(seq)
    probabilities = torch.softmax(logits, dim=1)

    idx = torch.argmax(probabilities, dim=1).item()
    confidence = float(probabilities[0, idx])

    return labels[idx], confidence

def movement_score(landmarks_frames,hand_weight=3):
    hand_indices = list(range(13))  # adjust based on your hand landmark indices
    diffs = []
    for i in range(len(landmarks_frames) - 1):
        diff = np.linalg.norm(landmarks_frames[i+1] - landmarks_frames[i], axis=1)  # shape (num_landmarks,)

        # Apply 2x weight to hand landmarks TODO: maybe only put weight on the hands?
        weighted_diff = diff.copy()
        weighted_diff[hand_indices] *= hand_weight

        total_diff = np.sum(weighted_diff)
        diffs.append(total_diff)

    avg_movement = np.mean(diffs)
    return avg_movement   

def update_ema(old_ema, new_score, alpha=0.3):
    if old_ema:
        return alpha * new_score + (1 - alpha) * old_ema
    else:
        return new_score
        
def add_to_sequence(gloss_arr, last_pred, word):
    global counter 
        
    if (last_pred == word):
        counter += 1
    else:
        counter = 1
        
    if (counter > 3 and gloss_arr[-1] != word):            
        gloss_arr.append(word)
        
def main():
    model = get_model()
    labels = get_labels()

    pose_extractor = PoseExtractor()
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(0)
    processed = []
    ema_score = None
    counter = 0 #can delete this later
    
    pause_counter = 0
    last_pred = None
    gloss_sequence = [""]
    
    while True:
        # with open('data.csv', 'w', newline='') as f:
        #     writer = csv.writer(f)
        #     writer.writerows(processed)
        ret, frame = cap.read()
        if not ret:
            break

        print(len(processed))
        
        # flip camera
        frame = cv2.flip(frame, 1)
        result = pose_extractor.extract_keypoints(frame)["people"][0]    
    
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
        # landmarks = []
                
        if result["multi_hand_landmarks"]:
            for handLms in result["multi_hand_landmarks"]:
                mp_drawing.draw_landmarks(
                    frame, handLms, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )
                # h, w, _ = frame.shape
                # for lm in handLms.landmark:
                #     landmarks.extend([lm.x, lm.y, lm.z])
        # print(len(landmarks)) # 0, 63 or 126
        
        body = result["pose_keypoints_2d"]
        left = result["hand_left_keypoints_2d"]
        right = result["hand_right_keypoints_2d"]
        
        # Combine: body (25) + left hand (21) + right hand (21) = 67 total landmarks
        combined = list(body) + list(left) + list(right)
        num_landmarks = len(combined) // 3 if len(combined) >= 3 else 0
        x_list = []
        y_list = []
        
        for j in range(num_landmarks):
            # Skip excluded body keypoints (indices 0-24 are body) 67 - 12 = 55
            if j < 25 and j in {9, 10, 11, 22, 23, 24, 12, 13, 14, 19, 20, 21}:
                continue
            
            xi = combined[j*3 + 0] if j*3 + 0 < len(combined) else 0.0
            yi = combined[j*3 + 1] if j*3 + 1 < len(combined) else 0.0
            
            # Normalize to [-1, 1] range: 2 * ((x / 256) - 0.5)
            x_norm = 2.0 * ((float(xi) / 256.0) - 0.5)
            y_norm = 2.0 * ((float(yi) / 256.0) - 0.5)
            
            x_list.append(x_norm)
            y_list.append(y_norm)
        
        # Should have exactly 55 keypoints: 13 body + 21 left + 21 right
        expected_keypoints = 55
        if len(x_list) != expected_keypoints:
            print(f"[WARNING]: Expected {expected_keypoints} keypoints, got {len(x_list)}")
            # Pad or truncate to 55
            while len(x_list) < expected_keypoints:
                x_list.append(0.0)
                y_list.append(0.0)
            x_list = x_list[:expected_keypoints]
            y_list = y_list[:expected_keypoints]
        
        xy_frame = np.stack([np.array(x_list), np.array(y_list)], axis=1).astype(np.float32)
        
        if (len(processed) > (WINDOW_SIZE - 1)):
            processed = processed[1:WINDOW_SIZE]
        processed.append(xy_frame)
        # save preprocessed frames
        # np.save(DIR / "outputs/preprocessed_frames.npy", np.array(processed))
        
        # Pad to exactly NUM_SAMPLES frames
        if len(processed) < WINDOW_SIZE:
            # Pad with last frame
            last_frame = processed[-1] if processed else np.zeros((INPUT_SIZE, 2), dtype=np.float32)
            num_padding = WINDOW_SIZE - len(processed)
            for _ in range(num_padding):
                processed.append(last_frame.copy())
            print(f"[PREPROCESS] Padded {num_padding} frames to reach {WINDOW_SIZE}")
        
        #at this point, processed is 50 frames x 55 features x 2 coordinates
        ema_score = update_ema(ema_score, movement_score(processed[-5:]))
        if ema_score > MOVEMENT_THRESHOLD:
            pause_counter = 0
            print(f"-- MOVEMENT THRESHOLD PASSED -- {counter}")
            counter+=1 
            # Reshape to model input format: (1, num_nodes, feature_len)
            # feature_len = num_samples * 2 (x,y coordinates across time)
            # Each node has: [x_t0, y_t0, x_t1, y_t1, ..., x_t49, y_t49]
            feature_len = WINDOW_SIZE * 2
            input_data = np.zeros((1, INPUT_SIZE, feature_len), dtype=np.float32)

            for node_idx in range(INPUT_SIZE):
                for t in range(WINDOW_SIZE):
                    frame_xy = processed[t]  # Shape: (55, 2)
                    x_val = frame_xy[node_idx, 0]
                    y_val = frame_xy[node_idx, 1]
                    input_data[0, node_idx, t*2 + 0] = x_val
                    input_data[0, node_idx, t*2 + 1] = y_val
                
            # only print if hands in frame     
            if np.sum(left) + np.sum(right) != 0:
                # current_pred = predict(model, labels, torch.from_numpy(input_data))
                # # print(current_pred)
                # add_to_sequence(gloss_sequence, last_pred, current_pred)
                
                current_pred, confidence = predict(model, labels, torch.from_numpy(input_data))
                # print(f"Predicted: {current_pred}, Confidence: {confidence:.4f}")
                if confidence > 0.5:
                    add_to_sequence(gloss_sequence, last_pred, current_pred)
                    last_pred = current_pred
                last_pred = current_pred
        else:
            pause_counter += 1
            if (pause_counter == PAUSE_THRESHOLD):
                add_to_sequence(gloss_sequence, last_pred, ".")
                last_pred = "."
                pause_counter = 0
                #TODO call gloss to sentence stuff
        
        cv2.imshow("Live Prediction", frame)
        print(gloss_sequence)
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
