from fastapi import FastAPI, WebSocket
import json
import numpy as np

# import from local files
from core.loader import get_model, get_labels
from core.formatter import convert_format_to_55, normalize_x_y_data
from core.predictor import predict
from utils.motion_analyzer import movement_score, update_ema 

DIR = "/home/sharmg36"
NUM_CLASSES = 300
MOVEMENT_THRESHOLD = 1.2
WINDOW_SIZE = 50
INPUT_SIZE = 55
PAUSE_THRESHOLD = 20 # number of frames

app = FastAPI()

# TODO: delete this once gloss_to_sentence is finished
def add_to_sequence(gloss_counter, gloss_arr, last_pred, word):        
    if (last_pred == word):
        gloss_counter += 1
    else:
        gloss_counter = 1
        
    if (gloss_counter > 3 and gloss_arr[-1] != word):            
        gloss_arr.append(word)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    model = get_model(DIR, NUM_CLASSES)
    labels = get_labels(DIR, NUM_CLASSES)
    
    processed = []
    ema_score = None
    gloss_counter = 0 #TODO: delete this later
    
    pause_counter = 0
    last_pred = None
    gloss_sequence = [""]
    
    await websocket.accept()
    while True:
        received = await websocket.receive_json()
        
        pose_landmarks = []
        if len(received["pose"]["landmarks"]) > 0:
            pose_landmarks = received["pose"]["landmarks"][0]
        converted_format = convert_format_to_55(pose_landmarks, received["hand"]["landmarks"], received["hand"]["handedness"])
        
        body = converted_format[0]
        left = converted_format[1]
        right = converted_format[2]
        
        combined = list(body) + list(left) + list(right)
        x_list, y_list = normalize_x_y_data(combined)
        
        # Should have exactly 55 keypoints: 13 body + 21 left + 21 right
        expected_keypoints = 55
        if len(x_list) != expected_keypoints:
            # print(f"[WARNING]: Expected {expected_keypoints} keypoints, got {len(x_list)}")
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
        
        # Pad to exactly NUM_SAMPLES frames (occurs at the very beginning) - TODO: replace with interpolation?
        if len(processed) < WINDOW_SIZE:
            # Pad with last frame
            last_frame = processed[-1] if processed else np.zeros((INPUT_SIZE, 2), dtype=np.float32)
            num_padding = WINDOW_SIZE - len(processed)
            for _ in range(num_padding):
                processed.append(last_frame.copy())
            # print(f"[PREPROCESS] Padded {num_padding} frames to reach {WINDOW_SIZE}")
         
        ema_score = update_ema(ema_score, movement_score(processed[-5:]))
        if ema_score > MOVEMENT_THRESHOLD:
            pause_counter = 0
            print("-- MOVEMENT THRESHOLD PASSED --")
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
                current_pred = predict(model, labels, torch.from_numpy(input_data))
                # print(current_pred)
                add_to_sequence(gloss_counter, gloss_sequence, last_pred, current_pred)
                last_pred = current_pred
        else:
            pause_counter += 1
            if (pause_counter == PAUSE_THRESHOLD):
                add_to_sequence(gloss_counter, gloss_sequence, last_pred, ".")
                last_pred = "."
                pause_counter = 0
                #TODO call gloss to sentence stuff
