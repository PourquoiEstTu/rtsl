import json 
import cv2
import mediapipe as mp
import numpy as np
import torch

# import from local files
from tgcn_model import GCN_muti_att
from configs import Config
from pose_extractor import PoseExtractor

DIR = "/Users/gauravsharma/Documents/capstone"
NUM_CLASSES = 1500
INPUT_SIZE = 55
WINDOW_SIZE = 50

if NUM_CLASSES not in [100, 300, 1000, 2000]:
    print("ERROR - INVALID NUM_CLASSES")
    exit()

def get_model():
    checkpoint_path = f"{DIR}/rtsl/backend/models/checkpoints/asl{NUM_CLASSES}/pytorch_model.bin"
    config_path = f"{DIR}/rtsl/backend/models/configs/asl{NUM_CLASSES}.ini"
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

def get_labels():
    labels_path = f"{DIR}/rtsl/backend/data_splits/{NUM_CLASSES}/idx_to_class.json"
    with open(labels_path, 'r') as f:
        labels = [w for w in json.load(f)]
    return labels
     
def predict(model, labels, seq):
    logits = model(seq)
    probabilities = torch.softmax(logits, dim=1)
    idx = torch.argmax(probabilities, dim=1).item()
    return labels[idx]


def main():
    model = get_model()
    labels = get_labels()
    
    pose_extractor = PoseExtractor()
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(0)
    buffer = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # flip camera
        frame = cv2.flip(frame, 1)
        result = pose_extractor.extract_keypoints(frame)
        
        landmarks = []
        buffer.append(landmarks)
        
        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, handLms, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )
                h, w, _ = frame.shape
                for lm in handLms.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                   
        #TODO: doublecheck how long landmarks usually is 
        # pad if needed
        if len(landmarks) < INPUT_SIZE:
            landmarks += [0.0] * (INPUT_SIZE - len(landmarks))
        elif len(landmarks) > INPUT_SIZE:
            landmarks = landmarks[:INPUT_SIZE]

        landmarks = np.array(landmarks)
        
        # Only predict if **hands are present**
        if np.sum(landmarks) != 0 and len(buffer) >= WINDOW_SIZE:
            seq = np.stack(np.stack(buffer[-WINDOW_SIZE:]))
            current_pred = predict(model, labels, seq)
        elif np.sum(landmarks) == 0:
            current_pred = ""  # no hands detected
            
        print(current_pred)
        
        cv2.imshow("Live Prediction", frame)
