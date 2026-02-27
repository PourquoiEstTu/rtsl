import torch
from tgcn_model import GCN_muti_att
from configs import Config
import json
import onnxruntime as ort
import os
# from transformers import AutoModel, AutoModelForSequenceClassification, PreTrainedModel
import numpy as np
from pose_extractor import PoseExtractor
from sys import exit

PRETRAINED_MODEL = 0
NUM_SAMPLES = 0
ROOT = "/u50/quyumr/rtsl/backend/app"

def generate_class_integer_mappings(directory: str, mappings_exist: bool, json_path: str, max_classes = None) :
    """
    Generate integer -> class and class -> integer mapping and load them.
    If mappings already exist, they should not be re-generated
    directory: is the output directory if mappings exist; is the input 
        directory when mappings do exist
    """
    if mappings_exist :
        with open(f"{directory}/idx_to_class.json", 'r') as f:
            idx_to_class = json.load(f)
            print("idx_to_class loaded from disk")
        with open(f"{directory}/class_to_idx.json", 'r') as f:
            class_to_idx = json.load(f)
            print("class_to_idx loaded from disk")
    else :
        with open(json_path, "r") as f:
            data = json.load(f)

        # get glosses and filter them down if desired
        all_glosses = sorted([entry["gloss"] for entry in data])
        if max_classes is not None:
            glosses = all_glosses[:max_classes]  # only keep the first max_classes glosses
        else:
            glosses = all_glosses

        class_to_idx = {g: i for i, g in enumerate(glosses)}
        with open(f"{directory}/class_to_idx.json", 'w') as f :
            json.dump(class_to_idx, f, indent=2)
        print("{class: idx} dictionary created")

        idx_to_class = {i: g for i, g in enumerate(glosses)}
        with open(f"{directory}/idx_to_class.json", 'w') as f :
            json.dump(idx_to_class, f, indent=2)
        print("{idx: class} dictionary created")

    return idx_to_class, class_to_idx

idx_to_class, class_to_idx = generate_class_integer_mappings("/u50/quyumr/rtsl/backend/app/splits/2000", mappings_exist=False, json_path="/u50/quyumr/rtsl/backend/app/splits/asl2000.json")
# print(idx_to_class[1845])
# exit()

# Load weights
# checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
# state_dict = checkpoint.get('state_dict', checkpoint)
# model.load_state_dict(state_dict, strict=False)
# model.eval()


def _preprocess_keypoints(frames_data):
    """
    Preprocess keypoint frames for model input.
    Matches Sign_Dataset.read_pose_file processing exactly.
    
    Training preprocessing:
    1. Exclude body keypoints: {9, 10, 11, 22, 23, 24, 12, 13, 14, 19, 20, 21}
    2. Normalize: 2 * ((x / 256.0) - 0.5) to get [-1, 1] range
    3. Result: 55 keypoints (13 body + 21 left hand + 21 right hand)
    4. Format: (55, num_samples*2) where features are [x1,y1,x2,y2,...] across time
    """
    body_pose_exclude = {9, 10, 11, 22, 23, 24, 12, 13, 14, 19, 20, 21}
    processed = []
    
    # Reduced logging for speed - only log if processing many frames
    if len(frames_data) > 30:
        print(f"[PREPROCESS] Processing {len(frames_data)} frames, target: {NUM_SAMPLES} frames")
    
    for frame_idx, frame_data in enumerate(frames_data):
        people = frame_data.get('people', [])
        if not people:
            # Empty frame: fill with zeros (55 keypoints * 2 = 110 values)
            processed.append(np.zeros((NUM_NODES, 2), dtype=np.float32))
            continue
        
        p = people[0]
        body = p.get('pose_keypoints_2d', [])
        left = p.get('hand_left_keypoints_2d', [])
        right = p.get('hand_right_keypoints_2d', [])
        
        # Combine: body (25) + left hand (21) + right hand (21) = 67 total landmarks
        combined = list(body) + list(left) + list(right)
        
        # Extract x, y coordinates, excluding specified body keypoints
        # Format: [x, y, confidence, x, y, confidence, ...]
        num_landmarks = len(combined) // 3 if len(combined) >= 3 else 0
        x_list = []
        y_list = []
        
        for j in range(num_landmarks):
            # Skip excluded body keypoints (indices 0-24 are body)
            if j < 25 and j in body_pose_exclude:
                continue
            
            xi = combined[j*3 + 0] if j*3 + 0 < len(combined) else 0.0
            yi = combined[j*3 + 1] if j*3 + 1 < len(combined) else 0.0
            
            # Normalize to [-1, 1] range: 2 * ((x / 256) - 0.5)
            x_norm = 2.0 * ((float(xi) / 256.0) - 0.5)
            y_norm = 2.0 * ((float(yi) / 256.0) - 0.5)
            
            x_list.append(x_norm)
            y_list.append(y_norm)
        
        # Should have exactly 55 keypoints: 13 body + 21 left + 21 right
        expected_keypoints = NUM_NODES
        if len(x_list) != expected_keypoints:
            print(f"[WARNING] Frame {frame_idx}: Expected {expected_keypoints} keypoints, got {len(x_list)}")
            # Pad or truncate to 55
            while len(x_list) < expected_keypoints:
                x_list.append(0.0)
                y_list.append(0.0)
            x_list = x_list[:expected_keypoints]
            y_list = y_list[:expected_keypoints]
        
        # Stack as (55, 2) - 55 keypoints with x,y coordinates
        xy_frame = np.stack([np.array(x_list), np.array(y_list)], axis=1).astype(np.float32)
        processed.append(xy_frame)
    
    print(f"[PREPROCESS] Extracted {len(processed)} frames with keypoints")
    
    # Pad or sample to exactly NUM_SAMPLES frames
    if len(processed) < NUM_SAMPLES:
        # Pad with last frame
        last_frame = processed[-1] if processed else np.zeros((NUM_NODES, 2), dtype=np.float32)
        num_padding = NUM_SAMPLES - len(processed)
        for _ in range(num_padding):
            processed.append(last_frame.copy())
        print(f"[PREPROCESS] Padded {num_padding} frames to reach {NUM_SAMPLES}")
    elif len(processed) > NUM_SAMPLES:
        # Uniformly sample NUM_SAMPLES frames
        original_count = len(processed)
        indices = np.linspace(0, len(processed) - 1, NUM_SAMPLES).astype(int)
        processed = [processed[i] for i in indices]
        print(f"[PREPROCESS] Sampled {NUM_SAMPLES} frames from {original_count}")
    
    # Reshape to model input format: (1, num_nodes, feature_len)
    # feature_len = num_samples * 2 (x,y coordinates across time)
    # Each node has: [x_t0, y_t0, x_t1, y_t1, ..., x_t49, y_t49]
    feature_len = NUM_SAMPLES * 2
    input_data = np.zeros((1, NUM_NODES, feature_len), dtype=np.float32)
    
    for node_idx in range(NUM_NODES):
        for t in range(NUM_SAMPLES):
            frame_xy = processed[t]  # Shape: (55, 2)
            x_val = frame_xy[node_idx, 0]
            y_val = frame_xy[node_idx, 1]
            input_data[0, node_idx, t*2 + 0] = x_val
            input_data[0, node_idx, t*2 + 1] = y_val
    
    # Reduced logging for speed
    # print(f"[PREPROCESS] Final input shape: {input_data.shape}")
    # print(f"[PREPROCESS] Input range: x=[{input_data[0, :, ::2].min():.3f}, {input_data[0, :, ::2].max():.3f}], "
    #       f"y=[{input_data[0, :, 1::2].min():.3f}, {input_data[0, :, 1::2].max():.3f}]")
    
    return input_data

# temp implementation
MODEL_LOAD_INFO = {
    "loaded": False,
    "error": None,
    "details": None
}

def _setup_model(onnx_path):
    """Load the ONNX model for inference."""
    
    if ort is None:
        print('ONNX Runtime not available')
        MODEL_LOAD_INFO['error'] = 'ONNX Runtime not installed'
        return
    if not os.path.exists(onnx_path):
        print(f'ONNX model not found at {onnx_path}')
        MODEL_LOAD_INFO['error'] = f'ONNX model not found. Please run convert_to_onnx.py first.'
        return
    
    try:
        providers = ['CPUExecutionProvider']
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        pretrained_model = ort.InferenceSession(onnx_path, sess_options=session_options, providers=providers)
        
        # Get input shape
        input_shape = pretrained_model.get_inputs()[0].shape
        if len(input_shape) >= 3:
            feature_len = input_shape[2] if input_shape[2] > 0 else 100
            NUM_SAMPLES = feature_len // 2
        
        MODEL_LOAD_INFO['loaded'] = True
        MODEL_LOAD_INFO['details'] = f'Loaded ONNX model from {onnx_path}'
        print(f'ONNX model loaded: {input_shape}, NUM_SAMPLES={NUM_SAMPLES}')
    except Exception as e:
        print(f'Failed loading ONNX model: {e}')
        import traceback
        traceback.print_exc()
        MODEL_LOAD_INFO['loaded'] = False
        MODEL_LOAD_INFO['error'] = str(e)
        pretrained_model = None
    return pretrained_model, NUM_SAMPLES
PRETRAINED_MODEL, NUM_SAMPLES = _setup_model(f"{ROOT}/splits/1000/asl1000.onnx")

NUM_NODES = 55

# dhruv test run of pose extractor and setupmodel
def test_run():
    extractor = PoseExtractor()
    frames_data = extractor.extract_from_video("/u50/quyumr/archive/videos/00639.mp4")
    input_tensor = _preprocess_keypoints(frames_data)
    print(f"Test run input shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")

    inputs = {PRETRAINED_MODEL.get_inputs()[0].name: input_tensor}
    outputs = PRETRAINED_MODEL.run(None, inputs)

    logits = outputs[0]
    pred_idx = int(np.argmax(logits))
    # c = float(np.max(logits))

    # print(pred_idx)
    # print(idx_to_class[1845])
    word = idx_to_class[pred_idx]
    print("Prediction:", word)

    return 

test_run()


# checkpoint_path = "/home/pourquoi/repos/rtsl/backend/app/checkpoints/asl1000/pytorch_model.bin"
# config_path =     "/home/pourquoi/repos/rtsl/backend/app/checkpoints/asl1000/config.ini"
# config = Config(config_path)
# 
# # initialize model
# model = GCN_muti_att(
#     input_feature=config.num_samples * 2,  # 50 * 2 = 100
#     hidden_feature=config.hidden_size,      # 256
#     num_class=2000,
#     p_dropout=config.drop_p,               # 0.3
#     num_stage=config.num_stages            # 24
# )
# 
# # Load weights
# checkpoint = torch.load(checkpoint_path, map_location='cpu', )
# state_dict = checkpoint.get('state_dict', checkpoint)
# model.load_state_dict(state_dict, strict=False)
# model.eval()
