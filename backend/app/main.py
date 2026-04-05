from fastapi import FastAPI, WebSocket
import json
import numpy as np
import torch

# import from local files
from core.loader import get_model, get_onnx_model, get_labels
from core.formatter import convert_format_to_55, normalize_x_y_data
from core.predictor import predict, onnx_predict
from core.gloss_to_sentence import Gloss_to_Sentence_Model
from utils.motion_analyzer import movement_score, update_ema

DIR = "/home/sharmg36"
NUM_CLASSES = 100
MOVEMENT_THRESHOLD = 0.8
WINDOW_SIZE = 50
INPUT_SIZE = 55
PAUSE_THRESHOLD = 20  # number of frames
MODEL_TYPE = "ONNX"

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    model = (
        get_onnx_model(DIR, NUM_CLASSES)
        if MODEL_TYPE == "ONNX"
        else get_model(DIR, NUM_CLASSES)
    )
    prediction_function = onnx_predict if MODEL_TYPE == "ONNX" else predict
    get_model(DIR, NUM_CLASSES)
    labels = get_labels(DIR, NUM_CLASSES)
    gloss_to_sentence = Gloss_to_Sentence_Model()

    processed = []
    ema_score = None
    gloss_counter = 0  # TODO: delete this later

    pause_counter = 0
    last_pred = None
    gloss_sequence = [""]

    await websocket.accept()
    while True:
        # print(gloss_sequence)
        received = await websocket.receive_json()

        if "model" in received:
            selected_model = received["model"]
            # Guarav does something here
            continue

        pose_landmarks = []
        if len(received["pose"]["landmarks"]) > 0:
            pose_landmarks = received["pose"]["landmarks"][0]
        converted_format = convert_format_to_55(
            pose_landmarks,
            received["hand"]["landmarks"],
            received["hand"]["handedness"],
        )

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

        xy_frame = np.stack([np.array(x_list), np.array(y_list)], axis=1).astype(
            np.float32
        )

        if len(processed) > (WINDOW_SIZE - 1):
            processed = processed[1:WINDOW_SIZE]
        processed.append(xy_frame)

        # Pad to exactly NUM_SAMPLES frames (occurs at the very beginning) - TODO: replace with interpolation?
        if len(processed) < WINDOW_SIZE:
            # Pad with last frame
            last_frame = (
                processed[-1]
                if processed
                else np.zeros((INPUT_SIZE, 2), dtype=np.float32)
            )
            num_padding = WINDOW_SIZE - len(processed)
            for _ in range(num_padding):
                processed.append(last_frame.copy())
            # print(f"[PREPROCESS] Padded {num_padding} frames to reach {WINDOW_SIZE}")

        ema_score = update_ema(ema_score, movement_score(processed[-5:]))
        if ema_score > MOVEMENT_THRESHOLD:
            pause_counter = 0
            # print("-- MOVEMENT THRESHOLD PASSED --")
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
                    input_data[0, node_idx, t * 2 + 0] = x_val
                    input_data[0, node_idx, t * 2 + 1] = y_val

            # only predict if hands in frame
            if np.sum(left) + np.sum(right) != 0:
                if MODEL_TYPE != "ONNX":
                    input_data = torch.from_numpy(input_data)
                current_pred = prediction_function(model, labels, input_data).upper()

                # predicting same word again,
                if last_pred == current_pred:
                    gloss_counter += 1
                else:
                    gloss_counter = 1

                # if seen the same prediction for multiple frames in a row, add to sequence and send to front_end
                if gloss_counter > 3 and gloss_sequence[-1] != current_pred:
                    gloss_sequence.append(current_pred)
                    await websocket.send_json({"word": current_pred, "sentence": ""})

                last_pred = current_pred
        else:
            pause_counter += 1
            if pause_counter == PAUSE_THRESHOLD:
                sentence = gloss_to_sentence.run_inference(
                    " ".join(gloss_sequence[1:])
                ).rstrip()
                correct_sentence = ""
                # fixes punctuation spacing
                for i in range(len(sentence)):
                    if ord(sentence[i]) != 32:
                        correct_sentence += sentence[i]
                    elif (
                        (96 < ord(sentence[i + 1]) < 123)
                        or (64 < ord(sentence[i + 1]) < 91)
                        or (47 < ord(sentence[i + 1]) < 58)
                    ):
                        correct_sentence += sentence[
                            i
                        ]  # only add space if it's followed by a lowercase or uppercase letter or number
                # print(f"GLOSSES: {gloss_sequence[1:]}")
                # print(f"SENTENCE: {correct_sentence}")

                await websocket.send_json({"word": "", "sentence": correct_sentence})
                last_pred = ""
                gloss_sequence = [""]
                pause_counter = 0
