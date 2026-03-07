import torch
import numpy as np
import os
import json
from sys import exit
from pose_extractor import PoseExtractor
from tgcn_wlasl import generate_class_integer_mappings, _setup_model, _preprocess_keypoints
# print("here")

ROOT = "/u50/quyumr/rtsl/backend/app"

def predict(model, labels, seq):
    logits = model(seq)
    probabilities = torch.softmax(logits, dim=1)
    idx = torch.argmax(probabilities, dim=1).item()
    print(f"idx: {idx}")
    return labels[idx]

# ALL PATHS NEED TO BE ADJUSTED TO YOUR OWN COMPUTER'S PATHS
##########################################################
#                                                        #
# NOT TESTED NOT TESTED NOT TESTED NOT TESTED NOT TESTED #
#                                                        #
##########################################################
def get_metrics():
    extractor = PoseExtractor()
    # idx_to_class, class_to_idx = generate_class_integer_mappings("/u50/quyumr/rtsl/backend/app/splits/300", mappings_exist=False, json_path="/u50/quyumr/rtsl/backend/app/splits/asl300.json")
    model_100 = _setup_model(f"{ROOT}/splits/100/asl100.onnx")
    model_300 = _setup_model(f"{ROOT}/splits/300/asl300.onnx")
    model_1000 = _setup_model(f"{ROOT}/splits/1000/asl1000.onnx")
    model_2000 = _setup_model(f"{ROOT}/splits/2000/asl2000.onnx")

    with open(f"splits/2000/class_to_idx.json", 'r') as f:
        class_to_idx = json.load(f)
    with open(f"splits/2000/idx_to_class.json", 'r') as f:
        idx_to_class = json.load(f) 
    with open(f"splits/asl2000.json", 'r') as f:
        data = json.load(f)

    for entry in data:
        # print(entry["gloss"])
        gloss = entry["gloss"] 
        for instance in entry["instances"]: 
            # print(instance["video_id"])
            video_id = instance["video_id"]
            try :
                print("here")
                frames_data = extractor.extract_from_video(f"/u50/quyumr/archive/videos/{video_id}.mp4")
                # frames_data = extractor.extract_from_video(f"/u50/chandd9/capstone/videos/{video_id}.mp4")
            except FileNotFoundError:
                print(f"{video_id}.mp4 not found")
                continue
            except ValueError:
                print(f"{video_id}.mp4 not found")
                continue
            except Exception as e: 
                raise e
            input_tensor = _preprocess_keypoints(frames_data)
            print(f"Test run input shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")
            # print(frames_data[0].keys())  # should show 'people' key

            # print(pred_idx)
            # word = idx_to_class[pred_idx]
            # print("Prediction:", word)
            # Get logits
            input_300 = model_300.get_inputs()[0].name
            output_300 = model_300.get_outputs()[0].name
            logits_300 = model_300.run([output_300], {input_name: input_300})[0]

            input_1000 = model_1000.get_inputs()[0].name
            output_1000 = model_1000.get_outputs()[0].name
            logits_1000 = model_1000.run([output_1000], {input_name: input_1000})[0]

            # Apply softmax
            exp_logits_300 = np.exp(logits_300 - np.max(logits_300, axis=1, keepdims=True))
            probs_300 = exp_logits_300 / np.sum(exp_logits_300, axis=1, keepdims=True)

            exp_logits_1000 = np.exp(logits_1000 - np.max(logits_1000, axis=1, keepdims=True))
            probs_1000 = exp_logits_1000 / np.sum(exp_logits_1000, axis=1, keepdims=True)
                
            # Get top prediction
            predicted_idx_300 = int(np.argmax(probs_300, axis=1)[0])
            confidence_300 = float(probs[0, predicted_idx_300])
            predicted_idx_1000 = int(np.argmax(probs_1000, axis=1)[0])
            confidence_1000 = float(probs[0, predicted_idx_1000])

            print(f"Predicted class index: {predicted_idx_300}, confidence: {confidence_300:.4f}")
            predicted_label_300 = idx_to_class.get(str(predicted_idx_300), "Unknown")
            print(f"Predicted label: {predicted_label_300}")

            print(f"Predicted class index: {predicted_idx_1000}, confidence: {confidence_1000:.4f}")
            predicted_label_1000 = idx_to_class.get(str(predicted_idx_1000), "Unknown")
            print(f"Predicted label: {predicted_label_1000}")



get_metrics()
