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
    model_100 = _setup_model(f"{ROOT}/splits/100/asl100.onnx")[0]
    model_300 = _setup_model(f"{ROOT}/splits/300/asl300.onnx")[0]
    model_1000 = _setup_model(f"{ROOT}/splits/1000/asl1000.onnx")[0]
    model_2000 = _setup_model(f"{ROOT}/splits/2000/asl2000.onnx")[0]

    with open(f"splits/2000/class_to_idx.json", 'r') as f:
        class_to_idx = json.load(f)
    with open(f"splits/2000/idx_to_class.json", 'r') as f:
        idx_to_class = json.load(f) 
    with open(f"splits/asl2000.json", 'r') as f:
        data = json.load(f)

    gloss_count = 0
    correct_predictions = {100:0, 300:0, 1000:0, 2000:0}
    total_predictions = 0
    for entry in data:
        # print(entry["gloss"])
        gloss = entry["gloss"] 
        print(f"Actual label: {gloss}")
        # print(gloss)
        gloss_idx = class_to_idx[gloss]
        # print(gloss_idx)
        # exit(0)
        for instance in entry["instances"]: 
            # print(instance["video_id"])
            video_id = instance["video_id"]
            try :
                frames_data = extractor.extract_from_video(f"/u50/quyumr/archive/videos/{video_id}.mp4")
                # frames_data = extractor.extract_from_video(f"/u50/chandd9/capstone/videos/{video_id}.mp4")
            except FileNotFoundError:
                # print(f"{video_id}.mp4 not found")
                continue
            except ValueError:
                # print(f"{video_id}.mp4 not found")
                continue
            except Exception as e: 
                raise e
            # print(f"{video_id}.mp4 found")
            input_tensor = _preprocess_keypoints(frames_data)
            total_predictions += 1
            # print(f"Test run input shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")
            # print(frames_data[0].keys())  # should show 'people' key

            # print(pred_idx)
            # word = idx_to_class[pred_idx]
            # print("Prediction:", word)

            if gloss_count < 100 :
                # Get logits
                input_100 = model_100.get_inputs()[0].name
                output_100 = model_100.get_outputs()[0].name
                logits_100 = model_100.run([output_100], {input_100: input_tensor})[0]
                # Apply softmax
                exp_logits_100 = np.exp(logits_100 - np.max(logits_100, axis=1, keepdims=True))
                probs_100 = exp_logits_100 / np.sum(exp_logits_100, axis=1, keepdims=True)
                # Get top prediction
                predicted_idx_100 = int(np.argmax(probs_100, axis=1)[0])
                confidence_100 = float(probs_100[0, predicted_idx_100])
                # print(f"Predicted class index: {predicted_idx_100}, confidence: {confidence_100:.4f}")
                predicted_label_100 = idx_to_class.get(str(predicted_idx_100), "Unknown")
                # print(f"Predicted 100 label: {predicted_label_100}")
                if predicted_idx_100 == gloss_idx :
                    correct_predictions[100] += 1
                if gloss_count == 99 :
                    print("On gloss 99, will stop using 100 gloss model")

            if gloss_count < 300 :
                # Get logits
                input_300 = model_300.get_inputs()[0].name
                output_300 = model_300.get_outputs()[0].name
                logits_300 = model_300.run([output_300], {input_300: input_tensor})[0]
                # Apply softmax
                exp_logits_300 = np.exp(logits_300 - np.max(logits_300, axis=1, keepdims=True))
                probs_300 = exp_logits_300 / np.sum(exp_logits_300, axis=1, keepdims=True)
                # Get top prediction
                predicted_idx_300 = int(np.argmax(probs_300, axis=1)[0])
                confidence_300 = float(probs_300[0, predicted_idx_300])
                # print(f"Predicted class index: {predicted_idx_300}, confidence: {confidence_300:.4f}")
                predicted_label_300 = idx_to_class.get(str(predicted_idx_300), "Unknown")
                # print(f"Predicted 300 label: {predicted_label_300}")
                if predicted_idx_300 == gloss_idx :
                    correct_predictions[300] += 1
                if gloss_count == 299 :
                    print("On gloss 299, will stop using 300 gloss model")

            if gloss_count < 1000 :
                input_1000 = model_1000.get_inputs()[0].name
                output_1000 = model_1000.get_outputs()[0].name
                logits_1000 = model_1000.run([output_1000], {input_1000: input_tensor})[0]

                exp_logits_1000 = np.exp(logits_1000 - np.max(logits_1000, axis=1, keepdims=True))
                probs_1000 = exp_logits_1000 / np.sum(exp_logits_1000, axis=1, keepdims=True)
                    
                predicted_idx_1000 = int(np.argmax(probs_1000, axis=1)[0])
                confidence_1000 = float(probs_1000[0, predicted_idx_1000])


                # print(f"Predicted class index: {predicted_idx_1000}, confidence: {confidence_1000:.4f}")
                predicted_label_1000 = idx_to_class.get(str(predicted_idx_1000), "Unknown")
                # print(f"Predicted 1000 label: {predicted_label_1000}")
                if predicted_idx_1000 == gloss_idx :
                    correct_predictions[1000] += 1
                if gloss_count == 999 :
                    print("On gloss 999, will stop using 100 gloss model")

            if gloss_count < 2000 :
                input_2000 = model_2000.get_inputs()[0].name
                output_2000 = model_2000.get_outputs()[0].name
                logits_2000 = model_2000.run([output_2000], {input_2000: input_tensor})[0]

                exp_logits_2000 = np.exp(logits_2000 - np.max(logits_2000, axis=1, keepdims=True))
                probs_2000 = exp_logits_2000 / np.sum(exp_logits_2000, axis=1, keepdims=True)
                    
                predicted_idx_2000 = int(np.argmax(probs_2000, axis=1)[0])
                confidence_2000 = float(probs_2000[0, predicted_idx_2000])


                # print(f"Predicted class index: {predicted_idx_2000}, confidence: {confidence_2000:.4f}")
                predicted_label_2000 = idx_to_class.get(str(predicted_idx_2000), "Unknown")
                # print(f"Predicted 2000 label: {predicted_label_2000}")
                if predicted_idx_2000 == gloss_idx :
                    correct_predictions[2000] += 1

        gloss_count += 1
        print("========================================================================")
        print(f"Num of correct predictions per model: {correct_predictions}")
        print("========================================================================")
            
        # print("======================\nGOT HERE\n======================")
        # exit(0)

    for i, v in correct_predictions.items():
        correct_predictions[i] /= total_predictions
    print(correct_predictions)
    return correct_predictions


get_metrics()
