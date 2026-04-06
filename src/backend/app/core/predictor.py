import torch
import numpy as np

def predict(model, labels, seq):
    logits = model(seq)
    probabilities = torch.softmax(logits, dim=1)
    idx = torch.argmax(probabilities, dim=1).item()
    confidence = float(probabilities[0, idx])
    #print(f"{labels[idx]}, {confidence}")
    if confidence > 0.5:
        return labels[idx]
    else:
        return ""

def onnx_predict(model, labels, seq):
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    logits = model.run([output_name], {input_name: seq})[0]
    # softmax calc
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    idx = np.argmax(probabilities, axis=1)[0]
    confidence = float(probabilities[0, idx])
    #print(f"{labels[idx]}, {confidence}")
    if confidence > 0.5:
        return labels[idx]
    else:
        return ""