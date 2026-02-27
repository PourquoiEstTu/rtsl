import torch
import numpy as np
import os

def predict(model, labels, seq):
    logits = model(seq)
    probabilities = torch.softmax(logits, dim=1)
    idx = torch.argmax(probabilities, dim=1).item()
    print(f"idx: {idx}")
    return labels[idx]