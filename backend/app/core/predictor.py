import torch

def predict(model, labels, seq):
    logits = model(seq)
    probabilities = torch.softmax(logits, dim=1)
    idx = torch.argmax(probabilities, dim=1).item()
    confidence = float(probabilities[0, idx])
    # print(f"idx: {idx}, confidence: {confidence}")
    return labels[idx]