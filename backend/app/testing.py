import os
import torch
import numpy as np
from preprocess_V2 import JSONASLDataset, le  # import label encoder
from model import SignLSTM

# CONFIG
video_id = "27172"                    # ID of the video to test
data_dir = "test_output"             # folder containing preprocessed .npy files
model_path = "sign_lstm.pth"          # trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LOAD PREPROCESSED VIDEO
npy_path = os.path.join(data_dir, f"{video_id}.npy")
if not os.path.exists(npy_path):
    raise FileNotFoundError(f"{npy_path} not found. Run preprocessing first.")

sequence = np.load(npy_path)          # shape: (target_length, 225)
input_seq = torch.tensor([sequence], dtype=torch.float32).to(device)  # batch of 1

# LOAD MODEL
model = SignLSTM(input_size=225, num_classes=len(le.classes_))
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# PREDICT
with torch.no_grad():
    logits = model(input_seq)
    pred_idx = logits.argmax(dim=1).item()
    pred_label = le.inverse_transform([pred_idx])[0]

print(f"Predicted ASL sign for video {video_id}: {pred_label}")
