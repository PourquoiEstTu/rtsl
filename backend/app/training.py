import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from preprocess_V2 import JSONASLDataset, le  # Import dataset class and label encoder
from model import SignLSTM
import numpy as np

# global vars
DATA_DIR = "debug_output"  # folder containing .npy files from preprocessing
BATCH_SIZE = 4
EPOCHS = 250
LEARNING_RATE = 0.0001
INPUT_SIZE = 225  # 225 (hands + pose)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LOAD FEATURES AND LABELS
all_feature_paths = sorted([os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".npy")])

# checks if the labels are saved in the same place as the videos
labels_path = os.path.join(DATA_DIR, "labels.npy")
if os.path.exists(labels_path):
    y_numeric = np.load(labels_path)
else:
    raise ValueError("labels.npy not found")

# Make sure features and labels match
valid_feature_paths = []
valid_labels = []

for i, feature_path in enumerate(all_feature_paths):
    # skip the labels.npy file itself
    if os.path.basename(feature_path) == "labels.npy":
        continue

    # ensure file exists and has data
    if os.path.exists(feature_path) and os.path.getsize(feature_path) > 0:
        valid_feature_paths.append(feature_path)
        valid_labels.append(y_numeric[i])

# convert labels back to numpy
y_numeric_filtered = np.array(valid_labels)

# Create dataset
dataset = JSONASLDataset(valid_feature_paths, y_numeric_filtered)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"Dataset ready. Number of samples: {len(dataset)}")
print(f"Classes: {le.classes_}")

# INITIALIZE MODEL
model = SignLSTM(input_size=INPUT_SIZE, num_classes=len(le.classes_)).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# TRAINING LOOP
for epoch in range(EPOCHS): # for each iteration
    model.train()
    total_loss, correct = 0, 0

    for X, y in dataloader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()

        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == y).sum().item()

    acc = correct / len(dataset)
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {total_loss:.4f} | Acc: {acc:.3f}")

# SAVE MODEL
torch.save(model.state_dict(), "sign_lstm.pth")
print("Model saved to sign_lstm.pth")
