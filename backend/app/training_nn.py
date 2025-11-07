import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
# from preprocess_V2 import JSONASLDataset, le  # Import dataset class and label encoder
from torch.utils.data import Dataset
from model import Model_CNN
import numpy as np

# global vars
DATA_DIR = "../../../train_output_cleaned"  # folder containing .npy files from preprocessing
LABEL_FILE_NAME = "ordered_labels.npy"
BATCH_SIZE = 4
EPOCHS = 250
LEARNING_RATE = 0.0001
INPUT_SIZE = 126  # 126 (2 hands, 63/3 each)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define a dataset to store features and labels
# remnant of preprocess_V2.py
class JSONASLDataset(Dataset):
    def __init__(self, features_paths, labels):
        self.features_paths = features_paths
        self.labels = labels

    def __len__(self):
        return len(self.features_paths)

    def __getitem__(self, idx):
        X = np.load(self.features_paths[idx])
        y = self.labels[idx]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long) #tensor of features and label

# single run of training
def train_model(model, batches, criterion, optimizer):
    model.train()
    # forward pass
    avg_loss=0.0
    for x_train, y_train in batches:
        x_train, y_train = x_train.to(DEVICE), y_train.to(DEVICE)
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
        # print(f"Loss: {loss.item()}")    
    return avg_loss / len(batches)

# TODO: implement evaluation function
def evaluate_model(model, test_x, test_y):
    return 0

# LOAD FEATURES (excluding labels.npy)
all_feature_paths = sorted([os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if (f.endswith(".npy") and f != LABEL_FILE_NAME)])

# checks if the labels are saved in the same place as the videos
labels_path = os.path.join(DATA_DIR, LABEL_FILE_NAME)
if not os.path.exists(labels_path):
    raise ValueError("ordered_labels.npy not found")
    exit(1)

y_train = np.load(labels_path)

# split into train and test (80-20 split)
split_index = int(0.8 * len(all_feature_paths))
train_feature_paths = all_feature_paths[:split_index]
test_feature_paths = all_feature_paths[split_index:]
y_test = y_train[split_index:]
y_train = y_train[:split_index]

# encode labels to numeric
le = LabelEncoder()
y_train_numeric = le.fit_transform(y_train)
# y_test_numeric = le.transform(y_test)

# create dataset and dataloader
dataset = JSONASLDataset(train_feature_paths, y_train_numeric)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# create model
model = Model_CNN(input_size=INPUT_SIZE, num_classes=len(le.classes_)).to(DEVICE)

# create loss function + optimizer
# apparently Adam is better for CNNs, dont really know how it works tbh
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()

for e in range(EPOCHS):
    print(f"Epoch {e+1}/{EPOCHS}")
    loss = train_model(model, train_loader, loss_fn, optimizer)
    acc = evaluate_model(model, X_test, y_test_numeric)
    print(f"Avg Loss: {loss}, Accuracy: {acc}")

# save the model
torch.save(model.state_dict(), "asl_model.pth")
