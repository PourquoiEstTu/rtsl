import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from model import Sign1DCNN, Model_CNN, SignTCN, GRU_1DCNN
import numpy as np
import matplotlib.pyplot as plt

# PARAM CONFIG
TRAIN_DIR = "/u50/chandd9/capstone/personal_preprocessed2/train_output_normalized"
VALIDATION_DIR  = "/u50/chandd9/capstone/personal_preprocessed2/validation_output_normalized"
TEST_DIR = "/u50/chandd9/capstone/personal_preprocessed2/test_output_normalized"
BATCH_SIZE = 64
EPOCHS = 1000
LEARNING_RATE = 0.0001
INPUT_SIZE = 126
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# LOAD TRAINING DATA
train_feature_path = sorted([
    os.path.join(TRAIN_DIR, f)
    for f in os.listdir(TRAIN_DIR)
    if f.endswith(".npy") and "ordered_labels_normalized" not in f
])

# LOAD TEST DATA
test_feature_paths = sorted([
    os.path.join(TEST_DIR, f)
    for f in os.listdir(TEST_DIR)
    if f.endswith(".npy") and "ordered_labels_normalized" not in f
])

# load labels
train_label_path = os.path.join(TRAIN_DIR, "ordered_labels_normalized.npy")
test_label_path = os.path.join(TEST_DIR, "ordered_labels_normalized.npy")

# load features and normalize
# normalizing doesnt improve it makes it worse?
X_train = np.concatenate([np.array([np.load(f) for f in train_feature_path]), np.array([np.load(f) for f in test_feature_paths])], axis=0)
# X_train = (X_train - X_train.mean(axis=(0,1), keepdims=True)) / (X_train.std(axis=(0,1), keepdims=True) + 1e-6)

y_train_label = np.concatenate([np.load(train_label_path), np.load(test_label_path)], axis=0)
le = LabelEncoder()
y_train_numeric = le.fit_transform(y_train_label)

# load validation data
val_feature_path = sorted([
    os.path.join(VALIDATION_DIR, f)
    for f in os.listdir(VALIDATION_DIR)
    if f.endswith(".npy") and "ordered_labels_normalized" not in f
])

# load validation labels
val_label_path = os.path.join(VALIDATION_DIR, "ordered_labels_normalized.npy")

# load and normalize validation features
# normalizing doesnt improve it makes it worse?
X_val = np.array([np.load(f) for f in val_feature_path])
# X_val = (X_val - X_train.mean(axis=(0,1), keepdims=True)) / (X_train.std(axis=(0,1), keepdims=True) + 1e-6)

y_test_label = np.load(val_label_path)
# Make sure test labels are encoded using the same classes as training
y_test_numeric = le.transform(y_test_label)

print(f"Training samples: {X_train.shape}, Val samples: {X_val.shape}")
print(f"Number of classes: {len(le.classes_)}")
# print(f"Classes: {le.classes_}")
# print(f"Sample label mapping (train): {y_train_label[0]} -> {y_train_numeric[0]}")
# print(f"Sample label mapping (test): {y_test_label[0]} -> {y_test_numeric[0]}")
# print(f"Feature sample (train): {X_train[0][:5]}")  # print first 5 frames of first sample
# print(f"Feature sample (test): {X_val[0][:5]}")    # print first 5 frames of first sample

# exit()
# DATASET + DATALOADER
# create TensorDatasets and DataLoaders to handle batching and shuffling
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train_numeric, dtype=torch.long))
val_dataset  = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                              torch.tensor(y_test_numeric, dtype=torch.long))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
test_loader  = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# model, criterion, optimizer definitions
num_classes = len(np.unique(y_train_label))
# model = Sign1DCNN(input_size=INPUT_SIZE, num_classes=num_classes).to(DEVICE) #1D CNN works 30% acc
model = GRU_1DCNN(input_size=INPUT_SIZE, num_classes=num_classes).to(DEVICE) #Better 35% acc
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)

# use this for plotting
history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': []
}


# TRAINING LOOP
for epoch in range(EPOCHS):
    model.train()
    train_loss, train_correct = 0, 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) #ngl no clue what this does
        optimizer.step()

        train_loss += loss.item() * X_batch.size(0)
        train_correct += (outputs.argmax(1) == y_batch).sum().item()

    train_loss /= len(train_dataset)
    train_acc = train_correct / len(train_dataset)

    # TEST EVALUATION
    model.eval()
    val_loss, test_correct = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            val_loss += loss.item() * X_batch.size(0)
            test_correct += (outputs.argmax(1) == y_batch).sum().item()

    val_loss /= len(val_dataset)
    val_acc = test_correct / len(val_dataset)

    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"| Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% "
              f"| Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc*100}%")

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc * 100)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc * 100)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot Loss
ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Training and Test Loss', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot Accuracy
ax2.plot(history['train_acc'], label='Train Accuracy', linewidth=2)
ax2.plot(history['val_acc'], label='Val Accuracy', linewidth=2)
ax2.axhline(y=100/num_classes, color='r', linestyle='--', label=f'Random Baseline ({100/num_classes:.2f}%)', alpha=0.7)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_title('Training and Val Accuracy', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 100])

plt.tight_layout()
plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')

# SAVE MODEL
torch.save(model.state_dict(), "trainingNN.pth")
print("Model saved to trainingNN.pth")
