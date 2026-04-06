import os
import csv
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

# Settings
data_dir = '/u50/chandd9/downloads/asl_alphabet_train/asl_alphabet_train'
save_dir = './saved_models'
results_dir = './results'
os.makedirs(save_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

batch_size = 64
num_epochs_phase1 = 5
num_epochs_phase2 = 10

val_ratio = 0.1
test_ratio = 0.1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transforms
train_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# Dataset Split─
dataset = ImageFolder(data_dir, transform=train_transforms)

total_size = len(dataset)
test_size = int(total_size * test_ratio)
val_size = int(total_size * val_ratio)
train_size = total_size - val_size - test_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size]
)

# override transforms
val_dataset.dataset.transform = val_transforms
test_dataset.dataset.transform = val_transforms

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

class_names = dataset.classes

# Mode
weights = MobileNet_V2_Weights.DEFAULT
model = mobilenet_v2(weights=weights)
model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
model = model.to(device)

criterion = nn.CrossEntropyLoss()

# Phase 1: Train classifier head
for param in model.features.parameters():
    param.requires_grad = False

optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)

best_val_acc = 0

print("=== Phase 1: Training head ===")

for epoch in range(num_epochs_phase1):

    model.train()

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

    # validation
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs,1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = correct / total

    print(f"Epoch {epoch+1}/{num_epochs_phase1} | Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:

        best_val_acc = val_acc
        path = os.path.join(save_dir,"mobilenet_phase1_best.pth")
        torch.save(model.state_dict(), path)
        print("Saved:", path)

# Phase 2: Fine-tuning─
for param in model.features.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

print("\n=== Phase 2: Fine tuning ===")

for epoch in range(num_epochs_phase2):

    model.train()

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

    # validation
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs,1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = correct / total

    print(f"Epoch {epoch+1}/{num_epochs_phase2} | Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:

        best_val_acc = val_acc
        path = os.path.join(save_dir,"mobilenet_best.pth")
        torch.save(model.state_dict(), path)
        print("Saved:", path)

# TEST EVALUATION─
print("\n=== TESTING BEST MODEL ===")

model.load_state_dict(torch.load(os.path.join(save_dir,"mobilenet_best.pth")))
model.eval()

correct = 0
total = 0

results_file = os.path.join(results_dir,"test_predictions.csv")

with open(results_file,"w",newline="") as f:

    writer = csv.writer(f)
    writer.writerow(["true_label","predicted_label"])

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs,1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for t,p in zip(labels,predicted):

                writer.writerow([
                    class_names[t.item()],
                    class_names[p.item()]
                ])

test_acc = correct / total

print("Test Accuracy:", test_acc)
print("Predictions saved to:", results_file)