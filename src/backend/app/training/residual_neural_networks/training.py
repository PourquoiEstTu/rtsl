import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import numpy as np
import json
import os
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sys import exit

# Dataset
# dataset class that contains the videos and their labels
# samples: list of (video_tensor_path, label)
# glosses: list of classes
# json_path is the json file listing all the glosses and their 
#   corresponding videos
class ASLVideoTensorDataset(Dataset):
    def __init__(self, tensor_dir="", json_path="", split="train", max_classes=None):
        # samples, glosses, and idx -> class and class -> dicts are  assumed to 
        #   be files in the tensor_dir
        self.samples = []
        self.glosses = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        if os.path.exists(f"{tensor_dir}/samples.json") and os.path.exists(f"{tensor_dir}/glosses.json") and os.path.exists(f"{tensor_dir}/idx_to_class.json") and os.path.exists(f"{tensor_dir}/class_to_idx.json"):
            with open(f"{tensor_dir}/samples.json", 'r') as f:
                self.samples = json.load(f)
            with open(f"{tensor_dir}/glosses.json", 'r') as g:
                self.glosses = json.load(g)
            with open(f"{tensor_dir}/idx_to_class.json", 'r') as h:
                self.idx_to_class = json.load(h)
            with open(f"{tensor_dir}/class_to_idx.json", 'r') as i:
                self.class_to_idx = json.load(i)
            print("Samples, glosses, and dictionary files found... Initialized")
        else : 
            print("No json files found... Creating")
            self.build_class_mapping(tensor_dir, json_path, split, max_classes)

    def build_class_mapping(self, tensor_dir, json_path, split="train", max_classes=None):
        with open(json_path, "r") as f:
            data = json.load(f)

        # get glosses and filter them down if desired
        all_glosses = sorted([entry["gloss"] for entry in data])
        if max_classes is not None:
            self.glosses = all_glosses[:max_classes]  # only keep the first max_classes glosses
        else:
            self.glosses = all_glosses
        with open(f"{tensor_dir}/glosses.json", 'w') as g :
            json.dump(self.glosses, g, indent=2)
        print("Glosses created")

        self.class_to_idx = {g: i for i, g in enumerate(self.glosses)}
        with open(f"{tensor_dir}/class_to_idx.json", 'w') as f :
            json.dump(self.class_to_idx, f, indent=2)
        print("{class: idx} dictionary created")

        self.idx_to_class = {i: g for i, g in enumerate(self.glosses)}
        with open(f"{tensor_dir}/idx_to_class.json", 'w') as f :
            json.dump(self.idx_to_class, f, indent=2)
        print("{idx: class} dictionary created")

        # add instances only for selected glosses
        for entry in data:
            gloss = entry["gloss"]
            if gloss not in self.class_to_idx:
                continue  # skip glosses beyond max_classes

            label = self.class_to_idx[gloss]

            for inst in entry["instances"]:
                if inst["split"] != split:
                    continue

                video_id = inst["video_id"]
                path = os.path.join(tensor_dir, f"{video_id}.npy")

                if os.path.exists(path):
                    self.samples.append((path, label))
        with open(f"{tensor_dir}/samples.json", 'w') as f :
            json.dump(self.samples, f, indent=2)
        print("samples created")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        video = torch.load(path)  # (T, 3, 224, 224)
        return video, label
# test = ASLVideoTensorDataset("/u50/quyumr/archive/test_output_json_video_padded", "/u50/quyumr/archive/WLASL_v0.3.json", split="test", max_classes=None)
# test = ASLVideoTensorDataset(samples_path="/u50/quyumr/archive/test_output_json_video_padded/samples.json", glosses_path="/u50/quyumr/archive/test_output_json_video_padded/glosses.json")
# test.build_class_mapping()
# exit()


# using this padding function for now because when i created the tensors for resnet i forgot about having to normalize the frame length
# need to preprecess the data to have same length videos
# def pad_collate_fn(batch):
#     videos, labels = zip(*batch)
#     lengths = [v.shape[0] for v in videos]
#     max_len = max(lengths)
#     padded_videos = []

#     for v in videos:
#         T, C, H, W = v.shape
#         if T < max_len:
#             pad = torch.zeros(max_len - T, C, H, W)
#             v = torch.cat([v, pad], dim=0)
#         padded_videos.append(v)

#     videos_tensor = torch.stack(padded_videos)  # (B, T_max, 3, 224, 224)
#     labels_tensor = torch.tensor(labels)
#     lengths_tensor = torch.tensor(lengths)
#     return videos_tensor, labels_tensor, lengths_tensor


# Model
class ResNetGRU(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # pretrained resnet 
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # this is use to extract frame-level features
        # remove the final fully connected layer and replace it with my own layers
        # (dont fully understand how this does the above but i trust it works)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # freeze pretrained parameters
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.conv = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.gru = nn.GRU(
            input_size=256,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, lengths):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)

        feats = self.backbone(x).squeeze(-1).squeeze(-1)
        feats = feats.view(B, T, 512)

        feats = feats.transpose(1, 2) # (batch, 512, T)
        feats = self.conv(feats)
        feats = feats.transpose(1, 2) # (batch, T, 256)

        # Pack padded sequence
        packed_feats = pack_padded_sequence(feats, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, h = self.gru(packed_feats)
        # h: (num_layers*2, B, hidden_size)
        h = torch.cat([h[-2], h[-1]], dim=1)  # (B, 512)

        return self.fc(h)


# Training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for x, y, lengths in train_loader:
            x, y, lengths = x.to(device), y.to(device), lengths.to(device)

            optimizer.zero_grad()
            out = model(x, lengths)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            train_correct += (out.argmax(1) == y).sum().item()
            train_total += y.size(0)

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for x, y, lengths in val_loader:
                x, y, lengths = x.to(device), y.to(device), lengths.to(device)
                out = model(x, lengths)
                loss = criterion(out, y)

                val_loss += loss.item() * x.size(0)
                val_correct += (out.argmax(1) == y).sum().item()
                val_total += y.size(0)

        scheduler.step()
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Acc: {train_correct/train_total:.4f} | " f"Val Acc: {val_correct/val_total:.4f} | Train Loss: {train_loss/train_total:.4f} | Val Loss: {val_loss/val_total:.4f}")

    return model


# Main
# def main():
#     num_epochs = 100
#     learning_rate = 0.001
#     batch_size = 8  # smaller because videos are big
# 
#     _dir = '/u50/chandd9/capstone/face_pose'
#     train_data_dir = f'{_dir}/train_output_resnet'
#     val_data_dir = f'{_dir}/val_output_resnet'
#     json_path = f'{_dir}/WLASL_v0.3.json'
# 
#     train_dataset = ASLVideoTensorDataset(train_data_dir, json_path, split="train", max_classes=100)
#     val_dataset   = ASLVideoTensorDataset(val_data_dir, json_path, split="val", max_classes=100)
# 
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader   = DataLoader(val_dataset, batch_size=batch_size)
# 
#     num_classes = len(train_dataset.glosses)
#     print(f"Num classes: {num_classes}")
# 
#     model = ResNetGRU(num_classes)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
# 
#     train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs)


# if __name__ == "__main__":
#     main()
