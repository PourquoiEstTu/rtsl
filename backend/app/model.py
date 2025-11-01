import os
import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import numpy as np
from collections import deque
from sklearn.preprocessing import LabelEncoder

# Model Definition
class SignLSTM(nn.Module):
    def __init__(self, input_size=225, hidden_size=128, num_classes=2, num_layers=2, dropout=0.5):
        super(SignLSTM, self).__init__()
        # innit LSTM
        # lstm is a type of recurrent neural network (RNN) that is capable of learning long-term dependencies in sequence data
        # This should be replaced with the better model architecture. for now use default LSTM model in pytorch just to get things working ig
        self.lstm = nn.LSTM(
            input_size=input_size, # number of features for each frame
            hidden_size=hidden_size, #internal memory
            num_layers=num_layers, #amount of layers
            batch_first=True, #(batch, seq, feature)
            dropout=dropout
        )
        # innit layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64), # converts the 128 output into 64
            nn.ReLU(), # usinga  reLU activation function
            nn.Dropout(dropout), # set dropout to 0.5 to prevent overfitting
            nn.Linear(64, num_classes) # final layer that mapes the 64 features to num_classes outputs (logits)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Use last frame
        return self.fc(out)
