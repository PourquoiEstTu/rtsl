import os
import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import numpy as np
from collections import deque
from sklearn.preprocessing import LabelEncoder

# Model Definition LSTM
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

# Model Definition CNN version
# For now WIP, and high level structure only
# in theory this works lol
class Model_CNN(nn.Module):
    def __init__(self, input_size, num_classes, frames=60):
        # call super constructor for CNN model from nn.Module
        super(Model_CNN, self).__init__()
        # define the layers of the CNN

        # convolutional layers
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        # pooling layers
        self.pool = nn.MaxPool1d(kernel_size=2)

        # after one pooling layer the frame size is halved
        # therefore we need to account for that in the fully connected layer
        post_pool_frames = frames // 2 

        # fully connected layers
        self.fc = nn.Linear(128 * post_pool_frames, num_classes)

    def forward(self, x):
        # x = (batch, seq_len, input_size)
        x = x.transpose(1, 2)
        # x = (batch, input_size, seq_len)
        # needed for conv1d (not too sure why, look into)
        x = F.relu(self.conv1(x)) #ensures non negative
        x = self.pool(x)
        x = F.relu(self.conv2(x)) #ensures non negative
        # flatten for fully connected as fc layers take (a, b) input not (a, b, c), i think
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x