import os
import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import numpy as np
from collections import deque
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F

# # Model Definition LSTM
# class SignLSTM(nn.Module):
#     def __init__(self, input_size=225, hidden_size=128, num_classes=2, num_layers=2, dropout=0.5):
#         super(SignLSTM, self).__init__()
#         # innit LSTM
#         # lstm is a type of recurrent neural network (RNN) that is capable of learning long-term dependencies in sequence data
#         # This should be replaced with the better model architecture. for now use default LSTM model in pytorch just to get things working ig
#         self.lstm = nn.LSTM(
#             input_size=input_size, # number of features for each frame
#             hidden_size=hidden_size, #internal memory
#             num_layers=num_layers, #amount of layers
#             batch_first=True, #(batch, seq, feature)
#             dropout=dropout
#         )
#         # innit layers
#         self.fc = nn.Sequential(
#             nn.Linear(hidden_size, 64), # converts the 128 output into 64
#             nn.ReLU(), # usinga  reLU activation function
#             nn.Dropout(dropout), # set dropout to 0.5 to prevent overfitting
#             nn.Linear(64, num_classes) # final layer that mapes the 64 features to num_classes outputs (logits)
#         )

#     def forward(self, x):
#         out, _ = self.lstm(x)
#         out = out[:, -1, :]  # Use last frame
#         return self.fc(out)

# Trying a 1 Dimensional CNN model for sign language recognition
# Because LSTM was not running well and CNNs have been shown to perform well on sequential data
# should try 2D CNNs too
class GRU_1DCNN(nn.Module):
    """
    Try using GRU instead with 1D CNN
    Combines 1D Convolutional layers for feature extraction with GRU layers for sequence modeling.
    The 1D CNN works well from prev testing (like 30% acc)
    The 1D CNN extracts local temporal features from the input sequences, while the GRU captures long-term dependencies.
    The final fully connected layers map the learned features to the output classes.
    """
    def __init__(self, input_size=126, num_classes=500, hidden_size=256, num_layers=2):
        super().__init__()

        # define sequence of convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

        # GRU is a type of RNN that is similar to LSTM but simpler
        self.gru = nn.GRU(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True, # clarifies that the first dimension is batch size
            dropout=0.3, 
            bidirectional=True # use bidirectional GRU to capture context from both directions
        )

        # fully connected linear layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        x = x.transpose(1, 2)
        # x: (batch, features, seq_len)
        x = self.conv_layers(x)
        # x: (batch, 256, seq_len)
        x = x.transpose(1, 2)
        # x: (batch, seq_len, features)
        gru_out, hidden_states = self.gru(x)

        forward_last = hidden_states[-2, :, :]  # Last layer forward
        backward_last = hidden_states[-1, :, :]  # Last layer backward

        # Concatenate final forward and backward hidden states
        final_hidden = torch.cat([forward_last, backward_last], dim=1)
        return self.fc(final_hidden)


class Sign1DCNN(nn.Module):
    """
    Uses 1D Convolutional layers to extract temporal features from sequential input data.
    The extracted features are globally pooled and passed through fully connected layers for classification.
    1D CNNs are effective for sequence data like time series or video frames.
    """
    def __init__(self, input_size=126, num_classes=500):
        super().__init__()

        # sqeuential allows for cleaner stacking of layers
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1), 
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

        # Global temporal pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # collapse time dimension

        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        x: (batch, seq_len, features)
        """
        x = x.transpose(1, 2)  # (batch, features, seq_len) for Conv1d
        out = self.conv_layers(x)
        out = self.global_pool(out).squeeze(-1)  # (batch, 256)
        return self.fc(out)



