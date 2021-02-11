from abc import ABC, ABCMeta
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


# TODO implement
# Just PyTorch neural network example
class QNet(nn.Module):

    def __init__(self, dropout_rate: float = 0.15):
        super().__init__()

        self.dropout_rate = dropout_rate

        self.fc1 = nn.Linear(69, 69)
        self.fc2 = nn.Linear(69, 24)
        self.fc3 = nn.Linear(24, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = nn.Dropout(self.dropout_rate)(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return torch.sigmoid(x)

    def save_parameters(self, weights_path: str):
        torch.save(self.state_dict(), weights_path)

    def load_parameters(self, weights_path: str):
        self.load_state_dict(torch.load(weights_path))
