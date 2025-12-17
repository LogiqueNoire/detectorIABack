import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, n_classes=2, dropout=0.4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x
