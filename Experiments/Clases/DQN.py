import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque
import math

class DQN(nn.Module):
    def __init__(self, input_dim=20, output_dim=8):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))  # [batch_size, 256]
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))  # [batch_size, 128]
        x = self.dropout(F.relu(self.fc3(x)))  # [batch_size, 64]
        x = self.fc4(x)  # ¡Esta línea faltaba! [batch_size, 8]
        return x