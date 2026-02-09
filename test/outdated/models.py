import torch
import torch.nn as nn

class GestureCNN(nn.Module):
    def __init__(self):
        super(GestureCNN, self).__init__()
        # Input: [Batch, 6 Axes, 120 Samples]
        # 8 filters, kernel size 3: Looks for local motion patterns
        self.conv1 = nn.Conv1d(6, 8, kernel_size=3, padding=1)
        # Global Average Pooling: Makes the model "position independent"
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(8, 4) # Output: 4 Move IDs

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class VoiceCNN(nn.Module):
    def __init__(self):
        super(VoiceCNN, self).__init__()
        # Input: [Batch, 40 MFCCs, 50 Time Windows]
        # 16 filters: Detects complex phoneme frequencies
        self.conv1 = nn.Conv1d(40, 16, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(16, 3) # Output: 3 Pokemon IDs

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)