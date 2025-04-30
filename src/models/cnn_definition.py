import torch
from torch import nn

from src.config import INPUT_SIZE  # Need this for flattened_size calculation


# --- Model Definition (Simple CNN) ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)
        # Calculate flattened size: 64 channels * (input_size / 2^3) * (input_size / 2^3)
        flattened_size = 64 * (INPUT_SIZE // 8) * (INPUT_SIZE // 8)
        self.fc1 = nn.Linear(flattened_size, 512)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # Add dropout for regularization
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = self.relu4(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
