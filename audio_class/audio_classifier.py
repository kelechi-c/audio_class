import torch
from torch import nn
from utils import config


class MusiClass(nn.Module):
    def __init__(self, out_classes=10):
        super().__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=2),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=2),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
        )

        self.linear_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16384, 128),
            nn.Dropout2d(0.1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Linear(128, out_classes),
        )

    def forward(self, input: torch.Tensor):
        input = input.unsqueeze(1)
        x = self.convnet(input)
        x = self.linear_fc(x)

        return x


classifier = MusiClass().to(config.device)
