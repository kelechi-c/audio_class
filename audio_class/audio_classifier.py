import torch
from torch import nn
from utils import config


# model class => convnet for audio classification
class MusiClass(nn.Module):
    def __init__(self, out_classes=18):
        super().__init__()
        self.audio_convnet = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
        )

        self.linear_fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 128), nn.ReLU(), nn.Linear(128, out_classes)
        )

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(1)

        x = self.audio_convnet(x)

        x = x.view(-1, 128 * 4 * 4)  # flatten output
        x = self.linear_fc(x)

        return x


classifier = MusiClass().to(config.device)
classifier = torch.compile(classifier)
