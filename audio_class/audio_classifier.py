import torch
from torch import nn
import torchvision


class MusiClass(nn.Module):
    def __init__(self, out_classes=10):
        super().__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=2),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=2),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
        )
        self.linear_1 = nn.Sequential(
            nn.Linear(16384, 128), nn.Dropout2d(0.1), nn.Linear(128, out_classes)
        )

    def forward(self, input):
        x = self.convnet(input)
        x = self.linear_1(x)

        return x


class MusicResnet(nn.Module):  # just for experimenting
    def __init__(self, out_classes=10):
        super().__init__()
        resnet = torchvision.models.resnet50(pretrained=True)
        modules = resnet.children()[-1]
        self.convnet = nn.Sequential(nn.ModuleList(*modules))
        self.linear1 = nn.Linear(1024, 512)
        self.layer_norm = nn.BatchNorm2d()
        self.linear2 = nn.Linear(512, out_classes)

        for par in self.convnet.parameters():
            par.requires_grad = True

    def forward(self, input):
        x = self.convnet(input)
        x = self.layer_norm(self.linear1(x))
        x = self.linear2(x)

        return x
