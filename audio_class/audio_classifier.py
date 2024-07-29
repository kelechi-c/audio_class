import torch
from torch import nn
from utils import config
from dataloader import train_loader

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

        # Calculate the output size after convolutions and pooling
        self.conv_output_size = self._get_conv_output_size((128, 2000))

        self.linear_fc = nn.Sequential(
            nn.Linear(self.conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, out_classes),
        )

    def _get_conv_output_size(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, 1, *shape))
        output_feat = self.audio_convnet(input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    def forward(self, x: torch.Tensor):
        # x shape: (batch_size, 128, 2000)
        x = x.unsqueeze(1)  # Now shape is (batch_size, 1, 128, 2000)
        x = self.audio_convnet(x)

        x = x.view(x.size(0), -1)  # Flatten

        x = self.linear_fc(x)

        return x  # Shape: (batch_size, out_classes)


classifier = MusiClass()

classifier = classifier.to(torch.bfloat16).to(config.device)
classifier = torch.compile(classifier)

x = next(iter(train_loader))[0]
print(x.shape)

k = classifier(x)

torch.cuda.empty_cache()

print(f"model pred shape => {k.shape}")
print(k)
