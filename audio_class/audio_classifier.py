import torch
import librosa
import gc
import numpy as np
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
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
        )
        # Dynamically calculate the size of the flattened conv output
        with torch.no_grad():
            sample_input = torch.randn(1, 1, 128, 937)
            conv_output = self.audio_convnet(sample_input)
            fs = conv_output.view(1, -1).size(1)

        self.linear_fc = nn.Sequential(nn.Linear(fs, 256), nn.ReLU())

        self.l2 = nn.Linear(256, out_classes)

    def _spectrogram(self, audio, config=config):
        if not isinstance(audio, np.ndarray):
            audio = audio.cpu().numpy()

        n_frames = int(librosa.time_to_frames(config.max_duration, sr=config.small_sr))
        audio = audio.squeeze(0)
        specgram = librosa.feature.melspectrogram(
            y=audio, sr=config.small_sr, n_mels=128, fmax=8000
        )
        specgram = librosa.power_to_db(specgram, ref=np.max)

        if specgram.shape[1] < n_frames:
            specgram = librosa.util.fix_length(specgram, size=n_frames, axis=1)
        else:
            specgram = specgram[:, :n_frames]

        specgram = torch.tensor(specgram, dtype=config.dtype).to(config.device)

        return specgram

    def forward(self, x: torch.Tensor):
        x = self._spectrogram(x)
        x = x.unsqueeze(0)  # Add channel dimension

        x = self.audio_convnet(x)

        x = torch.flatten(x)

        x = self.linear_fc(x)
        x = self.l2(x)
        x = x.unsqueeze(0)  # batch dimension
        return x  # Shape: (batch_size, out_classes)


classifier = MusiClass()

classifier = classifier.to(config.dtype).to(config.device)
classifier = torch.compile(classifier)

x = next(iter(train_loader))[0]
print(x.shape)
torch.cuda.empty_cache()

k = classifier(x)

torch.cuda.empty_cache()
gc.collect()

print(f"model pred shape => {k.shape}")
print(k)
