import torch
import torch.nn as nn
import librosa
import os
from torch import nn, optim
from torch.nn import functional as fnn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
import numpy as np
from datasets import load_dataset
from tqdm.auto import tqdm
import gc
import wandb


class config:
    batch_size = 16
    lr = 1e-3
    grad_acc_step = 4
    epochs = 50
    num_classes = 18
    split = 100
    sr = 16000
    train_split = 0.9
    target_size = 475000
    #     dtype = torch.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_outpath = "sound_detect"
    model_filename = "sound_detect.pth"
    dtype = torch.float32
    target_duration = 5.0
    sr = 16000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folder_path = "/kaggle/input/sesa-audio/SESA"


class AudioDataset(Dataset):
    def __init__(
        self,
        file_list,
        duration=config.target_duration,
        sr=config.sr,
        label_encoder=None,
    ):
        self.file_list = file_list
        self.duration = duration
        self.sr = sr
        self.num_samples = int(duration * sr)
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        audio_path = self.file_list[idx]
        # Extract class from filename
        class_name = os.path.basename(audio_path).split("_")[0]

        # Load and preprocess audio file
        waveform = self.load_audio_file(audio_path)

        # Encode the label
        label = self.label_encoder.transform([class_name])[0]

        return torch.tensor(waveform, dtype=config.dtype), torch.tensor(
            label, dtype=config.dtype
        )

    def load_audio_file(self, file_path):
        # Load audio file
        audio, _ = librosa.load(file_path, sr=self.sr, duration=self.duration)

        # Pad or trim to fixed length
        if len(audio) > self.num_samples:
            audio = audio[: self.num_samples]
        elif len(audio) < self.num_samples:
            audio = np.pad(audio, (0, self.num_samples - len(audio)))

        # Convert to tensor
        return audio


def get_wav_files(folder_path):
    wav_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(folder_path)
        for file in files
        if file.endswith(".wav")
    ]
    return wav_files


def create_dataset(wav_files, duration=5.0, sr=config.sr):
    class_files = {}

    # Organize files by class prefix
    for file_path in wav_files:
        # Extract class prefix
        class_prefix = os.path.basename(file_path).split("_")[0]

        # Add file to corresponding class list
        if class_prefix not in class_files:
            class_files[class_prefix] = []
        class_files[class_prefix].append(file_path)

    # Create and fit the LabelEncoder
    label_encoder = LabelEncoder()
    label_encoder.fit(list(class_files.keys()))

    # Create the PyTorch dataset
    dataset = AudioDataset(
        wav_files, duration=duration, sr=sr, label_encoder=label_encoder
    )

    return dataset, class_files, label_encoder


# Usage
wav_files = get_wav_files(config.folder_path)
dataset, class_files, label_encoder = create_dataset(
    wav_files, duration=5.0, sr=config.sr
)

# model definition


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
        self.flat = nn.Flatten()

        # Dynamically calculate the size of the flattened conv output
        with torch.no_grad():
            sample_input = torch.randn(16, 1, 128, 937)
            conv_output = self.audio_convnet(sample_input)
            fs = conv_output.view(16, -1).size(1)
            print(f"fs => {fs}")

        self.linear_fc = nn.Sequential(nn.Linear(159744, 256), nn.ReLU())

        self.l2 = nn.Linear(256, out_classes)

    def _spectrogram(self, audio, config=config):
        spectrograms = []
        for i in range(audio.shape[0]):
            audio_i = audio[i].cpu().numpy()

            n_frames = int(librosa.time_to_frames(
                config.target_duration, sr=config.sr))

            specgram = librosa.feature.melspectrogram(
                y=audio_i, sr=config.sr, n_mels=128, fmax=8000
            )
            specgram = librosa.power_to_db(specgram, ref=np.max)

            if specgram.shape[1] < n_frames:
                specgram = librosa.util.fix_length(
                    specgram, size=n_frames, axis=1)
            else:
                specgram = specgram[:, :n_frames]

            spectrograms.append(specgram)

        spectrograms = np.array(spectrograms)
        spectrograms = torch.tensor(
            spectrograms, dtype=config.dtype).to(config.device)
        return spectrograms

    def forward(self, x: torch.Tensor):
        x = self._spectrogram(x)
        x = x.unsqueeze(1)  # Add channel dimension

        x = self.audio_convnet(x)
        x = self.flat(x)

        x = self.linear_fc(x)
        x = self.l2(x)
        return x  # Shape: (batch_size, out_classes)


# Usage
classifier = MusiClass()
classifier = classifier.to(config.dtype).to(config.device)

# Assuming x is your input tensor with shape [16, audio_length]
# x = torch.randn(16, config.sr * config.target_duration)  # Example input
output = classifier(x)
print(f"model pred shape => {output.shape}")
print(output)

criterion = nn.CrossEntropyLoss()  # loss function
optimizer = optim.Adam(params=classifier.parameters(), lr=config.lr)
scaler = GradScaler()

# param_count = count_params(classifier)
# print(param_count)

epochs = config.epochs

# initilaize wandb
# wandb.login(key=wandb_key)
# train_run = wandb.init(project="musiclass", name="musiclass_1")
# wandb.watch(classifier, log_freq=100)


if os.path.exists(config.model_outpath) is not True:
    os.mkdir(config.model_outpath)

output_path = os.path.join(os.getcwd(), config.model_outpath)


def training_loop(
    model=classifier, train_loader=train_loader, epochs=epochs, config=config
):
    model.train()
    for epoch in tqdm(range(epochs)):
        torch.cuda.empty_cache()
        print(f"Training epoch {epoch}")
        train_loss = 0.0
        correct = 0
        total = 0
        accuracy = 0.0

        for x, (audio, label) in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()

            audio = audio.to(config.device)
            label = label.to(config.device)

            # every iterations
            torch.cuda.empty_cache()
            gc.collect()

            # Mixed precision training
            with autocast():
                outputs = model(audio)
                loss = criterion(outputs, label.long())
                train_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

                loss = loss / config.grad_acc_step  # Normalize the loss

            # Scales loss. Calls backward() on scaled loss to create scaled gradients.
            scaler.scale(loss).backward()

            if (x + 1) % config.grad_acc_step == 0:
                # Unscales the gradients of optimizer's assigned params in-place
                scaler.step(optimizer)
                # Updates the scale for next iteration
                scaler.update()
                optimizer.zero_grad()

        # Calculate average loss and accuracy for the epoch
        avg_loss = train_loss / len(train_loader)
        accuracy = 100 * correct / total

        print(
            f"Epoch {epoch} of {epochs}, train_loss: {avg_loss:.4f}, accuracy: {accuracy:.2f}%"
        )

        #         wandb.log({
        #             "epoch": epoch,
        #             "loss": avg_loss,
        #             "accuracy": accuracy
        #         })

        print(f"Epoch @ {epoch} complete!")

    print(
        f"End metrics for run of {epochs}, train_loss: {avg_loss:.4f}, accuracy: {accuracy:.2f}%"
    )
    torch.save(
        model.state_dict(), os.path.join(
            output_path, f"{config.model_filename}")
    )


training_loop()
torch.cuda.empty_cache()
gc.collect()
print("sound classifier training complete")
