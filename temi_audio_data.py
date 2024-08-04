import os
import torch
import librosa
from pydub import AudioSegment
from torch.utils.data import Dataset, IterableDataset, DataLoader
from datasets import load_dataset
import os
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder


class config:
    dtype = torch.float16
    target_duration = 5.0
    sr = 16000
    folder_path = "/kaggle/input/sesa-audio/SESA"


class VOICeData(Dataset):
    def __init__(self, audio_arrays, labels) -> None:
        super().__init__()
        self.audio_arrays = audio_arrays
        self.labels = labels

    def __len__(self):
        return len(self.audio_arrays)

    def __getitem__(self, idx):
        audio = self.audio_arrays[idx]
        audio = torch.tensor(audio, dtype=config.dtype)

        label = self.audio_arrays[idx]
        label = torch.tensor(label, dtype=config.dtype)

        return audio, label


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

        return waveform, label

    def load_audio_file(self, file_path):
        # Load audio file
        audio, _ = librosa.load(file_path, sr=self.sr, duration=self.duration)

        # Pad or trim to fixed length
        if len(audio) > self.num_samples:
            audio = audio[: self.num_samples]
        elif len(audio) < self.num_samples:
            audio = np.pad(audio, (0, self.num_samples - len(audio)))

        # Convert to tensor
        return torch.FloatTensor(audio)


def get_wav_files(folder_path):
    wav_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(folder_path)
        for file in files
        if file.endswith(".wav")
    ]
    return wav_files


def create_dataset(wav_files, duration=5.0, sr=22050):
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
wav_files = get_wav_files(folder_path)
dataset, class_files, label_encoder = create_dataset(wav_files, duration=5.0, sr=22050)

# Print some information
print(f"Total number of files: {len(dataset)}")
for class_name, files in class_files.items():
    encoded_label = label_encoder.transform([class_name])[0]
    print(f"Class '{class_name}' (encoded: {encoded_label}): {len(files)} files")

# To get the original labels back:
print("\nLabel encoding:")
for i, class_name in enumerate(label_encoder.classes_):
    print(f"Class '{class_name}': {i}")
