import torch
import librosa
import torchaudio
import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

dataset_id = "lewtun/music_genres"

hfdata = load_dataset(dataset_id, split="train")  # , streaming=True)

audio_files = hfdata["audio"]
genre_ids = hfdata["genre_id"]
genres = hfdata["genre"]


class MusicData(Dataset):
    def __init__(self, files, labels, extractor):
        super().__init__()
        self.audio_files = files
        self.labels = labels
        self.extractor = extractor

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        pass
