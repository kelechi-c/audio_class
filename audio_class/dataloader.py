import torch
import math
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, random_split
from utils import config, get_spectrogram
from tqdm.auto import tqdm

hfdata = load_dataset(config.dataset_id, split="train", streaming=True)

hfdata = hfdata.take(config.split)
hfdata_list = list(hfdata)

audio_arrays = [k for k in tqdm(hfdata_list["audio"])]
labels = [v for v in tqdm(hfdata_list["genre_ids"])]


class MusicData(Dataset):
    def __init__(self, audio_arrays=audio_arrays, labels=labels):
        super().__init__()
        self.audio_arrays = audio_arrays
        self.labels = labels

    def __len__(self):
        return len(self.audio_arrays)

    def __getitem__(self, idx):
        audio = get_spectrogram(self.audio_arrays[idx])
        audio_tensor = torch.tensor(audio)

        label = torch.tensor(self.labels[idx])

        return audio_tensor, label


music_genre_data = MusicData()

train_size = math.floor(len(music_genre_data) * config.train_split)
val_size = len(music_genre_data) - train_size

train_data, valid_data = random_split(music_genre_data, (train_size, val_size))

train_loader = DataLoader(train_data, config.batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, config.batch_size, shuffle=False)
