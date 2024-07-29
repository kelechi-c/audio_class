from matplotlib.pyplot import get
import torch
import math
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, random_split, IterableDataset
from utils import config, get_spectrogram
from tqdm.auto import tqdm


hfdata = load_dataset(config.dataset_id, split="train", streaming=True)
hfdata = hfdata.take(config.split)


class MusicData(IterableDataset):
    def __init__(self, dataset=hfdata):
        self.dataset = dataset

    def __iter__(self):
        for item in self.dataset:
            spectrogram = get_spectrogram(item["audio"])

            mspec = torch.tensor(spectrogram, dtype=torch.bfloat16)
            label = torch.tensor(item["genre_id"], dtype=torch.bfloat16)

            yield mspec, label


music_genre_data = MusicData()

train_loader = DataLoader(music_genre_data, batch_size=config.batch_size, shuffle=True)
