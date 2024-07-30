import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, IterableDataset
from utils import config, get_spectrogram


hfdata = load_dataset(config.dataset_id, split="train", streaming=True)
hfdata = hfdata.take(config.split)


class MusicData(IterableDataset):
    def __init__(self, dataset=hfdata):
        self.dataset = dataset
        self.target_shape = 1323119

    def __iter__(self):
        for item in self.dataset:
            audio = item["audio"]["array"]

            audio = librosa.resample(
                audio, orig_sr=item["audio"]["sampling_rate"], target_sr=config.small_sr
            )
            audio = librosa.util.fix_length(
                audio, size=self.target_shape, axis=0)

            label = item["genre_id"]

            yield audio, label


m_data = MusicData()

train_loader = DataLoader(m_data, batch_size=config.batch_size)
print("dataloader created")

x = next(iter(train_loader))[0]
print(x)
