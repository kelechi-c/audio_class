import torch
import librosa
import numpy as np
from pydub import AudioSegment
from matplotlib import pyplot as plt


class config:
    batch_size = 32
    lr = 1e-3
    max_duration = 30
    sample_rate = 44100
    epochs = 50
    num_classes = 18
    split = 15000
    train_split = 0.9
    audio_length = 2000
    dataset_id = "lewtun/music_genres"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_outpath = "musiclass"
    model_filename = "musiclass.pth"


def load_audio(file: str):
    if file.endswith(".mp3"):
        audio = AudioSegment.from_mp3(file)
        fname = file.split(".")[0]
        file = f"{fname}.wav"
        audio.export(file)

    waveform, sr = librosa.load(file, sr=config.sample_rate)

    return waveform, sr


def get_spectrogram(audio):
    array, sr = audio["array"], audio["sampling_rate"]
    target_shape = (128, config.audio_length)

    specgram = librosa.feature.melspectrogram(
        y=array, sr=sr, n_mels=128, fmax=8000)
    specgram = librosa.power_to_db(specgram, ref=np.max)

    specgram = librosa.util.fix_length(specgram, size=target_shape[1], axis=1)

    return specgram


def pad_audio(audio: np.ndarray, target_length: int):
    padding_needed = target_length - len(audio)
    if padding_needed > 0:
        return np.pad(audio, (0, padding_needed), mode="reflect")

    return audio


def display_melspec(audio):
    array, sr = audio["array"], audio["sampling_rate"]

    specgram = librosa.feature.melspectrogram(
        y=array, sr=sr, n_mels=128, fmax=8000)
    specgram = librosa.power_to_db(specgram, ref=np.max)

    plt.figure().set_figwidth(12)
    librosa.display.specshow(specgram, x_axis="time",
                             y_axis="mel", sr=sr, fmax=8000)
    plt.colorbar()

    return specgram


def count_params(model: torch.nn.Module):
    p_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return p_count
