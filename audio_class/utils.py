import librosa
import os
import torchaudio
from pydub import AudioSegment
from transformers import WhisperFeatureExtractor
from matplotlib import pyplot as plt


class config:
    batch_size = 32
    lr = 1e-3
    max_duration = 30
    sample_rate = 16000
    epochs = 50
    num_classes = 18


feature_extractor = WhisperFeatureExtractor.from_pretrained(
    "openai/whisper-small")


def load_audio(file: str):
    if file.endswith(".mp3"):
        audio = AudioSegment.from_mp3(file)
        file = f'{file.split('.')[0]}.wav'
        audio.export(file)

    waveform, sr = librosa.load(file, sr=config.sample_rate)

    return waveform


def prepare_dataset(sample):
    audio_sample = sample["audio"]
    audio_feature = feature_extractor(
        audio_sample["array"], sampling_rate=audio_sample["sampling_rate"], padding=True
    )

    return audio_feature


def display_melspec(audio):
    array, sr = audio["array"], audio["sampling_rate"]

    specgram = librosa.feature.melspectrogram(
        y=array, sr=sr, n_mels=128, fmax=8000)

    plt.figure().set_figwidth(12)
    librosa.display.specshow(specgram, x_axis="time",
                             y_axis="mel", sr=sr, fmax=8000)
    plt.colorbar()
