import librosa
import os
import torchaudio
from pydub import AudioSegment
from transformers import WhisperFeatureExtractor


class config:
    batch_size = 32
    lr = 1e-3
    max_duration = 30
    sample_rate = 16000
    epochs = 50


def load_audio(file: str):
    if file.endswith(".mp3"):
        audio = AudioSegment.from_mp3(file)
        file = f'{file.split('.')[0]}.wav'
        audio.export(file)

    waveform, sr = torchaudio.load(file)

    return waveform


def extract_audio_features(audio):
    pass
