import librosa
import numpy as np

SAMPLE_RATE = 22050
DURATION = 3
SAMPLES = SAMPLE_RATE * DURATION

def preprocess_audio(file_path):
    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    signal, _ = librosa.effects.trim(signal)

    signal = signal / np.max(np.abs(signal))

    if len(signal) < SAMPLES:
        signal = np.pad(signal, (0, SAMPLES - len(signal)))
    else:
        signal = signal[:SAMPLES]

    return signal
