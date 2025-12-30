import librosa
import numpy as np

SAMPLE_RATE = 22050
DURATION = 3  # seconds
SAMPLES = SAMPLE_RATE * DURATION

def preprocess_audio(file_path):
    # Load audio
    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

    # Remove silence
    signal, _ = librosa.effects.trim(signal)

    # Normalize
    signal = signal / np.max(np.abs(signal))

    # Pad or trim
    if len(signal) < SAMPLES:
        padding = SAMPLES - len(signal)
        signal = np.pad(signal, (0, padding))
    else:
        signal = signal[:SAMPLES]

    return signal
