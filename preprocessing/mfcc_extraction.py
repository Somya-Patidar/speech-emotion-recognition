import os
import numpy as np
import librosa
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from audio_preprocessing import preprocess_audio

DATASET_PATH = "../data/RAVDESS"
SAVE_PATH = "../processed_data"

emotion_map = {
    "01": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear"
}

N_MFCC = 40

X = []
y = []

for actor in tqdm(os.listdir(DATASET_PATH)):
    actor_path = os.path.join(DATASET_PATH, actor)
    if not os.path.isdir(actor_path):
        continue

    for file in os.listdir(actor_path):
        
        if not file.endswith(".wav"):
            continue

        parts = file.split("-")

        if len(parts) < 3:
            continue

        emotion_code = parts[2]

        if emotion_code not in emotion_map:
            continue

        file_path = os.path.join(actor_path, file)

        # Preprocess audio
        signal = preprocess_audio(file_path)

        # MFCC extraction
        mfcc = librosa.feature.mfcc(
            y=signal,
            sr=22050,
            n_mfcc=N_MFCC,
            n_fft=2048,
            hop_length=512
        )

        mfcc = mfcc.T  # (time_steps, n_mfcc)

        X.append(mfcc)
        y.append(emotion_map[emotion_code])

X = np.array(X, dtype=object)

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Save processed data
os.makedirs(SAVE_PATH, exist_ok=True)
np.save(os.path.join(SAVE_PATH, "X_mfcc.npy"), X)
np.save(os.path.join(SAVE_PATH, "y_labels.npy"), y_encoded)

print("MFCC features and labels saved successfully.")
