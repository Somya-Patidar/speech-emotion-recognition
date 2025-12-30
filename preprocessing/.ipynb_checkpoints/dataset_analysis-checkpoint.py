import os
from collections import Counter

DATASET_PATH = "../data/RAVDESS"

# RAVDESS emotion mapping
emotion_map = {
    "01": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear"
}

emotion_count = Counter()

for actor in os.listdir(DATASET_PATH):
    actor_path = os.path.join(DATASET_PATH, actor)
    if not os.path.isdir(actor_path):
        continue

    for file in os.listdir(actor_path):
        parts = file.split("-")
        emotion_code = parts[2]

        if emotion_code in emotion_map:
            emotion_count[emotion_map[emotion_code]] += 1

print("Emotion Distribution:")
for emotion, count in emotion_count.items():
    print(f"{emotion}: {count}")
