import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

def load_data():
    X = np.load("../processed_data/X_mfcc.npy", allow_pickle=True)
    y = np.load("../processed_data/y_labels.npy")

    # Pad sequences (important for LSTM/GRU)
    X = pad_sequences(X, padding="post", dtype="float32")

    num_classes = len(np.unique(y))
    y = to_categorical(y, num_classes=num_classes)

    # Train / Validation / Test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
