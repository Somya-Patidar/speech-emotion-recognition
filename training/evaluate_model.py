import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tensorflow.keras.models import load_model
from data_loader import load_data

X_train, X_val, X_test, y_train, y_val, y_test = load_data()

models = {
    "LSTM": "../models/lstm_model.h5",
    "GRU": "../models/gru_model.h5"
}

for name, path in models.items():
    print(f"\n🔍 Evaluating {name} model")
    model = load_model(path)

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    print(classification_report(y_true, y_pred_classes))

    cm = confusion_matrix(y_true, y_pred_classes)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.tight_layout()
    plt.savefig(f"../models/{name}_confusion_matrix.png")
    plt.close()

