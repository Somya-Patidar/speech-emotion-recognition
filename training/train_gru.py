from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from data_loader import load_data

X_train, X_val, X_test, y_train, y_val, y_test = load_data()

model = Sequential([
    GRU(128, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.3),
    Dense(y_train.shape[1], activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=40,
    batch_size=32,
    callbacks=[early_stop]
)

model.save("../models/gru_model.h5")
print("✅ GRU model saved")
