import os
import json
import tensorflow as tf
from src.preprocess import load_data
from src.model import build_model


ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint

DATA_DIR = "data/dev"
SAVE_MODEL = "models/signature_model.h5"


def train():
    (X_train, X_val, y_train, y_val), classes = load_data(DATA_DIR)
    model = build_model(X_train.shape[1:], len(classes))

    os.makedirs("models", exist_ok=True)  # ✅ Ensure folder exists

    checkpoint = ModelCheckpoint(SAVE_MODEL, save_best_only=True, monitor="val_accuracy", mode="max")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=16,
        callbacks=[checkpoint]
    )

    # ✅ Save classes
    with open("models/classes.json", "w") as f:
        json.dump(classes, f)

    return model, history, classes

if __name__ == "__main__":
    train()
