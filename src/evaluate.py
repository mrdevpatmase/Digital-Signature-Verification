from src.preprocess import load_data
from tensorflow.keras.models import load_model # type: ignore

DATA_DIR = "data/dev"
MODEL_PATH = "models/signature_model.h5"


def evaluate():
    (_, X_val, _, y_val), classes = load_data(DATA_DIR)

    model = load_model(MODEL_PATH)
    loss, acc = model.evaluate(X_val, y_val, verbose=1)

    print(f"✅ Validation Accuracy: {acc*100:.2f}%")
    print(f"Validation Loss: {loss:.4f}")

    return acc, loss

if __name__ == "__main__":
    evaluate()
