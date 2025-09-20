import cv2
import numpy as np
import tensorflow as tf
import json
import argparse

IMG_SIZE = (128, 128)

def predict(image_path, classes):
    # Load model
    model = tf.keras.models.load_model("models/signature_model.h5")

    # Read image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"❌ Could not read image: {image_path}")

    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)

    # Predict
    preds = model.predict(img)
    conf = float(np.max(preds))
    label = classes[int(np.argmax(preds))]
    return label, conf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to signature image")
    args = parser.parse_args()

    # Load class labels
    with open("models/classes.json") as f:
        classes = json.load(f)

    label, conf = predict(args.image, classes)
    print(f"✅ Predicted: {label} ({conf*100:.2f}%)")
