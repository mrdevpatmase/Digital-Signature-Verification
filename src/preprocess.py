import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

to_categorical = tf.keras.utils.to_categorical

def load_data(data_dir, img_size=(128, 128)):
    X, y = [], []
    classes = os.listdir(data_dir)

    for idx, cls in enumerate(classes):
        cls_path = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_path):
            continue

        for file in os.listdir(cls_path):
            file_path = os.path.join(cls_path, file)

            # ✅ Skip folders
            if os.path.isdir(file_path):
                continue

            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, img_size)
            img = img / 255.0
            X.append(img)
            y.append(idx)

    X = np.array(X).reshape(-1, img_size[0], img_size[1], 1)
    y = to_categorical(y, num_classes=len(classes))

    print(f"✅ Loaded {len(X)} images across {len(classes)} classes: {classes}")
    return train_test_split(X, y, test_size=0.2, random_state=42), classes
