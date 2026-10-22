"""Train a CNN classifier on the 28x28 pixel ISL dataset (CSV format).

Each row is a flattened 28x28 grayscale image with a numeric label.
Covers A-Z and digits 1-9 (35 classes).

Usage (from the src/ folder, venv activated):
    python train_landmarks.py

Saves to checkpoints/best_landmark_model.keras and class_names_landmarks.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

ROOT      = Path(__file__).resolve().parents[1]
DATA_DIR  = ROOT / "dataset" / "landmarks"
CKPT_DIR  = ROOT / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)

TRAIN_CSV = DATA_DIR / "Completedataset_Train.csv"
TEST_CSV  = DATA_DIR / "Completedataset_Test.csv"

IMG_SIZE   = 28
EPOCHS     = 30
BATCH_SIZE = 64
VAL_SPLIT  = 0.15

# Map numeric labels to readable names
# Labels 1-9 = digits, 10-35 = A-Z (adjust if your dataset differs)
def make_label_map(unique_labels):
    """Auto-generate a human-readable name for each numeric label."""
    sorted_labels = sorted(unique_labels)
    label_map = {}
    digit_count = 0
    letter_count = 0
    for l in sorted_labels:
        # Heuristic: if 35 classes, assume 9 digits + 26 letters
        if digit_count < 9:
            label_map[l] = str(digit_count + 1)
            digit_count += 1
        else:
            label_map[l] = chr(ord('A') + letter_count)
            letter_count += 1
    return label_map


def load_and_prepare(train_csv, test_csv):
    print("Loading CSVs...")
    train_df = pd.read_csv(train_csv)
    test_df  = pd.read_csv(test_csv)
    print(f"  Train: {train_df.shape}  Test: {test_df.shape}")

    pixel_cols = [c for c in train_df.columns if c.startswith("pixel")]

    # Extract features and reshape to 28x28x1
    X_train = train_df[pixel_cols].values.astype(np.float32) / 255.0
    X_test  = test_df[pixel_cols].values.astype(np.float32) / 255.0
    X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    X_test  = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    # Encode labels
    le = LabelEncoder()
    y_train = le.fit_transform(train_df["label"].values)
    y_test  = le.transform(test_df["label"].values)

    # Build readable class names
    label_map  = make_label_map(le.classes_)
    class_names = [label_map[c] for c in le.classes_]

    print(f"  Classes ({len(class_names)}): {class_names}")
    return X_train, y_train, X_test, y_test, class_names


def build_model(num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),

        # Augmentation (training only)
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomTranslation(0.05, 0.05),

        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ], name="isl_pixel_classifier")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    X_train, y_train, X_test, y_test, class_names = load_and_prepare(
        TRAIN_CSV, TEST_CSV
    )

    with open(CKPT_DIR / "class_names_landmarks.json", "w") as f:
        json.dump(class_names, f)

    model = build_model(len(class_names))
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            CKPT_DIR / "best_landmark_model.keras",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            verbose=1,
        ),
    ]

    model.fit(
        X_train, y_train,
        validation_split=VAL_SPLIT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy : {test_acc:.4f}")
    print(f"Test loss     : {test_loss:.4f}")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print(classification_report(y_test, y_pred,
                                 target_names=class_names, digits=3))

    model.save(CKPT_DIR / "final_landmark_model.keras")
    print(f"\nSaved to {CKPT_DIR}/")


if __name__ == "__main__":
    main()