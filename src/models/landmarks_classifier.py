"""Simple MLP classifier for static ISL signs from MediaPipe landmarks.

Input: flattened (x, y, z) coordinates for both hands — 126 features
       (42 landmarks × 3 coords). The 'uses_two_hands' column is dropped.
Output: softmax over N sign classes.
"""
import tensorflow as tf
from tensorflow.keras import layers, models


def build_landmark_model(num_features, num_classes, dropout=0.3):
    """
    Args:
        num_features: number of landmark coordinate columns (e.g. 126)
        num_classes:  number of sign classes (e.g. 26 for A-Z)
        dropout:      dropout rate between dense layers
    Returns:
        compiled Keras model
    """
    model = models.Sequential([
        layers.Input(shape=(num_features,)),

        # Normalise inputs within the network so scale differences don't matter
        layers.BatchNormalization(),

        layers.Dense(512, activation="relu"),
        layers.Dropout(dropout),

        layers.Dense(256, activation="relu"),
        layers.Dropout(dropout),

        layers.Dense(128, activation="relu"),
        layers.Dropout(dropout / 2),

        layers.Dense(num_classes, activation="softmax"),
    ], name="isl_landmark_classifier")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model