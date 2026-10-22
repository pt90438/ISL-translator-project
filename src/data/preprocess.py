import tensorflow as tf
from tensorflow.keras import layers

# No RandomFlip: mirroring a hand shape can change the meaning of a sign.
data_augmentation = tf.keras.Sequential(
    [
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.1),
        layers.RandomBrightness(0.15),
        layers.RandomContrast(0.15),
        layers.RandomTranslation(0.05, 0.05),
    ],
    name="augmentation",
)


def augment(image, label):
    return data_augmentation(image, training=True), label