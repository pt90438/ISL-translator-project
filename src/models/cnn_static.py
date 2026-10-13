import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


def build_model(num_classes, input_shape=(224, 224, 3)):
    base_model = MobileNetV2(
        input_shape=input_shape, include_top=False, weights="imagenet"
    )
    base_model.trainable = False  # phase 1: train only the new head

    inputs = tf.keras.Input(shape=input_shape)
    x = preprocess_input(inputs)  # scales pixels to [-1, 1] for MobileNetV2
    x = base_model(x, training=False)  # keep BatchNorm in inference mode
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="isl_static_classifier")
    return model, base_model