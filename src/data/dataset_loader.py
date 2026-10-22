import tensorflow as tf

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def load_split(path, shuffle):
    return tf.keras.utils.image_dataset_from_directory(
        path,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=shuffle,
        pad_to_aspect_ratio=True
    )


TRAIN_DIR = "Training"
VAL_DIR = "Validation"
TEST_DIR = "Testing"


def get_datasets(data_dir):
    """Returns train_ds, val_ds, test_ds and the list of class names.

    Expects: dataset/Training, dataset/Validation, dataset/Testing,
    each containing one subfolder per letter. Other folders in dataset/
    (e.g. landmarks, Letters) are ignored.
    """
    train_ds = load_split(f"{data_dir}/{TRAIN_DIR}", shuffle=True)
    val_ds = load_split(f"{data_dir}/{VAL_DIR}", shuffle=False)
    test_ds = load_split(f"{data_dir}/{TEST_DIR}", shuffle=False)
    return train_ds, val_ds, test_ds, train_ds.class_names