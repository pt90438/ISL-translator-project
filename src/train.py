import json
from pathlib import Path

import tensorflow as tf

from data.dataset_loader import get_datasets
from data.preprocess import augment
from models.cnn_static import build_model

# ---- Paths (relative to project root) ----
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset"
CKPT_DIR = ROOT / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)

# ---- Hyperparameters ----
EPOCHS_HEAD = 15       # phase 1: frozen backbone
EPOCHS_FINETUNE = 10   # phase 2: unfreeze top layers
FINETUNE_LAYERS = 30   # how many backbone layers to unfreeze
LR_HEAD = 1e-3
LR_FINETUNE = 1e-5


def main():
    # 1. Data
    train_ds, val_ds, test_ds, class_names = get_datasets(DATA_DIR)
    print(f"Classes ({len(class_names)}): {class_names}")

    # Save class order so inference uses the same label mapping
    with open(CKPT_DIR / "class_names.json", "w") as f:
        json.dump(class_names, f)

    # Augment training data only
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.map(augment, num_parallel_calls=autotune).prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)

    # 2. Model
    model, base_model = build_model(num_classes=len(class_names))

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            CKPT_DIR / "best_model.keras",
            monitor="val_accuracy",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2
        ),
    ]

    # 3. Phase 1: train the classification head only
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR_HEAD),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD, callbacks=callbacks)

    # 4. Phase 2: fine-tune the top backbone layers at a low learning rate
    base_model.trainable = True
    for layer in base_model.layers[:-FINETUNE_LAYERS]:
        layer.trainable = False
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False  # keep BN stats stable on small datasets

    model.compile(  # must re-compile after changing trainable flags
        optimizer=tf.keras.optimizers.Adam(LR_FINETUNE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_HEAD + EPOCHS_FINETUNE,
        initial_epoch=EPOCHS_HEAD,
        callbacks=callbacks,
    )

    # 5. Final evaluation on the untouched test set
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"\nTest accuracy: {test_acc:.4f}")

    model.save(CKPT_DIR / "final_model.keras")


if __name__ == "__main__":
    main()