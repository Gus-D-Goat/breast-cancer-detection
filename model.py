from pathlib import Path

import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, GlobalAveragePooling2D, GlobalMaxPooling2D, Concatenate, Input
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1_l2

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "model.h5"
WEIGHTS_PATH = BASE_DIR / "weights" / "modeldense1.h5"

LEGACY_LAYER_NAMES = (
    "densenet201",
    "batch_normalization",
    "dense",
    "batch_normalization_1",
    "dense_1",
)


def build_model(input_shape=(224, 224, 3), n_classes=8, dropout_rate=0.5, pretrained=True):
    conv_base = DenseNet201(
        input_shape=input_shape,
        include_top=False,
        pooling=None,
        weights="imagenet" if pretrained else None
    )
    conv_base.trainable = True

    inputs = Input(shape=input_shape)
    x = conv_base(inputs)
    avg_pool = GlobalAveragePooling2D()(x)
    max_pool = GlobalMaxPooling2D()(x)
    x = Concatenate()([avg_pool, max_pool])
    x = BatchNormalization()(x)
    x = Dense(1024, activation='relu', kernel_regularizer=l1_l2(0.001))(x)
    x = Dropout(dropout_rate)(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu', kernel_regularizer=l1_l2(0.001))(x)
    x = Dropout(dropout_rate)(x)
    x = BatchNormalization()(x)
    outputs = Dense(n_classes, activation='softmax')(x)
    return Model(inputs=inputs, outputs=outputs)


def build_legacy_model(input_shape=(224, 224, 3), n_classes=8, pretrained=False):
    """Sequential DenseNet201 classifier matching the published weights file."""
    conv_base = DenseNet201(
        input_shape=input_shape,
        include_top=False,
        pooling="max",
        weights="imagenet" if pretrained else None,
    )
    conv_base._name = "densenet201"

    model = tf.keras.Sequential(name="legacy")
    model.add(conv_base)
    model.add(BatchNormalization(name="batch_normalization"))
    model.add(Dense(2048, activation="relu", kernel_regularizer=l1_l2(0.01), name="dense"))
    model.add(BatchNormalization(name="batch_normalization_1"))
    model.add(Dense(n_classes, activation="softmax", name="dense_1"))
    return model


def saved_weight_layer_names(weights_path):
    """Return the top-level layer names stored in a weights-only HDF5 file."""
    try:
        import h5py

        with h5py.File(weights_path, "r") as handle:
            group = handle
            if "layer_names" not in group.attrs:
                if "model_weights" not in group:
                    return None
                group = group["model_weights"]
            names = group.attrs.get("layer_names", [])
            return [
                name.decode() if isinstance(name, bytes) else str(name)
                for name in names
            ]
    except Exception:
        return None


def build_model_for_weights(weights_path, input_shape=(224, 224, 3), n_classes=8, pretrained=False):
    """Build the architecture that matches the layer names saved in the weights file."""
    names = saved_weight_layer_names(weights_path)
    if names is not None and len(names) == len(LEGACY_LAYER_NAMES) and set(names) == set(LEGACY_LAYER_NAMES):
        return build_legacy_model(
            input_shape=input_shape,
            n_classes=n_classes,
            pretrained=pretrained,
        )
    return build_model(
        input_shape=input_shape,
        n_classes=n_classes,
        pretrained=pretrained,
    )


def load_model(path=MODEL_PATH, weights_path=WEIGHTS_PATH, pretrained=True):
    path = Path(path)
    weights_path = Path(weights_path)
    if path.exists() and path.suffix == ".h5":
        try:
            return tf.keras.models.load_model(path)
        except Exception:
            pass
    if weights_path.exists():
        model = build_model_for_weights(weights_path, pretrained=pretrained)
    else:
        model = build_model(pretrained=pretrained)
    if weights_path.exists():
        model.load_weights(weights_path)
        print(f"Loaded weights from {weights_path}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
        metrics=["accuracy"],
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    )
    return model


def save_model(model, path=MODEL_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
