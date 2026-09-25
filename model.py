import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, GlobalAveragePooling2D, GlobalMaxPooling2D, Concatenate, Input
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1_l2

MODEL_PATH = "model/model.h5"
WEIGHTS_PATH = "weights/modeldense1.h5"


def build_model(input_shape=(224, 224, 3), n_classes=8, dropout_rate=0.5):
    conv_base = DenseNet201(
        input_shape=input_shape,
        include_top=False,
        pooling=None,
        weights='imagenet'
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


def load_model(path=MODEL_PATH, weights_path=WEIGHTS_PATH):
    if os.path.exists(path) and path.endswith('.h5'):
        try:
            return tf.keras.models.load_model(path)
        except Exception:
            pass
    model = build_model()
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
        print(f"Loaded weights from {weights_path}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
        metrics=["accuracy"],
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    )
    return model


def save_model(model, path=MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
