"""
Complete training pipeline for OncologyBreastScanner.

Usage:
  python train.py --train-csv Training_set.csv --image-dir /path/to/images
  python train.py --train-csv Training_set.csv --image-dir /path/to/images --val-csv Validation_set.csv
  python train.py --train-csv Training_set.csv --image-dir /path/to/images --epochs 50 --batch-size 32
"""

import os
import sys
import math
import argparse
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from model import build_model, save_model, MODEL_PATH, WEIGHTS_PATH
from preprocess import preprocess_mammogram, normalize_for_model

CLASS_NAMES = [
    'Density1Benign', 'Density1Malignant',
    'Density2Benign', 'Density2Malignant',
    'Density3Benign', 'Density3Malignant',
    'Density4Benign', 'Density4Malignant',
]
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}


def set_seeds(seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)


class MammogramGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataframe, image_dir, batch_size=32, target_size=(224, 224),
                 augment=True, shuffle=True, apply_clahe=True, apply_sharpen=True):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.target_size = target_size
        self.augment = augment
        self.shuffle = shuffle
        self.apply_clahe = apply_clahe
        self.apply_sharpen = apply_sharpen
        self.indexes = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return math.ceil(len(self.df) / self.batch_size)

    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_df = self.df.iloc[batch_indexes]
        X, y = [], []
        for _, row in batch_df.iterrows():
            img_path = os.path.join(self.image_dir, row["filename"])
            img = self._load_image(img_path)
            if img is None:
                continue
            X.append(img)
            label = CLASS_TO_INDEX[row["label"]]
            y.append(tf.keras.utils.to_categorical(label, num_classes=8))
        return np.array(X), np.array(y)

    def _load_image(self, path):
        import cv2
        img = cv2.imread(str(path))
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = preprocess_mammogram(img, apply_clahe=self.apply_clahe,
                                   apply_sharpen=self.apply_sharpen,
                                   target_size=self.target_size)
        if img is None:
            return None
        img = normalize_for_model(img)
        if self.augment:
            img = self._augment(img)
        return img

    def _augment(self, img):
        if np.random.random() > 0.5:
            img = np.fliplr(img)
        angle = np.random.uniform(-10, 10)
        if abs(angle) > 1:
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        if np.random.random() > 0.7:
            scale = np.random.uniform(0.9, 1.1)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), 0, scale)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        if np.random.random() > 0.7:
            brightness = np.random.uniform(0.85, 1.15)
            img = np.clip(img * brightness, 0, 1)
        return img

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


def main():
    parser = argparse.ArgumentParser(description="Train breast cancer classifier")
    parser.add_argument("--train-csv", required=True, help="CSV with filename,label columns for training")
    parser.add_argument("--val-csv", help="CSV for validation (optional, will split from train if not given)")
    parser.add_argument("--image-dir", required=True, help="Directory containing images")
    parser.add_argument("--epochs", type=int, default=60, help="Total epochs (default: 60)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate (default: 1e-4)")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate (default: 0.5)")
    parser.add_argument("--fine-tune-layers", type=int, default=50,
                        help="Number of DenseNet layers to unfreeze (default: 50)")
    parser.add_argument("--val-split", type=float, default=0.15,
                        help="Validation split ratio (default: 0.15)")
    parser.add_argument("--head-only-epochs", type=int, default=20,
                        help="Epochs training only the head before unfreezing (default: 20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", default="training_output", help="Output directory for checkpoints")
    parser.add_argument("--resume", help="Path to checkpoint weights to resume from")
    args = parser.parse_args()

    set_seeds(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading dataset...")
    train_df = pd.read_csv(args.train_csv)

    if args.val_csv:
        val_df = pd.read_csv(args.val_csv)
    else:
        train_df, val_df = train_test_split(
            train_df, test_size=args.val_split,
            stratify=train_df["label"], random_state=args.seed
        )

    print(f"  Training samples:   {len(train_df)}")
    print(f"  Validation samples: {len(val_df)}")
    print(f"  Classes: {len(CLASS_NAMES)}")

    class_counts = train_df["label"].value_counts()
    print("\n  Class distribution (training):")
    for cls in CLASS_NAMES:
        count = class_counts.get(cls, 0)
        bar = "#" * max(1, count // 20)
        print(f"    {cls:25s} {count:5d} {bar}")

    class_weights = compute_class_weight(
        "balanced",
        classes=np.array(CLASS_NAMES),
        y=train_df["label"].values
    )
    class_weight_dict = {i: class_weights[i] for i in range(len(CLASS_NAMES))}
    print(f"\n  Class weights (min={min(class_weights):.3f}, max={max(class_weights):.3f})")

    train_gen = MammogramGenerator(
        train_df, args.image_dir,
        batch_size=args.batch_size,
        augment=True, shuffle=True
    )
    val_gen = MammogramGenerator(
        val_df, args.image_dir,
        batch_size=args.batch_size,
        augment=False, shuffle=False
    )

    print("\nPhase 1: Training head only (backbone frozen)...")
    model = build_model(
        fine_tune_layers=0,
        dropout_rate=args.dropout
    )

    if args.resume and os.path.exists(args.resume):
        model.load_weights(args.resume)
        print(f"  Resumed weights from {args.resume}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"]
    )

    callbacks_phase1 = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(args.output_dir, "phase1_best.weights.h5"),
            monitor="val_accuracy", save_best_only=True, save_weights_only=True,
            verbose=1, mode="max"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3,
            min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True, verbose=1
        ),
    ]

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.head_only_epochs,
        callbacks=callbacks_phase1,
        class_weight=class_weight_dict,
        verbose=1
    )

    print("\nPhase 2: Fine-tuning with unfrozen layers...")
    model2 = build_model(
        fine_tune_layers=args.fine_tune_layers,
        dropout_rate=args.dropout
    )
    model2.load_weights(os.path.join(args.output_dir, "phase1_best.weights.h5"))

    model2.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr / 10),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"]
    )

    callbacks_phase2 = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(args.output_dir, "best.weights.h5"),
            monitor="val_accuracy", save_best_only=True, save_weights_only=True,
            verbose=1, mode="max"
        ),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(args.output_dir, "checkpoint_epoch_{epoch:02d}.weights.h5"),
            save_weights_only=True, verbose=0
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4,
            min_lr=1e-7, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.CSVLogger(os.path.join(args.output_dir, "training_log.csv")),
    ]

    history = model2.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        initial_epoch=args.head_only_epochs,
        callbacks=callbacks_phase2,
        class_weight=class_weight_dict,
        verbose=1
    )

    model2.save(MODEL_PATH)
    model2.save_weights(WEIGHTS_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Weights saved to {WEIGHTS_PATH}")

    best_epoch = np.argmax(history.history["val_accuracy"]) + args.head_only_epochs
    best_val_acc = max(history.history["val_accuracy"])
    print(f"\nBest validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%) at epoch {best_epoch}")

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(14, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history["loss"], label="Train Loss")
        plt.plot(history.history["val_loss"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss")
        plt.subplot(1, 2, 2)
        plt.plot(history.history["accuracy"], label="Train Acc")
        plt.plot(history.history["val_accuracy"], label="Val Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Accuracy")
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "training_curves.png"), dpi=150)
        print(f"Training curves saved to {os.path.join(args.output_dir, 'training_curves.png')}")
    except Exception:
        pass

    print("\nDone. Run evaluation with:")
    print(f"  python evaluate.py --csv {args.val_csv or args.train_csv} --image-dir {args.image_dir}")


if __name__ == "__main__":
    main()
