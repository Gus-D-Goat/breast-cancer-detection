import os
import sys
import argparse
import csv
import numpy as np
import tensorflow as tf
from pathlib import Path
from collections import defaultdict

from model import load_model
from preprocess import load_and_preprocess

CLASS_NAMES = [
    'Benign with Density=1', 'Malignant with Density=1',
    'Benign with Density=2', 'Malignant with Density=2',
    'Benign with Density=3', 'Malignant with Density=3',
    'Benign with Density=4', 'Malignant with Density=4'
]

SHORT_NAMES = ['B D1', 'M D1', 'B D2', 'M D2', 'B D3', 'M D3', 'B D4', 'M D4']

LABEL_TO_INDEX = {
    'Density1Benign': 0, 'Density1Malignant': 1,
    'Density2Benign': 2, 'Density2Malignant': 3,
    'Density3Benign': 4, 'Density3Malignant': 5,
    'Density4Benign': 6, 'Density4Malignant': 7,
}

INDEX_TO_LABEL = {v: k for k, v in LABEL_TO_INDEX.items()}


def parse_label_from_filename(filename):
    import re
    stem = Path(filename).stem
    for key in LABEL_TO_INDEX:
        if re.search(r'\b' + re.escape(key) + r'\b', stem, re.IGNORECASE):
            return key
    return None


def load_from_csv(csv_path, image_dir):
    images, labels, filenames = [], [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row.get("filename")
            label = row.get("label")
            if not fname:
                continue
            img = load_and_preprocess(os.path.join(image_dir, fname))
            if img is None:
                print(f"  Warning: could not load {os.path.join(image_dir, fname)}", file=sys.stderr)
                continue
            images.append(img)
            filenames.append(fname)
            labels.append(LABEL_TO_INDEX.get(label, -1))
    return np.array(images), np.array(labels), filenames


def load_from_directory(image_dir):
    images, labels, filenames = [], [], []
    for fname in sorted(os.listdir(image_dir)):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue
        label_raw = parse_label_from_filename(fname)
        if label_raw is None:
            print(f"  Warning: could not parse label from {fname}, skipping", file=sys.stderr)
            continue
        img = load_and_preprocess(os.path.join(image_dir, fname))
        if img is None:
            print(f"  Warning: could not load {fname}", file=sys.stderr)
            continue
        images.append(img)
        labels.append(LABEL_TO_INDEX[label_raw])
        filenames.append(fname)
    return np.array(images), np.array(labels), filenames


def compute_metrics(y_true, y_pred, y_prob):
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, roc_auc_score, top_k_accuracy_score,
        average_precision_score
    )
    import pandas as pd

    n_classes = len(CLASS_NAMES)
    acc = accuracy_score(y_true, y_pred)

    try:
        top2_acc = top_k_accuracy_score(y_true, y_prob, k=2, labels=range(n_classes))
    except Exception:
        top2_acc = None

    precision = precision_score(y_true, y_pred, average=None, zero_division=0, labels=range(n_classes))
    recall = recall_score(y_true, y_pred, average=None, zero_division=0, labels=range(n_classes))
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0, labels=range(n_classes))

    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))

    try:
        auc_ovr = roc_auc_score(y_true, y_prob, multi_class="ovr", labels=range(n_classes))
    except Exception:
        auc_ovr = None

    per_class_auc = []
    per_class_ap = []
    for i in range(n_classes):
        y_true_bin = (y_true == i).astype(int)
        if len(np.unique(y_true_bin)) < 2:
            per_class_auc.append(None)
            per_class_ap.append(None)
            continue
        try:
            per_class_auc.append(roc_auc_score(y_true_bin, y_prob[:, i]))
        except Exception:
            per_class_auc.append(None)
        try:
            per_class_ap.append(average_precision_score(y_true_bin, y_prob[:, i]))
        except Exception:
            per_class_ap.append(None)

    return {
        "accuracy": acc,
        "top2_accuracy": top2_acc,
        "precision_per_class": precision,
        "recall_per_class": recall,
        "f1_per_class": f1,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm,
        "auc_ovr": auc_ovr,
        "auc_per_class": per_class_auc,
        "ap_per_class": per_class_ap,
    }


def compute_calibration(y_true, y_prob, n_bins=10):
    n_classes = len(CLASS_NAMES)
    results = {}
    for i in range(n_classes):
        y_true_bin = (y_true == i).astype(int)
        conf = y_prob[:, i]
        if len(np.unique(y_true_bin)) < 2:
            continue
        bins = np.linspace(0, 1, n_bins + 1)
        bin_acc = np.zeros(n_bins)
        bin_conf = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)
        for b in range(n_bins):
            mask = (conf >= bins[b]) & (conf < bins[b + 1])
            if b == n_bins - 1:
                mask = (conf >= bins[b]) & (conf <= bins[b + 1])
            bin_counts[b] = np.sum(mask)
            if bin_counts[b] > 0:
                bin_acc[b] = np.mean(y_true_bin[mask])
                bin_conf[b] = np.mean(conf[mask])
        ece = np.sum(bin_counts * np.abs(bin_acc - bin_conf)) / np.sum(bin_counts) if np.sum(bin_counts) > 0 else None
        mce = np.max(np.abs(bin_acc - bin_conf)) if np.sum(bin_counts) > 0 else None
        results[CLASS_NAMES[i]] = {
            "ece": ece,
            "mce": mce,
            "bin_acc": bin_acc,
            "bin_conf": bin_conf,
            "bin_counts": bin_counts,
        }
    overall_ece = np.mean([r["ece"] for r in results.values() if r["ece"] is not None])
    overall_mce = np.max([r["mce"] for r in results.values() if r["mce"] is not None])
    return results, overall_ece, overall_mce


def bootstrap_metrics(y_true, y_pred, y_prob, n_bootstrap=1000, seed=42):
    from sklearn.metrics import accuracy_score, f1_score
    rng = np.random.RandomState(seed)
    n = len(y_true)
    accs, f1s = [], []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        accs.append(accuracy_score(y_true[idx], y_pred[idx]))
        f1s.append(f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0))
    accs = np.array(accs)
    f1s = np.array(f1s)
    return {
        "accuracy_ci": (np.percentile(accs, 2.5), np.percentile(accs, 97.5)),
        "f1_macro_ci": (np.percentile(f1s, 2.5), np.percentile(f1s, 97.5)),
    }


def print_report(metrics, y_true, y_pred, filenames, calibration_results=None,
                 overall_ece=None, overall_mce=None, boot_ci=None, y_prob=None):
    print("=" * 72)
    print("CLASSIFICATION REPORT")
    print("=" * 72)

    ci_str = ""
    if boot_ci:
        ci_str = f"  (95% CI: {boot_ci['accuracy_ci'][0]:.4f}–{boot_ci['accuracy_ci'][1]:.4f})"
    print(f"\nOverall Accuracy:          {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%){ci_str}")

    if metrics['top2_accuracy'] is not None:
        print(f"Top-2 Accuracy:            {metrics['top2_accuracy']:.4f}  ({metrics['top2_accuracy']*100:.2f}%)")
    print(f"Macro Precision:           {metrics['precision_macro']:.4f}")
    print(f"Macro Recall:              {metrics['recall_macro']:.4f}")
    if boot_ci:
        print(f"Macro F1-Score:            {metrics['f1_macro']:.4f}  (95% CI: {boot_ci['f1_macro_ci'][0]:.4f}–{boot_ci['f1_macro_ci'][1]:.4f})")
    else:
        print(f"Macro F1-Score:            {metrics['f1_macro']:.4f}")
    print(f"Weighted F1-Score:         {metrics['f1_weighted']:.4f}")

    if overall_ece is not None:
        print(f"Expected Calibration Err: {overall_ece:.4f}")
        print(f"Max Calibration Err:      {overall_mce:.4f}")

    if metrics['auc_ovr'] is not None:
        print(f"ROC-AUC (OvR):             {metrics['auc_ovr']:.4f}")

    print("\n" + "-" * 72)
    print(f"{'Class':<28} {'Prec':>7} {'Recall':>7} {'F1':>7} {'AUC':>7} {'AP':>7} {'ECE':>7} {'Supp':>5}")
    print("-" * 72)

    for i in range(len(CLASS_NAMES)):
        support = int(np.sum(y_true == i))
        auc_val = metrics['auc_per_class'][i]
        auc_str = f"{auc_val:.3f}" if auc_val is not None else "  N/A"
        ap_val = metrics['ap_per_class'][i]
        ap_str = f"{ap_val:.3f}" if ap_val is not None else "  N/A"
        ece_val = None
        if calibration_results and CLASS_NAMES[i] in calibration_results:
            ece_val = calibration_results[CLASS_NAMES[i]]["ece"]
        ece_str = f"{ece_val:.3f}" if ece_val is not None else "  N/A"
        print(f"{CLASS_NAMES[i]:<28} {metrics['precision_per_class'][i]:>7.3f} {metrics['recall_per_class'][i]:>7.3f} {metrics['f1_per_class'][i]:>7.3f} {auc_str:>7} {ap_str:>7} {ece_str:>7} {support:>5}")

    print("-" * 72)

    print("\n\nCONFUSION MATRIX")
    cm = metrics['confusion_matrix']
    header = f"{'':>14}" + "".join(f"{s:>12}" for s in SHORT_NAMES)
    print(header)
    for i, row_label in enumerate(SHORT_NAMES):
        row = f"{row_label:>14}"
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                row += f"{cm[i, j]:>12}"
            else:
                row += f"{'.':>12}"
        print(row)
    print(f"{'':>14}" + "".join(f"{'---':>12}" for _ in SHORT_NAMES))
    print(f"(rows = true, columns = predicted)\n")

    incorrect = y_true != y_pred
    if np.any(incorrect) and 'y_prob' in locals():
        print("\n" + "-" * 72)
        print(f"MISCLASSIFIED SAMPLES ({int(np.sum(incorrect))} total)")
        print("-" * 72)
        indices = np.where(incorrect)[0]
        for idx in indices[:20]:
            fname = filenames[idx] if idx < len(filenames) else f"sample_{idx}"
            true_label = INDEX_TO_LABEL.get(y_true[idx], "unknown")
            pred_label = INDEX_TO_LABEL.get(y_pred[idx], "unknown")
            conf = np.max(y_prob[idx])
            print(f"  {fname:<40} true={true_label:<22} pred={pred_label:<22} conf={conf:.3f}")
        if len(indices) > 20:
            print(f"  ... and {len(indices) - 20} more")


def plot_confusion_matrix(metrics, save_path=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    cm = metrics['confusion_matrix']
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=SHORT_NAMES, yticklabels=SHORT_NAMES)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted', fontsize=13)
    plt.ylabel('True', fontsize=13)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Confusion matrix saved to {save_path}")
    plt.show()


def plot_roc_curves(y_true, y_prob, save_path=None):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    from itertools import cycle
    n_classes = len(CLASS_NAMES)
    plt.figure(figsize=(10, 8))
    colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'])
    plotted = False
    for i, color in zip(range(n_classes), colors):
        y_true_bin = (y_true == i).astype(int)
        if len(np.unique(y_true_bin)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true_bin, y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2, label=f'{SHORT_NAMES[i]} (AUC={roc_auc:.3f})')
        plotted = True
    if not plotted:
        print("Skipping ROC plot: need at least 2 classes with positive samples")
        return
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.title('ROC Curves (One-vs-Rest)', fontsize=16, fontweight='bold')
    plt.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"ROC curves saved to {save_path}")
    plt.show()


def plot_pr_curves(y_true, y_prob, save_path=None):
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, average_precision_score
    from itertools import cycle
    n_classes = len(CLASS_NAMES)
    plt.figure(figsize=(10, 8))
    colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'])
    plotted = False
    for i, color in zip(range(n_classes), colors):
        y_true_bin = (y_true == i).astype(int)
        if len(np.unique(y_true_bin)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_true_bin, y_prob[:, i])
        ap = average_precision_score(y_true_bin, y_prob[:, i])
        plt.plot(recall, precision, color=color, lw=2, label=f'{SHORT_NAMES[i]} (AP={ap:.3f})')
        plotted = True
    if not plotted:
        return
    plt.xlabel('Recall', fontsize=13)
    plt.ylabel('Precision', fontsize=13)
    plt.title('Precision-Recall Curves (One-vs-Rest)', fontsize=16, fontweight='bold')
    plt.legend(loc='lower left', fontsize=9)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"PR curves saved to {save_path}")
    plt.show()


def plot_reliability_diagram(calibration_results, save_path=None):
    import matplotlib.pyplot as plt
    n_classes = len(CLASS_NAMES)
    n_cols = 4
    n_rows = (n_classes + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()
    for i in range(n_classes):
        ax = axes[i]
        name = CLASS_NAMES[i]
        if name not in calibration_results:
            ax.set_visible(False)
            continue
        res = calibration_results[name]
        if np.sum(res["bin_counts"]) == 0:
            ax.set_visible(False)
            continue
        bin_centers = (np.linspace(0, 1, len(res["bin_conf"]) + 1)[:-1] + np.linspace(0, 1, len(res["bin_conf"]) + 1)[1:]) / 2
        ax.bar(bin_centers, res["bin_acc"] - res["bin_conf"],
               width=0.1, color='salmon', alpha=0.7, label='Gap')
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.scatter(res["bin_conf"], res["bin_acc"], c='navy', s=30)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f'{SHORT_NAMES[i]}  ECE={res["ece"]:.3f}', fontsize=10)
        ax.set_xlabel('Confidence', fontsize=8)
        ax.set_ylabel('Accuracy', fontsize=8)
    for j in range(n_classes, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Reliability diagram saved to {save_path}")
    plt.show()


def plot_prediction_distribution(y_true, y_pred, save_path=None):
    import matplotlib.pyplot as plt
    true_counts = np.bincount(y_true, minlength=len(CLASS_NAMES))
    pred_counts = np.bincount(y_pred, minlength=len(CLASS_NAMES))
    x = np.arange(len(CLASS_NAMES))
    w = 0.35
    plt.figure(figsize=(12, 6))
    plt.bar(x - w/2, true_counts, w, label='True', color='#3498db', alpha=0.8)
    plt.bar(x + w/2, pred_counts, w, label='Predicted', color='#e74c3c', alpha=0.8)
    plt.xticks(x, SHORT_NAMES, rotation=30, ha='right', fontsize=10)
    plt.ylabel('Count', fontsize=13)
    plt.title('True vs Predicted Class Distribution', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Prediction distribution saved to {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Evaluate breast cancer classifier")
    parser.add_argument("--csv", help="Path to CSV with filename,label columns")
    parser.add_argument("--image-dir", default="Test_images",
                        help="Directory of test images (default: Test_images)")
    parser.add_argument("--output-dir", default="evaluation_output",
                        help="Directory to save evaluation outputs")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip generating plots (headless mode)")
    parser.add_argument("--no-bootstrap", action="store_true",
                        help="Skip bootstrap confidence intervals (faster)")
    parser.add_argument("--model-path", default="model/model.h5",
                        help="Path to saved model")
    parser.add_argument("--weights-path", default="weights/modeldense1.h5",
                        help="Path to weights file")
    parser.add_argument("--no-clahe", action="store_true",
                        help="Disable CLAHE preprocessing")
    parser.add_argument("--no-sharpen", action="store_true",
                        help="Disable sharpening preprocessing")
    args = parser.parse_args()

    print("Loading model...")
    model = load_model(args.model_path)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
        metrics=["accuracy"],
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    )
    if os.path.exists(args.weights_path):
        model.load_weights(args.weights_path)
    print("Model loaded.\n")

    if args.csv:
        print(f"Loading test data from CSV: {args.csv}")
        X, y_true, filenames = load_from_csv(args.csv, args.image_dir or ".")
    else:
        print(f"Loading test data from directory: {args.image_dir}")
        X, y_true, filenames = load_from_directory(args.image_dir)

    if len(X) == 0:
        print("Error: no valid test images found!", file=sys.stderr)
        sys.exit(1)

    unlabeled = int(np.sum(y_true == -1))
    labeled_mask = y_true >= 0
    print(f"Loaded {len(X)} images ({int(np.sum(labeled_mask))} labeled, {unlabeled} unlabeled)\n")

    y_prob = model.predict(X, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)

    y_true_labeled = y_true[labeled_mask]
    y_pred_labeled = y_pred[labeled_mask]
    y_prob_labeled = y_prob[labeled_mask]

    labeled_filenames = [f for i, f in enumerate(filenames) if labeled_mask[i]]

    print("\n")
    metrics = compute_metrics(y_true_labeled, y_pred_labeled, y_prob_labeled)

    cal_results = None
    overall_ece = None
    overall_mce = None
    if not args.no_plots:
        cal_results, overall_ece, overall_mce = compute_calibration(y_true_labeled, y_prob_labeled)

    boot_ci = None
    if not args.no_bootstrap:
        print("Computing bootstrap confidence intervals (1000 resamples)...")
        boot_ci = bootstrap_metrics(y_true_labeled, y_pred_labeled, y_prob_labeled)
        print("Done.\n")

    print_report(metrics, y_true_labeled, y_pred_labeled, labeled_filenames,
                 cal_results, overall_ece, overall_mce, boot_ci, y_prob_labeled)

    if not args.no_plots:
        os.makedirs(args.output_dir, exist_ok=True)
        plot_confusion_matrix(metrics, save_path=os.path.join(args.output_dir, "confusion_matrix.png"))
        plot_roc_curves(y_true_labeled, y_prob_labeled,
                        save_path=os.path.join(args.output_dir, "roc_curves.png"))
        plot_pr_curves(y_true_labeled, y_prob_labeled,
                       save_path=os.path.join(args.output_dir, "pr_curves.png"))
        plot_prediction_distribution(y_true_labeled, y_pred_labeled,
                                     save_path=os.path.join(args.output_dir, "prediction_distribution.png"))
        if cal_results:
            plot_reliability_diagram(cal_results,
                                     save_path=os.path.join(args.output_dir, "reliability_diagram.png"))

        import pandas as pd
        results_df = pd.DataFrame({
            "filename": labeled_filenames,
            "true_label": [INDEX_TO_LABEL.get(y_true_labeled[i], "unknown") for i in range(len(labeled_filenames))],
            "predicted_label": [INDEX_TO_LABEL.get(y_pred_labeled[i], "unknown") for i in range(len(labeled_filenames))],
            "correct": y_true_labeled == y_pred_labeled,
            "confidence": [np.max(y_prob_labeled[i]) for i in range(len(labeled_filenames))],
        })
        for i, name in enumerate(CLASS_NAMES):
            results_df[f"prob_{name}"] = y_prob_labeled[:, i]
        csv_path = os.path.join(args.output_dir, "predictions.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"\nDetailed predictions saved to {csv_path}")

    if unlabeled > 0:
        print(f"\nNote: {unlabeled} unlabeled sample(s) had predictions saved")
        print(f"      to {os.path.join(args.output_dir, 'unlabeled_predictions.csv')}")
        unlabeled_mask = ~labeled_mask
        if np.any(unlabeled_mask):
            import pandas as pd
            ul_df = pd.DataFrame({
                "filename": [f for i, f in enumerate(filenames) if unlabeled_mask[i]],
                "predicted_label": [INDEX_TO_LABEL.get(y_pred[i], "unknown") for i in range(len(filenames)) if unlabeled_mask[i]],
                "confidence": [np.max(y_prob[i]) for i in range(len(filenames)) if unlabeled_mask[i]],
            })
            for i, name in enumerate(CLASS_NAMES):
                ul_df[f"prob_{name}"] = y_prob[unlabeled_mask, i]
            ul_path = os.path.join(args.output_dir, "unlabeled_predictions.csv")
            ul_df.to_csv(ul_path, index=False)
            print(f"      saved to {ul_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
