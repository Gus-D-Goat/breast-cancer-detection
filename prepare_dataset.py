"""
Prepares CBIS-DDSM dataset for training and evaluation.

Usage:
  python prepare_dataset.py --kaggle-dir cbis_ddsm --output-dir data
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import cv2

CLASS_LABELS = [
    'Density1Benign', 'Density1Malignant',
    'Density2Benign', 'Density2Malignant',
    'Density3Benign', 'Density3Malignant',
    'Density4Benign', 'Density4Malignant',
]


def pathology_to_binary(p):
    p = str(p).strip().upper()
    if 'BENIGN' in p:
        return 'Benign'
    elif 'MALIGNANT' in p:
        return 'Malignant'
    return None


def density_to_int(d):
    try:
        return int(float(str(d).strip()))
    except (ValueError, TypeError):
        return None


def get_density_col(row):
    for col in ['breast density', 'breast_density', 'density']:
        val = row.get(col)
        if val is not None and not pd.isna(val):
            return val
    return None


def make_label(density, pathology):
    d = density_to_int(density)
    p = pathology_to_binary(pathology)
    if d is None or p is None or d < 1 or d > 4:
        return None
    label = f'Density{d}{p}'
    return label if label in CLASS_LABELS else None


def extract_uid(dicom_path):
    """Extract the last UID from a DICOM path like .../UID/000000.dcm"""
    parts = str(dicom_path).strip().split('/')
    if len(parts) >= 2:
        return parts[-2]
    return None


def find_jpeg_for_dicom(dicom_path, jpeg_dir):
    """Given a DICOM path, find the corresponding JPEG in the JPEG directory."""
    uid = extract_uid(dicom_path)
    if not uid:
        return None
    uid_jpeg_dir = os.path.join(jpeg_dir, uid)
    if not os.path.isdir(uid_jpeg_dir):
        return None
    files = sorted(os.listdir(uid_jpeg_dir))
    jpgs = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not jpgs:
        return None
    return os.path.join(uid_jpeg_dir, jpgs[0])


def main():
    parser = argparse.ArgumentParser(description="Prepare CBIS-DDSM dataset")
    parser.add_argument('--kaggle-dir', required=True, help='Directory of extracted Kaggle dataset')
    parser.add_argument('--output-dir', default='data', help='Output directory')
    parser.add_argument('--no-copy', action='store_true', help='Skip copying images')
    args = parser.parse_args()

    kaggle_dir = Path(args.kaggle_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_dir = kaggle_dir / 'csv'
    jpeg_dir = kaggle_dir / 'jpeg'

    if not csv_dir.exists() or not jpeg_dir.exists():
        print("Expected 'csv/' and 'jpeg/' directories inside --kaggle-dir.")
        sys.exit(1)

    records = []
    for csv_name in [
        'mass_case_description_train_set.csv',
        'mass_case_description_test_set.csv',
        'calc_case_description_train_set.csv',
        'calc_case_description_test_set.csv',
    ]:
        csv_path = csv_dir / csv_name
        if not csv_path.exists():
            continue
        split = 'test' if 'test' in csv_name else 'train'
        df = pd.read_csv(csv_path)
        print(f"  {csv_name}: {len(df)} rows")

        for _, row in df.iterrows():
            label = make_label(get_density_col(row), row.get('pathology'))
            if label is None:
                continue

            jpeg_path = None
            for col in ['cropped image file path', 'image file path']:
                dcm = row.get(col)
                if not dcm or pd.isna(dcm):
                    continue
                jpeg_path = find_jpeg_for_dicom(dcm, jpeg_dir)
                if jpeg_path:
                    break

            if jpeg_path:
                records.append({
                    'source_path': jpeg_path,
                    'label': label,
                    'split': split,
                })

    if not records:
        print("\nNo usable records found. Checking jpeg/ structure...")
        print(f"  jpeg/ subdirectories: {len(list(jpeg_dir.iterdir()))}")
        sys.exit(1)

    print(f"\n  Total usable: {len(records)}")
    labels = [r['label'] for r in records]
    counts = pd.Series(labels).value_counts()
    for cls in CLASS_LABELS:
        c = counts.get(cls, 0)
        if c > 0:
            print(f"    {cls:25s} {c:5d}")
    print(f"    {'Total':25s} {len(records):5d}")

    if args.no_copy:
        return

    print(f"\nCopying images to {output_dir / 'images/'}...")
    images_out = output_dir / 'images'
    images_out.mkdir(parents=True, exist_ok=True)

    train_rows, test_rows = [], []
    for idx, rec in enumerate(records):
        src = rec['source_path']
        dst = images_out / f"{rec['label']}_{idx:05d}.png"
        try:
            img = cv2.imread(src)
            if img is None:
                continue
            img = cv2.resize(img, (224, 224))
            cv2.imwrite(str(dst), img)
        except Exception:
            continue

        row = {'filename': dst.name, 'label': rec['label']}
        if rec['split'] == 'train':
            train_rows.append(row)
        else:
            test_rows.append(row)

    if train_rows:
        pd.DataFrame(train_rows).to_csv(output_dir / 'train_set.csv', index=False)
        print(f"  train_set.csv: {len(train_rows)} images")
    if test_rows:
        pd.DataFrame(test_rows).to_csv(output_dir / 'test_set.csv', index=False)
        print(f"  test_set.csv:  {len(test_rows)} images")

    print(f"\nDone! To train:")
    print(f"  python train.py --train-csv {output_dir / 'train_set.csv'} --image-dir {output_dir / 'images'}")
    if test_rows:
        print(f"  python evaluate.py --csv {output_dir / 'test_set.csv'} --image-dir {output_dir / 'images'}")


if __name__ == '__main__':
    main()
