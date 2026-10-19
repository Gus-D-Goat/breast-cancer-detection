# OncologyBreastScanner — Complete Codebase Documentation

> **A deep learning system for breast cancer detection from mammogram images.**
> Classifies mammograms into 8 categories combining tissue density (BI-RADS 1-4)
> with pathology (benign vs malignant). Uses DenseNet201 fine-tuned on the
> CBIS-DDSM dataset, wrapped in a Streamlit web interface.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Full Directory Structure](#2-full-directory-structure)
3. [Core Source Files](#3-core-source-files)
4. [Training & Evaluation Pipeline](#4-training--evaluation-pipeline)
5. [Notebooks](#5-notebooks)
6. [Data Files](#6-data-files)
7. [Media Directories](#7-media-directories)
8. [Configuration Files](#8-configuration-files)
9. [Model & Weights](#9-model--weights)
10. [Generated & Runtime Directories](#10-generated--runtime-directories)
11. [Data Flow: End-to-End Walkthrough](#11-data-flow-end-to-end-walkthrough)
12. [Classification System](#12-classification-system)

---

## 1. Project Overview

This project implements a **Convolutional Neural Network (CNN)** based on
**DenseNet201** pre-trained on ImageNet, fine-tuned for an 8-class breast cancer
classification task. The 8 classes combine two dimensions:

| Dimension | Values |
|-----------|--------|
| **Tissue Density (BI-RADS)** | 1 (fatty), 2 (scattered), 3 (heterogeneously dense), 4 (extremely dense) |
| **Pathology** | Benign (non-cancerous), Malignant (cancerous) |

| Class Index | Label |
|-------------|-------|
| 0 | Benign with Density=1 |
| 1 | Malignant with Density=1 |
| 2 | Benign with Density=2 |
| 3 | Malignant with Density=2 |
| 4 | Benign with Density=3 |
| 5 | Malignant with Density=3 |
| 6 | Benign with Density=4 |
| 7 | Malignant with Density=4 |

The system is wrapped in a **Streamlit web interface** where users upload
mammogram images and receive real-time predictions with probabilities for all 8
categories.

**Dataset:** [CBIS-DDSM](https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM)
(Curated Breast Imaging Subset of DDSM) — a standardized and curated subset of
the Digital Database for Screening Mammography.

---

## 2. Full Directory Structure

```
breast/
├── streamlit_app.py              # Streamlit web application (entry point)
├── inference.py                  # Shared inference layer (preprocess → predict → top label)
├── model.py                      # DenseNet201 model architecture definition
├── weights.py                    # Downloads pre-trained weights (env URL or Google Drive)
├── preprocess.py                 # Image preprocessing pipeline (CLAHE, sharpening, normalization)
├── train.py                      # Full two-phase training pipeline with CLI
├── evaluate.py                   # Comprehensive model evaluation suite
├── prepare_dataset.py            # Prepares CBIS-DDSM dataset from raw Kaggle format
│
├── requirements.txt              # Python package dependencies
├── packages.txt                  # apt packages installed by hosted platforms (libgomp1, libglib2.0-0)
├── Dockerfile                    # Container image (python:3.11-slim + Streamlit on 8501)
├── render.yaml                   # Render.com blueprint (docker runtime)
├── .streamlit/
│   └── config.toml               # Streamlit theme + server settings
├── tests/
│   └── test_inference.py         # Unit tests for the shared inference layer
├── .gitignore                    # Git exclusion rules
├── .dockerignore                 # Docker build-context exclusion rules
├── README.md                     # Project README (high-level overview)
│
├── model/                        # Saved model files (gitignored)
│   └── model.h5                  # Full serialized Keras model (~87 MB)
│
├── weights/                      # Pre-trained weight files (gitignored)
│   └── modeldense1.h5            # Weights-only file (~256 MB, downloaded at runtime)
│
├── data/                         # Processed dataset (gitignored)
│   ├── train_set.csv             # Training split CSV (~1192 images)
│   └── test_set.csv              # Test split CSV (~703 images)
│
├── Test_images/                  # Sample mammograms for quick testing (4 images)
│   ├── Density1Malignant.jpg
│   ├── Density2Malignant.jpg
│   ├── Density3Benign.jpg
│   └── Density4Benign.jpg
│
├── image/                        # README & UI media assets (8 PNG files)
│   ├── after cropping.png        # Post-cropping visualization
│   ├── Begign.png / benign.png   # Benign mammogram examples (duplicate/typo variant)
│   ├── Maliggnant.png / malignant.png  # Malignant mammogram examples
│   ├── prediction.png            # Prediction output screenshot
│   ├── processing.png            # Preprocessing pipeline visualization
│   └── webpage.png               # Historical interface screenshot (unused)
│
├── evaluation_output/            # Saved evaluation plots (gitignored)
│   ├── confusion_matrix.png
│   └── roc_curves.png
│
├── training_output/              # Training artifacts (gitignored, empty by default)
│
├── colab_training.ipynb          # Google Colab training notebook
├── colab_training_v2.ipynb       # Google Colab training notebook v2
├── Demo_Notebook.ipynb           # Original demo/exploration notebook
├── test.ipynb                    # Quick weight file validation notebook
│
├── csv_calc_train.csv            # Intermediate CBIS-DDSM calcification train CSV (gitignored)
├── csv_calc_test.csv             # Intermediate CBIS-DDSM calcification test CSV (gitignored)
├── csv_mass_train.csv            # Intermediate CBIS-DDSM mass train CSV (gitignored)
├── csv_mass_test.csv             # Intermediate CBIS-DDSM mass test CSV (gitignored)
│
├── LICENSE                       # Project license
└── .DS_Store                     # macOS metadata (gitignored)
```

---

## 3. Core Source Files

### 3.1 `streamlit_app.py` — Streamlit Web Application

**File:** `streamlit_app.py` (327 lines)
**Purpose:** The main entry point. Builds the full Streamlit UI for inference.

**Imports:**
- `os`, `pathlib.Path` — environment lookups and filesystem paths
- `numpy` — array conversion for the inference call
- `streamlit` — web UI framework
- `PIL.Image` — decodes uploads and on-disk example images
- `inference` — `CLASS_NAMES`, `LOW_CONFIDENCE_THRESHOLD`,
  `load_inference_model`, `predict_array`, `top_prediction`

**Module-Level Constants:**
- `BASE_DIR = Path(__file__).resolve().parent` — anchors every asset path to the
  app directory
- `EXAMPLE_DIR = BASE_DIR / "Test_images"` — example mammograms
- `IMAGE_DIR = BASE_DIR / "image"` — guide media
- `EXAMPLES = sorted(EXAMPLE_DIR.glob("*.jpg"))` — the 4 example image paths

**Page Configuration:**
`st.set_page_config(page_title="OncologyBreastScanner", layout="wide",
initial_sidebar_state="collapsed")` — a wide layout with no sidebar. An inline
`<style>` block (injected via `st.markdown(..., unsafe_allow_html=True)`)
defines the hero banner, `.pill` tech badge, `.section-title`, `.note` callout,
and `.footer` disclaimer styling.

**Helper Functions:**
- `_setting(name)` — resolves configuration from `os.environ` first, then
  `st.secrets`, but only after `st.secrets.load_if_toml_exists()` confirms a
  secrets file exists. The guard avoids Streamlit printing a red
  "No secrets found" error box when no `secrets.toml` is present.
- `get_model(weights_url)` — decorated with
  `@st.cache_resource(show_spinner="Loading the model...")`, so the network is
  built once and reused across Streamlit reruns.
- `_set_image(image)` / `_select_example(path)` / `_clear_all()` — mutate
  session state, bumping `uploader_key` to force the file-uploader widget to
  reset.
- `_clear_result()` — drops any previously computed scores.

**Session State:**
Initialized once with `image=None`, `scores=None`, `uploader_key=0`, and
`upload_token=None`. Session state preserves the selected image and its scores
across reruns.

**Model Loading (module level, ~line 124):**

```python
weights_url = _setting("MODEL_WEIGHTS_URL")
try:
    model = get_model(weights_url)
except Exception as exc:
    st.error("The model weights could not be loaded. ...")
    with st.expander("Technical details"):
        st.exception(exc)
    st.stop()
```

The model loads once at startup (downloading weights if necessary). On failure a
friendly error plus a technical expander renders, then the script halts.

**UI Structure:**
Three tabs created with `st.tabs(["Analyze mammogram", "How it works",
"Visual guide"])`:

**Tab 1 — "Analyze mammogram":**
Two wide columns (`st.columns([1, 1], gap="large")`):
- **Left column:** a `st.file_uploader` (PNG/JPG/JPEG/BMP/TIF/TIFF) whose key
  includes `uploader_key`, so programmatic changes force a fresh widget. A
  `file_id`/`(name, size)` token dedupes repeated decodes. The selected image
  renders via `st.image(..., use_container_width=True)`, or an info hint
  appears. An **Analyze mammogram** button (`type="primary"`, disabled until an
  image exists) runs inference:
  ```python
  with st.spinner("Applying preprocessing and running the model..."):
      scores = predict_array(model, np.array(st.session_state.image))
  ```
  A **Clear** button resets state. Below, an **Example cases** row renders each
  `Test_images/` file with a button that loads it into session state.
- **Right column:** renders the result. When scores exist it calls
  `top_prediction(scores)`:
  - low confidence → `st.warning(...)`
  - otherwise → `st.metric(label="Most likely classification", value=label,
    delta=f"{confidence:.1%} confidence", delta_color="off")`

  Every class probability then renders as a `st.progress` bar sorted high to
  low, followed by a caption noting the 30% threshold and a `.note` card
  explaining that uploads are processed in memory and never saved.

**Tab 2 — "How it works":**
A four-step Markdown pipeline (upload → preprocess → classify → review), two
guide images (`image/after cropping.png`, `image/processing.png`) rendered only
if present, the list of all 8 `CLASS_NAMES`, and the 4-level density scale.

**Tab 3 — "Visual guide":**
Side-by-side `image/benign.png` and `image/malignant.png` examples (guarded by
`.exists()`), explanatory Markdown, and the research-use footer disclaimer.

**Privacy:** Uploads are decoded in memory only; no files or prediction history
are written to disk.

---

### 3.2 `inference.py` — Shared Inference Layer

**File:** `inference.py` (56 lines)
**Purpose:** A UI-agnostic inference contract shared by the Streamlit app and
the unit tests. Keeping prediction logic here decouples it from the web
framework.

**Import:**
- `weights.WEIGHTS_PATH`, `weights.download_weights` — resolve and fetch the
  weights file

**Module-Level Constants:**
- `CLASS_NAMES` — the 8 human-readable labels in model output order (index 0–7)
- `INPUT_SHAPE = (224, 224, 3)` — required model input dimensions
- `LOW_CONFIDENCE_THRESHOLD = 0.3` — minimum top score to report a category
- `LOW_CONFIDENCE_LABEL = "Low confidence — consider re-uploading"`

#### Function: `load_inference_model(weights_path=WEIGHTS_PATH, weights_url=None)`

Builds and loads a ready-to-use model:
1. Lazily imports `build_model` from `model.py` (keeps the TensorFlow import out
   of module import time).
2. Calls `download_weights(dest=weights_path, url=weights_url)`, which returns
   the resolved path whether the file already existed or was freshly downloaded.
3. Builds the architecture with `pretrained=False` (weights come from disk, not
   ImageNet).
4. Loads the weights with `model.load_weights(path)`.
5. Returns the model.

#### Function: `predict_array(model, image)`

The single inference entry point:
1. Returns `None` for a `None` image.
2. Lazily imports `preprocess_numpy_array` and runs it.
3. Returns `None` if preprocessing fails or the shape is not `INPUT_SHAPE`.
4. Reshapes to a single-item batch `(1, 224, 224, 3)` and runs
   `model.predict(batch, verbose=0)[0]`.
5. Returns `{class_name: float(probability)}` for all 8 classes.

#### Function: `top_prediction(scores)`

Reduces a score dict to a display decision, returning a
`(label, confidence, low_confidence)` tuple:
- Empty/`None` scores → `(None, 0.0, True)`
- Highest score below `LOW_CONFIDENCE_THRESHOLD` →
  `(LOW_CONFIDENCE_LABEL, confidence, True)`
- Otherwise → `(label, confidence, False)`

The confidence gate exists to avoid presenting a confident-looking but
unreliable result on out-of-distribution or corrupted images.

---

### 3.3 `model.py` — Model Architecture

**File:** `model.py` (62 lines)
**Purpose:** Defines, builds, loads, and saves the DenseNet201-based classifier.

**Imports:**
- `pathlib.Path` — location-independent path resolution
- `tensorflow` / `tensorflow.keras.*` — model construction and loading
- `DenseNet201` from Keras applications
- `l1_l2` regularizer

**Module-Level Constants (resolved relative to the file, not the CWD):**
- `BASE_DIR = Path(__file__).resolve().parent`
- `MODEL_PATH = BASE_DIR / "model" / "model.h5"` — default path for full model serialization
- `WEIGHTS_PATH = BASE_DIR / "weights" / "modeldense1.h5"` — default path for weights-only file

#### Function: `build_model(input_shape=(224,224,3), n_classes=8, dropout_rate=0.5, pretrained=True)`

Constructs the complete model architecture from scratch. Architecture diagram:

```
Input(224, 224, 3)
    |
    v
DenseNet201 (include_top=False, weights='imagenet' if pretrained else None, trainable=True)
    |
    +-- GlobalAveragePooling2D() --+
    +-- GlobalMaxPooling2D()     --+
    |                              |
    v                              v
         Concatenate()
              |
              v
       BatchNormalization
              |
       Dense(1024, relu, L1L2 reg=0.001)
              |
       Dropout(0.5)
              |
       BatchNormalization
              |
       Dense(512, relu, L1L2 reg=0.001)
              |
       Dropout(0.5)
              |
       BatchNormalization
              |
       Dense(8, softmax)
```

**Key design decisions:**
- **Dual pooling:** GlobalAveragePooling2D and GlobalMaxPooling2D are
  concatenated together. This captures both the overall presence of features
  (max pooling) and their distribution (average pooling) — proven effective
  for medical imaging.
- **Regularization:** L1+L2 kernel regularization (0.001) on both fully
  connected layers, plus 50% Dropout after each. This is aggressive
  regularization appropriate for a relatively small medical dataset.
- **BatchNorm:** Applied before each Dense block, following the modern
  "BatchNorm → Dense → Activation" pattern for training stability.
- **Entirely trainable backbone:** The DenseNet201 is fully trainable
  (`conv_base.trainable = True`), allowing end-to-end fine-tuning. The
  training script (`train.py`) handles the two-phase approach (head-only
  then full fine-tuning) externally.
- **`pretrained` flag:** `weights="imagenet" if pretrained else None`. Training
  and generic loading use `pretrained=True` to fetch ImageNet backbone weights;
  inference calls `build_model(pretrained=False)` because the trained weights
  are then loaded from the local `.h5` file, avoiding a network fetch and the
  associated memory spike on constrained hosts.

#### Function: `load_model(path=MODEL_PATH, weights_path=WEIGHTS_PATH, pretrained=True)`

A smart model loader with fallback logic:

1. If `path` exists and has a `.h5` suffix → attempt
   `tf.keras.models.load_model(path)`
   - This loads the architecture, weights, and optimizer state in one call
   - Wrapped in try/except to handle TF version mismatches (model was
     originally trained with TF 2.4.1)
2. If step 1 fails or the file doesn't exist → call
   `build_model(pretrained=pretrained)` for a fresh architecture
3. If `weights_path` exists → load weights into the architecture
4. Compile with Adam(learning_rate=1e-5),
   CategoricalCrossentropy(label_smoothing=0.1), accuracy metric

Returns a compiled, inference-ready model.

#### Function: `save_model(model, path=MODEL_PATH)`

Serializes the full model (architecture + weights + optimizer state) to disk.
Creates parent directories if needed.

---

### 3.4 `weights.py` — Weight Download

**File:** `weights.py` (55 lines)
**Purpose:** Ensures the 256 MB pre-trained weights file is present, preferring
a direct download URL from the `MODEL_WEIGHTS_URL` environment variable and
falling back to Google Drive.

**Imports:**
- `os` — environment variable lookup
- `shutil` — streamed file copy for URL downloads
- `pathlib.Path` — path handling
- `urllib.request` — stdlib HTTP download (no extra dependency)
- `gdown` (lazy import) — Google Drive fallback

**Module constants**

```python
BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = BASE_DIR / "weights"
WEIGHTS_PATH = WEIGHTS_DIR / "modeldense1.h5"
WEIGHTS_URL_ENV = "MODEL_WEIGHTS_URL"
GDRIVE_FILE_ID = "1--7p9rRJy7WU4OmomkzM8i0veetZctTT"
```

**Function: `download_weights(dest=WEIGHTS_PATH, url=None)`**

- **Idempotent:** Returns immediately if `dest` exists and is non-empty.
- **URL first:** Uses the explicit `url` argument, otherwise
  `os.environ.get("MODEL_WEIGHTS_URL")`, and streams it with `_download_url`.
- **Google Drive fallback:** When no URL is set, lazily imports `gdown` and
  downloads `GDRIVE_FILE_ID`. If `gdown` is missing or returns `None`, raises a
  `RuntimeError` pointing the operator at `MODEL_WEIGHTS_URL`.
- **Atomic write:** Downloads into `<name>.part` and `Path.replace()`s it onto
  `dest`, so a partial transfer never leaves a corrupt weights file.
- **Cleanup:** A `finally` block removes any leftover `.part` file.
- **Google Drive file ID:** `1--7p9rRJy7WU4OmomkzM8i0veetZctTT` — permanent
  identifier for the trained weights file (publicly shared, no credentials).

**Function: `_download_url(url, dest)`** — stdlib `urllib` download with a
120-second timeout and a `User-Agent` of `oncology-breast-scanner/1.0`,
streamed to disk with `shutil.copyfileobj`.

**Deployment note:** On Streamlit Community Cloud, set the `MODEL_WEIGHTS_URL`
secret to a direct (non–Google-Drive) weights URL to avoid Drive quota errors
on cold start; otherwise the gdown fallback is used.

---

### 3.5 `preprocess.py` — Image Preprocessing

**File:** `preprocess.py` (60 lines)
**Purpose:** Image preprocessing pipeline for mammogram inference.

**Import:** `cv2` (OpenCV), `numpy`

**Module-Level Constants:**
- `SHARPEN_KERNEL` — 3x3 convolution kernel:
  ```
  [[ 0, -1,  0],
   [-1,  5, -1],
   [ 0, -1,  0]]
  ```
  This kernel enhances edges by amplifying the center pixel while
  subtracting contributions from neighbors.

#### Function: `preprocess_mammogram(image, apply_clahe, apply_sharpen, target_size)` (lines 7–29)

The core preprocessing function. Processing steps:

1. **Null guard:** Returns None if image is None
2. **Color normalization:**
   - 1-channel (grayscale) → 3-channel RGB via `cv2.COLOR_GRAY2RGB`
   - 4-channel (RGBA) → 3-channel RGB via `cv2.COLOR_RGBA2RGB`
3. **Resize:** `cv2.resize(image, target_size)` — defaults to (224, 224)
   to match DenseNet201 input dimensions
4. **CLAHE** (optional, disabled by default for inference):
   - Converts RGB → LAB color space
   - Splits L (lightness), A, B channels
   - Applies CLAHE to L channel: `clipLimit=2.0, tileGridSize=(8, 8)`
   - Merges channels back and converts LAB → RGB
   - CLAHE enhances local contrast, particularly useful for mammograms
     where subtle tissue variations are clinically significant
5. **Sharpening** (enabled by default):
   - `cv2.filter2D(image, -1, SHARPEN_KERNEL)` — convolves the sharpen
     kernel over the image to enhance edges and fine details

#### Function: `normalize_for_model(image)` (lines 32–34)

Divides all pixel values by 255.0, normalizing to the [0, 1] range
expected by Keras models trained on ImageNet-normalized inputs.

#### Function: `load_and_preprocess(path, ...)` (lines 37–46)

End-to-end function for loading an image from disk:
1. `cv2.imread(path)` — loads in BGR format
2. `cv2.COLOR_BGR2RGB` — converts to RGB
3. `preprocess_mammogram()` — applies CLAHE, sharpening, resize
4. `normalize_for_model()` — scales to [0, 1]

Used by `evaluate.py` for batch processing (training uses
`preprocess_mammogram()` / `normalize_for_model()` directly).

#### Function: `preprocess_numpy_array(img_array, ...)` (lines 49–60)

The function called by `inference.py` for in-memory images (PIL images
converted to numpy arrays by `streamlit_app.py`):

1. **Null guard:** Returns None for empty/None arrays
2. **Dtype normalization:**
   - If the array is float in [0, 1] range, upscales to [0, 255] uint8
   - If the array is a non-uint8 type, casts to uint8
3. Runs `preprocess_mammogram()` then `normalize_for_model()`
4. Returns a (224, 224, 3) float32 array in [0, 1]

---

## 4. Training & Evaluation Pipeline

### 4.1 `train.py` — Training Pipeline

**File:** `train.py` (292 lines)
**Purpose:** Complete two-phase training pipeline for fine-tuning DenseNet201
on the CBIS-DDSM breast cancer dataset.

**Usage:**
```bash
python train.py --train-csv data/train_set.csv --image-dir data/images
python train.py --train-csv data/train_set.csv --image-dir data/images \
    --val-csv data/test_set.csv --epochs 50 --batch-size 32
```

**Imports:**
- `numpy`, `tensorflow`, `pandas` — core ML/data stack
- `sklearn.model_selection.train_test_split` — train/val split
- `sklearn.utils.class_weight.compute_class_weight` — handles class imbalance
- `model.build_model` / `model.save_model` — model construction and persistence
- `preprocess.preprocess_mammogram` / `preprocess.normalize_for_model` — per-image processing

**Module-Level Constants:**
- `CLASS_NAMES` — 8 training labels using a slightly different naming scheme
  than `inference.py`: `'Density1Benign'`, `'Density1Malignant'`, etc.
- `CLASS_TO_INDEX` — dict mapping label name → integer index

#### Class: `MammogramGenerator(tf.keras.utils.Sequence)` (lines 37–105)

A custom Keras Sequence data generator for efficient batch loading. Extends
`tf.keras.utils.Sequence` for multiprocessing-safe data loading.

**Constructor parameters:**
- `dataframe` — pandas DataFrame with `filename` and `label` columns
- `image_dir` — directory path containing the image files
- `batch_size` — samples per batch (default 32)
- `target_size` — output image dimensions (default 224x224)
- `augment` — enable/disable data augmentation (default True)
- `shuffle` — shuffle after each epoch (default True)
- `apply_clahe` — enable CLAHE contrast enhancement (default True for training)
- `apply_sharpen` — enable sharpening (default True)

**Key methods:**

`__len__()` — returns number of batches per epoch.

`__getitem__(idx)` — loads and processes a batch:
- Extracts batch rows from the DataFrame
- For each image: loads from disk, preprocesses, applies augmentation,
  converts label to one-hot encoding
- Returns (X_batch, y_batch) numpy arrays

`_load_image(path)` (lines 69–83):
- `cv2.imread()` → BGR→RGB → `preprocess_mammogram()` →
  `normalize_for_model()` → augmentation

`_augment(img)` (lines 85–101) — applies randomized augmentations:
- **Horizontal flip** (50% probability)
- **Rotation** (±10 degrees)
- **Scaling** (0.9x to 1.1x, 30% probability)
- **Brightness jitter** (0.85x to 1.15x, 30% probability)

`on_epoch_end()` — shuffles the index array for the next epoch.

#### Training Workflow (`main()`, lines 108–292)

**CLI Arguments (via argparse):**

| Argument | Default | Description |
|----------|---------|-------------|
| `--train-csv` | (required) | Training CSV with filename,label columns |
| `--val-csv` | (optional) | Validation CSV (split from train if not given) |
| `--image-dir` | (required) | Directory containing images |
| `--epochs` | 60 | Total training epochs |
| `--batch-size` | 32 | Batch size |
| `--lr` | 1e-4 | Initial learning rate |
| `--dropout` | 0.5 | Dropout rate |
| `--fine-tune-layers` | 50 | Number of DenseNet layers to unfreeze |
| `--val-split` | 0.15 | Validation split ratio |
| `--head-only-epochs` | 20 | Epochs training only the classification head |
| `--seed` | 42 | Random seed for reproducibility |
| `--output-dir` | `training_output` | Output for checkpoints and logs |
| `--resume` | (none) | Path to checkpoint weights |

**Execution flow:**

1. **Seed setting & data loading** — reads CSV(s), optionally splits into
   train/val, computes class weights with `sklearn.compute_class_weight`
   to handle class imbalance.

2. **Phase 1 — Head-only training** (lines 172–210):
   - Freezes the DenseNet201 backbone, trains only the new classification head
   - Callbacks: ModelCheckpoint (best val_accuracy), ReduceLROnPlateau
     (factor=0.5, patience=3), EarlyStopping (patience=8)
   - Saves best weights to `phase1_best.weights.h5`

3. **Phase 2 — Fine-tuning** (lines 212–253):
   - Builds a new model, loads phase 1 best weights
   - Unfreezes the last N DenseNet layers (`--fine-tune-layers`, default 50)
   - Lower learning rate: `lr / 10` (default: 1e-5)
   - Callbacks: two ModelCheckpoints (best + per-epoch snapshots),
     ReduceLROnPlateau (factor=0.5, patience=4), EarlyStopping (patience=10),
     CSVLogger
   - Saves final model to `model/model.h5` and `weights/modeldense1.h5`

4. **Post-training** (lines 260–288):
   - Reports best validation accuracy and epoch
   - Generates training curves plot (loss + accuracy) via matplotlib
   - Prints evaluation command for the user

---

### 4.2 `evaluate.py` — Model Evaluation

**File:** `evaluate.py` (539 lines)
**Purpose:** Comprehensive evaluation suite for the trained classifier.

**Usage:**
```bash
python evaluate.py --csv data/test_set.csv --image-dir data/images
python evaluate.py --csv data/test_set.csv --image-dir data/images --no-plots
```

**Imports:**
- `numpy`, `tensorflow` — inference
- `pathlib.Path`, `csv` — data loading
- `sklearn.metrics` — accuracy, precision, recall, F1, confusion matrix,
  ROC-AUC, top-k accuracy, average precision
- `matplotlib`, `seaborn` — visualization
- `model.load_model`, `preprocess.load_and_preprocess` — model and preprocessing

#### Data Loading

**`load_from_csv(csv_path, image_dir)` (lines 41–57):**
Reads a CSV with `filename` and `label` columns, loads each image via
`load_and_preprocess()`, returns numpy arrays of images, integer labels,
and filenames.

**`load_from_directory(image_dir)` (lines 60–76):**
Scans a directory for images, parses labels from filenames using regex
(pattern: `Density{N}{Benign|Malignant}`), loads and preprocesses each.

**`parse_label_from_filename(filename)` (lines 32–38):**
Uses regex to extract the label from a filename, e.g.,
`Density3Malignant_00001.png` → `"Density3Malignant"`.

#### Metrics Functions

**`compute_metrics(y_true, y_pred, y_prob)` (lines 79–142):**
Returns a comprehensive dict:
- `accuracy` — overall top-1 accuracy
- `top2_accuracy` — accuracy when correct answer is in top 2 predictions
- `precision_per_class` / `recall_per_class` / `f1_per_class` — per-class arrays
- `precision_macro` / `recall_macro` / `f1_macro` / `f1_weighted` — aggregated scores
- `confusion_matrix` — 8x8 numpy array
- `auc_ovr` — ROC-AUC one-vs-rest
- `auc_per_class` / `ap_per_class` — per-class AUC and average precision
  (None for classes with no positive samples)

**`compute_calibration(y_true, y_prob, n_bins=10)` (lines 145–176):**
Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
per class. Bins predictions into 10 confidence intervals, compares average
confidence to observed accuracy in each bin.

**`bootstrap_metrics(y_true, y_pred, y_prob, n_bootstrap=1000)` (lines 179–195):**
Computes 95% confidence intervals for accuracy and macro F1 via 1000
bootstrap resamples.

#### Visualization Functions

Each accepts a `save_path` argument for file output:

| Function | Output |
|----------|--------|
| `plot_confusion_matrix()` | Seaborn heatmap with per-class counts |
| `plot_roc_curves()` | 8 ROC curves (one per class) vs random baseline |
| `plot_pr_curves()` | 8 Precision-Recall curves with AP scores |
| `plot_reliability_diagram()` | 2x4 grid of calibration gap bar charts |
| `plot_prediction_distribution()` | Side-by-side bar chart of true vs predicted counts |

#### Report Output

**`print_report()` (lines 198–272):**
Prints a formatted ASCII report:
```
==========================================================
CLASSIFICATION REPORT
==========================================================
Overall Accuracy:          0.XXXX  (XX.XX%)  (95% CI: ...)
Top-2 Accuracy:            ...
Macro Precision:           ...
...
CONFUSION MATRIX
              B D1      M D1      B D2      ...
       B D1      XX        ..        ..
       M D1      ..        XX        ..
...
MISCLASSIFIED SAMPLES (N total)
  filename.jpg    true=Density3Benign    pred=Density3Malignant    conf=0.XXX
```

#### Main Evaluation Flow (`main()`, lines 417–539)

1. Loads model via `load_model()`, compiles, loads weights
2. Loads test data from CSV or directory
3. Runs `model.predict()` on all test images
4. Computes metrics, calibration, bootstrap CIs
5. Prints ASCII report
6. Generates plots (confusion matrix, ROC, PR, distribution, reliability)
7. Exports detailed per-sample predictions to `predictions.csv`
8. Handles unlabeled samples separately → `unlabeled_predictions.csv`

---

### 4.3 `prepare_dataset.py` — Dataset Preparation

**File:** `prepare_dataset.py` (187 lines)
**Purpose:** Converts raw CBIS-DDSM Kaggle dataset into a standardized format
for training and evaluation.

**Usage:**
```bash
python prepare_dataset.py --kaggle-dir cbis_ddsm --output-dir data
```

**Imports:**
- `pandas`, `numpy`, `pathlib.Path` — data manipulation
- `cv2` — image reading and resizing
- `argparse` — CLI

**Module-Level Constants:**
- `CLASS_LABELS` — 8 valid label strings in the training format:
  `'Density1Benign'`, `'Density1Malignant'`, ..., `'Density4Malignant'`

#### Label Parsing Functions

**`pathology_to_binary(p)` (lines 24–29):**
Converts raw pathology string (e.g., `"MALIGNANT"`, `"BENIGN_WITHOUT_CALLBACK"`)
→ `"Benign"` or `"Malignant"`.

**`density_to_int(d)` (lines 32–37):**
Converts raw density value to integer 1-4. Returns None for invalid values.

**`get_density_col(row)` (lines 40–45):**
Looks for density in multiple possible CSV column names:
`'breast density'`, `'breast_density'`, `'density'`.

**`make_label(density, pathology)` (lines 48–54):**
Combines density and pathology into a validated label string.
e.g., `density=3, pathology="MALIGNANT"` → `"Density3Malignant"`.
Validates against `CLASS_LABELS`.

#### DICOM-to-JPEG Mapping

**`extract_uid(dicom_path)` (lines 57–61):**
Extracts the patient/study UID from a CBIS-DDSM DICOM file path.
The UID is the second-to-last directory component.

**`find_jpeg_for_dicom(dicom_path, jpeg_dir)` (lines 65–77):**
CBIS-DDSM stores DICOM paths in CSVs but provides JPEG conversions
in a separate `jpeg/` directory. This function:
1. Extracts the UID from the DICOM path
2. Looks for a matching subdirectory under `jpeg/{UID}/`
3. Returns the first JPEG file found in that subdirectory

#### Main Flow (`main()`, lines 80–187)

1. Reads 4 CBIS-DDSM CSV files:
   - `mass_case_description_train_set.csv`
   - `mass_case_description_test_set.csv`
   - `calc_case_description_train_set.csv`
   - `calc_case_description_test_set.csv`
2. For each row, extracts label from density + pathology columns
3. Resolves DICOM paths to JPEG files using `find_jpeg_for_dicom()`
4. Collects valid `(source_path, label, split)` records
5. Copies and resizes images to `data/images/` as 224x224 PNGs
6. Writes `data/train_set.csv` and `data/test_set.csv` with
   `filename,label` columns

---

## 5. Notebooks

### 5.1 `colab_training.ipynb`

**Purpose:** Google Colab notebook for training the model on a GPU runtime.
Mounts Google Drive, installs dependencies, downloads CBIS-DDSM from Kaggle,
runs `prepare_dataset.py` and `train.py` in sequence. Designed for users
without local GPU access.

### 5.2 `colab_training_v2.ipynb`

**Purpose:** Updated version of the Colab training notebook with improvements
to the training workflow, potentially including different hyperparameters or
data handling.

### 5.3 `Demo_Notebook.ipynb`

**Purpose:** Original exploration/demo notebook (1046 lines). Contains early
experiments with:
- Data loading and visualization
- Image preprocessing with CLAHE and sharpening
- Model architecture experimentation
- `keras_preprocessing`, `albumentations` for augmentation
- Training loop setup

This was the development sandbox before the code was modularized into
separate `.py` files.

### 5.4 `test.ipynb`

**Purpose:** Small validation notebook (70 lines). Tests whether the
`weights/modeldense1.h5` file is a valid HDF5 file by attempting to
open it with `h5py`. Was likely used to debug weight file corruption
issues.

---

## 6. Data Files

### 6.1 `data/` Directory (gitignored)

| File | Description |
|------|-------------|
| `train_set.csv` | Training split — columns: `filename`, `label`. ~1192 images |
| `test_set.csv` | Test split — columns: `filename`, `label`. ~703 images |
| `images/` | Preprocessed 224x224 PNG images named `{label}_{idx:05d}.png` |

Generated by `prepare_dataset.py`. Not needed for inference — only for
retraining.

### 6.2 Root CSV Files (gitignored)

| File | Lines | Description |
|------|-------|-------------|
| `csv_mass_train.csv` | 2637 | Original CBIS-DDSM mass training metadata (patient_id, breast_density, pathology, DICOM paths, etc.) |
| `csv_mass_test.csv` | 757 | Original CBIS-DDSM mass test metadata |
| `csv_calc_train.csv` | 3093 | Original CBIS-DDSM calcification training metadata |
| `csv_calc_test.csv` | 653 | Original CBIS-DDSM calcification test metadata |

These are the raw Kaggle-format CSVs extracted from the CBIS-DDSM zip.
They are inputs to `prepare_dataset.py` and are gitignored due to size (~2 MB total).

---

## 7. Media Directories

### 7.1 `image/` Directory

Contains 8 PNG files used in the README and the Streamlit UI. Note that
`Begign.png` is a typo variant of `benign.png` and `Maliggnant.png` is a typo
variant of `malignant.png` — both pairs appear to be the same images. The app
loads only `benign.png` and `malignant.png` (via `IMAGE_DIR`); the remaining
files are legacy assets retained in the repository.

| File | Size | Used In | Description |
|------|------|---------|-------------|
| `benign.png` | 133 KB | UI Tab 3 + README | Benign mammogram example |
| `malignant.png` | 117 KB | UI Tab 3 + README | Malignant mammogram example |
| `Begign.png` | 133 KB | (legacy duplicate) | Typo variant of benign.png |
| `Maliggnant.png` | 117 KB | (legacy duplicate) | Typo variant of malignant.png |
| `processing.png` | 115 KB | (legacy) | Preprocessing pipeline visualization |
| `after cropping.png` | 160 KB | (legacy) | Post-cropping/enhancement result |
| `prediction.png` | 159 KB | (legacy) | Sample prediction output screenshot |
| `webpage.png` | 44 KB | (unused) | Historical Gradio interface screenshot |

### 7.2 `Test_images/` Directory

Contains 4 representative mammogram JPEGs for quick testing from the UI:

| File | Description |
|------|-------------|
| `Density1Malignant.jpg` | Malignant, fatty breast (density 1) |
| `Density2Malignant.jpg` | Malignant, scattered density (density 2) |
| `Density3Benign.jpg` | Benign, heterogeneously dense (density 3) |
| `Density4Benign.jpg` | Benign, extremely dense (density 4) |

These 4 files cover all 4 density levels and both pathology classes,
providing balanced example coverage in the UI example gallery.

---

## 8. Configuration Files

### 8.1 `requirements.txt`

Pinned Python package dependencies (Streamlit stack):

| Package | Version | Purpose |
|---------|---------|---------|
| `streamlit` | `1.40.1` | Web UI framework |
| `tensorflow-cpu` | `2.15.0` | CPU-only deep learning framework |
| `opencv-python-headless` | `4.9.0.80` | OpenCV without GUI (server-safe) |
| `pillow` | `10.4.0` | Image handling (PIL fork) |
| `numpy` | `1.26.4` | Numerical computing |
| `gdown` | `5.0.0` | Google Drive fallback download for weights |

**Note on TensorFlow:** Switched from the full `tensorflow` GPU wheel to
`tensorflow-cpu`, since deployment targets (Streamlit Community Cloud) have no
GPU. Version 2.15.0 is the last release supporting Python 3.11 with Keras 2,
matching the `.h5` weight format used here.

**Note on pins:** All versions are pinned exactly so local, Docker, and cloud
builds resolve identically.

**System packages:** `packages.txt` installs `libgomp1` (required by
TensorFlow) and `libglib2.0-0` (required by OpenCV at import time).

### 8.2 `.gitignore`

Excludes from version control:
- **Model artifacts:** `model/`, `weights/` (too large for Git, downloaded at runtime)
- **Training data:** `data/`, `csv_*_train.csv`, `csv_*_test.csv`
- **Generated outputs:** `evaluation_output/`, `training_output/`, `flagged/`
- **Runtime files:** `.gradio/` (legacy), `.venv/`, `__pycache__/`, `*.pyc`,
  `.ipynb_checkpoints/`
- **macOS:** `.DS_Store`

---

## 9. Model & Weights

### `model/model.h5` (gitignored, ~87 MB)

Full Keras serialized model containing:
- Model architecture (layer graph)
- Trained weights for all layers
- Optimizer state (Adam with current learning rate, momentum accumulators)
- Training configuration (loss, metrics)

Saved by `train.py` via `model.save()` and `model.py`'s `save_model()`.

Can be loaded directly with `tf.keras.models.load_model()`, but may fail
across major TF version differences.

### `weights/modeldense1.h5` (gitignored, ~256 MB)

Weights-only file in HDF5 format. Contains only the layer weight matrices —
no architecture, no optimizer state.

- Downloaded at runtime by `weights.download_weights()`: a direct URL from
  the `MODEL_WEIGHTS_URL` environment variable is preferred, with a `gdown`
  Google Drive fallback (file ID: `1--7p9rRJy7WU4OmomkzM8i0veetZctTT`)
- More portable than the full model — compatible across TF versions
  when loaded into a freshly built architecture
- Saved by `train.py` via `model.save_weights()`

---

## 10. Generated & Runtime Directories

| Directory | .gitignore? | Purpose |
|-----------|-------------|---------|
| `evaluation_output/` | Yes | Plots and CSVs from `evaluate.py` (confusion matrix, ROC, PR curves, predictions CSV) |
| `training_output/` | Yes | Training checkpoints, logs, and curves from `train.py` (`.weights.h5` files, `training_log.csv`, `training_curves.png`) |
| `weights/` | Yes | Runtime-downloaded model weights (`modeldense1.h5`); not committed due to size |
| `.venv/` | Yes | Local virtual environment |
| `__pycache__/` | Yes | Python bytecode cache |

> Legacy Gradio runtime directories (`.gradio/`, `flagged/`) are still listed
> in `.gitignore` but are no longer produced now that the UI is Streamlit.

---

## 11. Data Flow: End-to-End Walkthrough

### Inference Flow (User visits the web app)

```
User uploads a mammogram (st.file_uploader) or clicks an example
       │
       v
PIL Image opened → st.session_state["image"]
       │
       v
"Analyze" button → np.array(image)  (RGB, uint8)
       │
       v
preprocess_numpy_array(img) in preprocess.py
   → detect float [0,1] → upscale to uint8 [0,255]
   → grayscale/A → RGB conversion
   → cv2.resize(224, 224)
   → sharpen kernel convolution
   → divide by 255.0 → [0,1]
       │
       v
predict_array(model, img) in inference.py
   → model.predict(processed.reshape(1, 224, 224, 3), verbose=0)[0]
   → 8-class softmax probabilities → {class_name: probability}
       │
       v
top_prediction(scores)
   → (label, confidence, low_confidence)
   → if max < LOW_CONFIDENCE_THRESHOLD (0.3): low_confidence = True
       │
       v
streamlit_app.py renders:
   → st.metric (label + confidence %)  |  low-confidence st.warning
   → per-class probability bars (st.progress, sorted high → low)
```

### Training Flow (Retraining the model)

```
Raw CBIS-DDSM Kaggle dataset
  (jpeg/ directory + 4 CSV metadata files)
       │
       v
prepare_dataset.py
  → Parse CSVs, map DICOM paths to JPEGs
  → Resize all to 224x224 PNGs
  → Output: data/train_set.csv + data/test_set.csv + data/images/
       │
       v
train.py
  → Load CSVs, build train/val generators
  → Phase 1: freeze backbone, train head only (20 epochs)
  → Phase 2: unfreeze top 50 DenseNet layers, lower LR (40 epochs)
  → Class weighting for imbalance
  → Data augmentation: flips, rotation, scaling, brightness
  → Save: model/model.h5 + weights/modeldense1.h5
       │
       v
evaluate.py
  → Load model, run predictions on test set
  → Metrics: accuracy, precision, recall, F1, AUC, ECE
  → Bootstrap 95% confidence intervals
  → Plots: confusion matrix, ROC, PR, reliability, distribution
  → Export: predictions.csv
```

---

## 12. Classification System

### 8-Class Output

The model outputs 8 probability values summing to ~1.0 (softmax). Each index
represents the joint probability of a specific density-pathology combination:

| Index | Density | Pathology | Label |
|-------|---------|-----------|-------|
| 0 | 1 (Fatty) | Benign | Benign with Density=1 |
| 1 | 1 (Fatty) | Malignant | Malignant with Density=1 |
| 2 | 2 (Scattered) | Benign | Benign with Density=2 |
| 3 | 2 (Scattered) | Malignant | Malignant with Density=2 |
| 4 | 3 (Heterogeneously dense) | Benign | Benign with Density=3 |
| 5 | 3 (Heterogeneously dense) | Malignant | Malignant with Density=3 |
| 6 | 4 (Extremely dense) | Benign | Benign with Density=4 |
| 7 | 4 (Extremely dense) | Malignant | Malignant with Density=4 |

### BI-RADS Density Scale

| Level | Category | Description |
|-------|----------|-------------|
| 1 | Almost entirely fatty | Breast tissue is mostly fat, making abnormalities easier to detect |
| 2 | Scattered fibroglandular densities | Some areas of dense tissue scattered throughout |
| 3 | Heterogeneously dense | Many areas of dense tissue, may obscure small masses |
| 4 | Extremely dense | Very dense tissue, significantly reduces mammogram sensitivity |

Higher density levels correlate with both increased breast cancer risk and
reduced mammogram sensitivity (the "masking effect").

### Pathology Classes

| Class | Description |
|-------|-------------|
| **Benign** | Non-cancerous. Typically well-circumscribed with smooth borders. Includes fibroadenomas, cysts, and other benign masses. |
| **Malignant** | Cancerous. Often presents with irregular, spiculated borders and architectural distortion. Requires biopsy confirmation. |

### Confidence Threshold

The prediction function includes a confidence gate at **0.3**. If no class
exceeds this threshold, the model returns "Low confidence — consider
re-uploading" instead of a potentially misleading result. This prevents
the model from making confident-but-wrong predictions on out-of-distribution
or corrupted images.
