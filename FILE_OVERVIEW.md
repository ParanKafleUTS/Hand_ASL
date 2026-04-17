# File Overview — Hand ASL Repository

This document describes the purpose of every file in the repository.
The numbered scripts (`01_` → `08_`) form an end-to-end pipeline;
the remaining utility scripts can be run independently at any stage.

---

## Pipeline Scripts (run in order)

### `01_extract_and_balance.py`
**Dataset extraction and path resolution.**

- Extracts `archive (2).zip` into `ASL_Alphabet_Dataset/` (skips extraction if the folder already exists).
- Handles the double-h typo in the zip's folder names (`asl_alphabhet_train` / `asl_alphabhet_test`) as well as the correctly-spelled variants, trying each in priority order.
- Lists all class subfolders (A–Z, del, nothing, space) and prints the image count per class.
- Writes the resolved training folder path to `train_folder.txt` and the test folder path to `test_folder.txt` so every downstream script can locate the data without hard-coding paths.

**Outputs:** `train_folder.txt`, `test_folder.txt`

---

### `02_data_validation.py`
**Image integrity check and train / val / test split.**

- Reads the training folder path from `train_folder.txt`.
- Collects all image paths and their class labels using `imutils.paths`.
- Opens every image with OpenCV to detect corrupted files; removes any that cannot be read.
- Splits the clean dataset into **70 % train / 15 % validation / 15 % test** using stratified sampling so class proportions are preserved across all three sets.
- Saves the three path-and-label lists to `splits.pkl` for use by all model-training scripts.

**Outputs:** `splits.pkl`

---

### `03_landmark_mlp.py`
**MediaPipe landmark extraction + MLP classifier.**

- Loads `splits.pkl` to get all image paths.
- Runs **MediaPipe Hands** on every image to extract 21 hand keypoints (63 `x, y, z` coordinates per image).
- Images where no hand is detected are discarded.
- Trains a small three-layer **Multi-Layer Perceptron** (128 → 64 → softmax) on the 63-dimensional landmark vectors.
- Uses `EarlyStopping` and `ReduceLROnPlateau` callbacks.
- Saves the trained model and the `LabelEncoder` for inference.

> **Note:** Running MediaPipe on the full ~87 k-image dataset is slow on CPU. Prefer `04_raw_cnn.py` when speed is important.

**Outputs:** `landmark_mlp.h5`, `landmark_label_encoder.pkl`

---

### `04_raw_cnn.py`
**Transfer-learning CNNs on full (uncropped) images.**

- Reads `splits.pkl` and builds Keras `ImageDataGenerator` pipelines.
- **Training augmentation:** rotation ±15°, width/height shift ±10 %, brightness [0.8–1.2], zoom ±10 %.
- Trains two transfer-learning models back-to-back on the raw 224 × 224 images:
  - **MobileNetV2** (ImageNet weights, frozen base) → saved as `mobilenetv2_raw.h5`
  - **EfficientNetB0** (ImageNet weights, frozen base) → saved as `efficientnetb0_raw.h5`
- Both models use `EarlyStopping` (patience 4) and `ReduceLROnPlateau` (patience 2).

**Outputs:** `mobilenetv2_raw.h5`, `efficientnetb0_raw.h5`

---

### `05_crop_images.py`
**Hand-region cropping with MediaPipe.**

- Reads all image paths from `splits.pkl`.
- For each image, runs **MediaPipe Hands** and computes a bounding box around the detected landmarks (with a 20-pixel margin).
- Crops the hand region (or falls back to the full image if no hand is detected), then resizes to 224 × 224.
- Saves cropped images to `processed_images/<class>/`, mirroring the original folder structure.
- Re-splits the cropped images into train / val / test and saves the new split to `processed_splits.pkl`.

> **Note:** This step is CPU-intensive on the full dataset.

**Outputs:** `processed_images/` directory, `processed_splits.pkl`

---

### `06_cropped_cnn.py`
**Transfer-learning CNNs on cropped images.**

- Identical architecture and augmentation strategy to `04_raw_cnn.py`, but reads `processed_splits.pkl` (hand-cropped images from `05_crop_images.py`).
- Trains **MobileNetV2** and **EfficientNetB0** on the tighter hand-region crops.
- Saves both models.

**Outputs:** `mobilenetv2_crop.h5`, `efficientnetb0_crop.h5`

---

### `07_compare_models.py`
**Model comparison bar chart.**

- Takes manually entered test-accuracy values for all five trained models (Landmark MLP, MobileNetV2 raw, EfficientNetB0 raw, MobileNetV2 cropped, EfficientNetB0 cropped).
- Draws a colour-coded bar chart and identifies the best-performing model.
- Saves the chart as a PNG file.

> After running the training scripts, update the `accuracies` dictionary in this file with the printed test-accuracy values before running it.

**Outputs:** `model_comparison.png`

---

### `08_save_class_names.py`
**Persist ordered class names for inference.**

- Rebuilds the training `ImageDataGenerator` from `splits.pkl` to obtain the canonical `class_indices` mapping (alphabetically sorted).
- Inverts the mapping (index → class name) and writes each class name on its own line in `class_names.txt`.
- Used during inference to convert model output indices back to letter labels.

**Outputs:** `class_names.txt`

---

## Utility Scripts (independent / exploratory)

### `annotate_and_count.py`
**Batch landmark annotation with progress summary (CLI).**

- Accepts `--input` and `--output` command-line arguments; defaults to the path in `train_folder.txt`.
- Iterates over all class subfolders, runs **MediaPipe Hands** on each image, and draws the 21-point skeleton + connections onto the original image.
- Only images where a hand is detected are saved to the output directory (mirrors the class subfolder structure).
- Prints a final summary: total processed, successfully annotated, failed, and success rate.

**Outputs:** `annotated_images/` directory (or custom `--output` path)

---

### `annotate_all.py`
**Batch landmark annotation (importable module).**

- Functionally similar to `annotate_and_count.py` but designed to be imported as well as run directly (`annotate_images()` function is importable).
- Reads the source directory from `train_folder.txt` (falls back to `subset_5000`).
- Saves annotated images to `annotated_images/` and prints detection statistics and common failure reasons.

**Outputs:** `annotated_images/` directory

---

### `generate_skeletons_and_train.py`
**Skeleton-image generation + full CNN training pipeline.**

- All-in-one script combining preprocessing and model training.
- **Step 1 – Skeleton generation:** for each image, draws the MediaPipe hand skeleton (joints + bones) on a **white background** (not the original photo) and saves it to `skeleton_images/`.
- **Step 2 – Split preparation:** builds stratified train / val / test splits from the skeleton images and saves them to `skeleton_splits.pkl`.
- **Step 3 – Training:** trains **MobileNetV2** and **EfficientNetB0** on the skeleton images using the same augmentation / callback strategy as `04_raw_cnn.py`.
- **Step 4 – Comparison:** selects the best model, saves it as `best_skeleton_model.h5`, writes `skeleton_class_names.txt`, and produces `skeleton_model_comparison.png`.

**Outputs:** `skeleton_images/`, `skeleton_splits.pkl`, `mobilenetv2_skeleton.h5`, `efficientnetb0_skeleton.h5`, `best_skeleton_model.h5`, `skeleton_class_names.txt`, `skeleton_model_comparison.png`

---

### `show_landmarks.py`
**Single-image landmark visualiser (CLI).**

- Accepts `--image` (specific file), `--source_dir` (folder to pick from randomly), and `--output` (save instead of display) arguments.
- If no image is given, picks a random image from any class subfolder of the training directory.
- Runs **MediaPipe Hands**, draws the detected landmarks and connections on the image, then either displays it in a window or saves it to disk.
- Useful for quickly verifying that MediaPipe detects hands correctly on your dataset.

---

## Generated Artefacts (not tracked in git)

| File / Folder | Created by |
|---|---|
| `ASL_Alphabet_Dataset/` | `01_extract_and_balance.py` |
| `train_folder.txt` | `01_extract_and_balance.py` |
| `test_folder.txt` | `01_extract_and_balance.py` |
| `splits.pkl` | `02_data_validation.py` |
| `landmark_mlp.h5` | `03_landmark_mlp.py` |
| `landmark_label_encoder.pkl` | `03_landmark_mlp.py` |
| `mobilenetv2_raw.h5` | `04_raw_cnn.py` |
| `efficientnetb0_raw.h5` | `04_raw_cnn.py` |
| `processed_images/` | `05_crop_images.py` |
| `processed_splits.pkl` | `05_crop_images.py` |
| `mobilenetv2_crop.h5` | `06_cropped_cnn.py` |
| `efficientnetb0_crop.h5` | `06_cropped_cnn.py` |
| `model_comparison.png` | `07_compare_models.py` |
| `class_names.txt` | `08_save_class_names.py` |
| `annotated_images/` | `annotate_all.py` / `annotate_and_count.py` |
| `skeleton_images/` | `generate_skeletons_and_train.py` |
| `skeleton_splits.pkl` | `generate_skeletons_and_train.py` |
| `best_skeleton_model.h5` | `generate_skeletons_and_train.py` |
| `skeleton_class_names.txt` | `generate_skeletons_and_train.py` |
| `skeleton_model_comparison.png` | `generate_skeletons_and_train.py` |
