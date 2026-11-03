"""
Real-Time Facial Emotion Recognition
Training Script: ANN and CNN Models
Dataset: Image_data (7 emotion classes)
"""

# ─────────────────────────────────────────────
# 0.  REPRODUCIBILITY & IMPORTS
# ─────────────────────────────────────────────
import os, sys, time, hashlib, pickle, warnings, random
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from PIL import Image
from tqdm import tqdm

import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, precision_score,
                              recall_score, f1_score)
from sklearn.utils.class_weight import compute_class_weight

# Seed everything
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ─────────────────────────────────────────────
# 1.  PATHS
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "Image_data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUT_DIR    = os.path.join(BASE_DIR, "outputs")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUT_DIR,    exist_ok=True)

EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
CLASS_TO_IDX    = {c: i for i, c in enumerate(EMOTION_CLASSES)}
IDX_TO_CLASS    = {i: c for c, i in CLASS_TO_IDX.items()}
NUM_CLASSES     = len(EMOTION_CLASSES)
IMG_SIZE        = (64, 64)          # height x width
CHANNELS        = 1                 # grayscale
INPUT_SHAPE_ANN = IMG_SIZE[0] * IMG_SIZE[1] * CHANNELS
INPUT_SHAPE_CNN = (*IMG_SIZE, CHANNELS)

print("="*60)
print(" Real-Time Facial Emotion Recognition - Training")
print("="*60)
print(f"TensorFlow  : {tf.__version__}")
print(f"Data dir    : {DATA_DIR}")
print(f"Image size  : {IMG_SIZE}")
print(f"Classes     : {EMOTION_CLASSES}\n")

# ─────────────────────────────────────────────
# 2.  DATASET INSPECTION & CLEANING
# ─────────────────────────────────────────────
print("[STEP 2] Inspecting and cleaning dataset ...")
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

records = []          # list of (path, label_idx)
hashes  = {}          # md5 -> path  (for exact dup detection)
skipped = 0

for cls in EMOTION_CLASSES:
    cls_dir = os.path.join(DATA_DIR, cls)
    if not os.path.isdir(cls_dir):
        print(f"  WARNING: folder missing for class '{cls}'")
        continue
    for fname in os.listdir(cls_dir):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in VALID_EXTS:
            skipped += 1
            continue
        fpath = os.path.join(cls_dir, fname)
        # Validate image readability
        try:
            with Image.open(fpath) as img:
                img.verify()          # lightweight corruption check
        except Exception:
            skipped += 1
            continue
        # Exact duplicate detection via MD5
        with open(fpath, "rb") as f:
            md5 = hashlib.md5(f.read()).hexdigest()
        if md5 in hashes:
            skipped += 1
            continue
        hashes[md5] = fpath
        records.append((fpath, CLASS_TO_IDX[cls]))

print(f"  Total valid images : {len(records)}")
print(f"  Skipped (corrupt/dup/ext) : {skipped}")

class_counts = pd.Series([r[1] for r in records]).map(IDX_TO_CLASS).value_counts()
print("\n  Class distribution:")
for cls, cnt in class_counts.items():
    bar = "X" * (cnt // 500)
    print(f"    {cls:<10}: {cnt:>5}  {bar}")

# Class distribution bar chart
fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#e74c3c","#8e44ad","#3498db","#2ecc71","#95a5a6","#e67e22","#1abc9c"]
ax.bar(class_counts.index, class_counts.values, color=colors)
ax.set_title("Dataset Class Distribution", fontsize=16, fontweight="bold")
ax.set_xlabel("Emotion Class")
ax.set_ylabel("Number of Images")
for i, v in enumerate(class_counts.values):
    ax.text(i, v + 50, str(v), ha="center", fontsize=9)
ax.set_facecolor("#f8f9fa"); fig.patch.set_facecolor("#f8f9fa")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "class_distribution.png"), dpi=150)
plt.close()
print("\n  Saved: outputs/class_distribution.png")

# ─────────────────────────────────────────────
# 3.  LOAD & PREPROCESS IMAGES
# ─────────────────────────────────────────────
print("\n[STEP 3] Loading and preprocessing images ...")

def load_image(path, size=IMG_SIZE, channels=CHANNELS):
    """Load image -> resized grayscale array in [0,1]."""
    flag = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR
    img  = cv2.imread(path, flag)
    if img is None:
        return None
    img = cv2.resize(img, size)
    img = img.astype(np.float32) / 255.0
    if channels == 1:
        img = img[..., np.newaxis]   # (H,W,1)
    return img

X_paths = [r[0] for r in records]
y_all   = np.array([r[1] for r in records])

X_all = []
valid_idx = []
for i, path in enumerate(tqdm(X_paths, desc="  Loading")):
    img = load_image(path)
    if img is not None:
        X_all.append(img)
        valid_idx.append(i)

X_all = np.array(X_all, dtype=np.float32)
y_all = y_all[valid_idx]

print(f"\n  X_all shape : {X_all.shape}")
print(f"  y_all shape : {y_all.shape}")

# ─────────────────────────────────────────────
# 4.  TRAIN / TEST SPLIT  (80:20 stratified)
# ─────────────────────────────────────────────
print("\n[STEP 4] Splitting dataset 80/20 train-test (stratified) ...")
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_all, y_all, test_size=0.20, random_state=SEED, stratify=y_all)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.125, random_state=SEED, stratify=y_train_full)
# 0.125 x 0.80 = 0.10 -> 10% val from full

print(f"  Train : {X_train.shape[0]}")
print(f"  Val   : {X_val.shape[0]}")
print(f"  Test  : {X_test.shape[0]}")

# Class weights for imbalanced dataset
class_weights_arr = compute_class_weight(
    "balanced", classes=np.arange(NUM_CLASSES), y=y_train)
CLASS_WEIGHTS = {i: w for i, w in enumerate(class_weights_arr)}

# ─────────────────────────────────────────────
# 5.  SAMPLE VISUALIZATION
# ─────────────────────────────────────────────
print("\n[STEP 5] Saving sample images per class ...")
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
axes = axes.flatten()
plotted = set()
idx = 0
for i, (img, lbl) in enumerate(zip(X_train, y_train)):
    if lbl in plotted:
        continue
    axes[idx].imshow(img.squeeze(), cmap="gray")
    axes[idx].set_title(IDX_TO_CLASS[lbl].capitalize(), fontsize=12)
    axes[idx].axis("off")
    plotted.add(lbl)
    idx += 1
    if idx == NUM_CLASSES:
        break
axes[-1].axis("off")
plt.suptitle("Sample Images per Emotion Class", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "sample_images.png"), dpi=150)
plt.close()

# ─────────────────────────────────────────────
# 6.  ANN MODEL
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("[STEP 6] Building and Training ANN ...")
print("="*60)

X_train_flat = X_train.reshape(len(X_train), -1)
X_val_flat   = X_val.reshape(len(X_val), -1)
X_test_flat  = X_test.reshape(len(X_test), -1)

def build_ann(input_dim, num_classes):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(512, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(256, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation="softmax"),
    ], name="ANN_Emotion")
    return model

ann = build_ann(X_train_flat.shape[1], NUM_CLASSES)
ann.compile(optimizer=keras.optimizers.Adam(1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"])
ann.summary()

ann_callbacks = [
    callbacks.EarlyStopping(monitor="val_accuracy", patience=10,
                             restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                 patience=4, min_lr=1e-6, verbose=1),
    callbacks.ModelCheckpoint(
        os.path.join(MODELS_DIR, "ann_model.keras"),
        monitor="val_accuracy", save_best_only=True, verbose=0),
]

t0 = time.time()
ann_history = ann.fit(
    X_train_flat, y_train,
    validation_data=(X_val_flat, y_val),
    epochs=50, batch_size=64,
    class_weight=CLASS_WEIGHTS,
    callbacks=ann_callbacks,
    verbose=1)
ann_train_time = time.time() - t0
print(f"\n  ANN training time: {ann_train_time:.1f}s")

# ANN Evaluation
t1 = time.time()
ann_test_loss, ann_test_acc = ann.evaluate(X_test_flat, y_test, verbose=0)
ann_inf_time = (time.time() - t1) / len(X_test)
y_pred_ann = np.argmax(ann.predict(X_test_flat, verbose=0), axis=1)

ann_report = classification_report(y_test, y_pred_ann,
                                    target_names=EMOTION_CLASSES, output_dict=True)
ann_metrics = {
    "accuracy":  accuracy_score(y_test, y_pred_ann),
    "precision": precision_score(y_test, y_pred_ann, average="weighted", zero_division=0),
    "recall":    recall_score(y_test, y_pred_ann, average="weighted", zero_division=0),
    "f1":        f1_score(y_test, y_pred_ann, average="weighted", zero_division=0),
    "train_time": ann_train_time,
    "inf_time":   ann_inf_time,
    "report":     ann_report,
    "history":    ann_history.history,
}
print(f"\n  ANN Test Accuracy : {ann_metrics['accuracy']:.4f}")
print(classification_report(y_test, y_pred_ann, target_names=EMOTION_CLASSES))

# ─────────────────────────────────────────────
# 7.  CNN MODEL
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("[STEP 7] Building and Training CNN ...")
print("="*60)

# Data augmentation layer (applied only during training)
data_aug = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.05, 0.05),
], name="data_augmentation")

def build_cnn(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    x = data_aug(inputs)

    # Block 1
    x = layers.Conv2D(32, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Block 3
    x = layers.Conv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Block 4
    x = layers.Conv2D(256, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)

    # Dense head
    x = layers.Dense(256, activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs, name="CNN_Emotion")

cnn = build_cnn(INPUT_SHAPE_CNN, NUM_CLASSES)
cnn.compile(optimizer=keras.optimizers.Adam(1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"])
cnn.summary()

cnn_callbacks = [
    callbacks.EarlyStopping(monitor="val_accuracy", patience=12,
                             restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                 patience=5, min_lr=1e-6, verbose=1),
    callbacks.ModelCheckpoint(
        os.path.join(MODELS_DIR, "cnn_model.keras"),
        monitor="val_accuracy", save_best_only=True, verbose=0),
]

t0 = time.time()
cnn_history = cnn.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=60, batch_size=64,
    class_weight=CLASS_WEIGHTS,
    callbacks=cnn_callbacks,
    verbose=1)
cnn_train_time = time.time() - t0
print(f"\n  CNN training time: {cnn_train_time:.1f}s")

# CNN Evaluation
t1 = time.time()
cnn_test_loss, cnn_test_acc = cnn.evaluate(X_test, y_test, verbose=0)
cnn_inf_time = (time.time() - t1) / len(X_test)
y_pred_cnn = np.argmax(cnn.predict(X_test, verbose=0), axis=1)

cnn_report = classification_report(y_test, y_pred_cnn,
                                    target_names=EMOTION_CLASSES, output_dict=True)
cnn_metrics = {
    "accuracy":  accuracy_score(y_test, y_pred_cnn),
    "precision": precision_score(y_test, y_pred_cnn, average="weighted", zero_division=0),
    "recall":    recall_score(y_test, y_pred_cnn, average="weighted", zero_division=0),
    "f1":        f1_score(y_test, y_pred_cnn, average="weighted", zero_division=0),
    "train_time": cnn_train_time,
    "inf_time":   cnn_inf_time,
    "report":     cnn_report,
    "history":    cnn_history.history,
}
print(f"\n  CNN Test Accuracy : {cnn_metrics['accuracy']:.4f}")
print(classification_report(y_test, y_pred_cnn, target_names=EMOTION_CLASSES))

# ─────────────────────────────────────────────
# 8.  VISUALIZATIONS
# ─────────────────────────────────────────────
print("\n[STEP 8] Generating visualizations ...")

def plot_history(history, model_name, color):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history["accuracy"],     label="Train Acc",  color=color, linewidth=2)
    ax1.plot(history["val_accuracy"], label="Val Acc",    color=color, linewidth=2, linestyle="--")
    ax1.set_title(f"{model_name} - Accuracy", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(history["loss"],     label="Train Loss", color=color, linewidth=2)
    ax2.plot(history["val_loss"], label="Val Loss",   color=color, linewidth=2, linestyle="--")
    ax2.set_title(f"{model_name} - Loss", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.suptitle(f"{model_name} Training History", fontsize=16)
    plt.tight_layout()
    fname = os.path.join(OUT_DIR, f"{model_name.lower()}_history.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved: outputs/{model_name.lower()}_history.png")

plot_history(ann_history.history, "ANN", "#e74c3c")
plot_history(cnn_history.history, "CNN", "#3498db")

def plot_confusion_matrix(y_true, y_pred, model_name, cmap):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,
                xticklabels=EMOTION_CLASSES, yticklabels=EMOTION_CLASSES,
                ax=ax, linewidths=0.5)
    ax.set_title(f"{model_name} - Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    fname = os.path.join(OUT_DIR, f"{model_name.lower()}_confusion_matrix.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved: outputs/{model_name.lower()}_confusion_matrix.png")

plot_confusion_matrix(y_test, y_pred_ann, "ANN", "Reds")
plot_confusion_matrix(y_test, y_pred_cnn, "CNN", "Blues")

# ANN vs CNN Comparison Chart
metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
ann_vals = [ann_metrics["accuracy"], ann_metrics["precision"],
            ann_metrics["recall"],   ann_metrics["f1"]]
cnn_vals = [cnn_metrics["accuracy"], cnn_metrics["precision"],
            cnn_metrics["recall"],   cnn_metrics["f1"]]

x  = np.arange(len(metrics_names))
w  = 0.35
fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - w/2, ann_vals, w, label="ANN", color="#e74c3c", alpha=0.85)
bars2 = ax.bar(x + w/2, cnn_vals, w, label="CNN", color="#3498db", alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(metrics_names, fontsize=12)
ax.set_ylim(0, 1.1); ax.set_ylabel("Score")
ax.set_title("ANN vs CNN - Performance Comparison", fontsize=14, fontweight="bold")
ax.legend(fontsize=12); ax.grid(True, axis="y", alpha=0.3)
for bar in bars1:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
            f"{bar.get_height():.3f}", ha="center", fontsize=9)
for bar in bars2:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
            f"{bar.get_height():.3f}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "ann_vs_cnn_comparison.png"), dpi=150)
plt.close()
print("  Saved: outputs/ann_vs_cnn_comparison.png")

# Training time comparison
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(["ANN", "CNN"], [ann_train_time, cnn_train_time], color=["#e74c3c","#3498db"])
ax.set_title("Training Time Comparison (seconds)", fontsize=13, fontweight="bold")
ax.set_ylabel("Seconds"); ax.grid(True, axis="y", alpha=0.3)
for i, v in enumerate([ann_train_time, cnn_train_time]):
    ax.text(i, v+0.5, f"{v:.0f}s", ha="center", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "training_time_comparison.png"), dpi=150)
plt.close()

# Sample predictions
print("\n  Generating sample predictions ...")
fig, axes = plt.subplots(2, 7, figsize=(18, 6))
sample_indices = []
for cls in range(NUM_CLASSES):
    idxs = np.where(y_test == cls)[0]
    if len(idxs) > 0:
        sample_indices.append(idxs[0])

for col, idx in enumerate(sample_indices[:7]):
    img = X_test[idx]
    true_lbl  = IDX_TO_CLASS[y_test[idx]]
    # ANN
    ann_probs = ann.predict(img.reshape(1, -1), verbose=0)[0]
    ann_pred  = IDX_TO_CLASS[np.argmax(ann_probs)]
    ann_conf  = np.max(ann_probs)
    # CNN
    cnn_probs = cnn.predict(img[np.newaxis], verbose=0)[0]
    cnn_pred  = IDX_TO_CLASS[np.argmax(cnn_probs)]
    cnn_conf  = np.max(cnn_probs)

    axes[0, col].imshow(img.squeeze(), cmap="gray")
    color = "green" if ann_pred == true_lbl else "red"
    axes[0, col].set_title(f"ANN: {ann_pred}\n({ann_conf:.0%})", color=color, fontsize=9)
    axes[0, col].axis("off")

    axes[1, col].imshow(img.squeeze(), cmap="gray")
    color = "green" if cnn_pred == true_lbl else "red"
    axes[1, col].set_title(f"CNN: {cnn_pred}\n({cnn_conf:.0%})", color=color, fontsize=9)
    axes[1, col].axis("off")

axes[0, 0].set_ylabel("ANN Predictions", fontsize=10)
axes[1, 0].set_ylabel("CNN Predictions", fontsize=10)
plt.suptitle("Sample Predictions (Green=Correct, Red=Incorrect)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "sample_predictions.png"), dpi=150)
plt.close()
print("  Saved: outputs/sample_predictions.png")

# ─────────────────────────────────────────────
# 9.  SAVE MODELS AND ARTIFACTS
# ─────────────────────────────────────────────
print("\n[STEP 9] Saving model artifacts ...")

# Keras native format (reliable)
ann.save(os.path.join(MODELS_DIR, "ann_model.keras"))
cnn.save(os.path.join(MODELS_DIR, "cnn_model.keras"))

# Pickle metadata wrappers
ann_pkl = {
    "model_type":    "ANN",
    "model_path":    os.path.join(MODELS_DIR, "ann_model.keras"),
    "metrics":       {k: v for k, v in ann_metrics.items() if k != "history"},
    "input_shape":   INPUT_SHAPE_ANN,
    "class_to_idx":  CLASS_TO_IDX,
    "idx_to_class":  IDX_TO_CLASS,
    "img_size":      IMG_SIZE,
    "channels":      CHANNELS,
}
cnn_pkl = {
    "model_type":    "CNN",
    "model_path":    os.path.join(MODELS_DIR, "cnn_model.keras"),
    "metrics":       {k: v for k, v in cnn_metrics.items() if k != "history"},
    "input_shape":   INPUT_SHAPE_CNN,
    "class_to_idx":  CLASS_TO_IDX,
    "idx_to_class":  IDX_TO_CLASS,
    "img_size":      IMG_SIZE,
    "channels":      CHANNELS,
}
preprocessing_cfg = {
    "img_size":       IMG_SIZE,
    "channels":       CHANNELS,
    "class_to_idx":   CLASS_TO_IDX,
    "idx_to_class":   IDX_TO_CLASS,
    "num_classes":    NUM_CLASSES,
    "normalize":      "divide_by_255",
    "color_mode":     "grayscale" if CHANNELS == 1 else "rgb",
}

for name, obj in [("ann_model.pkl", ann_pkl),
                   ("cnn_model.pkl", cnn_pkl),
                   ("preprocessing_config.pkl", preprocessing_cfg)]:
    with open(os.path.join(MODELS_DIR, name), "wb") as f:
        pickle.dump(obj, f)

# Save metrics JSON for the API
import json
comparison_metrics = {
    "ann": {k: float(v) if isinstance(v, (np.floating, float)) else v
            for k, v in ann_metrics.items()
            if k not in ("history", "report")},
    "cnn": {k: float(v) if isinstance(v, (np.floating, float)) else v
            for k, v in cnn_metrics.items()
            if k not in ("history", "report")},
    "ann_report": ann_report,
    "cnn_report": cnn_report,
}
# make JSON serialisable
def make_serialisable(obj):
    if isinstance(obj, dict):
        return {k: make_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_serialisable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    return obj

with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
    json.dump(make_serialisable(comparison_metrics), f, indent=2)

# Save per-epoch history for frontend charts
history_data = {
    "ann": {k: [float(v) for v in vals] for k, vals in ann_history.history.items()},
    "cnn": {k: [float(v) for v in vals] for k, vals in cnn_history.history.items()},
}
with open(os.path.join(MODELS_DIR, "history.json"), "w") as f:
    json.dump(history_data, f, indent=2)

print("  Saved: models/ann_model.keras + ann_model.pkl")
print("  Saved: models/cnn_model.keras + cnn_model.pkl")
print("  Saved: models/preprocessing_config.pkl")
print("  Saved: models/metrics.json")
print("  Saved: models/history.json")

# ─────────────────────────────────────────────
# 10.  FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n" + "="*60)
print(" FINAL RESULTS")
print("="*60)
header = f"{'Metric':<20} {'ANN':>12} {'CNN':>12}"
print(header)
print("-"*len(header))
for metric, av, cv in [
    ("Accuracy",    ann_metrics["accuracy"],  cnn_metrics["accuracy"]),
    ("Precision",   ann_metrics["precision"], cnn_metrics["precision"]),
    ("Recall",      ann_metrics["recall"],    cnn_metrics["recall"]),
    ("F1-Score",    ann_metrics["f1"],        cnn_metrics["f1"]),
    ("Train Time s",ann_metrics["train_time"],cnn_metrics["train_time"]),
    ("Inf ms/img",  ann_metrics["inf_time"]*1000, cnn_metrics["inf_time"]*1000),
]:
    print(f"{metric:<20} {av:>12.4f} {cv:>12.4f}")

print("="*60)
print("\nTraining complete! All artifacts saved to models/ and outputs/")
print("Run `python app.py` to start the FastAPI backend.")
