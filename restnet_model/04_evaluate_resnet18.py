"""
04_evaluate_resnet18.py -- ResNet-18 Performance Evaluation
============================================================
Loads the trained ResNet-18 model and evaluates it on the full
test set.  Prints accuracy, precision, recall, F1-score, and
confusion matrix to the terminal.  Saves plots to resnet_outputs/.
"""

import os, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)

# ── Config ───────────────────────────────────────────────────────────────────
DATA_DIR    = os.path.join(PROJECT_DIR, "../image_dataset")
OUTPUT_DIR  = os.path.join(PROJECT_DIR, "resnet_outputs")
MODEL_PATH  = os.path.join(PROJECT_DIR, "resnet18_microplastic.pth")
IMG_SIZE    = 64
BATCH_SIZE  = 32
NUM_WORKERS = 0
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES     = ['PE', 'PET', 'PP', 'PS']

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Data ─────────────────────────────────────────────────────────────────────
test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model():
    """Load the trained ResNet-18 model."""
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, len(CLASSES)),
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model


def run_evaluation(model, loader):
    """Run inference on every batch; return ground-truth & predicted arrays."""
    all_preds, all_labels = [], []

    t0 = time.time()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    elapsed = time.time() - t0

    return np.array(all_labels), np.array(all_preds), elapsed


def print_metrics(y_true, y_pred, elapsed):
    """Print all performance metrics to the terminal."""
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec  = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1   = f1_score(y_true, y_pred, average=None, zero_division=0)

    macro_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)

    wtd_prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    wtd_rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    wtd_f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # ── Overall ──────────────────────────────────────────────────────────
    print(f"\n  Overall Accuracy : {acc*100:.2f}%  ({(y_true == y_pred).sum()}/{len(y_true)})")
    print(f"  Inference Time   : {elapsed:.2f}s  ({len(y_true)} images)")
    print(f"  Throughput       : {len(y_true)/elapsed:.1f} images/s")
    print(f"  Device           : {DEVICE}")

    # ── Per-class table ──────────────────────────────────────────────────
    print("\n  " + "-" * 58)
    print(f"  {'Class':<8} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("  " + "-" * 58)
    for i, cls in enumerate(CLASSES):
        support = (y_true == i).sum()
        print(f"  {cls:<8} {prec[i]*100:>9.2f}% {rec[i]*100:>9.2f}% {f1[i]*100:>9.2f}% {support:>10d}")
    print("  " + "-" * 58)
    print(f"  {'Macro':<8} {macro_prec*100:>9.2f}% {macro_rec*100:>9.2f}% {macro_f1*100:>9.2f}%")
    print(f"  {'Weighted':<8} {wtd_prec*100:>9.2f}% {wtd_rec*100:>9.2f}% {wtd_f1*100:>9.2f}%")

    # ── sklearn detailed report ──────────────────────────────────────────
    print("\n  Full Classification Report:")
    report = classification_report(y_true, y_pred, target_names=CLASSES, digits=4)
    for line in report.splitlines():
        print(f"  {line}")

    # ── Confusion Matrix (text) ──────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    print("\n  Confusion Matrix:")
    header = "          " + "  ".join(f"{c:>5}" for c in CLASSES)
    print(f"  {header}")
    for i, cls in enumerate(CLASSES):
        row = "  ".join(f"{v:>5d}" for v in cm[i])
        print(f"  {cls:<8}  {row}")

    return cm


def save_confusion_matrix(cm):
    """Save a confusion matrix heatmap to resnet_outputs/."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax,
                annot_kws={"size": 14})
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("ResNet-18 Confusion Matrix (Test Set)", fontsize=14)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "eval_confusion_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {path}")


def save_per_class_accuracy(y_true, y_pred):
    """Save a per-class accuracy bar chart to resnet_outputs/."""
    accs = []
    for i, cls in enumerate(CLASSES):
        mask = y_true == i
        accs.append((y_pred[mask] == i).mean() * 100 if mask.sum() > 0 else 0.0)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
    bars = ax.bar(CLASSES, accs, color=colors[:len(CLASSES)],
                  edgecolor="white", linewidth=1.5)
    for bar, a in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{a:.1f}%", ha="center", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.set_xlabel("Polymer Class", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Per-Class Test Accuracy (ResNet-18)", fontsize=14)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "eval_per_class_accuracy.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("      ResNet-18 PERFORMANCE EVALUATION (Test Set)      ")
    print("=" * 65)

    # 1. Load model
    print("\n[1/4] Loading model...")
    try:
        model = load_model()
        print(f"  Loaded: {MODEL_PATH}")
    except Exception as e:
        print(f"\n[ERROR] Could not load model: {e}")
        return

    # 2. Load test data
    print("\n[2/4] Loading test data...")
    test_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), test_transforms)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS)
    print(f"  Test images : {len(test_ds)}")
    print(f"  Classes     : {CLASSES}")

    # 3. Run evaluation
    print("\n[3/4] Running inference on test set...")
    y_true, y_pred, elapsed = run_evaluation(model, test_loader)

    # 4. Print & save metrics
    print("\n[4/4] Results:")
    cm = print_metrics(y_true, y_pred, elapsed)
    save_confusion_matrix(cm)
    save_per_class_accuracy(y_true, y_pred)

    print("\n" + "=" * 65)
    print("  Evaluation complete. Plots saved to: resnet_outputs/")
    print("=" * 65)


if __name__ == "__main__":
    main()
