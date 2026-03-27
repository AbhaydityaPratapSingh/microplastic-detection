"""
02_train_resnet18.py -- Train ResNet-18 for Microplastic Classification
========================================================================
Fine-tunes a pretrained ResNet-18 on cropped microplastic images (PE, PP, PS, PET).
Generates: trained model, training curves, confusion matrix, classification report.
"""

import os, sys, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
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
NUM_EPOCHS  = 30
LR          = 0.001
NUM_WORKERS = 0       # Windows-safe
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Data Transforms ─────────────────────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_data():
    """Load train/val/test datasets."""
    train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), train_transforms)
    val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"),   val_transforms)
    test_ds  = datasets.ImageFolder(os.path.join(DATA_DIR, "test"),  val_transforms)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    class_names = train_ds.classes
    print(f"  Classes: {class_names}")
    print(f"  Train: {len(train_ds)}  |  Val: {len(val_ds)}  |  Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader, class_names


def build_model(num_classes):
    """Load pretrained ResNet-18, replace final FC layer."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Freeze early layers (conv1 + layer1) for faster training
    for name, param in model.named_parameters():
        if "layer2" not in name and "layer3" not in name and "layer4" not in name and "fc" not in name:
            param.requires_grad = False

    # Replace final FC
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes),
    )
    return model.to(DEVICE)


def train_model(model, train_loader, val_loader, num_epochs):
    """Train with early stopping based on val accuracy."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    print(f"\n  Training on: {DEVICE}")
    print(f"  Epochs: {num_epochs}  |  LR: {LR}  |  Batch: {BATCH_SIZE}")
    print("-" * 60)

    for epoch in range(num_epochs):
        t0 = time.time()

        # ── Train phase ──────────────────────────────────────────────────
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # ── Val phase ────────────────────────────────────────────────────
        model.eval()
        running_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss = running_loss / total
        val_acc = correct / total

        scheduler.step()

        # Save history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        marker = " *BEST*" if val_acc > best_acc else ""
        print(f"  Epoch {epoch+1:2d}/{num_epochs} | "
              f"Train: {train_acc:.4f} ({train_loss:.4f}) | "
              f"Val: {val_acc:.4f} ({val_loss:.4f}) | "
              f"{elapsed:.1f}s{marker}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    print(f"\n  Best val accuracy: {best_acc:.4f}")

    # Load best weights
    model.load_state_dict(best_model_wts)
    return model, history


def evaluate_model(model, test_loader, class_names):
    """Run on test set and generate metrics."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Overall accuracy
    accuracy = (all_preds == all_labels).mean()
    print(f"\n  Test Accuracy: {accuracy:.4f} ({(all_preds == all_labels).sum()}/{len(all_labels)})")

    # Classification report
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print(f"\n{report}")

    report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"ResNet-18 Microplastic Classification Report\n")
        f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
        f.write(report)
    print(f"  Saved: {report_path}")

    return all_preds, all_labels


def plot_training_curves(history):
    """Plot loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)
    ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history["train_acc"], "b-", label="Train Accuracy", linewidth=2)
    ax2.plot(epochs, history["val_acc"], "r-", label="Val Accuracy", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "training_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_confusion_matrix(all_labels, all_preds, class_names):
    """Plot confusion matrix heatmap."""
    cm = confusion_matrix(all_labels, all_preds)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax,
                annot_kws={"size": 14})
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("ResNet-18 Confusion Matrix (Test Set)", fontsize=14)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_per_class_accuracy(all_labels, all_preds, class_names):
    """Bar chart of per-class accuracy."""
    per_class_acc = []
    for i, name in enumerate(class_names):
        mask = all_labels == i
        if mask.sum() > 0:
            acc = (all_preds[mask] == i).mean() * 100
        else:
            acc = 0.0
        per_class_acc.append(acc)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
    bars = ax.bar(class_names, per_class_acc, color=colors[:len(class_names)], edgecolor="white", linewidth=1.5)

    for bar, acc in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{acc:.1f}%", ha="center", fontsize=12, fontweight="bold")

    ax.set_ylim(0, 110)
    ax.set_xlabel("Polymer Class", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Per-Class Test Accuracy (ResNet-18)", fontsize=14)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "per_class_accuracy.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def save_model_summary(model, class_names):
    """Save model architecture summary."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total_params - trainable

    path = os.path.join(OUTPUT_DIR, "model_summary.txt")
    with open(path, "w") as f:
        f.write("ResNet-18 Microplastic Classifier - Model Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Architecture    : ResNet-18 (pretrained on ImageNet)\n")
        f.write(f"Input Size      : {IMG_SIZE}x{IMG_SIZE}x3\n")
        f.write(f"Output Classes  : {len(class_names)} ({', '.join(class_names)})\n")
        f.write(f"Total Parameters: {total_params:,}\n")
        f.write(f"Trainable       : {trainable:,}\n")
        f.write(f"Frozen          : {frozen:,}\n\n")
        f.write(f"Training Config:\n")
        f.write(f"  Optimizer     : Adam (lr={LR})\n")
        f.write(f"  Scheduler     : StepLR (step=10, gamma=0.5)\n")
        f.write(f"  Batch Size    : {BATCH_SIZE}\n")
        f.write(f"  Epochs        : {NUM_EPOCHS}\n")
        f.write(f"  Dropout       : 0.5\n")
        f.write(f"  Augmentation  : Flip, Rotate, ColorJitter, Affine, RandomErasing\n")
        f.write(f"  Device        : {DEVICE}\n")
    print(f"  Saved: {path}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  TRAIN ResNet-18 MICROPLASTIC CLASSIFIER")
    print("=" * 60)

    # 1. Load data
    print("\n[1/6] Loading data...")
    train_loader, val_loader, test_loader, class_names = load_data()

    # 2. Build model
    print("\n[2/6] Building model...")
    model = build_model(num_classes=len(class_names))
    save_model_summary(model, class_names)

    # 3. Train
    print("\n[3/6] Training...")
    model, history = train_model(model, train_loader, val_loader, NUM_EPOCHS)

    # 4. Save model
    print("\n[4/6] Saving model...")
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"  Saved: {MODEL_PATH}")

    # 5. Evaluate on test set
    print("\n[5/6] Evaluating on test set...")
    all_preds, all_labels = evaluate_model(model, test_loader, class_names)

    # 6. Generate plots
    print("\n[6/6] Generating plots...")
    plot_training_curves(history)
    plot_confusion_matrix(all_labels, all_preds, class_names)
    plot_per_class_accuracy(all_labels, all_preds, class_names)

    print("\n" + "=" * 60)
    print("  ALL DONE! Outputs saved to: resnet_outputs/")
    print("=" * 60)


if __name__ == "__main__":
    main()
