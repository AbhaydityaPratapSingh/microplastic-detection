"""
07_overfitting_check.py — Overfitting Check & Robustness Analysis
==================================================================
1. 5-Fold Cross-Validation (fresh model per fold, 10 epochs each)
2. Learning Curve Analysis (train vs val loss on last fold)
3. Data Leakage Audit (duplicate row detection)
4. Risk Assessment conclusion
"""

import os
import time

# Fix for PyTorch DLL loading on some Windows setups
# torch must be imported BEFORE numpy to avoid a DLL conflict.
_torch_lib = os.path.join(
    os.path.expanduser("~"),
    r"AppData\Roaming\Python\Python312\site-packages\torch\lib"
)
if os.path.isdir(_torch_lib):
    os.environ["PATH"] = _torch_lib + os.pathsep + os.environ.get("PATH", "")
    os.add_dll_directory(_torch_lib)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

# ── Hyperparameters ──────────────────────────────────────────────────────────
BATCH_SIZE    = 32
EPOCHS_PER_FOLD = 10
LEARNING_RATE = 0.001
N_SPLITS      = 5
LC_PATH       = "learning_curves.png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")


# ── Model definition ────────────────────────────────────────────────────────
class Microplastic1DCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ═════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═════════════════════════════════════════════════════════════════════════════
print("Loading dataset ...")
df = pd.read_csv("../full_spectra_with_labels.csv")

label_col = df.columns[1]
le = LabelEncoder()
y = le.fit_transform(df[label_col].values)
class_names = list(le.classes_)
num_classes = len(class_names)

X = df.iloc[:, 3:].values.astype(np.float32)
print(f"  X shape: {X.shape}   Classes ({num_classes}): {class_names}\n")

# ═════════════════════════════════════════════════════════════════════════════
# 1. 5-FOLD CROSS-VALIDATION
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 62)
print(f"  5-FOLD CROSS-VALIDATION  ({EPOCHS_PER_FOLD} epochs per fold)")
print("=" * 62)

kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
fold_accuracies = []

# We will capture learning curves from the LAST fold
last_fold_train_losses = []
last_fold_val_losses   = []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y), start=1):
    t0 = time.time()

    # Prepare data for this fold
    X_tr = torch.tensor(X[train_idx]).unsqueeze(1).to(device)
    y_tr = torch.tensor(y[train_idx], dtype=torch.long).to(device)
    X_vl = torch.tensor(X[val_idx]).unsqueeze(1).to(device)
    y_vl = torch.tensor(y[val_idx], dtype=torch.long).to(device)

    train_loader = DataLoader(TensorDataset(X_tr, y_tr),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_vl, y_vl),
                              batch_size=BATCH_SIZE, shuffle=False)

    # Fresh model for each fold
    model = Microplastic1DCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    fold_train_losses = []
    fold_val_losses   = []

    for epoch in range(1, EPOCHS_PER_FOLD + 1):
        # ── Train ────────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        fold_train_losses.append(train_loss)

        # ── Validate ─────────────────────────────────────────────────────
        model.eval()
        val_running = 0.0
        correct = 0
        total   = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                out = model(xb)
                val_running += criterion(out, yb).item() * xb.size(0)
                _, preds = torch.max(out, 1)
                correct += (preds == yb).sum().item()
                total   += yb.size(0)
        val_loss = val_running / total
        fold_val_losses.append(val_loss)

    val_acc = 100.0 * correct / total
    fold_accuracies.append(val_acc)
    elapsed = time.time() - t0

    print(f"  Fold {fold_idx}/{N_SPLITS}  |  "
          f"Val Acc: {val_acc:6.2f}%  |  "
          f"Final Train Loss: {fold_train_losses[-1]:.4f}  |  "
          f"Final Val Loss: {fold_val_losses[-1]:.4f}  |  "
          f"{elapsed:.1f}s")

    # Keep learning curves from last fold
    if fold_idx == N_SPLITS:
        last_fold_train_losses = fold_train_losses
        last_fold_val_losses   = fold_val_losses

mean_acc = np.mean(fold_accuracies)
std_acc  = np.std(fold_accuracies)

print("-" * 62)
print(f"  Mean Accuracy : {mean_acc:.2f}%")
print(f"  Std Deviation : {std_acc:.2f}%")
print("=" * 62)

# ═════════════════════════════════════════════════════════════════════════════
# 2. LEARNING CURVE ANALYSIS (from last fold)
# ═════════════════════════════════════════════════════════════════════════════
print(f"\nPlotting learning curves (Fold {N_SPLITS}) ...")

epochs_range = range(1, EPOCHS_PER_FOLD + 1)
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(epochs_range, last_fold_train_losses, "o-", label="Train Loss", linewidth=2)
ax.plot(epochs_range, last_fold_val_losses,   "s-", label="Val Loss",   linewidth=2)
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Loss (CrossEntropy)", fontsize=12)
ax.set_title("Learning Curves — Fold 5 (Train vs Validation Loss)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, linestyle="--", alpha=0.5)
ax.set_xticks(list(epochs_range))
fig.tight_layout()
fig.savefig(LC_PATH, dpi=300)
plt.close(fig)
print(f"  Saved -> {LC_PATH}")

# ═════════════════════════════════════════════════════════════════════════════
# 3. DATA LEAKAGE AUDIT
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("  DATA LEAKAGE AUDIT")
print("=" * 62)

# Check for exact duplicate rows (all columns)
num_duplicates = df.duplicated().sum()
print(f"  Total rows          : {len(df)}")
print(f"  Duplicate rows      : {num_duplicates}")
print(f"  Unique rows         : {len(df) - num_duplicates}")

if num_duplicates > 0:
    dup_pct = 100.0 * num_duplicates / len(df)
    print(f"  Duplicate percentage: {dup_pct:.1f}%")
    print("  [!] Duplicates detected — review if any leak across train/val splits.")
else:
    print("  [OK] No duplicate rows — no risk of data leakage from duplicates.")

# ═════════════════════════════════════════════════════════════════════════════
# 4. RISK ASSESSMENT CONCLUSION
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("  RISK ASSESSMENT")
print("=" * 62)

# Evaluate overfitting risk
gap = abs(last_fold_train_losses[-1] - last_fold_val_losses[-1])
overfit_risk = "LOW" if gap < 0.1 else ("MEDIUM" if gap < 0.3 else "HIGH")

cv_robust = std_acc < 1.0   # SD < 1% across folds => robust

if cv_robust and overfit_risk == "LOW":
    verdict = (
        "The model DOES NOT appear to be overfitting.\n"
        f"  - 5-Fold CV Mean: {mean_acc:.2f}% (SD={std_acc:.2f}%) shows consistent "
        "performance across all folds.\n"
        f"  - Train-Val loss gap ({gap:.4f}) is negligible, confirming good "
        "generalization.\n"
        "  - The ~99% accuracy is SCIENTIFICALLY SUPPORTED by the cross-validation "
        "and learning curve evidence."
    )
elif cv_robust and overfit_risk != "LOW":
    verdict = (
        f"  - CV is robust (SD={std_acc:.2f}%), but the train-val loss gap ({gap:.4f}) "
        f"suggests {overfit_risk} overfitting risk.\n"
        "  - Consider adding regularization or reducing model complexity."
    )
else:
    verdict = (
        f"  - High variance across folds (SD={std_acc:.2f}%) suggests the model's "
        "performance is unstable.\n"
        f"  - Overfitting risk: {overfit_risk} (loss gap={gap:.4f}).\n"
        "  - Consider using more data, data augmentation, or simpler architecture."
    )

print(f"  Overfitting Risk : {overfit_risk}")
print(f"  CV Robustness    : {'ROBUST' if cv_robust else 'UNSTABLE'} "
      f"(SD={std_acc:.2f}%)")
print()
print(f"  {verdict}")
print("=" * 62)
print("\nDone!")
