"""
presentation_outputs.py — Generate ALL Outputs for 20-min 1D-CNN Presentation
===============================================================================
Run this ONCE.  Everything is saved into  ./presentation_outputs/

Outputs generated:
  1. presentation_summary.txt      — console-friendly text dump of ALL results
  2. 01_dataset_class_distribution.png
  3. 02_sample_spectra.png          — one spectrum per class
  4. 03_model_architecture.txt      — architecture + param count
  5. 04_training_curves.png         — loss & accuracy over 25 epochs (re-train)
  6. 05_confusion_matrix.png        — on held-out 20%
  7. 06_classification_report.txt   — precision/recall/F1
  8. 07_per_class_accuracy.png      — bar chart
  9. 08_cross_validation.png        — 5-fold accuracies bar chart
  10. 09_learning_curves_cv.png     — train vs val loss (last fold)
  11. 10_live_inference_demo.txt     — 10 random sample predictions with timing
"""

import os
import sys
import time
import random

# ── PyTorch DLL fix (Windows) ────────────────────────────────────────────────
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
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)

OUT_DIR    = os.path.join(PROJECT_DIR, "presentation_outputs")
MODEL_PATH = "1d_cnn_spectral_model.pth"
CSV_PATH   = "../full_spectra_with_labels.csv"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Redirect print to both console and file ──────────────────────────────────
summary_path = os.path.join(OUT_DIR, "presentation_summary.txt")
summary_lines = []

_orig_print = print
def print(*args, **kwargs):
    _orig_print(*args, **kwargs)
    import io
    buf = io.StringIO()
    kwargs2 = dict(kwargs)
    kwargs2['file'] = buf
    _orig_print(*args, **kwargs2)
    summary_lines.append(buf.getvalue())

print("=" * 70)
print("  1D-CNN MICROPLASTIC CLASSIFIER — PRESENTATION OUTPUT GENERATOR")
print("=" * 70)
print(f"  Output folder: {OUT_DIR}\n")

# ═════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITION
# ═════════════════════════════════════════════════════════════════════════════
class Microplastic1DCNN(nn.Module):
    """
    1D-CNN for hyperspectral polymer classification.
    Input:  (Batch, 1, 256)
    Conv1d(1->16, k=5) -> ReLU -> MaxPool(2)  => (Batch, 16, 128)
    Conv1d(16->32, k=5) -> ReLU -> MaxPool(2) => (Batch, 32, 64)
    Flatten => (Batch, 2048)
    FC(2048->128) -> ReLU -> Dropout(0.5) -> FC(128->num_classes)
    """
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
# 1. LOAD DATA
# ═════════════════════════════════════════════════════════════════════════════
print("-" * 70)
print("  [1/10] LOADING DATASET")
print("-" * 70)
df = pd.read_csv(CSV_PATH)
label_col = df.columns[1]  # 'V0' — polymer type
le = LabelEncoder()
y = le.fit_transform(df[label_col].values)
class_names = list(le.classes_)
num_classes = len(class_names)
X = df.iloc[:, 3:].values.astype(np.float32)

print(f"  CSV shape           : {df.shape}")
print(f"  Feature channels    : {X.shape[1]}")
print(f"  Total samples       : {X.shape[0]}")
print(f"  Target column       : '{label_col}'")
print(f"  Classes ({num_classes})        : {class_names}")
print(f"  Samples per class   :")
for cls in class_names:
    count = (df[label_col] == cls).sum()
    print(f"    {cls:>4s} : {count:>5d}  ({100*count/len(df):.1f}%)")

# ═════════════════════════════════════════════════════════════════════════════
# 2. CLASS DISTRIBUTION CHART
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 70)
print("  [2/10] CLASS DISTRIBUTION")
print("-" * 70)

counts = df[label_col].value_counts()
colors = ['#0077B6', '#00B4D8', '#90E0EF', '#E76F51']
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(counts.index, counts.values, color=colors[:len(counts)], edgecolor='#333', linewidth=0.8)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            f'{val}', ha='center', va='bottom', fontsize=13, fontweight='bold')
ax.set_xlabel("Polymer Type", fontsize=13)
ax.set_ylabel("Number of Samples", fontsize=13)
ax.set_title("Dataset — Class Distribution", fontsize=15, fontweight='bold')
ax.grid(axis='y', linestyle='--', alpha=0.4)
fig.tight_layout()
path = os.path.join(OUT_DIR, "01_dataset_class_distribution.png")
fig.savefig(path, dpi=200)
plt.close(fig)
print(f"  Saved -> {path}")

# ═════════════════════════════════════════════════════════════════════════════
# 3. SAMPLE SPECTRA (one per class)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 70)
print("  [3/10] SAMPLE SPECTRA PER CLASS")
print("-" * 70)

# Try loading wavelength axis
try:
    import rdata
    wl_data = rdata.read_rda("../wavelength.rda")
    wavelengths = np.array(list(wl_data.values())[0]).flatten().astype(float)
    x_label = "Wavelength (nm)"
except Exception:
    wavelengths = np.arange(X.shape[1])
    x_label = "Channel Index"

spec_colors = ['#0077B6', '#E76F51', '#2A9D8F', '#E9C46A']
fig, ax = plt.subplots(figsize=(12, 5))
for i, cls in enumerate(class_names):
    idx = np.where(df[label_col].values == cls)[0][0]
    ax.plot(wavelengths, X[idx], color=spec_colors[i % len(spec_colors)],
            linewidth=1.2, label=cls, alpha=0.85)
ax.set_xlabel(x_label, fontsize=12)
ax.set_ylabel("Reflectance", fontsize=12)
ax.set_title("Sample Reflectance Spectra — One Per Polymer Class", fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, linestyle='--', alpha=0.4)
fig.tight_layout()
path = os.path.join(OUT_DIR, "02_sample_spectra.png")
fig.savefig(path, dpi=200)
plt.close(fig)
print(f"  Saved -> {path}")

# ═════════════════════════════════════════════════════════════════════════════
# 4. MODEL ARCHITECTURE
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 70)
print("  [4/10] MODEL ARCHITECTURE")
print("-" * 70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Microplastic1DCNN(num_classes=num_classes).to(device)
state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model.eval()

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n  Architecture:\n{model}\n")
print(f"  Total parameters     : {total_params:,}")
print(f"  Trainable parameters : {trainable_params:,}")
print(f"  Device               : {device}")

# Layer-by-layer summary
arch_text = []
arch_text.append("=" * 60)
arch_text.append("  1D-CNN MODEL ARCHITECTURE SUMMARY")
arch_text.append("=" * 60)
arch_text.append(f"\n{model}\n")
arch_text.append(f"Total parameters     : {total_params:,}")
arch_text.append(f"Trainable parameters : {trainable_params:,}")
arch_text.append("")
arch_text.append("Layer-by-layer parameter count:")
for name, param in model.named_parameters():
    arch_text.append(f"  {name:40s}  shape={str(list(param.shape)):20s}  params={param.numel():>8,}")
arch_text.append("=" * 60)

arch_path = os.path.join(OUT_DIR, "03_model_architecture.txt")
with open(arch_path, "w") as f:
    f.write("\n".join(arch_text))
print(f"  Saved -> {arch_path}")

# ═════════════════════════════════════════════════════════════════════════════
# 5. TRAINING CURVES (re-train quickly for plots)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 70)
print("  [5/10] TRAINING CURVES (25 epochs)")
print("-" * 70)

BATCH_SIZE = 32
EPOCHS = 25
LR = 0.001

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

X_train_t = torch.tensor(X_train).unsqueeze(1).to(device)
X_val_t   = torch.tensor(X_val).unsqueeze(1).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
y_val_t   = torch.tensor(y_val, dtype=torch.long).to(device)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=BATCH_SIZE, shuffle=False)

# Fresh model for training curves
train_model = Microplastic1DCNN(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(train_model.parameters(), lr=LR)

train_losses, val_losses, train_accs, val_accs = [], [], [], []

for epoch in range(1, EPOCHS + 1):
    # Train
    train_model.train()
    running_loss, correct, total = 0.0, 0, 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        out = train_model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
        _, preds = torch.max(out, 1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)
    train_losses.append(running_loss / total)
    train_accs.append(100.0 * correct / total)

    # Validate
    train_model.eval()
    v_loss, v_correct, v_total = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            out = train_model(xb)
            v_loss += criterion(out, yb).item() * xb.size(0)
            _, preds = torch.max(out, 1)
            v_correct += (preds == yb).sum().item()
            v_total += yb.size(0)
    val_losses.append(v_loss / v_total)
    val_accs.append(100.0 * v_correct / v_total)

    if epoch % 5 == 0 or epoch == 1:
        print(f"    Epoch {epoch:>3d}/{EPOCHS}  |  "
              f"Train Loss: {train_losses[-1]:.4f}  Train Acc: {train_accs[-1]:.2f}%  |  "
              f"Val Loss: {val_losses[-1]:.4f}  Val Acc: {val_accs[-1]:.2f}%")

# Plot training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss
ax1.plot(range(1, EPOCHS+1), train_losses, 'o-', color='#0077B6', label='Train Loss', linewidth=2, markersize=4)
ax1.plot(range(1, EPOCHS+1), val_losses,   's-', color='#E76F51', label='Val Loss',   linewidth=2, markersize=4)
ax1.set_xlabel("Epoch", fontsize=12)
ax1.set_ylabel("CrossEntropy Loss", fontsize=12)
ax1.set_title("Training & Validation Loss", fontsize=13, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, linestyle='--', alpha=0.4)

# Accuracy
ax2.plot(range(1, EPOCHS+1), train_accs, 'o-', color='#0077B6', label='Train Acc', linewidth=2, markersize=4)
ax2.plot(range(1, EPOCHS+1), val_accs,   's-', color='#E76F51', label='Val Acc',   linewidth=2, markersize=4)
ax2.set_xlabel("Epoch", fontsize=12)
ax2.set_ylabel("Accuracy (%)", fontsize=12)
ax2.set_title("Training & Validation Accuracy", fontsize=13, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, linestyle='--', alpha=0.4)

fig.suptitle("1D-CNN Training Progress — 25 Epochs", fontsize=15, fontweight='bold', y=1.02)
fig.tight_layout()
path = os.path.join(OUT_DIR, "04_training_curves.png")
fig.savefig(path, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"  Saved -> {path}")

# ═════════════════════════════════════════════════════════════════════════════
# 6. CONFUSION MATRIX (using saved model)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 70)
print("  [6/10] CONFUSION MATRIX")
print("-" * 70)

model.eval()
X_test_t = torch.tensor(X_val).unsqueeze(1).to(device)
with torch.no_grad():
    outputs = model(X_test_t)
    _, y_pred_t = torch.max(outputs, 1)
y_true = y_val
y_pred = y_pred_t.cpu().numpy()

cm = confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.8, linecolor='gray', ax=ax,
            annot_kws={"size": 14, "fontweight": "bold"})
ax.set_xlabel("Predicted Polymer", fontsize=13)
ax.set_ylabel("Actual Polymer", fontsize=13)
ax.set_title("Confusion Matrix — 1D-CNN Microplastic Classifier\n(Held-out 20% Test Set)",
             fontsize=14, fontweight='bold')
fig.tight_layout()
path = os.path.join(OUT_DIR, "05_confusion_matrix.png")
fig.savefig(path, dpi=200)
plt.close(fig)
print(f"  Saved -> {path}")

# ═════════════════════════════════════════════════════════════════════════════
# 7. CLASSIFICATION REPORT
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 70)
print("  [7/10] CLASSIFICATION REPORT")
print("-" * 70)

report = classification_report(y_true, y_pred, target_names=class_names)
print(f"\n{report}")

report_path = os.path.join(OUT_DIR, "06_classification_report.txt")
with open(report_path, "w") as f:
    f.write("=" * 60 + "\n")
    f.write("CLASSIFICATION REPORT — 1D-CNN Microplastic Classifier\n")
    f.write("(Held-out 20% Test Set)\n")
    f.write("=" * 60 + "\n\n")
    f.write(report)
    f.write("\n" + "=" * 60 + "\n")
print(f"  Saved -> {report_path}")

# ═════════════════════════════════════════════════════════════════════════════
# 8. PER-CLASS ACCURACY BAR CHART
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 70)
print("  [8/10] PER-CLASS ACCURACY")
print("-" * 70)

per_class_acc = []
for i, cls in enumerate(class_names):
    mask = (y_true == i)
    if mask.sum() > 0:
        acc = 100.0 * (y_pred[mask] == i).sum() / mask.sum()
    else:
        acc = 0.0
    per_class_acc.append(acc)
    print(f"    {cls:>4s} : {acc:.2f}%")

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(class_names, per_class_acc, color=colors[:num_classes],
              edgecolor='#333', linewidth=0.8)
for bar, val in zip(bars, per_class_acc):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=13, fontweight='bold')
ax.set_ylim(0, 105)
ax.set_xlabel("Polymer Type", fontsize=13)
ax.set_ylabel("Accuracy (%)", fontsize=13)
ax.set_title("Per-Class Accuracy — 1D-CNN Classifier", fontsize=14, fontweight='bold')
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.axhline(y=np.mean(per_class_acc), color='red', linestyle='--', linewidth=1.5,
           label=f'Mean: {np.mean(per_class_acc):.1f}%')
ax.legend(fontsize=11)
fig.tight_layout()
path = os.path.join(OUT_DIR, "07_per_class_accuracy.png")
fig.savefig(path, dpi=200)
plt.close(fig)
print(f"  Saved -> {path}")

# ═════════════════════════════════════════════════════════════════════════════
# 9. 5-FOLD CROSS-VALIDATION
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 70)
print("  [9/10] 5-FOLD CROSS-VALIDATION (10 epochs/fold)")
print("-" * 70)

N_SPLITS = 5
EPOCHS_CV = 10
kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
fold_accuracies = []
last_train_losses, last_val_losses = [], []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y), start=1):
    t0 = time.time()
    X_tr = torch.tensor(X[train_idx]).unsqueeze(1).to(device)
    y_tr = torch.tensor(y[train_idx], dtype=torch.long).to(device)
    X_vl = torch.tensor(X[val_idx]).unsqueeze(1).to(device)
    y_vl = torch.tensor(y[val_idx], dtype=torch.long).to(device)

    tr_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    vl_loader = DataLoader(TensorDataset(X_vl, y_vl), batch_size=BATCH_SIZE, shuffle=False)

    fold_model = Microplastic1DCNN(num_classes=num_classes).to(device)
    fold_crit = nn.CrossEntropyLoss()
    fold_opt = torch.optim.Adam(fold_model.parameters(), lr=LR)

    f_train_losses, f_val_losses = [], []

    for ep in range(1, EPOCHS_CV + 1):
        fold_model.train()
        rl = 0.0
        for xb, yb in tr_loader:
            fold_opt.zero_grad()
            loss = fold_crit(fold_model(xb), yb)
            loss.backward()
            fold_opt.step()
            rl += loss.item() * xb.size(0)
        f_train_losses.append(rl / len(tr_loader.dataset))

        fold_model.eval()
        vl_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in vl_loader:
                out = fold_model(xb)
                vl_loss += fold_crit(out, yb).item() * xb.size(0)
                _, preds = torch.max(out, 1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        f_val_losses.append(vl_loss / total)

    val_acc = 100.0 * correct / total
    fold_accuracies.append(val_acc)
    elapsed = time.time() - t0
    print(f"    Fold {fold_idx}/{N_SPLITS}  |  Val Acc: {val_acc:.2f}%  |  {elapsed:.1f}s")

    if fold_idx == N_SPLITS:
        last_train_losses = f_train_losses
        last_val_losses = f_val_losses

mean_acc = np.mean(fold_accuracies)
std_acc = np.std(fold_accuracies)
print(f"\n    Mean Accuracy : {mean_acc:.2f}%  ±  {std_acc:.2f}%")

# CV bar chart
fig, ax = plt.subplots(figsize=(8, 5))
fold_labels = [f"Fold {i+1}" for i in range(N_SPLITS)]
bars = ax.bar(fold_labels, fold_accuracies, color='#0077B6', edgecolor='#333', linewidth=0.8)
for bar, val in zip(bars, fold_accuracies):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f'{val:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax.axhline(y=mean_acc, color='red', linestyle='--', linewidth=1.5,
           label=f'Mean: {mean_acc:.2f}% ± {std_acc:.2f}%')
ax.set_ylim(min(fold_accuracies) - 2, 102)
ax.set_xlabel("Fold", fontsize=13)
ax.set_ylabel("Accuracy (%)", fontsize=13)
ax.set_title("5-Fold Cross-Validation Results", fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(axis='y', linestyle='--', alpha=0.4)
fig.tight_layout()
path = os.path.join(OUT_DIR, "08_cross_validation.png")
fig.savefig(path, dpi=200)
plt.close(fig)
print(f"  Saved -> {path}")

# Learning curves from last fold
fig, ax = plt.subplots(figsize=(9, 5))
epochs_range = range(1, EPOCHS_CV + 1)
ax.plot(epochs_range, last_train_losses, 'o-', color='#0077B6', label='Train Loss', linewidth=2)
ax.plot(epochs_range, last_val_losses,   's-', color='#E76F51', label='Val Loss',   linewidth=2)
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Loss (CrossEntropy)", fontsize=12)
ax.set_title("Learning Curves — Last Fold (Train vs Validation Loss)", fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, linestyle='--', alpha=0.4)
ax.set_xticks(list(epochs_range))
fig.tight_layout()
path = os.path.join(OUT_DIR, "09_learning_curves_cv.png")
fig.savefig(path, dpi=200)
plt.close(fig)
print(f"  Saved -> {path}")

# ═════════════════════════════════════════════════════════════════════════════
# 10. LIVE INFERENCE DEMO (10 random samples)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 70)
print("  [10/10] LIVE INFERENCE DEMO -- 10 Random Samples")
print("-" * 70)

y_test_labels = le.inverse_transform(y_val)
random.seed(None)

header = f"{'#':>3}  {'Actual':>6}  {'Predicted':>10}  {'Match':>5}  {'Time':>10}  {'Confidence':>10}"
print(f"\n{header}")
print("-" * len(header))

demo_lines = [header, "-" * len(header)]
correct_count = 0

for run in range(1, 11):
    idx = random.randint(0, len(X_val) - 1)
    actual = y_test_labels[idx]
    feat = X_val[idx]
    x_t = torch.tensor(feat).unsqueeze(0).unsqueeze(0).to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(x_t)
        probs = torch.softmax(out, dim=1)
        conf, pred_idx = torch.max(probs, 1)
    t1 = time.perf_counter()

    predicted = class_names[pred_idx.item()]
    elapsed_ms = (t1 - t0) * 1000
    match = "YES" if predicted == actual else "NO"
    if predicted == actual:
        correct_count += 1

    line = f"{run:>3}  {actual:>6}  {predicted:>10}  {match:>5}  {elapsed_ms:>8.2f}ms  {conf.item()*100:>8.1f}%"
    print(line)
    demo_lines.append(line)

demo_lines.append("-" * len(header))
demo_lines.append(f"Accuracy on demo: {correct_count}/10 = {100*correct_count/10:.0f}%")
print(f"\n  Demo accuracy: {correct_count}/10")

demo_path = os.path.join(OUT_DIR, "10_live_inference_demo.txt")
with open(demo_path, "w", encoding="utf-8") as f:
    f.write("LIVE INFERENCE DEMO — 10 Random Samples from Held-Out Test Set\n")
    f.write("=" * 70 + "\n\n")
    f.write("\n".join(demo_lines))
    f.write("\n")
print(f"  Saved -> {demo_path}")

# ═════════════════════════════════════════════════════════════════════════════
# RISK ASSESSMENT
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 70)
print("  FINAL RISK ASSESSMENT")
print("-" * 70)

gap = abs(last_train_losses[-1] - last_val_losses[-1])
overfit_risk = "LOW" if gap < 0.1 else ("MEDIUM" if gap < 0.3 else "HIGH")
cv_robust = std_acc < 1.0

print(f"  Overfitting risk     : {overfit_risk} (train-val loss gap = {gap:.4f})")
print(f"  CV robustness        : {'ROBUST' if cv_robust else 'UNSTABLE'} (SD = {std_acc:.2f}%)")
print(f"  Data leakage (dupes) : {df.duplicated().sum()} duplicate rows")

if cv_robust and overfit_risk == "LOW":
    print(f"\n  [OK] The model DOES NOT appear to be overfitting.")
    print(f"  [OK] The ~{mean_acc:.0f}% accuracy is SCIENTIFICALLY SUPPORTED.")
else:
    print(f"\n  [!] Further investigation recommended.")

# ═════════════════════════════════════════════════════════════════════════════
# SAVE COMPLETE SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("".join(summary_lines))

print(f"\n{'=' * 70}")
print(f"  ALL OUTPUTS SAVED TO: {OUT_DIR}")
print(f"{'=' * 70}")
print(f"\n  Files generated:")
for fname in sorted(os.listdir(OUT_DIR)):
    fsize = os.path.getsize(os.path.join(OUT_DIR, fname))
    print(f"    {fname:45s}  ({fsize:>8,} bytes)")
print(f"\n  DONE! Ready for your 20-min presentation.")
