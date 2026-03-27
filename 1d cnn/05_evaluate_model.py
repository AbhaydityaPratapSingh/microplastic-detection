"""
05_evaluate_model.py — Evaluate the trained 1D-CNN
===================================================
Loads the trained model, runs inference on the held-out 20% split,
prints a full classification report, and saves a confusion matrix.
"""

import os

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

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

MODEL_PATH = "1d_cnn_spectral_model.pth"
CM_PATH    = "confusion_matrix.png"

# ═════════════════════════════════════════════════════════════════════════════
# 1. RECREATE THE TEST ENVIRONMENT
# ═════════════════════════════════════════════════════════════════════════════
print("[1/4] Preparing test data ...")
df = pd.read_csv("../full_spectra_with_labels.csv")

# Encode target labels (column index 1 = 'V0')
label_col = df.columns[1]
le = LabelEncoder()
y = le.fit_transform(df[label_col].values)
class_names = list(le.classes_)
num_classes = len(class_names)
print(f"  Target column : '{label_col}'")
print(f"  Classes ({num_classes}): {class_names}")

# Extract 256 spectral features (column index 3 onward)
X = df.iloc[:, 3:].values.astype(np.float32)

# Split with SAME random_state=42 used during training
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"  Test samples: {X_test.shape[0]}")

# Convert to PyTorch tensor — shape (Batch, 1, 256)
X_test_t = torch.tensor(X_test).unsqueeze(1)
print(f"  Tensor shape : {tuple(X_test_t.shape)}")

# ═════════════════════════════════════════════════════════════════════════════
# 2. LOAD THE MODEL
# ═════════════════════════════════════════════════════════════════════════════
class Microplastic1DCNN(nn.Module):
    """
    Exact architecture from 02_train_1d_cnn.py.
    Conv1d(1->16) -> ReLU -> MaxPool
    Conv1d(16->32) -> ReLU -> MaxPool
    Flatten -> Linear(2048->128) -> ReLU -> Dropout(0.5) -> Linear(128->num_classes)
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=1,  out_channels=16,
                      kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=16, out_channels=32,
                      kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


print("\n[2/4] Loading model weights ...")
model = Microplastic1DCNN(num_classes=num_classes)
state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
model.load_state_dict(state_dict)
model.eval()
print("  Model loaded and set to eval() mode.")

# ═════════════════════════════════════════════════════════════════════════════
# 3. GENERATE PREDICTIONS & METRICS
# ═════════════════════════════════════════════════════════════════════════════
print("\n[3/4] Running inference on test set ...")
with torch.no_grad():
    outputs = model(X_test_t)
    _, y_pred_t = torch.max(outputs, 1)

y_true = y_test                          # numpy array
y_pred = y_pred_t.numpy()                # tensor -> numpy

# Classification report
print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)
report = classification_report(y_true, y_pred, target_names=class_names)
print(report)

# ═════════════════════════════════════════════════════════════════════════════
# 4. GENERATE & SAVE CONFUSION MATRIX
# ═════════════════════════════════════════════════════════════════════════════
print("[4/4] Generating confusion matrix ...")
cm = confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
    linewidths=0.5,
    linecolor="gray",
    ax=ax,
)
ax.set_xlabel("Predicted Polymer", fontsize=13)
ax.set_ylabel("Actual Polymer", fontsize=13)
ax.set_title("Confusion Matrix — Microplastic 1D-CNN", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(CM_PATH, dpi=300)
plt.close(fig)

print(f"  Saved -> {CM_PATH}")
print("\nDone!")
