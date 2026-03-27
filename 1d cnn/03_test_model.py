"""
03_test_model.py — Live Inference Demo for Project Review Panel
================================================================
Runs 5 random predictions with timing, then pops up a spectral
plot for the last prediction showing Actual vs Predicted label.
"""

import os
import time
import random

# ── PyTorch DLL fix (must come before numpy) ─────────────────────────────────
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
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

MODEL_PATH = "1d_cnn_spectral_model.pth"

# ═════════════════════════════════════════════════════════════════════════════
# 1. SETUP & LOAD MODEL
# ═════════════════════════════════════════════════════════════════════════════

# ── Model definition (exact architecture from training) ──────────────────────
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


# ── Load dataset for label encoder ──────────────────────────────────────────
print("Loading dataset & model ...\n")
df = pd.read_csv("../full_spectra_with_labels.csv")

label_col = df.columns[1]      # 'V0' — polymer type
le = LabelEncoder()
y = le.fit_transform(df[label_col].values)
class_names = list(le.classes_)
num_classes = len(class_names)

# Extract features and split with the SAME seed used during training
X = df.iloc[:, 3:].values.astype(np.float32)
y_labels = df[label_col].values
_, X_test, _, y_test = train_test_split(
    X, y_labels, test_size=0.20, random_state=42, stratify=y
)
print(f"  Test set (held-out 20%): {X_test.shape[0]} samples")

# ── Initialize model and load weights ────────────────────────────────────────
model = Microplastic1DCNN(num_classes=num_classes)
state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
model.load_state_dict(state_dict)
model.eval()

print(f"  Model loaded from: {MODEL_PATH}")
print(f"  Classes ({num_classes}): {class_names}\n")

# ═════════════════════════════════════════════════════════════════════════════
# 2. THE "STRESS TEST" — 5 Random Predictions
# ═════════════════════════════════════════════════════════════════════════════
random.seed(None)       # truly random for live demo

header = f"{'Run':>4}  |  {'Actual':>6}  |  {'Predicted':>10}  |  {'Match':>5}  |  {'Time':>10}"
separator = "-" * len(header)

print("=" * len(header))
print("  LIVE INFERENCE — STRESS TEST (5 Random Samples from Held-Out 20%)")
print("=" * len(header))
print(header)
print(separator)

last_features = None
last_actual   = None
last_predicted = None

for run in range(1, 6):
    # Pick a random sample from the held-out test set
    idx = random.randint(0, len(X_test) - 1)

    actual_label = y_test[idx]                # true polymer name
    features     = X_test[idx]                # 256 spectral values (already float32)

    # Build tensor: (1, 1, 256)
    x_tensor = torch.tensor(features).unsqueeze(0).unsqueeze(0)

    # Timed inference
    t_start = time.perf_counter()
    with torch.no_grad():
        output = model(x_tensor)
        _, pred_idx = torch.max(output, 1)
    t_end = time.perf_counter()

    predicted_label = class_names[pred_idx.item()]
    elapsed_ms = (t_end - t_start) * 1000.0
    match_str = "YES" if predicted_label == actual_label else "NO"

    print(f"{run:>4}  |  {actual_label:>6}  |  {predicted_label:>10}  |  "
          f"{match_str:>5}  |  {elapsed_ms:>8.2f}ms")

    # Remember last sample for the visual aid
    last_features  = features
    last_actual    = actual_label
    last_predicted = predicted_label

print(separator)
print()

# ═════════════════════════════════════════════════════════════════════════════
# 3. THE VISUAL AID — Spectral Curve of Last Sample
# ═════════════════════════════════════════════════════════════════════════════
print("Opening spectral plot for the panel ...")

# Try to load actual wavelengths; fall back to channel indices
try:
    import rdata
    wl_data = rdata.read_rda("../wavelength.rda")
    wavelengths = np.array(list(wl_data.values())[0]).flatten().astype(float)
    x_label = "Wavelength (nm)"
except Exception:
    wavelengths = np.arange(256)
    x_label = "Channel Index"

match_color = "#27AE60" if last_predicted == last_actual else "#E74C3C"
match_word  = "CORRECT" if last_predicted == last_actual else "MISMATCH"

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(wavelengths, last_features, color="#0077B6", linewidth=1.3)
ax.set_xlabel(x_label, fontsize=12)
ax.set_ylabel("Reflectance", fontsize=12)
ax.set_title(
    f"Live Prediction \u2014 Actual: {last_actual}  |  "
    f"Predicted: {last_predicted}  [{match_word}]",
    fontsize=14, fontweight="bold", color=match_color
)
ax.grid(True, linestyle="--", alpha=0.5)
fig.tight_layout()
plt.show()
