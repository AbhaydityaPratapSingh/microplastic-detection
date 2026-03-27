"""
02_train_1d_cnn.py — Phase 2: Microplastic 1D-CNN Classifier
=============================================================
Trains a 1D Convolutional Neural Network on 256-channel hyperspectral
reflectance data to classify polymer types (PE, PP, PS, PET).
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
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ── Hyperparameters ──────────────────────────────────────────────────────────
BATCH_SIZE    = 32
EPOCHS        = 25
LEARNING_RATE = 0.001
RANDOM_STATE  = 42
MODEL_PATH    = "1d_cnn_spectral_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ═════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & PREPROCESSING
# ═════════════════════════════════════════════════════════════════════════════
print("\n[1/4] Loading data ...")
df = pd.read_csv("../full_spectra_with_labels.csv")
print(f"  CSV shape: {df.shape}")

# ── Target labels (column index 1 = 'V0', the polymer type) ─────────────────
label_col = df.columns[1]  # 'V0'
le = LabelEncoder()
y = le.fit_transform(df[label_col].values)
num_classes = len(le.classes_)
print(f"  Target column: '{label_col}'")
print(f"  Discovered classes ({num_classes}): {list(le.classes_)}")
print(f"  Encoded labels:  {dict(zip(le.classes_, le.transform(le.classes_)))}")

# ── Features (columns from index 3 onward = 256 spectral channels) ──────────
X = df.iloc[:, 3:].values.astype(np.float32)
print(f"  Feature matrix X: {X.shape}  (samples x wavelength channels)")

# ── Train / validation split (80 / 20) ──────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
print(f"  Train: {X_train.shape[0]}   Val: {X_val.shape[0]}")

# ── Convert to PyTorch tensors — reshape to (Batch, 1, 256) ─────────────────
X_train_t = torch.tensor(X_train).unsqueeze(1)     # (N, 1, 256)
X_val_t   = torch.tensor(X_val).unsqueeze(1)
y_train_t = torch.tensor(y_train, dtype=torch.long)
y_val_t   = torch.tensor(y_val,   dtype=torch.long)

print(f"  Tensor shapes -> X_train: {tuple(X_train_t.shape)}, "
      f"X_val: {tuple(X_val_t.shape)}")

# ── DataLoaders ──────────────────────────────────────────────────────────────
train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=BATCH_SIZE, shuffle=True
)
val_loader = DataLoader(
    TensorDataset(X_val_t, y_val_t),
    batch_size=BATCH_SIZE, shuffle=False
)

# ═════════════════════════════════════════════════════════════════════════════
# 2. MODEL ARCHITECTURE
# ═════════════════════════════════════════════════════════════════════════════
class Microplastic1DCNN(nn.Module):
    """
    1D-CNN for hyperspectral polymer classification.

    Input:  (Batch, 1, 256)
    Conv1 -> ReLU -> MaxPool  =>  (Batch, 16, 128)
    Conv2 -> ReLU -> MaxPool  =>  (Batch, 32,  64)
    Flatten                   =>  (Batch, 32*64 = 2048)
    FC(128) -> ReLU -> Dropout(0.5) -> FC(num_classes)
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


model = Microplastic1DCNN(num_classes=num_classes).to(device)
print(f"\n[2/4] Model architecture:")
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"  Total parameters: {total_params:,}")

# ═════════════════════════════════════════════════════════════════════════════
# 3. TRAINING LOOP
# ═════════════════════════════════════════════════════════════════════════════
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"\n[3/4] Training for {EPOCHS} epochs  (batch_size={BATCH_SIZE}, "
      f"lr={LEARNING_RATE}) ...")
print("-" * 58)
print(f"{'Epoch':>6}  {'Train Loss':>12}  {'Val Accuracy':>14}")
print("-" * 58)

for epoch in range(1, EPOCHS + 1):
    # ── Training phase ───────────────────────────────────────────────────
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X_batch.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)

    # ── Validation phase ─────────────────────────────────────────────────
    model.eval()
    correct = 0
    total   = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, preds = torch.max(outputs, 1)
            correct += (preds == y_batch).sum().item()
            total   += y_batch.size(0)

    val_acc = 100.0 * correct / total
    print(f"{epoch:>6}  {epoch_loss:>12.4f}  {val_acc:>13.2f}%")

print("-" * 58)

# ═════════════════════════════════════════════════════════════════════════════
# 4. SAVE MODEL
# ═════════════════════════════════════════════════════════════════════════════
torch.save(model.state_dict(), MODEL_PATH)
print(f"\n[4/4] Model weights saved -> {MODEL_PATH}")
print("Done!")
