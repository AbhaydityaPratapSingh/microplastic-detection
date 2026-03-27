"""
08_repeated_kfold_cv.py -- Robustness Validation for 1D-CNN
============================================================
DEBUGGING MODE: Runs 5 different random weight initializations
on the exact same data split (Repeat 5, Fold 5: Index 24).
This proves whether the accuracy drop was due to bad data in 
this specific split, or just an unlucky random weight initialization.
"""

import os

# Fix for PyTorch DLL loading on some Windows setups
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
from sklearn.model_selection import RepeatedStratifiedKFold

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)

# ── Config ───────────────────────────────────────────────────────────────────
BATCH_SIZE    = 64     # Increased from original 32 for speed
EPOCHS        = 25     # Restored to original Epochs
LEARNING_RATE = 0.001
N_SPLITS      = 5
N_REPEATS     = 5
RANDOM_STATE  = 42     # Used ONLY for the data split
NUM_RUNS      = 5      # Number of times to train/eval the model

TARGET_INDEX  = 24     # Repeat 5, Fold 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Architecture ─────────────────────────────────────────────────────────────
class Microplastic1DCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 64, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def main():
    print("=" * 65)
    print(f"  DEBUGGING FOLD {TARGET_INDEX} : {NUM_RUNS} RANDOM INITIALIZATIONS")
    print("=" * 65)
    
    # ── 1. Load Data & Exact Same Split ────────────────────────────────────────
    print("Loading spectral data and creating exact data split...")
    df = pd.read_csv("../full_spectra_with_labels.csv")
    label_col = df.columns[1]
    
    le = LabelEncoder()
    y = le.fit_transform(df[label_col].values)
    X = df.iloc[:, 3:].values.astype(np.float32)
    classes = list(le.classes_)
    num_classes = len(classes)
    
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
    
    target_train_idx, target_val_idx = None, None
    for i, (train_idx, val_idx) in enumerate(rskf.split(X, y)):
        if i == TARGET_INDEX:
            target_train_idx, target_val_idx = train_idx, val_idx
            break
            
    if target_train_idx is None:
        print(f"Error: Could not find fold index {TARGET_INDEX}")
        return
        
    X_train, y_train = X[target_train_idx], y[target_train_idx]
    X_val, y_val     = X[target_val_idx], y[target_val_idx]
    
    print(f"Target Fold {TARGET_INDEX} isolated successfully.")
    print(f"Train samples: {len(X_train)} | Val samples: {len(X_val)}\n")

    # ── 2. Create Static DataLoaders ───────────────────────────────────────────
    train_ds = TensorDataset(torch.tensor(X_train).unsqueeze(1), torch.tensor(y_train, dtype=torch.long))
    val_ds   = TensorDataset(torch.tensor(X_val).unsqueeze(1),   torch.tensor(y_val, dtype=torch.long))
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    # ── 3. Run Loop: Train & Eval 5 times ──────────────────────────────────────
    accuracies = []
    
    for run in range(1, NUM_RUNS + 1):
        # We RE-SEED torch specifically here so PyTorch random initialization is different for each loop
        torch.manual_seed(run * 100)
        
        print("-" * 40)
        print(f"RUN {run}/{NUM_RUNS} (Random Seed: {run * 100})")
        print("-" * 40)
        
        # Fresh model initialization
        model = Microplastic1DCNN(num_classes).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # Train
        for epoch in range(EPOCHS):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

        # Evaluate
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = model(X_batch)
                
                # Convert logits to class index
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        # Exact match accuracy
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        acc = np.mean(all_preds == all_labels) * 100
        
        accuracies.append(acc)
        print(f"--> Accuracy on Run {run}: {acc:.2f}%\n")

    # ── 4. Final Aggregated Results ────────────────────────────────────────────
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    
    print("=" * 65)
    print("  FINAL AGGREGATED METRICS")
    print("=" * 65)
    print(f"Data Split      : Index {TARGET_INDEX} (Repeat 5, Fold 5)")
    print(f"Total Runs      : {NUM_RUNS} random initializations")
    print(f"Accuracy Mean   : {mean_acc:.3f}%")
    print(f"Accuracy StdDev : {std_acc:.3f}%")
    print("=" * 65)
    
    print("\nIndividual run accuracies:")
    for i, a in enumerate(accuracies):
        print(f"  Run {i+1}: {a:.2f}%")


if __name__ == "__main__":
    main()
