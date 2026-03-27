"""
04_view_model.py — Visualize / Inspect the trained 1D-CNN
=========================================================
Rebuilds the Microplastic1DCNN, loads saved weights, prints
the architecture and first-layer weights, and exports the
model to ONNX format for visual graphing in tools like Netron.
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
from sklearn.preprocessing import LabelEncoder

MODEL_PATH = "1d_cnn_spectral_model.pth"
ONNX_PATH  = "1d_cnn_model.onnx"

# ═════════════════════════════════════════════════════════════════════════════
# 1. DETERMINE CLASS COUNT
# ═════════════════════════════════════════════════════════════════════════════
print("[1/4] Determining class count from dataset ...")
df = pd.read_csv("../full_spectra_with_labels.csv")
label_col = df.columns[1]   # 'V0' — polymer type

le = LabelEncoder()
le.fit(df[label_col].values)
num_classes = len(le.classes_)
print(f"  Target column : '{label_col}'")
print(f"  Classes ({num_classes}): {list(le.classes_)}")

# ═════════════════════════════════════════════════════════════════════════════
# 2. REBUILD ARCHITECTURE & LOAD WEIGHTS
# ═════════════════════════════════════════════════════════════════════════════
class Microplastic1DCNN(nn.Module):
    """
    1D-CNN for hyperspectral polymer classification.

    Input:  (Batch, 1, 256)
    Conv1 -> ReLU -> MaxPool  =>  (Batch, 16, 128)
    Conv2 -> ReLU -> MaxPool  =>  (Batch, 32,  64)
    Flatten                   =>  (Batch, 2048)
    FC(128) -> ReLU -> Dropout(0.5) -> FC(num_classes)
    """
    def __init__(self, num_classes: int):
        super().__init__()
        # Named individually so we can access conv1 weights directly
        self.conv1 = nn.Conv1d(in_channels=1,  out_channels=16,
                               kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32,
                               kernel_size=5, padding=2)
        self.relu    = nn.ReLU()
        self.pool    = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1     = nn.Linear(32 * 64, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2     = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


print("\n[2/4] Loading model weights ...")
model = Microplastic1DCNN(num_classes=num_classes)

# The training script used nn.Sequential with keys like 'features.0.weight'.
# Remap those keys to our named-layer architecture.
state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)

KEY_MAP = {
    "features.0.weight": "conv1.weight",
    "features.0.bias":   "conv1.bias",
    "features.3.weight": "conv2.weight",
    "features.3.bias":   "conv2.bias",
    "classifier.1.weight": "fc1.weight",
    "classifier.1.bias":   "fc1.bias",
    "classifier.4.weight": "fc2.weight",
    "classifier.4.bias":   "fc2.bias",
}

remapped = {}
for old_key, tensor in state_dict.items():
    new_key = KEY_MAP.get(old_key, old_key)
    remapped[new_key] = tensor

model.load_state_dict(remapped)
model.eval()
print("  Weights loaded and model set to eval() mode.")

# ═════════════════════════════════════════════════════════════════════════════
# 3. CONSOLE OUTPUTS
# ═════════════════════════════════════════════════════════════════════════════
print("\n[3/4] Model architecture:")
print(model)

print("\n--- Learned weights of conv1 (first convolutional layer) ---")
print(f"  Shape: {model.conv1.weight.data.shape}  "
      f"(out_channels, in_channels, kernel_size)")
print(model.conv1.weight.data)

# ═════════════════════════════════════════════════════════════════════════════
# 4. EXPORT TO ONNX
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n[4/4] Exporting model to ONNX -> {ONNX_PATH} ...")
dummy_input = torch.randn(1, 1, 256)

torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    input_names=["spectral_input"],
    output_names=["class_logits"],
    dynamic_axes={
        "spectral_input": {0: "batch_size"},
        "class_logits":   {0: "batch_size"},
    },
    opset_version=11,
)

print(f"  ONNX export complete: {ONNX_PATH}")
print("\nTip: Open the .onnx file in https://netron.app to see the full graph.")
print("Done!")
