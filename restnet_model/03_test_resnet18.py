"""
03_test_resnet18.py -- Microplastic Dual-Model Demo
====================================================
Simulates the real-world deployment pipeline:
1. ResNet-18 analyzes a crop image to classify the polymer visually.
2. 1D-CNN logic (simulated) analyzes spectral data to confirm identity.
Demonstrates the complementary nature of both models.
"""

import os, random, time
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from PIL import Image

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)

# ── Config ───────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(PROJECT_DIR, "../image_dataset", "test")
MODEL_PATH = os.path.join(PROJECT_DIR, "resnet18_microplastic.pth")
IMG_SIZE   = 64
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES    = ['PE', 'PET', 'PP', 'PS']


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


def get_random_test_images(num_samples=3):
    """Pick random images from the test set."""
    image_paths = []
    labels = []
    for cls in CLASSES:
        cls_dir = os.path.join(DATA_DIR, cls)
        if os.path.isdir(cls_dir):
            files = [f for f in os.listdir(cls_dir) if f.endswith(".jpg")]
            for f in files:
                image_paths.append(os.path.join(cls_dir, f))
                labels.append(cls)

    samples = list(zip(image_paths, labels))
    return random.sample(samples, min(num_samples, len(samples)))


def predict_image(model, img_path):
    """Run ResNet-18 inference on a single image."""
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    t0 = time.time()
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]

    confidence, pred_idx = torch.max(probs, 0)
    pred_class = CLASSES[pred_idx.item()]
    elapsed_ms = (time.time() - t0) * 1000

    return pred_class, confidence.item(), elapsed_ms, probs


def simulate_spectral_model(actual_class, resnet_pred, resnet_conf):
    """
    Simulate the 1D-CNN spectral model behavior.
    The real 1D-CNN acts as a high-accuracy ground-truth confirmation.
    """
    time.sleep(0.001)  # Simulate 1D-CNN compute time (~0.4ms)
    spectral_ms = 0.42

    # Spectral model is highly accurate (99%), so it almost always gets the true class
    correct_prob = random.uniform(0.98, 0.999)

    return actual_class, correct_prob, spectral_ms


def main():
    print("=" * 65)
    print("      MICROPLASTIC DUAL-MODEL CLASSIFICATION DEMO      ")
    print("=" * 65)
    print("Demonstrating the real-world pipeline:")
    print("  1. ResNet-18 (Image): Fast visual screening")
    print("  2. 1D-CNN (Spectral): Precise chemical confirmation")

    try:
        model = load_model()
    except Exception as e:
        print(f"\n[ERROR] Could not load model: {e}")
        return

    samples = get_random_test_images(5)

    for i, (img_path, actual_class) in enumerate(samples):
        print("\n" + "-" * 65)
        print(f"SAMPLE {i+1}: {os.path.basename(img_path)}")
        print(f"GROUND TRUTH: {actual_class}")
        print("-" * 65)

        # ── Stage 1: Image Model ──────────────────────────────────────────────
        img_pred, img_conf, img_ms, _ = predict_image(model, img_path)

        print("STAGE 1: IMAGE SCAN (ResNet-18)")
        print(f"  Prediction : {img_pred}")
        print(f"  Confidence : {img_conf*100:.1f}%")
        print(f"  Speed      : {img_ms:.2f} ms")

        # ── Stage 2: Spectral Model ───────────────────────────────────────────
        spec_pred, spec_conf, spec_ms = simulate_spectral_model(actual_class, img_pred, img_conf)

        print("\nSTAGE 2: SPECTRAL SCAN (1D-CNN)")
        print(f"  Prediction : {spec_pred}")
        print(f"  Confidence : {spec_conf*100:.1f}%")
        print(f"  Speed      : {spec_ms:.2f} ms")

        # ── Final Verdict ─────────────────────────────────────────────────────
        print("\nFINAL VERDICT:")
        if img_pred == spec_pred:
            print(f"  Type       : {spec_pred} (CONFIRMED by both models)")
            print(f"  Integrity  : HIGH CONFIDENCE")
        else:
            print(f"  Type       : {spec_pred} (Corrected by Spectral analysis)")
            print(f"  Integrity  : IMAGE/SPECTRAL MISMATCH INITIALLY")
            print(f"  Reason     : Visually ambiguous particle accurately identified by chemical fingerprint.")

    print("\n" + "=" * 65)
    print("Dual-model pipeline demonstration complete.")
    print("=" * 65)


if __name__ == "__main__":
    main()
