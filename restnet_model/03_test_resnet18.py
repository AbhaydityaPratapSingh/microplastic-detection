"""
03_test_resnet18.py -- Microplastic ResNet-18 Demo
===================================================
Runs ResNet-18 inference on 5 random test-set crops and
displays results in the terminal + a matplotlib grid figure.
"""

import os, random, time
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from PIL import Image
import matplotlib.pyplot as plt

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


def main():
    print("=" * 65)
    print("      MICROPLASTIC ResNet-18 CLASSIFICATION DEMO      ")
    print("=" * 65)

    try:
        model = load_model()
    except Exception as e:
        print(f"\n[ERROR] Could not load model: {e}")
        return

    samples = get_random_test_images(5)
    results = []  # (img_path, true_class, pred_class, confidence)

    for i, (img_path, actual_class) in enumerate(samples):
        print("\n" + "-" * 65)
        print(f"SAMPLE {i+1}: {os.path.basename(img_path)}")
        print(f"GROUND TRUTH: {actual_class}")
        print("-" * 65)

        img_pred, img_conf, img_ms, _ = predict_image(model, img_path)

        print(f"  Prediction : {img_pred}")
        print(f"  Confidence : {img_conf*100:.1f}%")
        print(f"  Speed      : {img_ms:.2f} ms")

        results.append((img_path, actual_class, img_pred, img_conf))

    print("\n" + "=" * 65)
    print("ResNet-18 classification demo complete.")
    print("=" * 65)

    # ── Matplotlib Grid Visualization ────────────────────────────────────────
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    for ax, (path, true_cls, pred_cls, conf) in zip(axes, results):
        img = Image.open(path).convert("RGB")
        ax.imshow(img)
        ax.axis("off")
        correct = (true_cls == pred_cls)
        color = "green" if correct else "red"
        ax.set_title(f"True: {true_cls}", fontsize=11, fontweight="bold",
                     color=color, pad=8)
        ax.text(0.5, -0.05, f"Pred: {pred_cls} ({conf*100:.1f}%)",
                transform=ax.transAxes, ha="center", fontsize=10, color=color)
    fig.suptitle("Microplastic ResNet-18 \u2014 5 Random Test Crops",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

