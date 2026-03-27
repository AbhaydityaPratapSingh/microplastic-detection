"""
01_prepare_image_data.py — Crop Bounding Boxes from Pascal VOC XML
===================================================================
Parses both imaging/ and imaging2/ datasets, crops each bounding box,
removes PVC class, organizes crops into class folders for ResNet training.

Output structure:
  image_dataset/
    train/ PE/ PP/ PS/ PET/
    val/   PE/ PP/ PS/ PET/
    test/  PE/ PP/ PS/ PET/
"""

import os
import xml.etree.ElementTree as ET
from PIL import Image
from collections import defaultdict

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)

# ── Config ───────────────────────────────────────────────────────────────────
DATASETS = [
    os.path.join(PROJECT_DIR, "../imaging"),
    os.path.join(PROJECT_DIR, "../imaging2"),
]
OUTPUT_DIR = os.path.join(PROJECT_DIR, "../image_dataset")
SPLITS = ["train", "valid", "test"]
SPLIT_MAP = {"valid": "val"}   # rename valid -> val in output
KEEP_CLASSES = {"PE", "PP", "PS", "PET"}
SKIP_CLASSES = {"PVC"}
CROP_SIZE = 64  # resize all crops to 64x64
MIN_CROP_PX = 5  # skip tiny crops smaller than 5px in any dimension


def parse_voc_xml(xml_path):
    """Parse a Pascal VOC XML file and return list of (class_name, bbox) tuples."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)
        objects.append((name, (xmin, ymin, xmax, ymax)))
    return objects


def main():
    print("=" * 60)
    print("  PREPARE IMAGE DATA — Crop Bounding Boxes")
    print("=" * 60)

    # Create output directories
    for split in ["train", "val", "test"]:
        for cls in KEEP_CLASSES:
            os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

    stats = defaultdict(lambda: defaultdict(int))  # stats[split][class] = count
    skipped_pvc = 0
    skipped_tiny = 0
    crop_id = 0

    for dataset_dir in DATASETS:
        dataset_name = os.path.basename(dataset_dir)
        print(f"\nProcessing: {dataset_name}/")

        for split in SPLITS:
            split_dir = os.path.join(dataset_dir, split)
            if not os.path.isdir(split_dir):
                print(f"  [SKIP] {split}/ not found")
                continue

            out_split = SPLIT_MAP.get(split, split)  # valid -> val

            # Find all XML files
            xml_files = [f for f in os.listdir(split_dir) if f.endswith(".xml")]
            print(f"  {split}/: {len(xml_files)} annotation files")

            for xml_file in xml_files:
                xml_path = os.path.join(split_dir, xml_file)
                img_file = xml_file.replace(".xml", ".jpg")
                img_path = os.path.join(split_dir, img_file)

                if not os.path.exists(img_path):
                    continue

                try:
                    img = Image.open(img_path).convert("RGB")
                except Exception:
                    continue

                objects = parse_voc_xml(xml_path)

                for cls_name, (xmin, ymin, xmax, ymax) in objects:
                    # Skip PVC
                    if cls_name in SKIP_CLASSES:
                        skipped_pvc += 1
                        continue

                    # Skip unknown classes
                    if cls_name not in KEEP_CLASSES:
                        continue

                    # Skip tiny crops
                    w = xmax - xmin
                    h = ymax - ymin
                    if w < MIN_CROP_PX or h < MIN_CROP_PX:
                        skipped_tiny += 1
                        continue

                    # Crop and resize
                    crop = img.crop((xmin, ymin, xmax, ymax))
                    crop = crop.resize((CROP_SIZE, CROP_SIZE), Image.LANCZOS)

                    # Save
                    crop_id += 1
                    out_name = f"{cls_name}_{crop_id:05d}.jpg"
                    out_path = os.path.join(OUTPUT_DIR, out_split, cls_name, out_name)
                    crop.save(out_path, quality=95)

                    stats[out_split][cls_name] += 1

    # ── Print summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  DATASET SUMMARY")
    print("=" * 60)

    total = 0
    for split in ["train", "val", "test"]:
        split_total = sum(stats[split].values())
        total += split_total
        print(f"\n  {split}/  ({split_total} crops)")
        for cls in sorted(KEEP_CLASSES):
            count = stats[split].get(cls, 0)
            print(f"    {cls:>4s}: {count:>5d}")

    print(f"\n  Total crops     : {total}")
    print(f"  Skipped (PVC)   : {skipped_pvc}")
    print(f"  Skipped (tiny)  : {skipped_tiny}")
    print(f"  Output folder   : {OUTPUT_DIR}")
    print("=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
