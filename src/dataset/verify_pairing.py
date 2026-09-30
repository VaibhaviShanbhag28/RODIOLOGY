"""
STEP 4 — Verify that every image file, caption, and CUI list correctly
correspond to the same image_id, and that all split files reference
real, existing images with no overlap.
"""

import csv
import os
from PIL import Image

METADATA_PATH = "data/raw/metadata.csv"
SPLITS_DIR = "data/splits"

def load_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["image_id"]: row for row in reader}

def load_split_ids(split_name):
    path = os.path.join(SPLITS_DIR, f"{split_name}_ids.csv")
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row["image_id"] for row in reader]

def main():
    metadata = load_metadata()
    print(f"Loaded metadata for {len(metadata)} images.\n")

    all_split_ids = {}
    for split_name in ["train", "val", "test"]:
        ids = load_split_ids(split_name)
        all_split_ids[split_name] = ids
        print(f"{split_name}: {len(ids)} ids -> {ids}")

    print("\n=== Checking each image loads and matches metadata ===")
    for split_name, ids in all_split_ids.items():
        for image_id in ids:
            assert image_id in metadata, f"MISSING metadata for {image_id}"
            row = metadata[image_id]

            img_path = row["image_path"]
            assert os.path.exists(img_path), f"MISSING image file: {img_path}"

            img = Image.open(img_path)
            img.verify()  # raises if corrupted

            caption = row["caption"]
            cui = row["cui"]
            assert len(caption.strip()) > 0, f"EMPTY caption for {image_id}"

            print(f"  OK [{split_name}] {image_id}: caption_len={len(caption)}, cui={cui}")

    print("\n=== Checking no image_id appears in more than one split ===")
    train_set = set(all_split_ids["train"])
    val_set = set(all_split_ids["val"])
    test_set = set(all_split_ids["test"])
    assert train_set.isdisjoint(val_set), "OVERLAP between train and val!"
    assert train_set.isdisjoint(test_set), "OVERLAP between train and test!"
    assert val_set.isdisjoint(test_set), "OVERLAP between val and test!"
    print("  OK: no overlap.")

    print("\nShowing one full example (first train image):")
    example_id = all_split_ids["train"][0]
    example = metadata[example_id]
    print(f"  image_id: {example['image_id']}")
    print(f"  image_path: {example['image_path']}")
    print(f"  caption: {example['caption']}")
    print(f"  cui: {example['cui']}")

    print("\nALL CHECKS PASSED.")

if __name__ == "__main__":
    main()