"""
STEP 3 — Pull a reproducible 14-image sample from ROCOv2 (streaming),
save images + metadata locally, and write train/val/test ID files.
"""

import os
import csv
from datasets import load_dataset

DATASET_NAME = "eltorio/ROCOv2-radiology"
SEED = 42
N_TRAIN = 10
N_VAL = 2
N_TEST = 2
N_TOTAL = N_TRAIN + N_VAL + N_TEST

RAW_IMG_DIR = "data/raw/images"
SPLITS_DIR = "data/splits"
METADATA_PATH = "data/raw/metadata.csv"

def main():
    os.makedirs(RAW_IMG_DIR, exist_ok=True)
    os.makedirs(SPLITS_DIR, exist_ok=True)

    print(f"Connecting to {DATASET_NAME} (streaming, seed={SEED})...")
    ds = load_dataset(DATASET_NAME, split="train", streaming=True)
    ds = ds.shuffle(seed=SEED, buffer_size=1000)

    print(f"Pulling {N_TOTAL} records...")
    records = []
    for i, record in enumerate(ds):
        if i >= N_TOTAL:
            break
        records.append(record)
        print(f"  [{i+1}/{N_TOTAL}] {record['image_id']}")

    if len(records) < N_TOTAL:
        raise RuntimeError(f"Only got {len(records)} records, expected {N_TOTAL}")

    print("\nSaving images and metadata...")
    metadata_rows = []
    for record in records:
        image_id = record["image_id"]
        image = record["image"]
        caption = record["caption"]
        cui = record["cui"]

        img_path = os.path.join(RAW_IMG_DIR, f"{image_id}.png")
        image.save(img_path)

        metadata_rows.append({
            "image_id": image_id,
            "image_path": img_path,
            "caption": caption,
            "cui": "|".join(cui),  # pipe-separated since CUI lists vary in length
        })

    with open(METADATA_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "image_path", "caption", "cui"])
        writer.writeheader()
        writer.writerows(metadata_rows)

    print(f"Saved metadata to {METADATA_PATH}")

    # Deterministic split: first N_TRAIN go to train, next N_VAL to val, rest to test.
    # The shuffle above (fixed seed) already randomized order, so this slicing is
    # effectively a random split, and is 100% reproducible on every run.
    train_ids = [r["image_id"] for r in records[:N_TRAIN]]
    val_ids = [r["image_id"] for r in records[N_TRAIN:N_TRAIN + N_VAL]]
    test_ids = [r["image_id"] for r in records[N_TRAIN + N_VAL:]]

    def write_ids(path, ids):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image_id"])
            for _id in ids:
                writer.writerow([_id])

    write_ids(os.path.join(SPLITS_DIR, "train_ids.csv"), train_ids)
    write_ids(os.path.join(SPLITS_DIR, "val_ids.csv"), val_ids)
    write_ids(os.path.join(SPLITS_DIR, "test_ids.csv"), test_ids)

    print(f"\nSplit written:")
    print(f"  train: {len(train_ids)} -> data/splits/train_ids.csv")
    print(f"  val:   {len(val_ids)} -> data/splits/val_ids.csv")
    print(f"  test:  {len(test_ids)} -> data/splits/test_ids.csv")

    # Sanity check: no overlap between splits (Rule: no image in multiple splits)
    assert set(train_ids).isdisjoint(val_ids)
    assert set(train_ids).isdisjoint(test_ids)
    assert set(val_ids).isdisjoint(test_ids)
    print("\nNo overlap between splits confirmed.")

if __name__ == "__main__":
    main()