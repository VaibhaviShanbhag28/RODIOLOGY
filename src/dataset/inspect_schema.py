"""
STEP 2 — Connect to ROCOv2 and print the real schema of one record.
No data is saved yet. This is inspection only.
"""

from datasets import load_dataset

# ROCOv2 is published on Hugging Face as eltorio/ROCOv2-radiology.
# We use streaming=True so nothing is downloaded except the one record we ask for.
DATASET_NAME = "eltorio/ROCOv2-radiology"

def main():
    print(f"Connecting to {DATASET_NAME} (streaming mode)...")
    ds = load_dataset(DATASET_NAME, split="train", streaming=True)

    print("\nFetching first record...")
    first_record = next(iter(ds))

    print("\n=== RAW RECORD KEYS ===")
    for key in first_record.keys():
        value = first_record[key]
        value_type = type(value).__name__
        if hasattr(value, "size"):  # PIL Image has .size
            print(f"  '{key}': type={value_type}, size={value.size}")
        else:
            preview = str(value)[:150]
            print(f"  '{key}': type={value_type}, value_preview={preview!r}")

    print("\n=== FULL FEATURES SCHEMA (from dataset metadata) ===")
    print(ds.features)

if __name__ == "__main__":
    main()