"""
STEP 8 — Build a CUI (concept) vocabulary from TRAINING data only.
"""

import csv
import json
import os
from collections import Counter

METADATA_PATH = "../../data/raw/metadata.csv"
TRAIN_IDS_PATH = "../../data/splits/train_ids.csv"
CUI_VOCAB_OUTPUT_PATH = "../../data/processed/cui_vocab.json"

def load_train_cuis():
    with open(TRAIN_IDS_PATH, "r", encoding="utf-8") as f:
        train_ids = set(row["image_id"] for row in csv.DictReader(f))

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = {row["image_id"]: row for row in csv.DictReader(f)}

    all_cui_lists = []
    for tid in train_ids:
        cui_str = metadata[tid]["cui"]
        cui_list = cui_str.split("|") if cui_str else []
        all_cui_lists.append((tid, cui_list))

    return all_cui_lists

def build_cui_vocab(cui_lists):
    counter = Counter()
    for _tid, cuis in cui_lists:
        counter.update(cuis)

    # No special tokens needed here (unlike word vocab) — this is a
    # multi-label classification target space, not a sequence to decode.
    vocab = {cui: idx for idx, (cui, _count) in enumerate(counter.most_common())}
    return vocab, counter

if __name__ == "__main__":
    cui_lists = load_train_cuis()

    print(f"Loaded CUIs from {len(cui_lists)} TRAINING images:\n")
    for tid, cuis in cui_lists:
        print(f"  {tid}: {cuis}")

    vocab, counter = build_cui_vocab(cui_lists)

    print(f"\nTotal unique CUIs in training set: {len(vocab)}")
    print(f"CUI frequency: {counter.most_common()}")

    os.makedirs("../../data/processed", exist_ok=True)
    with open(CUI_VOCAB_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)

    print(f"\nSaved CUI vocabulary to {CUI_VOCAB_OUTPUT_PATH}")