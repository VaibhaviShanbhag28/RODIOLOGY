"""
STEP 7 — Build a word-level vocabulary from TRAINING captions only.
"""

import csv
import json
from collections import Counter
from text_preprocessing import tokenize

METADATA_PATH = "../../data/raw/metadata.csv"
TRAIN_IDS_PATH = "../../data/splits/train_ids.csv"
VOCAB_OUTPUT_PATH = "../../data/processed/vocab.json"

SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]

def load_train_captions():
    with open(TRAIN_IDS_PATH, "r", encoding="utf-8") as f:
        train_ids = set(row["image_id"] for row in csv.DictReader(f))

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = {row["image_id"]: row for row in csv.DictReader(f)}

    return [metadata[tid]["caption"] for tid in train_ids]

def build_vocab(captions):
    counter = Counter()
    for caption in captions:
        tokens = tokenize(caption)
        counter.update(tokens)

    # Special tokens always come first, at fixed indices
    vocab = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
    next_idx = len(SPECIAL_TOKENS)
    for word, _count in counter.most_common():
        if word not in vocab:
            vocab[word] = next_idx
            next_idx += 1

    return vocab, counter

if __name__ == "__main__":
    import os

    captions = load_train_captions()
    print(f"Building vocabulary from {len(captions)} TRAINING captions only.\n")

    vocab, counter = build_vocab(captions)

    print(f"Vocabulary size (including 4 special tokens): {len(vocab)}")
    print(f"Most common 10 words: {counter.most_common(10)}\n")

    os.makedirs("../../data/processed", exist_ok=True)
    with open(VOCAB_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)

    print(f"Saved vocabulary to {VOCAB_OUTPUT_PATH}")
    print(f"\nSpecial token indices: {[(t, vocab[t]) for t in SPECIAL_TOKENS]}")