"""
STEP 6 — Text preprocessing: clean and tokenize radiology report captions
into a consistent word-level format.
"""

import re

def clean_caption(text):
    """Lowercase, normalize whitespace, and separate punctuation from words."""
    text = text.lower().strip()
    # Add spaces around common punctuation so it tokenizes as separate tokens
    text = re.sub(r"([.,!?;:()\[\]])", r" \1 ", text)
    # Collapse multiple spaces into one
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text):
    """Split cleaned text into a list of word/punctuation tokens."""
    cleaned = clean_caption(text)
    return cleaned.split(" ")

if __name__ == "__main__":
    import csv

    with open("data/raw/metadata.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Testing text preprocessing on all {len(rows)} captions:\n")
    for row in rows[:3]:  # show first 3 as a sample
        image_id = row["image_id"]
        raw_caption = row["caption"]
        tokens = tokenize(raw_caption)

        print(f"[{image_id}]")
        print(f"  RAW:    {raw_caption}")
        print(f"  TOKENS: {tokens}")
        print(f"  COUNT:  {len(tokens)} tokens\n")

    # Sanity stats across all 14
    all_lengths = [len(tokenize(row["caption"])) for row in rows]
    print(f"Token count across all {len(rows)} captions: min={min(all_lengths)}, "
          f"max={max(all_lengths)}, avg={sum(all_lengths)/len(all_lengths):.1f}")