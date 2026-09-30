"""
STEP 10 — Concept Prediction Head: predicts which CUIs are present
from the visual feature map. Multi-label classification (sigmoid, not softmax).
"""

import torch
import torch.nn as nn

class ConceptHead(nn.Module):
    def __init__(self, visual_dim=2048, hidden_dim=512, num_concepts=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)  # global average pool over the 49 spatial positions
        self.mlp = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_concepts),
        )

    def forward(self, visual_features):
        """
        Input:  visual_features of shape [batch, 49, 2048]
        Output: concept_logits of shape [batch, num_concepts]
                 (raw logits — apply sigmoid separately for probabilities,
                  and use BCEWithLogitsLoss for training, which applies
                  sigmoid internally and is more numerically stable)
        """
        x = visual_features.permute(0, 2, 1)  # [batch, 2048, 49]
        x = self.pool(x).squeeze(-1)          # [batch, 2048]
        logits = self.mlp(x)                  # [batch, num_concepts]
        return logits

if __name__ == "__main__":
    import sys
    import json
    import csv
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(PROJECT_ROOT / "src" / "preprocessing"))
    sys.path.insert(0, str(PROJECT_ROOT / "src" / "models"))
    from image_preprocessing import preprocess_image
    from visual_encoder import VisualEncoder

    with open(PROJECT_ROOT / "data" / "processed" / "cui_vocab.json", "r", encoding="utf-8") as f:
        cui_vocab = json.load(f)
    num_concepts = len(cui_vocab)
    print(f"Number of concepts (from cui_vocab.json): {num_concepts}")

    train_ids_path = PROJECT_ROOT / "data" / "splits" / "train_ids.csv"
    metadata_path = PROJECT_ROOT / "data" / "raw" / "metadata.csv"
    with open(train_ids_path, "r", encoding="utf-8") as f:
        first_id = next(csv.DictReader(f))["image_id"]
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = {row["image_id"]: row for row in csv.DictReader(f)}

    image_path = PROJECT_ROOT / metadata[first_id]["image_path"]
    print(f"Testing on: {image_path}")

    tensor = preprocess_image(str(image_path)).unsqueeze(0)

    print("Running visual encoder...")
    visual_encoder = VisualEncoder()
    visual_encoder.eval()
    with torch.no_grad():
        visual_features = visual_encoder(tensor)
    print(f"Visual features shape: {visual_features.shape}")

    print("Running concept head...")
    concept_head = ConceptHead(visual_dim=2048, hidden_dim=512, num_concepts=num_concepts)
    concept_head.eval()
    with torch.no_grad():
        logits = concept_head(visual_features)

    print(f"\nConcept logits shape: {logits.shape}")
    print(f"Expected: [1, {num_concepts}]")
    assert logits.shape == (1, num_concepts), "SHAPE MISMATCH!"

    probs = torch.sigmoid(logits)
    print(f"\nRaw logits: {logits}")
    print(f"Sigmoid probabilities: {probs}")
    print(f"\nGround-truth CUI for this image: {metadata[first_id]['cui']}")
    print("\n(Note: probabilities are meaningless right now — the model hasn't been trained yet. "
          "This step only verifies the shapes and wiring are correct.)")

    print("\nShape check PASSED.")