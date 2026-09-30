"""
STEP 11 — CUI Embedding: convert predicted top-k CUI indices into
learned embedding vectors (not BioBERT — see design rationale in Section B.7).
"""

import torch
import torch.nn as nn

class ConceptEmbedding(nn.Module):
    def __init__(self, num_concepts=8, embed_dim=256, top_k=3):
        super().__init__()
        self.top_k = top_k
        self.embedding = nn.Embedding(num_concepts, embed_dim)

    def forward(self, concept_logits):
        """
        Input:  concept_logits of shape [batch, num_concepts] (raw logits from ConceptHead)
        Output: concept_embeddings of shape [batch, top_k, embed_dim]

        IMPORTANT (Rule 16/17): this only ever uses PREDICTED concepts
        (from concept_logits), never ground-truth CUIs. This is what makes
        it safe to call at inference time on unseen images.
        """
        probs = torch.sigmoid(concept_logits)  # [batch, num_concepts]
        top_k_probs, top_k_indices = torch.topk(probs, k=self.top_k, dim=-1)  # [batch, top_k]
        concept_embeddings = self.embedding(top_k_indices)  # [batch, top_k, embed_dim]
        return concept_embeddings, top_k_indices, top_k_probs

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
    from concept_head import ConceptHead

    with open(PROJECT_ROOT / "data" / "processed" / "cui_vocab.json", "r", encoding="utf-8") as f:
        cui_vocab = json.load(f)
    num_concepts = len(cui_vocab)
    idx_to_cui = {idx: cui for cui, idx in cui_vocab.items()}

    train_ids_path = PROJECT_ROOT / "data" / "splits" / "train_ids.csv"
    metadata_path = PROJECT_ROOT / "data" / "raw" / "metadata.csv"
    with open(train_ids_path, "r", encoding="utf-8") as f:
        first_id = next(csv.DictReader(f))["image_id"]
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = {row["image_id"]: row for row in csv.DictReader(f)}

    image_path = PROJECT_ROOT / metadata[first_id]["image_path"]
    tensor = preprocess_image(str(image_path)).unsqueeze(0)

    visual_encoder = VisualEncoder()
    concept_head = ConceptHead(visual_dim=2048, hidden_dim=512, num_concepts=num_concepts)
    concept_embed = ConceptEmbedding(num_concepts=num_concepts, embed_dim=256, top_k=3)

    visual_encoder.eval(); concept_head.eval(); concept_embed.eval()
    with torch.no_grad():
        visual_features = visual_encoder(tensor)
        concept_logits = concept_head(visual_features)
        embeddings, top_k_indices, top_k_probs = concept_embed(concept_logits)

    print(f"Concept embeddings shape: {embeddings.shape}")
    print(f"Expected: [1, 3, 256]")
    assert embeddings.shape == (1, 3, 256), "SHAPE MISMATCH!"

    print(f"\nTop-3 predicted CUI indices: {top_k_indices}")
    print(f"Top-3 predicted probabilities: {top_k_probs}")
    predicted_cuis = [idx_to_cui[idx.item()] for idx in top_k_indices[0]]
    print(f"Top-3 predicted CUIs (decoded): {predicted_cuis}")
    print(f"\nGround-truth CUI for this image: {metadata[first_id]['cui']}")
    print("(Predictions are meaningless pre-training — shape/wiring check only.)")

    print("\nShape check PASSED.")