"""
STEP 9 — Visual Encoder: ResNet-50 (ImageNet-pretrained), truncated before
the final pooling/classification layers, producing spatial feature maps.
"""

import torch
import torch.nn as nn
import torchvision.models as models

class VisualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x):
        """
        Input:  x of shape [batch, 3, 224, 224]
        Output: feature map of shape [batch, 49, 2048]
        """
        features = self.backbone(x)  # [batch, 2048, 7, 7]
        batch_size, channels, h, w = features.shape
        features = features.view(batch_size, channels, h * w)  # [batch, 2048, 49]
        features = features.permute(0, 2, 1)  # [batch, 49, 2048]
        return features

if __name__ == "__main__":
    import sys
    import csv
    from pathlib import Path

    # Find project root regardless of where this script is run from
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT / "src" / "preprocessing"))
    from image_preprocessing import preprocess_image

    train_ids_path = PROJECT_ROOT / "data" / "splits" / "train_ids.csv"
    metadata_path = PROJECT_ROOT / "data" / "raw" / "metadata.csv"

    with open(train_ids_path, "r", encoding="utf-8") as f:
        first_id = next(csv.DictReader(f))["image_id"]
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = {row["image_id"]: row for row in csv.DictReader(f)}

    # image_path in metadata.csv is relative to project root, so join it that way
    relative_image_path = metadata[first_id]["image_path"]
    image_path = PROJECT_ROOT / relative_image_path

    print(f"Testing visual encoder on: {image_path}")

    tensor = preprocess_image(str(image_path))
    batch = tensor.unsqueeze(0)
    print(f"Input batch shape: {batch.shape}")

    print("\nLoading ResNet-50 (first run downloads pretrained weights)...")
    encoder = VisualEncoder()
    encoder.eval()

    with torch.no_grad():
        output = encoder(batch)

    print(f"\nOutput feature map shape: {output.shape}")
    print(f"Expected: [1, 49, 2048]")
    assert output.shape == (1, 49, 2048), "SHAPE MISMATCH!"
    print("Shape check PASSED.")