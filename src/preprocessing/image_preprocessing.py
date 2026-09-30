"""
STEP 5 — Image preprocessing: load a raw ROCOv2 image and convert it
to a normalized tensor ready for a CNN (ResNet-50 ImageNet-pretrained).
"""

from PIL import Image
import torchvision.transforms as T

IMAGE_SIZE = 224

# Standard ImageNet normalization statistics — required because we will
# use an ImageNet-pretrained ResNet-50 backbone (Step 9). Using different
# normalization would mismatch what the pretrained weights expect.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_transform():
    """Returns the preprocessing pipeline applied to every image."""
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.Grayscale(num_output_channels=3),  # ROCOv2 images are often grayscale radiographs;
                                              # ResNet-50 expects 3 channels, so we replicate.
        T.ToTensor(),                        # converts to [C, H, W], scales pixel values to [0, 1]
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def preprocess_image(image_path):
    """Load a single image file and return a preprocessed tensor of shape [3, 224, 224]."""
    image = Image.open(image_path).convert("RGB")
    transform = get_transform()
    tensor = transform(image)
    return tensor

if __name__ == "__main__":
    # Quick manual test on one real image from our split
    import csv

    with open("data/splits/train_ids.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        first_id = next(reader)["image_id"]

    with open("data/raw/metadata.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        metadata = {row["image_id"]: row for row in reader}

    image_path = metadata[first_id]["image_path"]
    print(f"Testing preprocessing on: {image_path}")

    tensor = preprocess_image(image_path)
    print(f"Output tensor shape: {tensor.shape}")
    print(f"Output tensor dtype: {tensor.dtype}")
    print(f"Value range: min={tensor.min().item():.3f}, max={tensor.max().item():.3f}")
    print(f"Mean: {tensor.mean().item():.3f}, Std: {tensor.std().item():.3f}")