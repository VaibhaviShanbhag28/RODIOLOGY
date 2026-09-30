import torch
import torchvision
import numpy as np
import pandas as pd
import PIL
import yaml
import nltk
import tqdm

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("numpy:", np.__version__)
print("pandas:", pd.__version__)
print("pillow:", PIL.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Device that will be used:", "cuda" if torch.cuda.is_available() else "cpu")
print("\nAll imports successful.")