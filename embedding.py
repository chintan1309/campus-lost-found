"""
embedding.py — Feature extraction using a pre-trained MobileNetV2.

MobileNetV2 is loaded once at module import time (weights downloaded on first run).
Each image is resized to 224×224, preprocessed, and passed through the network
to produce a 1280-dimensional embedding vector (global average pool of the last
convolutional feature map).  No fine-tuning or re-training is required.
"""

import numpy as np
from PIL import Image

# --- lazy imports so TF only loads once ---
_model = None


def _get_model():
    """Load (or return cached) MobileNetV2 feature extractor."""
    global _model
    if _model is None:
        # Import here to avoid slow startup if embedding is not needed immediately
        from keras.applications import MobileNetV2
        _model = MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,       # remove classification head
            pooling='avg',           # global average pooling → 1280-dim vector
            weights='imagenet'       # pre-trained on ImageNet
        )
        _model.trainable = False     # inference only — no gradient tracking needed
    return _model


def extract_embedding(image_path: str) -> list:
    """
    Load an image from *image_path*, preprocess it, and return a 1280-dim
    embedding as a plain Python list (for JSON serialisation).

    Steps:
      1. Open with Pillow and convert to RGB (handles PNG alpha / grayscale).
      2. Resize to 224×224 (MobileNetV2 input resolution).
      3. Scale pixel values to [-1, 1] using MobileNetV2's own preprocess_fn.
      4. Add batch dimension and run forward pass.
      5. Squeeze batch dim and return as list.
    """
    from keras.applications.mobilenet_v2 import preprocess_input

    model = _get_model()

    # Open and prepare the image
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224), Image.LANCZOS)

    # Convert to numpy and scale
    img_array = np.array(img, dtype=np.float32)
    img_array = preprocess_input(img_array)        # scale to [-1, 1]
    img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)

    # Forward pass (inference only)
    embedding = model.predict(img_array, verbose=0)  # shape: (1, 1280)
    return embedding.squeeze().tolist()               # 1280-element list
