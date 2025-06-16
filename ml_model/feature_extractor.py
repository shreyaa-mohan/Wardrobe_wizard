# ml_model/feature_extractor.py
import torch
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import numpy as np
from .utils import extract_dominant_colors as extract_dom_colors_util # Alias to avoid name conflict

# --- Image Embedding Model ---
# Load pre-trained ResNet50 model
# Use the recommended weights API
try:
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
except Exception as e: # Fallback for older torchvision or if V2 not available
    print(f"Could not load ResNet50_Weights.IMAGENET1K_V2, trying ResNet50_Weights.IMAGENET1K_V1: {e}")
    try:
        weights = ResNet50_Weights.IMAGENET1K_V1
        model = resnet50(weights=weights)
    except Exception as e2:
        print(f"Could not load ResNet50_Weights.IMAGENET1K_V1 either, using pretrained=True (legacy): {e2}")
        model = resnet50(pretrained=True) # Legacy, if above fails

model.eval() # Set to evaluation mode: disables dropout, batchnorm updates etc.

# Remove the final classification layer (the fully connected layer)
# to get the feature vector before classification.
# For ResNet, this is typically the output of the average pooling layer.
feature_extractor_model = torch.nn.Sequential(*list(model.children())[:-1])

# Define image transformations based on what ResNet50 expects
preprocess = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_image_embedding(image_path):
    """Generates a feature vector (embedding) for a given image."""
    try:
        img = Image.open(image_path).convert('RGB') # Ensure image is RGB
        img_t = preprocess(img)
        batch_t = torch.unsqueeze(img_t, 0) # Create a mini-batch as expected by the model

        with torch.no_grad(): # Important: disable gradient calculations for inference
            embedding = feature_extractor_model(batch_t)
        
        # Flatten the embedding (output of avgpool is typically [1, num_features, 1, 1])
        embedding_flat = torch.flatten(embedding).numpy() # Convert to NumPy array
        return embedding_flat
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return np.zeros(2048) # ResNet50 feature vector size before final FC layer is 2048
    except Exception as e:
        print(f"Error generating embedding for {image_path}: {e}")
        return np.zeros(2048)

def extract_features(image_path):
    """Extracts both dominant colors and image embedding from an image."""
    # Call the aliased function from utils
    dominant_colors = extract_dom_colors_util(image_path, num_colors=3)
    embedding = get_image_embedding(image_path)
    return dominant_colors, embedding