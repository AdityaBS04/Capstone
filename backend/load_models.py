"""
Helper script to load the actual model weights when they are available.
Update the paths in app.py to use this script.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class BinaryClassifier(nn.Module):
    """Binary classifier: Normal vs Diseased"""
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.model = models.resnet18(pretrained=False)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class DiseaseClassifier(nn.Module):
    """Disease classifier: Esophagitis vs Polyps vs UC"""
    def __init__(self, num_classes=3):
        super(DiseaseClassifier, self).__init__()
        self.model = models.resnet18(pretrained=False)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(num_classes, num_classes)
        )

    def forward(self, x):
        return self.model(x)


def load_models(binary_path, disease_path, device='cpu'):
    """
    Load the trained models from disk.

    Args:
        binary_path: Path to binary classification model (.pth file)
        disease_path: Path to disease classification model (.pth file)
        device: Device to load models on ('cpu' or 'cuda')

    Returns:
        binary_model, disease_model
    """
    # Initialize models
    binary_model = BinaryClassifier().to(device)
    disease_model = DiseaseClassifier(num_classes=3).to(device)

    # Load weights
    binary_model.load_state_dict(torch.load(binary_path, map_location=device))
    disease_model.load_state_dict(torch.load(disease_path, map_location=device))

    # Set to evaluation mode
    binary_model.eval()
    disease_model.eval()

    print(f"✓ Binary model loaded from: {binary_path}")
    print(f"✓ Disease model loaded from: {disease_path}")

    return binary_model, disease_model


if __name__ == '__main__':
    # Test loading
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    binary_path = '../models/image/image_identification.pth'
    disease_path = '../models/image/image_classfication.pth'

    try:
        binary_model, disease_model = load_models(binary_path, disease_path, device)
        print("\n✓ Models loaded successfully!")
    except Exception as e:
        print(f"\n✗ Error loading models: {e}")
        print("\nMake sure the model files exist at:")
        print(f"  - {binary_path}")
        print(f"  - {disease_path}")
