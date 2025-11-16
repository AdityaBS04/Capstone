import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MicrobiomeClassifier(nn.Module):
    """Two-stage microbiome disease classifier"""

    def __init__(self, input_size=500, num_diseases=10):
        super(MicrobiomeClassifier, self).__init__()

        # Shared feature extractor
        self.feature_layers = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Stage 1: Binary classification (Healthy vs Diseased)
        self.stage1_head = nn.Linear(256, 2)

        # Stage 2: Multi-class disease classification (10 diseases)
        self.stage2_head = nn.Linear(256, num_diseases)

    def forward(self, x):
        features = self.feature_layers(x)
        stage1_output = self.stage1_head(features)
        stage2_output = self.stage2_head(features)
        return stage1_output, stage2_output


class MicrobiomePredictor:
    """Microbiome disease prediction with two-stage pipeline"""

    def __init__(self, model_stage1_path, model_stage2_path, scaler_path=None,
                 feature_selector_path=None, device='cpu'):
        self.device = device

        # Disease classes (11 total: 1 Healthy + 10 diseases)
        self.all_classes = [
            'Healthy',
            'COVID19',
            'Colorectal_Neoplasms',
            'Crohns',
            'Diabetes',
            'IBS',
            'KidneyFailure',
            'Parkinsons',
            'Ulcerative_Colitis',
            'cysticfibrosis',
            'nafld'
        ]

        self.disease_classes = self.all_classes[1:]  # Exclude Healthy

        # Load models
        self.model = MicrobiomeClassifier(input_size=500, num_diseases=10).to(device)

        # Load Stage 1 weights
        stage1_state = torch.load(model_stage1_path, map_location=device)
        # Load Stage 2 weights
        stage2_state = torch.load(model_stage2_path, map_location=device)

        # Combine weights into single model
        combined_state = {}
        for key, value in stage1_state.items():
            if 'stage1' in key or 'feature' in key:
                combined_state[key] = value
        for key, value in stage2_state.items():
            if 'stage2' in key or 'feature' in key:
                combined_state[key] = value

        self.model.load_state_dict(combined_state, strict=False)
        self.model.eval()

        # Load scaler and feature selector if provided
        self.scaler = None
        self.feature_selector = None

        if scaler_path:
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)

        if feature_selector_path:
            with open(feature_selector_path, 'rb') as f:
                self.feature_selector = pickle.load(f)

    def preprocess_data(self, data):
        """Preprocess microbiome data"""
        # Expecting data as dict with taxon names as keys and abundances as values
        # Convert to array (assuming 500 features expected)

        if isinstance(data, dict):
            # Convert dict to array (order matters - should match training)
            features = np.array(list(data.values()), dtype=np.float32)
        elif isinstance(data, (list, np.ndarray)):
            features = np.array(data, dtype=np.float32)
        else:
            raise ValueError("Data must be dict, list, or numpy array")

        # Reshape to 2D if needed
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Apply feature selection if available
        if self.feature_selector:
            features = self.feature_selector.transform(features)

        # Apply scaling if available
        if self.scaler:
            features = self.scaler.transform(features)

        return torch.FloatTensor(features).to(self.device)

    def predict(self, data):
        """Make prediction with two-stage pipeline"""
        # Preprocess
        features = self.preprocess_data(data)

        # Forward pass
        with torch.no_grad():
            stage1_output, stage2_output = self.model(features)

            # Get probabilities
            probs_stage1 = F.softmax(stage1_output, dim=1).cpu().numpy()[0]
            probs_stage2 = F.softmax(stage2_output, dim=1).cpu().numpy()[0]

        # Stage 1: Healthy vs Diseased
        healthy_prob = float(probs_stage1[0])
        disease_prob = float(probs_stage1[1])

        # Determine confidence level
        confidence_level = "Uncertain"
        if disease_prob > 0.7:
            confidence_level = "High"
        elif disease_prob > 0.3:
            confidence_level = "Low"

        # Calculate risk-weighted probabilities
        risk_scores = {
            'Healthy': healthy_prob
        }

        for i, disease in enumerate(self.disease_classes):
            risk_scores[disease] = disease_prob * probs_stage2[i]

        # Get top prediction
        predicted_class = max(risk_scores, key=risk_scores.get)

        return {
            'predicted': predicted_class,
            'confidence': confidence_level,
            'healthy_prob': healthy_prob,
            'disease_prob': disease_prob,
            'risk_weighted_probabilities': risk_scores,
            'stage2_probabilities': {
                disease: float(probs_stage2[i])
                for i, disease in enumerate(self.disease_classes)
            }
        }


# For demo purposes (when models not loaded)
def get_demo_prediction(data_length=500):
    """Generate demo prediction when models aren't loaded"""
    import random

    diseases = [
        'Healthy', 'COVID19', 'Colorectal_Neoplasms', 'Crohns',
        'Diabetes', 'IBS', 'KidneyFailure', 'Parkinsons',
        'Ulcerative_Colitis', 'cysticfibrosis', 'nafld'
    ]

    # Random probabilities
    probs = np.random.dirichlet(np.ones(len(diseases)))

    risk_scores = {disease: float(prob) for disease, prob in zip(diseases, probs)}
    predicted = max(risk_scores, key=risk_scores.get)

    return {
        'predicted': predicted,
        'confidence': random.choice(['High', 'Low', 'Uncertain']),
        'healthy_prob': float(probs[0]),
        'disease_prob': float(1 - probs[0]),
        'risk_weighted_probabilities': risk_scores
    }
