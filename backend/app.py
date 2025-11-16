from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import numpy as np
import cv2

app = Flask(__name__)
CORS(app)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# Model Architectures (from the notebook)
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
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)


# Preprocessing methods (from the notebook)
class UpdatedPreprocessor:
    @staticmethod
    def polyp_detection(image):
        # For now, just return the original image
        # You can add the full preprocessing logic from the notebook
        return image

    @staticmethod
    def ulcerative_colitis_enhancement(image):
        """Balanced UC enhancement focusing on texture and severe inflammation"""
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)

        # Detect VERY RED regions
        lower_red1 = np.array([0, 120, 100])
        upper_red1 = np.array([8, 255, 255])
        lower_red2 = np.array([172, 120, 100])
        upper_red2 = np.array([180, 255, 255])

        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        intense_red_mask = cv2.bitwise_or(mask_red1, mask_red2)

        # Texture detection
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        mean_filtered = cv2.blur(gray, (5, 5))
        variance = cv2.absdiff(gray, mean_filtered)
        _, texture_mask = cv2.threshold(variance, 15, 255, cv2.THRESH_BINARY)

        # Edge detection
        edges = cv2.Canny(gray, 40, 120)
        kernel_edge = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges_dilated = cv2.dilate(edges, kernel_edge, iterations=1)

        # LAB color space
        a_channel = lab[:, :, 1]
        _, red_lab_mask = cv2.threshold(a_channel, 140, 255, cv2.THRESH_BINARY)

        # Combine
        color_mask = cv2.bitwise_or(intense_red_mask, red_lab_mask)
        texture_combined = cv2.bitwise_or(texture_mask, edges_dilated)
        final_mask = cv2.bitwise_and(color_mask, texture_combined)

        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Remove small regions
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(final_mask, connectivity=8)
        min_size = 100
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_size:
                final_mask[labels == i] = 0

        # Enhance detected regions
        result_bgr = img_cv.copy().astype(np.float32)
        mask_regions = final_mask > 0

        if np.any(mask_regions):
            result_bgr[mask_regions, 2] = np.clip(result_bgr[mask_regions, 2] * 1.5 + 40, 0, 255)
            result_bgr[mask_regions, 1] = np.clip(result_bgr[mask_regions, 1] * 0.7, 0, 255)
            result_bgr[mask_regions, 0] = np.clip(result_bgr[mask_regions, 0] * 0.65, 0, 255)

        result_rgb = cv2.cvtColor(result_bgr.astype(np.uint8), cv2.COLOR_BGR2RGB)
        return Image.fromarray(result_rgb)

    @staticmethod
    def esophagitis_enhancement(image):
        """Esophagitis enhancement"""
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)

        lower_eso1 = np.array([175, 45, 150])
        upper_eso1 = np.array([180, 75, 240])
        lower_eso2 = np.array([173, 40, 140])
        upper_eso2 = np.array([180, 80, 250])
        lower_eso3 = np.array([170, 35, 130])
        upper_eso3 = np.array([180, 85, 255])
        lower_eso4 = np.array([0, 40, 140])
        upper_eso4 = np.array([5, 80, 250])

        mask1 = cv2.inRange(hsv, lower_eso1, upper_eso1)
        mask2 = cv2.inRange(hsv, lower_eso2, upper_eso2)
        mask3 = cv2.inRange(hsv, lower_eso3, upper_eso3)
        mask4 = cv2.inRange(hsv, lower_eso4, upper_eso4)

        inflammation_mask = cv2.bitwise_or(mask1, mask2)
        inflammation_mask = cv2.bitwise_or(inflammation_mask, mask3)
        inflammation_mask = cv2.bitwise_or(inflammation_mask, mask4)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        inflammation_mask = cv2.morphologyEx(inflammation_mask, cv2.MORPH_CLOSE, kernel)
        inflammation_mask = cv2.morphologyEx(inflammation_mask, cv2.MORPH_OPEN, kernel)

        result_bgr = img_cv.copy().astype(np.float32)
        mask_regions = inflammation_mask > 0
        if np.any(mask_regions):
            result_bgr[mask_regions, 2] = np.clip(result_bgr[mask_regions, 2] * 1.4 + 30, 0, 255)
            result_bgr[mask_regions, 1] = np.clip(result_bgr[mask_regions, 1] * 0.8, 0, 255)
            result_bgr[mask_regions, 0] = np.clip(result_bgr[mask_regions, 0] * 0.9, 0, 255)

        result_rgb = cv2.cvtColor(result_bgr.astype(np.uint8), cv2.COLOR_BGR2RGB)
        return Image.fromarray(result_rgb)


class MultiPreprocessingEvaluator:
    """Evaluator for testing with all preprocessing methods"""
    def __init__(self, binary_model, disease_model, device, transform):
        self.binary_model = binary_model
        self.disease_model = disease_model
        self.device = device
        self.transform = transform
        self.class_names = ['esophagitis', 'polyps', 'ulcerative_colitis']
        self.preprocessing_to_disease = {0: 0, 1: 1, 2: 2}

    def preprocess_all_methods(self, image):
        """Apply all three preprocessing methods"""
        return [
            UpdatedPreprocessor.esophagitis_enhancement(image),
            UpdatedPreprocessor.polyp_detection(image),
            UpdatedPreprocessor.ulcerative_colitis_enhancement(image)
        ]

    def evaluate_single_image(self, image):
        """Evaluate with all preprocessing methods"""
        # Binary classification first
        self.binary_model.eval()
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            binary_output = self.binary_model(img_tensor)
            binary_prob = binary_output.item()
            has_disease = binary_prob > 0.5

        binary_result = {
            'has_disease': bool(has_disease),
            'confidence': float(binary_prob if has_disease else 1 - binary_prob)
        }

        # If disease detected, classify it
        disease_result = None
        if has_disease:
            self.disease_model.eval()
            preprocessed_images = self.preprocess_all_methods(image)

            matched_probs = []
            all_predictions = []

            with torch.no_grad():
                for method_idx, processed_img in enumerate(preprocessed_images):
                    img_tensor = self.transform(processed_img).unsqueeze(0).to(self.device)
                    outputs = self.disease_model(img_tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()[0]
                    all_predictions.append(probs)

                    disease_idx = self.preprocessing_to_disease[method_idx]
                    matched_probs.append(probs[disease_idx])

            best_disease_idx = np.argmax(matched_probs)

            disease_result = {
                'predicted_class': int(best_disease_idx),
                'predicted_class_name': self.class_names[best_disease_idx],
                'confidence': float(matched_probs[best_disease_idx]),
                'matched_probabilities': {
                    'esophagitis': float(matched_probs[0]),
                    'polyps': float(matched_probs[1]),
                    'ulcerative_colitis': float(matched_probs[2])
                }
            }

        return {
            'binary': binary_result,
            'disease': disease_result
        }


# Load models (will be loaded when models are available)
try:
    # Image transform
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load models
    binary_model = BinaryClassifier().to(device)
    disease_model = DiseaseClassifier(num_classes=3).to(device)

    # Load actual model weights
    binary_model.load_state_dict(torch.load('../models/image/image_identification.pth', map_location=device))
    disease_model.load_state_dict(torch.load('../models/image/image_classfication.pth', map_location=device))

    # Set to eval mode
    binary_model.eval()
    disease_model.eval()

    # Create evaluator
    evaluator = MultiPreprocessingEvaluator(binary_model, disease_model, device, val_transform)

    models_loaded = True
    print("✓ Models loaded successfully with trained weights!")
except Exception as e:
    models_loaded = False
    print(f"Error loading models: {e}")


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'online',
        'models_loaded': models_loaded,
        'device': str(device)
    })


@app.route('/api/analyze-image', methods=['POST'])
def analyze_image():
    if not models_loaded:
        return jsonify({'error': 'Models not loaded. Please check model files.'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    try:
        # Read image
        image_file = request.files['image']
        image_bytes = image_file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Analyze
        results = evaluator.evaluate_single_image(image)

        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze-text', methods=['POST'])
def analyze_text():
    """Analyze microbiome data for disease prediction"""
    try:
        from text_model import MicrobiomePredictor, get_demo_prediction

        data = request.get_json()

        if not data or 'data' not in data:
            return jsonify({'error': 'No microbiome data provided'}), 400

        microbiome_data = data['data']

        # Try to load models
        try:
            predictor = MicrobiomePredictor(
                model_stage1_path='../models/text/text_identification.pth',
                model_stage2_path='../models/text/text_classification.pth',
                device=device
            )
            results = predictor.predict(microbiome_data)
        except Exception as model_error:
            print(f"Model loading error: {model_error}")
            # Use demo prediction
            data_len = len(microbiome_data) if isinstance(microbiome_data, (list, dict)) else 500
            results = get_demo_prediction(data_len)
            results['note'] = 'Using demo prediction (models not fully loaded)'

        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 MediScan AI Backend Server")
    print("="*60)
    print(f"Device: {device}")
    print(f"Models loaded: {models_loaded}")
    print("\nEndpoints:")
    print("  GET  /api/health       - Health check")
    print("  POST /api/analyze-image - Analyze medical image")
    print("  POST /api/analyze-text  - Analyze medical text (coming soon)")
    print("\nStarting server on http://localhost:5001")
    print("="*60 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5001)
