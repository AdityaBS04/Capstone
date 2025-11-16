# 🔧 MediScan AI - Backend API Documentation

## Overview

The MediScan AI backend is a Flask-based REST API that provides disease detection and classification services using PyTorch deep learning models.

**Base URL**: `http://localhost:5000`

---

## Endpoints

### 1. Health Check

Check if the API server is running and models are loaded.

**Endpoint**: `GET /api/health`

**Request**: None

**Response**:
```json
{
  "status": "online",
  "models_loaded": true,
  "device": "cpu"
}
```

**Status Codes**:
- `200 OK`: Server is healthy

**Example**:
```bash
curl http://localhost:5000/api/health
```

---

### 2. Analyze Image

Upload a medical image for disease detection and classification.

**Endpoint**: `POST /api/analyze-image`

**Request**:
- **Content-Type**: `multipart/form-data`
- **Body**:
  - `image` (file, required): Medical image file (JPG, PNG)

**Response**:
```json
{
  "binary": {
    "has_disease": true,
    "confidence": 0.89
  },
  "disease": {
    "predicted_class": 0,
    "predicted_class_name": "esophagitis",
    "confidence": 0.85,
    "matched_probabilities": {
      "esophagitis": 0.85,
      "polyps": 0.12,
      "ulcerative_colitis": 0.03
    }
  }
}
```

**Response Fields**:

**Binary Object**:
- `has_disease` (boolean): Whether disease is detected
- `confidence` (float 0-1): Confidence of the binary prediction

**Disease Object** (only present if `has_disease` is true):
- `predicted_class` (int): Disease class index (0, 1, or 2)
- `predicted_class_name` (string): Disease name
  - `"esophagitis"` (index 0)
  - `"polyps"` (index 1)
  - `"ulcerative_colitis"` (index 2)
- `confidence` (float 0-1): Confidence of the disease prediction
- `matched_probabilities` (object): Probability for each disease class

**Status Codes**:
- `200 OK`: Analysis successful
- `400 Bad Request`: No image provided
- `500 Internal Server Error`: Analysis failed or models not loaded

**Example (cURL)**:
```bash
curl -X POST http://localhost:5000/api/analyze-image \
  -F "image=@/path/to/image.jpg"
```

**Example (JavaScript/Axios)**:
```javascript
const formData = new FormData();
formData.append('image', imageFile);

const response = await axios.post(
  'http://localhost:5000/api/analyze-image',
  formData,
  {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  }
);

console.log(response.data);
```

**Example (Python/Requests)**:
```python
import requests

files = {'image': open('/path/to/image.jpg', 'rb')}
response = requests.post(
    'http://localhost:5000/api/analyze-image',
    files=files
)

print(response.json())
```

---

### 3. Analyze Text

Analyze medical text for disease prediction (Coming Soon).

**Endpoint**: `POST /api/analyze-text`

**Request**:
- **Content-Type**: `application/json`
- **Body**:
```json
{
  "text": "Patient symptoms and medical report text..."
}
```

**Response** (Current):
```json
{
  "error": "Text analysis models are not yet implemented. Coming soon!"
}
```

**Status Codes**:
- `501 Not Implemented`: Feature not yet available

---

## Data Flow

### Image Analysis Pipeline

```
1. Client uploads image via POST /api/analyze-image
          ↓
2. Flask receives image file
          ↓
3. Convert to PIL Image (RGB)
          ↓
4. Binary Classification:
   - Transform image (resize, normalize)
   - Pass through BinaryClassifier model
   - Get probability (sigmoid output)
   - Threshold at 0.5 to determine disease presence
          ↓
5. If disease detected:
   - Apply 3 preprocessing methods:
     a) Esophagitis enhancement
     b) Polyp detection
     c) Ulcerative colitis enhancement
   - Pass each through DiseaseClassifier
   - Get softmax probabilities for each
   - Extract "matched probability" for each method
   - Select disease with highest matched probability
          ↓
6. Return JSON response to client
```

---

## Models

### Binary Classifier

**Architecture**: ResNet18 with custom head
```
ResNet18 backbone
    ↓
Linear(512, 512)
    ↓
ReLU
    ↓
Dropout(0.3)
    ↓
Linear(512, 1)
    ↓
Sigmoid
```

**Input**: RGB image (224x224)
**Output**: Single probability value (0-1)
- < 0.5: Normal
- ≥ 0.5: Diseased

---

### Disease Classifier

**Architecture**: ResNet18 with custom head
```
ResNet18 backbone
    ↓
Linear(512, 512)
    ↓
ReLU
    ↓
Dropout(0.3)
    ↓
Linear(512, 3)
    ↓
Softmax
```

**Input**: RGB image (224x224) with preprocessing applied
**Output**: 3 probability values (sum to 1.0)
- Index 0: Esophagitis
- Index 1: Polyps
- Index 2: Ulcerative Colitis

---

## Preprocessing Methods

### 1. Esophagitis Enhancement
- **Focus**: Inflammation detection
- **Technique**: HSV color space filtering for red regions
- **Target**: Reddish inflammation markers

### 2. Polyp Detection
- **Focus**: Texture-based features
- **Technique**: Currently returns original (can add GLCM, LBP, HOG)
- **Target**: Polyp texture patterns

### 3. Ulcerative Colitis Enhancement
- **Focus**: Pattern and texture detection
- **Technique**:
  - Red region detection (HSV + LAB color spaces)
  - Texture analysis (variance, edges)
  - Morphological operations
- **Target**: UC-specific patterns

---

## Image Transforms

All images are transformed before model inference:

```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

---

## Error Handling

### Common Errors

**No Image Provided**:
```json
{
  "error": "No image provided"
}
```
Status: 400

**Models Not Loaded**:
```json
{
  "error": "Models not loaded. Please check model files."
}
```
Status: 500

**Invalid Image Format**:
```json
{
  "error": "Cannot identify image file"
}
```
Status: 500

**Processing Error**:
```json
{
  "error": "Error message details..."
}
```
Status: 500

---

## Configuration

### Device Selection

The backend automatically selects the best available device:

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

- **CUDA**: Used if NVIDIA GPU available (faster)
- **CPU**: Fallback option (slower but works everywhere)

### Model Loading

**Demo Mode** (current):
```python
binary_model.eval()
disease_model.eval()
# Uses randomly initialized weights
```

**Production Mode** (when models ready):
```python
binary_model.load_state_dict(
    torch.load('../models/image/image_identification.pth',
               map_location=device)
)
disease_model.load_state_dict(
    torch.load('../models/image/image_classfication.pth',
               map_location=device)
)
```

---

## Performance

### Expected Response Times

- **Health Check**: < 10ms
- **Image Analysis** (CPU): 1-3 seconds
- **Image Analysis** (GPU): 0.2-0.5 seconds

### Optimization Tips

1. **Use GPU**: Install CUDA and PyTorch with GPU support
2. **Batch Processing**: Process multiple images together
3. **Model Quantization**: Reduce model size for faster inference
4. **Caching**: Cache preprocessing results if analyzing same image multiple times

---

## CORS Configuration

CORS is enabled for all origins:

```python
from flask_cors import CORS
CORS(app)
```

For production, restrict to specific origins:

```python
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000", "https://yourdomain.com"]
    }
})
```

---

## Development

### Running in Debug Mode

```bash
cd backend
python3 app.py
```

- Auto-reload on code changes
- Detailed error messages
- Not suitable for production

### Running in Production

Use a production WSGI server:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

Or with Waitress (Windows compatible):

```bash
pip install waitress
waitress-serve --host=0.0.0.0 --port=5000 app:app
```

---

## Testing

### Manual Testing

```bash
# Health check
curl http://localhost:5000/api/health

# Upload image (replace path)
curl -X POST http://localhost:5000/api/analyze-image \
  -F "image=@/path/to/test_image.jpg"
```

### Automated Testing

Create `backend/test_api.py`:

```python
import requests
import json

BASE_URL = "http://localhost:5000"

def test_health():
    response = requests.get(f"{BASE_URL}/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    print("✓ Health check passed")

def test_analyze_image():
    files = {'image': open('test_image.jpg', 'rb')}
    response = requests.post(
        f"{BASE_URL}/api/analyze-image",
        files=files
    )
    assert response.status_code == 200
    data = response.json()
    assert "binary" in data
    assert "disease" in data or not data["binary"]["has_disease"]
    print("✓ Image analysis passed")

if __name__ == "__main__":
    test_health()
    test_analyze_image()
    print("\nAll tests passed!")
```

---

## Security Considerations

### Current Implementation (Development)
- ✅ CORS enabled for all origins
- ✅ No authentication required
- ✅ No rate limiting
- ✅ No input validation beyond file type

### Production Recommendations
- 🔒 Add authentication (JWT, API keys)
- 🔒 Implement rate limiting
- 🔒 Validate image files thoroughly
- 🔒 Restrict CORS to specific origins
- 🔒 Use HTTPS
- 🔒 Sanitize user inputs
- 🔒 Add logging and monitoring

---

## Deployment

### Docker (Recommended)

Create `backend/Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

Build and run:

```bash
docker build -t mediscan-backend .
docker run -p 5000:5000 mediscan-backend
```

### Cloud Platforms

**Heroku**:
```bash
heroku create mediscan-api
git push heroku main
```

**AWS Lambda**:
Use Zappa or AWS SAM for serverless deployment

**Google Cloud Run**:
Deploy container directly to Cloud Run

---

## Monitoring

### Logging

Add logging to track usage:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@app.route('/api/analyze-image', methods=['POST'])
def analyze_image():
    logging.info("Image analysis request received")
    # ... rest of code
```

### Metrics

Track:
- Request count
- Response times
- Error rates
- Model inference times
- Device usage (CPU vs GPU)

---

## Support

For issues or questions:
1. Check server logs
2. Verify model files are present
3. Test with curl
4. Check CORS configuration
5. Review error messages

---

**Built for medical AI innovation** 🔬
