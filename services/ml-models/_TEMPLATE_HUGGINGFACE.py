"""
TEMPLATE for HuggingFace Model Services
Copy this file to create new model services

Service ports:
- shifaa-unet: 8004 (24 features)
- ffhq-wrinkle: 8005 (10 features)
- skin-type: 8006 (10 features)
- spots-detection: 8007 (8 features)
- acne-detection: 8008 (7 features)
- sam: 8009 (7 features)
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from transformers import AutoModel, AutoProcessor
import torch
from PIL import Image
import io
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CONFIGURE THESE FOR YOUR SERVICE
SERVICE_NAME = "template-service"  # e.g., "skin-type"
SERVICE_PORT = 8000  # e.g., 8006
MODEL_NAME = "your-model-name"  # e.g., "driboune/skin_type"
FEATURE_COUNT = 0  # e.g., 10

app = FastAPI(title=f"{SERVICE_NAME} Service", version="1.0.0")

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Loading model {MODEL_NAME} on {device}")

try:
    model = AutoModel.from_pretrained(MODEL_NAME)
    model = model.to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    logger.info(f"Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None
    processor = None


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": SERVICE_NAME,
        "model": MODEL_NAME,
        "device": device,
        "model_loaded": model is not None
    }


@app.post("/extract")
async def extract_features(image: UploadFile = File(...)):
    """Extract features from image"""

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Read image
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')

        # Process with model
        with torch.no_grad():
            inputs = processor(images=img, return_tensors="pt").to(device)
            outputs = model(**inputs)

        # TODO: Extract features from outputs
        # This depends on your specific model
        features = extract_features_from_outputs(outputs, img)

        return {
            "status": "success",
            "features": features,
            "feature_count": len(features)
        }

    except Exception as e:
        logger.error(f"Feature extraction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def extract_features_from_outputs(outputs, image: Image.Image) -> dict:
    """
    Extract features from model outputs
    Implement this based on your specific model
    """

    # EXAMPLE: For classification models
    # logits = outputs.logits
    # probs = torch.softmax(logits, dim=-1)
    # predicted_class = torch.argmax(probs, dim=-1).item()

    # EXAMPLE: For segmentation models
    # masks = outputs.pred_masks
    # Calculate features from masks

    features = {
        "example_feature_1": 0.0,
        "example_feature_2": 0.0,
        # Add your features here
    }

    return features


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT)


# =============================================================================
# DOCKERFILE TEMPLATE
# =============================================================================
"""
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE {SERVICE_PORT}

HEALTHCHECK CMD curl -f http://localhost:{SERVICE_PORT}/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "{SERVICE_PORT}"]
"""

# =============================================================================
# REQUIREMENTS.TXT TEMPLATE
# =============================================================================
"""
fastapi==0.104.1
uvicorn[standard]==0.24.0
transformers==4.36.0
torch==2.1.0
pillow==10.1.0
python-multipart==0.0.6
numpy==1.26.2
"""
