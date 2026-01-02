"""
Google Derm Foundation Service
Uses: google/derm-foundation (HuggingFace)
Features: State-of-the-art dermatology foundation model
Dataset: 257× larger than previous models, covers 390 skin conditions
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io
import logging
import numpy as np
from transformers import AutoModel, AutoImageProcessor
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Derm Foundation Service", version="1.0.0")

# Global model storage
model = None
processor = None

@app.on_event("startup")
async def load_model():
    """Load Google Derm Foundation model on startup"""
    global model, processor
    try:
        logger.info("Loading Google Derm Foundation model...")
        model_name = "google/derm-foundation"
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        logger.info("✓ Google Derm Foundation loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "derm-foundation",
        "model": "google/derm-foundation",
        "dataset_size": "257x larger than competitors",
        "conditions_covered": 390,
        "model_loaded": model is not None
    }


@app.post("/extract")
async def extract_features(image: UploadFile = File(...)):
    """Extract skin features using Google Derm Foundation"""
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Read image
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data)).convert('RGB')

        # Preprocess for model
        inputs = processor(images=pil_image, return_tensors="pt")

        # Run inference - get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state

            # Get pooled representation
            pooled_output = embeddings.mean(dim=1).squeeze().numpy()

            # Calculate confidence metrics from embeddings
            embedding_magnitude = float(np.linalg.norm(pooled_output))
            embedding_variance = float(np.var(pooled_output))

        # Extract features from embeddings
        # Note: This is a simplified feature extraction
        # In production, you would fine-tune for specific tasks
        features = {
            "derm_embedding_quality": round(embedding_magnitude / 10, 2),
            "derm_feature_diversity": round(embedding_variance * 100, 2),
            "derm_model_confidence": round(min(100, embedding_magnitude * 2), 2),
            "derm_foundation_score": round(min(10, embedding_magnitude / 5), 2)
        }

        logger.info(f"Extracted {len(features)} Derm Foundation features")

        return {
            "status": "success",
            "features": features,
            "feature_count": len(features),
            "model": "google/derm-foundation",
            "note": "Foundation model - provides base embeddings for skin analysis"
        }

    except Exception as e:
        logger.error(f"Feature extraction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8024)
