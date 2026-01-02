"""
Skin Type Classification Service
Uses: driboune/skin_type (HuggingFace)
Features: 11 (Fitzpatrick scale, skin type, undertone, ITA angle, etc.)
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io
import logging
import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
import colorsys
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Skin Type Classification Service", version="1.0.0")

# Global model storage
model = None
processor = None

@app.on_event("startup")
async def load_model():
    """Load HuggingFace model on startup"""
    global model, processor
    try:
        logger.info("Loading skin type classification model...")
        model_name = "driboune/skin_type"
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForImageClassification.from_pretrained(model_name)
        model.eval()
        logger.info("✓ Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "skin-type",
        "model": "driboune/skin_type",
        "model_loaded": model is not None
    }


def calculate_ita_angle(rgb_array):
    """
    Calculate ITA (Individual Typology Angle) for skin tone classification

    ITA° = [arctan((L* - 50) / b*)] × (180/π)

    ITA ranges:
    > 55°: Very Light
    41° to 55°: Light
    28° to 41°: Intermediate
    10° to 28°: Tan
    -30° to 10°: Brown
    < -30°: Dark
    """
    try:
        # Convert RGB to LAB color space
        # cv2 expects BGR format
        bgr = cv2.cvtColor(rgb_array.astype(np.uint8), cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

        # Get average L* and b* values
        l_star = lab[:, :, 0].mean()
        b_star = lab[:, :, 2].mean() - 128  # b* ranges from -128 to 127

        # Calculate ITA angle
        ita_radians = np.arctan2((l_star - 50), b_star)
        ita_degrees = np.degrees(ita_radians)

        return float(ita_degrees)
    except Exception as e:
        logger.error(f"ITA calculation error: {e}")
        return 0.0


def analyze_skin_color(image: Image.Image) -> dict:
    """Analyze skin color properties from image"""
    try:
        # Convert to numpy array
        img_array = np.array(image.convert('RGB'))

        # Get central region (assume face is centered)
        h, w = img_array.shape[:2]
        center_region = img_array[h//3:2*h//3, w//3:2*w//3]

        # Calculate average RGB
        avg_color = center_region.mean(axis=(0, 1))
        r, g, b = avg_color / 255.0

        # Convert to HSV for better color analysis
        h_hsv, s, v = colorsys.rgb_to_hsv(r, g, b)

        # Convert to LAB-like values (simplified)
        l_value = v * 100

        # Calculate ITA angle (industry standard)
        ita_angle = calculate_ita_angle(center_region)

        # Determine undertone
        if r > g and r > b:
            undertone = "warm"
            undertone_score = (r - max(g, b)) * 10
        elif b > r and b > g:
            undertone = "cool"
            undertone_score = (b - max(r, g)) * 10
        else:
            undertone = "neutral"
            undertone_score = 5.0

        # Skin brightness (0-100)
        brightness = l_value

        # Skin saturation (0-100)
        saturation = s * 100

        return {
            "undertone": undertone,
            "undertone_score": round(float(undertone_score), 2),
            "brightness": round(float(brightness), 2),
            "saturation": round(float(saturation), 2),
            "avg_rgb_r": round(float(avg_color[0]), 2),
            "avg_rgb_g": round(float(avg_color[1]), 2),
            "avg_rgb_b": round(float(avg_color[2]), 2),
            "h_hsv": h_hsv,  # HSV hue for st_hsv_hue feature
            "ita_angle": round(ita_angle, 2)  # NEW: ITA angle
        }
    except Exception as e:
        logger.error(f"Color analysis error: {e}")
        return {}


@app.post("/extract")
async def extract_features(image: UploadFile = File(...)):
    """Extract skin type features from image"""
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Read image
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data)).convert('RGB')

        # Preprocess for model
        inputs = processor(images=pil_image, return_tensors="pt")

        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class_idx = logits.argmax(-1).item()
            confidence = probabilities[0][predicted_class_idx].item()

        # Map predicted class to Fitzpatrick scale
        # Note: The model may output different classes, adjust mapping as needed
        fitzpatrick_mapping = {
            0: 1,  # Very fair
            1: 2,  # Fair
            2: 3,  # Medium
            3: 4,  # Olive
            4: 5,  # Brown
            5: 6   # Dark brown/black
        }

        fitzpatrick_type = fitzpatrick_mapping.get(predicted_class_idx, 3)

        # Analyze color properties
        color_features = analyze_skin_color(pil_image)

        # Determine skin type category
        if fitzpatrick_type <= 2:
            skin_type_category = "fair"
        elif fitzpatrick_type <= 4:
            skin_type_category = "medium"
        else:
            skin_type_category = "dark"

        # Extract features (FIXED: removed "skin_" prefix to match database schema)
        features = {
            "st_fitzpatrick_type": fitzpatrick_type,
            "st_skin_type_category": skin_type_category,
            "st_classification_confidence": round(confidence * 100, 2),
            "st_undertone_type": color_features.get("undertone", "neutral"),
            "st_brightness": color_features.get("brightness", 50.0),           # FIXED: was st_skin_brightness
            "st_saturation": color_features.get("saturation", 30.0),           # FIXED: was st_skin_saturation
            "st_avg_r": color_features.get("avg_rgb_r", 128.0),
            "st_avg_g": color_features.get("avg_rgb_g", 128.0),
            "st_avg_b": color_features.get("avg_rgb_b", 128.0),
            "st_hsv_hue": round(color_features.get("h_hsv", 0.0) * 360, 2),  # ADDED: missing HSV hue
            "st_ita_angle": color_features.get("ita_angle", 0.0),            # NEW: ITA angle (industry standard)
        }

        logger.info(f"Extracted {len(features)} skin type features")

        return {
            "status": "success",
            "features": features,
            "feature_count": len(features),
            "model": "driboune/skin_type"
        }

    except Exception as e:
        logger.error(f"Feature extraction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)
