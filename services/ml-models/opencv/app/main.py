"""
OpenCV Service - Moisture & Hydration Analysis
Extracts 6 moisture-related features
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OpenCV Service", version="1.0.0")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "opencv", "version": "1.0.0"}


@app.post("/extract")
async def extract_features(image: UploadFile = File(...)):
    """Extract 6 moisture/hydration features"""

    try:
        # Read image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        # Extract features
        features = analyze_moisture(img)

        return {
            "status": "success",
            "features": features,
            "feature_count": len(features)
        }

    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def analyze_moisture(img: np.ndarray) -> dict:
    """Analyze moisture and hydration using OpenCV color analysis"""

    # Convert to HSV for better color analysis
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Convert to LAB for skin analysis
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b_channel = cv2.split(lab)

    features = {}

    # 1. Overall moisture level (based on saturation)
    saturation_mean = np.mean(s)
    features["hd_moisture"] = round(float(saturation_mean), 2)

    # 2. Hydration uniformity (lower std = more uniform)
    saturation_std = np.std(s)
    uniformity = max(0, 100 - (saturation_std / 2.55))
    features["hydration_uniformity"] = round(float(uniformity), 2)

    # 3. Dehydration areas (percentage of low-moisture pixels)
    dehydration_mask = s < (saturation_mean * 0.7)
    dehydration_percentage = (np.sum(dehydration_mask) / dehydration_mask.size) * 100
    features["dehydration_areas"] = round(float(dehydration_percentage), 2)

    # 4. Moisture level indicator (normalized 0-100)
    moisture_level = (saturation_mean / 255.0) * 100
    features["moisture_level"] = round(float(moisture_level), 2)

    # 5. Moisture level indicator (alternative calculation)
    features["moisture_level_indicator"] = round(float(moisture_level), 2)

    # 6. Hydration level health (composite score)
    health_score = (uniformity * 0.6) + (moisture_level * 0.4)
    features["hydration_level_health"] = round(float(health_score), 2)

    logger.info(f"Extracted {len(features)} moisture features")

    return features


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
