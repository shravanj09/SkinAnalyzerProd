"""
FFHQ-Wrinkle Detection Service
Uses: Computer Vision-based wrinkle analysis
Features: 10 (wrinkle density, depth, regional analysis, texture metrics)
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io
import logging
import numpy as np
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="FFHQ-Wrinkle Detection Service", version="1.0.0")


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "ffhq-wrinkle",
        "model": "CV-based wrinkle analysis"
    }


def analyze_wrinkles(image: Image.Image) -> dict:
    """Analyze wrinkles using computer vision techniques"""
    try:
        # Convert to numpy and grayscale
        img_array = np.array(image.convert('RGB'))
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape

        # Define facial regions
        regions = {
            'forehead': gray[0:h//3, w//4:3*w//4],
            'crow_feet_left': gray[h//4:h//2, 0:w//4],
            'crow_feet_right': gray[h//4:h//2, 3*w//4:w],
            'nasolabial_left': gray[h//3:2*h//3, w//4:w//2],
            'nasolabial_right': gray[h//3:2*h//3, w//2:3*w//4],
            'mouth_area': gray[2*h//3:h, w//3:2*w//3]
        }

        # 1. Overall wrinkle density using edge detection
        edges = cv2.Canny(gray, 50, 150)
        overall_density = (np.sum(edges > 0) / edges.size) * 100

        # 2. Wrinkle depth/severity using gradient magnitude
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobelx**2 + sobely**2)
        wrinkle_depth = np.mean(gradient_mag) / 10

        # 3. Texture roughness using standard deviation
        texture_roughness = np.std(gray) / 25.5

        # 4. Regional wrinkle analysis
        regional_scores = {}
        for region_name, region_img in regions.items():
            if region_img.size > 0:
                region_edges = cv2.Canny(region_img, 50, 150)
                regional_density = (np.sum(region_edges > 0) / region_edges.size) * 100
                regional_scores[f"wrinkle_{region_name}"] = round(float(regional_density), 2)
            else:
                regional_scores[f"wrinkle_{region_name}"] = 0.0

        # 5. Fine line count estimation (high-frequency edges)
        fine_edges = cv2.Canny(gray, 100, 200)
        fine_line_density = (np.sum(fine_edges > 0) / fine_edges.size) * 100

        # 6. Deep wrinkle count estimation (low-frequency edges)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        deep_edges = cv2.Canny(blurred, 30, 100)
        deep_wrinkle_density = (np.sum(deep_edges > 0) / deep_edges.size) * 100

        # 7. Skin smoothness (inverse of texture variance)
        local_variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        skin_smoothness = max(0, 10 - (local_variance / 100))

        # Calculate dominant wrinkle region
        max_region = max(regional_scores.items(), key=lambda x: x[1])
        dominant_region = max_region[0].replace("wrinkle_", "")

        # Overall wrinkle severity score (0-10)
        severity_score = min(10, (overall_density * 0.5 + wrinkle_depth * 0.3 + texture_roughness * 0.2))

        return {
            "wrinkle_overall_density": round(float(overall_density), 2),
            "wrinkle_depth_severity": round(float(wrinkle_depth), 2),
            "wrinkle_texture_roughness": round(float(texture_roughness), 2),
            "wrinkle_fine_lines": round(float(fine_line_density), 2),
            "wrinkle_deep_lines": round(float(deep_wrinkle_density), 2),
            "wrinkle_skin_smoothness": round(float(skin_smoothness), 2),
            "wrinkle_severity_score": round(float(severity_score), 2),
            "wrinkle_dominant_region": dominant_region,
            **regional_scores  # Adds 6 regional features
        }
    except Exception as e:
        logger.error(f"Wrinkle analysis error: {e}")
        return {
            "wrinkle_overall_density": 0.0,
            "wrinkle_depth_severity": 0.0,
            "wrinkle_texture_roughness": 0.0,
            "wrinkle_fine_lines": 0.0,
            "wrinkle_deep_lines": 0.0,
            "wrinkle_skin_smoothness": 5.0,
            "wrinkle_severity_score": 0.0,
            "wrinkle_dominant_region": "unknown",
            "wrinkle_forehead": 0.0,
            "wrinkle_crow_feet_left": 0.0,
            "wrinkle_crow_feet_right": 0.0,
            "wrinkle_nasolabial_left": 0.0,
            "wrinkle_nasolabial_right": 0.0,
            "wrinkle_mouth_area": 0.0
        }


@app.post("/extract")
async def extract_features(image: UploadFile = File(...)):
    """Extract wrinkle detection features from image"""
    try:
        # Read image
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data)).convert('RGB')

        # Analyze wrinkles
        features = analyze_wrinkles(pil_image)

        logger.info(f"Extracted {len(features)} wrinkle features")

        return {
            "status": "success",
            "features": features,
            "feature_count": len(features),
            "model": "CV-based wrinkle analysis"
        }

    except Exception as e:
        logger.error(f"Feature extraction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
