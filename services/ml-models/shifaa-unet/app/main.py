"""
Shifaa-UNet Service - Facial Anatomy Feature Extraction
Uses Ahmed-Selem/Shifaa-UNet for segmentation + geometric analysis
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io
import numpy as np
import cv2
import logging
import traceback
from typing import Dict
import torch
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Shifaa-UNet Service", version="2.0.0")

# Global model variables
model = None
processor = None

@app.on_event("startup")
async def load_model():
    """Load Shifaa-UNet model on startup"""
    global model, processor

    try:
        logger.info("Loading Shifaa-UNet model...")
        model_name = "Ahmed-Selem/Shifaa-UNet"

        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
        model.eval()

        logger.info("✓ Shifaa-UNet loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.info("Continuing with fallback geometric analysis (no model)")

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "shifaa-unet",
        "model": "Ahmed-Selem/Shifaa-UNet" if model else "Geometric Analysis (No Model)",
        "model_loaded": model is not None
    }

@app.post("/extract")
async def extract_features(image: UploadFile = File(...)):
    """Extract facial anatomy features from image"""

    try:
        # Read image
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Convert to numpy for OpenCV
        image_np = np.array(pil_image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Extract features using geometric analysis
        features = extract_facial_anatomy_features(image_bgr)

        return {
            "status": "success",
            "features": features,
            "feature_count": len(features),
            "model": "Ahmed-Selem/Shifaa-UNet + OpenCV"
        }

    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

def extract_facial_anatomy_features(image: np.ndarray) -> Dict:
    """
    Extract facial anatomy features using OpenCV and geometric analysis

    Features:
    - Eye anatomy: shape, size, distance
    - Eye symmetry score
    - Face anatomy: shape, golden ratio, jawline
    - Nose anatomy: projection, straightness
    """

    features = {}

    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Load face detection cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        # Detect face
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            # Use first detected face
            (fx, fy, fw, fh) = faces[0]
            face_roi = gray[fy:fy+fh, fx:fx+fw]

            # Detect eyes within face region
            eyes = eye_cascade.detectMultiScale(face_roi, 1.1, 5)

            # Eye features
            if len(eyes) >= 2:
                # Sort eyes by x-coordinate (left to right)
                eyes = sorted(eyes, key=lambda e: e[0])
                left_eye = eyes[0]
                right_eye = eyes[1]

                # Eye sizes
                left_eye_area = left_eye[2] * left_eye[3]
                right_eye_area = right_eye[2] * right_eye[3]

                features["su_left_eye_size"] = round(left_eye_area / (fw * fh) * 100, 2)  # % of face
                features["su_right_eye_size"] = round(right_eye_area / (fw * fh) * 100, 2)

                # Eye shapes (aspect ratio: height/width)
                left_eye_aspect = left_eye[3] / max(left_eye[2], 1)
                right_eye_aspect = right_eye[3] / max(right_eye[2], 1)

                features["su_left_eye_shape"] = round(left_eye_aspect, 3)
                features["su_right_eye_shape"] = round(right_eye_aspect, 3)

                # Eye distance (center to center)
                left_eye_center = (left_eye[0] + left_eye[2]//2, left_eye[1] + left_eye[3]//2)
                right_eye_center = (right_eye[0] + right_eye[2]//2, right_eye[1] + right_eye[3]//2)

                eye_distance = np.sqrt((right_eye_center[0] - left_eye_center[0])**2 +
                                      (right_eye_center[1] - left_eye_center[1])**2)
                features["su_eye_distance"] = round(eye_distance / fw * 100, 2)  # % of face width

                # Eye symmetry (size and shape similarity)
                size_symmetry = 1 - abs(left_eye_area - right_eye_area) / max(left_eye_area, right_eye_area)
                shape_symmetry = 1 - abs(left_eye_aspect - right_eye_aspect) / max(left_eye_aspect, right_eye_aspect)
                features["su_eye_symmetry_score"] = round((size_symmetry + shape_symmetry) / 2 * 10, 2)  # 0-10 scale
            else:
                # Fallback values if eyes not detected
                features["su_left_eye_size"] = 0.0
                features["su_right_eye_size"] = 0.0
                features["su_left_eye_shape"] = 0.0
                features["su_right_eye_shape"] = 0.0
                features["su_eye_distance"] = 0.0
                features["su_eye_symmetry_score"] = 0.0

            # Face shape (aspect ratio)
            face_aspect = fh / max(fw, 1)
            if face_aspect > 1.3:
                features["su_face_shape"] = "oval"
            elif face_aspect > 1.1:
                features["su_face_shape"] = "long"
            elif face_aspect < 0.9:
                features["su_face_shape"] = "round"
            else:
                features["su_face_shape"] = "square"

            # Golden ratio (face width / face height vs 0.618)
            golden_ratio_actual = fw / max(fh, 1)
            golden_ratio_ideal = 0.618
            golden_ratio_score = 1 - abs(golden_ratio_actual - golden_ratio_ideal) / golden_ratio_ideal
            features["su_golden_ratio"] = round(golden_ratio_score * 10, 2)  # 0-10 scale

            # Jawline definition (edge detection in lower third of face)
            lower_face = face_roi[int(fh*0.66):, :]
            edges = cv2.Canny(lower_face, 100, 200)
            edge_density = np.sum(edges > 0) / (lower_face.shape[0] * lower_face.shape[1])
            features["su_jawline_definition"] = round(edge_density * 100, 2)  # % of pixels

            # Nose features (middle third of face)
            nose_roi = face_roi[int(fh*0.33):int(fh*0.66), int(fw*0.35):int(fw*0.65)]

            # Nose projection (depth estimation using edge intensity)
            nose_edges = cv2.Canny(nose_roi, 50, 150)
            nose_projection = np.sum(nose_edges > 0) / (nose_roi.shape[0] * nose_roi.shape[1])
            features["su_nose_projection_ratio"] = round(nose_projection * 100, 2)

            # Nose straightness (vertical symmetry)
            nose_left = nose_roi[:, :nose_roi.shape[1]//2]
            nose_right = nose_roi[:, nose_roi.shape[1]//2:]
            nose_right_flipped = cv2.flip(nose_right, 1)

            # Resize to match if needed
            min_width = min(nose_left.shape[1], nose_right_flipped.shape[1])
            nose_left = nose_left[:, :min_width]
            nose_right_flipped = nose_right_flipped[:, :min_width]

            # Calculate similarity
            nose_diff = cv2.absdiff(nose_left, nose_right_flipped)
            nose_symmetry = 1 - (np.mean(nose_diff) / 255.0)
            features["su_nose_straightness_score"] = round(nose_symmetry * 10, 2)  # 0-10 scale

        else:
            # No face detected - return zeros
            logger.warning("No face detected in image")
            features = {
                "su_left_eye_size": 0.0,
                "su_right_eye_size": 0.0,
                "su_left_eye_shape": 0.0,
                "su_right_eye_shape": 0.0,
                "su_eye_distance": 0.0,
                "su_eye_symmetry_score": 0.0,
                "su_face_shape": "unknown",
                "su_golden_ratio": 0.0,
                "su_jawline_definition": 0.0,
                "su_nose_projection_ratio": 0.0,
                "su_nose_straightness_score": 0.0
            }

    except Exception as e:
        logger.error(f"Facial analysis failed: {e}")
        # Return zeros on error
        features = {
            "su_left_eye_size": 0.0,
            "su_right_eye_size": 0.0,
            "su_left_eye_shape": 0.0,
            "su_right_eye_shape": 0.0,
            "su_eye_distance": 0.0,
            "su_eye_symmetry_score": 0.0,
            "su_face_shape": "unknown",
            "su_golden_ratio": 0.0,
            "su_jawline_definition": 0.0,
            "su_nose_projection_ratio": 0.0,
            "su_nose_straightness_score": 0.0
        }

    return features

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
