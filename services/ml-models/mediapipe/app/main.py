"""
MediaPipe Face Mesh Service
Extracts 33 facial landmark-based features
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import mediapipe as mp
import cv2
import numpy as np
from PIL import Image
import io
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MediaPipe Service", version="1.0.0")

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

logger.info("MediaPipe Face Mesh initialized")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "mediapipe", "version": "1.0.0"}


@app.post("/extract")
async def extract_features(image: UploadFile = File(...)):
    """Extract 33 facial landmark features"""

    try:
        # Read image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width = img.shape[:2]

        # Process with MediaPipe
        results = face_mesh.process(img_rgb)

        if not results.multi_face_landmarks:
            raise HTTPException(status_code=400, detail="No face detected")

        landmarks = results.multi_face_landmarks[0]

        # Extract features
        features = extract_facial_features(landmarks, width, height)

        return {
            "status": "success",
            "features": features,
            "feature_count": len(features)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def extract_facial_features(landmarks, width: int, height: int) -> dict:
    """
    Extract 33 features from MediaPipe landmarks

    Features:
    - Basic measurements (face length, width, proportions)
    - Eye measurements (spacing, size, dimensions)
    - Eyebrow measurements
    - Nose measurements
    - Lip measurements
    - Facial symmetry scores
    """

    # Convert landmarks to pixel coordinates
    points = []
    for landmark in landmarks.landmark:
        x = landmark.x * width
        y = landmark.y * height
        points.append((x, y))

    features = {}

    # Key landmark indices (MediaPipe 468 landmarks)
    # Face contour
    face_top = points[10]  # Forehead
    face_bottom = points[152]  # Chin
    face_left = points[234]
    face_right = points[454]

    # Eyes
    left_eye_left = points[33]
    left_eye_right = points[133]
    right_eye_left = points[362]
    right_eye_right = points[263]

    # Calculate features
    # 1-3: Basic face measurements
    features["face_length"] = round(euclidean_distance(face_top, face_bottom), 2)
    features["face_width"] = round(euclidean_distance(face_left, face_right), 2)
    features["facial_ratio"] = round(features["face_length"] / features["face_width"], 3)

    # 4-6: Eye measurements
    left_eye_width = euclidean_distance(left_eye_left, left_eye_right)
    right_eye_width = euclidean_distance(right_eye_left, right_eye_right)
    eye_distance = euclidean_distance(left_eye_right, right_eye_left)

    features["left_eye_width"] = round(left_eye_width, 2)
    features["right_eye_width"] = round(right_eye_width, 2)
    features["eye_spacing"] = round(eye_distance, 2)

    # 7-10: Eye dimensions
    left_eye_top = points[159]
    left_eye_bottom = points[145]
    right_eye_top = points[386]
    right_eye_bottom = points[374]

    features["left_eye_height"] = round(euclidean_distance(left_eye_top, left_eye_bottom), 2)
    features["right_eye_height"] = round(euclidean_distance(right_eye_top, right_eye_bottom), 2)
    features["left_eye_aspect_ratio"] = round(features["left_eye_height"] / left_eye_width, 3)
    features["right_eye_aspect_ratio"] = round(features["right_eye_height"] / right_eye_width, 3)

    # 11-14: Eyebrow measurements
    left_eyebrow_inner = points[70]
    left_eyebrow_outer = points[46]
    right_eyebrow_inner = points[300]
    right_eyebrow_outer = points[276]

    features["left_eyebrow_length"] = round(euclidean_distance(left_eyebrow_inner, left_eyebrow_outer), 2)
    features["right_eyebrow_length"] = round(euclidean_distance(right_eyebrow_inner, right_eyebrow_outer), 2)
    features["eyebrow_spacing"] = round(euclidean_distance(left_eyebrow_inner, right_eyebrow_inner), 2)
    features["eyebrow_to_eye_left"] = round(euclidean_distance(left_eyebrow_inner, left_eye_top), 2)

    # 15-17: Nose measurements
    nose_tip = points[1]
    nose_left = points[98]
    nose_right = points[327]

    features["nose_width"] = round(euclidean_distance(nose_left, nose_right), 2)
    features["nose_height"] = round(euclidean_distance(face_top, nose_tip), 2)
    features["nose_tip_to_chin"] = round(euclidean_distance(nose_tip, face_bottom), 2)

    # 18-20: Lip measurements
    upper_lip_top = points[0]
    lower_lip_bottom = points[17]
    mouth_left = points[61]
    mouth_right = points[291]

    features["mouth_width"] = round(euclidean_distance(mouth_left, mouth_right), 2)
    features["lip_height"] = round(euclidean_distance(upper_lip_top, lower_lip_bottom), 2)
    features["mouth_to_chin"] = round(euclidean_distance(lower_lip_bottom, face_bottom), 2)

    # 21-24: Cheekbone measurements
    left_cheekbone = points[234]
    right_cheekbone = points[454]
    features["cheekbone_width"] = round(euclidean_distance(left_cheekbone, right_cheekbone), 2)
    features["cheekbone_prominence_left"] = round(left_cheekbone[0] - face_left[0], 2)
    features["cheekbone_prominence_right"] = round(face_right[0] - right_cheekbone[0], 2)
    features["cheekbone_height_left"] = round(face_top[1] - left_cheekbone[1], 2)

    # 25-28: Facial symmetry
    features["eye_symmetry"] = round(abs(left_eye_width - right_eye_width), 2)
    features["eyebrow_symmetry"] = round(abs(features["left_eyebrow_length"] - features["right_eyebrow_length"]), 2)
    features["facial_symmetry"] = round(calculate_symmetry_score(features), 3)
    features["vertical_thirds_ratio"] = round(calculate_vertical_thirds(points), 3)

    # 29-33: Additional proportions
    features["eye_to_eyebrow_ratio"] = round(features["eyebrow_to_eye_left"] / features["left_eye_height"], 3)
    features["nose_to_mouth_ratio"] = round(features["nose_tip_to_chin"] / features["mouth_to_chin"], 3)
    features["face_symmetry_score"] = round((1.0 - features["eye_symmetry"] / 100), 3)
    features["golden_ratio_approximation"] = round(features["face_length"] / features["face_width"], 3)
    features["horizontal_thirds_ratio"] = round(features["eye_spacing"] / features["face_width"], 3)

    return features


def euclidean_distance(p1: tuple, p2: tuple) -> float:
    """Calculate Euclidean distance between two points"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def calculate_symmetry_score(features: dict) -> float:
    """Calculate overall facial symmetry score (0-1, 1 = perfect symmetry)"""
    symmetry_features = [
        features["eye_symmetry"],
        features["eyebrow_symmetry"]
    ]

    # Normalize and invert (higher = more symmetric)
    avg_difference = np.mean(symmetry_features)
    symmetry_score = max(0, 1 - (avg_difference / 50))

    return symmetry_score


def calculate_vertical_thirds(points: list) -> float:
    """Calculate vertical thirds ratio (ideal = 1.0)"""
    # Face should divide into 3 equal vertical sections
    face_top = points[10][1]
    eyebrow_level = points[70][1]
    nose_tip = points[1][1]
    chin = points[152][1]

    upper_third = eyebrow_level - face_top
    middle_third = nose_tip - eyebrow_level
    lower_third = chin - nose_tip

    # Calculate ratio of uniformity
    thirds = [upper_third, middle_third, lower_third]
    avg_third = np.mean(thirds)
    variance = np.var(thirds)

    # Lower variance = more ideal proportions
    ratio = 1.0 - (variance / (avg_third ** 2)) if avg_third > 0 else 0

    return max(0, min(1, ratio))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
