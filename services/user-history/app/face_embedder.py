"""
Face Embedding Generator using FaceNet512
Generates unique 512-dimensional vectors for face identification
"""
import numpy as np
from deepface import DeepFace
import cv2
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class FaceEmbedder:
    """Generate face embeddings using FaceNet512"""

    def __init__(self, model_name: str = "Facenet512"):
        """
        Initialize face embedder
        Args:
            model_name: Model to use (Facenet512, Facenet, VGG-Face, etc.)
        """
        self.model_name = model_name
        logger.info(f"FaceEmbedder initialized with model: {model_name}")

    def generate_embedding(self, face_img: np.ndarray) -> Optional[List[float]]:
        """
        Generate embedding from face image
        Args:
            face_img: Face image as numpy array (RGB or BGR)
        Returns:
            512-dimensional embedding vector or None if failed
        """
        try:
            # Ensure image is in correct format
            if len(face_img.shape) == 2:  # Grayscale
                face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
            elif face_img.shape[2] == 4:  # RGBA
                face_img = cv2.cvtColor(face_img, cv2.COLOR_RGBA2RGB)

            # Generate embedding using DeepFace
            embedding_objs = DeepFace.represent(
                img_path=face_img,
                model_name=self.model_name,
                enforce_detection=False,  # Don't fail if face detection fails
                detector_backend="skip"  # Skip face detection (already cropped)
            )

            if embedding_objs and len(embedding_objs) > 0:
                embedding = embedding_objs[0]["embedding"]
                logger.debug(f"Generated embedding: shape={len(embedding)}")
                return embedding
            else:
                logger.warning("No embedding generated")
                return None

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None

    @staticmethod
    def cosine_similarity(embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings
        Returns value between 0 and 1 (1 = identical, 0 = completely different)
        """
        try:
            emb1 = np.array(embedding1)
            emb2 = np.array(embedding2)

            # Normalize vectors
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            # Calculate cosine similarity
            similarity = np.dot(emb1, emb2) / (norm1 * norm2)
            return float(similarity)

        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
