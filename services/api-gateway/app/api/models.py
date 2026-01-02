"""Model information endpoints"""
from fastapi import APIRouter
from typing import Dict, List
import logging

from app.core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/models/available")
async def get_available_models() -> Dict:
    """
    Get list of available ML models and their status

    Returns information about:
    - Which models are enabled
    - Model URLs
    - Feature counts
    - Tier classification
    """

    tier1_models = {
        "mediapipe": {
            "enabled": settings.ENABLE_MEDIAPIPE,
            "url": settings.MEDIAPIPE_URL,
            "features": 33,
            "tier": 1,
            "description": "Facial landmarks and proportions",
            "accuracy": "95-97%"
        },
        "opencv": {
            "enabled": settings.ENABLE_OPENCV,
            "url": settings.OPENCV_URL,
            "features": 6,
            "tier": 1,
            "description": "Moisture and hydration analysis",
            "accuracy": "70-75%"
        },
        "shifaa-unet": {
            "enabled": settings.ENABLE_SHIFAA_UNET,
            "url": settings.SHIFAA_UNET_URL,
            "features": 24,
            "tier": 1,
            "description": "Facial segmentation",
            "accuracy": "85-92%"
        },
        "ffhq-wrinkle": {
            "enabled": settings.ENABLE_FFHQ_WRINKLE,
            "url": settings.FFHQ_WRINKLE_URL,
            "features": 10,
            "tier": 1,
            "description": "Wrinkle detection",
            "accuracy": "75-85%"
        },
        "skin-type": {
            "enabled": settings.ENABLE_SKIN_TYPE,
            "url": settings.SKIN_TYPE_URL,
            "features": 10,
            "tier": 1,
            "description": "Skin type and color analysis",
            "accuracy": "90-94%"
        },
        "spots-detection": {
            "enabled": settings.ENABLE_SPOTS_DETECTION,
            "url": settings.SPOTS_DETECTION_URL,
            "features": 8,
            "tier": 1,
            "description": "Age spots and lesions",
            "accuracy": "92-97%"
        },
        "acne-detection": {
            "enabled": settings.ENABLE_ACNE_DETECTION,
            "url": settings.ACNE_DETECTION_URL,
            "features": 7,
            "tier": 1,
            "description": "Acne and blemishes",
            "accuracy": "87-92%"
        },
        "sam": {
            "enabled": settings.ENABLE_SAM,
            "url": settings.SAM_URL,
            "features": 7,
            "tier": 1,
            "description": "Oiliness detection",
            "accuracy": "73-78%"
        },
        "claude-api": {
            "enabled": settings.ENABLE_CLAUDE_API,
            "url": settings.CLAUDE_API_URL,
            "features": 13,
            "tier": 1,
            "description": "Subjective assessments",
            "accuracy": "85-90%"
        }
    }

    tier2_models = {
        "unet-dark-circles": {
            "enabled": settings.ENABLE_UNET_DARK_CIRCLES,
            "url": settings.UNET_DARK_CIRCLES_URL,
            "features": 6,
            "tier": 2,
            "description": "Dark circles detection",
            "accuracy": "83-88%",
            "improvement": "+30% over OpenCV"
        },
        "unet-redness": {
            "enabled": settings.ENABLE_UNET_REDNESS,
            "url": settings.UNET_REDNESS_URL,
            "features": 13,
            "tier": 2,
            "description": "Redness and inflammation",
            "accuracy": "74-83%",
            "improvement": "+15% over OpenCV"
        },
        "efficientnet-texture": {
            "enabled": settings.ENABLE_EFFICIENTNET_TEXTURE,
            "url": settings.EFFICIENTNET_TEXTURE_URL,
            "features": 9,
            "tier": 2,
            "description": "Texture analysis",
            "accuracy": "80-90%",
            "improvement": "+30-40% over OpenCV"
        }
    }

    # Calculate totals
    tier1_enabled = sum(1 for m in tier1_models.values() if m["enabled"])
    tier1_features = sum(m["features"] for m in tier1_models.values() if m["enabled"])

    tier2_enabled = sum(1 for m in tier2_models.values() if m["enabled"])
    tier2_features = sum(m["features"] for m in tier2_models.values() if m["enabled"])

    total_features = tier1_features + tier2_features

    return {
        "summary": {
            "tier1": {
                "models_enabled": tier1_enabled,
                "models_total": len(tier1_models),
                "features_enabled": tier1_features,
                "features_total": 118,
                "status": "production-ready"
            },
            "tier2": {
                "models_enabled": tier2_enabled,
                "models_total": len(tier2_models),
                "features_enabled": tier2_features,
                "features_total": 28,
                "status": "requires-training"
            },
            "total_features": total_features,
            "max_features": 146
        },
        "tier1_models": tier1_models,
        "tier2_models": tier2_models
    }


@router.get("/models/{model_name}/info")
async def get_model_info(model_name: str) -> Dict:
    """Get detailed information about a specific model"""

    # This would query the actual model service
    # For now, return placeholder

    return {
        "model_name": model_name,
        "status": "not_implemented",
        "message": "Model info endpoint coming soon"
    }
