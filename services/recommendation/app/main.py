"""
Recommendation Service - Product and routine recommendations
"""
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Recommendation Service", version="1.0.0")


class RecommendationRequest(BaseModel):
    features: Dict[str, Any]


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "recommendation"}


@app.post("/recommend")
async def get_recommendations(request: RecommendationRequest):
    """Generate product and routine recommendations based on features"""

    features = request.features

    # Simple rule-based recommendations
    products = []
    routines = []

    # Acne-prone skin
    if features.get("acne_severity_score", 0) > 50:
        products.append({
            "name": "Salicylic Acid Cleanser",
            "type": "cleanser",
            "description": "Deep cleansing formula with 2% salicylic acid",
            "priority": "high",
            "reason": "Helps treat acne"
        })

    # Dry skin
    if features.get("moisture_level", 100) < 40:
        products.append({
            "name": "Hyaluronic Acid Serum",
            "type": "serum",
            "description": "Ultra-hydrating serum with multi-molecular hyaluronic acid",
            "priority": "high",
            "reason": "Boosts hydration"
        })

    # Wrinkles
    if features.get("wrinkle_depth_severity", 0) > 5:
        products.append({
            "name": "Retinol Night Cream",
            "type": "treatment",
            "description": "0.5% retinol anti-aging night treatment",
            "priority": "high",
            "reason": "Reduces wrinkles"
        })

    # Dark circles
    if features.get("dark_circle_severity", 0) > 5:
        products.append({
            "name": "Vitamin C Eye Cream",
            "type": "eye-cream",
            "description": "15% vitamin C brightening eye treatment",
            "priority": "medium",
            "reason": "Brightens under-eye area"
        })

    # Basic routine
    routines.append({
        "name": "Morning Routine",
        "type": "daily",
        "description": "Essential morning skincare routine for healthy skin",
        "priority": "high",
        "reason": "Protects and prepares skin for the day",
        "steps": [
            "Cleanser",
            "Toner",
            "Vitamin C Serum",
            "Moisturizer",
            "SPF 50 Sunscreen"
        ]
    })

    routines.append({
        "name": "Evening Routine",
        "type": "daily",
        "description": "Night-time repair and treatment routine",
        "priority": "high",
        "reason": "Repairs and rejuvenates skin overnight",
        "steps": [
            "Cleanser",
            "Toner",
            "Treatment (Retinol/AHA/BHA)",
            "Night Cream"
        ]
    })

    return {
        "products": products if products else [],
        "routines": routines if routines else []
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8021)
