"""Analysis request/response schemas"""
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime


class AnalysisRequest(BaseModel):
    """Analysis request schema"""
    user_id: Optional[str] = None
    selected_models: Optional[List[str]] = None


class FeatureExtraction(BaseModel):
    """Individual feature extraction result"""
    value: Any
    confidence: Optional[float] = None
    source_model: str


class ModelResult(BaseModel):
    """Individual model execution result"""
    status: str  # success | error | timeout
    latency_ms: Optional[float] = None
    error: Optional[str] = None


class Recommendation(BaseModel):
    """Product or routine recommendation"""
    type: str  # product | routine
    name: str
    description: str
    priority: str  # high | medium | low
    reason: str


class AnalysisResponse(BaseModel):
    """Complete analysis response"""
    status: str = "success"
    analysis_id: str
    processing_time_seconds: float
    features: Dict[str, Any]
    feature_count: int
    categorized_features: Optional[Dict[str, Any]] = None
    hierarchical_features: Optional[Dict[str, Any]] = None
    model_results: Dict[str, ModelResult]
    overall_score: float = Field(..., ge=0, le=10, description="Overall skin health score (0-10)")
    recommendations: Dict[str, List[Recommendation]]
    image_url: Optional[str] = None
    errors: Optional[Dict[str, str]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "analysis_id": "550e8400-e29b-41d4-a716-446655440000",
                "processing_time_seconds": 1.85,
                "feature_count": 118,
                "features": {
                    "face_length": 185.4,
                    "face_width": 142.3,
                    "skin_smoothness": 78.5,
                    "acne_severity_score": 12.3
                },
                "model_results": {
                    "mediapipe": {"status": "success", "latency_ms": 95.2},
                    "claude-api": {"status": "success", "latency_ms": 1823.5}
                },
                "overall_score": 7.8,
                "recommendations": {
                    "products": [],
                    "routines": []
                }
            }
        }
