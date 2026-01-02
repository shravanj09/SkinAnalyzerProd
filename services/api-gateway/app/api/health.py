"""Health check endpoints"""
from fastapi import APIRouter
from typing import Dict
import httpx
import logging

from app.core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health")
async def health_check() -> Dict:
    """Basic health check"""
    return {
        "status": "healthy",
        "service": "api-gateway",
        "version": settings.VERSION
    }


@router.get("/health/detailed")
async def detailed_health_check() -> Dict:
    """Detailed health check with service status"""

    services = {
        "mediapipe": settings.MEDIAPIPE_URL,
        "opencv": settings.OPENCV_URL,
        "shifaa-unet": settings.SHIFAA_UNET_URL,
        "ffhq-wrinkle": settings.FFHQ_WRINKLE_URL,
        "skin-type": settings.SKIN_TYPE_URL,
        "spots-detection": settings.SPOTS_DETECTION_URL,
        "acne-detection": settings.ACNE_DETECTION_URL,
        "sam": settings.SAM_URL,
        "claude-api": settings.CLAUDE_API_URL,
        "image-storage": settings.IMAGE_STORAGE_URL,
        "user-history": settings.USER_HISTORY_URL,
        "recommendation": settings.RECOMMENDATION_URL,
    }

    service_status = {}

    async with httpx.AsyncClient(timeout=5.0) as client:
        for service_name, url in services.items():
            try:
                response = await client.get(f"{url}/health")
                service_status[service_name] = {
                    "status": "healthy" if response.status_code == 200 else "unhealthy",
                    "latency_ms": response.elapsed.total_seconds() * 1000
                }
            except Exception as e:
                service_status[service_name] = {
                    "status": "unavailable",
                    "error": str(e)
                }

    all_healthy = all(s["status"] == "healthy" for s in service_status.values())

    return {
        "status": "healthy" if all_healthy else "degraded",
        "service": "api-gateway",
        "version": settings.VERSION,
        "services": service_status
    }


@router.get("/ready")
async def readiness_check() -> Dict:
    """Kubernetes readiness probe"""
    # Check if critical services are available
    critical_services = [
        settings.MEDIAPIPE_URL,
        settings.IMAGE_STORAGE_URL
    ]

    async with httpx.AsyncClient(timeout=3.0) as client:
        for url in critical_services:
            try:
                await client.get(f"{url}/health")
            except Exception:
                return {"status": "not_ready"}, 503

    return {"status": "ready"}
