"""
Configuration management using Pydantic Settings
"""
from pydantic_settings import BaseSettings
from typing import List
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings"""

    # Application
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "production"
    DEBUG: bool = False

    # API Gateway
    API_GATEWAY_HOST: str = "0.0.0.0"
    API_GATEWAY_PORT: int = 8000

    # Security
    JWT_SECRET: str = "change-this-secret-key-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_HOURS: int = 24

    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://frontend:3000"
    ]

    # Database
    DATABASE_URL: str = "postgresql://postgres:postgres@postgres:5432/facial_analysis"

    # Redis
    REDIS_URL: str = "redis://redis:6379/0"

    # ML Model Services
    MEDIAPIPE_URL: str = "http://mediapipe:8001"
    OPENCV_URL: str = "http://opencv:8003"
    SHIFAA_UNET_URL: str = "http://shifaa-unet:8004"
    FFHQ_WRINKLE_URL: str = "http://ffhq-wrinkle:8005"
    SKIN_TYPE_URL: str = "http://skin-type:8006"
    SPOTS_DETECTION_URL: str = "http://spots-detection:8007"
    ACNE_DETECTION_URL: str = "http://acne-detection:8008"
    SAM_URL: str = "http://sam:8009"
    CLAUDE_API_URL: str = "http://claude-api:8010"
    FACIAL_ALIGNMENT_URL: str = "http://facial-alignment:8025"
    DERM_FOUNDATION_URL: str = "http://derm-foundation:8024"

    # Tier 2 Models (optional)
    UNET_DARK_CIRCLES_URL: str = "http://unet-dark-circles:8011"
    UNET_REDNESS_URL: str = "http://unet-redness:8012"
    EFFICIENTNET_TEXTURE_URL: str = "http://efficientnet-texture:8013"

    # Infrastructure Services
    IMAGE_STORAGE_URL: str = "http://image-storage:8022"
    USER_HISTORY_URL: str = "http://user-history:8023"
    RECOMMENDATION_URL: str = "http://recommendation:8021"

    # Feature Flags
    ENABLE_MEDIAPIPE: bool = True
    ENABLE_OPENCV: bool = True
    ENABLE_SHIFAA_UNET: bool = True
    ENABLE_FFHQ_WRINKLE: bool = True
    ENABLE_SKIN_TYPE: bool = True
    ENABLE_SPOTS_DETECTION: bool = True
    ENABLE_ACNE_DETECTION: bool = True
    ENABLE_SAM: bool = True
    ENABLE_CLAUDE_API: bool = True
    ENABLE_FACIAL_ALIGNMENT: bool = True
    ENABLE_DERM_FOUNDATION: bool = True

    # Tier 2 Feature Flags
    ENABLE_UNET_DARK_CIRCLES: bool = False
    ENABLE_UNET_REDNESS: bool = False
    ENABLE_EFFICIENTNET_TEXTURE: bool = False

    # Performance
    REQUEST_TIMEOUT: int = 30
    MODEL_INFERENCE_TIMEOUT: int = 10
    MAX_PARALLEL_MODELS: int = 10

    # Rate Limiting
    ENABLE_RATE_LIMITING: bool = True
    RATE_LIMIT_PER_MINUTE: int = 20
    RATE_LIMIT_BURST: int = 5

    # Caching
    ENABLE_CACHING: bool = True
    CACHE_TTL_RESULTS: int = 3600

    # Image Processing
    MAX_IMAGE_SIZE_MB: int = 10
    MAX_IMAGE_DIMENSION: int = 4096
    ALLOWED_IMAGE_FORMATS: List[str] = ["jpg", "jpeg", "png", "webp"]

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Global settings instance
settings = get_settings()
