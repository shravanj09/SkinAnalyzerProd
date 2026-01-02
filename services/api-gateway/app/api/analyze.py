"""Analysis endpoints - Main facial analysis"""
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import logging
from typing import Optional

from app.services.orchestrator import Orchestrator
from app.schemas.analysis import AnalysisResponse, AnalysisRequest
from app.core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_face(
    image: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    user_id: Optional[str] = None
):
    """
    Analyze facial image and extract all features

    **Process:**
    1. Validate and store image
    2. Fan-out to all ML model services (parallel)
    3. Aggregate results
    4. Generate recommendations
    5. Return complete analysis

    **Returns:**
    - 118 features (Tier 1) - immediate
    - 146 features (Tier 2) - if trained models enabled
    - AR overlay data
    - Product recommendations
    - Overall skin health score
    """

    try:
        # Validate image
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Check file size
        contents = await image.read()
        file_size_mb = len(contents) / (1024 * 1024)

        if file_size_mb > settings.MAX_IMAGE_SIZE_MB:
            raise HTTPException(
                status_code=400,
                detail=f"Image too large. Max size: {settings.MAX_IMAGE_SIZE_MB}MB"
            )

        logger.info(f"Received image: {image.filename} ({file_size_mb:.2f}MB)")

        # Initialize orchestrator
        orchestrator = Orchestrator()

        # Process image through all services
        result = await orchestrator.analyze_image(
            image_data=contents,
            filename=image.filename,
            user_id=user_id
        )

        logger.info(f"Analysis complete: {len(result.get('features', {}))} features extracted")

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/analyze/selective")
async def analyze_selective(
    image: UploadFile = File(...),
    models: Optional[str] = None,  # Comma-separated list
    user_id: Optional[str] = None
):
    """
    Analyze with selected models only

    **Example:**
    ```
    models=mediapipe,claude-api,skin-type
    ```

    Useful for:
    - Testing specific models
    - Cost optimization (skip expensive models)
    - Fast analysis (skip slow models like SAM)
    """

    try:
        contents = await image.read()

        # Parse selected models
        selected_models = models.split(',') if models else None

        orchestrator = Orchestrator()
        result = await orchestrator.analyze_image(
            image_data=contents,
            filename=image.filename,
            user_id=user_id,
            selected_models=selected_models
        )

        return result

    except Exception as e:
        logger.error(f"Selective analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analyze/status/{analysis_id}")
async def get_analysis_status(analysis_id: str):
    """
    Get status of analysis (for async processing)

    **States:**
    - pending: Analysis queued
    - processing: ML models running
    - completed: Results ready
    - failed: Error occurred
    """

    # TODO: Implement Redis-based status tracking
    return {
        "analysis_id": analysis_id,
        "status": "not_implemented",
        "message": "Status tracking coming in next version"
    }
