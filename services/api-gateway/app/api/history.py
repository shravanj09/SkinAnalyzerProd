"""History endpoints - User analysis history"""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional
import logging
import httpx

from app.core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/history")
async def get_analysis_history(
    user_id: str = Query(..., description="User ID to fetch history for"),
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """
    Get user's analysis history

    **Parameters:**
    - user_id: User identifier (required)
    - limit: Number of results (max 100)
    - offset: Pagination offset

    **Returns:**
    List of previous analyses with thumbnails and key metrics
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{settings.USER_HISTORY_URL}/history",
                params={"user_id": user_id, "limit": limit, "offset": offset}
            )

            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Failed to fetch history: {response.text}"
                )

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="History service timeout")
    except httpx.RequestError as e:
        logger.error(f"Error fetching history: {e}")
        raise HTTPException(status_code=503, detail="History service unavailable")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{analysis_id}")
async def get_analysis_result(analysis_id: str):
    """Get specific analysis result by ID"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{settings.USER_HISTORY_URL}/history/{analysis_id}"
            )

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                raise HTTPException(status_code=404, detail="Analysis not found")
            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Failed to fetch analysis: {response.text}"
                )

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="History service timeout")
    except httpx.RequestError as e:
        logger.error(f"Error fetching analysis: {e}")
        raise HTTPException(status_code=503, detail="History service unavailable")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/user/{user_id}")
async def get_user_info(user_id: str):
    """Get user information"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{settings.USER_HISTORY_URL}/user/{user_id}"
            )

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                raise HTTPException(status_code=404, detail="User not found")
            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Failed to fetch user: {response.text}"
                )

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="History service timeout")
    except httpx.RequestError as e:
        logger.error(f"Error fetching user: {e}")
        raise HTTPException(status_code=503, detail="History service unavailable")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/history/{analysis_id}")
async def delete_analysis(analysis_id: str):
    """Delete analysis result (not implemented yet)"""
    raise HTTPException(
        status_code=501,
        detail="Delete functionality coming soon"
    )
