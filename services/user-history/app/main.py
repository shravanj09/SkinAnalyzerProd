"""
User & History Service - Track analysis history with face embedding detection
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import uuid
import numpy as np
from typing import Optional, Dict
import cv2

from app.database import UserHistoryDB
from app.face_embedder import FaceEmbedder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
db = UserHistoryDB()
embedder = None  # Initialize after startup


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global embedder
    # Startup
    logger.info("Starting User History Service...")
    await db.initialize()
    logger.info("Initializing Face Embedder (may take 30-60s on first run)...")
    embedder = FaceEmbedder(model_name="Facenet512")
    logger.info("✓ User History Service ready")
    yield
    # Shutdown
    logger.info("Shutting down User History Service")
    from app.database import close_pool
    await close_pool()


app = FastAPI(
    title="User History Service",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "user-history"}


@app.post("/detect-user")
async def detect_user(
    image: UploadFile = File(...),
    similarity_threshold: float = 0.6
):
    """
    Detect if user is new or returning based on face embedding

    Args:
        image: Face image file
        similarity_threshold: Cosine similarity threshold (default 0.6 = 60% match)

    Returns:
        {
            "user_id": str,
            "is_repeat_user": bool,
            "is_new_user": bool,
            "similarity_score": float,
            "visit_count": int,
            "notification": str
        }
    """
    try:
        # Check if embedder is ready
        if embedder is None:
            raise HTTPException(status_code=503, detail="Face embedder not initialized yet. Please wait...")

        # Read image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        # Generate face embedding
        embedding = embedder.generate_embedding(img)

        if embedding is None:
            raise HTTPException(status_code=400, detail="Could not generate face embedding")

        # Search for matching user
        match_result = await db.find_matching_user(embedding, threshold=similarity_threshold)

        if match_result:
            # RETURNING USER
            user_id, similarity = match_result
            user_info = await db.get_user_info(user_id)

            # Add new embedding for this visit
            await db.add_face_embedding(user_id, embedding)

            notification = (
                f"👋 Welcome Back!\n\n"
                f"User ID: {user_id}\n"
                f"Total Visits: {user_info['total_analyses'] + 1}\n"
                f"Last Visit: {user_info['last_analysis_at']}\n"
                f"Match Confidence: {similarity:.1%}"
            )

            logger.info(f"Repeat user detected: {user_id} (similarity: {similarity:.4f})")

            return {
                "user_id": user_id,
                "is_repeat_user": True,
                "is_new_user": False,
                "similarity_score": float(similarity),
                "visit_count": user_info['total_analyses'] + 1,
                "notification": notification,
                "user_info": user_info
            }

        else:
            # NEW USER
            new_user_id = f"USER_{uuid.uuid4().hex[:8].upper()}"

            # Create user
            await db.create_or_get_user(new_user_id, embedding)

            # Store embedding
            await db.add_face_embedding(new_user_id, embedding)

            notification = (
                f"✨ Welcome!\n\n"
                f"New User ID: {new_user_id}\n"
                f"First Visit - Your analysis history starts now!"
            )

            logger.info(f"New user registered: {new_user_id}")

            return {
                "user_id": new_user_id,
                "is_repeat_user": False,
                "is_new_user": True,
                "similarity_score": 0.0,
                "visit_count": 1,
                "notification": notification,
                "user_info": {"user_id": new_user_id, "total_analyses": 0}
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in detect_user: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/results")
async def store_result(result_data: Dict):
    """
    Store analysis result in database

    Expected payload:
    {
        "user_id": str,
        "analysis_id": str,
        "features": dict,
        "overall_score": float,
        "recommendations": dict,
        "processing_time": float,
        "image_url": str (optional),
        "is_repeat_user": bool,
        "similarity_score": float (optional)
    }
    """
    try:
        success = await db.save_analysis_result(
            user_id=result_data["user_id"],
            analysis_id=result_data["analysis_id"],
            features=result_data["features"],
            overall_score=result_data["overall_score"],
            recommendations=result_data["recommendations"],
            processing_time=result_data["processing_time"],
            image_url=result_data.get("image_url"),
            is_repeat_user=result_data.get("is_repeat_user", False),
            similarity_score=result_data.get("similarity_score")
        )

        if success:
            return {
                "status": "success",
                "message": "Analysis result stored",
                "analysis_id": result_data["analysis_id"]
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to store result")

    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing required field: {e}")
    except Exception as e:
        logger.error(f"Error storing result: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history")
async def get_history(user_id: str, limit: int = 10, offset: int = 0):
    """Get user analysis history"""
    try:
        history = await db.get_user_history(user_id, limit, offset)
        return history
    except Exception as e:
        logger.error(f"Error getting history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/{analysis_id}")
async def get_analysis(analysis_id: str):
    """Get specific analysis result"""
    try:
        result = await db.get_analysis_by_id(analysis_id)
        if result:
            return result
        else:
            raise HTTPException(status_code=404, detail="Analysis not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/user/{user_id}")
async def get_user(user_id: str):
    """Get user information"""
    try:
        user_info = await db.get_user_info(user_id)
        if user_info:
            return user_info
        else:
            raise HTTPException(status_code=404, detail="User not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8023)
