"""
Image Storage Service - MinIO/S3 integration
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from minio import Minio
from minio.error import S3Error
import os
import uuid
from io import BytesIO
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Image Storage Service", version="1.0.0")

# MinIO configuration
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "facial-analysis-images")
MINIO_USE_SSL = os.getenv("MINIO_USE_SSL", "false").lower() == "true"

# Initialize MinIO client
try:
    minio_client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_USE_SSL
    )

    # Create bucket if it doesn't exist
    if not minio_client.bucket_exists(MINIO_BUCKET):
        minio_client.make_bucket(MINIO_BUCKET)
        logger.info(f"Created bucket: {MINIO_BUCKET}")
    else:
        logger.info(f"Using existing bucket: {MINIO_BUCKET}")

except Exception as e:
    logger.error(f"MinIO initialization failed: {e}")
    minio_client = None


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "image-storage",
        "minio_available": minio_client is not None
    }


@app.post("/store")
async def store_image(image: UploadFile = File(...)):
    """Store image in MinIO"""

    if not minio_client:
        raise HTTPException(status_code=503, detail="Storage not available")

    try:
        # Generate unique filename
        file_id = str(uuid.uuid4())
        extension = image.filename.split('.')[-1] if '.' in image.filename else 'jpg'
        object_name = f"{file_id}.{extension}"

        # Read image data
        image_data = await image.read()
        image_stream = BytesIO(image_data)
        image_size = len(image_data)

        # Upload to MinIO
        minio_client.put_object(
            MINIO_BUCKET,
            object_name,
            image_stream,
            length=image_size,
            content_type=image.content_type or "image/jpeg"
        )

        # Generate URL
        image_url = f"/image/{object_name}"

        logger.info(f"Stored image: {object_name} ({image_size} bytes)")

        return {
            "status": "success",
            "image_id": file_id,
            "image_url": image_url,
            "object_name": object_name,
            "size_bytes": image_size
        }

    except S3Error as e:
        logger.error(f"MinIO error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Storage failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/image/{object_name}")
async def get_image(object_name: str):
    """Retrieve image from MinIO"""

    if not minio_client:
        raise HTTPException(status_code=503, detail="Storage not available")

    try:
        response = minio_client.get_object(MINIO_BUCKET, object_name)
        image_data = response.read()

        from fastapi.responses import Response
        return Response(content=image_data, media_type="image/jpeg")

    except S3Error as e:
        raise HTTPException(status_code=404, detail="Image not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8022)
