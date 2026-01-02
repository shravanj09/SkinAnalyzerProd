"""
Orchestrator Service - Coordinates all ML model services
Fans out requests to all models in parallel and aggregates results
"""
import asyncio
import httpx
import logging
from typing import Dict, List, Optional, Any
import time
from io import BytesIO

from app.core.config import settings
from app.utils.feature_mapper import add_csv_aliases, categorize_features
from app.utils.category_hierarchy import build_hierarchical_response

logger = logging.getLogger(__name__)


class Orchestrator:
    """Orchestrates parallel execution of all ML model services"""

    def __init__(self):
        self.timeout = httpx.Timeout(settings.REQUEST_TIMEOUT, connect=5.0)

    async def analyze_image(
        self,
        image_data: bytes,
        filename: str,
        user_id: Optional[str] = None,
        selected_models: Optional[List[str]] = None
    ) -> Dict:
        """
        Main orchestration method - fans out to all ML models

        Args:
            image_data: Image bytes
            filename: Original filename
            user_id: Optional user identifier
            selected_models: Optional list of specific models to use

        Returns:
            Complete analysis results with all features
        """

        start_time = time.time()

        # Define all available models
        all_models = self._get_model_configs()

        # Filter by selected models if specified
        if selected_models:
            all_models = {
                k: v for k, v in all_models.items()
                if k in selected_models
            }

        logger.info(f"Starting analysis with {len(all_models)} models")

        # 1. Detect user (new or returning) using face embedding
        user_info = await self._detect_user(image_data)
        detected_user_id = user_info.get("user_id")
        is_repeat_user = user_info.get("is_repeat_user", False)
        similarity_score = user_info.get("similarity_score", 0.0)
        user_notification = user_info.get("notification", "")

        logger.info(
            f"User detected: {detected_user_id}, "
            f"repeat={is_repeat_user}, similarity={similarity_score:.2f}"
        )

        # 2. Store image
        image_url = None
        try:
            image_url = await self._store_image(image_data, filename)
            logger.info(f"Image stored: {image_url}")
        except Exception as e:
            logger.error(f"Failed to store image: {e}")
            # Continue without stored image URL

        # 3. Fan-out to all ML models in parallel
        tasks = {}
        for model_name, model_config in all_models.items():
            if model_config["enabled"]:
                tasks[model_name] = self._call_model_service(
                    model_name=model_name,
                    url=model_config["url"],
                    image_data=image_data,
                    timeout=model_config.get("timeout", 10)
                )

        # 3. Wait for all results (with timeout)
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        # 4. Aggregate results
        features = {}
        model_results = {}
        errors = {}

        for (model_name, result) in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                logger.error(f"Model {model_name} failed: {result}")
                errors[model_name] = str(result)
                model_results[model_name] = {"status": "error", "error": str(result)}
            else:
                model_results[model_name] = {"status": "success", "latency_ms": result.get("latency_ms", 0)}
                # Merge features
                if "features" in result:
                    features.update(result["features"])

        # Add CSV-compatible aliases to all features
        features = add_csv_aliases(features)

        # Calculate composite health scores (enterprise-grade metrics)
        composite_scores = self._calculate_composite_scores(features)
        features.update(composite_scores)

        # Generate categorized view
        categorized_features = categorize_features(features, use_csv_names=True)

        # Build hierarchical structure (Category → Model → Features)
        hierarchical_features = build_hierarchical_response(features, model_results)
        logger.info(f"Hierarchical features built: {len(hierarchical_features)} categories")
        logger.debug(f"Hierarchical features: {list(hierarchical_features.keys())}")

        # 5. Generate recommendations (if recommendation service available)
        recommendations = await self._get_recommendations(features)

        # 6. Calculate overall score
        overall_score = self._calculate_overall_score(features)

        processing_time = time.time() - start_time
        analysis_id = self._generate_analysis_id()

        # 7. Save results to database (in background)
        await self._save_results_to_db(
            user_id=detected_user_id,
            analysis_id=analysis_id,
            features=features,
            overall_score=overall_score,
            recommendations=recommendations,
            processing_time=processing_time,
            image_url=image_url,
            is_repeat_user=is_repeat_user,
            similarity_score=similarity_score
        )

        return {
            "status": "success",
            "analysis_id": analysis_id,
            "processing_time_seconds": round(processing_time, 2),
            "features": features,
            "feature_count": len(features),
            "categorized_features": categorized_features,
            "hierarchical_features": hierarchical_features,
            "model_results": model_results,
            "overall_score": overall_score,
            "recommendations": recommendations,
            "image_url": image_url,
            "errors": errors if errors else None,
            # User info
            "user_id": detected_user_id,
            "is_repeat_user": is_repeat_user,
            "similarity_score": similarity_score,
            "user_notification": user_notification,
            "visit_count": user_info.get("visit_count", 1)
        }

    def _get_model_configs(self) -> Dict:
        """Get configuration for all ML models"""
        return {
            "mediapipe": {
                "enabled": settings.ENABLE_MEDIAPIPE,
                "url": settings.MEDIAPIPE_URL,
                "timeout": 30  # Increased for mediapipe processing
            },
            "opencv": {
                "enabled": settings.ENABLE_OPENCV,
                "url": settings.OPENCV_URL,
                "timeout": 30  # Increased for opencv processing
            },
            "shifaa-unet": {
                "enabled": settings.ENABLE_SHIFAA_UNET,
                "url": settings.SHIFAA_UNET_URL,
                "timeout": 15
            },
            "ffhq-wrinkle": {
                "enabled": settings.ENABLE_FFHQ_WRINKLE,
                "url": settings.FFHQ_WRINKLE_URL,
                "timeout": 30  # Increased for CV processing
            },
            "skin-type": {
                "enabled": settings.ENABLE_SKIN_TYPE,
                "url": settings.SKIN_TYPE_URL,
                "timeout": 60  # Increased for HuggingFace model download
            },
            "spots-detection": {
                "enabled": settings.ENABLE_SPOTS_DETECTION,
                "url": settings.SPOTS_DETECTION_URL,
                "timeout": 60  # Increased for HuggingFace model
            },
            "acne-detection": {
                "enabled": settings.ENABLE_ACNE_DETECTION,
                "url": settings.ACNE_DETECTION_URL,
                "timeout": 60  # Increased for ResNet-50 model
            },
            "sam": {
                "enabled": settings.ENABLE_SAM,
                "url": settings.SAM_URL,
                "timeout": 40  # SAM can be slow
            },
            "claude-api": {
                "enabled": settings.ENABLE_CLAUDE_API,
                "url": settings.CLAUDE_API_URL,
                "timeout": 30  # Claude API can be slow
            },
            "ml-custom": {
                "enabled": settings.ENABLE_ML_CUSTOM,
                "url": settings.ML_CUSTOM_URL,
                "timeout": 90  # FAN is CPU-intensive, needs 60-90s for 68 landmarks
            },
            "derm-foundation": {
                "enabled": settings.ENABLE_DERM_FOUNDATION,
                "url": settings.DERM_FOUNDATION_URL,
                "timeout": 60  # Transformer model, GPU accelerated
            }
        }

    async def _call_model_service(
        self,
        model_name: str,
        url: str,
        image_data: bytes,
        timeout: int = 10
    ) -> Dict:
        """Call individual ML model service"""

        start_time = time.time()

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                files = {"image": ("image.jpg", image_data, "image/jpeg")}

                response = await client.post(
                    f"{url}/extract",
                    files=files
                )

                response.raise_for_status()

                result = response.json()
                latency_ms = (time.time() - start_time) * 1000

                result["latency_ms"] = latency_ms

                logger.debug(f"{model_name} completed in {latency_ms:.0f}ms")

                return result

        except httpx.TimeoutException:
            logger.warning(f"{model_name} timed out after {timeout}s")
            raise Exception(f"Timeout after {timeout}s")
        except httpx.HTTPStatusError as e:
            logger.error(f"{model_name} HTTP error: {e.response.status_code}")
            raise Exception(f"HTTP {e.response.status_code}")
        except Exception as e:
            logger.error(f"{model_name} failed: {e}")
            raise

    async def _store_image(self, image_data: bytes, filename: str) -> str:
        """Store image using image-storage service"""

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                files = {"image": (filename, image_data, "image/jpeg")}

                response = await client.post(
                    f"{settings.IMAGE_STORAGE_URL}/store",
                    files=files
                )

                response.raise_for_status()
                result = response.json()

                return result.get("image_url")

        except Exception as e:
            logger.error(f"Failed to store image: {e}")
            raise

    async def _get_recommendations(self, features: Dict) -> Dict:
        """Get product recommendations based on features"""

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    f"{settings.RECOMMENDATION_URL}/recommend",
                    json={"features": features}
                )

                if response.status_code == 200:
                    return response.json()

        except Exception as e:
            logger.error(f"Failed to get recommendations: {e}")

        return {"products": [], "routines": []}

    def _calculate_composite_scores(self, features: Dict) -> Dict:
        """
        Calculate composite health scores with enterprise-grade accuracy
        Returns 3 comprehensive health metrics
        """

        def safe_get(key, default=0.0):
            """Safely get feature value"""
            value = features.get(key, default)
            return float(value) if isinstance(value, (int, float)) else default

        # 1. Overall Skin Condition Score (0-100)
        # Combines: texture, tone, clarity, aging signs

        # Positive factors (higher is better)
        skin_smoothness = safe_get("wrinkle_skin_smoothness", 5.0)  # 0-10
        skin_quality = safe_get("skin_quality_overall", 7.0)  # 0-10

        # Negative factors (higher is worse, need to invert)
        acne_severity = safe_get("acne_severity", 0.0)  # 0-10
        wrinkle_severity = safe_get("wrinkle_severity", 0.0)  # 0-10
        spots_severity = safe_get("spots_severity", 0.0)  # 0-10
        dark_circles = safe_get("undereye_darkness_level", 0.0)  # 0-10

        # Calculate overall condition (0-100 scale)
        overall_condition = (
            (skin_smoothness * 15) +  # Smoothness: 15%
            (skin_quality * 15) +  # Quality: 15%
            ((10 - acne_severity) * 20) +  # Acne (inverted): 20%
            ((10 - wrinkle_severity) * 20) +  # Wrinkles (inverted): 20%
            ((10 - spots_severity) * 15) +  # Spots (inverted): 15%
            ((10 - dark_circles) * 15)  # Dark circles (inverted): 15%
        )

        overall_skin_condition_score = max(0, min(100, round(overall_condition, 1)))

        # 2. Skin Maturity Score (0-100)
        # Reflects biological aging vs chronological age

        wrinkle_density = safe_get("wrinkle_density", 0.0)  # 0-100
        wrinkle_depth = safe_get("wrinkle_depth", 0.0)  # 0-10
        fine_lines = safe_get("fine_lines_count", 0.0)  # 0-100
        age_spots = safe_get("age_spots_present", 0.0)  # 0/1
        skin_firmness = safe_get("eyelid_firmness_score", 7.0)  # 0-10

        # Calculate maturity (higher = more mature/aged)
        maturity_score = (
            (wrinkle_density * 0.3) +  # 30%
            (wrinkle_depth * 3.0) +  # 30% (scaled from 0-10 to 0-30)
            (fine_lines * 0.2) +  # 20%
            (age_spots * 10) +  # 10%
            ((10 - skin_firmness) * 1.0)  # 10% (inverted)
        )

        skin_maturity_score = max(0, min(100, round(maturity_score, 1)))

        # 3. Skin Health Index (0-100)
        # Holistic health metric combining multiple dimensions

        # Hydration & moisture
        moisture_level = safe_get("moisture_level", 50.0)  # 0-100
        hydration = safe_get("hydration_uniformity", 50.0)  # 0-100

        # Symmetry & structure
        facial_symmetry = safe_get("left_right_balance_score", 5.0)  # 0-10

        # Skin type factors
        sebum_level = safe_get("sebum_level", 5.0)  # 0-10

        # Pigmentation
        hyperpigmentation = safe_get("hyperpigmentation_level", 0.0)  # 0-10
        uneven_tone = safe_get("uneven_skin_tone", 0.0)  # 0-10

        # Calculate health index
        health_index = (
            (moisture_level * 0.15) +  # 15%
            (hydration * 0.15) +  # 15%
            (facial_symmetry * 10.0) +  # 10% (scaled)
            ((10 - abs(sebum_level - 5)) * 1.0) +  # 10% (optimal at 5)
            ((10 - hyperpigmentation) * 2.0) +  # 20%
            ((10 - uneven_tone) * 2.0) +  # 20%
            (overall_skin_condition_score * 0.1)  # 10%
        )

        skin_health_index = max(0, min(100, round(health_index, 1)))

        return {
            "overall_skin_condition_score": overall_skin_condition_score,
            "skin_maturity_score": skin_maturity_score,
            "skin_health_index": skin_health_index
        }

    def _calculate_overall_score(self, features: Dict) -> float:
        """Calculate overall skin health score from features (legacy, uses new composite)"""

        # Use the new comprehensive health index as the overall score
        # Normalize to 0-10 scale
        health_index = features.get("skin_health_index", 75.0)
        return round(health_index / 10, 1)

    def _generate_analysis_id(self) -> str:
        """Generate unique analysis ID"""
        import uuid
        return str(uuid.uuid4())

    async def _detect_user(self, image_data: bytes) -> Dict:
        """
        Detect user using face embedding
        Returns user info including user_id, is_repeat_user, etc.
        """
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                files = {"image": ("image.jpg", image_data, "image/jpeg")}

                response = await client.post(
                    f"{settings.USER_HISTORY_URL}/detect-user",
                    files=files
                )

                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"User detection failed: {response.status_code}")
                    # Return anonymous user fallback
                    return {
                        "user_id": f"ANON_{uuid.uuid4().hex[:8].upper()}",
                        "is_repeat_user": False,
                        "is_new_user": True,
                        "similarity_score": 0.0,
                        "visit_count": 1,
                        "notification": "Anonymous session"
                    }

        except Exception as e:
            logger.error(f"Error detecting user: {e}")
            # Return anonymous user fallback
            import uuid
            return {
                "user_id": f"ANON_{uuid.uuid4().hex[:8].upper()}",
                "is_repeat_user": False,
                "is_new_user": True,
                "similarity_score": 0.0,
                "visit_count": 1,
                "notification": "Anonymous session"
            }

    async def _save_results_to_db(
        self,
        user_id: str,
        analysis_id: str,
        features: Dict,
        overall_score: float,
        recommendations: Dict,
        processing_time: float,
        image_url: Optional[str] = None,
        is_repeat_user: bool = False,
        similarity_score: Optional[float] = None
    ):
        """Save analysis results to database"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                payload = {
                    "user_id": user_id,
                    "analysis_id": analysis_id,
                    "features": features,
                    "overall_score": overall_score,
                    "recommendations": recommendations,
                    "processing_time": processing_time,
                    "image_url": image_url,
                    "is_repeat_user": is_repeat_user,
                    "similarity_score": similarity_score
                }

                response = await client.post(
                    f"{settings.USER_HISTORY_URL}/results",
                    json=payload
                )

                if response.status_code == 200:
                    logger.info(f"Results saved to database: {analysis_id}")
                else:
                    logger.error(f"Failed to save results: {response.status_code}")

        except Exception as e:
            logger.error(f"Error saving results to database: {e}")


async def check_services_health():
    """Check health of all ML model services on startup"""

    orchestrator = Orchestrator()
    model_configs = orchestrator._get_model_configs()

    logger.info("Checking ML model services health...")

    async with httpx.AsyncClient(timeout=5.0) as client:
        for model_name, config in model_configs.items():
            if not config["enabled"]:
                continue

            try:
                response = await client.get(f"{config['url']}/health")
                if response.status_code == 200:
                    logger.info(f"✓ {model_name} is healthy")
                else:
                    logger.warning(f"✗ {model_name} returned status {response.status_code}")
            except Exception as e:
                logger.warning(f"✗ {model_name} is unavailable: {e}")

    logger.info("Health check complete")
