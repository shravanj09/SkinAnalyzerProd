"""
Claude API Service
Extracts 13 subjective assessment features using Claude Vision
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
import anthropic
import base64
import os
import json
import logging
from typing import Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Claude API Service", version="1.0.0")

# Initialize Anthropic client
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
if not CLAUDE_API_KEY:
    logger.warning("CLAUDE_API_KEY not set - service will not work")
    client = None
else:
    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    logger.info("Claude API client initialized")


ANALYSIS_PROMPT = """Analyze this facial image and provide precise numerical assessments for skin health features.

Return ONLY a JSON object with these exact keys and numerical values (0-100 scale unless specified):

{
  "wrinkle_depth_severity": <0-10 scale, assess overall wrinkle depth>,
  "fine_lines_presence": <0-100, density of fine lines>,
  "melasma_severity": <0-100, presence of melasma patches>,
  "freckles_count": <0-200, estimated number of freckles>,
  "post_acne_marks_count": <0-100, count of post-acne marks>,
  "uneven_skin_tone": <0-100, skin tone uniformity issue severity>,
  "dark_circle_type": <string: "pigmented", "vascular", "structural", or "mixed">,
  "dark_circle_severity": <0-10 scale>,
  "eye_bags_puffiness": <0-100, puffiness severity>,
  "tear_trough_hollowness": <0-100, hollowness under eyes>,
  "skin_quality_overall": <0-10 scale, overall skin quality>,
  "primary_concerns": <array of top 3-5 skin concerns as strings>,
  "estimated_age_appearance": <20-80, estimated age in years>
}

Be precise and objective. Focus on visible characteristics only."""


@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "service": "claude-api",
        "version": "1.0.0",
        "claude_available": client is not None
    }


@app.post("/extract")
async def extract_features(image: UploadFile = File(...)):
    """Extract 13 subjective features using Claude Vision"""

    if not client:
        raise HTTPException(status_code=503, detail="Claude API key not configured")

    try:
        # Read and encode image
        contents = await image.read()
        image_data = base64.standard_b64encode(contents).decode("utf-8")

        # Determine media type
        media_type = image.content_type or "image/jpeg"

        # Call Claude API
        message = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=4096,
            temperature=0.3,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": ANALYSIS_PROMPT
                        }
                    ],
                }
            ],
        )

        # Extract response
        response_text = message.content[0].text

        logger.info(f"Claude response: {response_text[:200]}...")

        # Parse JSON response - handle markdown code blocks
        import json
        import re

        # Try to extract JSON from markdown code blocks or plain text
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON object directly
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = response_text.strip()

        features = json.loads(json_str)

        # Validate features
        expected_keys = [
            "wrinkle_depth_severity",
            "fine_lines_presence",
            "melasma_severity",
            "freckles_count",
            "post_acne_marks_count",
            "uneven_skin_tone",
            "dark_circle_type",
            "dark_circle_severity",
            "eye_bags_puffiness",
            "tear_trough_hollowness",
            "skin_quality_overall",
            "primary_concerns",
            "estimated_age_appearance"
        ]

        for key in expected_keys:
            if key not in features:
                logger.warning(f"Missing feature: {key}")
                # Provide default value
                features[key] = 0 if key != "primary_concerns" else []

        logger.info(f"Successfully extracted {len(features)} features")

        return {
            "status": "success",
            "features": features,
            "feature_count": len(features)
        }

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Claude response as JSON: {e}")
        raise HTTPException(status_code=500, detail="Invalid response format from Claude")
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)
