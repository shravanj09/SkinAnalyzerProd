"""
Enterprise Feature Mapper - ONE unique name per feature
Maps ML service outputs to canonical enterprise feature names
"""

# Canonical feature name mapping: service_output -> enterprise_name
FEATURE_NAME_MAPPING = {
    # Skin Type Service (11 features)
    "st_fitzpatrick_type": "fitzpatrick_type",
    "st_skin_type_category": "skin_type_category",
    "st_classification_confidence": "skin_type_confidence",
    "st_undertone_type": "undertone_type",
    "st_brightness": "skin_brightness",
    "st_saturation": "skin_saturation",
    "st_avg_r": "skin_avg_red",
    "st_avg_g": "skin_avg_green",
    "st_avg_b": "skin_avg_blue",
    "st_hsv_hue": "skin_hue",
    "st_ita_angle": "ita_angle",

    # Acne Detection Service (9 features)
    "acne_severity_score": "acne_severity",
    "acne_count_estimate": "acne_count",
    "acne_classification_confidence": "acne_confidence",
    "acne_forehead": "forehead_acne",
    "acne_left_cheek": "left_cheek_acne",
    "acne_right_cheek": "right_cheek_acne",
    "acne_chin": "chin_acne",
    "acne_nose": "nose_acne",
    "acne_inflammation_level": "inflammation_level",

    # Spots Detection Service (9 features)
    "spots_severity_score": "spots_severity",
    "spots_classification_confidence": "spots_confidence",
    "spots_age_spots_detected": "age_spots_present",
    "spots_hyperpigmentation_level": "hyperpigmentation_level",
    "spots_lesion_presence": "lesion_detected",
    "total_spot_count": "total_spots_count",
    "spots_forehead": "forehead_spots",
    "spots_cheeks": "cheeks_spots",
    "spots_chin": "chin_spots",
    # spots_pigmentation_variance: no mapping, keep as-is

    # SAM Oiliness Service (7 features)
    "sam_t_zone_oiliness": "t_zone_oiliness",
    "sam_sebum_level": "sebum_level",
    "sam_pore_visibility": "pore_visibility",
    "sam_forehead_oiliness": "forehead_oiliness",
    "sam_nose_oiliness": "nose_oiliness",
    "sam_cheek_oiliness": "cheek_oiliness",
    "sam_skin_moisture_type": "moisture_type",

    # FFHQ-Wrinkle Service (14 features)
    "wrinkle_overall_density": "wrinkle_density",
    "wrinkle_depth_severity": "wrinkle_depth",
    "wrinkle_texture_roughness": "texture_roughness",
    "wrinkle_fine_lines": "fine_lines_count",
    "wrinkle_deep_lines": "deep_wrinkles_count",
    "wrinkle_skin_smoothness": "skin_smoothness",
    "wrinkle_severity_score": "wrinkle_severity",
    "wrinkle_dominant_region": "wrinkle_dominant_area",
    "wrinkle_forehead": "forehead_wrinkles",
    "wrinkle_crow_feet_left": "left_crow_feet",
    "wrinkle_crow_feet_right": "right_crow_feet",
    "wrinkle_nasolabial_left": "left_nasolabial",
    "wrinkle_nasolabial_right": "right_nasolabial",
    "wrinkle_mouth_area": "mouth_wrinkles",

    # OpenCV, Shifaa-UNet, Derm-Foundation, MediaPipe:
    # Keep original names (already clean or have necessary prefixes)

    # Claude API: Keep original names (already clean)
    # ML-Custom: Keep original names (already descriptive)
    # API-Gateway composites: Keep original names
}

# Features to REMOVE (extracted by multiple models, keeping only best)
FEATURES_TO_REMOVE = {
    "claude-api": [
        "wrinkle_depth_severity"  # ffhq-wrinkle is more accurate
    ],
    "facial-alignment": [
        "left_eye_aspect_ratio",  # mediapipe is more accurate
        "right_eye_aspect_ratio"  # mediapipe is more accurate
    ]
}


def normalize_feature_names(features: dict, service_name: str = None) -> dict:
    """
    Normalize feature names to enterprise canonical names.
    Removes duplicate features from less accurate models.

    Args:
        features: Raw features from ML service
        service_name: Name of the service (for filtering duplicates)

    Returns:
        Features with enterprise canonical names, duplicates removed
    """
    normalized = {}

    for feature_name, feature_value in features.items():
        # Skip features that should be removed from this service
        if service_name and service_name in FEATURES_TO_REMOVE:
            if feature_name in FEATURES_TO_REMOVE[service_name]:
                continue  # Skip this duplicate feature

        # Rename to canonical name if mapping exists
        canonical_name = FEATURE_NAME_MAPPING.get(feature_name, feature_name)
        normalized[canonical_name] = feature_value

    return normalized


# Export for backwards compatibility
def add_csv_aliases(features: dict, service_name: str = None) -> dict:
    """
    DEPRECATED: Use normalize_feature_names() instead.
    This function is kept for backwards compatibility but now just returns normalized names.
    """
    return normalize_feature_names(features, service_name)


# Category definitions for enterprise features
ENTERPRISE_CATEGORIES = {
    "Skin Type & Color": [
        "fitzpatrick_type",
        "skin_type_category",
        "undertone_type",
        "skin_brightness",
        "skin_saturation",
        "skin_avg_red",
        "skin_avg_green",
        "skin_avg_blue",
        "skin_hue",
        "ita_angle",
        "skin_type_confidence"
    ],
    "Oiliness & Sebum": [
        "t_zone_oiliness",
        "sebum_level",
        "forehead_oiliness",
        "nose_oiliness",
        "cheek_oiliness",
        "moisture_type",
        "pore_visibility"
    ],
    "Acne & Blemishes": [
        "acne_severity",
        "acne_count",
        "acne_confidence",
        "forehead_acne",
        "left_cheek_acne",
        "right_cheek_acne",
        "chin_acne",
        "nose_acne",
        "inflammation_level"
    ],
    "Wrinkles & Aging": [
        "wrinkle_density",
        "wrinkle_depth",
        "fine_lines_count",
        "deep_wrinkles_count",
        "skin_smoothness",
        "wrinkle_severity",
        "wrinkle_dominant_area",
        "forehead_wrinkles",
        "left_crow_feet",
        "right_crow_feet",
        "left_nasolabial",
        "right_nasolabial",
        "mouth_wrinkles",
        "fine_lines_presence"
    ],
    "Spots & Pigmentation": [
        "spots_severity",
        "spots_confidence",
        "age_spots_present",
        "hyperpigmentation_level",
        "lesion_detected",
        "total_spots_count",
        "forehead_spots",
        "cheeks_spots",
        "chin_spots",
        "spots_pigmentation_variance"
    ],
    "Texture & Smoothness": [
        "texture_roughness",
        "skin_smoothness",
        "df_texture_uniformity",
        "df_overall_texture_score"
    ]
}
