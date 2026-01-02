"""
Feature Name Mapping Utility
Maps service-specific feature names to CSV-standardized names
Allows both naming schemes to coexist
"""

# Complete mapping: service_name -> csv_name
SERVICE_TO_CSV_MAPPING = {
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
    "spots_total_spot_count": "total_spots_count",
    "spots_forehead": "forehead_spots",
    "spots_cheeks": "cheeks_spots",
    "spots_chin": "chin_spots",

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

    # Claude API Service (13 features) - Already mostly CSV-compatible
    # These don't need mapping as they match CSV names
}

# Reverse mapping for bidirectional lookup
CSV_TO_SERVICE_MAPPING = {v: k for k, v in SERVICE_TO_CSV_MAPPING.items()}

# Category definitions matching CSV structure
CSV_CATEGORIES = {
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
        "moisture_type"
    ],
    "Pores": [
        "pore_visibility"
    ],
    "Acne & Inflammation": [
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
        "melasma_severity",
        "freckles_count",
        "post_acne_marks_count",
        "uneven_skin_tone"
    ],
    "Wrinkles & Aging": [
        "wrinkle_density",
        "wrinkle_depth",
        "wrinkle_severity",
        "fine_lines_count",
        "deep_wrinkles_count",
        "fine_lines_presence",
        "forehead_wrinkles",
        "left_crow_feet",
        "right_crow_feet",
        "left_nasolabial",
        "right_nasolabial",
        "mouth_wrinkles",
        "wrinkle_dominant_area"
    ],
    "Texture & Quality": [
        "texture_roughness",
        "skin_smoothness",
        "skin_quality_overall"
    ],
    "Dark Circles & Eye Area": [
        "dark_circle_type",
        "dark_circle_severity",
        "eye_bags_puffiness",
        "tear_trough_hollowness"
    ],
    "Overall Assessment": [
        "estimated_age_appearance",
        "primary_concerns"
    ]
}


def add_csv_aliases(features: dict) -> dict:
    """
    Add CSV-compatible aliases to feature dictionary
    Returns new dict with both service names and CSV names

    Example:
        Input: {"st_fitzpatrick_type": 3}
        Output: {"st_fitzpatrick_type": 3, "fitzpatrick_type": 3}
    """
    result = features.copy()

    for service_name, csv_name in SERVICE_TO_CSV_MAPPING.items():
        if service_name in features:
            # Add CSV alias
            result[csv_name] = features[service_name]

    return result


def get_csv_name(service_name: str) -> str:
    """
    Get CSV-compatible name for a service feature name
    Returns original name if no mapping exists
    """
    return SERVICE_TO_CSV_MAPPING.get(service_name, service_name)


def get_service_name(csv_name: str) -> str:
    """
    Get service feature name from CSV name
    Returns original name if no mapping exists
    """
    return CSV_TO_SERVICE_MAPPING.get(csv_name, csv_name)


def categorize_features(features: dict, use_csv_names: bool = True) -> dict:
    """
    Categorize features according to CSV category structure

    Args:
        features: Dictionary of feature name -> value
        use_csv_names: If True, use CSV names for categorization; else use service names

    Returns:
        Dictionary with structure: {category_name: {feature_name: value}}
    """
    categorized = {category: {} for category in CSV_CATEGORIES.keys()}
    categorized["Uncategorized"] = {}

    for feature_name, feature_value in features.items():
        # Determine which name to use for lookup
        lookup_name = feature_name
        if not use_csv_names and feature_name in CSV_TO_SERVICE_MAPPING:
            lookup_name = CSV_TO_SERVICE_MAPPING[feature_name]
        elif use_csv_names and feature_name in SERVICE_TO_CSV_MAPPING:
            lookup_name = SERVICE_TO_CSV_MAPPING[feature_name]

        # Find category
        found_category = False
        for category, category_features in CSV_CATEGORIES.items():
            if lookup_name in category_features:
                categorized[category][feature_name] = feature_value
                found_category = True
                break

        if not found_category:
            categorized["Uncategorized"][feature_name] = feature_value

    # Remove empty categories
    return {k: v for k, v in categorized.items() if v}


def get_feature_info(feature_name: str) -> dict:
    """
    Get comprehensive information about a feature

    Returns:
        {
            "service_name": str,
            "csv_name": str,
            "category": str,
            "has_mapping": bool
        }
    """
    # Determine if this is a service name or CSV name
    is_service_name = feature_name in SERVICE_TO_CSV_MAPPING
    is_csv_name = feature_name in CSV_TO_SERVICE_MAPPING

    service_name = feature_name if is_service_name else CSV_TO_SERVICE_MAPPING.get(feature_name, feature_name)
    csv_name = feature_name if is_csv_name else SERVICE_TO_CSV_MAPPING.get(feature_name, feature_name)

    # Find category
    category = "Unknown"
    for cat, features in CSV_CATEGORIES.items():
        if csv_name in features:
            category = cat
            break

    return {
        "service_name": service_name,
        "csv_name": csv_name,
        "category": category,
        "has_mapping": is_service_name or is_csv_name
    }
