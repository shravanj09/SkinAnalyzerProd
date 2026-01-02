"""
Complete Feature Schema for Analysis Results
Maps all 118 features to database columns (101 implemented + 17 unimplemented)
"""

# Database column definitions for all features
FEATURE_COLUMNS = {
    # ========================================================================
    # MEDIAPIPE FEATURES (33) - Port 8001 ✓
    # ========================================================================
    "mp_face_width": "FLOAT",
    "mp_face_height": "FLOAT",
    "mp_face_aspect_ratio": "FLOAT",
    "mp_eye_distance": "FLOAT",
    "mp_nose_length": "FLOAT",
    "mp_mouth_width": "FLOAT",
    "mp_forehead_height": "FLOAT",
    "mp_jaw_width": "FLOAT",
    "mp_cheek_width": "FLOAT",
    "mp_facial_symmetry": "FLOAT",
    "mp_left_eye_openness": "FLOAT",
    "mp_right_eye_openness": "FLOAT",
    "mp_eye_symmetry": "FLOAT",
    "mp_lip_fullness": "FLOAT",
    "mp_upper_lip_height": "FLOAT",
    "mp_lower_lip_height": "FLOAT",
    "mp_nose_width": "FLOAT",
    "mp_nose_bridge_width": "FLOAT",
    "mp_nostril_width": "FLOAT",
    "mp_eyebrow_distance": "FLOAT",
    "mp_left_eyebrow_thickness": "FLOAT",
    "mp_right_eyebrow_thickness": "FLOAT",
    "mp_chin_length": "FLOAT",
    "mp_jaw_angle_left": "FLOAT",
    "mp_jaw_angle_right": "FLOAT",
    "mp_cheekbone_prominence": "FLOAT",
    "mp_temple_width": "FLOAT",
    "mp_forehead_width": "FLOAT",
    "mp_face_oval_score": "FLOAT",
    "mp_face_roundness": "FLOAT",
    "mp_face_squareness": "FLOAT",
    "mp_golden_ratio_score": "FLOAT",
    "mp_total_landmarks_detected": "INTEGER",

    # ========================================================================
    # OPENCV FEATURES (6) - Port 8003 ✓
    # ========================================================================
    "cv_moisture_score": "FLOAT",
    "cv_hydration_level": "FLOAT",
    "cv_dryness_score": "FLOAT",
    "cv_oiliness_score": "FLOAT",
    "cv_skin_brightness": "FLOAT",
    "cv_uniformity_score": "FLOAT",

    # ========================================================================
    # CLAUDE API FEATURES (13) - Port 8010 ✓
    # ========================================================================
    "claude_overall_skin_health": "FLOAT",
    "claude_skin_clarity": "FLOAT",
    "claude_skin_tone_evenness": "FLOAT",
    "claude_dark_circles_severity": "FLOAT",
    "claude_puffiness_level": "FLOAT",
    "claude_fine_lines_visibility": "FLOAT",
    "claude_skin_texture_quality": "FLOAT",
    "claude_redness_level": "FLOAT",
    "claude_skin_radiance": "FLOAT",
    "claude_age_estimate": "INTEGER",
    "claude_skin_concerns": "TEXT",  # JSON array
    "claude_skin_recommendations": "TEXT",  # JSON array
    "claude_analysis_confidence": "FLOAT",

    # ========================================================================
    # FFHQ-WRINKLE FEATURES (14) - Port 8005 ✓
    # ========================================================================
    "wrinkle_overall_density": "FLOAT",
    "wrinkle_depth_severity": "FLOAT",
    "wrinkle_texture_roughness": "FLOAT",
    "wrinkle_fine_lines": "FLOAT",
    "wrinkle_deep_lines": "FLOAT",
    "wrinkle_skin_smoothness": "FLOAT",
    "wrinkle_severity_score": "FLOAT",
    "wrinkle_dominant_region": "VARCHAR(50)",
    "wrinkle_forehead": "FLOAT",
    "wrinkle_crow_feet_left": "FLOAT",
    "wrinkle_crow_feet_right": "FLOAT",
    "wrinkle_nasolabial_left": "FLOAT",
    "wrinkle_nasolabial_right": "FLOAT",
    "wrinkle_mouth_area": "FLOAT",

    # ========================================================================
    # SKIN TYPE FEATURES (10) - Port 8006 ✓
    # ========================================================================
    "st_fitzpatrick_type": "INTEGER",
    "st_skin_type_category": "VARCHAR(50)",
    "st_classification_confidence": "FLOAT",
    "st_undertone_type": "VARCHAR(50)",
    "st_brightness": "FLOAT",
    "st_saturation": "FLOAT",
    "st_avg_r": "FLOAT",
    "st_avg_g": "FLOAT",
    "st_avg_b": "FLOAT",
    "st_hsv_hue": "FLOAT",

    # ========================================================================
    # SPOTS DETECTION FEATURES (9) - Port 8007 ✓
    # ========================================================================
    "spots_severity_score": "FLOAT",
    "spots_classification_confidence": "FLOAT",
    "spots_age_spots_detected": "INTEGER",  # 0 or 1
    "spots_hyperpigmentation_level": "FLOAT",
    "spots_lesion_presence": "INTEGER",  # 0 or 1
    "total_spot_count": "INTEGER",
    "spots_forehead": "INTEGER",
    "spots_cheeks": "INTEGER",
    "spots_chin": "INTEGER",

    # ========================================================================
    # ACNE DETECTION FEATURES (9) - Port 8008 ✓
    # ========================================================================
    "acne_severity_score": "FLOAT",
    "acne_count_estimate": "INTEGER",
    "acne_classification_confidence": "FLOAT",
    "acne_forehead": "FLOAT",
    "acne_left_cheek": "FLOAT",
    "acne_right_cheek": "FLOAT",
    "acne_chin": "FLOAT",
    "acne_nose": "FLOAT",
    "acne_inflammation_level": "FLOAT",

    # ========================================================================
    # SAM OILINESS FEATURES (7) - Port 8009 ✓
    # ========================================================================
    "sam_t_zone_oiliness": "FLOAT",
    "sam_sebum_level": "FLOAT",
    "sam_pore_visibility": "FLOAT",
    "sam_forehead_oiliness": "FLOAT",
    "sam_nose_oiliness": "FLOAT",
    "sam_cheek_oiliness": "FLOAT",
    "sam_skin_moisture_type": "VARCHAR(50)",

    # ========================================================================
    # UNIMPLEMENTED FEATURES (17) - Shifaa-UNet (Port 8004) ✗
    # Missing - these will be NULL until alternative models are found
    # ========================================================================
    # Face Shape Analysis (8 features)
    "face_shape_category": "VARCHAR(50)",  # NULL for now
    "face_contour_smoothness": "FLOAT",  # NULL
    "face_width_height_ratio": "FLOAT",  # NULL
    "face_lower_third_ratio": "FLOAT",  # NULL
    "face_middle_third_ratio": "FLOAT",  # NULL
    "face_upper_third_ratio": "FLOAT",  # NULL
    "face_symmetry_score": "FLOAT",  # NULL
    "face_jawline_definition": "FLOAT",  # NULL

    # Eye Anatomy (5 features)
    "eye_shape_category": "VARCHAR(50)",  # NULL
    "eye_size_ratio": "FLOAT",  # NULL
    "eye_canthal_tilt": "FLOAT",  # NULL
    "eye_interpupillary_distance": "FLOAT",  # NULL
    "eye_lid_contour": "FLOAT",  # NULL

    # Eyebrow Analysis (4 features)
    "eyebrow_shape_category": "VARCHAR(50)",  # NULL
    "eyebrow_arch_height": "FLOAT",  # NULL
    "eyebrow_symmetry": "FLOAT",  # NULL
    "eyebrow_fullness_score": "FLOAT",  # NULL
}

# Total: 101 implemented + 17 unimplemented = 118 features

def generate_create_table_sql():
    """Generate SQL CREATE TABLE statement with all feature columns"""
    columns = []

    # Add all feature columns
    for feature_name, column_type in FEATURE_COLUMNS.items():
        columns.append(f"    {feature_name} {column_type} NULL")

    return ",\n".join(columns)


def generate_alter_table_sql():
    """Generate SQL ALTER TABLE statements to add feature columns"""
    statements = []

    for feature_name, column_type in FEATURE_COLUMNS.items():
        statements.append(
            f"ALTER TABLE analysis_results ADD COLUMN IF NOT EXISTS {feature_name} {column_type} NULL;"
        )

    return "\n".join(statements)


# Feature categories for documentation
FEATURE_CATEGORIES = {
    "mediapipe": {
        "count": 33,
        "status": "implemented",
        "port": 8001,
        "features": [k for k in FEATURE_COLUMNS.keys() if k.startswith("mp_")]
    },
    "opencv": {
        "count": 6,
        "status": "implemented",
        "port": 8003,
        "features": [k for k in FEATURE_COLUMNS.keys() if k.startswith("cv_")]
    },
    "claude": {
        "count": 13,
        "status": "implemented",
        "port": 8010,
        "features": [k for k in FEATURE_COLUMNS.keys() if k.startswith("claude_")]
    },
    "wrinkle": {
        "count": 14,
        "status": "implemented",
        "port": 8005,
        "features": [k for k in FEATURE_COLUMNS.keys() if k.startswith("wrinkle_")]
    },
    "skin_type": {
        "count": 10,
        "status": "implemented",
        "port": 8006,
        "features": [k for k in FEATURE_COLUMNS.keys() if k.startswith("st_")]
    },
    "spots": {
        "count": 9,
        "status": "implemented",
        "port": 8007,
        "features": [k for k in FEATURE_COLUMNS.keys() if k.startswith("spots_") or k == "total_spot_count"]
    },
    "acne": {
        "count": 9,
        "status": "implemented",
        "port": 8008,
        "features": [k for k in FEATURE_COLUMNS.keys() if k.startswith("acne_")]
    },
    "sam": {
        "count": 7,
        "status": "implemented",
        "port": 8009,
        "features": [k for k in FEATURE_COLUMNS.keys() if k.startswith("sam_")]
    },
    "shifaa_unet": {
        "count": 17,
        "status": "unimplemented",
        "port": 8004,
        "features": [k for k in FEATURE_COLUMNS.keys() if k.startswith(("face_", "eye_", "eyebrow_"))]
    }
}
