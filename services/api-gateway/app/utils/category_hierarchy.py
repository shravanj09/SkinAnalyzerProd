"""
Enterprise Category Hierarchy - 184 Unique Features
ONE canonical name per feature, best model per feature
Production-grade feature mapping for SkinAnalyzer
"""

# Category → Model → Features mapping with ENTERPRISE CANONICAL NAMES
CATEGORY_HIERARCHY = {
    "Skin Tone & Type": {
        "total_planned": 11,
        "models": {
            "driboune/skin_type": {
                "service_name": "skin-type",
                "features": [
                    "fitzpatrick_type", "skin_type_category", "skin_type_confidence",
                    "undertone_type", "skin_brightness", "skin_saturation",
                    "skin_avg_red", "skin_avg_green", "skin_avg_blue",
                    "skin_hue", "ita_angle"
                ]
            }
        }
    },

    "Wrinkles & Fine Lines": {
        "total_planned": 15,
        "models": {
            "FFHQ-Wrinkle CV": {
                "service_name": "ffhq-wrinkle",
                "features": [
                    "wrinkle_density", "wrinkle_depth", "texture_roughness",
                    "fine_lines_count", "deep_wrinkles_count", "skin_smoothness",
                    "wrinkle_severity", "wrinkle_dominant_area",
                    "forehead_wrinkles", "left_crow_feet", "right_crow_feet",
                    "left_nasolabial", "right_nasolabial", "mouth_wrinkles"
                ]
            },
            "Claude Vision API": {
                "service_name": "claude-api",
                "features": ["fine_lines_presence"]
            }
        }
    },

    "Oiliness & Sebum": {
        "total_planned": 7,
        "models": {
            "SAM (Segment Anything)": {
                "service_name": "sam",
                "features": [
                    "t_zone_oiliness", "sebum_level", "pore_visibility",
                    "forehead_oiliness", "nose_oiliness",
                    "cheek_oiliness", "moisture_type"
                ]
            }
        }
    },

    "Acne & Blemishes": {
        "total_planned": 9,
        "models": {
            "imfarzanansari/skintelligent-acne": {
                "service_name": "acne-detection",
                "features": [
                    "acne_severity", "acne_count", "acne_confidence",
                    "forehead_acne", "left_cheek_acne", "right_cheek_acne",
                    "chin_acne", "nose_acne", "inflammation_level"
                ]
            }
        }
    },

    "Spots & Lesions": {
        "total_planned": 10,
        "models": {
            "Anwarkh1/Skin_Cancer": {
                "service_name": "spots-detection",
                "features": [
                    "spots_severity", "spots_confidence",
                    "age_spots_present", "hyperpigmentation_level",
                    "lesion_detected", "total_spots_count",
                    "forehead_spots", "cheeks_spots", "chin_spots",
                    "spots_pigmentation_variance"
                ]
            }
        }
    },

    "Pigmentation & Melanin": {
        "total_planned": 12,
        "models": {
            "Google Derm Foundation": {
                "service_name": "derm-foundation",
                "features": [
                    "df_melanin_concentration", "df_pigmentation_uniformity",
                    "df_hyperpigmentation_level", "df_dark_spots_percentage",
                    "df_age_spots_indicator", "df_melasma_severity",
                    "df_uneven_skin_tone", "df_post_inflammatory_marks"
                ]
            },
            "Claude Vision API": {
                "service_name": "claude-api",
                "features": [
                    "melasma_severity", "freckles_count",
                    "post_acne_marks_count", "uneven_skin_tone"
                ]
            }
        }
    },

    "Dark Circles & Under-Eye": {
        "total_planned": 10,
        "models": {
            "ML-Custom FAN": {
                "service_name": "facial-alignment",
                "features": [
                    "hd_dark_circle", "left_eye_darkness", "right_eye_darkness",
                    "darkness_asymmetry", "undereye_darkness_level",
                    "dark_circle_intensity_advanced"
                ]
            },
            "Claude Vision API": {
                "service_name": "claude-api",
                "features": [
                    "dark_circle_type", "dark_circle_severity",
                    "eye_bags_puffiness", "tear_trough_hollowness"
                ]
            }
        }
    },

    "Eye Anatomy": {
        "total_planned": 15,
        "models": {
            "MediaPipe": {
                "service_name": "mediapipe",
                "features": [
                    "left_eye_width", "right_eye_width", "eye_spacing",
                    "left_eye_height", "right_eye_height",
                    "left_eye_aspect_ratio", "right_eye_aspect_ratio",
                    "eye_symmetry"
                ]
            },
            "ML-Custom FAN": {
                "service_name": "facial-alignment",
                "features": ["eye_symmetry_score"]
            },
            "Shifaa-UNet": {
                "service_name": "shifaa-unet",
                "features": [
                    "su_left_eye_size", "su_right_eye_size",
                    "su_left_eye_shape", "su_right_eye_shape",
                    "su_eye_distance", "su_eye_symmetry_score"
                ]
            }
        }
    },

    "Eyebrow Anatomy": {
        "total_planned": 10,
        "models": {
            "MediaPipe": {
                "service_name": "mediapipe",
                "features": [
                    "left_eyebrow_length", "right_eyebrow_length",
                    "eyebrow_spacing", "eyebrow_to_eye_left",
                    "eyebrow_symmetry"
                ]
            },
            "ML-Custom FAN": {
                "service_name": "facial-alignment",
                "features": [
                    "left_eyebrow_height_lfa", "right_eyebrow_height_lfa",
                    "left_eyebrow_arch_lfa", "right_eyebrow_arch_lfa",
                    "eyebrow_symmetry_score_lfa"
                ]
            }
        }
    },

    "Facial Structure & Proportions": {
        "total_planned": 9,
        "models": {
            "MediaPipe": {
                "service_name": "mediapipe",
                "features": [
                    "face_length", "face_width", "facial_ratio",
                    "vertical_thirds_ratio", "horizontal_thirds_ratio"
                ]
            },
            "Shifaa-UNet": {
                "service_name": "shifaa-unet",
                "features": [
                    "su_face_shape", "su_golden_ratio",
                    "su_jawline_definition", "su_nose_projection_ratio"
                ]
            }
        }
    },

    "Facial Symmetry": {
        "total_planned": 11,
        "models": {
            "MediaPipe": {
                "service_name": "mediapipe",
                "features": [
                    "facial_symmetry", "eye_symmetry", "eyebrow_symmetry",
                    "face_symmetry_score", "golden_ratio_approximation"
                ]
            },
            "ML-Custom FAN": {
                "service_name": "facial-alignment",
                "features": [
                    "symmetry_variance", "eyes_symmetry", "eyebrows_symmetry",
                    "cheeks_symmetry", "nose_symmetry", "left_right_balance_score"
                ]
            }
        }
    },

    "Nose Anatomy": {
        "total_planned": 6,
        "models": {
            "MediaPipe": {
                "service_name": "mediapipe",
                "features": [
                    "nose_width", "nose_height", "nose_tip_to_chin"
                ]
            },
            "ML-Custom FAN": {
                "service_name": "facial-alignment",
                "features": [
                    "nose_projection_ratio", "nose_straightness_score"
                ]
            },
            "Shifaa-UNet": {
                "service_name": "shifaa-unet",
                "features": [
                    "su_nose_straightness_score"
                ]
            }
        }
    },

    "Lip Anatomy": {
        "total_planned": 4,
        "models": {
            "MediaPipe": {
                "service_name": "mediapipe",
                "features": [
                    "mouth_width", "lip_height", "mouth_to_chin"
                ]
            },
            "ML-Custom FAN": {
                "service_name": "facial-alignment",
                "features": ["lip_color_saturation"]
            }
        }
    },

    "Eyelid & Ptosis": {
        "total_planned": 1,
        "models": {
            "ML-Custom FAN": {
                "service_name": "facial-alignment",
                "features": ["eyelid_firmness_score"]
            }
        }
    },

    "Cheekbone Anatomy": {
        "total_planned": 7,
        "models": {
            "MediaPipe": {
                "service_name": "mediapipe",
                "features": [
                    "cheekbone_width", "cheekbone_prominence_left",
                    "cheekbone_prominence_right", "cheekbone_height_left"
                ]
            },
            "ML-Custom FAN": {
                "service_name": "facial-alignment",
                "features": [
                    "left_cheekbone_prominence_lfa", "right_cheekbone_prominence_lfa",
                    "cheekbone_symmetry_score_lfa"
                ]
            }
        }
    },

    "Skin Health Metrics": {
        "total_planned": 11,
        "models": {
            "Google Derm Foundation": {
                "service_name": "derm-foundation",
                "features": [
                    "df_skin_quality_overall", "df_skin_health_index",
                    "df_skin_maturity_score", "df_skin_radiance", "df_dullness_severity"
                ]
            },
            "API Gateway Composites": {
                "service_name": "api-gateway",
                "features": [
                    "overall_skin_condition_score", "skin_maturity_score", "skin_health_index"
                ]
            },
            "Claude Vision API": {
                "service_name": "claude-api",
                "features": [
                    "skin_quality_overall", "estimated_age_appearance", "primary_concerns"
                ]
            }
        }
    },

    "Aging & Appearance": {
        "total_planned": 1,
        "models": {
            "Claude Vision API": {
                "service_name": "claude-api",
                "features": ["estimated_age_appearance"]
            }
        }
    },

    "Pores": {
        "total_planned": 7,
        "models": {
            "Google Derm Foundation": {
                "service_name": "derm-foundation",
                "features": [
                    "df_pore_visibility", "df_pore_severity_score", "df_pore_count_estimate",
                    "df_pore_avg_size", "df_tzone_pore_severity", "df_cheek_pore_severity"
                ]
            },
            "SAM (Segment Anything)": {
                "service_name": "sam",
                "features": ["pore_visibility"]
            }
        }
    },

    "Redness & Inflammation": {
        "total_planned": 15,
        "models": {
            "Google Derm Foundation": {
                "service_name": "derm-foundation",
                "features": [
                    "df_redness_intensity", "df_redness_percentage", "df_redness_severity_score",
                    "df_cheek_redness_level", "df_nose_redness_level", "df_overall_redness",
                    "df_inflammation_score", "df_inflammation_signs", "df_redness_uniformity",
                    "df_flushing_severity", "df_rosacea_indicators", "df_vascular_visibility",
                    "df_eczema_indicator", "df_psoriasis_indicator", "df_skin_condition_severity"
                ]
            }
        }
    },

    "Texture & Smoothness": {
        "total_planned": 6,
        "models": {
            "Google Derm Foundation": {
                "service_name": "derm-foundation",
                "features": [
                    "df_skin_smoothness", "df_texture_roughness", "df_texture_uniformity",
                    "df_overall_texture_score"
                ]
            },
            "FFHQ-Wrinkle CV": {
                "service_name": "ffhq-wrinkle",
                "features": [
                    "texture_roughness", "skin_smoothness"
                ]
            }
        }
    },

    "Moisture & Hydration": {
        "total_planned": 6,
        "models": {
            "OpenCV Custom": {
                "service_name": "opencv",
                "features": [
                    "hd_moisture", "hydration_uniformity", "dehydration_areas",
                    "moisture_level", "moisture_level_indicator", "hydration_level_health"
                ]
            }
        }
    },

    "Firmness & Elasticity": {
        "total_planned": 1,
        "models": {
            "ML-Custom FAN": {
                "service_name": "facial-alignment",
                "features": ["eyelid_firmness_score"]
            }
        }
    },

    "Radiance & Luminosity": {
        "total_planned": 2,
        "models": {
            "Google Derm Foundation": {
                "service_name": "derm-foundation",
                "features": ["df_skin_radiance", "df_dullness_severity"]
            }
        }
    },

    "Environmental Damage": {
        "total_planned": 5,
        "models": {
            "Google Derm Foundation": {
                "service_name": "derm-foundation",
                "features": [
                    "df_uv_damage_index", "df_environmental_stress_level",
                    "df_oxidative_damage_index", "df_free_radical_accumulation",
                    "df_environmental_impact_score"
                ]
            }
        }
    },

    "Color Harmony & Analysis": {
        "total_planned": 8,
        "models": {
            "driboune/skin_type": {
                "service_name": "skin-type",
                "features": [
                    "skin_brightness", "skin_saturation", "skin_avg_red",
                    "skin_avg_green", "skin_avg_blue", "skin_hue"
                ]
            },
            "MediaPipe": {
                "service_name": "mediapipe",
                "features": ["eye_to_eyebrow_ratio", "nose_to_mouth_ratio"]
            }
        }
    },

    "Vascular & Blood Vessels": {
        "total_planned": 1,
        "models": {
            "Google Derm Foundation": {
                "service_name": "derm-foundation",
                "features": ["df_vascular_visibility"]
            }
        }
    },

    "Scars & Post-Inflammatory": {
        "total_planned": 1,
        "models": {
            "Google Derm Foundation": {
                "service_name": "derm-foundation",
                "features": ["df_post_inflammatory_marks"]
            }
        }
    }
}


def get_category_info(category_name: str) -> dict:
    """Get information about a category"""
    return CATEGORY_HIERARCHY.get(category_name, {})


def get_all_features() -> list:
    """Get all unique enterprise feature names"""
    features = set()
    for category_data in CATEGORY_HIERARCHY.values():
        for model_config in category_data.get('models', {}).values():
            features.update(model_config.get('features', []))
    return sorted(list(features))


def build_hierarchical_response(flat_features: dict, model_results: dict) -> dict:
    """
    Build hierarchical category response from flat features

    Args:
        flat_features: Flat dictionary of all features
        model_results: Dictionary of model execution status

    Returns:
        Hierarchical structure: {category: {models, features, status, counts}}
    """
    hierarchical = {}

    for category_name, category_data in CATEGORY_HIERARCHY.items():
        total_planned = category_data.get("total_planned", 0)
        models_info = category_data.get("models", {})

        extracted_features = {}
        expected_features = set()

        # Build per-model feature data
        models_with_features = {}
        for model_name, model_config in models_info.items():
            expected_model_features = model_config.get("features", [])
            expected_features.update(expected_model_features)

            # Extract features for THIS model
            model_extracted_features = {}
            for feature_name in expected_model_features:
                if feature_name in flat_features:
                    model_extracted_features[feature_name] = flat_features[feature_name]

            # Build model data with extracted features
            models_with_features[model_name] = {
                "service_name": model_config.get("service_name", "pending"),
                "features": expected_model_features,
                "extracted_features": model_extracted_features,
                "extracted_count": len(model_extracted_features),
                "total_features": len(expected_model_features),
                "pending_features": [f for f in expected_model_features if f not in model_extracted_features],
                "status": "success" if len(model_extracted_features) > 0 else "pending"
            }

        # Extract features that exist in flat_features
        for feature_name in expected_features:
            if feature_name in flat_features:
                extracted_features[feature_name] = flat_features[feature_name]

        extracted_count = len(extracted_features)

        # Determine status
        if extracted_count == 0:
            status = "pending"
        elif extracted_count >= total_planned:
            status = "complete"
        else:
            status = "partial"

        hierarchical[category_name] = {
            "models": models_with_features,
            "features": extracted_features,
            "extracted_count": extracted_count,
            "total_planned": total_planned,
            "status": status,
            "percentage": round((extracted_count / total_planned * 100) if total_planned > 0 else 0, 1)
        }

    return hierarchical
