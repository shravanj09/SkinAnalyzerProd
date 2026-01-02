"""
Category Hierarchy - Maps 27 categories to models and features
Based on FEATURES_EXTRACTION_COMPLETE-new.csv specification
"""

# Category → Model → Features mapping
CATEGORY_HIERARCHY = {
    "Skin Tone & Type": {
        "total_planned": 11,
        "models": {
            "driboune/skin_type": {
                "service_name": "skin-type",
                "features": [
                    "fitzpatrick_type", "skin_type_category", "undertone_type",
                    "skin_brightness", "skin_saturation", "skin_avg_red",
                    "skin_avg_green", "skin_avg_blue", "skin_hue", "skin_type_confidence",
                    "ita_angle"
                ]
            }
        }
    },

    "Wrinkles & Fine Lines": {
        "total_planned": 12,
        "models": {
            "FFHQ-Wrinkle CV": {
                "service_name": "ffhq-wrinkle",
                "features": [
                    "wrinkle_density", "wrinkle_depth", "wrinkle_severity",
                    "fine_lines_count", "deep_wrinkles_count", "wrinkle_dominant_area",
                    "forehead_wrinkles", "left_crow_feet", "right_crow_feet",
                    "left_nasolabial", "right_nasolabial", "mouth_wrinkles",
                    "texture_roughness", "skin_smoothness"
                ]
            },
            "Claude Vision API": {
                "service_name": "claude-api",
                "features": ["wrinkle_depth_severity", "fine_lines_presence"]
            }
        }
    },

    "Oiliness & Sebum": {
        "total_planned": 7,
        "models": {
            "SAM (Segment Anything)": {
                "service_name": "sam",
                "features": [
                    "t_zone_oiliness", "sebum_level", "forehead_oiliness",
                    "nose_oiliness", "cheek_oiliness", "moisture_type", "pore_visibility"
                ]
            }
        }
    },

    "Acne & Blemishes": {
        "total_planned": 7,
        "models": {
            "imfarzanansari/skintelligent-acne": {
                "service_name": "acne-detection",
                "features": [
                    "acne_severity", "acne_count", "forehead_acne",
                    "left_cheek_acne", "right_cheek_acne", "chin_acne",
                    "nose_acne", "inflammation_level", "acne_confidence"
                ]
            }
        }
    },

    "Spots & Lesions": {
        "total_planned": 6,
        "models": {
            "Anwarkh1/Skin_Cancer": {
                "service_name": "spots-detection",
                "features": [
                    "spots_severity", "age_spots_present", "total_spots_count",
                    "forehead_spots", "cheeks_spots", "chin_spots",
                    "hyperpigmentation_level", "lesion_detected", "spots_confidence"
                ]
            }
        }
    },

    "Pigmentation & Melanin": {
        "total_planned": 15,
        "models": {
            "Google Derm Foundation": {
                "service_name": "derm-foundation",
                "features": [
                    "df_melanin_concentration", "df_hyperpigmentation_level",
                    "df_hypopigmentation_level", "df_melasma_severity", "df_age_spots_count",
                    "df_sun_damage_severity", "df_pigmentation_uniformity",
                    "df_post_inflammatory_hyperpigmentation"
                ]
            },
            "Claude Vision API": {
                "service_name": "claude-api",
                "features": [
                    "melasma_severity", "freckles_count", "post_acne_marks_count",
                    "uneven_skin_tone"
                ]
            },
            "Anwarkh1/Skin_Cancer": {
                "service_name": "spots-detection",
                "features": ["hyperpigmentation_level"]
            }
        }
    },

    "Dark Circles & Under-Eye": {
        "total_planned": 10,
        "models": {
            "Claude Vision API": {
                "service_name": "claude-api",
                "features": [
                    "dark_circle_type", "dark_circle_severity",
                    "eye_bags_puffiness", "tear_trough_hollowness"
                ]
            },
            "ML-Custom (Advanced LAB Analysis)": {
                "service_name": "ml-custom",
                "features": [
                    "hd_dark_circle", "left_eye_darkness", "right_eye_darkness",
                    "darkness_asymmetry", "undereye_darkness_level",
                    "dark_circle_intensity_advanced"
                ]
            }
        }
    },

    "Eye Anatomy": {
        "total_planned": 18,
        "models": {
            "MediaPipe": {
                "service_name": "mediapipe",
                "features": [
                    "left_eye_angle", "left_eye_width", "left_eye_height",
                    "right_eye_angle", "right_eye_width", "right_eye_height"
                ]
            },
            "Ahmed-Selem/Shifaa-UNet": {
                "service_name": "shifaa-unet",
                "features": [
                    "left_eye_shape", "left_eye_size", "right_eye_shape",
                    "right_eye_size", "eye_distance"
                ]
            },
            "ML-Custom (FAN)": {
                "service_name": "ml-custom",
                "features": [
                    "right_eye_aspect_ratio", "left_eye_aspect_ratio",
                    "eye_symmetry_score"
                ]
            }
        }
    },

    "Eyebrow Anatomy": {
        "total_planned": 12,
        "models": {
            "MediaPipe": {
                "service_name": "mediapipe",
                "features": [
                    "left_eyebrow_shape", "left_eyebrow_length",
                    "right_eyebrow_shape", "right_eyebrow_length"
                ]
            },
            "Ahmed-Selem/Shifaa-UNet": {
                "service_name": "shifaa-unet",
                "features": [
                    "left_eyebrow_thickness", "right_eyebrow_thickness",
                    "eyebrow_distance"
                ]
            },
            "ML-Custom (FAN)": {
                "service_name": "ml-custom",
                "features": [
                    "right_eyebrow_height_lfa", "left_eyebrow_height_lfa",
                    "eyebrow_symmetry_score_lfa", "right_eyebrow_arch_lfa",
                    "left_eyebrow_arch_lfa"
                ]
            }
        }
    },

    "Facial Structure & Proportions": {
        "total_planned": 7,
        "models": {
            "MediaPipe": {
                "service_name": "mediapipe",
                "features": [
                    "face_length", "face_width", "facial_symmetry",
                    "eye_spacing"
                ]
            },
            "Ahmed-Selem/Shifaa-UNet": {
                "service_name": "shifaa-unet",
                "features": [
                    "face_shape", "golden_ratio", "jawline_definition"
                ]
            }
        }
    },

    "Facial Symmetry": {
        "total_planned": 10,
        "models": {
            "MediaPipe": {
                "service_name": "mediapipe",
                "features": [
                    "overall_facial_symmetry", "mouth_symmetry",
                    "jawline_symmetry", "forehead_symmetry"
                ]
            },
            "ML-Custom (FAN)": {
                "service_name": "ml-custom",
                "features": [
                    "symmetry_variance", "eyes_symmetry", "eyebrows_symmetry",
                    "cheeks_symmetry", "nose_symmetry", "left_right_balance_score"
                ]
            }
        }
    },

    "Nose Anatomy": {
        "total_planned": 5,
        "models": {
            "MediaPipe": {
                "service_name": "mediapipe",
                "features": ["nose_width", "nose_length", "nose_angle"]
            },
            "ML-Custom (FAN)": {
                "service_name": "ml-custom",
                "features": ["nose_projection_ratio", "nose_straightness_score"]
            }
        }
    },

    "Lip Anatomy": {
        "total_planned": 4,
        "models": {
            "MediaPipe": {
                "service_name": "mediapipe",
                "features": ["lip_fullness", "lip_height", "lip_width"]
            },
            "ML-Custom (FAN)": {
                "service_name": "ml-custom",
                "features": ["lip_color_saturation"]
            }
        }
    },

    "Eyelid & Ptosis": {
        "total_planned": 7,
        "models": {
            "MediaPipe": {
                "service_name": "mediapipe",
                "features": [
                    "upper_eyelid_drooping_right", "upper_eyelid_drooping_left",
                    "lower_eyelid_laxity_right", "lower_eyelid_laxity_left"
                ]
            },
            "Ahmed-Selem/Shifaa-UNet": {
                "service_name": "shifaa-unet",
                "features": ["eyelid_asymmetry_degree", "ptosis_severity"]
            },
            "ML-Custom (FAN)": {
                "service_name": "ml-custom",
                "features": ["eyelid_firmness_score"]
            }
        }
    },

    "Cheekbone Anatomy": {
        "total_planned": 8,
        "models": {
            "Ahmed-Selem/Shifaa-UNet": {
                "service_name": "shifaa-unet",
                "features": [
                    "left_cheekbone_prominence", "left_cheekbone_shape",
                    "right_cheekbone_prominence", "right_cheekbone_shape",
                    "cheekbone_symmetry_score"
                ]
            },
            "ML-Custom (FAN)": {
                "service_name": "ml-custom",
                "features": [
                    "right_cheekbone_prominence_lfa", "left_cheekbone_prominence_lfa",
                    "cheekbone_symmetry_score_lfa"
                ]
            }
        }
    },

    "Skin Health Metrics": {
        "total_planned": 15,
        "models": {
            "Google Derm Foundation": {
                "service_name": "derm-foundation",
                "features": [
                    "df_eczema_indicator", "df_psoriasis_indicator", "df_dermatitis_indicator",
                    "df_skin_quality_overall", "df_skin_health_index", "df_skin_barrier_function",
                    "df_skin_resilience", "df_overall_condition_score"
                ]
            },
            "Claude Vision API": {
                "service_name": "claude-api",
                "features": [
                    "skin_quality_overall", "primary_concerns",
                    "estimated_age_appearance"
                ]
            },
            "API Gateway (Computed)": {
                "service_name": "api-gateway",
                "features": [
                    "overall_skin_condition_score", "skin_maturity_score",
                    "skin_health_index"
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

    # Pending categories (models not yet implemented)
    "Pores": {
        "total_planned": 23,
        "models": {
            "Google Derm Foundation": {
                "service_name": "derm-foundation",
                "features": [
                    "df_pore_visibility", "df_pore_severity_score", "df_pore_count_estimate",
                    "df_pore_size_average", "df_pore_distribution", "df_enlarged_pores_percentage",
                    "df_tzone_pore_density", "df_cheek_pore_density", "df_pore_clogging_level",
                    "df_texture_roughness"
                ]
            },
            "YOLOv8 Pore Detection": {
                "service_name": "pending",
                "features": [
                    "hd_pores_by_region", "hd_pore_count_overall", "hd_pore_avg_size",
                    "hd_pore_prominence_overall", "affected_pore_regions", "pore_severity_score",
                    "pore_count_advanced", "pore_visibility_advanced", "pore_size_variance"
                ]
            },
            "Claude Vision API": {
                "service_name": "claude-api",
                "features": [
                    "pore_size_claude", "pore_prominence_claude",
                    "tzone_pore_severity_claude", "cheek_pore_severity_claude"
                ]
            }
        }
    },

    "Redness & Inflammation": {
        "total_planned": 26,
        "models": {
            "Google Derm Foundation": {
                "service_name": "derm-foundation",
                "features": [
                    "df_redness_intensity", "df_inflammation_score", "df_rosacea_indicator",
                    "df_redness_percentage", "df_redness_uniformity", "df_cheek_redness",
                    "df_nose_redness", "df_forehead_redness", "df_chin_redness",
                    "df_inflammation_type", "df_vascular_visibility", "df_erythema_level"
                ]
            },
            "U-Net Redness": {
                "service_name": "pending",
                "features": [
                    "hd_redness_percentage", "redness_intensity", "redness_by_region",
                    "overall_redness", "high_redness_intensity", "redness_uniformity",
                    "redness_concentration", "cheek_redness_level", "redness_severity_score"
                ]
            },
            "Claude Vision API": {
                "service_name": "claude-api",
                "features": [
                    "rosacea_severity", "flushing_patches", "cheek_redness_level_claude",
                    "nose_redness_level_claude", "inflammation_signs"
                ]
            }
        }
    },

    "Texture & Smoothness": {
        "total_planned": 11,
        "models": {
            "ResNet-50 Texture": {
                "service_name": "pending",
                "features": [
                    "surface_roughness", "skin_smoothness", "texture_variance",
                    "texture_score", "flaking_peeling", "hd_texture_by_region",
                    "hd_texture_overall", "texture_consistency", "skin_roughness_index"
                ]
            },
            "Claude Vision API": {
                "service_name": "claude-api",
                "features": ["texture_roughness_claude", "skin_smoothness_claude"]
            }
        }
    },

    "Moisture & Hydration": {
        "total_planned": 6,
        "models": {
            "OpenCV Texture": {
                "service_name": "opencv",
                "features": [
                    "hd_moisture", "hydration_uniformity", "dehydration_areas",
                    "moisture_level", "moisture_level_indicator", "hydration_level_health"
                ]
            }
        }
    },

    "Firmness & Elasticity": {
        "total_planned": 8,
        "models": {
            "Ahmed-Selem/Shifaa-UNet": {
                "service_name": "shifaa-unet",
                "features": [
                    "jawline_definition_analysis", "facial_sagging_index",
                    "skin_elasticity_score_landmark"
                ]
            },
            "MediaPipe": {
                "service_name": "mediapipe",
                "features": ["skin_firmness_score_landmark"]
            },
            "FaceXLib": {
                "service_name": "pending",
                "features": [
                    "skin_elasticity_advanced", "skin_elasticity_health",
                    "skin_firmness_health", "skin_resilience"
                ]
            }
        }
    },

    "Radiance & Luminosity": {
        "total_planned": 6,
        "models": {
            "OpenCV Custom": {
                "service_name": "pending",
                "features": [
                    "hd_radiance", "glow_coverage", "highlight_intensity",
                    "radiance_uniformity"
                ]
            },
            "Claude Vision API": {
                "service_name": "claude-api",
                "features": ["skin_luminosity", "dullness_severity"]
            }
        }
    },

    "Environmental Damage": {
        "total_planned": 10,
        "models": {
            "Google Derm Foundation": {
                "service_name": "derm-foundation",
                "features": [
                    "df_uv_damage_index", "df_oxidative_damage_index",
                    "df_photoaging_level", "df_pollution_damage", "df_free_radical_damage"
                ]
            },
            "OpenCV Custom": {
                "service_name": "pending",
                "features": [
                    "uv_damage_index", "environmental_stress_level",
                    "oxidative_damage_index", "free_radical_accumulation",
                    "environmental_impact_score"
                ]
            }
        }
    },

    "Color Harmony & Analysis": {
        "total_planned": 9,
        "models": {
            "OpenCV Custom": {
                "service_name": "pending",
                "features": [
                    "eye_color_hex", "eye_color", "lip_color_hex",
                    "eyebrow_color_hex", "hair_color_hex", "hair_color",
                    "color_harmony", "color_depth", "vibrancy_index"
                ]
            }
        }
    },

    "Vascular & Blood Vessels": {
        "total_planned": 3,
        "models": {
            "OpenCV Custom": {
                "service_name": "pending",
                "features": ["vascular_visibility"]
            },
            "Claude Vision API": {
                "service_name": "claude-api",
                "features": ["spider_veins_visible", "vascular_visibility_claude"]
            }
        }
    },

    "Scars & Post-Inflammatory": {
        "total_planned": 1,
        "models": {
            "OpenCV Custom": {
                "service_name": "pending",
                "features": ["scar_visibility_index"]
            }
        }
    }
}


def get_category_info(category_name: str) -> dict:
    """Get information about a category"""
    return CATEGORY_HIERARCHY.get(category_name, {})


def get_all_categories() -> list:
    """Get list of all category names"""
    return list(CATEGORY_HIERARCHY.keys())


def get_model_info(category_name: str, model_name: str) -> dict:
    """Get information about a specific model in a category"""
    category = CATEGORY_HIERARCHY.get(category_name, {})
    models = category.get("models", {})
    return models.get(model_name, {})


def build_hierarchical_response(flat_features: dict, model_results: dict) -> dict:
    """
    Transform flat feature dictionary into hierarchical structure

    Args:
        flat_features: Dictionary of all features (with both service and CSV names)
        model_results: Dictionary of model execution results

    Returns:
        Hierarchical structure organized by category → model → features
    """
    hierarchical = {}

    for category_name, category_data in CATEGORY_HIERARCHY.items():
        total_planned = category_data.get("total_planned", 0)
        models_data = category_data.get("models", {})

        category_info = {
            "total_planned": total_planned,
            "extracted_count": 0,
            "models": {}
        }

        # Process each model in the category
        for model_name, model_config in models_data.items():
            service_name = model_config.get("service_name")
            expected_features = model_config.get("features", [])

            # Get model execution result
            model_status = "pending"
            latency_ms = 0

            if service_name and service_name != "pending":
                model_result = model_results.get(service_name, {})
                model_status = model_result.get("status", "pending")
                latency_ms = model_result.get("latency_ms", 0)

            # Extract features that belong to this model
            extracted_features = {}
            for feature_name in expected_features:
                if feature_name in flat_features:
                    extracted_features[feature_name] = flat_features[feature_name]

            category_info["models"][model_name] = {
                "service_name": service_name,
                "status": model_status,
                "latency_ms": latency_ms,
                "total_features": len(expected_features),
                "extracted_features": extracted_features,
                "extracted_count": len(extracted_features),
                "pending_features": [f for f in expected_features if f not in flat_features]
            }

            category_info["extracted_count"] += len(extracted_features)

        # Determine category status
        if category_info["extracted_count"] == 0:
            category_info["status"] = "pending"
        elif category_info["extracted_count"] >= total_planned:
            category_info["status"] = "complete"
        else:
            category_info["status"] = "partial"

        hierarchical[category_name] = category_info

    return hierarchical
