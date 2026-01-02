# Production Database Schema - Skin Analyzer

**Complete database schema with 118 individual feature columns**

## Overview

- **Database:** PostgreSQL 16
- **Total Tables:** 3
- **Total Features:** 118 (101 implemented + 17 unimplemented)
- **Analysis Results Columns:** 129

---

## Tables

### 1. `users` Table
Stores user information with face embedding data.

```sql
CREATE TABLE users (
    id UUID PRIMARY KEY,                    -- Internal UUID
    user_id VARCHAR(50) UNIQUE NOT NULL,    -- User identifier (e.g., "USER_A3F2B891")
    created_at TIMESTAMP,                   -- First seen
    last_analysis_at TIMESTAMP,             -- Last analysis
    total_analyses INTEGER DEFAULT 0,       -- Total analysis count
    avg_embedding FLOAT[]                   -- Average face embedding for matching
);
```

**Indexes:**
- `idx_users_user_id` on `user_id`

---

### 2. `face_embeddings` Table
Stores individual face embeddings for user identification.

```sql
CREATE TABLE face_embeddings (
    id UUID PRIMARY KEY,
    user_id VARCHAR(50) REFERENCES users(user_id),
    embedding FLOAT[512],                   -- FaceNet512 embedding vector
    created_at TIMESTAMP
);
```

**Indexes:**
- `idx_embeddings_user_id` on `user_id`

---

### 3. `analysis_results` Table
**Main table with ALL 118 feature columns + metadata**

#### Metadata Columns (10)
```sql
id UUID PRIMARY KEY
user_id VARCHAR(50)                      -- Links to users table
analysis_id VARCHAR(100) UNIQUE          -- Unique analysis identifier
created_at TIMESTAMP
image_url TEXT                           -- Stored image URL
overall_score FLOAT                      -- Overall skin health score (0-10)
recommendations JSONB                    -- Product recommendations
processing_time_seconds FLOAT
is_repeat_user BOOLEAN
similarity_score FLOAT                   -- Face match confidence (0-1)
features_raw JSONB                       -- Full features backup
```

#### MediaPipe Features (33) ✅ IMPLEMENTED
```sql
mp_face_width FLOAT
mp_face_height FLOAT
mp_face_aspect_ratio FLOAT
mp_eye_distance FLOAT
mp_nose_length FLOAT
mp_mouth_width FLOAT
mp_forehead_height FLOAT
mp_jaw_width FLOAT
mp_cheek_width FLOAT
mp_facial_symmetry FLOAT
mp_left_eye_openness FLOAT
mp_right_eye_openness FLOAT
mp_eye_symmetry FLOAT
mp_lip_fullness FLOAT
mp_upper_lip_height FLOAT
mp_lower_lip_height FLOAT
mp_nose_width FLOAT
mp_nose_bridge_width FLOAT
mp_nostril_width FLOAT
mp_eyebrow_distance FLOAT
mp_left_eyebrow_thickness FLOAT
mp_right_eyebrow_thickness FLOAT
mp_chin_length FLOAT
mp_jaw_angle_left FLOAT
mp_jaw_angle_right FLOAT
mp_cheekbone_prominence FLOAT
mp_temple_width FLOAT
mp_forehead_width FLOAT
mp_face_oval_score FLOAT
mp_face_roundness FLOAT
mp_face_squareness FLOAT
mp_golden_ratio_score FLOAT
mp_total_landmarks_detected INTEGER
```

#### OpenCV Features (6) ✅ IMPLEMENTED
```sql
cv_moisture_score FLOAT
cv_hydration_level FLOAT
cv_dryness_score FLOAT
cv_oiliness_score FLOAT
cv_skin_brightness FLOAT
cv_uniformity_score FLOAT
```

#### Claude API Features (13) ✅ IMPLEMENTED
```sql
claude_overall_skin_health FLOAT
claude_skin_clarity FLOAT
claude_skin_tone_evenness FLOAT
claude_dark_circles_severity FLOAT
claude_puffiness_level FLOAT
claude_fine_lines_visibility FLOAT
claude_skin_texture_quality FLOAT
claude_redness_level FLOAT
claude_skin_radiance FLOAT
claude_age_estimate INTEGER
claude_skin_concerns TEXT                -- JSON array
claude_skin_recommendations TEXT         -- JSON array
claude_analysis_confidence FLOAT
```

#### FFHQ-Wrinkle Features (14) ✅ IMPLEMENTED
```sql
wrinkle_overall_density FLOAT
wrinkle_depth_severity FLOAT
wrinkle_texture_roughness FLOAT
wrinkle_fine_lines FLOAT
wrinkle_deep_lines FLOAT
wrinkle_skin_smoothness FLOAT
wrinkle_severity_score FLOAT
wrinkle_dominant_region VARCHAR(50)
wrinkle_forehead FLOAT
wrinkle_crow_feet_left FLOAT
wrinkle_crow_feet_right FLOAT
wrinkle_nasolabial_left FLOAT
wrinkle_nasolabial_right FLOAT
wrinkle_mouth_area FLOAT
```

#### Skin Type Features (10) ✅ IMPLEMENTED
```sql
st_fitzpatrick_type INTEGER              -- 1-6 scale
st_skin_type_category VARCHAR(50)        -- fair/medium/dark
st_classification_confidence FLOAT
st_undertone_type VARCHAR(50)            -- warm/cool/neutral
st_brightness FLOAT
st_saturation FLOAT
st_avg_r FLOAT
st_avg_g FLOAT
st_avg_b FLOAT
st_hsv_hue FLOAT
```

#### Spots Detection Features (9) ✅ IMPLEMENTED
```sql
spots_severity_score FLOAT
spots_classification_confidence FLOAT
spots_age_spots_detected INTEGER         -- 0 or 1
spots_hyperpigmentation_level FLOAT
spots_lesion_presence INTEGER            -- 0 or 1
total_spot_count INTEGER
spots_forehead INTEGER
spots_cheeks INTEGER
spots_chin INTEGER
```

#### Acne Detection Features (9) ✅ IMPLEMENTED
```sql
acne_severity_score FLOAT
acne_count_estimate INTEGER
acne_classification_confidence FLOAT
acne_forehead FLOAT
acne_left_cheek FLOAT
acne_right_cheek FLOAT
acne_chin FLOAT
acne_nose FLOAT
acne_inflammation_level FLOAT
```

#### SAM Oiliness Features (7) ✅ IMPLEMENTED
```sql
sam_t_zone_oiliness FLOAT
sam_sebum_level FLOAT
sam_pore_visibility FLOAT
sam_forehead_oiliness FLOAT
sam_nose_oiliness FLOAT
sam_cheek_oiliness FLOAT
sam_skin_moisture_type VARCHAR(50)       -- oily/dry/combination
```

#### Unimplemented Features (17) ❌ PLACEHOLDERS
**These columns exist but will be NULL until alternative models are implemented**

##### Face Shape Analysis (8)
```sql
face_shape_category VARCHAR(50)          -- NULL
face_contour_smoothness FLOAT            -- NULL
face_width_height_ratio FLOAT            -- NULL
face_lower_third_ratio FLOAT             -- NULL
face_middle_third_ratio FLOAT            -- NULL
face_upper_third_ratio FLOAT             -- NULL
face_symmetry_score FLOAT                -- NULL
face_jawline_definition FLOAT            -- NULL
```

##### Eye Anatomy (5)
```sql
eye_shape_category VARCHAR(50)           -- NULL
eye_size_ratio FLOAT                     -- NULL
eye_canthal_tilt FLOAT                   -- NULL
eye_interpupillary_distance FLOAT        -- NULL
eye_lid_contour FLOAT                    -- NULL
```

##### Eyebrow Analysis (4)
```sql
eyebrow_shape_category VARCHAR(50)       -- NULL
eyebrow_arch_height FLOAT                -- NULL
eyebrow_symmetry FLOAT                   -- NULL
eyebrow_fullness_score FLOAT             -- NULL
```

**Indexes:**
- `idx_analysis_user_id` on `user_id`
- `idx_analysis_created_at` on `created_at DESC`

---

## Query Examples

### Basic Queries

```sql
-- Count total analyses
SELECT COUNT(*) FROM analysis_results;

-- Get recent analyses
SELECT
    user_id,
    created_at,
    overall_score,
    is_repeat_user,
    similarity_score
FROM analysis_results
ORDER BY created_at DESC
LIMIT 10;

-- Get user statistics
SELECT
    user_id,
    total_analyses,
    created_at as first_seen,
    last_analysis_at
FROM users
ORDER BY total_analyses DESC;
```

### Feature-Specific Queries

```sql
-- Find users with high acne severity
SELECT
    user_id,
    acne_severity_score,
    acne_count_estimate,
    created_at
FROM analysis_results
WHERE acne_severity_score > 7.0
ORDER BY acne_severity_score DESC;

-- Get wrinkle analysis for a user
SELECT
    created_at,
    wrinkle_overall_density,
    wrinkle_depth_severity,
    wrinkle_forehead,
    wrinkle_crow_feet_left,
    wrinkle_crow_feet_right
FROM analysis_results
WHERE user_id = 'USER_A3F2B891'
ORDER BY created_at DESC;

-- Find oily skin types
SELECT
    user_id,
    sam_t_zone_oiliness,
    sam_sebum_level,
    sam_skin_moisture_type
FROM analysis_results
WHERE sam_skin_moisture_type = 'oily'
    AND sam_sebum_level > 7.0;

-- Compare skin type distribution
SELECT
    st_skin_type_category,
    COUNT(*) as count,
    AVG(overall_score) as avg_score
FROM analysis_results
GROUP BY st_skin_type_category
ORDER BY count DESC;

-- Track user progress over time
SELECT
    created_at,
    overall_score,
    acne_severity_score,
    wrinkle_depth_severity,
    spots_severity_score
FROM analysis_results
WHERE user_id = 'USER_A3F2B891'
ORDER BY created_at ASC;
```

### Advanced Analytics

```sql
-- Get comprehensive user profile
SELECT
    u.user_id,
    u.total_analyses,
    u.last_analysis_at,
    a.overall_score,
    a.acne_severity_score,
    a.wrinkle_depth_severity,
    a.spots_severity_score,
    a.sam_sebum_level,
    a.st_skin_type_category
FROM users u
LEFT JOIN analysis_results a ON u.user_id = a.user_id
WHERE a.created_at = (
    SELECT MAX(created_at)
    FROM analysis_results
    WHERE user_id = u.user_id
)
ORDER BY u.total_analyses DESC
LIMIT 10;

-- Feature availability report
SELECT
    'MediaPipe' as service,
    COUNT(*) FILTER (WHERE mp_face_width IS NOT NULL) as populated_count,
    COUNT(*) as total_analyses,
    ROUND(100.0 * COUNT(*) FILTER (WHERE mp_face_width IS NOT NULL) / COUNT(*), 2) as percentage
FROM analysis_results
UNION ALL
SELECT
    'Wrinkle',
    COUNT(*) FILTER (WHERE wrinkle_overall_density IS NOT NULL),
    COUNT(*),
    ROUND(100.0 * COUNT(*) FILTER (WHERE wrinkle_overall_density IS NOT NULL) / COUNT(*), 2)
FROM analysis_results
UNION ALL
SELECT
    'Acne',
    COUNT(*) FILTER (WHERE acne_severity_score IS NOT NULL),
    COUNT(*),
    ROUND(100.0 * COUNT(*) FILTER (WHERE acne_severity_score IS NOT NULL) / COUNT(*), 2)
FROM analysis_results;
```

---

## Data Access via API

### Get User History
```bash
curl "http://localhost:8000/api/v1/history?user_id=USER_A3F2B891&limit=10"
```

### Get Specific Analysis
```bash
curl "http://localhost:8000/api/v1/history/550e8400-e29b-41d4-a716-446655440000"
```

### Get User Info
```bash
curl "http://localhost:8000/api/v1/user/USER_A3F2B891"
```

---

## Summary

✅ **118 Feature Columns** (101 implemented + 17 placeholders)
✅ **Individual columns** for each feature (fully queryable)
✅ **JSONB backup** in `features_raw` column
✅ **Production-ready** with proper indexes
✅ **Automatic save** on every analysis
✅ **Face embedding** user detection
✅ **Complete history** tracking
