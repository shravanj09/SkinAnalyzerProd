-- Database initialization script for facial analysis platform

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_analysis_at TIMESTAMP,
    total_analyses INTEGER DEFAULT 0
);

-- Analysis results table
CREATE TABLE IF NOT EXISTS analysis_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    image_url TEXT,
    features JSONB NOT NULL,
    overall_score FLOAT,
    recommendations JSONB,
    processing_time_seconds FLOAT
);

-- Create indices
CREATE INDEX IF NOT EXISTS idx_analysis_user_id ON analysis_results(user_id);
CREATE INDEX IF NOT EXISTS idx_analysis_created_at ON analysis_results(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_features_gin ON analysis_results USING gin(features);

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
