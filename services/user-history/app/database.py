"""
Database connection and models for user history service
"""
import asyncpg
from typing import Optional, List, Dict
import os
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Database connection pool
_pool = None


async def get_pool():
    """Get or create database connection pool"""
    global _pool
    if _pool is None:
        database_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/facial_analysis")
        _pool = await asyncpg.create_pool(database_url, min_size=2, max_size=10)
        logger.info("Database connection pool created")
    return _pool


async def close_pool():
    """Close database connection pool"""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        logger.info("Database connection pool closed")


class UserHistoryDB:
    """Database operations for user history"""

    def __init__(self):
        self.pool = None

    async def initialize(self):
        """Initialize database connection"""
        self.pool = await get_pool()
        await self._create_tables()

    async def _create_tables(self):
        """Create necessary tables if they don't exist"""
        async with self.pool.acquire() as conn:
            # Users table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id VARCHAR(50) UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_analysis_at TIMESTAMP,
                    total_analyses INTEGER DEFAULT 0,
                    avg_embedding FLOAT[] NULL
                )
            """)

            # Face embeddings table for user identification
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS face_embeddings (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id VARCHAR(50) REFERENCES users(user_id) ON DELETE CASCADE,
                    embedding FLOAT[] NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Analysis results table with ALL 118 features as explicit columns
            from app.schema_features import generate_create_table_sql

            feature_columns_sql = generate_create_table_sql()

            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id VARCHAR(50) REFERENCES users(user_id) ON DELETE SET NULL,
                    analysis_id VARCHAR(100) UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    image_url TEXT,
                    overall_score FLOAT,
                    recommendations JSONB,
                    processing_time_seconds FLOAT,
                    is_repeat_user BOOLEAN DEFAULT FALSE,
                    similarity_score FLOAT,

                    -- ALL 118 FEATURE COLUMNS (101 implemented + 17 unimplemented)
{feature_columns_sql},

                    -- Keep JSONB as backup/full dump
                    features_raw JSONB
                )
            """)

            # Create indices
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_users_user_id ON users(user_id)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_embeddings_user_id ON face_embeddings(user_id)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_analysis_user_id ON analysis_results(user_id)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_analysis_created_at ON analysis_results(created_at DESC)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_features_gin ON analysis_results USING gin(features)
            """)

            logger.info("Database tables created/verified")

    async def create_or_get_user(self, user_id: str, embedding: Optional[List[float]] = None) -> Dict:
        """Create new user or get existing user"""
        async with self.pool.acquire() as conn:
            # Check if user exists
            existing = await conn.fetchrow(
                "SELECT * FROM users WHERE user_id = $1",
                user_id
            )

            if existing:
                return dict(existing)

            # Create new user
            result = await conn.fetchrow("""
                INSERT INTO users (user_id, created_at, total_analyses, last_analysis_at, avg_embedding)
                VALUES ($1, $2, 0, NULL, $3)
                RETURNING *
            """, user_id, datetime.utcnow(), embedding)

            logger.info(f"Created new user: {user_id}")
            return dict(result)

    async def add_face_embedding(self, user_id: str, embedding: List[float]) -> bool:
        """Store face embedding for a user"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO face_embeddings (user_id, embedding, created_at)
                VALUES ($1, $2, $3)
            """, user_id, embedding, datetime.utcnow())

            # Update average embedding for faster matching
            avg_embedding = await conn.fetchval("""
                SELECT AVG(embedding) FROM face_embeddings WHERE user_id = $1
            """, user_id)

            await conn.execute("""
                UPDATE users SET avg_embedding = $1 WHERE user_id = $2
            """, avg_embedding, user_id)

            logger.info(f"Stored face embedding for user: {user_id}")
            return True

    async def find_matching_user(self, embedding: List[float], threshold: float = 0.6) -> Optional[tuple]:
        """
        Find matching user by comparing embeddings using cosine similarity
        Returns (user_id, similarity_score) or None
        """
        async with self.pool.acquire() as conn:
            # Calculate cosine similarity for all users with embeddings
            result = await conn.fetch("""
                WITH user_similarities AS (
                    SELECT
                        user_id,
                        (
                            SELECT AVG(
                                (
                                    SELECT SUM(a * b) / (
                                        SQRT(SUM(a * a)) * SQRT(SUM(b * b))
                                    )
                                    FROM UNNEST($1::FLOAT[]) WITH ORDINALITY AS t1(a, ord)
                                    JOIN UNNEST(embedding) WITH ORDINALITY AS t2(b, ord2)
                                        ON t1.ord = t2.ord2
                                )
                            )
                            FROM face_embeddings fe
                            WHERE fe.user_id = u.user_id
                        ) as similarity
                    FROM users u
                    WHERE u.avg_embedding IS NOT NULL
                )
                SELECT user_id, similarity
                FROM user_similarities
                WHERE similarity >= $2
                ORDER BY similarity DESC
                LIMIT 1
            """, embedding, threshold)

            if result:
                return (result[0]['user_id'], float(result[0]['similarity']))
            return None

    async def save_analysis_result(
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
    ) -> bool:
        """
        Save analysis result to database with ALL feature columns
        Maps features dict to individual columns (implemented features)
        Leaves unimplemented features as NULL
        """
        from app.schema_features import FEATURE_COLUMNS

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Build dynamic INSERT query with all feature columns
                column_names = [
                    "user_id", "analysis_id", "created_at", "image_url",
                    "overall_score", "recommendations", "processing_time_seconds",
                    "is_repeat_user", "similarity_score", "features_raw"
                ]

                # Add all feature columns
                column_names.extend(FEATURE_COLUMNS.keys())

                # Build values list
                values = [
                    user_id,
                    analysis_id,
                    datetime.utcnow(),
                    image_url,
                    overall_score,
                    json.dumps(recommendations),
                    processing_time,
                    is_repeat_user,
                    similarity_score,
                    json.dumps(features)  # Store full features as backup
                ]

                # Map features to columns (NULL for missing ones)
                for feature_name in FEATURE_COLUMNS.keys():
                    # Get feature value from features dict, or None if not present
                    feature_value = features.get(feature_name)

                    # Convert to appropriate type
                    if feature_value is not None:
                        # Handle string conversions for specific types
                        column_type = FEATURE_COLUMNS[feature_name]
                        if "VARCHAR" in column_type or "TEXT" in column_type:
                            # Keep as string
                            if isinstance(feature_value, (list, dict)):
                                feature_value = json.dumps(feature_value)
                            else:
                                feature_value = str(feature_value)
                        elif "INTEGER" in column_type:
                            try:
                                feature_value = int(float(feature_value))
                            except (ValueError, TypeError):
                                feature_value = None
                        elif "FLOAT" in column_type:
                            try:
                                feature_value = float(feature_value)
                            except (ValueError, TypeError):
                                feature_value = None

                    values.append(feature_value)

                # Generate placeholders ($1, $2, ...)
                placeholders = ", ".join(f"${i+1}" for i in range(len(values)))

                # Build INSERT query
                columns_str = ", ".join(column_names)
                query = f"""
                    INSERT INTO analysis_results ({columns_str})
                    VALUES ({placeholders})
                """

                await conn.execute(query, *values)

                # Update user stats
                await conn.execute("""
                    UPDATE users
                    SET total_analyses = total_analyses + 1,
                        last_analysis_at = $1
                    WHERE user_id = $2
                """, datetime.utcnow(), user_id)

                logger.info(
                    f"Saved analysis result: {analysis_id} for user: {user_id} "
                    f"with {len([v for v in values[10:] if v is not None])} "
                    f"out of {len(FEATURE_COLUMNS)} features"
                )
                return True

    async def get_user_history(
        self,
        user_id: str,
        limit: int = 10,
        offset: int = 0
    ) -> Dict:
        """Get analysis history for a user"""
        async with self.pool.acquire() as conn:
            # Get total count
            total = await conn.fetchval("""
                SELECT COUNT(*) FROM analysis_results WHERE user_id = $1
            """, user_id)

            # Get results
            results = await conn.fetch("""
                SELECT
                    analysis_id,
                    created_at,
                    image_url,
                    overall_score,
                    processing_time_seconds,
                    is_repeat_user,
                    similarity_score
                FROM analysis_results
                WHERE user_id = $1
                ORDER BY created_at DESC
                LIMIT $2 OFFSET $3
            """, user_id, limit, offset)

            return {
                "user_id": user_id,
                "total": total,
                "results": [dict(r) for r in results]
            }

    async def get_analysis_by_id(self, analysis_id: str) -> Optional[Dict]:
        """
        Get specific analysis result
        Reconstructs features dict from individual columns
        """
        from app.schema_features import FEATURE_COLUMNS

        async with self.pool.acquire() as conn:
            result = await conn.fetchrow("""
                SELECT * FROM analysis_results WHERE analysis_id = $1
            """, analysis_id)

            if result:
                data = dict(result)

                # Parse JSONB fields
                if 'recommendations' in data and data['recommendations']:
                    data['recommendations'] = json.loads(data['recommendations']) if isinstance(data['recommendations'], str) else data['recommendations']

                # Reconstruct features dict from individual columns
                features = {}
                for feature_name in FEATURE_COLUMNS.keys():
                    if feature_name in data and data[feature_name] is not None:
                        features[feature_name] = data[feature_name]
                        # Remove from main data dict (keep data clean)
                        del data[feature_name]

                data['features'] = features

                # Also include features_raw for comparison/debugging
                if 'features_raw' in data and data['features_raw']:
                    data['features_raw_backup'] = json.loads(data['features_raw']) if isinstance(data['features_raw'], str) else data['features_raw']

                return data
            return None

    async def get_user_info(self, user_id: str) -> Optional[Dict]:
        """Get user information"""
        async with self.pool.acquire() as conn:
            user = await conn.fetchrow("""
                SELECT * FROM users WHERE user_id = $1
            """, user_id)

            if user:
                return dict(user)
            return None
