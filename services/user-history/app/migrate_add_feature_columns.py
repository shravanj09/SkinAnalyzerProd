"""
Database Migration Script
Adds all 118 individual feature columns to existing analysis_results table
Run this once to migrate from JSONB-only schema to column-based schema
"""
import asyncio
import asyncpg
import os
from schema_features import FEATURE_COLUMNS


async def migrate_database():
    """Add all feature columns to analysis_results table"""

    database_url = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/facial_analysis"
    )

    print("Connecting to database...")
    conn = await asyncpg.connect(database_url)

    try:
        print(f"\nAdding {len(FEATURE_COLUMNS)} feature columns...")

        # Check if table exists
        table_exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name = 'analysis_results'
            )
        """)

        if not table_exists:
            print("❌ Table 'analysis_results' does not exist yet")
            print("   Run the application first to create tables")
            return

        # Add each feature column
        added_count = 0
        already_exists_count = 0

        for feature_name, column_type in FEATURE_COLUMNS.items():
            # Check if column exists
            column_exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.columns
                    WHERE table_schema = 'public'
                    AND table_name = 'analysis_results'
                    AND column_name = $1
                )
            """, feature_name)

            if column_exists:
                already_exists_count += 1
                print(f"  ✓ {feature_name} (already exists)")
            else:
                # Add column
                await conn.execute(f"""
                    ALTER TABLE analysis_results
                    ADD COLUMN {feature_name} {column_type} NULL
                """)
                added_count += 1
                print(f"  + {feature_name} ({column_type})")

        # Also add features_raw if not exists
        features_raw_exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.columns
                WHERE table_schema = 'public'
                AND table_name = 'analysis_results'
                AND column_name = 'features_raw'
            )
        """)

        if not features_raw_exists:
            await conn.execute("""
                ALTER TABLE analysis_results
                ADD COLUMN features_raw JSONB
            """)
            print("  + features_raw (JSONB)")
            added_count += 1

        print(f"\n✅ Migration complete!")
        print(f"   Added: {added_count} new columns")
        print(f"   Already existed: {already_exists_count} columns")
        print(f"   Total feature columns: {len(FEATURE_COLUMNS)}")

        # Show table info
        total_columns = await conn.fetchval("""
            SELECT COUNT(*)
            FROM information_schema.columns
            WHERE table_schema = 'public'
            AND table_name = 'analysis_results'
        """)
        print(f"   Total columns in table: {total_columns}")

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(migrate_database())
