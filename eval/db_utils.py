# eval/db_utils.py - Utility functions for database operations
# Used to get a database connection to avoid repeating code for evaluation scripts.
# Also in case we need pooled connections, we can use this utility function to get a connection.

import os
import re
import psycopg2

def get_db_conn():
    return psycopg2.connect(os.getenv("DATABASE_URL"))

def get_latest_version():
    """
    Query database for the latest version string.
    Returns None if no versions exist.
    """
    conn = get_db_conn()
    cur = conn.cursor()
    
    # Get the latest version (by created_at, not just highest version string)
    cur.execute("""
        SELECT version FROM model_versions 
        ORDER BY created_at DESC 
        LIMIT 1
    """)
    row = cur.fetchone()
    cur.close()
    conn.close()
    
    if not row:
        return None
    
    # fetchone() returns a tuple like ('v1.0.0',) - we need the first element
    # row[0] extracts the version string from the tuple
    return row[0]

def increment_version(version, increment_type='patch'):
    """
    Increment a semantic version string.
    
    Args:
        version: Version string like "v1.2.3"
        increment_type: 'major', 'minor', or 'patch'
    
    Returns:
        New version string (e.g., "v1.2.4" for patch increment)
    """
    match = re.match(r'v(\d+)\.(\d+)\.(\d+)', version)
    if not match:
        raise ValueError(f"Invalid version format: {version}")
    
    major, minor, patch = map(int, match.groups())
    
    if increment_type == 'major':
        return f"v{major + 1}.0.0"
    elif increment_type == 'minor':
        return f"v{major}.{minor + 1}.0"
    elif increment_type == 'patch':
        return f"v{major}.{minor}.{patch + 1}"
    else:
        raise ValueError(f"Invalid increment_type: {increment_type}")

def get_next_version():
    """
    Get the next semantic version by incrementing the patch version.
    Queries the database for the latest version and increments it.
    Returns v1.0.0 if no versions exist.
    
    Semantic versioning:
    - MAJOR (v1.0.0 -> v2.0.0): Breaking changes, major architecture changes
    - MINOR (v1.0.0 -> v1.1.0): New features, significant improvements  
    - PATCH (v1.0.0 -> v1.0.1): Bug fixes, retraining with same architecture
    
    For automated retraining, we increment PATCH. Major/minor changes should be
    done manually via SQL or by calling increment_version() with 'major'/'minor'.
    """
    latest_version = get_latest_version()
    
    if latest_version is None:
        # First model version - no rows exist yet
        return "v1.0.0"
    
    try:
        # Increment patch version for retraining (v1.0.0 -> v1.0.1)
        return increment_version(latest_version, increment_type='patch')
    except ValueError as e:
        # Log warning if version parsing fails (import here to avoid circular deps)
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Error parsing version {latest_version}: {e}, defaulting to v1.0.0")
        return "v1.0.0"