# eval/db_utils.py - Utility functions for database operations
# Used to get a database connection to avoid repeating code for evaluation scripts.
# Also in case we need pooled connections, we can use this utility function to get a connection.

import os
import re
import time
import logging
import json
import psycopg2

logger = logging.getLogger(__name__)

# Exceptions that are typically transient (connection/network) and worth retrying
RETRYABLE_DB_ERRORS = (psycopg2.OperationalError, psycopg2.InterfaceError)


def get_db_conn():
    return psycopg2.connect(os.getenv("DATABASE_URL"))


def retry_db_write(write_fn, *args, max_retries=3, backoff_seconds=1.0, **kwargs):
    """
    Execute a DB write with retries on transient connection errors.

    write_fn(conn, *args, **kwargs) must perform one or more writes and call conn.commit().
    The connection is created and closed by this helper; do not close conn inside write_fn.

    Retries on psycopg2.OperationalError and InterfaceError with exponential backoff.
    """
    last_exc = None
    for attempt in range(max_retries):
        conn = None
        try:
            conn = get_db_conn()
            write_fn(conn, *args, **kwargs)
            return
        except RETRYABLE_DB_ERRORS as e:
            last_exc = e
            if attempt < max_retries - 1:
                delay = backoff_seconds * (2 ** attempt)
                logger.warning(
                    "DB write failed (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1,
                    max_retries,
                    delay,
                    e,
                )
                time.sleep(delay)
            else:
                raise
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass


def _write_prediction(conn, payload: dict):
    """Insert a single prediction row. Used by save_prediction_to_db with retry."""
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO predictions 
           (request_id, model_version, input_data, prediction_prob, prediction_class, latency_ms) 
           VALUES (%s, %s, %s, %s, %s, %s)""",
        (
            payload["request_id"],
            payload["model_version"],
            json.dumps(payload["input_data"]) if isinstance(payload.get("input_data"), dict) else payload.get("input_data"),
            payload["prediction_prob"],
            payload["prediction_class"],
            payload["latency_ms"],
        ),
    )
    conn.commit()


def save_prediction_to_db(payload: dict):
    """
    Persist a prediction to the database with retries on transient errors.
    payload must contain: request_id, model_version, input_data, prediction_prob, prediction_class, latency_ms.
    """
    request_id = payload.get("request_id", "?")
    logger.info("Attempting to save prediction to database for request %s", request_id)
    try:
        retry_db_write(_write_prediction, payload)
        logger.info("SUCCESS: Prediction saved to database for request %s", request_id)
    except Exception as e:
        logger.error("FAILURE: Write request %s failed: %s", request_id, e, exc_info=True)


def get_latest_version(active_only=False) -> str:
    """
    Query database for the latest version string.
    Raises ValueError if no versions exist.
    """
    conn = get_db_conn()
    cur = conn.cursor()
    
    # Get the latest version (by created_at, not just highest version string)
    if active_only:
        cur.execute("""
            SELECT version FROM model_versions 
            WHERE is_active = TRUE
            ORDER BY created_at DESC 
            LIMIT 1
        """)
    else:
        cur.execute("""
            SELECT version FROM model_versions 
            ORDER BY created_at DESC 
            LIMIT 1
        """)
    row = cur.fetchone()
    cur.close()
    conn.close()
    
    if not row:
        raise ValueError("No versions found in the database") # If this is your first run, you need to manually insert a version first
    
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

def get_next_version(increment_type='patch'):
    """
    Get the next semantic version by incrementing the patch version.
    Queries the database for the latest version and increments it.
    Raises ValueError if no versions exist.
    
    Semantic versioning:
    - MAJOR (v1.0.0 -> v2.0.0): Breaking changes, major architecture changes
    - MINOR (v1.0.0 -> v1.1.0): New features, significant improvements  
    - PATCH (v1.0.0 -> v1.0.1): Bug fixes, retraining with same architecture
    
    For automated retraining, we increment PATCH. Major/minor changes should be
    done manually via SQL or by calling increment_version() with 'major'/'minor'.

    Args:
        increment_type: 'major', 'minor', or 'patch'
    
    Returns:
        New, incremented version string (e.g., "v1.0.1" for patch increment)
        Raises ValueError if no versions exist or if the version parsing fails, or if the increment type is invalid
    """
    latest_version = get_latest_version()
    
    if latest_version is None:
        # First model version - no rows exist yet, need to manually insert a preliminary model that is versioned
        raise ValueError("No versions found in the database")
    
    try:
        # Increment by default: patch version for retraining (v1.0.0 -> v1.0.1)
        return increment_version(latest_version, increment_type=increment_type)
    except ValueError as e:
        # Log warning if version parsing fails (import here to avoid circular deps)
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Error parsing version {latest_version}: {e}")
        raise ValueError(f"Error parsing version {latest_version}: {e}")