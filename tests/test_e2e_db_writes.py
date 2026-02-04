"""End-to-end / integration tests for DB writes and retry against a real database.

Requires DATABASE_URL and a running Postgres with schema (e.g. docker-compose up -d db,
then run: docker-compose run --rm -e DATABASE_URL=... eval_worker python -m pytest tests/ -v)
Or from host: DATABASE_URL=postgresql://user:pass@localhost:5433/dbname pytest tests/ -v
"""
import os
import random
import uuid
import pytest
import psycopg2


@pytest.fixture(scope="module")
def db_url():
    url = os.getenv("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL not set")
    return url


@pytest.fixture
def request_id():
    return str(uuid.uuid4())


@pytest.fixture
def cleanup_predictions(db_url, request_id):
    """Yield for the test, then delete the prediction row for this request_id."""
    yield
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    cur.execute("DELETE FROM predictions WHERE request_id = %s", (request_id,))
    conn.commit()
    cur.close()
    conn.close()


@pytest.fixture
def drift_metric_cleanup(db_url):
    """Yield a unique metric_value for the drift test; teardown deletes that row from metrics."""
    value = random.random()
    yield value
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM metrics WHERE metric_name = %s AND metric_value = %s",
        ("drift_income_p_value", value),
    )
    conn.commit()
    cur.close()
    conn.close()


@pytest.mark.integration
class TestSavePredictionE2E:
    """Integration tests for save_prediction_to_db with real DB."""

    def test_save_prediction_to_db_writes_row(self, db_url, request_id, cleanup_predictions):
        from eval.db_utils import save_prediction_to_db

        payload = {
            "request_id": request_id,
            "model_version": "v1.0.0",
            "input_data": {"income": 50000, "debt": 10000, "credit_score": 650},
            "prediction_prob": 0.82,
            "prediction_class": 1,
            "latency_ms": 12.5,
        }
        save_prediction_to_db(payload)

        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        cur.execute(
            "SELECT request_id, model_version, prediction_class FROM predictions WHERE request_id = %s",
            (request_id,),
        )
        row = cur.fetchone()
        cur.close()
        conn.close()
        assert row is not None
        assert row[0] == request_id
        assert row[1] == "v1.0.0"
        assert row[2] == 1

    def test_retry_db_write_against_real_db_succeeds(self, db_url, request_id, cleanup_predictions):
        """Retry path: force a transient error on first attempt, then succeed on second."""
        from eval.db_utils import retry_db_write, get_db_conn

        call_count = 0

        def do_write(rid):
            nonlocal call_count
            call_count += 1
            conn = get_db_conn()
            try:
                if call_count == 1:
                    raise psycopg2.OperationalError("simulated connection loss")
                cur = conn.cursor()
                cur.execute(
                    """INSERT INTO predictions 
                       (request_id, model_version, input_data, prediction_prob, prediction_class, latency_ms) 
                       VALUES (%s, %s, %s, %s, %s, %s)""",
                    (rid, "v1.0.0", "{}", 0.5, 0, 10.0),
                )
                conn.commit()
            finally:
                conn.close()

        wrapped = retry_db_write(max_retries=3, backoff_seconds=0.05)(do_write)
        wrapped(request_id)
        assert call_count == 2

        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM predictions WHERE request_id = %s", (request_id,))
        assert cur.fetchone() is not None
        cur.close()
        conn.close()


@pytest.mark.integration
class TestEvalWritesE2E:
    """Integration tests for eval pipeline writes (metrics, drift metric) with retry."""

    def test_drift_metric_write_via_retry(self, db_url, drift_metric_cleanup):
        """Run the same write pattern drift.py uses (retry_db_write + execute_values)."""
        from eval.db_utils import retry_db_write, get_db_conn, get_latest_version
        from psycopg2.extras import execute_values
        from datetime import datetime, timedelta, timezone

        try:
            current_version = get_latest_version()
        except ValueError:
            pytest.skip("No model_versions row (run db init + update_v2)")

        insert_query = """
            INSERT INTO metrics (metric_name, metric_value, model_version, window_start, window_end)
            VALUES %s
        """
        window_end = datetime.now(timezone.utc)
        window_start = window_end - timedelta(hours=1)
        metric_value = drift_metric_cleanup
        rows = [("drift_income_p_value", metric_value, current_version, window_start, window_end)]

        @retry_db_write()
        def write_metric(query, rows):
            conn = get_db_conn()
            try:
                execute_values(conn.cursor(), query, rows)
                conn.commit()
            finally:
                conn.close()

        write_metric(insert_query, rows)

        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        cur.execute(
            "SELECT metric_value FROM metrics WHERE metric_name = %s AND metric_value = %s",
            ("drift_income_p_value", metric_value),
        )
        row = cur.fetchone()
        cur.close()
        conn.close()
        assert row is not None
        assert row[0] == metric_value
