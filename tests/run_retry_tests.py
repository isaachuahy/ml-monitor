#!/usr/bin/env python3
"""Run retry logic tests without pytest. Usage: python tests/run_retry_tests.py (from repo root)."""
import os
import sys
from unittest.mock import MagicMock, patch

# Repo root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import psycopg2
from eval.db_utils import retry_db_write

def test_succeeds_first_try():
    mock_conn = MagicMock()
    with patch("eval.db_utils.get_db_conn", return_value=mock_conn), patch("eval.db_utils.time.sleep"):
        call_count = 0
        def write_fn(conn, x):
            nonlocal call_count
            call_count += 1
            conn.commit()
        retry_db_write(write_fn, 42)
        assert call_count == 1, "write_fn should be called once"
        mock_conn.close.assert_called_once()
    print("  OK test_succeeds_first_try")

def test_succeeds_after_two_failures():
    mock_conn = MagicMock()
    with patch("eval.db_utils.get_db_conn", return_value=mock_conn), patch("eval.db_utils.time.sleep"):
        call_count = 0
        def write_fn(conn, x):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise psycopg2.OperationalError("connection lost")
            conn.commit()
        retry_db_write(write_fn, 1, max_retries=5, backoff_seconds=0.01)
        assert call_count == 3, f"write_fn should be called 3 times, got {call_count}"
    print("  OK test_succeeds_after_two_failures")

def test_exhausted_retries_raises():
    mock_conn = MagicMock()
    with patch("eval.db_utils.get_db_conn", return_value=mock_conn), patch("eval.db_utils.time.sleep"):
        call_count = 0
        def write_fn(conn, x):
            nonlocal call_count
            call_count += 1
            raise psycopg2.OperationalError("connection refused")
        try:
            retry_db_write(write_fn, 1, max_retries=3, backoff_seconds=0.01)
            assert False, "should have raised"
        except psycopg2.OperationalError as e:
            assert "connection refused" in str(e)
        assert call_count == 3, f"write_fn should be called 3 times, got {call_count}"
    print("  OK test_exhausted_retries_raises")

def test_non_retryable_propagates_immediately():
    mock_conn = MagicMock()
    with patch("eval.db_utils.get_db_conn", return_value=mock_conn), patch("eval.db_utils.time.sleep"):
        call_count = 0
        def write_fn(conn, x):
            nonlocal call_count
            call_count += 1
            raise ValueError("not transient")
        try:
            retry_db_write(write_fn, 1, max_retries=3)
            assert False, "should have raised"
        except ValueError as e:
            assert "not transient" in str(e)
        assert call_count == 1, "no retry for non-retryable"
    print("  OK test_non_retryable_propagates_immediately")

def main():
    print("Running retry_db_write unit tests (no pytest)...")
    test_succeeds_first_try()
    test_succeeds_after_two_failures()
    test_exhausted_retries_raises()
    test_non_retryable_propagates_immediately()
    print("All 4 retry tests passed.")

if __name__ == "__main__":
    main()
