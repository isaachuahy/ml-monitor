#!/usr/bin/env python3
"""Run retry logic tests without pytest. Usage: python tests/run_retry_tests.py (from repo root)."""
import os
import sys
from unittest.mock import MagicMock, patch

# Repo root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import psycopg2
from eval import db_utils
from eval.db_utils import retry_db_write


def test_succeeds_first_try():
    mock_conn = MagicMock()
    with patch("eval.db_utils.get_db_conn", return_value=mock_conn), patch("eval.db_utils.time.sleep"):
        call_count = 0

        def do_write(x):
            nonlocal call_count
            call_count += 1
            conn = db_utils.get_db_conn()
            conn.commit()
            conn.close()

        wrapped = retry_db_write()(do_write)
        wrapped(42)
        assert call_count == 1, "do_write should be called once"
        mock_conn.close.assert_called()
    print("  OK test_succeeds_first_try")


def test_succeeds_after_two_failures():
    mock_conn = MagicMock()
    with patch("eval.db_utils.get_db_conn", return_value=mock_conn), patch("eval.db_utils.time.sleep"):
        call_count = 0

        def do_write(x):
            nonlocal call_count
            call_count += 1
            conn = db_utils.get_db_conn()
            conn.close()
            if call_count < 3:
                raise psycopg2.OperationalError("connection lost")
            conn.commit()
            conn.close()

        wrapped = retry_db_write(max_retries=5, backoff_seconds=0.01)(do_write)
        wrapped(1)
        assert call_count == 3, f"do_write should be called 3 times, got {call_count}"
    print("  OK test_succeeds_after_two_failures")


def test_exhausted_retries_raises():
    mock_conn = MagicMock()
    with patch("eval.db_utils.get_db_conn", return_value=mock_conn), patch("eval.db_utils.time.sleep"):
        call_count = 0

        def do_write(x):
            nonlocal call_count
            call_count += 1
            conn = db_utils.get_db_conn()
            conn.close()
            raise psycopg2.OperationalError("connection refused")

        wrapped = retry_db_write(max_retries=3, backoff_seconds=0.01)(do_write)
        try:
            wrapped(1)
            assert False, "should have raised"
        except psycopg2.OperationalError as e:
            assert "connection refused" in str(e)
        assert call_count == 3, f"do_write should be called 3 times, got {call_count}"
    print("  OK test_exhausted_retries_raises")


def test_non_retryable_propagates_immediately():
    mock_conn = MagicMock()
    with patch("eval.db_utils.get_db_conn", return_value=mock_conn), patch("eval.db_utils.time.sleep"):
        call_count = 0

        def do_write(x):
            nonlocal call_count
            call_count += 1
            conn = db_utils.get_db_conn()
            conn.close()
            raise ValueError("not transient")

        wrapped = retry_db_write(max_retries=3)(do_write)
        try:
            wrapped(1)
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
