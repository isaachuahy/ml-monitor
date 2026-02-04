"""Unit tests for retry_db_write and related DB write behavior."""
import pytest
from unittest.mock import MagicMock, patch
import psycopg2

from eval.db_utils import retry_db_write, save_prediction_to_db

# note that decorator order mirrors parameter order, since patches are applied from bottom to top

class TestRetryDbWrite:
    """Tests for retry_db_write behavior."""

    @patch("eval.db_utils.get_db_conn")
    @patch("eval.db_utils.time.sleep")
    def test_succeeds_first_try(self, mock_sleep, mock_get_conn):
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        call_count = 0

        def write_fn(conn, x):
            nonlocal call_count
            call_count += 1
            conn.commit()

        retry_db_write(write_fn, 42)
        assert call_count == 1
        mock_conn.close.assert_called_once()
        mock_sleep.assert_not_called()

    @patch("eval.db_utils.get_db_conn")
    @patch("eval.db_utils.time.sleep")
    def test_succeeds_after_two_operational_errors(self, mock_sleep, mock_get_conn):
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        call_count = 0

        def write_fn(conn, x):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise psycopg2.OperationalError("connection lost")
            conn.commit()

        retry_db_write(write_fn, 1, max_retries=5, backoff_seconds=0.01)
        assert call_count == 3
        assert mock_sleep.call_count == 2  # slept before 2nd and 3rd attempt
        assert mock_sleep.call_args_list[0][0][0] == pytest.approx(0.01)
        assert mock_sleep.call_args_list[1][0][0] == pytest.approx(0.02)

    @patch("eval.db_utils.get_db_conn")
    @patch("eval.db_utils.time.sleep")
    def test_exhausted_retries_raises(self, mock_sleep, mock_get_conn):
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        call_count = 0

        def write_fn(conn, x):
            nonlocal call_count
            call_count += 1
            raise psycopg2.OperationalError("connection refused")

        with pytest.raises(psycopg2.OperationalError, match="connection refused"):
            retry_db_write(write_fn, 1, max_retries=3, backoff_seconds=0.01)
        assert call_count == 3
        assert mock_sleep.call_count == 2

    @patch("eval.db_utils.get_db_conn")
    @patch("eval.db_utils.time.sleep")
    def test_interface_error_retried_then_succeeds(self, mock_sleep, mock_get_conn):
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        call_count = 0

        def write_fn(conn, x):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise psycopg2.InterfaceError("connection closed")
            conn.commit()

        retry_db_write(write_fn, 1, max_retries=3, backoff_seconds=0.01)
        assert call_count == 2
        mock_sleep.assert_called_once()

    @patch("eval.db_utils.get_db_conn")
    @patch("eval.db_utils.time.sleep")
    def test_non_retryable_exception_propagates_immediately(self, mock_sleep, mock_get_conn):
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        call_count = 0

        def write_fn(conn, x):
            nonlocal call_count
            call_count += 1
            raise ValueError("not a transient error")

        with pytest.raises(ValueError, match="not a transient error"):
            retry_db_write(write_fn, 1, max_retries=3)
        assert call_count == 1
        mock_sleep.assert_not_called()

    @patch("eval.db_utils.get_db_conn")
    @patch("eval.db_utils.time.sleep")
    def test_conn_closed_in_finally_even_on_exception(self, mock_sleep, mock_get_conn):
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn

        def write_fn(conn, x):
            raise psycopg2.OperationalError("fail")

        with pytest.raises(psycopg2.OperationalError):
            retry_db_write(write_fn, 1, max_retries=1)
        mock_conn.close.assert_called()
