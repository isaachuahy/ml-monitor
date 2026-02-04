"""Pytest configuration and shared fixtures."""
import os
import sys
import pytest

# Ensure repo root is on path so "from eval.db_utils import ..." works
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests that require a real database (deselect with -m 'not integration')"
    )


@pytest.fixture(scope="session")
def db_url():
    """DATABASE_URL from env; skip integration tests if not set."""
    url = os.getenv("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL not set; skipping integration test")
    return url
