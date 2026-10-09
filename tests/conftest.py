"""
Shared pytest fixtures and configuration.

Sets VESSEL_DB_PATH to a temp file before any module-level import in the
test files happens, so vessel_profile_store.DB_PATH is always a test path.
"""
import os

# Must be set before vessel_profile_store is imported anywhere in the test session.
# Individual tests that need true isolation override this via monkeypatch.
if "VESSEL_DB_PATH" not in os.environ:
    os.environ["VESSEL_DB_PATH"] = ":memory:"
