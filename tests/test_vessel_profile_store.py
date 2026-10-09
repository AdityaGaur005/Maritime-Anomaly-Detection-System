"""
Unit tests for vessel_profile_store.py.

Uses an in-memory SQLite database (via VESSEL_DB_PATH=:memory: env var)
so tests are fast, isolated, and leave no files on disk.

IMPORTANT: vessel_profile_store uses a module-level DB_PATH that is set at
import time. We set the env var BEFORE importing the module so tests use the
in-memory database.
"""
import os
import sys
from pathlib import Path

import pytest

# Point the store at an in-memory SQLite DB for tests
os.environ["VESSEL_DB_PATH"] = ":memory:"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mlops"))

# Import AFTER setting env var
import vessel_profile_store as vps
from vessel_profile_store import (
    init_db,
    get_profile,
    maybe_update_profile,
    update_static_attributes,
    MIN_BASELINE_POINTS,
    SCORE_UPDATE_THRESHOLD,
    SAMPLE_SIZE_CAP,
)


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    """
    Give each test its own fresh SQLite file so tests are fully isolated.
    We can't use :memory: easily across connections, so use a temp file instead.
    """
    db_file = tmp_path / "test_profiles.db"
    monkeypatch.setenv("VESSEL_DB_PATH", str(db_file))
    # Patch the module-level DB_PATH so get_db_connection() uses the temp file
    monkeypatch.setattr(vps, "DB_PATH", db_file)
    init_db()
    yield


# ---------------------------------------------------------------------------
# init_db
# ---------------------------------------------------------------------------
def test_init_db_creates_tables():
    """init_db must be idempotent — calling it twice should not raise."""
    init_db()  # second call
    profile = get_profile("999999999")
    assert profile is not None  # fallback profile returned for unknown vessel


# ---------------------------------------------------------------------------
# get_profile — cold-start / fallback behaviour
# ---------------------------------------------------------------------------
def test_get_profile_unknown_vessel_returns_fallback():
    profile = get_profile("000000001")
    assert profile["baseline_is_fallback"] == 1
    assert profile["confidence"] == "low"
    assert profile["point_count"] == 0
    # Population fallback values must be present and numeric
    assert isinstance(profile["speed_mean"], float)
    assert isinstance(profile["lat_centroid"], float)


def test_get_profile_below_min_points_returns_fallback():
    mmsi = "TEST_BELOW_MIN"
    # Add fewer than MIN_BASELINE_POINTS low-score updates
    for i in range(MIN_BASELINE_POINTS - 1):
        maybe_update_profile(mmsi, {
            "speed_over_ground_knots": 10.0 + i * 0.1,
            "heading_change_deg": 5.0,
            "lat": 21.3,
            "lon": -157.8,
        },0.10)
    profile = get_profile(mmsi)
    assert profile["baseline_is_fallback"] == 1
    assert profile["confidence"] == "low"


# ---------------------------------------------------------------------------
# maybe_update_profile — score gate
# ---------------------------------------------------------------------------
def test_high_score_does_not_update_baseline():
    mmsi = "TEST_HIGH_SCORE"
    normal_point = {
        "speed_over_ground_knots": 12.0,
        "heading_change_deg": 3.0,
        "lat": 21.3,
        "lon": -157.8,
    }
    # Attempt update with score above the threshold
    maybe_update_profile(mmsi, normal_point,SCORE_UPDATE_THRESHOLD + 0.01)
    profile = get_profile(mmsi)
    assert profile["point_count"] == 0


def test_low_score_updates_baseline():
    mmsi = "TEST_LOW_SCORE"
    normal_point = {
        "speed_over_ground_knots": 12.0,
        "heading_change_deg": 3.0,
        "lat": 21.3,
        "lon": -157.8,
    }
    maybe_update_profile(mmsi, normal_point,0.10)
    profile = get_profile(mmsi)
    assert profile["point_count"] == 1


def test_speed_and_heading_stored_correctly():
    """The P0 bug: speed_over_ground_knots and heading_change_deg must not be NULL."""
    mmsi = "TEST_KEY_FIX"
    point = {
        "speed_over_ground_knots": 15.5,
        "heading_change_deg": 7.2,
        "lat": 21.3,
        "lon": -157.8,
    }
    maybe_update_profile(mmsi, point,0.05)

    # Query the raw sample table to confirm the values were stored
    import sqlite3
    conn = sqlite3.connect(str(vps.DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "SELECT speed_over_ground_knots, heading_change_deg FROM vessel_sample_points WHERE mmsi = ?",
        (mmsi,),
    )
    row = cursor.fetchone()
    conn.close()

    assert row is not None, "No sample was stored despite low score"
    assert row["speed_over_ground_knots"] is not None, (
        "speed_over_ground_knots is NULL — key mismatch bug not fixed"
    )
    assert row["heading_change_deg"] is not None, (
        "heading_change_deg is NULL — key mismatch bug not fixed"
    )
    assert row["speed_over_ground_knots"] == pytest.approx(15.5)
    assert row["heading_change_deg"] == pytest.approx(7.2)


# ---------------------------------------------------------------------------
# Established baseline
# ---------------------------------------------------------------------------
def test_established_baseline_returns_non_null_speed_mean():
    """
    After MIN_BASELINE_POINTS low-score updates, get_profile must return a
    numeric speed_mean (not None). This was broken by the P0 key mismatch bug.
    """
    mmsi = "TEST_ESTABLISHED"
    for i in range(MIN_BASELINE_POINTS):
        maybe_update_profile(mmsi, {
            "speed_over_ground_knots": 10.0 + i * 0.05,
            "heading_change_deg": 5.0,
            "lat": 21.3 + i * 0.001,
            "lon": -157.8,
        },0.05)

    profile = get_profile(mmsi)
    assert profile["baseline_is_fallback"] == 0
    assert profile["confidence"] == "high"
    assert profile["speed_mean"] is not None, (
        "speed_mean is None after baseline established — P0 bug not fixed"
    )
    assert isinstance(profile["speed_mean"], float)
    assert profile["speed_mean"] > 0


def test_established_baseline_zscore_does_not_crash():
    """
    Verify the arithmetic that crashed before the fix:
    (float - None) would raise TypeError.
    """
    mmsi = "TEST_ZSCORE"
    for i in range(MIN_BASELINE_POINTS):
        maybe_update_profile(mmsi, {
            "speed_over_ground_knots": 12.0,
            "heading_change_deg": 3.0,
            "lat": 21.3,
            "lon": -157.8,
        },0.05)

    profile = get_profile(mmsi)
    # This must not raise TypeError
    speed_zscore = (12.0 - profile["speed_mean"]) / (profile["speed_std"] + 1e-8)
    assert isinstance(speed_zscore, float)


# ---------------------------------------------------------------------------
# SAMPLE_SIZE_CAP eviction
# ---------------------------------------------------------------------------
def test_sample_cap_is_enforced():
    mmsi = "TEST_CAP"
    # Insert more than the cap
    for i in range(SAMPLE_SIZE_CAP + 10):
        maybe_update_profile(mmsi, {
            "speed_over_ground_knots": float(i % 20),
            "heading_change_deg": 5.0,
            "lat": 21.3,
            "lon": -157.8,
        },0.05)

    import sqlite3
    conn = sqlite3.connect(str(vps.DB_PATH))
    cursor = conn.cursor()
    cursor.execute(
        "SELECT COUNT(*) FROM vessel_sample_points WHERE mmsi = ?", (mmsi,)
    )
    count = cursor.fetchone()[0]
    conn.close()
    assert count <= SAMPLE_SIZE_CAP


# ---------------------------------------------------------------------------
# update_static_attributes
# ---------------------------------------------------------------------------
def test_update_static_attributes():
    mmsi = "TEST_STATIC"
    # Create the profile row first by inserting a sample
    maybe_update_profile(mmsi, {
        "speed_over_ground_knots": 10.0,
        "heading_change_deg": 2.0,
        "lat": 21.3,
        "lon": -157.8,
    },0.05)
    update_static_attributes(mmsi, vessel_type=70, length=150.0, width=25.0)
    profile = get_profile(mmsi)
    assert profile["vessel_type_code"] == 70
    assert profile["length_m"] == pytest.approx(150.0)
    assert profile["width_m"] == pytest.approx(25.0)
