"""
Unit tests for kinematic helper functions and delta computation logic.

Pure-math helpers (_clamp, _angular_diff_deg, _parse_timestamp) are tested
via the lightweight kinematics.py module — no torch/xgboost/geopy needed.

Delta-computation tests (_compute_deltas) require feature_factory which loads
all heavy deps; those tests are skipped if torch is not installed.
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

MLOPS_DIR = Path(__file__).resolve().parent.parent / "mlops"
sys.path.insert(0, str(MLOPS_DIR))

from kinematics import angular_diff_deg as _angular_diff_deg
from kinematics import clamp as _clamp
from kinematics import parse_timestamp as _parse_timestamp

# _compute_deltas lives in feature_factory which requires torch
torch_available = True
try:
    from feature_factory import (
        ACCELERATION_BOUNDS,
        COMPUTED_SPEED_BOUNDS,
        HEADING_CHANGE_BOUNDS,
        MIN_DELTA_TIME_SEC,
        _compute_deltas,
        _last_raw_point,
        _vessel_buffers,
    )
except ModuleNotFoundError:
    torch_available = False

requires_torch = pytest.mark.skipif(not torch_available, reason="torch not installed")


# ---------------------------------------------------------------------------
# _clamp
# ---------------------------------------------------------------------------
class TestClamp:
    def test_within_bounds(self):
        assert _clamp(5.0, (0.0, 10.0)) == 5.0

    def test_at_lower_bound(self):
        assert _clamp(0.0, (0.0, 10.0)) == 0.0

    def test_at_upper_bound(self):
        assert _clamp(10.0, (0.0, 10.0)) == 10.0

    def test_below_lower_clamps(self):
        assert _clamp(-1.0, (0.0, 10.0)) == 0.0

    def test_above_upper_clamps(self):
        assert _clamp(15.0, (0.0, 10.0)) == 10.0


# ---------------------------------------------------------------------------
# _angular_diff_deg
# ---------------------------------------------------------------------------
class TestAngularDiff:
    def test_same_heading(self):
        assert _angular_diff_deg(90.0, 90.0) == 0.0

    def test_simple_difference(self):
        assert _angular_diff_deg(100.0, 80.0) == pytest.approx(20.0)

    def test_wraparound_across_360(self):
        # 10° and 350° → shortest arc is 20°, not 340°
        assert _angular_diff_deg(10.0, 350.0) == pytest.approx(20.0)

    def test_wraparound_other_direction(self):
        assert _angular_diff_deg(350.0, 10.0) == pytest.approx(20.0)

    def test_exactly_180(self):
        assert _angular_diff_deg(0.0, 180.0) == pytest.approx(180.0)

    def test_north_south(self):
        assert _angular_diff_deg(0.0, 359.0) == pytest.approx(1.0)

    def test_result_always_in_0_to_180(self):
        for a in range(0, 360, 30):
            for b in range(0, 360, 30):
                result = _angular_diff_deg(float(a), float(b))
                assert 0.0 <= result <= 180.0


# ---------------------------------------------------------------------------
# _parse_timestamp
# ---------------------------------------------------------------------------
class TestParseTimestamp:
    def test_iso_string_with_z(self):
        ts = _parse_timestamp("2020-01-15T12:00:00Z")
        assert ts.tzinfo is not None
        assert ts.hour == 12

    def test_iso_string_with_offset(self):
        ts = _parse_timestamp("2020-01-15T12:00:00+00:00")
        assert ts.tzinfo is not None

    def test_naive_datetime_becomes_utc(self):
        naive = datetime(2020, 1, 15, 12, 0, 0)
        ts = _parse_timestamp(naive)
        assert ts.tzinfo is not None

    def test_aware_datetime_passthrough(self):
        aware = datetime(2020, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        ts = _parse_timestamp(aware)
        assert ts == aware


# ---------------------------------------------------------------------------
# _compute_deltas
# ---------------------------------------------------------------------------
def _make_raw(lat=21.3, lon=-157.8, sog=12.0, cog=90.0, ts=None):
    if ts is None:
        ts = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    return {"lat": lat, "lon": lon, "sog": sog, "cog": cog, "ts": ts}


@requires_torch
class TestComputeDeltas:
    def setup_method(self):
        # Each test gets a unique MMSI to avoid shared state pollution
        self._counter = getattr(self, "_counter", 0) + 1
        self.mmsi = f"TEST_KINEMATICS_{id(self)}_{self._counter}"
        # Clear any stale state for this MMSI
        _vessel_buffers.pop(self.mmsi, None)
        _last_raw_point.pop(self.mmsi, None)

    def test_first_point_returns_none(self):
        result = _compute_deltas(self.mmsi, _make_raw())
        assert result is None

    def test_second_point_returns_enriched(self):
        t0 = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        t1 = datetime(2020, 1, 1, 0, 1, 30, tzinfo=timezone.utc)  # 90s gap
        _compute_deltas(self.mmsi, _make_raw(ts=t0))
        result = _compute_deltas(self.mmsi, _make_raw(ts=t1))
        assert result is not None
        assert "computed_speed" in result
        assert "acceleration" in result
        assert "heading_change" in result
        assert "delta_time" in result

    def test_sub_second_gap_returns_none(self):
        t0 = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        t1 = t0 + timedelta(milliseconds=500)
        _compute_deltas(self.mmsi, _make_raw(ts=t0))
        result = _compute_deltas(self.mmsi, _make_raw(ts=t1))
        assert result is None

    def test_negative_time_gap_returns_none(self):
        t0 = datetime(2020, 1, 1, 0, 1, 0, tzinfo=timezone.utc)
        t_earlier = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        _compute_deltas(self.mmsi, _make_raw(ts=t0))
        result = _compute_deltas(self.mmsi, _make_raw(ts=t_earlier))
        assert result is None

    def test_computed_speed_clamped_to_bounds(self):
        # Ridiculously large distance in short time → should clamp to max speed
        t0 = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        t1 = t0 + timedelta(seconds=5)
        _compute_deltas(self.mmsi, _make_raw(lat=0.0, lon=0.0, ts=t0))
        result = _compute_deltas(self.mmsi, _make_raw(lat=89.0, lon=179.0, ts=t1))
        assert result is not None
        assert result["computed_speed"] <= COMPUTED_SPEED_BOUNDS[1]
        assert result["computed_speed"] >= COMPUTED_SPEED_BOUNDS[0]

    def test_acceleration_clamped_to_bounds(self):
        t0 = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        t1 = t0 + timedelta(seconds=2)
        _compute_deltas(self.mmsi, _make_raw(sog=0.0, ts=t0))
        result = _compute_deltas(self.mmsi, _make_raw(sog=100.0, ts=t1))
        assert result is not None
        assert result["acceleration"] <= ACCELERATION_BOUNDS[1]
        assert result["acceleration"] >= ACCELERATION_BOUNDS[0]

    def test_heading_change_always_in_bounds(self):
        t0 = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        t1 = t0 + timedelta(seconds=90)
        _compute_deltas(self.mmsi, _make_raw(cog=1.0, ts=t0))
        result = _compute_deltas(self.mmsi, _make_raw(cog=359.0, ts=t1))
        assert result is not None
        # 1° and 359° should produce heading_change of 2°, not 358°
        assert result["heading_change"] == pytest.approx(2.0, abs=0.01)

    def test_delta_time_correct(self):
        t0 = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        t1 = t0 + timedelta(seconds=90)
        _compute_deltas(self.mmsi, _make_raw(ts=t0))
        result = _compute_deltas(self.mmsi, _make_raw(ts=t1))
        assert result is not None
        assert result["delta_time"] == pytest.approx(90.0)
