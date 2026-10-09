"""
Pure Python / numpy kinematic helper functions used by feature_factory.py.

Kept in a separate module with no heavy dependencies (no torch, xgboost, geopy)
so they can be unit-tested in lightweight CI environments that don't have the
full ML stack installed.
"""
from datetime import datetime, timezone
from typing import Tuple


def clamp(value: float, bounds: Tuple[float, float]) -> float:
    lo, hi = bounds
    return max(lo, min(hi, value))


def angular_diff_deg(a: float, b: float) -> float:
    """Smallest absolute angular difference between two headings, in [0, 180]."""
    diff = abs(a - b) % 360.0
    return diff if diff <= 180.0 else 360.0 - diff


def parse_timestamp(ts) -> datetime:
    """Parse a timestamp (ISO string or datetime) and ensure it is timezone-aware."""
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts
    dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt
