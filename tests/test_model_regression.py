"""
Model regression test.

Runs a fixed synthetic AIS sequence through the full pipeline
(feature_factory → XGBoost) and asserts the output score matches a stored
golden value within a tight tolerance.

If the golden value does not exist yet (first run), the test writes it and
passes.  On subsequent runs it enforces the golden value.

IMPORTANT: This test loads all three ML models.  It is automatically skipped
when model artifacts are not present (e.g., in lightweight CI environments
that only run unit tests).
"""
import os
import sys
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_FILE = REPO_ROOT / "tests" / "golden_score.json"

sys.path.insert(0, str(REPO_ROOT / "mlops"))

_MODELS_PRESENT = all([
    (REPO_ROOT / "hybrid" / "xgboost_hybrid_final.json").exists(),
    (REPO_ROOT / "model" / "best_model_large.pt").exists(),
    (REPO_ROOT / "model" / "lstm_ae_run" / "lstm_ae_best.pt").exists(),
    (REPO_ROOT / "processing" / "norm_mean.npy").exists(),
    (REPO_ROOT / "processing" / "norm_std.npy").exists(),
])

pytestmark = pytest.mark.skipif(
    not _MODELS_PRESENT,
    reason="Model artifacts not present — skipping regression test",
)

# Use a temp DB so this test leaves no state on disk
os.environ.setdefault("VESSEL_DB_PATH", ":memory:")


def _build_sequence(mmsi: str):
    """Return the same 50-point deterministic sequence used in the smoke test."""
    base_ts = 1577836800  # 2020-01-01 00:00:00 UTC
    points = []
    for i in range(30):
        points.append({
            "MMSI": mmsi,
            "lat": 21.3,
            "lon": -157.8 + i * 0.001,
            "speed_over_ground_knots": 12.0,
            "course_over_ground_deg": 90.0,
            "datetime_hst": datetime.fromtimestamp(
                base_ts + i * 90, tz=timezone.utc
            ).isoformat(),
        })
    for i in range(30, 40):
        points.append({
            "MMSI": mmsi,
            "lat": 21.3,
            "lon": -157.8 + i * 0.001,
            "speed_over_ground_knots": 0.3,
            "course_over_ground_deg": 90.0,
            "datetime_hst": datetime.fromtimestamp(
                base_ts + i * 90, tz=timezone.utc
            ).isoformat(),
        })
    for i in range(40, 50):
        lon = -157.8 + 40 * 0.001 + (i - 40) * 0.00003
        points.append({
            "MMSI": mmsi,
            "lat": 21.3,
            "lon": lon,
            "speed_over_ground_knots": 0.3,
            "course_over_ground_deg": 180.0 + (i - 40) * 9.0,
            "datetime_hst": datetime.fromtimestamp(
                base_ts + i * 90, tz=timezone.utc
            ).isoformat(),
        })
    return points


def test_score_matches_golden():
    import vessel_profile_store as vps
    # Use in-memory DB for this test
    vps.DB_PATH = Path(":memory:")
    vps.init_db()

    # Import feature_factory AFTER setting up the DB path so it picks up the
    # patched module-level state
    from feature_factory import process_ais_point, _vessel_buffers, _last_raw_point

    mmsi = "REGRESSION_TEST_MMSI_v1"
    # Clean any stale state from earlier test runs in the same process
    _vessel_buffers.pop(mmsi, None)
    _last_raw_point.pop(mmsi, None)

    points = _build_sequence(mmsi)
    scores = []
    for pt in points:
        result = process_ais_point(pt)
        if result["ready"]:
            scores.append(result["anomaly_score"])

    assert len(scores) > 0, "No scores produced — buffer never reached 30 points"
    final_score = scores[-1]

    if not GOLDEN_FILE.exists():
        # First run: write the golden value
        GOLDEN_FILE.write_text(
            json.dumps({"final_score": final_score, "n_scores": len(scores)}, indent=2)
        )
        pytest.skip(
            f"Golden file created with score={final_score:.6f}. "
            "Re-run tests to enforce the golden value."
        )

    golden = json.loads(GOLDEN_FILE.read_text())
    golden_score = golden["final_score"]

    assert abs(final_score - golden_score) < 1e-4, (
        f"Score regression detected: got {final_score:.6f}, "
        f"expected {golden_score:.6f} (golden). "
        "If this is intentional (model update), delete tests/golden_score.json and re-run."
    )
