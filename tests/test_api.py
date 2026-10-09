"""
Integration tests for the FastAPI endpoints.

Uses FastAPI's TestClient (synchronous, no server required) and a temporary
SQLite database so the vessel_profiles.db on disk is not touched.

The full ML models ARE loaded here — this is an integration test, not a unit
test.  If model artifacts are missing the test is skipped automatically.
"""
import os
import sys
from pathlib import Path

import pytest

# Temp DB for all API tests
os.environ.setdefault("VESSEL_DB_PATH", ":memory:")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "mlops"))

# Check that model artifacts exist before attempting import
_MODELS_PRESENT = all([
    (REPO_ROOT / "hybrid" / "xgboost_hybrid_final.json").exists(),
    (REPO_ROOT / "model" / "best_model_large.pt").exists(),
    (REPO_ROOT / "model" / "lstm_ae_run" / "lstm_ae_best.pt").exists(),
    (REPO_ROOT / "processing" / "norm_mean.npy").exists(),
    (REPO_ROOT / "processing" / "norm_std.npy").exists(),
])

pytestmark = pytest.mark.skipif(
    not _MODELS_PRESENT,
    reason="Model artifacts not present — skipping integration tests",
)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("db")
    db_file = tmp / "test.db"
    os.environ["VESSEL_DB_PATH"] = str(db_file)

    # Re-patch the module-level DB_PATH so init_db uses the temp file
    import vessel_profile_store as vps
    vps.DB_PATH = db_file

    from fastapi.testclient import TestClient
    from main import app
    with TestClient(app) as c:
        yield c


def _point(mmsi, i, sog=12.0, cog=90.0):
    from datetime import datetime, timezone
    base_ts = 1577836800  # 2020-01-01 00:00:00 UTC
    return {
        "MMSI": mmsi,
        "lat": 21.3,
        "lon": -157.8 + i * 0.001,
        "speed_over_ground_knots": sog,
        "course_over_ground_deg": cog,
        "datetime_hst": datetime.fromtimestamp(
            base_ts + i * 90, tz=timezone.utc
        ).isoformat(),
    }


# ---------------------------------------------------------------------------
# /health and /ready
# ---------------------------------------------------------------------------
def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ready(client):
    resp = client.get("/ready")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ready"}


# ---------------------------------------------------------------------------
# /predict — warmup phase
# ---------------------------------------------------------------------------
def test_predict_not_ready_before_30_points(client):
    mmsi = "API_TEST_WARMUP"
    for i in range(29):
        resp = client.post("/predict", json=_point(mmsi, i))
        assert resp.status_code == 200
        data = resp.json()
        assert data["anomaly_score"] is None
        assert data["is_anomaly"] is False
        assert data["mmsi"] == mmsi


def test_predict_ready_at_31st_point(client):
    """
    First point sets _last_raw_point but produces no enriched point.
    Points 2–31 each produce an enriched point; buffer reaches 30 at point 31.
    """
    mmsi = "API_TEST_READY"
    result = None
    for i in range(35):
        resp = client.post("/predict", json=_point(mmsi, i))
        assert resp.status_code == 200
        result = resp.json()

    # By point 35 the buffer is definitely full
    assert result["anomaly_score"] is not None
    assert isinstance(result["anomaly_score"], float)
    assert 0.0 <= result["anomaly_score"] <= 1.0
    assert result["mmsi"] == mmsi


# ---------------------------------------------------------------------------
# /predict — input validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("field,value", [
    ("lat", 91.0),
    ("lat", -91.0),
    ("lat", float("nan")),
    ("lon", 181.0),
    ("lon", -181.0),
    ("speed_over_ground_knots", -1.0),
    ("speed_over_ground_knots", 200.0),
    ("course_over_ground_deg", -1.0),
    ("course_over_ground_deg", 361.0),
    ("MMSI", ""),
    ("MMSI", "A" * 65),
])
def test_invalid_input_returns_422(client, field, value):
    payload = _point("VALIDATION_TEST", 0)
    payload[field] = value
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422, (
        f"Expected 422 for {field}={value!r}, got {resp.status_code}: {resp.text}"
    )


# ---------------------------------------------------------------------------
# /predict — response schema
# ---------------------------------------------------------------------------
def test_response_schema_complete(client):
    mmsi = "API_SCHEMA_TEST"
    # Send enough points for a ready response
    resp = None
    for i in range(35):
        resp = client.post("/predict", json=_point(mmsi, i))
    data = resp.json()
    required_keys = {
        "mmsi", "anomaly_score", "confidence", "is_anomaly",
        "baseline_established", "point_count",
    }
    assert required_keys.issubset(data.keys())
    assert data["confidence"] in ("low", "high")
    assert isinstance(data["is_anomaly"], bool)
    assert isinstance(data["baseline_established"], bool)
    assert isinstance(data["point_count"], int)
