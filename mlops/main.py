"""
FastAPI serving layer for the maritime anomaly detection system.

Endpoints:
- POST /predict   : Accept a single AIS point, return anomaly score and confidence.
- GET  /health    : Health check (returns {"status": "ok"}).
- GET  /ready     : Readiness probe (returns {"status": "ready"}).

KNOWN LIMITATION: In-memory per-vessel buffers (vessel_buffers) and SQLite
profile writes are not thread/async-safe for concurrent requests targeting
the same MMSI. This is acceptable for a single-client demo. For production
scaling with multiple workers, replace the in-memory dict with a Redis-backed
store — see mlops/feature_factory.py module docstring.
"""

import logging
import math
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "mlops"))

from feature_factory import get_buffer_length, process_ais_point
from vessel_profile_store import get_profile, init_db

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------
ANOMALY_THRESHOLD = 0.85  # P95 of clean-2020 score distribution


# --------------------------------------------------------------------
# Lifespan: runs once at startup and once at shutdown
# --------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialise DB tables then load all ML models into memory
    init_db()
    from feature_factory import _get_resources
    _get_resources()
    logger.info("Database initialised and models loaded.")
    yield
    # Shutdown: nothing to clean up for this demo


# --------------------------------------------------------------------
# FastAPI App
# --------------------------------------------------------------------
app = FastAPI(
    title="Maritime Anomaly Detection API",
    description="Real-time vessel trajectory anomaly detection using XGBoost Hybrid.",
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------
# Request / Response Models
# --------------------------------------------------------------------
class AISPoint(BaseModel):
    MMSI: str = Field(..., description="9-digit vessel identifier")
    lat: float = Field(..., description="Latitude in decimal degrees (-90 to 90)")
    lon: float = Field(..., description="Longitude in decimal degrees (-180 to 180)")
    speed_over_ground_knots: float = Field(..., description="Speed over ground (knots, 0–102.4)")
    course_over_ground_deg: float = Field(..., description="Course over ground (degrees, 0–360)")
    datetime_hst: str = Field(..., description="Timestamp (ISO 8601 string, any timezone)")

    @field_validator("lat")
    @classmethod
    def validate_lat(cls, v: float) -> float:
        if not math.isfinite(v) or not (-90.0 <= v <= 90.0):
            raise ValueError(f"lat must be a finite number in [-90, 90], got {v}")
        return v

    @field_validator("lon")
    @classmethod
    def validate_lon(cls, v: float) -> float:
        if not math.isfinite(v) or not (-180.0 <= v <= 180.0):
            raise ValueError(f"lon must be a finite number in [-180, 180], got {v}")
        return v

    @field_validator("speed_over_ground_knots")
    @classmethod
    def validate_sog(cls, v: float) -> float:
        if not math.isfinite(v) or not (0.0 <= v <= 102.4):
            raise ValueError(f"speed_over_ground_knots must be finite and in [0, 102.4], got {v}")
        return v

    @field_validator("course_over_ground_deg")
    @classmethod
    def validate_cog(cls, v: float) -> float:
        if not math.isfinite(v) or not (0.0 <= v <= 360.0):
            raise ValueError(f"course_over_ground_deg must be finite and in [0, 360], got {v}")
        return v

    @field_validator("MMSI")
    @classmethod
    def validate_mmsi(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("MMSI must not be empty")
        if len(v) > 64:
            raise ValueError("MMSI must not exceed 64 characters")
        return v


class PredictionResponse(BaseModel):
    mmsi: str
    anomaly_score: Optional[float] = None
    confidence: str          # "low" or "high" (from vessel profile)
    is_anomaly: bool
    baseline_established: bool
    point_count: int         # normal points accumulated for this vessel


# --------------------------------------------------------------------
# Health / Readiness endpoints
# --------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    return {"status": "ready"}


# --------------------------------------------------------------------
# Prediction endpoint
# --------------------------------------------------------------------
@app.post("/predict", response_model=PredictionResponse)
async def predict(point: AISPoint) -> Dict[str, Any]:
    """
    Accept a single AIS point, process it through the full pipeline, and return
    the anomaly score.  Returns is_anomaly=False and anomaly_score=None while
    the per-vessel buffer is warming up (< 30 points).
    """
    raw = point.model_dump()
    result = process_ais_point(raw)

    profile = get_profile(result["mmsi"])
    buf_len = get_buffer_length(result["mmsi"])

    if not result["ready"]:
        return {
            "mmsi": result["mmsi"],
            "anomaly_score": None,
            "confidence": profile["confidence"],
            "is_anomaly": False,
            "baseline_established": profile["baseline_is_fallback"] == 0,
            "point_count": buf_len,
        }

    score = result["anomaly_score"]
    logger.info(
        "prediction mmsi=%s score=%.4f is_anomaly=%s baseline=%s",
        result["mmsi"], score, score >= ANOMALY_THRESHOLD,
        profile["baseline_is_fallback"] == 0,
    )

    return {
        "mmsi": result["mmsi"],
        "anomaly_score": score,
        "confidence": profile["confidence"],
        "is_anomaly": score >= ANOMALY_THRESHOLD,
        "baseline_established": profile["baseline_is_fallback"] == 0,
        "point_count": profile["point_count"],
    }


# --------------------------------------------------------------------
# Run with: uvicorn mlops.main:app --host 0.0.0.0 --port 8000
# --------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
