"""
FastAPI serving layer for the maritime anomaly detection system.

Endpoints:
- POST /predict   : Accept a single AIS point, return anomaly score and confidence.
- GET  /health    : Health check (returns {"status": "ok"}).
- GET  /ready     : Readiness probe (returns {"status": "ready"}).


KNOWN LIMITATION: In-memory per-vessel buffers (vessel_buffers) and SQLite
profile writes are not thread/async-safe for concurrent requests targeting
the same MMSI. This is acceptable for a single-client demo. For production
scaling, replace the in-memory dict with a Redis-backed store.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

# Add the parent directory to sys.path so we can import mlops modules
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "mlops"))

from feature_factory import process_ais_point

# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------
# Anomaly threshold (set to 95th percentile of clean-2020 scores from your diagnostic)
ANOMALY_THRESHOLD = 0.85  # Adjust if needed

# --------------------------------------------------------------------
# FastAPI App
# --------------------------------------------------------------------
app = FastAPI(
    title="Maritime Anomaly Detection API",
    description="Real-time vessel trajectory anomaly detection using XGBoost Hybrid.",
    version="1.0.0",
)

# --------------------------------------------------------------------
# Request / Response Models
# --------------------------------------------------------------------
class AISPoint(BaseModel):
    MMSI: str = Field(..., description="Vessel identifier")
    lat: float = Field(..., description="Latitude in decimal degrees")
    lon: float = Field(..., description="Longitude in decimal degrees")
    speed_over_ground_knots: float = Field(..., description="Speed over ground (knots)")
    course_over_ground_deg: float = Field(..., description="Course over ground (degrees, 0-360)")
    datetime_hst: str = Field(..., description="Timestamp in Hawaii Standard Time (ISO format)")

class PredictionResponse(BaseModel):
    mmsi: str
    anomaly_score: Optional[float] = None
    confidence: str          # "low" or "high" (from vessel profile)
    is_anomaly: bool
    baseline_established: bool
    point_count: int         # number of normal points accumulated for this vessel

# --------------------------------------------------------------------
# Health / Readiness endpoints
# --------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/ready")
async def ready():
    # You could add checks for model loading here
    return {"status": "ready"}

# --------------------------------------------------------------------
# Prediction endpoint (FIXED)
# --------------------------------------------------------------------
@app.post("/predict", response_model=PredictionResponse)
async def predict(point: AISPoint) -> Dict[str, Any]:
    """
    Accept a single AIS point, process it, and return the anomaly score.
    """
    raw = point.dict()
    result = process_ais_point(raw)
    
    # Import these here to avoid circular import issues
    from vessel_profile_store import get_profile
    from feature_factory import _vessel_buffers
    
    # Get the vessel's profile (for confidence, baseline status, and point_count)
    profile = get_profile(result["mmsi"])
    
    # Get the actual buffer length (even if not ready yet)
    buf_len = len(_vessel_buffers.get(result["mmsi"], []))
    
    if not result["ready"]:
        # Not enough points yet – return a "pending" response with REAL data
        return {
            "mmsi": result["mmsi"],
            "anomaly_score": None,
            "confidence": profile["confidence"],
            "is_anomaly": False,
            "baseline_established": profile["baseline_is_fallback"] == 0,
            "point_count": buf_len,
        }
    
    score = result["anomaly_score"]
    
    return {
        "mmsi": result["mmsi"],
        "anomaly_score": score,
        "confidence": profile["confidence"],
        "is_anomaly": score >= ANOMALY_THRESHOLD,
        "baseline_established": profile["baseline_is_fallback"] == 0,
        "point_count": profile["point_count"],
    }
# --------------------------------------------------------------------
# Optional: warmup on startup (load models to avoid first-request latency)
# --------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    # Force loading of models and resources
    from feature_factory import _get_resources
    _get_resources()
    print("Models loaded and ready.")

# --------------------------------------------------------------------
# Run with: uvicorn main:app --reload --host 0.0.0.0 --port 8000
# --------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)