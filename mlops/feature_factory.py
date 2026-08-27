"""
Feature Factory for MLOps Serving.

Consumes one AIS point at a time, maintains a per-vessel rolling buffer,
computes the 55 features in the exact order xgboost_hybrid_final.json expects,
and returns an anomaly score.

Design notes (read before deploying):
- Kinematic deltas (computed_speed_knots, acceleration_knots_per_sec,
  heading_change_deg, delta_time_sec) are computed INCREMENTALLY as each new
  point arrives, using the previously-seen raw point for that vessel. This is
  O(1) per point instead of recomputing the whole 30-point window's deltas
  every time.
- ACCELERATION: uses reported speed_over_ground_knots, NOT computed_speed_knots.
  If your training preprocess.py used computed_speed_knots instead, change
  `raw_point["sog"]` to `prev["computed_speed"]` in _compute_deltas.
- Training-time outlier filtering DROPPED rows outside:
    computed_speed_knots in [0, 60]
    acceleration_knots_per_sec in [-5, 5]
    heading_change_deg in [0, 180]
    delta_time_sec >= 1
  You cannot drop a live point mid-stream without breaking the buffer, so
  this serving code CLAMPS to those bounds instead. That is a real, stated
  behavioral difference from training -- document it as a limitation.
- The very first point received for a vessel (this process's lifetime) has no
  predecessor, so no deltas can be computed. It is cached as `_last_raw` and
  not added to the enriched buffer until a second point arrives.
- IMPORTANT: the same MIN_DELTA_TIME_SEC guard that skips a first point also
  skips ANY point whose gap since the vessel's last message is < 1s -- and
  the same guard fires (with dt_sec negative) if a point arrives *before*
  the vessel's last recorded timestamp, e.g. out-of-order delivery or two
  independent synthetic test runs using overlapping/reset clocks in the same
  process. Either way `_vessel_buffers[mmsi]` is only created/updated once a
  point clears this guard. Callers must not index `_vessel_buffers[mmsi]`
  directly without a `.get(mmsi, [])` fallback.
- TransformerVAE and LSTM are scored deterministically: encode -> mu ->
  decode(mu). Do NOT call model(x) directly for scoring -- that samples via
  reparameterize() and is non-deterministic even in eval mode (this bit the
  project during the per-incident-type diagnostic).
- maybe_update_profile() is called with the just-computed score on every
  ready point. This feeds the anomaly score back into the vessel's baseline
  profile, so the gating logic inside vessel_profile_store.py (deciding what
  counts as "normal enough" to fold into the baseline) is load-bearing for
  correctness -- confirm it actually rejects high-score points before
  trusting this in production, otherwise a vessel drifting into genuinely
  anomalous behavior could get "normalized" into its own new baseline.
- In-process buffers are fine for a single-worker FastAPI deployment. If you
  scale to multiple workers/replicas, you'll need a shared store (Redis) to
  keep vessel state consistent across processes.
"""

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone

import numpy as np
import torch
import xgboost as xgb
from geopy.distance import distance

from vessel_profile_store import get_profile, maybe_update_profile

import sys

# ============================================================
# Paths & Configuration
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent  # maritime/

# Make project root importable
sys.path.insert(0, str(BASE_DIR))

XGB_MODEL_PATH = BASE_DIR / "hybrid" / "xgboost_hybrid_final.json"
TRANSFORMER_CHECKPOINT = BASE_DIR / "model" / "best_model_large.pt"
LSTM_CHECKPOINT = BASE_DIR / "model" / "lstm_ae_run" / "lstm_ae_best.pt"
NORM_MEAN_PATH = BASE_DIR / "processing" / "norm_mean.npy"
NORM_STD_PATH = BASE_DIR / "processing" / "norm_std.npy"

from model.model import TransformerVAE
from model.train_lstm_ae import LSTMAutoencoder

WINDOW_SIZE = 30
MAX_BUFFER = 60  # keep some headroom beyond the window
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Outlier clamp bounds (mirrors training-time filter thresholds; clamped
# here instead of dropped -- see module docstring)
COMPUTED_SPEED_BOUNDS = (0.0, 60.0)
ACCELERATION_BOUNDS = (-5.0, 5.0)
HEADING_CHANGE_BOUNDS = (0.0, 180.0)
MIN_DELTA_TIME_SEC = 1.0

FEATURE_NAMES = [
    "baseline_is_fallback",
    "vessel_type_code",
    "length_m",
    "width_m",
    "lat_mean", "lat_std", "lat_min", "lat_max",
    "lon_mean", "lon_std", "lon_min", "lon_max",
    "speed_over_ground_knots_mean", "speed_over_ground_knots_std", "speed_over_ground_knots_min", "speed_over_ground_knots_max",
    "course_over_ground_deg_mean", "course_over_ground_deg_std", "course_over_ground_deg_min", "course_over_ground_deg_max",
    "computed_speed_knots_mean", "computed_speed_knots_std", "computed_speed_knots_min", "computed_speed_knots_max",
    "acceleration_knots_per_sec_mean", "acceleration_knots_per_sec_std", "acceleration_knots_per_sec_min", "acceleration_knots_per_sec_max",
    "heading_change_deg_mean", "heading_change_deg_std", "heading_change_deg_min", "heading_change_deg_max",
    "delta_time_mean", "delta_time_max", "delta_time_std",
    "speed_zscore_vs_vessel",
    "heading_zscore_vs_vessel",
    "dist_from_vessel_centroid_km_mean",
    "dist_from_vessel_centroid_km_max",
    "trans_err_lat", "trans_err_lon", "trans_err_speed", "trans_err_course",
    "trans_err_computed_speed", "trans_err_acceleration", "trans_err_heading_change",
    "trans_err_agg",
    "lstm_err_lat", "lstm_err_lon", "lstm_err_speed", "lstm_err_course",
    "lstm_err_computed_speed", "lstm_err_acceleration", "lstm_err_heading_change",
    "lstm_err_agg",
]
assert len(FEATURE_NAMES) == 55, "Feature count mismatch"

CHANNEL_NAMES = ["lat", "lon", "speed", "course", "computed_speed", "acceleration", "heading_change"]
KIN_ORDER = ["lat", "lon", "sog", "cog", "computed_speed", "acceleration", "heading_change"]


def _clamp(value: float, bounds: Tuple[float, float]) -> float:
    lo, hi = bounds
    return max(lo, min(hi, value))


def _angular_diff_deg(a: float, b: float) -> float:
    """Smallest absolute angular difference between two headings, in [0, 180]."""
    diff = abs(a - b) % 360.0
    return diff if diff <= 180.0 else 360.0 - diff


def _parse_timestamp(ts) -> datetime:
    if isinstance(ts, datetime):
        # If naive, assume UTC
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts
    # If string, parse and ensure timezone-aware
    dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


# ============================================================
# Deep model loading (lazy singletons)
# ============================================================
_transformer_model = None
_lstm_model = None
_xgb_booster = None
_norm_mean = None
_norm_std = None


def _get_resources():
    global _transformer_model, _lstm_model, _xgb_booster, _norm_mean, _norm_std
    if _transformer_model is None:
        _transformer_model = TransformerVAE(
            n_features=7, seq_len=WINDOW_SIZE,
            d_model=256, nhead=8, num_layers=6,
            latent_dim=64, dim_feedforward=512, dropout=0.1,
        ).to(DEVICE)
        _transformer_model.load_state_dict(torch.load(TRANSFORMER_CHECKPOINT, map_location=DEVICE))
        _transformer_model.eval()
    if _lstm_model is None:
        _lstm_model = LSTMAutoencoder(
            n_features=7, hidden_dim=128, latent_dim=64, num_layers=2, dropout=0.2,
        ).to(DEVICE)
        _lstm_model.load_state_dict(torch.load(LSTM_CHECKPOINT, map_location=DEVICE))
        _lstm_model.eval()
    if _xgb_booster is None:
        _xgb_booster = xgb.Booster()
        _xgb_booster.load_model(str(XGB_MODEL_PATH))
    if _norm_mean is None:
        _norm_mean = np.load(NORM_MEAN_PATH)
        _norm_std = np.load(NORM_STD_PATH)
    return _transformer_model, _lstm_model, _xgb_booster, _norm_mean, _norm_std


def _deterministic_transformer_score(buffer_raw: np.ndarray) -> Tuple[np.ndarray, float]:
    """buffer_raw: (30, 7) raw unnormalized values -> (per-channel MSE, agg MSE)."""
    model, _, _, mean, std = _get_resources()
    x_norm = (buffer_raw - mean) / (std + 1e-8)
    x = torch.from_numpy(x_norm).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        mu, _logvar = model.encode(x)
        recon = model.decode(mu)  # deterministic: no reparameterize() sampling
    err = ((recon - x) ** 2).mean(dim=1).squeeze(0).cpu().numpy()
    return err, float(err.mean())


def _deterministic_lstm_score(buffer_raw: np.ndarray) -> Tuple[np.ndarray, float]:
    _, model, _, mean, std = _get_resources()
    x_norm = (buffer_raw - mean) / (std + 1e-8)
    x = torch.from_numpy(x_norm).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        recon, _ = model(x)   # LSTM returns (recon, latent)
    err = ((recon - x) ** 2).mean(dim=1).squeeze(0).cpu().numpy()
    return err, float(err.mean())


# ============================================================
# Per-vessel state (in-memory, single-process only)
# ============================================================
_vessel_buffers: Dict[str, List[Dict]] = {}   # enriched points (have all 7 kinematic fields)
_last_raw_point: Dict[str, Dict] = {}         # last raw point seen per vessel, for delta calc


def _compute_deltas(mmsi: str, raw_point: Dict) -> Optional[Dict]:
    """
    Combine raw_point with the vessel's last raw point to produce an enriched
    point carrying all 7 kinematic features. Returns None if this is the
    vessel's first observed point (no predecessor to diff against) or if the
    gap since the last point is < MIN_DELTA_TIME_SEC (treated as a
    duplicate/out-of-order message, per the training-time filter; this also
    fires if dt_sec is negative, e.g. a point arrives "before" the last one).

    NOTE: _vessel_buffers[mmsi] is only created once this returns non-None.
    Any code reading _vessel_buffers[mmsi] directly (rather than through
    process_ais_point's return value) must use .get(mmsi, []).
    """
    prev = _last_raw_point.get(mmsi)
    _last_raw_point[mmsi] = raw_point  # always advance the "previous" pointer

    if prev is None:
        return None

    dt_sec = (raw_point["ts"] - prev["ts"]).total_seconds()
    if dt_sec < MIN_DELTA_TIME_SEC:
        # Duplicate, out-of-order, or negative-gap message; skip enrichment
        # for this point rather than dividing by a near-zero/negative interval.
        return None

    dist_km = distance((prev["lat"], prev["lon"]), (raw_point["lat"], raw_point["lon"])).km
    computed_speed = (dist_km / (dt_sec / 3600.0)) / 1.852
    computed_speed = _clamp(computed_speed, COMPUTED_SPEED_BOUNDS)

    # Acceleration uses reported SOG, NOT computed_speed (assumption -- see docstring)
    acceleration = (raw_point["sog"] - prev["sog"]) / dt_sec
    acceleration = _clamp(acceleration, ACCELERATION_BOUNDS)

    heading_change = _angular_diff_deg(raw_point["cog"], prev["cog"])
    heading_change = _clamp(heading_change, HEADING_CHANGE_BOUNDS)

    return {
        "lat": raw_point["lat"],
        "lon": raw_point["lon"],
        "sog": raw_point["sog"],
        "cog": raw_point["cog"],
        "computed_speed": computed_speed,
        "acceleration": acceleration,
        "heading_change": heading_change,
        "delta_time": dt_sec,
    }


def _add_enriched_point(mmsi: str, enriched: Dict):
    buf = _vessel_buffers.setdefault(mmsi, [])
    buf.append(enriched)
    if len(buf) > MAX_BUFFER:
        del buf[0]


def _extract_features(mmsi: str, window: List[Dict]) -> np.ndarray:
    """window: last WINDOW_SIZE enriched points -> 55-length feature vector."""
    arrays = {k: np.array([p[k] for p in window]) for k in KIN_ORDER}
    dt_arr = np.array([p["delta_time"] for p in window])

    features: Dict[str, float] = {}
    kin_col_names = [
        "lat", "lon", "speed_over_ground_knots", "course_over_ground_deg",
        "computed_speed_knots", "acceleration_knots_per_sec", "heading_change_deg",
    ]
    for col_name, key in zip(kin_col_names, KIN_ORDER):
        arr = arrays[key]
        features[f"{col_name}_mean"] = float(arr.mean())
        features[f"{col_name}_std"] = float(arr.std())
        features[f"{col_name}_min"] = float(arr.min())
        features[f"{col_name}_max"] = float(arr.max())

    features["delta_time_mean"] = float(dt_arr.mean())
    features["delta_time_max"] = float(dt_arr.max())
    features["delta_time_std"] = float(dt_arr.std())

    profile = get_profile(mmsi)
    features["baseline_is_fallback"] = float(profile["baseline_is_fallback"])
    features["vessel_type_code"] = float(profile["vessel_type_code"])
    features["length_m"] = float(profile["length_m"])
    features["width_m"] = float(profile["width_m"])

    features["speed_zscore_vs_vessel"] = (
        arrays["sog"].mean() - profile["speed_mean"]
    ) / (profile["speed_std"] + 1e-8)

    features["heading_zscore_vs_vessel"] = (
        arrays["heading_change"].mean() - profile["heading_mean"]
    ) / (profile["heading_std"] + 1e-8)

    dist_from_centroid = np.array([
        distance((lat, lon), (profile["lat_centroid"], profile["lon_centroid"])).km
        for lat, lon in zip(arrays["lat"], arrays["lon"])
    ])
    features["dist_from_vessel_centroid_km_mean"] = float(dist_from_centroid.mean())
    features["dist_from_vessel_centroid_km_max"] = float(dist_from_centroid.max())

    # Deep model errors
    buffer_raw = np.stack([arrays[k] for k in KIN_ORDER], axis=1).astype(np.float32)  # (30, 7)
    trans_err, trans_agg = _deterministic_transformer_score(buffer_raw)
    lstm_err, lstm_agg = _deterministic_lstm_score(buffer_raw)
    for i, ch in enumerate(CHANNEL_NAMES):
        features[f"trans_err_{ch}"] = float(trans_err[i])
        features[f"lstm_err_{ch}"] = float(lstm_err[i])
    features["trans_err_agg"] = trans_agg
    features["lstm_err_agg"] = lstm_agg

    # Return in FEATURE_NAMES order
    return np.array([features[name] for name in FEATURE_NAMES], dtype=np.float32)


def _score(feature_vector: np.ndarray) -> float:
    _, _, booster, _, _ = _get_resources()
    dmat = xgb.DMatrix(feature_vector.reshape(1, -1), feature_names=FEATURE_NAMES)
    return float(booster.predict(dmat)[0])


# ============================================================
# Public API
# ============================================================
def process_ais_point(raw_point: Dict) -> Dict:
    """
    Main entry point for the serving layer (call once per incoming AIS message).

    Args:
        raw_point: dict with keys MMSI, lat, lon, speed_over_ground_knots,
                   course_over_ground_deg, and a timestamp under
                   'datetime_hst' or 'timestamp' (ISO 8601 string or datetime).

    Returns:
        {
          "mmsi": str,
          "ready": bool,          # False until buffer has >= WINDOW_SIZE points
          "anomaly_score": float | None,
          "feature_vector": np.ndarray | None,
        }
    """
    mmsi = str(raw_point["MMSI"])
    normalized_raw = {
        "lat": float(raw_point["lat"]),
        "lon": float(raw_point["lon"]),
        "sog": float(raw_point["speed_over_ground_knots"]),
        "cog": float(raw_point["course_over_ground_deg"]),
        "ts": _parse_timestamp(raw_point.get("datetime_hst", raw_point.get("timestamp"))),
    }

    enriched = _compute_deltas(mmsi, normalized_raw)
    if enriched is None:
        return {"mmsi": mmsi, "ready": False, "anomaly_score": None, "feature_vector": None}

    _add_enriched_point(mmsi, enriched)
    buf = _vessel_buffers[mmsi]
    if len(buf) < WINDOW_SIZE:
        return {"mmsi": mmsi, "ready": False, "anomaly_score": None, "feature_vector": None}

    window = buf[-WINDOW_SIZE:]
    feature_vector = _extract_features(mmsi, window)
    score = _score(feature_vector)
    maybe_update_profile(mmsi, normalized_raw, score)

    return {
        "mmsi": mmsi,
        "ready": True,
        "anomaly_score": score,
        "feature_vector": feature_vector,
    }


# ============================================================
# Quick smoke test
# ============================================================
if __name__ == "__main__":
    # Use synthetic timestamps spaced like real AIS reports (~90s apart),
    # NOT wall-clock time.sleep(). Consecutive messages closer together than
    # MIN_DELTA_TIME_SEC (1s) -- or arriving "before" the vessel's last
    # recorded timestamp -- are skipped by _compute_deltas. Since
    # _vessel_buffers[mmsi] is only created/extended on a successful
    # enrichment, always use .get(mmsi, []) when inspecting buffer state
    # from outside process_ais_point.
    #
    # NOTE: run this module only once per process for a given mmsi.
    # _vessel_buffers and _last_raw_point are module-level dicts that
    # persist for the life of the process. If you run a second synthetic
    # scenario for the same mmsi afterwards (e.g. a second block like this
    # one), its `datetime.now()` will be only milliseconds after this run's,
    # which is *earlier* than the timestamps this run advanced _last_raw_point
    # to -- every point will look like it arrived in the past, dt_sec goes
    # negative, and the two runs' data silently mix together. Use a fresh
    # process (or a fresh mmsi) per scenario instead of appending another
    # `if __name__` block to this file.
    #
    # Boundary math: the very first raw point for a vessel never enriches
    # (no predecessor), so with WINDOW_SIZE=30 the window first becomes ready
    # at point 31 (30 successful enrichments, from points 2 through 31).
    mmsi = "TEST_MMSI"
    base_ts = datetime.now(timezone.utc).timestamp()
    points = []

    # Phase 1: normal sailing, points 1-30 (constant speed/course)
    for i in range(30):
        lon = -157.8 + i * 0.001
        points.append({
            "MMSI": mmsi, "lat": 21.3, "lon": lon,
            "speed_over_ground_knots": 12.0,
            "course_over_ground_deg": 90.0,
            "timestamp": datetime.fromtimestamp(base_ts + i * 90, tz=timezone.utc).isoformat(),
        })

    # Phase 2: loss of propulsion, points 31-40 (speed collapses)
    for i in range(30, 40):
        lon = -157.8 + i * 0.001
        points.append({
            "MMSI": mmsi, "lat": 21.3, "lon": lon,
            "speed_over_ground_knots": 0.3,
            "course_over_ground_deg": 90.0,
            "timestamp": datetime.fromtimestamp(base_ts + i * 90, tz=timezone.utc).isoformat(),
        })

    # Phase 3: sharp turn while still stalled, points 41-50 (course swings 90 deg)
    for i in range(40, 50):
        lon = -157.8 + 40 * 0.001 + (i - 40) * 0.00003
        points.append({
            "MMSI": mmsi, "lat": 21.3, "lon": lon,
            "speed_over_ground_knots": 0.3,
            "course_over_ground_deg": 180.0 + (i - 40) * 9.0,
            "timestamp": datetime.fromtimestamp(base_ts + i * 90, tz=timezone.utc).isoformat(),
        })

    print("Feeding points...")
    scores = []
    for i, pt in enumerate(points, start=1):
        result = process_ais_point(pt)
        if result["ready"]:
            scores.append(result["anomaly_score"])
            print(f"Point {i}: Score = {result['anomaly_score']:.4f}")
        else:
            buf_len = len(_vessel_buffers.get(mmsi, []))
            print(f"Point {i}: Buffer not ready yet ({buf_len} points)")

    # scores[0] corresponds to Point 31, the first point where the 30-window
    # is full (see boundary math above) -- so 10 scores of loss-of-propulsion
    # followed by 10 scores of the turn.
    print("\n--- Score Summary ---")
    print(f"Loss of propulsion (points 31-40): {scores[0:10]}")
    print(f"Sharp turn (points 41-50): {scores[10:20]}")