import sqlite3
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

# ============================================================
# CONFIGURATION (Tune these)
# ============================================================
MIN_BASELINE_POINTS = 50
SAMPLE_SIZE_CAP = 2000
SCORE_UPDATE_THRESHOLD = 0.77  
DB_PATH = Path("vessel_profiles.db")
POP_FALLBACK_PATH = Path("pop_fallback.json")

# ============================================================
# POPULATION FALLBACK (Load from training config)
# ============================================================
def load_pop_fallback() -> Dict:
    """Load the frozen population statistics used in training."""
    if POP_FALLBACK_PATH.exists():
        with open(POP_FALLBACK_PATH, "r") as f:
            return json.load(f)
    return {
        "speed_mean": 2.684884,
        "speed_std": 4.533485,
        "heading_mean": 18.655938,
        "heading_std": 37.439271,
        "lat_centroid": 21.310270,
        "lon_centroid": -157.869640,
    }

POP_FALLBACK = load_pop_fallback()

# ============================================================
# DATABASE SETUP
# ============================================================
def get_db_connection():
    """Return a connection to the SQLite database."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row  # Access columns by name
    return conn

def init_db():
    """Create tables if they don't exist."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Main profile table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vessel_profile (
                mmsi                 TEXT PRIMARY KEY,
                point_count          INTEGER NOT NULL DEFAULT 0,
                sample_count         INTEGER NOT NULL DEFAULT 0,
                speed_mean           REAL,
                speed_std            REAL,
                heading_mean         REAL,
                heading_std          REAL,
                lat_centroid         REAL,
                lon_centroid         REAL,
                vessel_type_code     INTEGER,
                length_m             REAL,
                width_m              REAL,
                static_is_default    INTEGER NOT NULL DEFAULT 1,
                baseline_established INTEGER NOT NULL DEFAULT 0,
                last_updated         TEXT
            )
        """)
        
        # Raw sample points table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vessel_sample_points (
                id                     INTEGER PRIMARY KEY AUTOINCREMENT,
                mmsi                   TEXT NOT NULL,
                speed_over_ground_knots REAL,
                heading_change_deg     REAL,
                lat                    REAL,
                lon                    REAL,
                observed_at            TEXT NOT NULL
            )
        """)
        
        # Index for fast lookups and eviction
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sample_mmsi ON vessel_sample_points(mmsi, observed_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sample_mmsi_id ON vessel_sample_points(mmsi, id)")
        
        conn.commit()
        print("Database initialized successfully.")

# ============================================================
# PROFILE QUERY (Cold-start logic)
# ============================================================
STATIC_FALLBACK_PATH = Path("static_fallback.json")

def load_static_fallback() -> Dict:
    if STATIC_FALLBACK_PATH.exists():
        with open(STATIC_FALLBACK_PATH, "r") as f:
            return json.load(f)
    return {"vessel_type_code": 0, "length_m": 30.0, "width_m": 8.0}  # sensible defaults

STATIC_FALLBACK = load_static_fallback()

def get_profile(mmsi: str) -> Dict:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM vessel_profile WHERE mmsi = ?", (mmsi,))
        row = cursor.fetchone()
    
    # Static attributes: use row values if a real broadcast was received, else training-set medians
    static = {
        "vessel_type_code": row["vessel_type_code"] if row and not row["static_is_default"] else STATIC_FALLBACK["vessel_type_code"],
        "length_m": row["length_m"] if row and not row["static_is_default"] else STATIC_FALLBACK["length_m"],
        "width_m": row["width_m"] if row and not row["static_is_default"] else STATIC_FALLBACK["width_m"],
    }
    
    if row is None or row["point_count"] < MIN_BASELINE_POINTS:
        return {
            **POP_FALLBACK,
            **static,
            "baseline_is_fallback": 1,
            "confidence": "low",
            "point_count": row["point_count"] if row else 0,
        }
    
    return {
        "speed_mean": row["speed_mean"],
        "speed_std": row["speed_std"],
        "heading_mean": row["heading_mean"],
        "heading_std": row["heading_std"],
        "lat_centroid": row["lat_centroid"],
        "lon_centroid": row["lon_centroid"],
        **static,
        "baseline_is_fallback": 0,
        "confidence": "high",
        "point_count": row["point_count"],
    }
# ============================================================
# SAMPLE MANAGEMENT & AGGREGATE RECOMPUTATION
# ============================================================
def _evict_oldest_if_over_cap(mmsi: str, cap: int = SAMPLE_SIZE_CAP):
    """Keep the sample size bounded by deleting the oldest points."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Count current points
        cursor.execute(
            "SELECT COUNT(*) as cnt FROM vessel_sample_points WHERE mmsi = ?",
            (mmsi,)
        )
        count = cursor.fetchone()["cnt"]
        
        if count > cap:
            # Delete the oldest (cap - count) points, sorted by ID (insertion order)
            to_delete = count - cap
            cursor.execute(
                """
                DELETE FROM vessel_sample_points
                WHERE id IN (
                    SELECT id FROM vessel_sample_points
                    WHERE mmsi = ?
                    ORDER BY id ASC
                    LIMIT ?
                )
                """,
                (mmsi, to_delete)
            )
            conn.commit()

def _recompute_aggregates(mmsi: str):
    """
    Recompute mean/std of speed & heading_change, and median lat/lon
    from the current sample of points for this vessel.
    Updates the vessel_profile row.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Fetch all sample points for this vessel
        cursor.execute(
            """
            SELECT speed_over_ground_knots, heading_change_deg, lat, lon
            FROM vessel_sample_points
            WHERE mmsi = ?
            """,
            (mmsi,)
        )
        rows = cursor.fetchall()
        
        if not rows:
            # No points left? Reset profile to NULL
            cursor.execute(
                """
                UPDATE vessel_profile
                SET sample_count = 0,
                    speed_mean = NULL, speed_std = NULL,
                    heading_mean = NULL, heading_std = NULL,
                    lat_centroid = NULL, lon_centroid = NULL,
                    baseline_established = 0,
                    last_updated = ?
                WHERE mmsi = ?
                """,
                (datetime.now(timezone.utc).isoformat(), mmsi)
            )
            conn.commit()
            return
        
        speeds = np.array([r[0] for r in rows if r[0] is not None])
        headings = np.array([r[1] for r in rows if r[1] is not None])
        lats = np.array([r[2] for r in rows if r[2] is not None])
        lons = np.array([r[3] for r in rows if r[3] is not None])
        
        speed_mean = float(np.mean(speeds)) if len(speeds) > 0 else None
        speed_std = float(np.std(speeds)) + 1e-6 if len(speeds) > 1 else 1.0
        heading_mean = float(np.mean(headings)) if len(headings) > 0 else None
        heading_std = float(np.std(headings)) + 1e-6 if len(headings) > 1 else 1.0
        lat_centroid = float(np.median(lats)) if len(lats) > 0 else None
        lon_centroid = float(np.median(lons)) if len(lons) > 0 else None
        
        sample_count = len(rows)
        baseline_established = 1 if sample_count >= MIN_BASELINE_POINTS else 0
        
        # Update profile
        cursor.execute(
            """
            UPDATE vessel_profile
            SET sample_count = ?,
                speed_mean = ?, speed_std = ?,
                heading_mean = ?, heading_std = ?,
                lat_centroid = ?, lon_centroid = ?,
                baseline_established = ?,
                point_count = point_count + 1,
                last_updated = ?
            WHERE mmsi = ?
            """,
            (
                sample_count,
                speed_mean, speed_std,
                heading_mean, heading_std,
                lat_centroid, lon_centroid,
                baseline_established,
                datetime.now(timezone.utc).isoformat(),
                mmsi
            )
        )
        conn.commit()

# ============================================================
# UPDATE ENTRY POINT (Called by Feature Factory)
# ============================================================
def maybe_update_profile(mmsi: str, point: Dict, model_score: float):
    """
    Conditionally add an incoming AIS point to the vessel's baseline sample.
    
    Args:
        mmsi: Vessel identifier.
        point: Dict with keys 'speed_over_ground_knots', 'heading_change_deg', 'lat', 'lon'.
        model_score: The anomaly score from the XGBoost model for the window containing this point.
                     Only update if score is confidently low (normal).
    """
    # 1. Score check
    if model_score >= SCORE_UPDATE_THRESHOLD:
        return  # Don't contaminate baseline with anomalous/ambiguous points
    
    # 2. Ensure vessel_profile row exists (insert if missing)
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR IGNORE INTO vessel_profile (mmsi) VALUES (?)",
            (mmsi,)
        )
        conn.commit()
    
    # 3. Insert the point into the sample
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO vessel_sample_points
            (mmsi, speed_over_ground_knots, heading_change_deg, lat, lon, observed_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                mmsi,
                point.get("speed_over_ground_knots"),
                point.get("heading_change_deg"),
                point.get("lat"),
                point.get("lon"),
                datetime.now(timezone.utc).isoformat()
            )
        )
        conn.commit()
    
    # 4. Enforce cap (evict oldest if over)
    _evict_oldest_if_over_cap(mmsi)
    
    # 5. Recompute aggregates from the current sample
    _recompute_aggregates(mmsi)

# ============================================================
# CONVENIENCE: Get static vessel attributes (if available)
# ============================================================
def update_static_attributes(mmsi: str, vessel_type: int, length: float, width: float):
    """Update the static attributes of a vessel (from AIS static reports)."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE vessel_profile
            SET vessel_type_code = ?,
                length_m = ?,
                width_m = ?,
                static_is_default = 0,
                last_updated = ?
            WHERE mmsi = ?
            """,
            (
                vessel_type,
                length,
                width,
                datetime.now(timezone.utc).isoformat(),
                mmsi
            )
        )
        conn.commit()

# ============================================================
# TEST / DEBUG
# ============================================================
if __name__ == "__main__":
    # Initialize the database
    init_db()
    
    # Simulate a new vessel
    mmsi = "TEST_123"
    
    # Simulate a low-score (normal) point
    normal_point = {
        "speed_over_ground_knots": 12.5,
        "heading_change_deg": 3.2,
        "lat": 21.3,
        "lon": -157.8,
    }
    low_score = 0.02  # Below threshold
    
    print(f"Adding normal point for {mmsi} (score={low_score})...")
    maybe_update_profile(mmsi, normal_point, low_score)
    
    # Check the profile
    profile = get_profile(mmsi)
    print("Profile:", profile)
    
    # Try to add a high-score (risky) point
    high_score = 0.8
    print(f"\nAttempting to add risky point (score={high_score})...")
    maybe_update_profile(mmsi, normal_point, high_score)  # Should be ignored
    
    # Check that point_count didn't increment
    profile2 = get_profile(mmsi)
    print("Profile after ignored point:", profile2)