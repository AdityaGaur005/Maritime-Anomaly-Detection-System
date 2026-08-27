"""
Test client for the anomaly detection API.
Sends a sequence of simulated AIS points and prints the responses.
"""
import requests
import json
from datetime import datetime, timezone
import time

API_URL = "http://localhost:8000/predict"
MMSI = "TEST_MMSI"

# Generate a sequence of points (same as the smoke test in feature_factory)
def generate_points():
    base_ts = datetime.now(timezone.utc).timestamp()
    points = []
    # Normal sailing (30 points)
    for i in range(30):
        lon = -157.8 + i * 0.001
        points.append({
            "MMSI": MMSI,
            "lat": 21.3,
            "lon": lon,
            "speed_over_ground_knots": 12.0,
            "course_over_ground_deg": 90.0,
            "datetime_hst": datetime.fromtimestamp(base_ts + i * 90, tz=timezone.utc).isoformat(),
        })
    # Loss of propulsion (10 points)
    for i in range(30, 40):
        lon = -157.8 + i * 0.001
        points.append({
            "MMSI": MMSI,
            "lat": 21.3,
            "lon": lon,
            "speed_over_ground_knots": 0.3,
            "course_over_ground_deg": 90.0,
            "datetime_hst": datetime.fromtimestamp(base_ts + i * 90, tz=timezone.utc).isoformat(),
        })
    # Sharp turn (10 points)
    for i in range(40, 50):
        lon = -157.8 + 40 * 0.001 + (i - 40) * 0.00003
        points.append({
            "MMSI": MMSI,
            "lat": 21.3,
            "lon": lon,
            "speed_over_ground_knots": 0.3,
            "course_over_ground_deg": 180.0 + (i - 40) * 9.0,
            "datetime_hst": datetime.fromtimestamp(base_ts + i * 90, tz=timezone.utc).isoformat(),
        })
    return points

def main():
    points = generate_points()
    print(f"Sending {len(points)} points...")
    for i, pt in enumerate(points, 1):
        resp = requests.post(API_URL, json=pt)
        if resp.status_code != 200:
            print(f"Point {i}: Error {resp.status_code} - {resp.text}")
            continue
        data = resp.json()
        if data["anomaly_score"] is None:
            print(f"Point {i}: Not ready yet (buffer < 30).")
        else:
            print(f"Point {i}: Score = {data['anomaly_score']:.4f}, Confidence = {data['confidence']}, Is anomaly = {data['is_anomaly']}")

if __name__ == "__main__":
    main()