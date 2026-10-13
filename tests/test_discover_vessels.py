"""Offline tests for mlops/discover_vessels.py (the watchlist builder)."""
import json
import sys
from pathlib import Path

MLOPS_DIR = Path(__file__).resolve().parent.parent / "mlops"
sys.path.insert(0, str(MLOPS_DIR))

import ais_consumer as ac
from discover_vessels import VesselTally, write_watchlist


def _pos(mmsi, sog, second, name="SHIP"):
    return {
        "MessageType": "PositionReport",
        "MetaData": {
            "MMSI": mmsi,
            "ShipName": f"{name}   ",
            "time_utc": f"2026-10-13 15:{second // 60:02d}:{second % 60:02d}.5 +0000 UTC",
        },
        "Message": {"PositionReport": {
            "UserID": mmsi, "Latitude": 21.3, "Longitude": -157.9,
            "Sog": sog, "Cog": 90.0, "Valid": True,
        }},
    }


def _tally(*msgs):
    t = VesselTally()
    for m in msgs:
        t.add(m)
    return t


def test_underway_vessels_outrank_moored_ones():
    t = _tally(
        *[_pos(111111111, 0.0, s * 60) for s in range(10)],   # moored, 10 reports
        *[_pos(222222222, 8.0, s * 60) for s in range(4)],    # under way, 4 reports
    )
    ranked = t.rank(top=5, min_underway=3)
    assert [r["mmsi"] for r in ranked] == ["222222222"]


def test_min_underway_filters_marginal_vessels():
    t = _tally(*[_pos(222222222, 8.0, s * 60) for s in range(2)])
    assert t.rank(top=5, min_underway=3) == []
    assert len(t.rank(top=5, min_underway=2)) == 1


def test_top_limits_and_orders_by_underway_reports():
    t = _tally(
        *[_pos(111111111, 5.0, s * 60) for s in range(3)],
        *[_pos(222222222, 5.0, s * 60) for s in range(6)],
        *[_pos(333333333, 5.0, s * 60) for s in range(4)],
    )
    assert [r["mmsi"] for r in t.rank(top=2, min_underway=1)] == ["222222222", "333333333"]


def test_unusable_and_non_position_messages_are_ignored():
    t = _tally(
        _pos(111111111, 102.3, 0),                       # speed not available
        {"MessageType": "ShipStaticData", "Message": {}},
        {"MessageType": "PositionReport", "MetaData": {}, "Message": {}},
    )
    assert t.vessels_seen == 0


def test_class_b_positions_are_counted():
    raw = _pos(444444444, 6.0, 0)
    raw["MessageType"] = "StandardClassBPositionReport"
    raw["Message"] = {"StandardClassBPositionReport": raw["Message"]["PositionReport"]}
    assert _tally(raw).vessels_seen == 1


def test_name_and_span_are_recorded():
    t = _tally(_pos(222222222, 8.0, 0, name="FERRY"), _pos(222222222, 8.0, 600, name="FERRY"))
    row = t.rank(top=1, min_underway=1)[0]
    assert row["name"] == "FERRY"
    assert row["span_min"] == 10.0


def test_written_watchlist_is_readable_by_the_consumer(tmp_path, monkeypatch):
    t = _tally(*[_pos(222222222, 8.0, s * 60) for s in range(4)])
    out = tmp_path / "watchlist.json"
    assert write_watchlist(out, t, minutes=8, top=25, min_underway=3) == 1
    assert json.loads(out.read_text())["mmsis"] == ["222222222"]

    monkeypatch.delenv("WATCHLIST_MMSIS", raising=False)
    monkeypatch.setenv("WATCHLIST_FILE", str(out))
    assert ac.load_watchlist() == ["222222222"]
