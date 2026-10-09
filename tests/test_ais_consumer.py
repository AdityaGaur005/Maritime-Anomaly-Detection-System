"""
Unit tests for mlops/ais_consumer.py.

All tests are fully offline — no real WebSocket or HTTP connections are made.
Heavy ML dependencies (torch, xgboost) are NOT needed.
"""
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from pydantic import ValidationError

MLOPS_DIR = Path(__file__).resolve().parent.parent / "mlops"
sys.path.insert(0, str(MLOPS_DIR))

from ais_consumer import (
    AISStreamMessage,
    ConsumerStats,
    INITIAL_RECONNECT_SEC,
    MAX_RECONNECT_SEC,
    QUEUE_MAXSIZE,
    ShipStaticDataMessage,
    StaticDataReportMessage,
    handle_static_message,
    post_prediction,
    translate_to_predict_payload,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_position_message(
    mmsi: int = 123456789,
    lat: float = 21.3,
    lon: float = -157.8,
    sog: float = 12.0,
    cog: float = 90.0,
    time_utc: str = "2024-01-01T00:00:00Z",
) -> Dict[str, Any]:
    return {
        "MessageType": "PositionReport",
        "MetaData": {
            "MMSI": mmsi,
            "MMSI_String": str(mmsi),
            "ShipName": "TEST VESSEL",
            "time_utc": time_utc,
        },
        "Message": {
            "PositionReport": {
                "UserID": mmsi,
                "Latitude": lat,
                "Longitude": lon,
                "Sog": sog,
                "Cog": cog,
                "TrueHeading": 90,
                "NavigationalStatus": 0,
            }
        },
    }


def _make_ship_static_message(
    mmsi: int = 123456789,
    vessel_type: int = 70,
    bow: int = 10,
    stern: int = 140,
    port: int = 8,
    starboard: int = 17,
) -> Dict[str, Any]:
    return {
        "MessageType": "ShipStaticData",
        "MetaData": {"MMSI": mmsi, "time_utc": "2024-01-01T00:00:00Z"},
        "Message": {
            "ShipStaticData": {
                "UserID": mmsi,
                "Name": "CARGO SHIP",
                "Type": vessel_type,
                "Dimension": {"A": bow, "B": stern, "C": port, "D": starboard},
            }
        },
    }


def _make_static_report_part_b(
    mmsi: int = 123456789,
    vessel_type: int = 70,
    bow: int = 10,
    stern: int = 140,
    port: int = 8,
    starboard: int = 17,
) -> Dict[str, Any]:
    return {
        "MessageType": "StaticDataReport",
        "MetaData": {"MMSI": mmsi, "time_utc": "2024-01-01T00:00:00Z"},
        "Message": {
            "StaticDataReport": {
                "UserID": mmsi,
                "PartNumber": 1,
                "ReportB": {
                    "TypeOfShipAndCargoType": vessel_type,
                    "Dimension": {"A": bow, "B": stern, "C": port, "D": starboard},
                },
            }
        },
    }


def _make_static_report_part_a(mmsi: int = 123456789) -> Dict[str, Any]:
    return {
        "MessageType": "StaticDataReport",
        "MetaData": {"MMSI": mmsi, "time_utc": "2024-01-01T00:00:00Z"},
        "Message": {
            "StaticDataReport": {
                "UserID": mmsi,
                "PartNumber": 0,  # Part A — name only, no dimensions
            }
        },
    }


def _make_mock_response(status_code: int, body: Dict[str, Any]) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = body
    return resp


# ===========================================================================
# AISStreamMessage (PositionReport)
# ===========================================================================
class TestAISStreamMessageValidation:
    def test_valid_position_report_parsed(self):
        msg = AISStreamMessage.model_validate(_make_position_message())
        assert msg.MetaData.MMSI == 123456789
        assert msg.Message.PositionReport.Latitude == pytest.approx(21.3)
        assert msg.Message.PositionReport.Sog == pytest.approx(12.0)

    def test_wrong_message_type_raises(self):
        raw = _make_position_message()
        raw["MessageType"] = "SafetyMessage"
        with pytest.raises(ValidationError) as exc_info:
            AISStreamMessage.model_validate(raw)
        assert any("PositionReport" in str(e) for e in exc_info.value.errors())

    def test_missing_mmsi_raises(self):
        raw = _make_position_message()
        del raw["MetaData"]["MMSI"]
        with pytest.raises(ValidationError):
            AISStreamMessage.model_validate(raw)

    def test_missing_latitude_raises(self):
        raw = _make_position_message()
        del raw["Message"]["PositionReport"]["Latitude"]
        with pytest.raises(ValidationError):
            AISStreamMessage.model_validate(raw)

    def test_optional_fields_absent(self):
        raw = _make_position_message()
        del raw["MetaData"]["ShipName"]
        del raw["Message"]["PositionReport"]["TrueHeading"]
        msg = AISStreamMessage.model_validate(raw)
        assert msg.MetaData.ShipName is None
        assert msg.Message.PositionReport.TrueHeading is None


# ===========================================================================
# ShipStaticDataMessage validation
# ===========================================================================
class TestShipStaticDataMessageValidation:
    def test_valid_ship_static_parsed(self):
        msg = ShipStaticDataMessage.model_validate(_make_ship_static_message())
        assert msg.MetaData.MMSI == 123456789
        assert msg.Message.ShipStaticData.Type == 70
        assert msg.Message.ShipStaticData.Dimension.A == 10
        assert msg.Message.ShipStaticData.Dimension.B == 140

    def test_dimension_optional(self):
        raw = _make_ship_static_message()
        del raw["Message"]["ShipStaticData"]["Dimension"]
        msg = ShipStaticDataMessage.model_validate(raw)
        assert msg.Message.ShipStaticData.Dimension is None

    def test_type_optional(self):
        raw = _make_ship_static_message()
        del raw["Message"]["ShipStaticData"]["Type"]
        msg = ShipStaticDataMessage.model_validate(raw)
        assert msg.Message.ShipStaticData.Type is None

    def test_missing_userid_raises(self):
        raw = _make_ship_static_message()
        del raw["Message"]["ShipStaticData"]["UserID"]
        with pytest.raises(ValidationError):
            ShipStaticDataMessage.model_validate(raw)


# ===========================================================================
# StaticDataReportMessage validation
# ===========================================================================
class TestStaticDataReportMessageValidation:
    def test_valid_part_b_parsed(self):
        msg = StaticDataReportMessage.model_validate(_make_static_report_part_b())
        assert msg.Message.StaticDataReport.ReportB.TypeOfShipAndCargoType == 70
        assert msg.Message.StaticDataReport.ReportB.Dimension.A == 10

    def test_part_a_has_no_report_b(self):
        msg = StaticDataReportMessage.model_validate(_make_static_report_part_a())
        assert msg.Message.StaticDataReport.ReportB is None
        assert msg.Message.StaticDataReport.PartNumber == 0

    def test_missing_userid_raises(self):
        raw = _make_static_report_part_b()
        del raw["Message"]["StaticDataReport"]["UserID"]
        with pytest.raises(ValidationError):
            StaticDataReportMessage.model_validate(raw)


# ===========================================================================
# handle_static_message
# ===========================================================================
class TestHandleStaticMessage:
    def test_ship_static_data_calls_update(self):
        with patch("ais_consumer.update_static_attributes") as mock_update:
            handle_static_message("ShipStaticData", _make_ship_static_message())
        mock_update.assert_called_once_with(
            "123456789",
            vessel_type=70,
            length=pytest.approx(150.0),  # 10 + 140
            width=pytest.approx(25.0),    # 8 + 17
        )

    def test_static_report_part_b_calls_update(self):
        with patch("ais_consumer.update_static_attributes") as mock_update:
            handle_static_message("StaticDataReport", _make_static_report_part_b())
        mock_update.assert_called_once_with(
            "123456789",
            vessel_type=70,
            length=pytest.approx(150.0),
            width=pytest.approx(25.0),
        )

    def test_static_report_part_a_does_not_call_update(self):
        # Part A has no dimensions or type — nothing useful to persist
        with patch("ais_consumer.update_static_attributes") as mock_update:
            handle_static_message("StaticDataReport", _make_static_report_part_a())
        mock_update.assert_not_called()

    def test_unknown_type_is_silently_ignored(self):
        with patch("ais_consumer.update_static_attributes") as mock_update:
            handle_static_message("VoyageData", {"MessageType": "VoyageData"})
        mock_update.assert_not_called()

    def test_validation_error_does_not_raise(self):
        # Malformed data should log a warning and return, not crash
        with patch("ais_consumer.update_static_attributes") as mock_update:
            handle_static_message("ShipStaticData", {"MessageType": "ShipStaticData", "Message": {}})
        mock_update.assert_not_called()

    def test_zero_dimensions_does_not_call_update(self):
        raw = _make_ship_static_message(bow=0, stern=0, port=0, starboard=0)
        raw["Message"]["ShipStaticData"]["Type"] = None
        with patch("ais_consumer.update_static_attributes") as mock_update:
            handle_static_message("ShipStaticData", raw)
        mock_update.assert_not_called()

    def test_mmsi_converted_to_string(self):
        """MMSI must be passed as a string to update_static_attributes."""
        with patch("ais_consumer.update_static_attributes") as mock_update:
            handle_static_message("ShipStaticData", _make_ship_static_message(mmsi=987654321))
        args, _ = mock_update.call_args
        assert isinstance(args[0], str)
        assert args[0] == "987654321"


# ===========================================================================
# translate_to_predict_payload
# ===========================================================================
class TestTranslateToPayload:
    def test_mmsi_converted_to_string(self):
        msg = AISStreamMessage.model_validate(_make_position_message(mmsi=123456789))
        payload = translate_to_predict_payload(msg)
        assert payload["MMSI"] == "123456789"
        assert isinstance(payload["MMSI"], str)

    def test_all_required_keys_present(self):
        msg = AISStreamMessage.model_validate(_make_position_message())
        payload = translate_to_predict_payload(msg)
        for key in ("MMSI", "lat", "lon", "speed_over_ground_knots",
                    "course_over_ground_deg", "datetime_hst"):
            assert key in payload, f"Missing key: {key}"

    def test_values_round_trip_correctly(self):
        msg = AISStreamMessage.model_validate(
            _make_position_message(lat=21.31, lon=-157.85, sog=8.4, cog=270.0)
        )
        payload = translate_to_predict_payload(msg)
        assert payload["lat"] == pytest.approx(21.31)
        assert payload["lon"] == pytest.approx(-157.85)
        assert payload["speed_over_ground_knots"] == pytest.approx(8.4)
        assert payload["course_over_ground_deg"] == pytest.approx(270.0)

    def test_timestamp_passed_through(self):
        ts = "2024-06-15T10:30:00Z"
        msg = AISStreamMessage.model_validate(_make_position_message(time_utc=ts))
        payload = translate_to_predict_payload(msg)
        assert payload["datetime_hst"] == ts


# ===========================================================================
# post_prediction
# ===========================================================================
class TestPostPrediction:
    def _payload(self) -> Dict[str, Any]:
        return {
            "MMSI": "123456789",
            "lat": 21.3,
            "lon": -157.8,
            "speed_over_ground_knots": 12.0,
            "course_over_ground_deg": 90.0,
            "datetime_hst": "2024-01-01T00:00:00Z",
        }

    async def test_successful_non_anomaly(self):
        stats = ConsumerStats()
        client = AsyncMock()
        client.post.return_value = _make_mock_response(
            200, {"mmsi": "123456789", "is_anomaly": False, "anomaly_score": 0.3}
        )
        await post_prediction(client, self._payload(), stats)
        assert stats.posted == 1
        assert stats.anomalies == 0
        assert stats.server_errors == 0

    async def test_anomaly_increments_counter(self):
        stats = ConsumerStats()
        client = AsyncMock()
        client.post.return_value = _make_mock_response(
            200, {"mmsi": "123456789", "is_anomaly": True, "anomaly_score": 0.91}
        )
        await post_prediction(client, self._payload(), stats)
        assert stats.anomalies == 1

    async def test_422_counts_as_server_error(self):
        stats = ConsumerStats()
        client = AsyncMock()
        client.post.return_value = _make_mock_response(
            422, {"detail": [{"msg": "value is not a valid float"}]}
        )
        await post_prediction(client, self._payload(), stats)
        assert stats.posted == 1
        assert stats.server_errors == 1

    async def test_500_counts_as_server_error(self):
        stats = ConsumerStats()
        client = AsyncMock()
        client.post.return_value = _make_mock_response(500, {})
        await post_prediction(client, self._payload(), stats)
        assert stats.server_errors == 1

    async def test_timeout_counts_as_server_error(self):
        import httpx
        stats = ConsumerStats()
        client = AsyncMock()
        client.post.side_effect = httpx.TimeoutException("timed out", request=None)
        await post_prediction(client, self._payload(), stats)
        assert stats.posted == 0
        assert stats.server_errors == 1

    async def test_request_error_counts_as_server_error(self):
        import httpx
        stats = ConsumerStats()
        client = AsyncMock()
        client.post.side_effect = httpx.ConnectError("refused", request=None)
        await post_prediction(client, self._payload(), stats)
        assert stats.server_errors == 1


# ===========================================================================
# ConsumerStats
# ===========================================================================
class TestConsumerStats:
    def test_initial_values_zero(self):
        stats = ConsumerStats()
        assert stats.received == 0
        assert stats.invalid == 0
        assert stats.posted == 0
        assert stats.anomalies == 0
        assert stats.server_errors == 0
        assert stats.dropped == 0

    def test_dropped_field_exists(self):
        stats = ConsumerStats()
        stats.dropped += 1
        assert stats.dropped == 1

    def test_log_includes_dropped(self, caplog):
        import logging
        stats = ConsumerStats()
        stats.received = 1000
        stats.dropped = 5
        with caplog.at_level(logging.INFO, logger="ais_consumer"):
            stats.log()
        assert "dropped=5" in caplog.text

    def test_log_does_not_raise(self, caplog):
        import logging
        stats = ConsumerStats()
        stats.received = 500
        with caplog.at_level(logging.INFO, logger="ais_consumer"):
            stats.log()
        assert "consumer_stats" in caplog.text


# ===========================================================================
# Queue overflow: drop behaviour
# ===========================================================================
class TestQueueDrop:
    async def test_queue_full_increments_dropped(self):
        """When the queue is full, put_nowait raises QueueFull; consumer counts it."""
        stats = ConsumerStats()
        queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        # Pre-fill the queue
        await queue.put({"MMSI": "111"})
        assert queue.full()

        # Simulate the drop path directly
        import asyncio as _asyncio
        payload = {"MMSI": "222"}
        try:
            queue.put_nowait(payload)
        except _asyncio.QueueFull:
            stats.dropped += 1

        assert stats.dropped == 1
        assert queue.qsize() == 1  # original item still there

    def test_queue_maxsize_constant(self):
        assert QUEUE_MAXSIZE > 0


# ===========================================================================
# Backoff constants
# ===========================================================================
def test_backoff_constants():
    assert INITIAL_RECONNECT_SEC == 1.0
    assert MAX_RECONNECT_SEC == 60.0
    backoff = INITIAL_RECONNECT_SEC
    for _ in range(20):
        backoff = min(backoff * 2, MAX_RECONNECT_SEC)
    assert backoff == MAX_RECONNECT_SEC
