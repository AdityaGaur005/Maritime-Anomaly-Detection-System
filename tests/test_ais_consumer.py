"""
Unit tests for mlops/ais_consumer.py.

All tests are fully offline — no real WebSocket or HTTP connections are made.
Heavy ML dependencies (torch, xgboost) are NOT needed.
"""
import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from pydantic import ValidationError

MLOPS_DIR = Path(__file__).resolve().parent.parent / "mlops"
sys.path.insert(0, str(MLOPS_DIR))

from ais_consumer import (
    INITIAL_RECONNECT_SEC,
    MAX_RECONNECT_SEC,
    MAX_UNCONFIRMED_CONNECTIONS,
    MAX_WATCHLIST,
    QUEUE_MAXSIZE,
    AISStreamMessage,
    ConsumerStats,
    PositionThrottle,
    ShipStaticDataMessage,
    StaticDataReportMessage,
    SubscriptionRejected,
    as_position_report,
    build_subscription,
    consume_stream,
    handle_static_message,
    load_watchlist,
    parse_mmsis,
    position_is_usable,
    post_prediction,
    run_consumer,
    to_iso_utc,
    translate_to_predict_payload,
)
from kinematics import parse_timestamp

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "aisstream_samples.json"
AIS_TIME = "2026-10-13 15:01:50.32030298 +0000 UTC"  # exactly what aisstream.io sends


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
            "MMSI_String": mmsi,  # an int on the real wire, despite the name
            "ShipName": "TEST VESSEL   ",
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
                "Valid": True,
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
                "PartNumber": True,
                "ReportB": {
                    "ShipType": vessel_type,
                    "Valid": True,
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
                "PartNumber": False,  # Part A — name only
                "ReportA": {"Name": "SOME NAME", "Valid": True},
                # The real feed always includes this all-zero placeholder.
                "ReportB": {
                    "ShipType": 0,
                    "Valid": False,
                    "Dimension": {"A": 0, "B": 0, "C": 0, "D": 0},
                },
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
        assert msg.Message.StaticDataReport.ReportB.ShipType == 70
        assert msg.Message.StaticDataReport.ReportB.Dimension.A == 10

    def test_part_a_carries_only_an_invalid_placeholder_report_b(self):
        msg = StaticDataReportMessage.model_validate(_make_static_report_part_a())
        assert msg.Message.StaticDataReport.PartNumber is False
        assert msg.Message.StaticDataReport.ReportB.Valid is False

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
        # Part A's ReportB is an all-zero, Valid=false placeholder — nothing to persist
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
# run_consumer auth fail-fast (401 / 403)
# ===========================================================================
class TestRunConsumerAuthFail:
    """
    A 401 or 403 during the WebSocket handshake must cause run_consumer to
    return immediately without calling asyncio.sleep (i.e. no retry).

    We patch consume_stream directly — raising websockets.exceptions.InvalidStatus
    from the inner function — to avoid needing a real WebSocket connection.
    """

    async def _run_and_collect_sleeps(self, status_code: int) -> list:
        from contextlib import ExitStack

        import websockets.exceptions

        # Minimal response object: InvalidStatus only needs .status_code + .headers
        mock_response = type("R", (), {"status_code": status_code, "headers": {}})()

        sleep_calls: list = []

        async def _mock_sleep(delay: float) -> None:
            sleep_calls.append(delay)

        with ExitStack() as stack:
            stack.enter_context(patch("ais_consumer.AISSTREAM_API_KEY", "fake-key"))
            stack.enter_context(patch("ais_consumer.load_watchlist", return_value=[]))
            stack.enter_context(
                patch(
                    "ais_consumer.consume_stream",
                    AsyncMock(
                        side_effect=websockets.exceptions.InvalidStatus(mock_response)
                    ),
                )
            )
            stack.enter_context(patch("asyncio.sleep", _mock_sleep))
            mock_cls = stack.enter_context(patch("ais_consumer.httpx.AsyncClient"))
            mock_cls.return_value = AsyncMock()
            await run_consumer()

        return sleep_calls

    async def test_401_returns_without_retrying(self):
        sleep_calls = await self._run_and_collect_sleeps(401)
        assert sleep_calls == [], (
            f"run_consumer should not sleep/retry on 401, got: {sleep_calls}"
        )

    async def test_403_returns_without_retrying(self):
        sleep_calls = await self._run_and_collect_sleeps(403)
        assert sleep_calls == [], (
            f"run_consumer should not sleep/retry on 403, got: {sleep_calls}"
        )


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


# ===========================================================================
# Regression: real messages captured from the live aisstream.io feed
# (mocks built from assumptions hid a 100% validation failure -- keep these)
# ===========================================================================
@pytest.fixture(scope="module")
def real():
    return json.loads(FIXTURE.read_text())


class TestRealCaptures:
    def test_every_real_class_a_position_validates(self, real):
        assert real["PositionReport"], "fixture is empty"
        for raw in real["PositionReport"]:
            msg = AISStreamMessage.model_validate(raw)
            assert position_is_usable(msg.Message.PositionReport)

    def test_real_class_b_position_validates_after_normalising(self, real):
        raw = real["StandardClassBPositionReport"][0]
        msg = AISStreamMessage.model_validate(as_position_report(raw))
        assert msg.Message.PositionReport.UserID == raw["Message"]["StandardClassBPositionReport"]["UserID"]

    def test_real_timestamps_are_accepted_by_the_predict_pipeline(self, real):
        for raw in real["PositionReport"]:
            payload = translate_to_predict_payload(AISStreamMessage.model_validate(raw))
            assert parse_timestamp(payload["datetime_hst"]).tzinfo is not None

    def test_real_ship_static_data_is_persisted(self, real):
        raw = real["ShipStaticData"][0]
        d = raw["Message"]["ShipStaticData"]["Dimension"]
        with patch("ais_consumer.update_static_attributes") as upd:
            handle_static_message("ShipStaticData", raw)
        upd.assert_called_once_with(
            str(raw["MetaData"]["MMSI"]),
            vessel_type=raw["Message"]["ShipStaticData"]["Type"],
            length=float(d["A"] + d["B"]),
            width=float(d["C"] + d["D"]),
        )

    def test_real_part_a_placeholders_are_ignored(self, real):
        with patch("ais_consumer.update_static_attributes") as upd:
            for raw in real["StaticDataReport"]:
                if raw["Message"]["StaticDataReport"]["PartNumber"] is False:
                    handle_static_message("StaticDataReport", raw)
        upd.assert_not_called()


# ===========================================================================
# Timestamp conversion
# ===========================================================================
class TestToIsoUtc:
    def test_aisstream_format_is_converted(self):
        assert to_iso_utc(AIS_TIME) == "2026-10-13T15:01:50.320302+00:00"

    def test_nine_fractional_digits_truncate_to_six(self):
        assert to_iso_utc("2026-10-13 15:01:50.123456789 +0000 UTC").endswith(".123456+00:00")

    def test_short_fraction_is_padded_for_python_38(self):
        assert to_iso_utc("2026-10-13 15:01:50.5 +0000 UTC") == "2026-10-13T15:01:50.500000+00:00"

    def test_no_fraction(self):
        assert to_iso_utc("2026-10-13 15:01:50 +0000 UTC") == "2026-10-13T15:01:50+00:00"

    def test_iso_input_passes_through(self):
        assert to_iso_utc("2024-06-15T10:30:00Z") == "2024-06-15T10:30:00Z"

    def test_converted_value_parses(self):
        assert parse_timestamp(to_iso_utc(AIS_TIME)).year == 2026

    def test_garbage_timestamp_fails_validation(self):
        raw = _make_position_message(time_utc="yesterday-ish")
        with pytest.raises(ValidationError):
            AISStreamMessage.model_validate(raw)


# ===========================================================================
# Usable-position filter (AIS "not available" sentinels)
# ===========================================================================
class TestPositionIsUsable:
    def _pr(self, **over):
        raw = _make_position_message()
        raw["Message"]["PositionReport"].update(over)
        return AISStreamMessage.model_validate(raw).Message.PositionReport

    def test_normal_fix_is_usable(self):
        assert position_is_usable(self._pr())

    @pytest.mark.parametrize("over", [
        {"Sog": 102.3},          # speed not available
        {"Cog": 360.0},          # course not available
        {"Latitude": 91.0},      # position not available
        {"Longitude": 181.0},
        {"Sog": -1.0},
        {"Valid": False},
        {"Latitude": float("nan")},
    ])
    def test_sentinels_and_bad_fixes_rejected(self, over):
        assert not position_is_usable(self._pr(**over))


# ===========================================================================
# Per-vessel downsampling
# ===========================================================================
class TestPositionThrottle:
    def _t(self, seconds):
        return datetime(2026, 10, 13, 15, 0, 0, tzinfo=timezone.utc) + timedelta(seconds=seconds)

    def test_first_report_always_allowed(self):
        assert PositionThrottle(60).allow("111111111", self._t(0))

    def test_report_inside_interval_is_thinned(self):
        th = PositionThrottle(60)
        th.allow("111111111", self._t(0))
        assert not th.allow("111111111", self._t(10))

    def test_report_after_interval_is_allowed(self):
        th = PositionThrottle(60)
        th.allow("111111111", self._t(0))
        assert th.allow("111111111", self._t(60))

    def test_thinned_report_does_not_reset_the_clock(self):
        th = PositionThrottle(60)
        th.allow("111111111", self._t(0))
        th.allow("111111111", self._t(50))          # rejected
        assert th.allow("111111111", self._t(61))   # measured from t=0, not t=50

    def test_vessels_are_independent(self):
        th = PositionThrottle(60)
        th.allow("111111111", self._t(0))
        assert th.allow("222222222", self._t(1))

    def test_out_of_order_report_is_rejected(self):
        th = PositionThrottle(60)
        th.allow("111111111", self._t(100))
        assert not th.allow("111111111", self._t(40))


# ===========================================================================
# Watchlist + subscription
# ===========================================================================
class TestParseMmsis:
    def test_valid_and_deduplicated(self):
        assert parse_mmsis(["338796000", 367056982, "338796000"]) == ["338796000", "367056982"]

    @pytest.mark.parametrize("bad", ["12345", "1234567890", "abcdefghi", "", "36 705698"])
    def test_invalid_mmsi_rejected(self, bad):
        with pytest.raises(ValueError):
            parse_mmsis([bad])

    def test_over_limit_rejected(self):
        with pytest.raises(ValueError):
            parse_mmsis([str(100000000 + i) for i in range(MAX_WATCHLIST + 1)])

    def test_exactly_the_limit_is_fine(self):
        assert len(parse_mmsis([str(100000000 + i) for i in range(MAX_WATCHLIST)])) == MAX_WATCHLIST


class TestLoadWatchlist:
    def test_env_var_wins(self, monkeypatch, tmp_path):
        f = tmp_path / "w.json"
        f.write_text('["111111111"]')
        monkeypatch.setenv("WATCHLIST_FILE", str(f))
        monkeypatch.setenv("WATCHLIST_MMSIS", "222222222, 333333333")
        assert load_watchlist() == ["222222222", "333333333"]

    def test_plain_list_file(self, monkeypatch, tmp_path):
        monkeypatch.delenv("WATCHLIST_MMSIS", raising=False)
        f = tmp_path / "w.json"
        f.write_text('["111111111", "222222222"]')
        monkeypatch.setenv("WATCHLIST_FILE", str(f))
        assert load_watchlist() == ["111111111", "222222222"]

    def test_discovery_tool_file_format(self, monkeypatch, tmp_path):
        monkeypatch.delenv("WATCHLIST_MMSIS", raising=False)
        f = tmp_path / "w.json"
        f.write_text('{"mmsis": ["111111111"], "vessels": []}')
        monkeypatch.setenv("WATCHLIST_FILE", str(f))
        assert load_watchlist() == ["111111111"]

    def test_missing_file_means_whole_bounding_box(self, monkeypatch, tmp_path):
        monkeypatch.delenv("WATCHLIST_MMSIS", raising=False)
        monkeypatch.setenv("WATCHLIST_FILE", str(tmp_path / "nope.json"))
        assert load_watchlist() == []

    def test_bad_entry_raises(self, monkeypatch):
        monkeypatch.setenv("WATCHLIST_MMSIS", "not-an-mmsi")
        with pytest.raises(ValueError):
            load_watchlist()


class TestBuildSubscription:
    BOX = [[[18.5, -161.0], [22.8, -154.5]]]

    def test_uses_documented_apikey_field_name(self):
        assert "APIKey" in build_subscription("k", self.BOX)

    def test_no_mmsi_filter_without_watchlist(self):
        assert "FiltersShipMMSI" not in build_subscription("k", self.BOX)

    def test_mmsi_filter_added_with_watchlist(self):
        sub = build_subscription("k", self.BOX, ["111111111"])
        assert sub["FiltersShipMMSI"] == ["111111111"]

    def test_subscribes_to_class_a_class_b_and_static(self):
        types = set(build_subscription("k", self.BOX)["FilterMessageTypes"])
        assert {"PositionReport", "StandardClassBPositionReport", "ShipStaticData",
                "StaticDataReport"} <= types

    def test_message_types_override(self):
        sub = build_subscription("k", self.BOX, message_types=["PositionReport"])
        assert sub["FilterMessageTypes"] == ["PositionReport"]


# ===========================================================================
# consume_stream end-to-end with a fake WebSocket
# ===========================================================================
class _FakeWS:
    def __init__(self, messages, close_error=None):
        self._messages = messages
        self._close_error = close_error
        self.sent = []

    async def send(self, data):
        self.sent.append(json.loads(data))

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for m in self._messages:
            yield m if isinstance(m, str) else json.dumps(m)
        if self._close_error:
            raise self._close_error


class _FakeConnect:
    def __init__(self, ws):
        self.ws = ws

    async def __aenter__(self):
        return self.ws

    async def __aexit__(self, *exc):
        return False


def _closed_error():
    import websockets.exceptions as wse
    return wse.ConnectionClosedError(None, None)


CONFIRM = {"MessageType": "SubscriptionConfirmation"}


class TestConsumeStream:
    async def _run(self, messages, mmsis=(), close_error=None):
        ws = _FakeWS(messages, close_error)
        queue: asyncio.Queue = asyncio.Queue()
        stats = ConsumerStats()
        with patch("ais_consumer.AISSTREAM_API_KEY", "fake-key"), \
                patch("ais_consumer.websockets.connect", lambda *a, **k: _FakeConnect(ws)):
            await consume_stream(queue, stats, mmsis, PositionThrottle(60))
        return ws, queue, stats

    async def test_sends_subscription_with_watchlist(self):
        ws, _, _ = await self._run([CONFIRM], mmsis=["338796000"])
        sub = ws.sent[0]
        assert sub["APIKey"] == "fake-key"
        assert sub["FiltersShipMMSI"] == ["338796000"]

    async def test_real_messages_flow_through_to_the_queue(self, real):
        msgs = [CONFIRM, *real["PositionReport"], *real["StandardClassBPositionReport"]]
        _, queue, stats = await self._run(msgs)
        assert queue.qsize() == len(msgs) - 1
        assert stats.invalid == 0
        assert stats.throttled == 0
        payload = queue.get_nowait()
        assert set(payload) >= {"MMSI", "lat", "lon", "speed_over_ground_knots",
                                "course_over_ground_deg", "datetime_hst"}

    async def test_fast_reports_from_one_vessel_are_thinned(self):
        msgs = [
            CONFIRM,
            _make_position_message(time_utc="2026-10-13 15:00:00.1 +0000 UTC"),
            _make_position_message(time_utc="2026-10-13 15:00:05.1 +0000 UTC"),   # thinned
            _make_position_message(time_utc="2026-10-13 15:00:30.1 +0000 UTC"),   # thinned
            _make_position_message(time_utc="2026-10-13 15:01:05.1 +0000 UTC"),   # kept
        ]
        _, queue, stats = await self._run(msgs)
        assert queue.qsize() == 2
        assert stats.throttled == 2

    async def test_unusable_and_malformed_messages_are_counted_invalid(self):
        msgs = [
            CONFIRM,
            _make_position_message(mmsi=111111111, sog=102.3),  # speed not available
            "{not json",
            {"MessageType": "PositionReport", "MetaData": {}, "Message": {}},
        ]
        _, queue, stats = await self._run(msgs)
        assert queue.qsize() == 0
        assert stats.invalid == 3

    async def test_static_messages_update_the_profile_store(self, real):
        msgs = [CONFIRM, real["ShipStaticData"][0]]
        with patch("ais_consumer.update_static_attributes") as upd:
            _, queue, _ = await self._run(msgs)
        upd.assert_called_once()
        assert queue.qsize() == 0

    async def test_close_before_confirmation_is_a_rejected_subscription(self):
        with pytest.raises(SubscriptionRejected):
            await self._run([], close_error=_closed_error())

    async def test_clean_close_before_confirmation_is_also_rejected(self):
        with pytest.raises(SubscriptionRejected):
            await self._run([])

    async def test_close_after_confirmation_is_a_plain_connection_error(self):
        import websockets.exceptions as wse
        with pytest.raises(wse.ConnectionClosed):
            await self._run([CONFIRM], close_error=_closed_error())


# ===========================================================================
# run_consumer: real-world auth failure = connection dropped, never confirmed
# ===========================================================================
class TestRunConsumerSubscriptionRejected:
    async def test_stops_after_consecutive_unconfirmed_connections(self):
        sleeps = []

        async def _sleep(delay):
            sleeps.append(delay)

        consume = AsyncMock(side_effect=SubscriptionRejected("dropped"))
        with patch("ais_consumer.AISSTREAM_API_KEY", "fake-key"), \
                patch("ais_consumer.load_watchlist", return_value=[]), \
                patch("ais_consumer.consume_stream", consume), \
                patch("asyncio.sleep", _sleep), \
                patch("ais_consumer.httpx.AsyncClient") as cls:
            cls.return_value = AsyncMock()
            await run_consumer()
        assert consume.await_count == MAX_UNCONFIRMED_CONNECTIONS
        assert len(sleeps) == MAX_UNCONFIRMED_CONNECTIONS - 1   # no sleep after the final failure

    async def test_invalid_watchlist_exits_before_connecting(self):
        consume = AsyncMock()
        with patch("ais_consumer.AISSTREAM_API_KEY", "fake-key"), \
                patch("ais_consumer.load_watchlist", side_effect=ValueError("bad mmsi")), \
                patch("ais_consumer.consume_stream", consume):
            with pytest.raises(SystemExit):
                await run_consumer()
        consume.assert_not_awaited()

    async def test_missing_api_key_exits(self):
        with patch("ais_consumer.AISSTREAM_API_KEY", ""):
            with pytest.raises(SystemExit):
                await run_consumer()


class TestStaticWriteResilience:
    def test_database_error_does_not_propagate(self, real):
        import sqlite3
        with patch("ais_consumer.update_static_attributes",
                   side_effect=sqlite3.OperationalError("no such table: vessel_profile")):
            handle_static_message("ShipStaticData", real["ShipStaticData"][0])  # must not raise

    async def test_run_consumer_initialises_the_profile_db(self):
        with patch("ais_consumer.AISSTREAM_API_KEY", "fake-key"), \
                patch("ais_consumer.load_watchlist", return_value=[]), \
                patch("ais_consumer.init_db") as init, \
                patch("ais_consumer.consume_stream",
                      AsyncMock(side_effect=asyncio.CancelledError())), \
                patch("ais_consumer.httpx.AsyncClient") as cls:
            cls.return_value = AsyncMock()
            await run_consumer()
        init.assert_called_once()


class TestRejectionVersusNetworkOutage:
    async def _run(self, ws=None, connect_error=None):
        stats = ConsumerStats()

        class _Connect:
            async def __aenter__(self_inner):
                if connect_error:
                    raise connect_error
                return ws

            async def __aexit__(self_inner, *exc):
                return False

        with patch("ais_consumer.AISSTREAM_API_KEY", "fake-key"), \
                patch("ais_consumer.websockets.connect", lambda *a, **k: _Connect()):
            await consume_stream(asyncio.Queue(), stats, (), PositionThrottle(60))
        return stats

    async def test_connection_reset_after_subscribing_is_a_rejection(self):
        ws = _FakeWS([], close_error=ConnectionResetError(104, "Connection reset by peer"))
        with pytest.raises(SubscriptionRejected):
            await self._run(ws)

    async def test_failure_to_connect_is_not_a_rejection(self):
        """Network down must keep retrying forever, never trip the auth fail-fast."""
        with pytest.raises(OSError) as exc:
            await self._run(connect_error=OSError("Network is unreachable"))
        assert not isinstance(exc.value, SubscriptionRejected)

    async def test_confirmation_is_counted(self):
        ws = _FakeWS([CONFIRM])
        stats = ConsumerStats()
        with patch("ais_consumer.AISSTREAM_API_KEY", "fake-key"), \
                patch("ais_consumer.websockets.connect", lambda *a, **k: _FakeConnect(ws)):
            await consume_stream(asyncio.Queue(), stats, (), PositionThrottle(60))
        assert stats.confirmed == 1

    async def test_streak_survives_network_blips_between_rejections(self):
        """reject, network error, reject, reject -> still stops (network errors don't reset the streak)."""
        sleeps = []

        async def _sleep(d):
            sleeps.append(d)

        consume = AsyncMock(side_effect=[
            SubscriptionRejected("a"), OSError("blip"),
            SubscriptionRejected("b"), SubscriptionRejected("c"),
        ])
        with patch("ais_consumer.AISSTREAM_API_KEY", "fake-key"), \
                patch("ais_consumer.load_watchlist", return_value=[]), \
                patch("ais_consumer.consume_stream", consume), \
                patch("asyncio.sleep", _sleep), \
                patch("ais_consumer.httpx.AsyncClient") as cls:
            cls.return_value = AsyncMock()
            await run_consumer()
        assert consume.await_count == 4
