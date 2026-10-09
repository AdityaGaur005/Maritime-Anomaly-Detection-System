"""
Live AIS ingestion consumer for the Maritime Anomaly Detection System.

Connects to wss://stream.aisstream.io/v0/stream, subscribes to PositionReport
and static vessel data messages (ShipStaticData / StaticDataReport), validates
each message, and either POSTs to /predict (position) or updates the vessel
profile store (static).

Architecture:
    WebSocket reader  →  asyncio.Queue(maxsize=QUEUE_MAXSIZE)
    N worker tasks    ←  drain queue, POST to /predict

Backpressure: if the queue is full (API slower than stream), the oldest item
is NOT dropped — the incoming item is dropped and counted in stats.dropped.
For the Hawaiian-waters demo bounding box (~10 vessels at a time) this should
never trigger; document it as a known limitation for a larger bounding box.

Reconnection: exponential backoff 1 s → 2 s → … → MAX_RECONNECT_SEC on
WebSocket errors.  Resets to 1 s on a clean server-side close.

Auth fail-fast: if the WebSocket handshake returns HTTP 401 or 403, the
consumer logs a FATAL error and exits rather than retrying indefinitely.

Usage:
    AISSTREAM_API_KEY=<key> python -m mlops.ais_consumer

Environment variables:
    AISSTREAM_API_KEY   Required. Your aisstream.io API key.
    AISSTREAM_WS_URL    Default wss://stream.aisstream.io/v0/stream
    PREDICT_URL         Default http://localhost:8000/predict
    BOUNDING_BOXES      JSON list e.g. [[[17.5,-162],[23.5,-153]]]
    REQUEST_TIMEOUT_SEC Default 5
    MAX_RECONNECT_SEC   Default 60
    WORKER_COUNT        Default 3  — parallel /predict posting workers
    QUEUE_MAXSIZE       Default 500
"""

import asyncio
import json
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import websockets
import websockets.exceptions
from pydantic import BaseModel, Field, ValidationError, field_validator

# vessel_profile_store lives alongside this module in mlops/
_MLOPS_DIR = Path(__file__).resolve().parent
if str(_MLOPS_DIR) not in sys.path:
    sys.path.insert(0, str(_MLOPS_DIR))

from vessel_profile_store import update_static_attributes

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
AISSTREAM_API_KEY: str = os.environ.get("AISSTREAM_API_KEY", "")
AISSTREAM_WS_URL: str = os.environ.get(
    "AISSTREAM_WS_URL", "wss://stream.aisstream.io/v0/stream"
)
PREDICT_URL: str = os.environ.get("PREDICT_URL", "http://localhost:8000/predict")

# Hawaiian waters [[lat_min, lon_min], [lat_max, lon_max]]
_DEFAULT_BOXES = json.dumps([[[17.5, -162.0], [23.5, -153.0]]])
BOUNDING_BOXES: List[List[List[float]]] = json.loads(
    os.environ.get("BOUNDING_BOXES", _DEFAULT_BOXES)
)

REQUEST_TIMEOUT_SEC: float = float(os.environ.get("REQUEST_TIMEOUT_SEC", "5"))
MAX_RECONNECT_SEC: float = float(os.environ.get("MAX_RECONNECT_SEC", "60"))
INITIAL_RECONNECT_SEC: float = 1.0
WORKER_COUNT: int = int(os.environ.get("WORKER_COUNT", "3"))
QUEUE_MAXSIZE: int = int(os.environ.get("QUEUE_MAXSIZE", "500"))

_STATS_LOG_INTERVAL: int = 1000
_STATIC_MSG_TYPES = frozenset({"ShipStaticData", "StaticDataReport"})
_SUBSCRIBE_TYPES = ["PositionReport", "ShipStaticData", "StaticDataReport"]


# ---------------------------------------------------------------------------
# Shared sub-models
# ---------------------------------------------------------------------------
class _MetaData(BaseModel):
    MMSI: int
    MMSI_String: Optional[str] = None
    ShipName: Optional[str] = None
    time_utc: str


class _Dimension(BaseModel):
    """AIS antenna-to-hull distances in metres. A+B = length, C+D = beam."""
    A: Optional[int] = None  # bow
    B: Optional[int] = None  # stern
    C: Optional[int] = None  # port
    D: Optional[int] = None  # starboard


# ---------------------------------------------------------------------------
# PositionReport message (AIS types 1 / 2 / 3)
# ---------------------------------------------------------------------------
class _PositionReport(BaseModel):
    UserID: int
    Latitude: float
    Longitude: float
    Sog: float = Field(..., description="Speed over ground, knots")
    Cog: float = Field(..., description="Course over ground, degrees")
    TrueHeading: Optional[int] = None
    NavigationalStatus: Optional[int] = None


class _PositionReportMsg(BaseModel):
    PositionReport: _PositionReport


class AISStreamMessage(BaseModel):
    """Validated PositionReport message from aisstream.io."""

    MessageType: str
    MetaData: _MetaData
    Message: _PositionReportMsg

    @field_validator("MessageType")
    @classmethod
    def must_be_position_report(cls, v: str) -> str:
        if v != "PositionReport":
            raise ValueError(f"Expected PositionReport, got {v!r}")
        return v


# ---------------------------------------------------------------------------
# ShipStaticData message (AIS type 5 — voyage / static data)
# ---------------------------------------------------------------------------
class _ShipStaticDataPayload(BaseModel):
    UserID: int
    Name: Optional[str] = None
    Type: Optional[int] = None
    Dimension: Optional[_Dimension] = None


class _ShipStaticDataMsg(BaseModel):
    ShipStaticData: _ShipStaticDataPayload


class ShipStaticDataMessage(BaseModel):
    """Validated ShipStaticData (Type 5) message from aisstream.io."""

    MessageType: str
    MetaData: _MetaData
    Message: _ShipStaticDataMsg


# ---------------------------------------------------------------------------
# StaticDataReport message (AIS type 24 — Class B static report)
# ---------------------------------------------------------------------------
class _StaticReportB(BaseModel):
    TypeOfShipAndCargoType: Optional[int] = None
    Dimension: Optional[_Dimension] = None


class _StaticDataReportPayload(BaseModel):
    UserID: int
    PartNumber: Optional[int] = None  # 0=Part A (name), 1=Part B (type/dims)
    ReportB: Optional[_StaticReportB] = None


class _StaticDataReportMsg(BaseModel):
    StaticDataReport: _StaticDataReportPayload


class StaticDataReportMessage(BaseModel):
    """Validated StaticDataReport (Type 24) message from aisstream.io."""

    MessageType: str
    MetaData: _MetaData
    Message: _StaticDataReportMsg


# ---------------------------------------------------------------------------
# Translation helpers
# ---------------------------------------------------------------------------
def translate_to_predict_payload(msg: AISStreamMessage) -> Dict[str, Any]:
    """Convert a validated AISStreamMessage to the /predict request body."""
    pr = msg.Message.PositionReport
    return {
        "MMSI": str(msg.MetaData.MMSI),
        "lat": pr.Latitude,
        "lon": pr.Longitude,
        "speed_over_ground_knots": pr.Sog,
        "course_over_ground_deg": pr.Cog,
        "datetime_hst": msg.MetaData.time_utc,
    }


def _sum_dim(dim: Optional[_Dimension], key_a: str, key_b: str) -> Optional[float]:
    """Sum two optional dimension components; return None if both are 0/missing."""
    if dim is None:
        return None
    a = getattr(dim, key_a) or 0
    b = getattr(dim, key_b) or 0
    return float(a + b) if (a + b) > 0 else None


def handle_static_message(msg_type: str, data: Dict[str, Any]) -> None:
    """
    Parse a ShipStaticData or StaticDataReport message and persist vessel
    type, length, and beam to the profile store.

    Ships broadcast static data infrequently (~every 6 minutes), so this
    runs synchronously in the WS reader loop — the SQLite write is fast
    enough that it won't stall the stream.
    """
    mmsi: Optional[str] = None
    vessel_type: Optional[int] = None
    length: Optional[float] = None
    width: Optional[float] = None

    try:
        if msg_type == "ShipStaticData":
            msg = ShipStaticDataMessage.model_validate(data)
            sd = msg.Message.ShipStaticData
            mmsi = str(msg.MetaData.MMSI)
            vessel_type = sd.Type
            length = _sum_dim(sd.Dimension, "A", "B")
            width = _sum_dim(sd.Dimension, "C", "D")

        elif msg_type == "StaticDataReport":
            msg = StaticDataReportMessage.model_validate(data)
            sd = msg.Message.StaticDataReport
            mmsi = str(msg.MetaData.MMSI)
            # Only Part B carries type and dimensions
            if sd.ReportB:
                vessel_type = sd.ReportB.TypeOfShipAndCargoType
                length = _sum_dim(sd.ReportB.Dimension, "A", "B")
                width = _sum_dim(sd.ReportB.Dimension, "C", "D")
        else:
            return

    except ValidationError as exc:
        logger.warning(
            "static_validation_error msg_type=%s errors=%s", msg_type, exc.errors()
        )
        return

    if mmsi and any(v is not None for v in (vessel_type, length, width)):
        update_static_attributes(mmsi, vessel_type=vessel_type, length=length, width=width)
        logger.debug(
            "static_updated mmsi=%s type=%s length=%s width=%s",
            mmsi, vessel_type, length, width,
        )


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
class ConsumerStats:
    def __init__(self) -> None:
        self.received: int = 0
        self.invalid: int = 0
        self.posted: int = 0
        self.anomalies: int = 0
        self.server_errors: int = 0
        self.dropped: int = 0  # queue-overflow drops

    def log(self) -> None:
        logger.info(
            "consumer_stats received=%d invalid=%d posted=%d"
            " anomalies=%d server_errors=%d dropped=%d",
            self.received, self.invalid, self.posted,
            self.anomalies, self.server_errors, self.dropped,
        )


_stats = ConsumerStats()


# ---------------------------------------------------------------------------
# HTTP posting
# ---------------------------------------------------------------------------
async def post_prediction(
    client: httpx.AsyncClient,
    payload: Dict[str, Any],
    stats: ConsumerStats = _stats,
) -> None:
    """POST one AIS point to the prediction API. Logs and returns on any error."""
    mmsi = payload.get("MMSI", "?")
    try:
        resp = await client.post(PREDICT_URL, json=payload, timeout=REQUEST_TIMEOUT_SEC)
        stats.posted += 1

        if resp.status_code == 422:
            logger.warning(
                "validation_rejected mmsi=%s detail=%s",
                mmsi, resp.json().get("detail"),
            )
            stats.server_errors += 1
            return

        if resp.status_code >= 400:
            logger.error("server_error status=%d mmsi=%s", resp.status_code, mmsi)
            stats.server_errors += 1
            return

        result = resp.json()
        if result.get("is_anomaly"):
            stats.anomalies += 1
            logger.warning(
                "ANOMALY_DETECTED mmsi=%s score=%.4f",
                result.get("mmsi", mmsi),
                result.get("anomaly_score") or -1.0,
            )

    except httpx.TimeoutException:
        logger.warning("predict_timeout mmsi=%s", mmsi)
        stats.server_errors += 1
    except httpx.RequestError as exc:
        logger.error("predict_request_error mmsi=%s error=%s", mmsi, exc)
        stats.server_errors += 1


# ---------------------------------------------------------------------------
# Worker: drains the prediction queue
# ---------------------------------------------------------------------------
async def _worker(
    queue: "asyncio.Queue[Dict[str, Any]]",
    client: httpx.AsyncClient,
    stats: ConsumerStats,
) -> None:
    while True:
        payload = await queue.get()
        try:
            await post_prediction(client, payload, stats)
        finally:
            queue.task_done()


# ---------------------------------------------------------------------------
# WebSocket producer (single connection lifetime)
# ---------------------------------------------------------------------------
async def consume_stream(
    queue: "asyncio.Queue[Dict[str, Any]]",
    stats: ConsumerStats = _stats,
) -> None:
    """
    Open one WebSocket connection and push PositionReport payloads onto the
    queue. Static messages are handled synchronously (fast SQLite write).

    Returns normally on a clean server-side close.
    Raises websockets.exceptions.WebSocketException on connection failure.
    Raises websockets.exceptions.InvalidStatus on non-101 HTTP responses
    (e.g. 401 auth failure — caller checks and exits on 401/403).
    """
    async with websockets.connect(AISSTREAM_WS_URL) as ws:
        subscription = {
            "APIkey": AISSTREAM_API_KEY,
            "BoundingBoxes": BOUNDING_BOXES,
            "FilterMessageTypes": _SUBSCRIBE_TYPES,
        }
        await ws.send(json.dumps(subscription))
        logger.info("aisstream_subscribed url=%s boxes=%s", AISSTREAM_WS_URL, BOUNDING_BOXES)

        async for raw_message in ws:
            stats.received += 1

            try:
                data = json.loads(raw_message)
            except json.JSONDecodeError as exc:
                logger.warning("json_decode_error error=%s", exc)
                stats.invalid += 1
                continue

            msg_type = data.get("MessageType", "")

            # Static data: synchronous SQLite write (~1 ms), no queue needed
            if msg_type in _STATIC_MSG_TYPES:
                handle_static_message(msg_type, data)
                continue

            if msg_type != "PositionReport":
                continue

            try:
                msg = AISStreamMessage.model_validate(data)
            except ValidationError as exc:
                logger.warning("position_validation_error errors=%s", exc.errors())
                stats.invalid += 1
                continue

            payload = translate_to_predict_payload(msg)

            # Drop incoming item on overflow rather than blocking the WS reader
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                stats.dropped += 1
                logger.warning("queue_full dropped_mmsi=%s queue_size=%d", payload.get("MMSI"), queue.qsize())

            if stats.received % _STATS_LOG_INTERVAL == 0:
                stats.log()


# ---------------------------------------------------------------------------
# Reconnect loop + worker lifecycle
# ---------------------------------------------------------------------------
async def run_consumer() -> None:
    """
    Run the AIS consumer indefinitely.

    Worker tasks live for the consumer's full lifetime (across reconnects).
    The WebSocket reader is restarted on failure using exponential backoff.
    """
    if not AISSTREAM_API_KEY:
        logger.error(
            "AISSTREAM_API_KEY is not set. "
            "Export it as an environment variable before starting."
        )
        sys.exit(1)

    queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
    backoff = INITIAL_RECONNECT_SEC

    async with httpx.AsyncClient() as client:
        workers = [
            asyncio.create_task(_worker(queue, client, _stats))
            for _ in range(WORKER_COUNT)
        ]
        try:
            while True:
                try:
                    await consume_stream(queue)
                    # Clean server-side close
                    logger.info(
                        "websocket_closed_cleanly reconnecting_in=%.1fs",
                        INITIAL_RECONNECT_SEC,
                    )
                    backoff = INITIAL_RECONNECT_SEC

                except websockets.exceptions.InvalidStatus as exc:
                    # Non-101 HTTP response during handshake
                    status = exc.response.status_code
                    if status in (401, 403):
                        logger.error(
                            "auth_failed status=%d — "
                            "AISSTREAM_API_KEY is invalid or revoked; not retrying",
                            status,
                        )
                        return  # no point retrying an auth failure
                    logger.warning(
                        "websocket_handshake_error status=%d reconnecting_in=%.1fs",
                        status, backoff,
                    )
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, MAX_RECONNECT_SEC)
                    continue

                except (
                    websockets.exceptions.WebSocketException,
                    OSError,
                    ConnectionRefusedError,
                ) as exc:
                    logger.warning(
                        "websocket_error error=%s reconnecting_in=%.1fs", exc, backoff
                    )
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, MAX_RECONNECT_SEC)
                    continue

                except asyncio.CancelledError:
                    logger.info("consumer_cancelled")
                    _stats.log()
                    return

                await asyncio.sleep(INITIAL_RECONNECT_SEC)

        finally:
            # Drain remaining queue items before stopping workers
            try:
                await asyncio.wait_for(queue.join(), timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning("queue_drain_timeout %d items remaining", queue.qsize())
            for w in workers:
                w.cancel()
            await asyncio.gather(*workers, return_exceptions=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )


async def _async_main() -> None:
    _setup_logging()
    loop = asyncio.get_running_loop()
    main_task = asyncio.current_task()

    def _shutdown(sig_name: str) -> None:
        logger.info("shutdown_signal signal=%s", sig_name)
        if main_task:
            main_task.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _shutdown, sig.name)
        except NotImplementedError:
            pass  # Windows

    await run_consumer()


def main() -> None:
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
