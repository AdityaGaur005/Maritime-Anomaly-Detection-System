"""
Live AIS ingestion consumer for the Maritime Anomaly Detection System.

Connects to wss://stream.aisstream.io/v0/stream, subscribes to position and
static vessel messages, validates each one, and either POSTs it to /predict
(position) or updates the vessel profile store (static).

Architecture:
    WebSocket reader  ->  per-vessel throttle  ->  asyncio.Queue(QUEUE_MAXSIZE)
    N worker tasks    <-  drain queue, POST to /predict

What aisstream.io's free service does and does not give us (docs + live probes,
2026-10-13):
    * No documented monthly quota. Real volume is tiny: ~10 messages/min across
      all of the main Hawaiian Islands (~30 KB per 5 min).
    * FiltersShipMMSI: up to 200 vessels per subscription; 3 subscribed
      connections per account; one subscription update per second.
    * Terrestrial receivers only: a 5-minute probe of 0-30N / 60-100E (all of
      India + Arabian Sea + Bay of Bengal) returned ZERO position reports, so
      India cannot be monitored. Hawaii returned ~18 ships in 5 minutes.
    * Real-time only, no history -- a vessel's track exists only while we are
      connected, which is why long-running, vessel-specific monitoring (the
      watchlist below) is what builds useful baselines.
    * Uncompressed connections are bandwidth-limited from Sept 2026, so
      permessage-deflate is requested explicitly.

Watchlist (track specific ships): put 9-digit MMSIs in watchlist.json (a list,
or {"mmsis": [...]} as written by mlops/discover_vessels.py) or in the
WATCHLIST_MMSIS env var. Empty watchlist = every vessel in the bounding box.

Downsampling: the models were trained on ~60-90 s sampling (XGBoost
delta_time_mean splits start at ~65 s). Class A ships under way broadcast every
2-10 s, which would squeeze a 30-point window into minutes instead of ~40
minutes, so positions are thinned to MIN_POSITION_INTERVAL_SEC per vessel.

Backpressure: if the queue is full (API slower than stream) the incoming item
is dropped and counted in stats.dropped.

Reconnection: exponential backoff 1 s -> 2 s -> ... -> MAX_RECONNECT_SEC.

Auth fail-fast: aisstream does NOT answer a bad API key with HTTP 401/403 or an
error message -- it just drops the connection (observed). So the consumer stops
after MAX_UNCONFIRMED_CONNECTIONS consecutive connections that close before a
SubscriptionConfirmation arrives. HTTP 401/403 at the handshake is also treated
as fatal in case that behaviour changes.

Usage:
    AISSTREAM_API_KEY=<key> python -m mlops.ais_consumer

Environment variables:
    AISSTREAM_API_KEY         Required. Read from the environment only; never
                              commit it.
    AISSTREAM_WS_URL          Default wss://stream.aisstream.io/v0/stream
    PREDICT_URL               Default http://localhost:8000/predict
    BOUNDING_BOXES            JSON list e.g. [[[18.5,-161],[22.8,-154.5]]]
    WATCHLIST_MMSIS           Comma-separated MMSIs (overrides WATCHLIST_FILE)
    WATCHLIST_FILE            Default <repo>/watchlist.json
    MIN_POSITION_INTERVAL_SEC Default 60 -- per-vessel downsampling
    REQUEST_TIMEOUT_SEC       Default 5
    MAX_RECONNECT_SEC         Default 60
    WORKER_COUNT              Default 3
    QUEUE_MAXSIZE             Default 500
"""

import asyncio
import json
import logging
import os
import re
import signal
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import httpx
import websockets
import websockets.exceptions
from pydantic import BaseModel, Field, ValidationError, field_validator

# kinematics / vessel_profile_store live alongside this module in mlops/
_MLOPS_DIR = Path(__file__).resolve().parent
if str(_MLOPS_DIR) not in sys.path:
    sys.path.insert(0, str(_MLOPS_DIR))

from kinematics import parse_timestamp
from vessel_profile_store import init_db, update_static_attributes

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
AISSTREAM_API_KEY: str = os.environ.get("AISSTREAM_API_KEY", "")
AISSTREAM_WS_URL: str = os.environ.get(
    "AISSTREAM_WS_URL", "wss://stream.aisstream.io/v0/stream"
)
PREDICT_URL: str = os.environ.get("PREDICT_URL", "http://localhost:8000/predict")

# Main Hawaiian Islands [[lat_min, lon_min], [lat_max, lon_max]]. The models
# were trained here (learned lat ~19.6-22.7, lon ~-159.3 to -155.7) and the
# cold-start centroid is Honolulu, so scoring ships elsewhere is meaningless.
_DEFAULT_BOXES = json.dumps([[[18.5, -161.0], [22.8, -154.5]]])
BOUNDING_BOXES: List[List[List[float]]] = json.loads(
    os.environ.get("BOUNDING_BOXES", _DEFAULT_BOXES)
)

REQUEST_TIMEOUT_SEC: float = float(os.environ.get("REQUEST_TIMEOUT_SEC", "5"))
MAX_RECONNECT_SEC: float = float(os.environ.get("MAX_RECONNECT_SEC", "60"))
INITIAL_RECONNECT_SEC: float = 1.0
WORKER_COUNT: int = int(os.environ.get("WORKER_COUNT", "3"))
QUEUE_MAXSIZE: int = int(os.environ.get("QUEUE_MAXSIZE", "500"))
MIN_POSITION_INTERVAL_SEC: float = float(os.environ.get("MIN_POSITION_INTERVAL_SEC", "60"))

MAX_WATCHLIST: int = 200  # aisstream FiltersShipMMSI limit
MAX_UNCONFIRMED_CONNECTIONS: int = 3
_DEFAULT_WATCHLIST_FILE = _MLOPS_DIR.parent / "watchlist.json"

_STATS_LOG_INTERVAL: int = 1000
_STATS_LOG_SEC: float = 300.0

# Class B transponders (small craft, fishing, tour boats) report types 18/19
# with the same fields as Class A types 1-3 under a different key.
_CLASS_B_TYPES = frozenset({"StandardClassBPositionReport", "ExtendedClassBPositionReport"})
_POSITION_TYPES = frozenset({"PositionReport"}) | _CLASS_B_TYPES
_STATIC_MSG_TYPES = frozenset({"ShipStaticData", "StaticDataReport"})
_SUBSCRIBE_TYPES = sorted(_POSITION_TYPES | _STATIC_MSG_TYPES)

# AIS "not available" sentinels: Sog 102.3 kn, Cog 360 deg, lat 91, lon 181.
_SOG_NOT_AVAILABLE = 102.3
_COG_NOT_AVAILABLE = 360.0


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------
_AIS_TIME_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2})(?:\.(\d+))? ([+-])(\d{2})(\d{2}) UTC$"
)


def to_iso_utc(ts: str) -> str:
    """
    aisstream sends '2026-10-13 15:01:50.32030298 +0000 UTC' (up to 9 fractional
    digits, trailing 'UTC'), which datetime.fromisoformat rejects. Convert it to
    ISO-8601 with exactly 6 fractional digits; anything else passes through.
    """
    m = _AIS_TIME_RE.match(ts.strip())
    if not m:
        return ts
    day, clock, frac, sign, hh, mm = m.groups()
    micro = f".{frac[:6].ljust(6, '0')}" if frac else ""
    return f"{day}T{clock}{micro}{sign}{hh}:{mm}"


# ---------------------------------------------------------------------------
# Shared sub-models
# ---------------------------------------------------------------------------
class _MetaData(BaseModel):
    MMSI: int
    ShipName: Optional[str] = None
    time_utc: str

    @field_validator("time_utc")
    @classmethod
    def _normalise_time(cls, v: str) -> str:
        iso = to_iso_utc(v)
        try:
            parse_timestamp(iso)
        except ValueError as exc:
            raise ValueError(f"unparseable time_utc {v!r}") from exc
        return iso


class _Dimension(BaseModel):
    """AIS antenna-to-hull distances in metres. A+B = length, C+D = beam."""
    A: Optional[int] = None  # bow
    B: Optional[int] = None  # stern
    C: Optional[int] = None  # port
    D: Optional[int] = None  # starboard


# ---------------------------------------------------------------------------
# Position messages (AIS types 1/2/3 Class A, 18/19 Class B)
# ---------------------------------------------------------------------------
class _PositionReport(BaseModel):
    UserID: int
    Latitude: float
    Longitude: float
    Sog: float = Field(..., description="Speed over ground, knots")
    Cog: float = Field(..., description="Course over ground, degrees")
    TrueHeading: Optional[int] = None
    NavigationalStatus: Optional[int] = None
    Valid: Optional[bool] = None


class _PositionReportMsg(BaseModel):
    PositionReport: _PositionReport


class AISStreamMessage(BaseModel):
    """Validated position message from aisstream.io (Class B is normalised to this shape)."""

    MessageType: str
    MetaData: _MetaData
    Message: _PositionReportMsg

    @field_validator("MessageType")
    @classmethod
    def must_be_position_report(cls, v: str) -> str:
        if v != "PositionReport":
            raise ValueError(f"Expected PositionReport, got {v!r}")
        return v


def as_position_report(data: Dict[str, Any]) -> Dict[str, Any]:
    """Present a Class B position message in the Class A shape."""
    msg_type = data.get("MessageType")
    if msg_type in _CLASS_B_TYPES:
        body = (data.get("Message") or {}).get(msg_type)
        return {**data, "MessageType": "PositionReport", "Message": {"PositionReport": body}}
    return data


def position_is_usable(pr: _PositionReport) -> bool:
    """Reject invalid fixes and AIS 'not available' sentinel values (NaN fails every check)."""
    return (
        pr.Valid is not False
        and 0.0 <= pr.Sog < _SOG_NOT_AVAILABLE
        and 0.0 <= pr.Cog < _COG_NOT_AVAILABLE
        and -90.0 <= pr.Latitude <= 90.0
        and -180.0 <= pr.Longitude <= 180.0
    )


# ---------------------------------------------------------------------------
# ShipStaticData message (AIS type 5 -- voyage / static data)
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
# StaticDataReport message (AIS type 24 -- Class B static report)
# ---------------------------------------------------------------------------
class _StaticReportB(BaseModel):
    ShipType: Optional[int] = None
    Dimension: Optional[_Dimension] = None
    Valid: Optional[bool] = None


class _StaticDataReportPayload(BaseModel):
    UserID: int
    PartNumber: Optional[bool] = None  # False = Part A (name only), True = Part B (type/dims)
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


def _known_type(code: Optional[int]) -> Optional[int]:
    """AIS ship type 0 means 'not available'."""
    return code or None


def handle_static_message(msg_type: str, data: Dict[str, Any]) -> None:
    """
    Parse a ShipStaticData or StaticDataReport message and persist vessel
    type, length, and beam to the profile store.

    Ships broadcast static data infrequently (~every 6 minutes), so this
    runs synchronously in the WS reader loop -- the SQLite write is fast
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
            vessel_type = _known_type(sd.Type)
            length = _sum_dim(sd.Dimension, "A", "B")
            width = _sum_dim(sd.Dimension, "C", "D")

        elif msg_type == "StaticDataReport":
            msg = StaticDataReportMessage.model_validate(data)
            sd = msg.Message.StaticDataReport
            mmsi = str(msg.MetaData.MMSI)
            # Only a valid Part B carries type and dimensions; Part A messages still
            # include an all-zero, Valid=false ReportB placeholder that must be ignored.
            rb = sd.ReportB
            if rb is not None and sd.PartNumber is not False and rb.Valid is not False:
                vessel_type = _known_type(rb.ShipType)
                length = _sum_dim(rb.Dimension, "A", "B")
                width = _sum_dim(rb.Dimension, "C", "D")
        else:
            return

    except ValidationError as exc:
        logger.warning(
            "static_validation_error msg_type=%s errors=%s", msg_type, exc.errors()
        )
        return

    if mmsi and any(v is not None for v in (vessel_type, length, width)):
        try:
            update_static_attributes(mmsi, vessel_type=vessel_type, length=length, width=width)
        except sqlite3.Error as exc:
            # A locked/unavailable profile DB must not take the whole stream down.
            logger.error("static_write_failed mmsi=%s error=%s", mmsi, exc)
            return
        logger.debug(
            "static_updated mmsi=%s type=%s length=%s width=%s",
            mmsi, vessel_type, length, width,
        )


# ---------------------------------------------------------------------------
# Watchlist + subscription
# ---------------------------------------------------------------------------
def parse_mmsis(values: Iterable[Any]) -> List[str]:
    """Validate and de-duplicate MMSIs (aisstream wants nine-character strings, max 200)."""
    out: List[str] = []
    for v in values:
        s = str(v).strip()
        if not re.fullmatch(r"\d{9}", s):
            raise ValueError(f"invalid MMSI {v!r}: must be exactly 9 digits")
        if s not in out:
            out.append(s)
    if len(out) > MAX_WATCHLIST:
        raise ValueError(f"watchlist has {len(out)} MMSIs; aisstream allows at most {MAX_WATCHLIST}")
    return out


def load_watchlist() -> List[str]:
    """MMSIs to track. WATCHLIST_MMSIS env wins over WATCHLIST_FILE; empty list = whole box."""
    env = os.environ.get("WATCHLIST_MMSIS", "").strip()
    if env:
        return parse_mmsis(part for part in env.replace(" ", "").split(",") if part)
    path = Path(os.environ.get("WATCHLIST_FILE", str(_DEFAULT_WATCHLIST_FILE)))
    if not path.is_file():
        return []
    raw = json.loads(path.read_text())
    if isinstance(raw, dict):
        raw = raw.get("mmsis", [])
    return parse_mmsis(raw)


def build_subscription(
    api_key: str,
    boxes: List[List[List[float]]],
    mmsis: Sequence[str] = (),
    message_types: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    sub: Dict[str, Any] = {
        "APIKey": api_key,
        "BoundingBoxes": boxes,
        "FilterMessageTypes": list(message_types or _SUBSCRIBE_TYPES),
    }
    if mmsis:
        sub["FiltersShipMMSI"] = list(mmsis)
    return sub


# ---------------------------------------------------------------------------
# Per-vessel downsampling
# ---------------------------------------------------------------------------
class PositionThrottle:
    """Accept at most one position per vessel per min_interval seconds (message time)."""

    def __init__(self, min_interval_sec: float) -> None:
        self.min_interval_sec = min_interval_sec
        self._last: Dict[str, datetime] = {}

    def allow(self, mmsi: str, ts: datetime) -> bool:
        last = self._last.get(mmsi)
        # Written as 'not >=' so out-of-order (negative) gaps are rejected too.
        if last is not None and not (ts - last).total_seconds() >= self.min_interval_sec:
            return False
        self._last[mmsi] = ts
        return True


_throttle = PositionThrottle(MIN_POSITION_INTERVAL_SEC)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
class ConsumerStats:
    def __init__(self) -> None:
        self.received: int = 0
        self.invalid: int = 0
        self.throttled: int = 0  # positions thinned by the per-vessel downsampler
        self.posted: int = 0
        self.anomalies: int = 0
        self.server_errors: int = 0
        self.dropped: int = 0  # queue-overflow drops
        self.confirmed: int = 0  # SubscriptionConfirmations received (connection health)

    def log(self) -> None:
        logger.info(
            "consumer_stats received=%d invalid=%d throttled=%d posted=%d"
            " anomalies=%d server_errors=%d dropped=%d",
            self.received, self.invalid, self.throttled, self.posted,
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
class SubscriptionRejected(Exception):
    """The connection closed before aisstream confirmed our subscription."""


async def consume_stream(
    queue: "asyncio.Queue[Dict[str, Any]]",
    stats: ConsumerStats = _stats,
    mmsis: Sequence[str] = (),
    throttle: PositionThrottle = _throttle,
) -> None:
    """
    Open one WebSocket connection and push position payloads onto the queue.
    Static messages are handled synchronously (fast SQLite write).

    Returns normally on a clean, confirmed server-side close.
    Raises SubscriptionRejected if the connection ends before a
    SubscriptionConfirmation (how aisstream treats a bad key or bad subscription).
    Raises websockets.exceptions.WebSocketException on other connection failures
    and InvalidStatus on non-101 handshake responses.
    """
    confirmed = False
    last_stats_log = time.monotonic()

    async with websockets.connect(AISSTREAM_WS_URL, compression="deflate") as ws:
        try:
            await ws.send(json.dumps(build_subscription(AISSTREAM_API_KEY, BOUNDING_BOXES, mmsis)))
            logger.info(
                "aisstream_subscribed url=%s boxes=%s watchlist_size=%d",
                AISSTREAM_WS_URL, BOUNDING_BOXES, len(mmsis),
            )

            async for raw_message in ws:
                stats.received += 1
                now = time.monotonic()
                if (
                    stats.received % _STATS_LOG_INTERVAL == 0
                    or now - last_stats_log >= _STATS_LOG_SEC
                ):
                    stats.log()
                    last_stats_log = now

                try:
                    data = json.loads(raw_message)
                except json.JSONDecodeError as exc:
                    logger.warning("json_decode_error error=%s", exc)
                    stats.invalid += 1
                    continue

                msg_type = data.get("MessageType", "")

                if msg_type == "SubscriptionConfirmation":
                    confirmed = True
                    stats.confirmed += 1
                    logger.info("aisstream_subscription_confirmed")
                    continue

                if "error" in data or "Error" in data:
                    logger.error("aisstream_error payload=%s", data)
                    stats.invalid += 1
                    continue

                # Static data: synchronous SQLite write (~1 ms), no queue needed
                if msg_type in _STATIC_MSG_TYPES:
                    handle_static_message(msg_type, data)
                    continue

                if msg_type not in _POSITION_TYPES:
                    continue

                try:
                    msg = AISStreamMessage.model_validate(as_position_report(data))
                except ValidationError as exc:
                    logger.warning("position_validation_error errors=%s", exc.errors())
                    stats.invalid += 1
                    continue

                if not position_is_usable(msg.Message.PositionReport):
                    stats.invalid += 1
                    continue

                if not throttle.allow(str(msg.MetaData.MMSI), parse_timestamp(msg.MetaData.time_utc)):
                    stats.throttled += 1
                    continue

                payload = translate_to_predict_payload(msg)

                # Drop incoming item on overflow rather than blocking the WS reader
                try:
                    queue.put_nowait(payload)
                except asyncio.QueueFull:
                    stats.dropped += 1
                    logger.warning(
                        "queue_full dropped_mmsi=%s queue_size=%d",
                        payload.get("MMSI"), queue.qsize(),
                    )
        except (websockets.exceptions.ConnectionClosed, OSError) as exc:
            # Dropped (or reset) after we subscribed but before aisstream confirmed.
            # Failures to connect at all are raised by websockets.connect() above
            # this block, so a network outage is never mistaken for a rejection.
            if not confirmed:
                raise SubscriptionRejected(str(exc)) from exc
            raise

    if not confirmed:
        raise SubscriptionRejected("connection closed before SubscriptionConfirmation")


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

    try:
        watchlist = load_watchlist()
    except (ValueError, OSError) as exc:
        logger.error("invalid_watchlist error=%s", exc)
        sys.exit(1)
    init_db()  # idempotent; the API process may not have created the shared tables yet
    logger.info(
        "watchlist_loaded size=%d mode=%s",
        len(watchlist), "vessel-specific" if watchlist else "whole-bounding-box",
    )

    queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
    backoff = INITIAL_RECONNECT_SEC
    unconfirmed = 0

    async with httpx.AsyncClient() as client:
        workers = [
            asyncio.create_task(_worker(queue, client, _stats))
            for _ in range(WORKER_COUNT)
        ]
        try:
            while True:
                confirmed_before = _stats.confirmed
                try:
                    await consume_stream(queue, mmsis=watchlist)
                    # Clean, confirmed server-side close
                    logger.info(
                        "websocket_closed_cleanly reconnecting_in=%.1fs",
                        INITIAL_RECONNECT_SEC,
                    )
                    backoff = INITIAL_RECONNECT_SEC
                    unconfirmed = 0

                except SubscriptionRejected as exc:
                    unconfirmed += 1
                    if unconfirmed >= MAX_UNCONFIRMED_CONNECTIONS:
                        logger.error(
                            "subscription_rejected count=%d error=%s -- the connection closes "
                            "before confirmation, which aisstream does for an invalid "
                            "AISSTREAM_API_KEY or a malformed subscription; not retrying",
                            unconfirmed, exc,
                        )
                        return
                    logger.warning(
                        "subscription_unconfirmed count=%d/%d reconnecting_in=%.1fs",
                        unconfirmed, MAX_UNCONFIRMED_CONNECTIONS, backoff,
                    )
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, MAX_RECONNECT_SEC)
                    continue

                except websockets.exceptions.InvalidStatus as exc:
                    # Non-101 HTTP response during handshake
                    status = exc.response.status_code
                    if status in (401, 403):
                        logger.error(
                            "auth_failed status=%d -- "
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
                    if _stats.confirmed > confirmed_before:
                        unconfirmed = 0  # a confirmed connection later dropped: not an auth problem
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
