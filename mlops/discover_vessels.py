"""
Find the vessels worth putting on the watchlist.

aisstream.io is real-time only, so a vessel's track exists only while we are
listening. The anomaly model needs 30 consecutive points (~40 min) and a vessel
baseline needs 50 samples, so the best ships to follow are the ones that are
under way (a moored ship is a flat, uninformative track) and report steadily.

This listens to the live stream for a while, ranks vessels by how many
under-way position reports they produced, and writes a watchlist file that
mlops/ais_consumer.py reads. Edit the file by hand to add or remove ships.

Usage:
    AISSTREAM_API_KEY=<key> python -m mlops.discover_vessels \
        --minutes 30 --top 25 --out watchlist.json

Run it at different times of day (ferries, tugs and tour boats keep schedules)
and re-run it whenever the list goes stale.
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import websockets
from pydantic import ValidationError

_MLOPS_DIR = Path(__file__).resolve().parent
if str(_MLOPS_DIR) not in sys.path:
    sys.path.insert(0, str(_MLOPS_DIR))

import ais_consumer as ac

UNDERWAY_KNOTS = 0.5


class VesselTally:
    """Counts position reports per vessel from raw aisstream messages."""

    def __init__(self) -> None:
        self._v: Dict[str, Dict[str, Any]] = {}

    def add(self, data: Dict[str, Any]) -> None:
        if data.get("MessageType") not in ac._POSITION_TYPES:
            return
        try:
            msg = ac.AISStreamMessage.model_validate(ac.as_position_report(data))
        except ValidationError:
            return
        pr = msg.Message.PositionReport
        if not ac.position_is_usable(pr):
            return
        ts = ac.parse_timestamp(msg.MetaData.time_utc)
        mmsi = str(msg.MetaData.MMSI)
        v = self._v.setdefault(
            mmsi, {"name": "", "reports": 0, "underway": 0, "first": ts, "last": ts}
        )
        v["name"] = (msg.MetaData.ShipName or v["name"]).strip()
        v["reports"] += 1
        v["underway"] += pr.Sog > UNDERWAY_KNOTS
        v["first"], v["last"] = min(v["first"], ts), max(v["last"], ts)

    def rank(self, top: int, min_underway: int) -> List[Dict[str, Any]]:
        rows = [
            {
                "mmsi": mmsi,
                "name": v["name"],
                "reports": v["reports"],
                "underway_reports": v["underway"],
                "span_min": round((v["last"] - v["first"]).total_seconds() / 60, 1),
            }
            for mmsi, v in self._v.items()
            if v["underway"] >= min_underway
        ]
        rows.sort(key=lambda r: (-r["underway_reports"], -r["reports"]))
        return rows[:top]

    @property
    def vessels_seen(self) -> int:
        return len(self._v)


async def listen(tally: VesselTally, minutes: float) -> None:
    """Collect position reports for `minutes`; no MMSI filter, positions only."""
    sub = ac.build_subscription(
        ac.AISSTREAM_API_KEY, ac.BOUNDING_BOXES, message_types=sorted(ac._POSITION_TYPES)
    )
    deadline = time.monotonic() + minutes * 60
    async with websockets.connect(ac.AISSTREAM_WS_URL, compression="deflate") as ws:
        await ws.send(json.dumps(sub))
        while (remaining := deadline - time.monotonic()) > 0:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
            except asyncio.TimeoutError:
                break
            tally.add(json.loads(raw))


def write_watchlist(path: Path, tally: VesselTally, minutes: float, top: int, min_underway: int) -> int:
    vessels = tally.rank(top, min_underway)
    path.write_text(json.dumps(
        {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "listened_minutes": minutes,
            "bounding_boxes": ac.BOUNDING_BOXES,
            "mmsis": [v["mmsi"] for v in vessels],
            "vessels": vessels,
        },
        indent=2,
    ) + "\n")
    return len(vessels)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--minutes", type=float, default=30, help="how long to listen (default 30)")
    p.add_argument("--top", type=int, default=25, help="max vessels to keep (aisstream limit 200)")
    p.add_argument("--min-underway", type=int, default=3,
                   help="min under-way reports to qualify (default 3)")
    p.add_argument("--out", type=Path, default=ac._DEFAULT_WATCHLIST_FILE)
    args = p.parse_args(argv)

    if not ac.AISSTREAM_API_KEY:
        print("AISSTREAM_API_KEY is not set.", file=sys.stderr)
        return 1
    if not 1 <= args.top <= ac.MAX_WATCHLIST:
        print(f"--top must be between 1 and {ac.MAX_WATCHLIST}.", file=sys.stderr)
        return 1

    tally = VesselTally()
    print(f"Listening for {args.minutes:g} min in {ac.BOUNDING_BOXES} (Ctrl-C to stop early)...")
    try:
        asyncio.run(listen(tally, args.minutes))
    except KeyboardInterrupt:
        print("Interrupted; ranking what was collected so far.")
    except websockets.exceptions.ConnectionClosed as exc:
        print(f"Connection closed before data arrived ({exc}); check AISSTREAM_API_KEY.",
              file=sys.stderr)
        return 1

    n = write_watchlist(args.out, tally, args.minutes, args.top, args.min_underway)
    print(f"Saw {tally.vessels_seen} vessels; wrote {n} to {args.out}")
    return 0 if n else 2


if __name__ == "__main__":
    sys.exit(main())
