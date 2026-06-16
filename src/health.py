"""Run health / freshness signal for the engine.

The pipeline's connectors all degrade gracefully to empty frames, which means
a source outage looks identical to a quiet match on the dashboard. This module
makes the difference observable: every persisted run writes a small JSON
heartbeat (data/run_status.json) describing what was actually captured, and the
dashboard turns that heartbeat into a LIVE / STALE / DEGRADED / NO-DATA badge so
"the crowd is calm" can never be confused with "the feed is down".

The heartbeat is intentionally tiny and only written when the engine actually
persists state, so it never adds no-op commits between matches.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STATUS_PATH = ROOT / "data" / "run_status.json"

# How long after the last capture committed state is still considered fresh.
# The live flywheel ticks every 20 minutes, so 25 gives one missed tick of slack.
STALE_AFTER_MINUTES = 25

_SOURCE_FROM_PREFIX = {"ESPN": "live", "FEED": "feed", "SIM": "simulator"}


def stream_health(
    chat: pd.DataFrame,
    commentary: pd.Series,
    match_id: str,
    now: datetime | None = None,
) -> dict[str, object]:
    """Summarise one ingestion run into a JSON-serialisable heartbeat."""
    moment = now or datetime.now(timezone.utc)
    prefix = str(match_id).split("-", 1)[0]
    source = _SOURCE_FROM_PREFIX.get(prefix, "unknown")
    by_source: dict[str, int] = {}
    if not chat.empty and "source" in chat.columns:
        by_source = {
            str(name): int(count)
            for name, count in chat["source"].value_counts().items()
        }
    return {
        "last_run_utc": moment.isoformat(timespec="seconds"),
        "match_id": str(match_id),
        "source": source,
        "live": source == "live",
        "n_reactions": int(len(chat)),
        "n_commentary": int(len(commentary)),
        "reactions_by_source": by_source,
        "fetch_ok": bool(source == "live" and len(commentary) > 0),
    }


def write_status(status: dict[str, object], path: Path = STATUS_PATH) -> None:
    """Persist the heartbeat next to the committed state Parquet."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(status, handle, indent=2)
        handle.write("\n")


def load_status(path: Path = STATUS_PATH) -> dict[str, object] | None:
    """Read the heartbeat, or None when it is absent or unreadable."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def _age_minutes(status: dict[str, object], now: datetime) -> float | None:
    raw = status.get("last_run_utc")
    if not isinstance(raw, str):
        return None
    try:
        stamp = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return (now - stamp).total_seconds() / 60.0


def freshness(
    status: dict[str, object] | None,
    now: datetime | None = None,
    stale_after_minutes: int = STALE_AFTER_MINUTES,
) -> dict[str, object]:
    """Classify the committed state for the dashboard badge.

    Returns {"level", "label", "detail"} where level is one of
    live | stale | degraded | no-data.
    """
    moment = now or datetime.now(timezone.utc)
    if not status or int(status.get("n_reactions", 0) or 0) == 0:
        return {
            "level": "no-data",
            "label": "No data",
            "detail": "No captured engine state yet.",
        }
    age = _age_minutes(status, moment)
    age_text = "unknown age" if age is None else f"{int(age)} min ago"
    sources = status.get("reactions_by_source") or {}
    source_text = ", ".join(f"{k} {v}" for k, v in sources.items()) or "no sources"
    if not status.get("live", False):
        return {
            "level": "degraded",
            "label": "Demo / fallback data",
            "detail": (
                f"No live match captured — showing {status.get('source', 'fallback')} "
                f"data ({age_text})."
            ),
        }
    if age is not None and age > stale_after_minutes:
        return {
            "level": "stale",
            "label": "Stale",
            "detail": (
                f"Last live capture {age_text} — between matches or the feed is "
                "delayed."
            ),
        }
    return {
        "level": "live",
        "label": "Live capture",
        "detail": (
            f"{status.get('n_reactions', 0)} reactions ({source_text}); "
            f"updated {age_text}."
        ),
    }
