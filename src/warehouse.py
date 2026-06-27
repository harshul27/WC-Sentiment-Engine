"""Mirror the engine's data into Supabase (key-gated, best-effort).

The committed Parquet files stay the durable, keyless source of truth; this
module additionally pushes the same rows into a hosted Supabase Postgres so the
data is queryable from a real structured platform (SQL editor, REST API, table
viewer) - useful for sharing and inspection.

It uses Supabase's PostgREST endpoint via plain `requests` (no SDK dependency),
upserts on the table primary keys, and degrades to a no-op when the
SUPABASE_URL / SUPABASE_KEY secrets are absent or a request fails, so it can
never block or break the pipeline.

CLI:
  python -m warehouse --dry-run   # print what would be sent, no network
  python -m warehouse             # sync the committed data files to Supabase
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

# committed file -> Supabase table name
TABLES: dict[str, str] = {
    "state.parquet": "engine_state",
    "match_archive.parquet": "match_archive",
    "match_results.parquet": "match_results",
}


def enabled() -> bool:
    """True when both Supabase secrets are configured."""
    return bool(os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_KEY"))


def _headers() -> dict[str, str]:
    key = os.environ.get("SUPABASE_KEY", "")
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates,return=minimal",
    }


def _clean(value: object) -> object:
    """Make one cell JSON/PostgREST-safe."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        if pd.isna(value):
            return None
        return pd.Timestamp(value).isoformat()
    if hasattr(value, "item"):  # numpy scalar
        try:
            return value.item()
        except (ValueError, TypeError):
            return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        # pd.isna raises on non-scalar values (lists/dicts); those are already
        # JSON-safe, so fall through and return them unchanged.
        return value
    return value


def records(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Convert a DataFrame into a list of JSON-safe row dicts."""
    if frame is None or frame.empty:
        return []
    return [
        {col: _clean(row[col]) for col in frame.columns}
        for _, row in frame.iterrows()
    ]


def push_records(
    table: str, rows: list[dict[str, object]], timeout: float = 30.0
) -> int:
    """Upsert rows into a Supabase table; returns rows sent, 0 on no-op/failure."""
    if not enabled() or not rows:
        return 0
    base = os.environ["SUPABASE_URL"].rstrip("/")
    try:
        response = requests.post(
            f"{base}/rest/v1/{table}",
            headers=_headers(),
            data=json.dumps(rows),
            timeout=timeout,
        )
        response.raise_for_status()
    except (requests.RequestException, ValueError) as exc:
        print(f"[warehouse] {table}: push failed ({type(exc).__name__})")
        return 0
    return len(rows)


def push_frame(table: str, frame: pd.DataFrame) -> int:
    return push_records(table, records(frame))


def push_status(status: dict[str, object]) -> int:
    """Mirror the run heartbeat as a single upserted row (keyed by match_id)."""
    if not status:
        return 0
    row = {k: _clean(v) for k, v in status.items() if not isinstance(v, (dict, list))}
    return push_records("run_status", [row])


def push_reactions(chat: pd.DataFrame, match_id: str) -> int:
    """Persist a match's individual reactions (keyed so re-fetches upsert).

    The live UI keeps the rolling window in memory; this stores the raw
    cleaned reactions in Supabase instead of git, so the full per-match set
    accumulates over the flywheel's runs without bloating the repo. The key is
    (match_id, message_hash), so the same reaction seen on a later run merges.
    """
    if not enabled() or chat is None or chat.empty or "message" not in chat.columns:
        return 0
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    mid = str(match_id)
    for _, row in chat.iterrows():
        message = str(row.get("message", "")).strip()
        if not message:
            continue
        digest = hashlib.sha256(message.encode("utf-8")).hexdigest()[:24]
        key = (mid, digest)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "match_id": mid,
                "message_hash": digest,
                "minute": _clean(row.get("minute")),
                "message": message,
                "source": str(row.get("source", "")),
                "team": str(row.get("team", "")),
                "author": str(row.get("author", "")),
            }
        )
    return push_records("reactions", rows)


def sync_from_disk(data_dir: Path = DATA_DIR) -> dict[str, int]:
    """Push the committed data files to Supabase; best-effort per table."""
    counts: dict[str, int] = {}
    if not enabled():
        return counts
    for filename, table in TABLES.items():
        path = Path(data_dir) / filename
        if not path.exists():
            continue
        try:
            frame = pd.read_parquet(path)
        except (OSError, ValueError):
            continue
        counts[table] = push_frame(table, frame)
    status_path = Path(data_dir) / "run_status.json"
    if status_path.exists():
        try:
            counts["run_status"] = push_status(
                json.loads(status_path.read_text(encoding="utf-8"))
            )
        except (OSError, ValueError):
            # Best-effort mirror: an unreadable/malformed heartbeat must never
            # break the sync of the other tables.
            counts["run_status"] = 0
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Mirror engine data to Supabase")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print what would be sent (row counts + sample) without any network",
    )
    args = parser.parse_args()
    if args.dry_run:
        for filename, table in TABLES.items():
            path = DATA_DIR / filename
            if not path.exists():
                print(f"[dry-run] {table}: (no {filename})")
                continue
            rows = records(pd.read_parquet(path))
            sample = json.dumps(rows[0], default=str)[:160] if rows else "{}"
            print(f"[dry-run] {table}: {len(rows)} rows; sample {sample}")
        print(f"[dry-run] supabase enabled: {enabled()}")
        return
    if not enabled():
        print("[warehouse] SUPABASE_URL/SUPABASE_KEY not set; nothing to do.")
        sys.exit(0)
    counts = sync_from_disk()
    print(f"[warehouse] synced: {counts}")


if __name__ == "__main__":
    main()
