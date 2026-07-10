"""One-off backfill: historical fan reactions for already-archived fixtures.

Raw reactions were never persisted before the Supabase `reactions` table
existed, but two sources remain queryable after the fact:

  Bluesky - searchPosts supports since/until, so each match's window
            (kickoff-15min .. kickoff+130min) can be re-fetched (keyless)
  Reddit  - match threads persist; fetched when REDDIT_CLIENT_ID/SECRET
            are set (time_filter="year" to find old threads)

YouTube live chat and Mastodon history are not recoverable (chat is deleted
after streams end; tag timelines don't support time windows) - so the backfill
is honest-partial by design.

Usage (needs SUPABASE_URL/SUPABASE_KEY in env; runs from repo root):
  python scripts/backfill_reactions.py            # all archived fixtures
  python scripts/backfill_reactions.py ESPN-760415  # one fixture
"""
from __future__ import annotations

import sys
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import archive
import live
import sources
import teams
import warehouse


def backfill_match(row: pd.Series) -> int:
    """Fetch, clean, tag, minute-map and push one fixture's reactions."""
    match_id = str(row["match_id"])
    home, away = str(row["home_team"]), str(row["away_team"])
    kickoff = pd.Timestamp(row["kickoff_utc"])
    if kickoff.tzinfo is None:
        kickoff = kickoff.tz_localize("UTC")
    since = (kickoff - timedelta(minutes=15)).strftime("%Y-%m-%dT%H:%M:%SZ")
    until = (kickoff + timedelta(minutes=130)).strftime("%Y-%m-%dT%H:%M:%SZ")
    terms = [t for t in (home, away) if t and t != "unknown"]
    if not terms:
        return 0
    frames = [sources.fetch_bluesky_window(terms, since, until)]
    reddit = sources.fetch_reddit(terms, limit=200, time_filter="year")
    if not reddit.empty:
        frames.append(reddit)
    merged = pd.concat([f for f in frames if not f.empty], ignore_index=True) if any(
        not f.empty for f in frames
    ) else pd.DataFrame()
    if merged.empty:
        return 0
    cleaned = sources.clean_reactions(merged, terms)
    if cleaned.empty:
        return 0
    cleaned = cleaned.sort_values("created_utc").tail(sources.COMMENT_WINDOW)
    cleaned["team"] = teams.tag_reactions(cleaned["message"], home, away)
    chat = live.posts_to_chat(cleaned, kickoff)
    if chat.empty:
        return 0
    return warehouse.push_reactions(chat, match_id)


def main() -> None:
    if not warehouse.enabled():
        print("[backfill] SUPABASE_URL/SUPABASE_KEY not set; aborting.")
        sys.exit(1)
    results = archive.load_results()
    if results.empty:
        print("[backfill] no archived fixtures found.")
        return
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    if only:
        results = results.loc[results["match_id"] == only]
    total = 0
    for _, row in results.sort_values("kickoff_utc").iterrows():
        pushed = backfill_match(row)
        total += pushed
        print(
            f"[backfill] {row['match_id']} {row['home_team']} vs "
            f"{row['away_team']}: {pushed} reactions",
            flush=True,
        )
        time.sleep(2)  # politeness gap between fixtures (rate-limit headroom)
    print(f"[backfill] done - {total} reactions across {len(results)} fixtures")


if __name__ == "__main__":
    main()
