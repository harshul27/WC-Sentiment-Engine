"""DuckDB ingestion, stream parsing, and Parquet pipeline.

Commands (run from the repository root):

  python src/pipeline.py run       -> ingest streams, score agents, write
                                      data/state.parquet + DuckDB tables
  python src/pipeline.py optimize  -> nightly self-correction: grid-search
                                      the flag threshold against outcomes
                                      and update data/model_config.json
  python src/pipeline.py all       -> run + optimize in sequence
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

sys.path.append(str(Path(__file__).resolve().parent))

import archive
import health
import live
import situation
from emotion import EMOTION_COLUMNS, EmotionAgent
from model import (
    ArbitrageSelector,
    MatchProgressionAgent,
    grid_search_threshold,
    load_config,
    parse_commentary,
    save_config,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "database.duckdb"
STATE_PATH = DATA_DIR / "state.parquet"
CONFIG_PATH = DATA_DIR / "model_config.json"
STATUS_PATH = DATA_DIR / "run_status.json"

# Self-correction guards: never tune the live threshold on a thin or absent
# real-match corpus (a single match would let one game dictate the threshold),
# and never on the simulator. The corpus is the committed archive, which only
# ever holds real ESPN fixtures.
MIN_TRAIN_MINUTES = 180

TEAMS: tuple[str, str] = ("Brazil", "Argentina")

_MINUTE_LINE = re.compile(r"^\s*\d+(?:\+\d+)?'")

PANIC_CHAT: tuple[str, ...] = (
    "this is a disaster we are done",
    "total panic, we always bottle it",
    "sack the manager, this is embarrassing",
    "I'm so nervous, we're choking again",
    "we are throwing it away, classic collapse",
    "no chance we hold on, terrible defending",
    "awful, absolutely awful, I'm scared",
)

CALM_CHAT: tuple[str, ...] = (
    "we look completely in control",
    "stay calm, we're dominating possession",
    "comfortable win incoming, no worries",
    "so composed at the back, easy game",
    "cruising, we've got this",
    "relax everyone, we're winning this",
)

NEUTRAL_CHAT: tuple[str, ...] = (
    "what a tournament this is",
    "ref needs to keep up with play",
    "good tempo from both sides",
    "atmosphere in the stadium is unreal",
    "midfield battle is fascinating",
)


def simulate_streams(seed: int, minutes: int = 90) -> tuple[pd.DataFrame, pd.Series]:
    """Deterministic mock of YouTube live chat + play-by-play commentary.

    Includes a scripted decoupling window where fan panic spikes while the
    underlying match state stays statistically stable - the exact pattern
    the Arbitrage Selector exists to catch.
    """
    rng = np.random.default_rng(seed)
    commentary: list[str] = []
    chat_rows: list[tuple[int, str]] = []
    panic_start = int(minutes * 0.6)
    panic_window = range(panic_start, min(panic_start + 12, minutes + 1))
    for minute in range(minutes + 1):
        team = TEAMS[int(rng.integers(0, 2))]
        roll = float(rng.random())
        if roll < 0.04:
            line = f"{minute}' {team}: shot on target, forces a save"
        elif roll < 0.07:
            line = f"{minute}' {team}: shot goes wide of the far post"
        elif roll < 0.10:
            line = f"{minute}' {team}: wins a corner on the right"
        elif roll < 0.115:
            line = f"{minute}' {team}: big chance missed, one on one with the keeper"
        elif roll < 0.122:
            line = f"{minute}' {team}: goal! clinical finish from the edge of the box"
        else:
            line = f"{minute}' {team}: keeps possession in midfield"
        commentary.append(line)
        for _ in range(int(rng.integers(2, 6))):
            if minute in panic_window:
                pool = PANIC_CHAT if rng.random() < 0.85 else NEUTRAL_CHAT
            else:
                pool = CALM_CHAT if rng.random() < 0.5 else NEUTRAL_CHAT
            chat_rows.append((minute, pool[int(rng.integers(0, len(pool)))]))
    chat = pd.DataFrame(chat_rows, columns=["minute", "message"]).astype(
        {"minute": "int64", "message": "str"}
    )
    chat["source"] = "simulator"
    return chat, pd.Series(commentary, dtype="str", name="line")


def fetch_live_commentary(url: str, timeout: float = 15.0) -> pd.Series:
    """Scrape minute-stamped commentary lines from a public text feed.

    Returns an empty series on any network or parsing failure so callers
    can fall back to the deterministic simulator.
    """
    try:
        response = requests.get(
            url, timeout=timeout, headers={"User-Agent": "wc-sentiment-engine/0.1"}
        )
        response.raise_for_status()
    except requests.RequestException:
        return pd.Series(dtype="str", name="line")
    soup = BeautifulSoup(response.text, "html.parser")
    lines = [tag.get_text(" ", strip=True) for tag in soup.find_all(["p", "li"])]
    return pd.Series(
        [line for line in lines if _MINUTE_LINE.match(line)], dtype="str", name="line"
    )


def persist_to_duckdb(
    chat: pd.DataFrame,
    commentary: pd.Series,
    events: pd.DataFrame,
    state: pd.DataFrame,
    db_path: Path = DB_PATH,
    state_path: Path = STATE_PATH,
) -> None:
    """Write raw streams + scored state into DuckDB, export zstd Parquet.

    The connection is always closed in a finally block so GitHub Action
    runners never leave the database file locked.
    """
    commentary_frame = commentary.to_frame(name="line")
    connection = duckdb.connect(str(db_path))
    try:
        connection.register("chat_view", chat)
        connection.register("commentary_view", commentary_frame)
        connection.register("events_view", events)
        connection.register("state_view", state)
        connection.execute("CREATE OR REPLACE TABLE raw_chat AS SELECT * FROM chat_view")
        connection.execute(
            "CREATE OR REPLACE TABLE raw_commentary AS SELECT * FROM commentary_view"
        )
        connection.execute("CREATE OR REPLACE TABLE match_events AS SELECT * FROM events_view")
        connection.execute(
            "CREATE OR REPLACE TABLE arbitrage_state AS SELECT * FROM state_view"
        )
        connection.execute(
            "COPY arbitrage_state TO ? (FORMAT PARQUET, COMPRESSION ZSTD)",
            [str(state_path)],
        )
    finally:
        connection.close()


def gather_streams(
    allow_simulator: bool = True,
) -> tuple[pd.DataFrame, pd.Series, str, pd.Series | None]:
    """Source selection: live or post-window ESPN match -> feed URL -> simulator.

    Returns (chat, commentary, match_id, match_row); match_row is the
    scoreboard row when a real fixture was captured, else None. When
    allow_simulator is False and no real source is available, the match_id is
    the ``NONE-`` sentinel so callers can skip writing simulated data into the
    committed production state.
    """
    seed = int(date.today().strftime("%Y%m%d"))
    match = live.current_capture_match()
    if match is not None:
        chat, commentary = live.live_streams(match)
        if not commentary.empty:
            return chat, commentary, f"ESPN-{match['event_id']}", match
    feed_url = os.environ.get("COMMENTARY_FEED_URL", "")
    if feed_url:
        scraped = fetch_live_commentary(feed_url)
        if not scraped.empty:
            chat, _ = simulate_streams(seed)
            return chat, scraped, f"FEED-{seed}", None
    if not allow_simulator:
        empty_chat = pd.DataFrame(columns=["minute", "message", "source"]).astype(
            {"minute": "int64", "message": "str", "source": "str"}
        )
        return empty_chat, pd.Series(dtype="str", name="line"), f"NONE-{seed}", None
    chat, commentary = simulate_streams(seed)
    return chat, commentary, f"SIM-{seed}", None


def fill_emotion_columns(state: pd.DataFrame) -> pd.DataFrame:
    """Carry emotion fields across minutes the chat stream didn't cover."""
    for column in (*EMOTION_COLUMNS, "emotional_volatility"):
        if column in state.columns:
            state[column] = state[column].ffill().fillna(0.0)
    if "dominant_emotion" in state.columns:
        state["dominant_emotion"] = state["dominant_emotion"].ffill().fillna("neutral")
    if "comment_volume" in state.columns:
        state["comment_volume"] = (
            state["comment_volume"].ffill().fillna(0).astype("int64")
        )
    return state


def run_ingest(allow_simulator: bool = True) -> pd.DataFrame:
    """Full ingestion pass: streams -> agents -> DuckDB -> state.parquet.

    When allow_simulator is False (the live flywheel tick) and no real match
    is available, the committed production state is left untouched rather than
    being overwritten with simulated data - so the dashboard never shows a fake
    match between fixtures, and idle ticks produce no commits.
    """
    config = load_config(str(CONFIG_PATH))
    params = config["hyperparameters"]
    chat, commentary, match_id, match_row = gather_streams(allow_simulator=allow_simulator)
    if match_id.startswith("NONE-"):
        print("[run] no live or post-window match; committed state left unchanged")
        return pd.DataFrame()
    social_agent = EmotionAgent(window_minutes=5)
    match_agent = MatchProgressionAgent(
        window_minutes=int(params["xg_rolling_window_minutes"])
    )
    selector = ArbitrageSelector(threshold=float(params["arbitrage_flag_threshold"]))
    state = situation.classify(
        fill_emotion_columns(
            selector.run(social_agent.run(chat), match_agent.run(commentary))
        )
    )
    state.insert(0, "match_id", match_id)
    events = parse_commentary(commentary)
    persist_to_duckdb(chat, commentary, events, state, DB_PATH, STATE_PATH)
    flagged = int(state["flagged"].sum())
    archived = 0
    if match_row is not None and not state.empty:
        archived = archive.archive_match(
            state,
            {
                "match_id": match_id,
                "home_team": match_row.get("home_team"),
                "away_team": match_row.get("away_team"),
                "kickoff_utc": match_row.get("kickoff_utc"),
                "final_score": match_row.get("score"),
                "state": match_row.get("state"),
            },
            db_path=DB_PATH,
            archive_path=DB_PATH.parent / "match_archive.parquet",
            results_path=DB_PATH.parent / "match_results.parquet",
        )
    health.write_status(
        health.stream_health(chat, commentary, match_id), STATUS_PATH
    )
    print(
        f"[run] match_id={match_id} rows={len(state)} "
        f"flagged_minutes={flagged} archived_rows={archived}"
    )
    return state


def derive_overreaction_truth(
    state: pd.DataFrame, events: pd.DataFrame, horizon: int = 15
) -> pd.Series:
    """Ground truth: panic that the eventual outcome never justified.

    A minute counts as a real arbitrage moment (1.0) when the crowd was
    clearly panicking yet no goal or penalty arrived within the look-ahead
    horizon - i.e. the market overreacted to noise.
    """
    goal_minutes = events.loc[
        events["event_type"].isin(["goal", "penalty"]), "minute"
    ].to_numpy(dtype=np.int64)
    minutes = state["minute"].to_numpy(dtype=np.int64)
    upcoming = np.array(
        [bool(((goal_minutes > m) & (goal_minutes <= m + horizon)).any()) for m in minutes]
    )
    panicking = state["crowd_panic_score"].abs().to_numpy(dtype=np.float64) > 0.4
    return pd.Series((panicking & ~upcoming).astype(np.float64), name="overreaction")


def derive_corpus_truth(
    corpus: pd.DataFrame, horizon: int = 15, threat_floor: float = 0.30
) -> pd.Series:
    """Outcome-grounded labels from the archived real-match corpus.

    For each match, a minute is a genuine arbitrage moment (1.0) when the
    crowd was clearly panicking yet no real attacking threat arrived over the
    next ``horizon`` minutes - i.e. the panic was never vindicated on the
    pitch. The forward threat is approximated from the rising edge of the
    archived rolling xG (a goal or big chance entering the window bumps it),
    which is a different, forward-looking quantity from the current-minute
    arbitrage index, so the label is not circular.
    """
    if corpus.empty:
        return pd.Series(dtype="float64", name="overreaction")
    labels = pd.Series(0.0, index=corpus.index, name="overreaction")
    for _, group in corpus.groupby("match_id"):
        ordered = group.sort_values("minute")
        rolling = ordered["rolling_xg"].to_numpy(dtype=np.float64)
        arrivals = np.clip(np.diff(rolling, prepend=rolling[:1]), 0.0, None)
        forward = np.array(
            [arrivals[i + 1 : i + 1 + horizon].sum() for i in range(len(arrivals))]
        )
        panicking = ordered["crowd_panic_score"].abs().to_numpy(dtype=np.float64) > 0.4
        threat_arrived = forward >= threat_floor
        labels.loc[ordered.index] = (panicking & ~threat_arrived).astype(np.float64)
    return labels


def run_optimize(min_minutes: int = MIN_TRAIN_MINUTES) -> dict[str, float]:
    """Nightly MLOps loop: re-fit the flag threshold on the real-match corpus.

    Trains on the committed archive (real ESPN fixtures only - never the
    simulator), and refuses to move the threshold until enough real minutes
    with both outcome classes have accumulated, so a single match or an idle
    night can never dictate the production threshold.
    """
    corpus = archive.load_archive(archive_path=DB_PATH.parent / "match_archive.parquet")
    config = load_config(str(CONFIG_PATH))
    current = float(config["hyperparameters"]["arbitrage_flag_threshold"])

    def _skip(reason: str) -> dict[str, float]:
        print(f"[optimize] skipped ({reason}); threshold unchanged at {current}.")
        return {
            "arbitrage_flag_threshold": current,
            "log_loss": float("nan"),
            "status": "skipped",
            "evaluated_minutes": int(len(corpus)),
        }

    if corpus.empty or len(corpus) < min_minutes:
        return _skip(f"{len(corpus)} archived real minutes < {min_minutes} required")
    truth = derive_corpus_truth(corpus)
    positives = float(truth.sum())
    if positives == 0.0 or positives == float(len(truth)):
        return _skip("corpus has only one outcome class so far")

    result = grid_search_threshold(corpus["arbitrage_index"], truth)
    config["hyperparameters"]["arbitrage_flag_threshold"] = result[
        "arbitrage_flag_threshold"
    ]
    config["log_loss_history"].append(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "log_loss": round(result["log_loss"], 6),
            "threshold": result["arbitrage_flag_threshold"],
            "evaluated_minutes": int(len(corpus)),
            "matches": int(corpus["match_id"].nunique()),
            "source": "match_archive",
        }
    )
    save_config(str(CONFIG_PATH), config)
    print(
        f"[optimize] threshold={result['arbitrage_flag_threshold']} "
        f"log_loss={result['log_loss']:.6f} "
        f"minutes={len(corpus)} matches={corpus['match_id'].nunique()}"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="WC Sentiment Arbitrage pipeline")
    parser.add_argument(
        "command",
        choices=("run", "optimize", "all"),
        nargs="?",
        default="all",
        help="run = ingest streams, optimize = nightly self-correction",
    )
    parser.add_argument(
        "--live-only",
        action="store_true",
        help=(
            "skip the deterministic simulator: only write committed state when "
            "a real (or feed) match is captured. Used by the live flywheel tick."
        ),
    )
    args = parser.parse_args()
    if args.command in ("run", "all"):
        run_ingest(allow_simulator=not args.live_only)
    if args.command in ("optimize", "all"):
        run_optimize()


if __name__ == "__main__":
    main()
