"""Streamlit frontend: real-time arbitrage ticker for the WC Sentiment Engine.

Three display modes:
  Live match      - polls free ESPN + multi-source crowd APIs every 60
                    seconds and recomputes the agents against the real match
  Simulator       - animated replay of a deterministic mock match
  Committed state - latest data/state.parquet produced by the GitHub Action

No custom HTML/CSS - native Streamlit components only, per CLAUDE.md.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent))

from emotion import EMOTION_COLUMNS, EmotionAgent, generate_takeaways
from live import fetch_scoreboard, live_streams, utc_now
from matchstats import control_index, fetch_boxscore
from model import ArbitrageSelector, MatchProgressionAgent, load_config
from pipeline import fill_emotion_columns, simulate_streams

ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "data" / "state.parquet"
CONFIG_PATH = ROOT / "data" / "model_config.json"
CHART_COLUMNS = ["crowd_panic_score", "delta_xg_10min", "arbitrage_index"]
MODE_LIVE = "🔴 Live match"
MODE_SIM = "🎮 Simulator"
MODE_STATE = "📦 Committed state"
EMOTION_LABELS = {col: col.removeprefix("emo_").title() for col in EMOTION_COLUMNS}
TONE_RENDERERS = {"warning": st.warning, "positive": st.success, "info": st.info}


@st.cache_data(ttl=300)
def load_state() -> pd.DataFrame:
    """Local parquet first, then the public raw URL, then an empty frame."""
    if STATE_PATH.exists():
        try:
            return pd.read_parquet(STATE_PATH)
        except (OSError, ValueError):
            pass
    raw_url = os.environ.get("STATE_PARQUET_URL", "")
    if raw_url:
        try:
            return pd.read_parquet(raw_url)
        except (OSError, ValueError):
            pass
    return pd.DataFrame(
        columns=["match_id", "minute", *CHART_COLUMNS, "rolling_xg", "flagged"]
    )


@st.cache_data(ttl=55, show_spinner=False)
def load_scoreboard() -> pd.DataFrame:
    return fetch_scoreboard()


@st.cache_data(show_spinner="Simulating live match streams...")
def build_simulation(seed: int) -> pd.DataFrame:
    """Run the full agent pipeline on a deterministic mock match."""
    chat, commentary = simulate_streams(seed)
    return run_selector(chat, commentary)


def run_selector(chat: pd.DataFrame, commentary: pd.Series) -> pd.DataFrame:
    config = load_config(str(CONFIG_PATH))
    params = config["hyperparameters"]
    social = EmotionAgent(window_minutes=5).run(chat)
    match = MatchProgressionAgent(
        window_minutes=int(params["xg_rolling_window_minutes"])
    ).run(commentary)
    selector = ArbitrageSelector(threshold=float(params["arbitrage_flag_threshold"]))
    return fill_emotion_columns(selector.run(social, match))


def active_threshold() -> float:
    config = load_config(str(CONFIG_PATH))
    return float(config["hyperparameters"]["arbitrage_flag_threshold"])


def render_metrics(frame: pd.DataFrame, container) -> None:
    """Native st.metric badges for the latest tick of the supplied frame."""
    latest = frame.iloc[-1]
    previous = frame.iloc[-2] if len(frame) > 1 else latest
    cols = container.columns(5)
    cols[0].metric(
        "Crowd Panic Score",
        f"{latest['crowd_panic_score']:+.2f}",
        delta=f"{latest['crowd_panic_score'] - previous['crowd_panic_score']:+.2f}",
        delta_color="inverse",
    )
    cols[1].metric(
        "xG Stability (10 min)",
        f"{latest['delta_xg_10min']:.2f}",
        delta=f"{latest['delta_xg_10min'] - previous['delta_xg_10min']:+.2f}",
    )
    cols[2].metric(
        "Arbitrage Index",
        f"{latest['arbitrage_index']:.2f}",
        delta=f"{latest['arbitrage_index'] - previous['arbitrage_index']:+.2f}",
        delta_color="inverse",
    )
    if "dominant_emotion" in frame.columns:
        cols[3].metric("Crowd Mood", str(latest.get("dominant_emotion", "neutral")).title())
    cols[4].metric("Flagged Minutes", int(frame["flagged"].sum()))


def render_chart(frame: pd.DataFrame, container) -> None:
    container.line_chart(
        frame.set_index("minute")[CHART_COLUMNS],
        height=340,
        use_container_width=True,
    )


def render_emotions(frame: pd.DataFrame, container) -> None:
    """Stacked emotion-share area chart plus mood stability badge."""
    available = [c for c in EMOTION_COLUMNS if c in frame.columns]
    if not available or frame[available].sum().sum() == 0:
        container.caption("Emotion profile: no classified reactions yet.")
        return
    container.subheader("🎭 Crowd Emotion Profile")
    chart_frame = frame.set_index("minute")[available].rename(columns=EMOTION_LABELS)
    container.area_chart(chart_frame, height=260, use_container_width=True)
    if "emotional_volatility" in frame.columns:
        latest = frame.iloc[-1]
        cols = container.columns(3)
        cols[0].metric("Mood Volatility", f"{latest['emotional_volatility']:.2f}")
        if "comment_volume" in frame.columns:
            cols[1].metric("Reactions This Minute", int(latest["comment_volume"]))
        cols[2].metric(
            "Dominant Emotion", str(latest.get("dominant_emotion", "neutral")).title()
        )


def render_takeaways(frame: pd.DataFrame, match_stats: dict | None = None) -> None:
    st.subheader("💡 What This Means Right Now")
    for takeaway in generate_takeaways(frame, active_threshold(), match_stats):
        renderer = TONE_RENDERERS.get(takeaway["tone"], st.info)
        renderer(f"**{takeaway['headline']}** — {takeaway['detail']}")


def render_flags(frame: pd.DataFrame) -> None:
    flagged = frame.loc[frame["flagged"]].sort_values("arbitrage_index", ascending=False)
    st.subheader("🚩 Flagged Market Panic Moments")
    if flagged.empty:
        st.info("No arbitrage events flagged in the current state window.")
        return
    badge_cols = st.columns(min(4, len(flagged)))
    for slot, (_, row) in zip(badge_cols, flagged.iterrows()):
        slot.metric(
            f"Minute {int(row['minute'])}'",
            f"{row['arbitrage_index']:.2f}",
            delta=f"panic {row['crowd_panic_score']:+.2f} | xG {row['delta_xg_10min']:.2f}",
            delta_color="off",
        )
    st.dataframe(
        flagged[["minute", *CHART_COLUMNS, "rolling_xg"]].round(3),
        use_container_width=True,
        hide_index=True,
    )


def render_reactions(chat: pd.DataFrame) -> None:
    """The 200-comment multi-source window behind the emotion profile."""
    with st.expander(f"💬 Crowd reactions analysed ({len(chat)})"):
        if chat.empty:
            st.caption("No fan reactions matched this fixture yet.")
            return
        if "source" in chat.columns:
            counts = chat["source"].value_counts()
            st.caption(
                "Sources: "
                + ", ".join(f"{src} {n}" for src, n in counts.items())
            )
        st.dataframe(chat.tail(30), use_container_width=True, hide_index=True)


@st.fragment(run_every=60)
def live_panel(event_id: str) -> None:
    """Auto-refreshing live view: refetches all streams every 60 seconds."""
    board = load_scoreboard()
    rows = board.loc[board["event_id"] == event_id]
    if rows.empty:
        st.warning("Fixture no longer present on the scoreboard; try reselecting.")
        return
    match = rows.iloc[0]
    head = st.columns(4)
    head[0].metric("Fixture", str(match["short_name"]))
    head[1].metric("Score", str(match["score"]))
    head[2].metric("Clock", f"{int(match['clock_minute'])}'")
    head[3].metric("Status", str(match["state"]).upper())
    if match["state"] == "pre":
        st.info(f"Kickoff at {match['kickoff_utc']:%H:%M UTC}. Panel refreshes automatically.")
        return
    chat, commentary = live_streams(match)
    if commentary.empty:
        st.info("Waiting for the first commentary entries from the feed...")
        return
    state = run_selector(chat, commentary)
    if state.empty:
        st.info("Streams connected; not enough data to score yet.")
        return
    match_stats = fetch_boxscore(event_id)
    control = control_index(match_stats)
    render_metrics(state, st.container())
    render_chart(state, st.empty())
    render_takeaways(state, match_stats)
    if control is not None and match_stats:
        first_team = next(iter(match_stats))
        st.caption(f"Match control index: {first_team} {control:.0%} of the contest.")
    render_emotions(state, st.container())
    render_flags(state)
    render_reactions(chat)
    st.caption(f"Live mode - last refresh {utc_now():%H:%M:%S UTC}, next in ~60s.")


def render_live_mode() -> None:
    board = load_scoreboard()
    if board.empty:
        st.warning(
            "Scoreboard unreachable or no fixtures today. "
            "Switch to Simulator mode to see the engine in action."
        )
        return
    labels = [
        f"{row['short_name']} ({row['state'].upper()}, {row['kickoff_utc']:%d %b %H:%M} UTC)"
        for _, row in board.iterrows()
    ]
    choice = st.selectbox("Fixture", labels, key="fixture")
    event_id = str(board.iloc[labels.index(choice)]["event_id"])
    live_panel(event_id)


def render_simulator_mode(seed: int, speed: float) -> None:
    metrics_slot = st.empty()
    chart_slot = st.empty()
    if st.button("▶ Live Stream Feed Simulator", type="primary"):
        state = build_simulation(seed)
        progress = st.progress(0, text="Streaming simulated match feed...")
        for tick in range(5, len(state) + 1):
            view = state.iloc[:tick]
            render_metrics(view, metrics_slot.container())
            render_chart(view, chart_slot)
            progress.progress(tick / len(state), text=f"Minute {int(view.iloc[-1]['minute'])}'")
            time.sleep(speed)
        progress.empty()
        st.session_state["sim_state"] = state
    state = st.session_state.get("sim_state", build_simulation(seed))
    render_metrics(state, metrics_slot.container())
    render_chart(state, chart_slot)
    render_takeaways(state)
    render_emotions(state, st.container())
    render_flags(state)


def render_state_mode() -> None:
    state = load_state()
    if state.empty:
        st.warning(
            "No committed engine state available. Run `python src/pipeline.py run` "
            "or wait for the next flywheel cycle."
        )
        return
    if "match_id" in state.columns and len(state):
        st.caption(f"Source run: {state['match_id'].iloc[-1]}")
    render_metrics(state, st.container())
    render_chart(state, st.empty())
    render_takeaways(state)
    render_emotions(state, st.container())
    render_flags(state)


def main() -> None:
    st.set_page_config(
        page_title="WC Sentiment Arbitrage Engine", page_icon="⚽", layout="wide"
    )
    st.title("⚽ WC Sentiment Arbitrage Engine")
    st.caption(
        "Wisdom-of-crowds divergence tracker: crowd emotions vs. real match "
        "stability, flagged when sentiment decouples from the pitch."
    )

    with st.sidebar:
        st.header("Mode")
        mode = st.radio(
            "Data source", [MODE_LIVE, MODE_SIM, MODE_STATE], key="mode", index=0
        )
        st.divider()
        st.header("Simulator Controls")
        seed = int(st.number_input("Match seed", min_value=1, value=20260610, step=1))
        speed = float(st.slider("Tick speed (seconds)", 0.01, 0.50, 0.05, 0.01))
        config = load_config(str(CONFIG_PATH))
        st.divider()
        st.subheader("Active Model Config")
        st.json(config["hyperparameters"])
        history = config.get("log_loss_history", [])
        if history:
            st.caption(f"Last self-correction log-loss: {history[-1]['log_loss']}")

    if mode == MODE_LIVE:
        render_live_mode()
    elif mode == MODE_SIM:
        render_simulator_mode(seed, speed)
    else:
        render_state_mode()


if __name__ == "__main__":
    main()
