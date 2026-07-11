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
import requests
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent))

import consistency
import glossary
import health
import odds
import publish
from advanced import fixture_prior
from archive import load_archive
from emotion import (
    EMOTION_COLUMNS,
    EmotionAgent,
    generate_takeaways,
    headline_outcome,
    scored_share,
    team_emotion_summary,
)
from live import (
    POST_GRACE,
    capture_phase,
    fetch_scoreboard,
    live_streams_buffered,
    utc_now,
)
from matchstats import (
    ADVANCED_STATS,
    KEY_STATS,
    control_index,
    fetch_match_detail,
    fetch_sofascore_momentum,
    goal_scorers,
    keeper_pressure,
    top_performers,
)
from model import ArbitrageSelector, MatchProgressionAgent, load_config
from pipeline import fill_emotion_columns, simulate_streams
from situation import classify, metrics_that_matter, situation_brief

ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "data" / "state.parquet"
STATUS_PATH = ROOT / "data" / "run_status.json"
CONFIG_PATH = ROOT / "data" / "model_config.json"
BENCHMARK_PATH = ROOT / "data" / "benchmarks" / "emotion_benchmark.json"
CHART_COLUMNS = ["crowd_panic_score", "delta_xg_10min", "arbitrage_index"]
CHART_LABELS = {col: glossary.label(col) for col in CHART_COLUMNS}
MODE_LIVE = "🔴 Live match"
MODE_SIM = "🎮 Simulator"
MODE_STATE = "📦 Committed state"
MODE_PUBLISH = "📤 Publish Insights"
EMOTION_LABELS = {col: col.removeprefix("emo_").title() for col in EMOTION_COLUMNS}
TONE_RENDERERS = {"warning": st.warning, "positive": st.success, "info": st.info}
# Friendly stat labels for the per-team live match-stats panel.
STAT_LABELS = {
    "possessionPct": "Possession %",
    "totalShots": "Shots",
    "shotsOnTarget": "On Target",
    "wonCorners": "Corners",
    "saves": "Saves",
    "foulsCommitted": "Fouls",
}
ADVANCED_STAT_LABELS = {
    "accuratePasses": "Accurate Passes",
    "passPct": "Pass %",
    "totalTackles": "Tackles",
    "interceptions": "Interceptions",
    "effectiveClearance": "Clearances",
    "blockedShots": "Blocked Shots",
    "accurateCrosses": "Accurate Crosses",
}
FLAG_REASON_LABELS = {
    "panic-vs-stable": "Panic while the match was stable",
    "positive-while-losing": "Celebration while losing",
    "panic-while-ahead": "Panic while ahead",
}


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


@st.cache_data(ttl=55, show_spinner=False)
def load_run_status() -> dict | None:
    """Engine heartbeat: local file first, then the committed raw URL sibling."""
    status = health.load_status(STATUS_PATH)
    if status is not None:
        return status
    raw_url = os.environ.get("STATE_PARQUET_URL", "")
    if raw_url and "/" in raw_url:
        status_url = raw_url.rsplit("/", 1)[0] + "/run_status.json"
        try:
            response = requests.get(status_url, timeout=10)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError):
            return None
    return None


def render_status_badge() -> None:
    """LIVE / STALE / DEGRADED / NO-DATA badge so an outage never reads as calm."""
    verdict = health.freshness(load_run_status())
    detail = f"**{verdict['label']}** — {verdict['detail']}"
    if verdict["level"] == "live":
        st.success(detail)
    elif verdict["level"] in ("stale", "degraded"):
        st.warning(detail)
    else:
        st.info(detail)


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
    return classify(fill_emotion_columns(selector.run(social, match)))


def active_threshold() -> float:
    config = load_config(str(CONFIG_PATH))
    return float(config["hyperparameters"]["arbitrage_flag_threshold"])


def render_metrics(frame: pd.DataFrame, container) -> None:
    """Native st.metric badges for the latest tick of the supplied frame."""
    latest = frame.iloc[-1]
    previous = frame.iloc[-2] if len(frame) > 1 else latest
    cols = container.columns(5)
    cols[0].metric(
        glossary.label("crowd_panic_score"),
        f"{latest['crowd_panic_score']:+.2f}",
        delta=f"{latest['crowd_panic_score'] - previous['crowd_panic_score']:+.2f}",
        delta_color="inverse",
        help=glossary.tooltip("crowd_panic_score"),
    )
    cols[1].metric(
        glossary.label("delta_xg_10min"),
        f"{latest['delta_xg_10min']:.2f}",
        delta=f"{latest['delta_xg_10min'] - previous['delta_xg_10min']:+.2f}",
        help=glossary.tooltip("delta_xg_10min"),
    )
    cols[2].metric(
        glossary.label("arbitrage_index"),
        f"{latest['arbitrage_index']:.2f}",
        delta=f"{latest['arbitrage_index'] - previous['arbitrage_index']:+.2f}",
        delta_color="inverse",
        help=glossary.tooltip("arbitrage_index"),
    )
    if "dominant_emotion" in frame.columns:
        cols[3].metric(
            "Loudest Emotion",
            str(latest.get("dominant_emotion", "neutral")).title(),
            help="The single emotion the most fans are expressing right now.",
        )
    cols[4].metric(
        glossary.label("flagged"),
        int(frame["flagged"].sum()),
        help=glossary.tooltip("flagged"),
    )


def render_chart(frame: pd.DataFrame, container) -> None:
    container.line_chart(
        frame.set_index("minute")[CHART_COLUMNS].rename(columns=CHART_LABELS),
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


def render_situation(frame: pd.DataFrame, priors_note: str = "") -> None:
    """The classifier's verdict + the metrics that matter in this situation."""
    if frame.empty or "situation" not in frame.columns:
        return
    latest = frame.iloc[-1]
    brief = situation_brief(str(latest["situation"]))
    st.subheader("🧭 Match Read (live)")
    cols = st.columns(2 + len(brief["metrics"]))
    cols[0].metric(
        "Match Read",
        str(brief["label"]),
        help="The model's plain read of the moment, classified every minute.",
    )
    cols[1].metric("Confidence", f"{float(latest['situation_confidence']):.0%}")
    for slot, (column, label, value) in zip(
        cols[2:], metrics_that_matter(frame)
    ):
        slot.metric(label, f"{value:+.2f}" if column == "crowd_panic_score" else f"{value:.2f}")
    st.caption(str(brief["read"]) + (f" {priors_note}" if priors_note else ""))


@st.cache_data(ttl=300, show_spinner=False)
def load_market(home_team: str, away_team: str) -> dict | None:
    """De-vigged bookmaker market for the fixture (cached; key-gated)."""
    if not odds.odds_enabled():
        return None
    return odds.fixture_market(home_team, away_team)


def render_market(market: dict, latest_panic: float) -> None:
    """Real market consensus next to the crowd signal (read-only odds)."""
    st.subheader("📊 Bookmakers vs Crowd")
    cols = st.columns(4)
    for slot, key, label in (
        (cols[0], "home_prob", "Home win"),
        (cols[1], "draw_prob", "Draw"),
        (cols[2], "away_prob", "Away win"),
    ):
        value = market.get(key)
        slot.metric(label, "—" if value is None else f"{value:.0%}")
    divergence = odds.sentiment_market_divergence(latest_panic, market["certainty"])
    cols[3].metric(
        glossary.label("sentiment_market_divergence"),
        f"{divergence:.2f}",
        help=glossary.tooltip("sentiment_market_divergence"),
    )
    st.caption(
        f"Market-implied favorite: {market['favorite']} "
        f"({market['certainty']:.0%} certainty). Read-only bookmaker consensus — "
        "not betting advice."
    )


def render_takeaways(
    frame: pd.DataFrame, match_stats: dict | None = None, keeper: dict | None = None
) -> None:
    st.subheader("💡 What This Means Right Now")
    for takeaway in generate_takeaways(frame, active_threshold(), match_stats, keeper):
        renderer = TONE_RENDERERS.get(takeaway["tone"], st.info)
        renderer(f"**{takeaway['headline']}** — {takeaway['detail']}")


def render_flags(frame: pd.DataFrame, conflicts: list[dict] | None = None) -> None:
    flagged = frame.loc[frame["flagged"]].sort_values("arbitrage_index", ascending=False)
    st.subheader("🚩 Overreaction Moments")
    st.caption(
        "**Definition:** minutes where fan mood conflicts with the game "
        "situation — panic or anger while the match is stable, or celebration "
        "and confidence while the team is losing and creating nothing."
    )
    for moment in conflicts or []:
        label = FLAG_REASON_LABELS.get(str(moment.get("reason")), "Mood-vs-game conflict")
        st.warning(f"**{label} — {moment.get('team', '')} (now):** {moment.get('detail', '')}")
    if flagged.empty:
        if not conflicts:
            st.info("No overreaction moments in the current window.")
        return
    st.caption("Flagged minutes (reason: panic while the match was stable):")
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
    """The rolling multi-source reaction window behind the emotion profile."""
    with st.expander(f"💬 Crowd reactions analysed ({len(chat)})"):
        if chat.empty:
            st.caption("No fan reactions matched this fixture yet.")
            return
        st.caption(
            "Rolling window of up to 1,000 reactions, accumulated across the "
            "match and cleaned of bots, links, emoji-only spam and off-topic chatter."
        )
        if "source" in chat.columns:
            counts = chat["source"].value_counts()
            st.caption(
                "Sources: "
                + ", ".join(f"{src} {n}" for src, n in counts.items())
            )
        if "team" in chat.columns:
            team_counts = chat["team"].value_counts().to_dict()
            st.caption(
                "Team tag: "
                + ", ".join(f"{k} {v}" for k, v in team_counts.items())
            )
        if "message" in chat.columns:
            share = scored_share(chat["message"])
            st.caption(
                f"Emotion-lexicon coverage: {share:.0%} of comments matched a "
                "term (English + major non-English + emoji). Lower coverage means "
                "the panic score rests on a smaller slice of the crowd."
            )
        st.dataframe(chat.tail(30), use_container_width=True, hide_index=True)


def render_headline(
    state: pd.DataFrame, team_summary: dict | None = None
) -> None:
    """The single plain-language 'what's happening right now' sentence."""
    sentence = headline_outcome(state, active_threshold(), team_summary)
    st.subheader("📣 Right Now")
    st.info(sentence)


def render_team_moods(team_summary: dict, verdicts: dict | None = None) -> None:
    """Per-team crowd mood with clarity + a mood-vs-game consistency check."""
    if not team_summary:
        return
    st.subheader("👥 Mood by Team")
    sides = [s for s in ("home", "away") if s in team_summary]
    cols = st.columns(len(sides))
    icons = {"home": "🏠", "away": "🛫"}
    for slot, side in zip(cols, sides):
        info = team_summary[side]
        verdict = (verdicts or {}).get(side, {})
        volume = int(info.get("volume", 0))
        slot.metric(
            f"{icons[side]} {info.get('team', side)} fans",
            str(info.get("dominant", "neutral")).title(),
            help=f"Loudest emotion among {volume} reactions mentioning this team "
            f"(reading coverage {float(info.get('coverage', 0.0)):.0%}).",
        )
        clarity = verdict.get("clarity")
        if clarity is None:
            clarity = consistency.clarity_score(info)
        slot.metric(
            glossary.label("clarity"),
            f"{float(clarity):.0%}",
            help=glossary.tooltip("clarity"),
        )
        slot.caption(f"{volume} reactions")
        if verdict.get("verdict") == "conflict":
            slot.warning(f"⚠️ {verdict.get('explanation', 'Mood conflicts with the game situation.')}")
        elif verdict:
            slot.caption(f"✓ {verdict.get('explanation', '')}")
    st.caption(
        "Each reaction is tagged by the team it names; posts mentioning both "
        "count for both sides. Mood is checked against the live scoreline - a "
        "losing side reading as joyful is flagged, not hidden."
    )


def render_match_stats(match_stats: dict, momentum: pd.DataFrame | None = None) -> None:
    """Clear per-team live match statistics beside the crowd mood."""
    if not match_stats:
        st.caption("Live match statistics not available from the feed yet.")
        return
    st.subheader("📈 Live Match Stats")
    teams_order = list(match_stats.keys())
    table = {
        STAT_LABELS[key]: [match_stats[t].get(key, "—") for t in teams_order]
        for key in KEY_STATS
        if key in STAT_LABELS
    }
    frame = pd.DataFrame(table, index=teams_order).T
    st.dataframe(frame, use_container_width=True)
    advanced = {
        ADVANCED_STAT_LABELS[key]: [match_stats[t].get(key, "—") for t in teams_order]
        for key in ADVANCED_STATS
        if key in ADVANCED_STAT_LABELS
        and any(key in match_stats[t] for t in teams_order)
    }
    if advanced:
        with st.expander("📊 Advanced team stats (passing & defending)"):
            st.dataframe(
                pd.DataFrame(advanced, index=teams_order).T,
                use_container_width=True,
            )
    control = control_index(match_stats)
    if control is not None and teams_order:
        st.caption(
            f"Match Control: {teams_order[0]} holds {control:.0%} of the contest "
            "(possession + shots blend)."
        )
    if momentum is not None and not momentum.empty:
        st.caption("Attack momentum (Sofascore) — above 0 favours the home side:")
        st.area_chart(
            momentum.set_index("minute")["momentum"], height=160,
            use_container_width=True,
        )
    else:
        st.caption(
            "Deeper xG/attack-momentum runs in local mode only — Sofascore "
            "blocks cloud server IPs, so the public app shows the ESPN stats above."
        )


def render_performers(
    leaders: dict, key_events: list, keeper: dict | None = None
) -> None:
    """Key performers per team + goalscorers, from the same ESPN payload."""
    lines = top_performers(leaders)
    goals = goal_scorers(key_events)
    if not lines and not goals and not keeper:
        return
    st.subheader("⭐ Key Performers")
    if goals:
        st.caption("Goals: " + " | ".join(goals))
    for team, line in lines.items():
        st.caption(f"**{team}** — {line}")
    for team, info in (keeper or {}).items():
        saves = float(info.get("saves", 0.0))
        if saves >= 3:
            st.caption(
                f"🧤 {info.get('keeper', '')} ({team}) is busy: {saves:.0f} saves"
                + (
                    f" from {float(info.get('shots_faced', 0.0)):.0f} shots faced."
                    if float(info.get("shots_faced", 0.0)) > 0
                    else "."
                )
            )


@st.cache_data(ttl=3600, show_spinner=False)
def load_benchmark() -> dict | None:
    import json

    if not BENCHMARK_PATH.exists():
        return None
    try:
        return json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def render_validation(live_coverage: float | None = None) -> None:
    """Surface the emotion model's real validation so categorisation is trusted."""
    bench = load_benchmark()
    with st.expander("✅ How accurate is the emotion model?"):
        if not bench:
            st.caption("Validation record not available.")
            return
        trained = bench.get("trained_emotion", {})
        lexicon = bench.get("lexicon", {})
        acc = float(trained.get("test_accuracy", 0.0))
        cv_mean = float(trained.get("cv_accuracy_mean", 0.0))
        cv_std = float(trained.get("cv_accuracy_std", 0.0))
        st.markdown(
            f"**The model agrees with human-labelled data {acc:.0%} of the time** "
            f"on a held-out test set, and {cv_mean:.0%} (±{cv_std:.0%}) across "
            f"5-fold cross-validation — versus only "
            f"{float(lexicon.get('emotion_accuracy', 0.0)):.0%} for a plain "
            "keyword approach."
        )
        st.dataframe(
            pd.DataFrame(
                {
                    "Trained model": [
                        f"{acc:.1%}",
                        f"{float(trained.get('test_macro_f1', 0.0)):.1%}",
                        f"{cv_mean:.1%} ± {cv_std:.1%}",
                    ],
                    "Keyword baseline": [
                        f"{float(lexicon.get('emotion_accuracy', 0.0)):.1%}",
                        f"{float(lexicon.get('emotion_macro_f1', 0.0)):.1%}",
                        "—",
                    ],
                },
                index=["Test accuracy", "Macro F1", "Cross-validation"],
            ),
            use_container_width=True,
        )
        langs = bench.get("trained_sentiment", {}).get("per_language_accuracy", {})
        if langs:
            st.caption(
                "Multilingual sentiment accuracy by language: "
                + ", ".join(f"{k} {float(v):.0%}" for k, v in langs.items())
            )
        datasets = bench.get("datasets", {})
        st.caption(
            "Trained & tested on public datasets: "
            + ", ".join(f"{k} ({v})" for k, v in datasets.items())
            + ". Exported to pure-numpy with parity to scikit-learn."
        )
        st.caption(
            "For football-specific phrasing (e.g. 'we are done', 'disgrace'), the "
            "model is anchored by a curated football+emoji lexicon so the live "
            "categorisation stays accurate on this domain, not just the benchmark."
        )
        if live_coverage is not None:
            st.caption(
                f"On the current live window the model could read "
                f"{live_coverage:.0%} of captured posts ({glossary.label('scored_share')})."
            )


def render_full_panel(
    state: pd.DataFrame,
    chat: pd.DataFrame,
    match_stats: dict,
    priors_note: str = "",
    team_summary: dict | None = None,
    momentum: pd.DataFrame | None = None,
    verdicts: dict | None = None,
    detail: dict | None = None,
) -> None:
    keeper = keeper_pressure(detail.get("players", {})) if detail else None
    conflicts = consistency.conflict_moments(verdicts) if verdicts else []
    render_headline(state, team_summary)
    render_metrics(state, st.container())
    if team_summary:
        render_team_moods(team_summary, verdicts)
    render_chart(state, st.empty())
    render_situation(state, priors_note)
    render_match_stats(match_stats, momentum)
    if detail:
        render_performers(
            detail.get("leaders", {}), detail.get("key_events", []), keeper
        )
    render_takeaways(state, match_stats, keeper)
    render_emotions(state, st.container())
    render_flags(state, conflicts)
    live_coverage = scored_share(chat["message"]) if "message" in chat.columns else None
    render_validation(live_coverage)
    render_reactions(chat)


def priors_note_for(match: pd.Series) -> str:
    """One-line FBref priors context for the fixture, when available."""
    prior = fixture_prior(str(match.get("home_team", "")), str(match.get("away_team", "")))
    if prior is None:
        return ""
    parts: list[str] = []
    for side in ("home", "away"):
        row = prior[side]
        if row is not None:
            parts.append(
                f"{row['team']}: {row['goals_for_per_match']:.1f} gf / "
                f"{row['goals_against_per_match']:.1f} ga per match"
            )
    note = "Tournament priors (FBref) - " + "; ".join(parts) + "."
    if prior["edge"] is not None:
        note += f" Scoring edge {prior['edge']:+.2f} to the home side."
    return note


@st.fragment(run_every=60)
def live_panel(event_id: str) -> None:
    """Auto-refreshing live view with a post-match start/stop filter.

    Collection phases: pre (idle) -> live (full fetch) -> post-window
    (15 more minutes of fetching after full time to capture the emotional
    settle) -> frozen (all fetching stops; archived data only).
    """
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

    post_seen = st.session_state.setdefault("post_first_seen", {})
    if match["state"] == "post" and event_id not in post_seen:
        post_seen[event_id] = utc_now()
    phase = capture_phase(
        str(match["state"]), match["kickoff_utc"], post_first_seen=post_seen.get(event_id)
    )

    if phase == "pre":
        st.info(f"Kickoff at {match['kickoff_utc']:%H:%M UTC}. Panel refreshes automatically.")
        return

    if phase == "frozen":
        snapshot = st.session_state.get(f"final_snapshot_{event_id}")
        if snapshot is not None:
            st.info(
                "🏁 Match ended — data collection stopped 15 minutes after "
                "full time. Showing the final captured state."
            )
            render_full_panel(*snapshot)
            return
        archived = load_archive(f"ESPN-{event_id}")
        if not archived.empty:
            st.info(
                "🏁 Match ended — collection window closed. Showing the "
                "archived match record."
            )
            render_metrics(archived, st.container())
            render_chart(archived, st.empty())
            render_emotions(archived, st.container())
            render_flags(archived)
            return
        st.info(
            "🏁 Match ended and its collection window has closed. No archive "
            "is available for this fixture yet."
        )
        return

    buffers = st.session_state.setdefault("reaction_buffer", {})
    chat, commentary, merged_raw = live_streams_buffered(match, buffers.get(event_id))
    buffers[event_id] = merged_raw
    if commentary.empty:
        if str(match["state"]) == "in":
            st.warning(
                "⚠️ Match is in progress but the commentary feed returned nothing — "
                "the ESPN source may be rate-limited or briefly unreachable. "
                "Retrying automatically; this is a source outage, not a quiet crowd."
            )
        else:
            st.info("Waiting for the first commentary entries from the feed...")
        return
    state = run_selector(chat, commentary)
    if state.empty:
        st.info("Streams connected; not enough data to score yet.")
        return
    detail = fetch_match_detail(event_id)
    match_stats = detail.get("stats", {})
    home_team = str(match.get("home_team") or "")
    away_team = str(match.get("away_team") or "")
    team_summary = team_emotion_summary(chat, home_team, away_team)
    momentum = fetch_sofascore_momentum(home_team, away_team)
    latest_threat = float(state.iloc[-1]["delta_xg_10min"]) if not state.empty else 0.0
    context = consistency.game_context(
        str(match.get("score") or ""),
        home_team,
        away_team,
        latest_threat,
        keeper_pressure(detail.get("players", {})),
    )
    verdicts = consistency.mood_consistency(team_summary, context)
    if phase == "post-window":
        first_seen = post_seen.get(event_id, utc_now())
        remaining = max(0, int((POST_GRACE - (utc_now() - first_seen)).total_seconds() // 60))
        st.warning(
            f"⏱️ Full time — post-match capture window open, collection "
            f"stops in ~{remaining} min."
        )
        st.session_state[f"final_snapshot_{event_id}"] = (state, chat, match_stats)
    render_full_panel(
        state,
        chat,
        match_stats,
        priors_note_for(match),
        team_summary,
        momentum,
        verdicts,
        detail,
    )
    market = load_market(str(match.get("home_team", "")), str(match.get("away_team", "")))
    if market:
        render_market(market, float(state.iloc[-1]["crowd_panic_score"]))
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
    render_headline(state)
    render_metrics(state, metrics_slot.container())
    render_chart(state, chart_slot)
    render_situation(state)
    render_takeaways(state)
    render_emotions(state, st.container())
    render_flags(state)
    render_validation()


def render_state_mode() -> None:
    render_status_badge()
    state = load_state()
    if state.empty:
        st.warning(
            "No committed engine state available. Run `python src/pipeline.py run` "
            "or wait for the next flywheel cycle."
        )
        return
    if "match_id" in state.columns and len(state):
        st.caption(f"Source run: {state['match_id'].iloc[-1]}")
    render_headline(state)
    render_metrics(state, st.container())
    render_chart(state, st.empty())
    render_situation(state)
    render_takeaways(state)
    render_emotions(state, st.container())
    render_flags(state)
    render_validation()


def render_publish_mode() -> None:
    """Draft honest, labelled insight posts for Bluesky and read their reach."""
    st.subheader("📤 Publish Insights to Bluesky")
    st.caption(
        "Shares the engine's genuine, clearly-labelled analytics from your own "
        "account (only when you click), and reads organic engagement on those "
        "posts. Not an influence tool — no autonomous posting, no crowd targeting."
    )
    state = load_state()
    headline = headline_outcome(state, active_threshold()) if not state.empty else ""
    draft = publish.draft_post(state, headline)
    text = st.text_area("Post preview (editable)", value=draft, height=160)
    st.caption(f"{len(text)}/290 characters")
    if publish.enabled():
        if st.button("Post to Bluesky", type="primary"):
            uri = publish.post_insight(text)
            if uri:
                st.success("Posted. It may take a moment to appear below.")
            else:
                st.error("Post failed — check the app-password secret and try again.")
    else:
        st.info(
            "Preview only. Set BLUESKY_HANDLE + BLUESKY_APP_PASSWORD (an app "
            "password from Bluesky → Settings → App Passwords) to enable posting."
        )
    st.divider()
    st.subheader("📈 Engagement on your recent posts")
    feed = publish.recent_engagement()
    if feed.empty:
        st.caption("No posts found yet (set BLUESKY_HANDLE to read your feed).")
    else:
        st.dataframe(feed, use_container_width=True, hide_index=True)
        st.caption(
            "Descriptive reach only — likes/reposts/replies on your posts. Crowd "
            "mood during a live match is driven by the game, not by these posts."
        )


def main() -> None:
    st.set_page_config(
        page_title="World Cup Crowd Mood Engine", page_icon="⚽", layout="wide"
    )
    st.title(glossary.TITLE)
    st.caption(glossary.SUBTITLE)
    with st.expander("❓ What am I looking at?"):
        for heading, body in glossary.GUIDE:
            st.markdown(f"**{heading}** — {body}")
    st.caption(
        "ℹ️ The Hype-vs-Reality Gap measures **crowd mood vs. real match "
        "action** as a proxy for fan overreaction. It is not connected to any "
        "live betting market and is not financial or betting advice."
    )

    with st.sidebar:
        st.header("Mode")
        mode = st.radio(
            "Data source",
            [MODE_LIVE, MODE_SIM, MODE_STATE, MODE_PUBLISH],
            key="mode",
            index=0,
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
    elif mode == MODE_PUBLISH:
        render_publish_mode()
    else:
        render_state_mode()


if __name__ == "__main__":
    main()
