# 🧠 Project Memory Bank: WC Sentiment Arbitrage Engine

## 🎯 System Intent & Core Niche
This system is an autonomous, 100% free-to-run, self-correcting MLOps pipeline. It tracks real-time crowd sentiment spikes across social media (X/YouTube Live Chat streams) and maps them against live pitch event data during the FIFA World Cup. It flags **Market Arbitrage Moments**—instances where public sentiment drastically decouples from actual underlying match efficiency (e.g., extreme crowd panic despite stable match control).

---

## 🏗️ Free Infrastructure Architecture
To stay completely zero-cost, the system strictly implements this stateless serverless configuration:
* **Orchestration:** GitHub Actions (Event-driven & Midnight Cron loops).
* **Storage Engine:** DuckDB (OLAP relational processing engine reading/writing directly to local states).
* **Data Serialization:** Compressed `.parquet` database files saved to the Git tree or Hugging Face Dataset buckets.
* **Frontend UI:** Streamlit Web Application deployed via Streamlit Community Cloud (streaming parquet values directly via public repository Raw URLs).
* **LLM Inference / Listeners:** Free OpenRouter endpoint mappings (Llama 3 / Mistral 7B) or Groq API free allowances.

---

## 🤖 AI Agentic & Mathematical Workflows
The application bypasses heavy framework overheads (like LangChain) and uses direct, vectorized Python Multi-Agent modules:

```text
 [Social Media Stream/Scrapes] ──> [Social Listening Agent] ──> Computed "Crowd Panic Score"
                                                                         │
 [Match Commentary Text RSS]   ──> [Match Progression Agent] ──> Rolling Expected Goals (xG)
                                                                         │
                                                                         ▼
                                   [Arbitrage Selector]      ──> Flags Disconnect Delta ($Arbitrage_{Index}$)
```

### Core Arbitrage Equation
The divergence index is dynamically formulated using:

$Arbitrage_{Index} = |Crowd\ Panic\ Score| \times (1.0 - \Delta xG_{10min})$

Where Crowd Panic Score is bounded between [-1.0, 1.0] and $\Delta xG$ represents the real-time offensive threat stability index.

---

## 📂 File Registry & Token Guardrails

```text
WC-sentiment-Engine/
├── .github/workflows/
│   ├── engine_flywheel.yml    # Daily cron executor & parameter optimization (test-gated)
│   ├── ci.yml                 # Push/PR gate: ruff, bandit, pip-audit, pytest, smoke run
│   └── codeql.yml             # Weekly + PR CodeQL static security analysis
├── src/
│   ├── model.py               # Vectorized sentiment math, agents, & arbitrage calculations
│   ├── emotion.py             # Custom 6-emotion classifier, mood volatility, takeaway generator
│   ├── sources.py             # Multi-source reaction aggregator (Bluesky/Mastodon keyless; Reddit/YouTube key-gated), 200-comment window
│   ├── matchstats.py          # ESPN boxscore control index + optional ScraperFC/Sofascore momentum
│   ├── live.py                # ESPN connectors, minute mapping, capture_phase start/stop lifecycle
│   ├── archive.py             # NOT NULL match archive: match_archive + match_results (DuckDB + Parquet mirror)
│   ├── pipeline.py            # Live-first DuckDB ingestion, parsing, and Parquet pipeline
│   └── app.py                 # Streamlit frontend: live mode (60s auto-refresh), simulator, committed state
├── tests/
│   ├── conftest.py            # src path setup + offline/no-LLM-key isolation fixture
│   ├── test_model.py          # Unit tests: sentiment math, parsing, equation, grid search
│   ├── test_live.py           # Offline fixture tests for ESPN/Bluesky parsers & failure paths
│   ├── test_pipeline.py       # Integration tests: DuckDB, Parquet, source selection, self-correction
│   └── test_app.py            # Streamlit AppTest smoke tests across display modes
├── data/
│   ├── database.duckdb        # Local testing instance (Git-ignored)
│   ├── model_config.json      # Dynamic hyperparameters & log-loss backpropagation history
│   └── state.parquet          # Compressed operational UI dataset source
├── requirements.txt           # Explicitly pinned runtime library weights (CVE-audited)
├── requirements-dev.txt       # Pinned test & security tooling (pytest/ruff/bandit/pip-audit)
└── CLAUDE.md                  # Context persistence engine (This File)
```

---

## ⛔ Coding Constraints (Do Not Violate)
1. **No Token Bloat:** Do not implement custom HTML/CSS wrappers inside Streamlit UI files.
2. **Resource Integrity:** Every operation calling DuckDB connections must utilize `try...finally` resource closure statements to guarantee zero file locking errors on GitHub Action virtual machines.
3. **No Code Stubs:** Every script must be 100% operational. Never commit files containing `# Insert logic here` placeholders.

---

## ⏳ Project Progress & Evolution Logs
**Status:** Engine fully operational (v0.1.0).

**Completed (2026-06-10):**
1. Workspace shell + `requirements.txt` pinned to verified local versions.
2. `src/model.py` — vectorized lexicon sentiment, commentary event parser, rolling xG stability, arbitrage equation, log-loss grid search, three agent dataclasses, optional OpenRouter/Groq LLM refinement (graceful fallback when no API key).
3. `src/pipeline.py` — deterministic stream simulator (date-seeded), optional live commentary scraping (`COMMENTARY_FEED_URL` + BeautifulSoup), DuckDB table management with `try...finally` closure, zstd Parquet export via DuckDB `COPY`, nightly `optimize` self-correction writing back to `model_config.json`.
4. `src/app.py` — wide-layout Streamlit dashboard: divergence line chart, Live Stream Feed Simulator button with animated ticker, `st.metric` panic badges. Verified via headless boot + `AppTest` (zero exceptions, button path included).
5. `.github/workflows/engine_flywheel.yml` — midnight cron + manual dispatch, runs `pipeline.py all`, commits refreshed `state.parquet`/`model_config.json`.

**Completed (2026-06-10, CI/CD hardening pass):**
6. `tests/` — 24-test suite (unit + integration + Streamlit AppTest), all passing locally; autouse fixture strips LLM keys so tests never make network calls.
7. `.github/workflows/ci.yml` — push/PR gate: ruff lint, bandit static security scan, pip-audit CVE check on pinned deps, pytest, and a full pipeline smoke run. Least-privilege `contents: read`, actions pinned to commit SHAs, `persist-credentials: false`, concurrency cancellation.
8. `.github/workflows/codeql.yml` — CodeQL `security-and-quality` analysis on push/PR/weekly cron.
9. `engine_flywheel.yml` hardened — actions pinned to SHAs, top-level `contents: read` with job-level `write`, concurrency lock, 30-min timeout, and the nightly run now gates on the test suite before touching state.
10. Dependency CVE remediation: pip-audit flagged streamlit 1.46.1 (CVE-2026-33682) and pyarrow 20.0.0 (PYSEC-2026-113); both upgraded and re-pinned (streamlit 1.54.0, pyarrow 23.0.1), full suite re-verified green, audit now clean.

**Completed (2026-06-10, real-time live-match pass):**
11. `src/live.py` — zero-cost, no-API-key live connectors: ESPN public site API (`fifa.world` scoreboard + per-match play-by-play commentary, league overridable via `ESPN_LEAGUE`) for Agent B, and Bluesky public post search for Agent A's real crowd sentiment. All fetchers verified against live endpoints (real 2026 WC fixtures returned; 95 real posts scored through the sentiment agent). Reddit JSON was evaluated and rejected (403 for unauthenticated clients).
12. `pipeline.py` `gather_streams()` — live-first source selection: in-progress ESPN match → `COMMENTARY_FEED_URL` scrape → deterministic simulator. `match_id` prefixes: `ESPN-` / `FEED-` / `SIM-`.
13. `app.py` three modes — **Live match** (fixture picker + `st.fragment(run_every=60)` panel that refetches both streams and recomputes agents every 60s while the page is open, with score/clock/status badges and a crowd-post expander), Simulator, Committed state.
14. `engine_flywheel.yml` dual cron — `*/20 * * * *` live refresh (commit step no-ops when state unchanged, so off-match ticks produce no commits) + `0 0 * * *` nightly test-gated self-correction.
15. Real-data regression fixed: Bluesky mixed-precision timestamps coerced via `pd.to_datetime(utc=True)` in `posts_to_chat`. Suite now 37 tests, all offline/deterministic.

**Real-time latency profile:** portal open during a match ≈ 60s end-to-end; committed state for closed browsers ≈ 20 min via Actions; sentiment source is Bluesky (free X/YouTube firehoses do not exist).

**Deployed (2026-06-10):** Repository live at https://github.com/harshul27/WC-Sentiment-Engine. First production validation complete: CI green on GitHub runners, manual flywheel run passed the test gate, ingested, self-corrected, and autonomously committed `data/state.parquet` + `model_config.json` back to main (`0daaa4e`). Dependabot immediately opened 3 update PRs, each gated by CI — the security loop is functioning.

**Completed (2026-06-11, emotion intelligence & product pass):**
16. `src/emotion.py` — custom vectorized 6-emotion classifier (panic/anger/joy/confidence/despair/surprise) with per-minute distributions, dominant-emotion tracking, mood-volatility index, emotion-derived panic score, and a rule-based takeaway generator that relates crowd emotion to match state in plain language.
17. `src/sources.py` — multi-platform reaction aggregator, unified schema, 200-comment rolling window (`COMMENT_WINDOW`). Keyless: Bluesky + Mastodon. Key-gated (free tiers): Reddit match-thread comments (`REDDIT_CLIENT_ID`/`REDDIT_CLIENT_SECRET`), YouTube live chat (`YOUTUBE_API_KEY`, ~105 units/poll). Verified live: 200/200 window filled (116 Mastodon + 84 Bluesky).
18. `src/matchstats.py` — ESPN boxscore (possession/shots/on-target/corners/saves, keyless, cloud-safe, verified vs 2022 final) → weighted match-control index; Sofascore attack momentum via ScraperFC behind `ENABLE_SOFASCORE=1` (Sofascore 403s plain HTTP and blocked botasaurus during testing — best-effort local enrichment only, NOT in requirements.txt).
19. Dashboard v2 — emotion area chart, crowd-mood badges, "What This Means Right Now" takeaway panel, source-labelled reaction window. State schema extended with emo_* shares, dominant_emotion, emotional_volatility, comment_volume.
20. Suite now 64 tests, all offline. First REAL live ingestion achieved during the MEX–RSA opener: `match_id=ESPN-760415`, 115 minutes scored, 2 arbitrage flags.

**Completed (2026-06-11, capture lifecycle & match archive pass):**
21. `live.capture_phase` start/stop filter — pre → live → post-window (15-minute grace after FT to capture the emotional settle, `POST_GRACE`) → frozen (all fetching stops; 180-min `MATCH_MAX_DURATION` guard freezes long-finished fixtures immediately). `current_capture_match` extends pipeline source selection to finished matches still inside their window.
22. Dashboard lifecycle — live panel shows a countdown banner during the post-window, snapshots the final state into session, and when frozen serves the snapshot or the committed archive with zero network fetches.
23. `src/archive.py` — strict schema: `match_archive` (PK match_id+minute, every column NOT NULL with CHECK range constraints on scores/shares/indexes) + `match_results` (fixture metadata, final score = ground truth for the self-correction loop). Idempotent delete+insert upserts; DuckDB tables mirrored to committed `data/match_archive.parquet` / `match_results.parquet` so the corpus survives ephemeral runners. Flywheel commit step now `git add -A data/`.
24. First real archive captured: ESPN-760415 Mexico 2-0 South Africa, 131 validated minute-rows, zero nulls. Suite: 74 tests.

**Next Session Focus:** Wire `run_optimize` to train on the growing `match_archive` + `match_results` corpus (real outcomes) instead of simulated ground truth once a few matches accumulate. Reboot Streamlit Cloud app. Optional secrets for deeper coverage: `REDDIT_CLIENT_ID`/`REDDIT_CLIENT_SECRET`, `YOUTUBE_API_KEY`.
