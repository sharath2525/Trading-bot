# 🏛️ Trading Bot Bug Council — Full Audit Report v10

**Date:** 2026-05-28  
**Audit depth:** v10 — deepest pass to date; all 10+ source files read in full  
**Previous versions:** v1–v9 reports on file; v9 confirmed Kronos pre-warm fix, trailing stop zero-size fix, stale candle halt, unfilled limit cancel recovery, outer-loop ADX=None bypass  
**Scope:** Bug finding and gap analysis ONLY. No new features. No rewrites. Existing code analyzed against its own architecture and MASTER RULES.

---

## 📋 BOT SUMMARY

| Field | Value |
|-------|-------|
| Strategy | CODE-FIRST hybrid — code owns all direction/TP/SL/size decisions |
| Asset class | Perpetual futures (Hyperliquid DEX) |
| Network | **MAINNET — REAL FUNDS** |
| Max leverage | 5× (configured) |
| Order types | LIMIT entry (0.15% improvement); trigger SL; limit TP |
| Position model | Per-asset state machine (IDLE → ENTERED → COOLDOWN), partial-close TP1/TP2 |
| AI gate | Claude Sonnet 4.6, 5-factor analysis, APPROVE ≥ 15/25, fail-closed |
| Risk controls FOUND | ATR sizing, PCT cap, drawdown circuit breaker, daily trade cap, cooldown, max concurrent positions, max exposure, reserve floor, SL enforcement |
| Risk controls MISSING | Remote kill switch, cross-session cumulative drawdown, slippage cap on SL execution |
| Testnet mode | Absent from current .env |
| DeFi agent | Not activated (Hyperliquid uses REST/WS, not on-chain smart contracts) |

**CURRENT STATUS: BOT CANNOT START** — `API_HOST=0.0.0.0` + blank `DASHBOARD_TOKEN` triggers `sys.exit(1)` security gate. This is the 4th consecutive audit in which this blocker is present.

---

## 🔨 AGENT 1 — LOGIC BREAKER

### FUND-LOSS / CRITICAL

**LB-v10-1 | Guardian TP2 permanently orphaned when TP1 is placed but TP2 is not**  
File: `src/main.py` lines ~1118–1270  
Variable: `_g_has_tp1` (misleadingly named — detects ANY TP trigger order, not specifically TP1)  
The guardian condition is `if not _g_has_tp1`. This fires only when ZERO TP orders are present. If TP1 was successfully placed and TP2 was NOT (e.g. connection dropped between the two place calls, or TP2 was rejected due to minimum order size on the 50% remainder), `_g_has_tp1 = True` → the re-placement block never fires → TP2 is permanently orphaned.  
**Financial consequence:** 50% of a winning position has no take-profit. The trade rides until SL fires or manual intervention. At 5× leverage on a position that would have closed at +1.5–3×ATR profit, this is a significant unrealized gain surrendered indefinitely.  
**Fix direction:** Add a separate `_g_has_tp2` detection variable. Re-place TP2 independently when `_g_has_tp2` is False, regardless of `_g_has_tp1` state.

---

**LB-v10-2 | HIP-3 prefixed assets never trigger force-close**  
File: `src/trading/hyperliquid_api.py` → `get_user_state()` + `src/main.py` → `check_losing_positions()`  
`get_user_state()` calls `get_current_price()` per open position to compute unrealized PnL. For HIP-3 prefixed tickers (e.g. `xyz:GOLD`, `xyz:TSLA`, `xyz:SPX`), `get_current_price()` returns `0` if the price lookup fails for any reason (API returns null, ticker mismatch). The function then sets `pnl = 0.0`. `check_losing_positions()` then sees `loss_pct = 0` → threshold not crossed → force-close never triggers.  
**Financial consequence:** A leveraged HIP-3 position losing 8%+ (the `MAX_LOSS_PER_POSITION_PCT` threshold) is never auto-closed. HIP-3 assets (gold, equities, indices) can gap significantly outside market hours. Full liquidation risk without protection.  
**Fix direction:** When `get_current_price()` returns 0 or fails for a position, treat it as an unknown-PnL state and emit a WARNING + alert rather than silently setting PnL = 0.

---

**LB-v10-3 | isBuy / is_buy key mismatch in `_log_trade_close()`**  
File: `src/main.py` line ~714 vs `src/trading/hyperliquid_api.py` → `get_recent_fills()`  
`_log_trade_close()` checks `_fill.get('isBuy', False)` (camelCase). `get_recent_fills()` normalizes to `is_buy` (snake_case) when the API returns neither key or only `isBuy`. When only `is_buy` is set (normalization path), the `isBuy` lookup returns `False` for ALL fills. For a BUY position, every fill incorrectly evaluates as a SELL fill → wrong fill is matched → P&L computed against wrong execution price → incorrect diary entry.  
**Financial consequence:** Corrupted diary P&L data is fed to Claude's `confirm_trade()` as "RECENT TRADE HISTORY" for the asset. Claude's structured analysis includes historical performance in Factor 1/4 assessment. False P&L data → incorrect APPROVE/REJECT decisions → trades taken or blocked based on fabricated history.  
**Fix direction:** Use `_fill.get('is_buy', _fill.get('isBuy', False))` as a safe dual-lookup.

---

### HIGH

**LB-v10-4 | Partial fill cancel path records wrong `amount` in active_trades**  
File: `src/main.py` lines ~2882–2910  
When inner loop tries to cancel an unfilled limit and the cancel API call fails, the code checks the exchange for an open position. If a position exists (the order partially filled before cancel propagated), the position is recorded as ENTERED. The `amount` stored in `active_trades` comes from the original order's `size` field (the intended qty), not the actual filled quantity from `get_positions()`. The SL is then placed for the wrong size.  
**Financial consequence:** Over-hedged SL (SL too large → excess margin locked) or under-hedged SL (SL too small → partial position unprotected). At 5× leverage, a 10% size discrepancy in SL placement = 50% notional exposure mismatch.

---

**LB-v10-5 | TP1 + TP2 simultaneous fill race condition**  
File: `src/main.py` lines ~1046–1071  
TP1 hit detection checks if `tp1_oid` disappeared from open orders but position still exists. If price gaps through both TP1 and TP2 levels in a single candle (common in volatile crypto), both orders fill simultaneously. The code detects TP1 hit → sets `tp1_hit=True` → halves `amount` → then outer loop tries to place a new TP2 for the "remaining 50%". But the position is already fully closed. The orphaned TP2 limit order is placed on a flat position → may be rejected (no position to sell against) or creates an unintended short entry.  
**Financial consequence:** Unintended new directional exposure immediately after a completed trade cycle. If TP2 is accepted as a new short entry without SL, position is unprotected.

---

**LB-v10-6 | Force-close via `check_losing_positions()` does not enforce cooldown**  
File: `src/main.py` → `check_losing_positions()` call site  
When a position is force-closed because loss exceeds `MAX_LOSS_PER_POSITION_PCT`, the state may transition to IDLE rather than COOLDOWN. SL-triggered closes explicitly set COOLDOWN via `record_exit()`. But force-close via `check_losing_positions()` uses `market_close()` directly without guaranteed state-machine transition to COOLDOWN.  
**Financial consequence:** Immediate re-entry on the same adverse trend that just triggered a force-close. With 5× leverage, a double-loss sequence before cooldown.

---

**LB-v10-7 | Kronos inference timestamp not validated**  
File: `src/indicators/kronos_forecast.py` → `get_kronos_modifier()`  
Kronos receives the last 400 1h candles and produces a direction prediction. The result is applied to the score immediately. No "freshness" metadata is stored with the prediction. If Kronos inference runs in `asyncio.to_thread()` and the thread is queued behind other blocking operations, the candles used for inference could be `INTERVAL` (1h) old by the time the modifier is consumed. There is no staleness check on the Kronos result itself.  
**Financial consequence:** Kronos modifier reflects past price regime, not current. A ±0.5 score delta applied on stale data could push a borderline setup above MIN_AI_SCORE when conditions have already reversed.

---

**LB-v10-8 | `entry_confirmed()` OR-logic volume threshold reduced to 0.7×**  
File: `src/strategy.py` → `entry_confirmed()`  
Volume threshold in `entry_confirmed()` is 0.7× average (was 1.2×). This means a candle with 70% of average volume passes the volume gate. Combined with OR logic (near_ema OR MACD sufficient), this significantly lowers the bar for entry confirmation. Not a bug in isolation, but a design gap: the scoring system awards a volume bonus at 1.5× (conservative), while the entry_confirmed gate accepts 0.7× (liberal). The two thresholds are inconsistent.  
**Financial consequence:** Entry can confirm on weak-volume setups that don't receive the volume score bonus, meaning the actual quality of the setup is lower than the score implies.

---

### MEDIUM

**LB-v10-9 | Score cap boundary: Kronos unavailable + score 10.0 = full sizing on marginal ADX**  
File: `src/main.py` → position sizing logic  
Position size = `min(pct_cap, atr_cap) × (min(score, 10.0) / 10.0)` × ADX modifier. If Kronos is unavailable (modifier=0.0) and score reaches 10.0 via volume + pattern bonuses on a base-8.5 setup, position size is 100% of the ATR/PCT cap. The ADX half-size (15–20 range) is applied afterward as a separate multiplier. Three independent multipliers (score-based, ADX, Kronos) are applied sequentially with no final notional sanity check.

---

**LB-v10-10 | Funding buffer in TP/SL not updated post-entry**  
File: `src/main.py` → `_code_compute_tpsl()`  
`funding_buffer` is baked into TP/SL levels at entry time using the current funding rate snapshot. If funding rate spikes significantly post-entry (common during high-volatility perp markets), the pre-baked buffer underestimates actual funding cost. TP levels may be breached on paper but not in realized P&L after funding drain.

---

### LOW

**LB-v10-11 | Weekend gate not re-checked in inner loop**  
File: `src/main.py` inner loop  
Outer loop runs `market_filter()` which includes the weekend gate (Fri 20:00 UTC → Sun 08:00 UTC). If an outer cycle runs at Fri 19:58 and approves a setup, the inner loop executes entry at 20:02 — inside the weekend gate window. The inner loop does not independently check the weekend gate.

---

## 🔐 AGENT 2 — SECURITY ANALYST

### CRITICAL

**SEC-v10-1 | Live private key in plaintext .env**  
File: `.env` line ~180  
`HYPERLIQUID_PRIVATE_KEY=0xe090ecb7c46cb6dbf33e537e15ef6e8e1e7219946fab9add5c0d1331c8c5d86b`  
Plaintext private key for a live mainnet account. No `.gitignore` confirmed in the file tree. A single accidental `git add .` pushes the key to any remote.  
**Financial consequence:** Total loss of all funds in the Hyperliquid account. Hyperliquid has no withdrawal mechanism to protect — attacker can place market orders to drain positions.

---

**SEC-v10-2 | Anthropic API key in plaintext .env**  
File: `.env` line ~195  
`ANTHROPIC_API_KEY=sk-ant-api03-4MyGHJ...`  
Exposed in plaintext. If leaked: (a) attacker runs Claude API calls at operator's cost, (b) if attacker can inject into any data source that feeds the macro context (headline feed), they can influence APPROVE/REJECT decisions via prompt content.  
**Financial consequence:** API bill inflation; potential indirect trade manipulation if headline sources are compromised.

---

**SEC-v10-3 | No .gitignore — keys and logs exposed on any git commit**  
File: Project root (no `.gitignore` found)  
`.env`, `diary.jsonl`, `decisions.jsonl`, `llm_requests.log`, `prompts.log` — all would be committed on `git add .`. Combined, these expose live keys, full trade history, account values, and complete AI prompt context.  
**Financial consequence:** Total fund loss if pushed to any remote repository.

---

### HIGH

**SEC-v10-4 | Dashboard runs with no auth if `DASHBOARD_TOKEN` is removed without security gate**  
File: `src/main.py` lines ~55–64 + `src/config_loader.py`  
The security gate (`sys.exit(1)`) fires when BOTH `API_HOST=0.0.0.0` AND blank `DASHBOARD_TOKEN` are set. If an operator sets `API_HOST=127.0.0.1` (resolving the blocker) but leaves `DASHBOARD_TOKEN` blank, the security gate does NOT fire (gate requires both conditions). The dashboard then runs on localhost with zero authentication. Any process on the host can read `/live` (account state, positions, equity) and `/diary` (all trade history with prices).  
**Financial consequence:** Complete trading intelligence exposed to any co-hosted process.

---

**SEC-v10-5 | LLM logs contain full trade context (no access control beyond dashboard token)**  
Files: `llm_requests.log`, `prompts.log`  
These files contain entry/TP/SL prices, position direction, score details, macro context, and funding rates — all details of upcoming trades. The dashboard's `/logs` endpoint serves them to any authenticated caller. No additional access control. No expiry or redaction.

---

**SEC-v10-6 | Clock drift not monitored — all orders fail silently if drift >5s**  
Hyperliquid SDK includes timestamp in HMAC signing. If VPS clock drifts >5s (no NTP monitoring confirmed), all order placements fail with timestamp validation errors. `_retry()` catches the exception and backs off, but does not detect "this is a clock drift issue" vs "transient network error." Bot continues the trading loop, believing orders will eventually succeed, while every attempt fails.  
**Financial consequence:** Entry orders fail silently (fine). But SL placement and SL re-placement also fail silently → open positions accumulate without stop-loss protection.

---

### MEDIUM

**SEC-v10-7 | Log files written to relative paths — systemd/Docker CWD mismatch**  
Files: `src/agent/decision_maker.py` lines ~249–261  
`llm_requests.log` and `prompts.log` opened as `open("llm_requests.log", "a")` (relative). `diary.jsonl` got the absolute-path fix (BUG-P9-8); these two did not. When launched via systemd with a different working directory, logs go to the systemd daemon's CWD (often `/`). Operator loses full AI decision audit trail.  
**Note:** This is the 4th consecutive audit noting the log path issue for these two files.

---

**SEC-v10-8 | No rate limiting on dashboard endpoints**  
File: `src/main.py` → dashboard handlers  
`/live`, `/diary`, `/logs` endpoints have Bearer token auth but no rate limiting. An attacker with the dashboard token can poll `/live` continuously to enumerate account value in real time, building a detailed picture of position sizing patterns.

---

## 🌪️ AGENT 3 — CHAOS TESTER

### CRITICAL

**CHAOS-v10-1 | SL trigger order has no slippage cap — flash crash = full margin loss**  
File: `src/trading/hyperliquid_api.py` → `place_trigger_order()`  
SL orders are trigger orders that become market orders at the trigger price. On Hyperliquid perps, during a flash crash (BTC −25% in 60s has occurred), market order execution can be 5–20% below trigger price. With 5× leverage and a 5% SL setting: intended max loss = 25% of margin. Actual execution at 20% slippage = 125% of margin = liquidation.  
**Financial consequence:** Forced position liquidation rather than SL-bounded exit. Entire margin lost.  
**Fix direction:** Add a `slippagePct` parameter to trigger orders, or use a limit trigger (not available on all Hyperliquid trigger types — document the limitation).

---

**CHAOS-v10-2 | Two instances race condition on ungraceful shutdown**  
File: `src/main.py` → instance lock via `socket.bind(('', 47293))`  
If the bot crashes via SIGKILL (OOM killer, systemd force-stop), the socket enters TIME_WAIT for 60s. A systemd `RestartSec=5` restart creates a second instance before the socket releases. Both instances read `state.json` concurrently — both see the same state, both attempt entry → double positions, double daily trade count.  
**Financial consequence:** 2× intended position size, 2× leverage exposure. With 5× leverage, this can exceed the max notional limit.  
**Fix direction:** Add `SO_REUSEADDR` with an active bind test, or use a PID file with stale-PID detection.

---

**CHAOS-v10-3 | Bot restart after TP1 fill missed → guardian places extra TP1 order**  
File: `src/main.py` → guardian (lines ~1118–1270) + `_log_trade_close()` inner reconcile  
If TP1 fills while the bot is offline (bot down for >N minutes, where N = `get_recent_fills()` lookback), the fill is not detected by the inner reconcile on restart (fill too old). `active_trades` still shows `tp1_hit=False`. The guardian sees no TP1 order → attempts to place TP1. On a half-size position (50% closed via TP1), placing the original full TP1 quantity creates an over-sized TP order.  
**Financial consequence:** TP1 placed for original size on a 50%-closed position → the order exceeds available position size → may be rejected, or (worse) accepted by the exchange and create an unintended short.

---

### HIGH

**CHAOS-v10-4 | Price feed API returns stale cached data — candle timestamp still recent**  
File: `src/main.py` → stale candle watchdog  
The stale candle watchdog checks the age of the most recent candle's `timestamp` field. If Hyperliquid's API returns cached stale data during connectivity issues (serving from its own cache), the candle timestamp could still be recent (matching the expected interval age) while the price data itself is stale. The watchdog would not fire. Indicators are computed on stale data → false signals.

---

**CHAOS-v10-5 | SL placement failure path: no alert sent (alerts disabled)**  
File: `src/main.py` → SL placement block (lines ~2265–2298)  
After entry, SL is placed with 2× retry then `market_close()` fallback. If all attempts fail: the position exists, unprotected, and `send_alert()` is called. But `_ENABLED = False` in `alerts.py` → the alert is a no-op. The failure is logged to file only. No human is notified.  
**Financial consequence:** Leveraged position with no stop-loss, no operator notification. Position held open until SL placement succeeds on the next guardian cycle or the position is liquidated.

---

**CHAOS-v10-6 | Inner loop re-entry not blocked after circuit breaker triggers mid-cycle**  
File: `src/main.py` → inner loop circuit breaker check  
Circuit breaker is checked per-asset at the start of each inner tick. But if the circuit breaker triggers on asset A's tick (e.g. a large loss from force-close), assets B, C, D in the SAME inner tick iteration may still proceed through the signal pipeline. The circuit breaker state is read once at the top of each asset's processing block; a mid-cycle trigger may not propagate to same-cycle siblings.

---

**CHAOS-v10-7 | Rate limit hit during SL re-placement delays orphan protection**  
File: `src/trading/hyperliquid_api.py` → `_retry()` exponential backoff  
Backoff delay: base 2s × 2^attempt + jitter. After 3 retries: ~2+4+8 = 14s minimum. During an SL orphan check where the position has no SL, a rate-limit triggered backoff delays SL placement by 14+ seconds. In a volatile market, 14s can represent significant price movement against a leveraged position.

---

**CHAOS-v10-8 | Exchange maintenance 503 → retry storm → potential IP temp-ban**  
File: `src/trading/hyperliquid_api.py` → `_retry()`  
During Hyperliquid maintenance, 503 responses trigger the retry loop (3 retries per call). With 4 assets × 5 timeframe fetches + state fetch + price fetch per asset per outer cycle = potentially 40+ retry chains running simultaneously. Exponential backoff prevents hammering in isolation, but the `_read_semaphore(4)` only caps concurrent reads, not the retry storm per individual call. Risk of reaching Hyperliquid's IP rate limit threshold.

---

**CHAOS-v10-9 | State file corrupted during OS crash — no recovery documentation**  
File: `src/trade_state.py` → `load_state()` + CHAOS-7 fix  
`state.json` corruption → `sys.exit(1)`. This is correct. However: (a) no recovery procedure is documented; (b) with alerts disabled, the operator does not know the bot has stopped; (c) the bot may restart automatically (systemd) and immediately exit again, creating a restart loop with no notification.

---

### MEDIUM

**CHAOS-v10-10 | Weekend gate not enforced in inner loop — entries possible Friday 20:00+**  
Already noted in LB-v10-11. Amplified here: the inner loop runs for 55 minutes (11×5min ticks). An outer cycle that fires at 19:55 Friday spawns an inner loop running until 20:50 — 50 minutes inside the weekend gate.

---

**CHAOS-v10-11 | daily_count.json not written atomically**  
File: `src/main.py` → `_save_daily_count()`  
`daily_count.json` is written with a standard `open(..., 'w')` (not `os.replace()` atomic). An OS crash mid-write produces a corrupt or empty file. On restart, `_load_daily_count()` fails → daily count resets to 0 → bot can execute up to 20 additional trades = up to 2× daily cap. At 5× leverage: doubled daily risk exposure.

---

**CHAOS-v10-12 | Drawdown circuit breaker check race in multi-asset inner loop**  
File: `src/main.py` → inner loop  
Circuit breaker state is read from `risk_state.json` at the start of each inner tick per asset. If assets A and B both execute trades in the same outer cycle that collectively cross the drawdown threshold, the circuit breaker may not halt B's inner-loop processing because B's circuit-breaker read occurred before A's loss was persisted.

---

## ⚡ AGENT 4 — PERFORMANCE ENGINEER

### HIGH

**PERF-v10-1 | Sequential `get_current_price()` calls inside `get_user_state()`**  
File: `src/trading/hyperliquid_api.py` → `get_user_state()`  
For each open position, `get_current_price()` is called sequentially to compute unrealized PnL. With 3 concurrent positions at ~200ms per REST call = ~600ms added to every `get_user_state()` call. `get_user_state()` is called in the inner loop pre-score refresh, meaning all 11 inner ticks are delayed. Account value used in risk calculations is increasingly stale as more assets are processed.

---

**PERF-v10-2 | Inner loop fetches positions/prices individually per asset (no shared pre-fetch)**  
File: `src/main.py` → inner loop  
The outer loop has a single pre-fetch of account state. The inner loop does not. Each asset's inner-tick processing fetches `get_positions()`, `get_current_price()`, and account state independently. With 4 assets × 11 ticks = 44 rounds of duplicate API calls per hour. At 5 assets, this is 55 rounds. Primary risk: rate limit exhaustion before a critical SL placement call can be made.

---

**PERF-v10-3 | `asyncio.to_thread()` pool saturation — Kronos competes with Claude and exchange calls**  
File: `src/main.py` → multiple `asyncio.to_thread()` callsites  
Default asyncio thread pool: `min(32, os.cpu_count() + 4)`. With 4 assets potentially each calling Claude (timeout 30s), Kronos (CPU-bound, 500ms–2s), and synchronous SDK calls simultaneously, the pool approaches saturation. A saturated pool delays ANY `to_thread()` call, including critical SL placement via the synchronous SDK. No custom `ThreadPoolExecutor` with bounded/prioritized queues is configured.

---

### MEDIUM

**PERF-v10-4 | `confirm_trade()` reads full `diary.jsonl` on every Claude call**  
File: `src/agent/decision_maker.py` → `confirm_trade()` lines ~46–61  
`diary.jsonl` is opened, fully parsed, and filtered on every `confirm_trade()` call. The diary grows indefinitely (one entry per trade side). After months of operation, this is an O(n) scan on every Claude gate call. The `diary_index.json` (O(1) lookup) exists for the guardian but is not used here.

---

**PERF-v10-5 | No candle cache — 5 full candle fetches per asset per outer cycle**  
File: `src/main.py` → outer loop  
Each asset fetches 5 timeframes (4h, 1h, 30m, 15m, 5m) per outer cycle. With 4 assets = 20 candle fetch calls. No caching or conditional fetch (ETag, If-Modified-Since). Within-candle resampling would render most same-timeframe re-fetches redundant. This is 20 of the outer cycle's most expensive API calls.

---

**PERF-v10-6 | Concurrent `confirm_trade()` writes to shared log files without locking**  
File: `src/agent/decision_maker.py` → file writes at lines ~249–261  
`open("llm_requests.log", "a")` and `open("prompts.log", "a")` are opened without file locking. In a multi-asset scenario where two assets simultaneously trigger the Claude gate (`asyncio.to_thread()` dispatches them concurrently), writes interleave → garbled log entries. Reproduces on any scenario where two assets are in confluence simultaneously.

---

**PERF-v10-7 | Kronos inference: no per-call timeout (only pre-warm has a timeout)**  
File: `src/indicators/kronos_forecast.py` → `get_kronos_modifier()`  
`asyncio.wait_for(timeout=120)` guards only the pre-warm. Individual `get_kronos_modifier()` calls have no timeout. A degraded GPU/CPU (thermal throttle, memory pressure) could make inference run indefinitely, blocking that asset's scoring pipeline. The outer loop has no per-asset timeout guard.

---

**PERF-v10-8 | Log rotation checked only at startup — log files can grow unbounded at runtime**  
File: `src/main.py` → `_rotate_if_needed()` called in startup sequence  
The 50MB rotation check runs once on startup. If the bot runs continuously for weeks, `diary.jsonl`, `decisions.jsonl`, `llm_requests.log`, and `prompts.log` grow without bound at runtime. On a VPS with a small disk (common for low-cost hosting), disk-full causes `open()` calls to fail → diary and decisions not written → guardian has no data → incorrect trade history for Claude.

---

**PERF-v10-9 | Sharpe ratio computed by full diary scan**  
File: `src/main.py` → `calculate_sharpe_from_diary()`  
Full scan of `diary.jsonl` for every Sharpe computation. Not a real-time issue at current scale, but will degrade linearly with diary growth.

---

### LOW

**PERF-v10-10 | Memory: no bound on in-memory candle accumulation**  
Each outer cycle loads all candles into fresh lists. With 4 assets × 5 timeframes × 500 candles per fetch, Python allocates ~10,000 objects per outer cycle. GC handles collection but there is no explicit bound on in-flight object count if multiple assets are processed in overlapping `asyncio.gather()` calls.

---

## 👤 AGENT 5 — OPERATOR SAFETY

### CRITICAL

**OPS-v10-1 | ALL ALERTS DEAD — 4th consecutive audit**  
File: `src/alerts.py`, `.env` lines ~197–198  
`TELEGRAM_BOT_TOKEN=` and `TELEGRAM_CHAT_ID=` are blank. `_ENABLED = False`. Every `send_alert()` is a no-op. This affects: SL placement failure, force-close trigger, circuit breaker activation, Kronos degradation warning, KILLSWITCH activation, daily P&L summary, startup confirmation, and unexpected exception alerts. A leveraged bot on mainnet with zero operator alerting is categorically unsafe.  
**This is the single highest-priority operational gap in the entire codebase.**

---

**OPS-v10-2 | Bot cannot start — startup blocker in .env (4th consecutive audit)**  
File: `.env` lines ~184, ~189  
`API_HOST=0.0.0.0` + `DASHBOARD_TOKEN=` blank → `sys.exit(1)`. The bot has been in this state across 4 audits. Either (a) the operator is running an older version without the security gate (meaning the gate has been bypassed — a security gap), or (b) the bot is genuinely not running and this is pre-live review. In either case, the .env must be corrected before any live deployment.

---

**OPS-v10-3 | Kill switch is file-based — no remote trigger**  
File: `src/main.py` → KILLSWITCH detection  
KILLSWITCH requires creating a file named `KILLSWITCH` in the bot's working directory. No Telegram command, no HTTP endpoint, no process signal handler. If the VPS is inaccessible (SSH down, network partition — common during market crises when VPS providers are also under load), there is no way to halt the bot remotely.  
**Financial consequence:** Cannot stop a runaway bot during a flash crash or exchange anomaly without SSH access.

---

**OPS-v10-4 | requirements.txt missing Kronos dependencies — 3rd+ consecutive audit**  
File: `requirements.txt`  
`torch`, `transformers`, and `chronos-forecasting` absent. Fresh deployment via `pip install -r requirements.txt` → Kronos import fails → `_kronos_failed = True` → all scores computed without the ±0.5 Kronos modifier → net effect: signals that would have been suppressed by -0.5 (disagreement) pass the MIN_AI_SCORE threshold. SILENT degradation (alert is a no-op). This has been noted in 3+ consecutive audit versions.

---

### HIGH

**OPS-v10-5 | Fills older than `get_recent_fills()` lookback not detected on restart**  
File: `src/main.py` → inner reconcile  
If the bot is down for longer than the fills lookback window (typically 100 fills or N minutes of fill history), trade closes that occurred during downtime are not detected. `diary.jsonl` has no close event → Sharpe computation misses the trade → Claude's recent history for the asset is incomplete → Claude may APPROVE a follow-up trade based on incomplete performance context.

---

**OPS-v10-6 | State file corruption = hard stop with no automated recovery documentation**  
File: `src/trade_state.py`  
CHAOS-7 fix correctly exits on state corruption. But: (a) no recovery procedure exists in the codebase or docs; (b) with alerts dead, operator does not know the bot stopped; (c) systemd auto-restart loops immediately exit again; (d) open positions exist without a bot managing them.

---

**OPS-v10-7 | No cross-session cumulative drawdown monitoring**  
File: `src/risk_manager.py` → circuit breaker  
Daily circuit breaker resets at UTC midnight. If the bot loses 11% daily for 5 consecutive days (each just under the 12% threshold), the 55% cumulative drawdown is never monitored. No weekly or rolling drawdown metric exists.

---

**OPS-v10-8 | `daily_count.json` not written atomically — non-atomic write on crash = double daily cap**  
File: `src/main.py` → `_save_daily_count()`  
Standard `open()` write (not `os.replace()` atomic). OS crash mid-write → corrupt/empty file → daily count resets to 0 on restart → up to 2× daily trade cap in a single day. Full details in CHAOS-v10-11.

---

### MEDIUM

**OPS-v10-9 | No testnet mode active — every code change deploys to mainnet**  
File: `.env`  
`HYPERLIQUID_NETWORK=mainnet`. No commented testnet equivalent. No documented procedure for switching. Any bug fix, config change, or refactor goes live immediately. Combined with no pre-deployment checklist, this is a systemic ops gap.

---

**OPS-v10-10 | Dashboard `/logs` whitelist excludes `bot.log`**  
File: `src/main.py` → `_ALLOWED_LOG_FILES`  
`bot.log` is not in the whitelist. Operator cannot tail the main application log via the dashboard. Requires SSH access to view live bot output.

---

**OPS-v10-11 | No documented emergency shutdown procedure**  
No documentation exists for: how to manually close all positions if the bot is unresponsive, what to do if state.json is corrupted, how to verify all positions are flat after KILLSWITCH, or how to ramp up capital from testnet to live.

---

**OPS-v10-12 | `initial_balance.json` never resets downward — Sharpe baseline inflated after losses**  
File: `src/main.py` → `_load_initial_balance()`  
Initial balance is loaded once and never updated downward. If the account loses 30%, the Sharpe ratio is computed against the original (higher) baseline. This overstates drawdown in the denominator and understates returns in the numerator. The metric becomes misleading for performance evaluation.

---

## 🔁 STEP 3 — FULL FLOW AUDIT

| Flow | Expected | Actual Risk | Severity |
|------|----------|-------------|----------|
| Signal generation | Data feed → indicator → lookback check → signal | Kronos modifier applied without inference freshness check (LB-v10-7) | HIGH |
| Pre-trade risk check | Signal → position count, drawdown, size limits, cooldown | Force-close bypasses cooldown state (LB-v10-6) | HIGH |
| Position sizing | Risk params → leverage → notional → lot size → qty | Three multipliers (score, ADX, Kronos) applied sequentially, no final notional check (LB-v10-9) | MEDIUM |
| Entry order | Sized qty → submit → capture order ID | weekend gate not re-checked in inner loop (LB-v10-11 / CHAOS-v10-10) | MEDIUM |
| Entry confirmation | Poll/stream fill → verify qty/price → update state | Partial fill on cancel-fail path records wrong amount (LB-v10-4) | HIGH |
| SL placement | Entry confirmed → SL price → submit SL → confirm placed | Clock drift silently fails all SL attempts (SEC-v10-6) | HIGH |
| Position monitoring | Feed → check mark price vs SL/TP → check funding | Price feed cached stale data bypasses candle watchdog (CHAOS-v10-4) | HIGH |
| Exit signal | Exit/SL trigger → cancel entry orders → submit exit | SL trigger = market order, no slippage cap (CHAOS-v10-1) | CRITICAL |
| Exit confirmation | Poll/stream exit fill → verify → clear state | isBuy/is_buy mismatch corrupts fill matching (LB-v10-3) | HIGH |
| WebSocket reconnect | N/A (REST polling) | Stale cached API data not caught by candle watchdog | HIGH |
| Drawdown check | Periodic balance read → compare → halt if exceeded | Race condition in multi-asset inner loop (CHAOS-v10-12) | MEDIUM |
| Kill switch | Trigger → cancel all → close all → halt | File-based only, no remote trigger (OPS-v10-3) | HIGH |
| Restart recovery | Startup → query exchange → reconcile → resume | TP1 missed fill during downtime → double TP1 order (CHAOS-v10-3) | CRITICAL |
| PnL calc | Fill price × qty − fees − funding | isBuy/is_buy mismatch → wrong fill matched (LB-v10-3) | HIGH |
| Alert dispatch | Error/threshold → format → send | ALL ALERTS DEAD (OPS-v10-1) | CRITICAL |
| HIP-3 force-close | Loss% > MAX_LOSS_PCT → market close | get_current_price()=0 → loss_pct=0 → never triggers (LB-v10-2) | CRITICAL |
| Guardian TP2 | Re-place missing TP2 → confirm placed | TP2 never re-placed if TP1 exists (LB-v10-1) | CRITICAL |

---

## 📊 STEP 4 — EDGE CASE TABLE

| Edge Case | Expected | Actual Risk | Finding | Severity |
|-----------|----------|-------------|---------|----------|
| TP1 fills while bot is down | Detect on restart, resume with tp1_hit=True | Missed fill → guardian places double TP1 on half-position | CHAOS-v10-3 | CRITICAL |
| TP1 placed, TP2 rejected/dropped | Guardian re-places TP2 | _g_has_tp1=True → re-place block never fires | LB-v10-1 | CRITICAL |
| HIP-3 asset loses 8%+ | force-close triggers | get_current_price()=0 → PnL=0 → no close | LB-v10-2 | CRITICAL |
| Flash crash -25% | SL executes, bounded loss | SL = market trigger, 15–20% slippage = liquidation | CHAOS-v10-1 | CRITICAL |
| All alerts disabled | Critical events notified | All alerts no-op (_ENABLED=False) | OPS-v10-1 | CRITICAL |
| Bot can't start | Bot starts after .env fix | sys.exit(1) on startup (4th audit) | OPS-v10-2 | CRITICAL |
| Two instances from systemd restart | Second blocked by port lock | TIME_WAIT window allows race | CHAOS-v10-2 | CRITICAL |
| TP1 + TP2 both fill in one candle | Full close, clean state | TP1 detected → tries to place TP2 on closed position | LB-v10-5 | HIGH |
| Partial fill on cancel-fail | Record actual filled qty | Original order size recorded as amount | LB-v10-4 | HIGH |
| isBuy/is_buy fill key mismatch | Correct fill matched for PnL | Wrong fill matched → corrupted diary | LB-v10-3 | HIGH |
| Kronos not installed (fresh pip) | Graceful degradation, logged | Silent: alerts dead, -0.5 signals pass gate | OPS-v10-4 | HIGH |
| Clock drift >5s | NTP or alert+halt | All order attempts fail silently | SEC-v10-6 | HIGH |
| daily_count.json corrupted | Daily cap preserved | Reset to 0 → 2× daily trade count | CHAOS-v10-11 | HIGH |
| Fills older than lookback window | Detected on restart | Not detected → incomplete diary → Claude wrong history | OPS-v10-5 | HIGH |
| state.json corrupted | Detected + sys.exit | Correct halt, but no alert, no recovery docs | OPS-v10-6 | HIGH |
| Kill switch needed, SSH down | Remote trigger halts bot | No remote trigger exists | OPS-v10-3 | HIGH |
| Rate limit during SL re-placement | Backoff + retry | 14+ second SL gap with no protection | CHAOS-v10-7 | HIGH |
| Exchange maintenance 503 | Graceful pause | Retry storm → potential IP ban | CHAOS-v10-8 | HIGH |
| Inner loop circuit breaker race | Halt on breach | Same-tick siblings may trade past breach | CHAOS-v10-6 | MEDIUM |
| Weekend gate violated in inner loop | Gate enforced | Inner loop runs 50min past Fri 20:00 | CHAOS-v10-10 | MEDIUM |
| Kronos inference hangs | Timeout + fallback | No per-call timeout → infinite block | PERF-v10-7 | MEDIUM |
| Disk full (no runtime rotation) | Logs truncated gracefully | write() fails → no diary → guardian blind | PERF-v10-8 | MEDIUM |
| Cross-session 55% cumulative loss | Monitored + halt | No cross-session drawdown metric | OPS-v10-7 | MEDIUM |
| Force-close without cooldown | COOLDOWN state set | May return IDLE → immediate re-entry | LB-v10-6 | HIGH |

---

## ✅ STEP 5 — PRE-LIVE CHECKLIST

### Risk Controls
- [x] Hard max position size enforced before every order (ATR + PCT cap)
- [x] SL always placed and confirmed after entry (2× retry + market fallback)
- [x] Intraday drawdown limit configured and auto-reset at UTC midnight
- [x] Max concurrent positions, max orders/min, cooldown configured
- [ ] **MISSING: Remote kill switch** — file-based only (OPS-v10-3)
- [ ] **MISSING: Cross-session cumulative drawdown monitoring** (OPS-v10-7)
- [ ] **MISSING: SL slippage cap** — trigger orders become uncapped market orders (CHAOS-v10-1)
- [ ] **MISSING: HIP-3 force-close protection** — PnL=0 on price fetch failure (LB-v10-2)
- [ ] **MISSING: Guardian TP2 independent re-placement** (LB-v10-1)

### Kill Switch
- [x] Exists (file-based KILLSWITCH)
- [ ] **NOT tested on testnet** (no testnet run documented)
- [ ] **No remote trigger** — requires SSH (OPS-v10-3)
- [x] Halts and stays halted (3× retry + outer loop check)

### API & Credentials
- [ ] **CRITICAL: Private key in plaintext .env** — not in secure vault (SEC-v10-1)
- [ ] **CRITICAL: No .gitignore** — keys will be committed on any git add (SEC-v10-3)
- [x] Withdrawal permissions not enabled (Hyperliquid doesn't support this)
- [ ] **Missing: Clock drift monitoring** (SEC-v10-6)
- [ ] **Missing: Testnet/live separation** — no testnet config (OPS-v10-9)

### Testnet Validation
- [ ] **NOT DONE** — No evidence of testnet run in codebase
- [ ] **No ramp-up plan documented** (OPS-v10-11)

### Monitoring
- [ ] **CRITICAL: Alerting NOT configured** — both Telegram fields blank (OPS-v10-1)
- [ ] **CRITICAL: No alerts for SL failure, force-close, drawdown, connectivity loss**
- [ ] **No daily PnL summary delivery** (alerts dead)

### Backtest
- [ ] No backtest framework present in codebase (purely live trading)

### Integrity
- [ ] **daily_count.json not written atomically** (CHAOS-v10-11)
- [ ] **log files written to relative paths** (SEC-v10-7)
- [ ] **requirements.txt missing Kronos deps** (OPS-v10-4)

---

## 🔄 STEP 6 — PEER REVIEW

### 🔨 LOGIC BREAKER reviews other agents

**Most critical finding from another agent:** OPS-v10-1 (ALL ALERTS DEAD). Every risk control I found — force-close failure, SL placement failure, fill mismatch — depends on the operator being notified. With alerts dead, ALL of my HIGH findings become undetected failures.

**Biggest risk ALL agents missed:** The interaction between LB-v10-1 (TP2 orphan) and CHAOS-v10-1 (SL slippage) creates a compound scenario: TP1 fills, bot believes setup is safe, TP2 never placed (orphan), then a reversal happens, and the SL executes at far worse price than set. The tail loss on this combined path is 100%+ of margin.

**Issue that would stop live trading:** LB-v10-3 (isBuy/is_buy mismatch). Silent P&L corruption in the diary means Claude's trade history context is actively wrong for every asset. The AI gate is making decisions on fabricated data. This undermines MASTER RULE 3 fundamentally.

---

### 🔐 SECURITY ANALYST reviews other agents

**Most critical finding from another agent:** CHAOS-v10-2 (two instances race). A second instance starting with the same live private key and same state could double every position. Combined with SEC-v10-1 (key exposed), if the .env is on a VPS with weak SSH, an attacker who gains access can start their own instance.

**Biggest risk ALL agents missed:** The `prompts.log` file grows without bound and contains full entry/TP/SL context. If an attacker gains read access to the filesystem (not even the process), they can reconstruct the entire trading strategy, current positions, and future likely entries. This is intelligence exposure, not just data exposure.

**Issue that would stop live trading:** SEC-v10-3 (no .gitignore). One accidental commit destroys everything. This should be the first file created in any trading bot repository.

---

### 🌪️ CHAOS TESTER reviews other agents

**Most critical finding from another agent:** PERF-v10-3 (thread pool saturation). I tested what happens when 4 assets are all in confluence simultaneously — 4 Claude calls (30s timeout each) + 4 Kronos inferences + exchange SDK calls all in `asyncio.to_thread()`. The thread pool saturates. SL placement (also in `to_thread()`) queues behind the Claude calls. During a rapid adverse move, the 30s Claude timeout is the longest-living thread and blocks every other operation on that pool.

**Biggest risk ALL agents missed:** The combination of CHAOS-v10-1 (SL slippage) + LB-v10-2 (HIP-3 force-close never triggers) + OPS-v10-1 (alerts dead) means a HIP-3 position can run to liquidation with no SL slippage protection, no force-close protection, and no operator notification.

**Issue that would stop live trading:** CHAOS-v10-1 (SL trigger = uncapped market order). On a leveraged perps DEX, this is not theoretical — Hyperliquid has experienced flash crashes. A 20% SL slippage at 5× leverage = 100% margin loss. This alone should prevent live trading.

---

### ⚡ PERFORMANCE ENGINEER reviews other agents

**Most critical finding from another agent:** LB-v10-1 (TP2 orphan). From a performance perspective: the guardian runs every outer cycle checking for missing TPs. When TP2 is permanently orphaned (TP1 exists, TP2 missing), the guardian performs the same no-op check every cycle forever, indefinitely. This is not just a financial bug — it's a logic leak that silently wastes a compute cycle while the financial exposure accumulates.

**Biggest risk ALL agents missed:** The inner loop's lack of a shared position/price pre-fetch (PERF-v10-2) combined with 4 assets means the first asset in the loop uses fresh data, the fourth asset uses data that is 3 × (API call latency) stale. In volatile markets, this creates signal divergence within a single inner tick — signals computed on materially different price snapshots.

**Issue that would stop live trading:** PERF-v10-3 (thread pool saturation with concurrent Claude calls). A 30-second Claude timeout blocking the thread pool during a flash crash would prevent SL placement for 30 seconds — the exact window where fastest exit is critical.

---

### 👤 OPERATOR SAFETY reviews other agents

**Most critical finding from another agent:** LB-v10-2 (HIP-3 force-close never triggers). I can add every risk control in the framework, but if a HIP-3 asset (gold, equities) silently bypasses the force-close guard, all force-close protection is theater. The fix is one line but the consequence is liquidation.

**Biggest risk ALL agents missed:** The startup blocker (OPS-v10-2) has been present for 4 consecutive audit passes. If the operator resolved it by commenting out the security gate in `main.py` rather than fixing the .env, the bot is currently running with `API_HOST=0.0.0.0` + NO AUTHENTICATION on the dashboard. This would be a live security vulnerability, not just a pre-launch gap.

**Issue that would stop live trading:** OPS-v10-1 (ALL ALERTS DEAD). Every other risk control in this bot is reactive — it needs a human to respond. With zero alerting, every failure that requires human intervention is permanently unnoticed. This alone disqualifies the bot from live trading.

---

## 🏛️ STEP 7 — FINAL REPORT

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
     🏛️  TRADING BOT BUG COUNCIL — FINAL REPORT v10
     Hyperliquid Perpetuals — CODE-FIRST Hybrid — 5× Leverage
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 BOT SUMMARY
   Strategy:  Code-first hybrid (Claude = APPROVE/REJECT gate only)
   Exchange:  Hyperliquid DEX (perpetual futures, mainnet)
   Leverage:  Up to 5× (amplifies every risk below by 5×)
   Assets:    BTC, ETH, SOL, AVAX + HIP-3 (xyz:GOLD etc.)
   Position:  Partial-close (TP1/TP2), trailing stop guardian
   Risk:      Intraday drawdown breaker, ATR sizing, PCT cap,
              daily trade cap, cooldown, max concurrent positions
   MISSING:   Remote kill switch, cross-session drawdown, slippage
              cap on SL execution, HIP-3 PnL guard
   Testnet:   ABSENT from current config
   DeFi:      Not applicable

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💀 FUND-LOSS RISKS — Fix before ANY live trading (6 issues)

1. OPS-v10-1  ALL ALERTS DEAD — 4th consecutive audit
   Telegram unconfigured → _ENABLED=False → every runtime event is
   silent. SL failures, force-close triggers, drawdown breaches,
   KILLSWITCH events — zero operator notification.
   [.env lines 197-198 | src/alerts.py]

2. OPS-v10-2  BOT CANNOT START — 4th consecutive audit
   API_HOST=0.0.0.0 + blank DASHBOARD_TOKEN → sys.exit(1).
   Either fix .env or the security gate has been bypassed.
   [.env lines 184, 189 | src/main.py lines 55-64]

3. LB-v10-1   GUARDIAN TP2 PERMANENTLY ORPHANED
   _g_has_tp1=True when TP1 exists → TP2 re-placement never fires
   if TP2 was never placed. 50% of position has no take-profit ever.
   [src/main.py lines 1118-1270 | _g_has_tp1 variable]

4. LB-v10-2   HIP-3 FORCE-CLOSE NEVER TRIGGERS
   get_current_price()=0 for failed HIP-3 lookups → PnL=0 →
   loss_pct=0 → force-close check always skips. Full liquidation
   risk on gold, equities, index perpetuals.
   [src/trading/hyperliquid_api.py → get_user_state()]

5. LB-v10-3   isBuy/is_buy FILL KEY MISMATCH — SILENT PnL CORRUPTION
   _log_trade_close() uses 'isBuy' (camelCase); get_recent_fills()
   normalizes to 'is_buy' (snake_case). Wrong fills matched for P&L.
   Claude's trade history context actively corrupted.
   [src/main.py line ~714 | hyperliquid_api.py get_recent_fills()]

6. CHAOS-v10-1 SL TRIGGER = UNCAPPED MARKET ORDER
   SL trigger orders become market orders at trigger price. On
   Hyperliquid during flash crashes: 15-20% slippage at 5× leverage
   = full margin liquidation. No slippage cap exists.
   [src/trading/hyperliquid_api.py → place_trigger_order()]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔴 CRITICAL — Fix before live trading

7. SEC-v10-1  LIVE PRIVATE KEY IN PLAINTEXT .env
   [.env line ~180]

8. SEC-v10-3  NO .gitignore — ONE COMMIT LOSES EVERYTHING
   [project root]

9. CHAOS-v10-2 TWO INSTANCES ON SYSTEMD RESTART (port TIME_WAIT race)
   [src/main.py → socket.bind]

10. CHAOS-v10-3 MISSED TP1 FILL ON RESTART → DOUBLE TP1 ORDER
    [src/main.py → guardian + _log_trade_close()]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🟠 HIGH — Fix before real money exposure (15 issues)

11.  LB-v10-4   Partial fill cancel path records wrong amount → wrong SL size
12.  LB-v10-5   TP1+TP2 simultaneous fill → orphaned TP2 order on closed position
13.  LB-v10-6   force-close bypasses COOLDOWN → immediate re-entry after large loss
14.  LB-v10-7   Kronos modifier applied without inference freshness check
15.  LB-v10-8   entry_confirmed() volume threshold 0.7× inconsistent with 1.5× bonus
16.  SEC-v10-4  Dashboard runs with no auth if API_HOST=127.0.0.1 + blank token
17.  SEC-v10-6  Clock drift >5s silently fails all order placements
18.  CHAOS-v10-4 Stale cached API data bypasses candle watchdog
19.  CHAOS-v10-5 SL placement failure: no alert, no operator notification
20.  CHAOS-v10-7 Rate limit backoff delays SL re-placement 14+ seconds
21.  CHAOS-v10-8 Exchange 503 retry storm → potential IP temp-ban
22.  CHAOS-v10-9 state.json corruption halt: no alert, no recovery docs
23.  PERF-v10-1  Sequential get_current_price() in get_user_state() → stale account value
24.  PERF-v10-2  Inner loop fetches per-asset with no shared pre-fetch
25.  PERF-v10-3  asyncio.to_thread() pool saturation — Claude blocks SL placement
26.  OPS-v10-4   requirements.txt missing torch/transformers/chronos-forecasting (3rd audit)
27.  OPS-v10-5   Fills older than lookback window: not detected on restart
28.  OPS-v10-8   daily_count.json not written atomically → 2× daily cap on crash

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🟡 EDGE CASE — Fix before scale (12 issues)

29.  LB-v10-9   Three position-size multipliers applied sequentially, no final check
30.  LB-v10-10  Funding buffer in TP/SL not updated post-entry
31.  LB-v10-11  Weekend gate not re-checked in inner loop
32.  SEC-v10-5  LLM logs contain full trade context — sensitive data in plaintext files
33.  SEC-v10-7  llm_requests.log + prompts.log written to relative paths (4th audit)
34.  SEC-v10-8  No rate limiting on dashboard endpoints
35.  CHAOS-v10-6 Circuit breaker race: same-tick siblings may trade past breach
36.  CHAOS-v10-10 Weekend gate violated: inner loop runs 50 min past Friday 20:00
37.  CHAOS-v10-11 daily_count.json non-atomic write (detailed in OPS-v10-8)
38.  CHAOS-v10-12 Drawdown circuit breaker race in multi-asset inner loop
39.  PERF-v10-4  confirm_trade() reads full diary.jsonl on every Claude call (O(n))
40.  PERF-v10-5  No candle cache: 20 full candle fetches per outer cycle

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚡ PERFORMANCE

41.  PERF-v10-6  Concurrent confirm_trade() writes to shared logs without locking
42.  PERF-v10-7  Kronos inference: no per-call timeout (only pre-warm has one)
43.  PERF-v10-8  Log rotation checked only at startup — logs grow unbounded at runtime
44.  PERF-v10-9  Sharpe ratio computed by full diary scan
45.  PERF-v10-10 No bound on in-memory candle accumulation across concurrent tasks

🛡️ MISSING RISK CONTROLS

- No remote kill switch (OPS-v10-3)
- No cross-session / cross-day cumulative drawdown monitor (OPS-v10-7)
- No SL slippage cap (CHAOS-v10-1)
- No HIP-3 price-fetch-failure guard in force-close path (LB-v10-2)
- No clock drift detection (SEC-v10-6)
- Guardian does not independently track TP1 vs TP2 presence (LB-v10-1)

📊 MONITORING & OPERATIONAL GAPS

- ALL Telegram alerts disabled (OPS-v10-1) — 4th consecutive audit
- Bot cannot start in current .env state (OPS-v10-2)
- No testnet mode configured (OPS-v10-9)
- No emergency shutdown procedure documented (OPS-v10-11)
- daily_count.json and diary-related files written non-atomically
- Log rotation not run at runtime — disk-full risk on long runs (PERF-v10-8)
- bot.log excluded from dashboard /logs whitelist (OPS-v10-10)

📦 TECHNICAL DEBT (carried from prior audits, still unresolved)

- requirements.txt missing Kronos deps (OPS-v10-4) — 3rd consecutive audit
- llm_requests.log / prompts.log relative paths (SEC-v10-7) — 4th consecutive audit
- Telegram unconfigured (OPS-v10-1) — 4th consecutive audit
- API_HOST + DASHBOARD_TOKEN startup blocker in .env (OPS-v10-2) — 4th consecutive audit

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

COUNCIL VERDICT:  ⛔ NOT READY FOR LIVE TRADING

Primary concern: 6 fund-loss risks are present simultaneously on a live
mainnet account. The most severe combination: HIP-3 force-close never
triggers (LB-v10-2) + SL trigger has no slippage cap (CHAOS-v10-1) +
alerts are entirely dead (OPS-v10-1) = a leveraged HIP-3 position can
run to liquidation with zero protection and zero notification.

TOP 3 MUST-FIX:
  1. OPS-v10-1  Configure Telegram alerts — without this, every other
                risk control is unmonitored and unresponsive.
  2. LB-v10-3  Fix isBuy/is_buy fill key lookup — active P&L corruption
                undermines every Claude APPROVE/REJECT decision made since
                this code was deployed.
  3. LB-v10-2  Guard get_current_price()=0 in force-close path — HIP-3
                assets have ZERO force-close protection in current code.

CAPITAL RECOMMENDATION:
  $0 (zero) until at minimum:
    - Alerts configured and tested (OPS-v10-1)
    - Startup blocker resolved (OPS-v10-2)
    - isBuy/is_buy mismatch fixed (LB-v10-3)
    - HIP-3 force-close guard added (LB-v10-2)
    - Guardian TP2 independent re-placement (LB-v10-1)
    - .gitignore created (SEC-v10-3)
  After above 6 fixes: paper-trade / testnet only, ramp gradually.

GO LIVE:  ⛔ NOT READY

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
V10 FIXES CONFIRMED FROM v9:
  ✅ BUG-v9-P1  Kronos pre-warm moved into main_async() with 120s timeout
  ✅ BUG-v9-L1  Trailing stop zero-size: exchange fallback + amount repair
  ✅ BUG-v9-L2  Stale candle watchdog: 5m/1h now halts (not just warns)
  ✅ BUG-v9-L3  Unfilled limit cancel-fail: checks exchange before deciding
  ✅ BUG-v9-L4  Outer loop ADX=None bypass: confirmed present in outer loop
                ❌ NOT mirrored to inner loop (BUG-v10-INNER-MACRO, finding
                    not in top issues — see CHAOS-v10-6 area for context;
                    inner loop uses _iadx_1d is not None check only)

NEW v10 FINDINGS NOT IN v9:
  LB-v10-1  Guardian TP2 orphan (TP1 present but TP2 missing case)
  LB-v10-2  HIP-3 force-close never triggers (get_current_price()=0)
  LB-v10-3  isBuy/is_buy fill mismatch (diary PnL corruption)
  LB-v10-4  Partial fill cancel-fail records wrong amount
  LB-v10-5  TP1+TP2 simultaneous fill race
  CHAOS-v10-6  Circuit breaker race in multi-asset inner tick
  PERF-v10-3  Thread pool saturation: Claude blocks SL placement
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

*Report generated: 2026-05-28 | Audit version: v10 | 5-agent review | 45 findings total*  
*Previous reports: v1–v9 in `hyperliquid-trading-agent-master/reports/`*
