
# 🏛️ TRADING BOT BUG COUNCIL — v9 RE-AUDIT REPORT

**Date:** 2026-05-28
**Pass:** v9 (re-audit — post-v8 fixes applied; verifying all fixes landed and hunting new/remaining gaps)
**Files read:** main.py (2721+ lines, 4 chunks), strategy.py, risk_manager.py, trade_state.py, hyperliquid_api.py, decision_maker.py, config_loader.py, alerts.py, kronos_forecast.py, .env, .gitignore, requirements.txt
**Leverage:** 5× — all position risks multiplied accordingly
**Exchange:** Hyperliquid perpetuals (mainnet, REAL FUNDS)
**Order model:** LIMIT entry (0.15% better than market); code-first hybrid with Claude as APPROVE/REJECT gate
**DeFi Agent:** Not activated (no on-chain / no web3 calls)

---

## ✅ V8 FIXES — VERIFICATION STATUS

All prior-cycle bugs individually verified against the codebase. Results:

| Bug ID | Description | Status |
|--------|-------------|--------|
| BUG-v8-L1 | Inner loop KILLSWITCH 3× retry — mirrors outer loop fix | ✅ CONFIRMED FIXED |
| BUG-v8-L2 | Kronos candle count raised from 100→400 (outer + inner) | ✅ CONFIRMED FIXED |
| BUG-v8-L3 | Volume bonus uses last CLOSED candle `_c5m[-2]` | ✅ CONFIRMED FIXED |
| BUG-v8-L4 | `calculate_sharpe_from_diary()` now defined (line 3186) | ✅ CONFIRMED FIXED |
| BUG-v8-L5 | BTC correlation filter fetches BTC 1h when BTC not in `--assets` | ✅ CONFIRMED FIXED |
| BUG-v8-L6 | TP/SL sized to `filled_qty`, not full intended amount | ✅ CONFIRMED FIXED |
| BUG-v8-L7 | Trailing stop zero-size guard added (logs + skips) | ✅ PARTIAL — no fallback protection |
| BUG-v8-S4 | `send_alert_sync` replaced with synchronous urllib call | ✅ CONFIRMED FIXED |
| BUG-v8-S6 | `.gitignore` confirmed to include `.env` | ✅ CONFIRMED FIXED |
| CHAOS-3 | Stale candle watchdog — warns if last candle >3× interval old | ✅ CONFIRMED FIXED |
| CHAOS-4/OPS-4 | Instance lock via socket bind (port 47293) | ✅ CONFIRMED FIXED |
| CHAOS-7 | state.json corrupt JSON → sys.exit(1) not silent IDLE reset | ✅ CONFIRMED FIXED |
| CHAOS-8 | Funding rate buffer now baked into `_code_compute_tpsl()` | ✅ CONFIRMED FIXED |
| OPS-5 | `max_concurrent_positions` fallback aligned: both `or 3` now | ✅ CONFIRMED FIXED |
| PERF-1 | Kronos pre-warm scheduled at startup | ✅ PARTIAL — implementation bug (BUG-v9-P1) |
| PERF-2 | Account balance refreshed BEFORE scoring in inner loop | ✅ CONFIRMED FIXED |
| PERF-4/PERF-5 | Log rotation: RotatingFileHandler + `_rotate_if_needed` for JSONL | ✅ CONFIRMED FIXED |

---

## 📋 BOT SUMMARY

- **Type:** Code-first hybrid perpetual futures scalper/swing with Claude as structured APPROVE/REJECT gate
- **Asset class:** Crypto perps (Hyperliquid, 229+ markets)
- **Leverage:** 5× (explicit — all position sizing consequences multiply by 5)
- **Order types:** LIMIT entry (0.15% better than market, 5-min TTL via inner loop); MARKET for TP/SL/emergency
- **Position model:** Long or short; max 3 concurrent; per-asset state machine (IDLE → ENTERED → COOLDOWN)
- **Risk controls FOUND:** ATR position sizing, drawdown circuit breaker (persisted), daily trade cap (persisted), per-asset cooldown, max concurrent positions, mandatory SL (2× retry + market-close fallback), ADX regime filter, score threshold gate, BTC correlation filter, S&R zone gate, session/time gate, OI confirmation, funding rate gate, candle completion gate, instance lock
- **Risk controls MISSING:** No testnet mode; Telegram unconfigured (all alerts silent); .env operational blocker (bot won't start as configured)
- **Testnet:** Absent — no testnet flag or testnet endpoint configured

---

## 🔨 AGENT 1 — LOGIC BREAKER

### BUG-v9-L1 | HIGH | Trailing Stop Zero-Size: Warning Only, No Protection Restored
**File:** `src/main.py` lines 1297–1302
**Issue:** After TP1 hits and `active_trades["amount"]` is halved (BUG-P11-1 fix), the trailing stop guard correctly checks `if _tr_size <= 0: logging.warning(...); continue`. However this only logs a warning and silently skips the trailing stop — it does NOT fall back to querying the exchange for the actual position size to recover protection. The position runs unprotected by a trailing stop for the remainder of its life after TP1, if lot-size rounding produced a zero.
**Financial consequence:** After TP1 fires, the remaining 50% position has no trailing stop. If price reverses from +1.5×ATR it can run all the way back to SL or below without the trailing-stop moving the SL to breakeven. At 5× leverage, a full reversal from peak back to entry = TP1 profit fully given back, plus SL loss on the original 50%.
**Fix direction:** On `_tr_size <= 0`, query `hyperliquid.get_positions()` for the live position size and use that as a fallback amount before skipping.

### BUG-v9-L2 | MEDIUM | `active_trades.json` Race Condition Under Two Simultaneous Fills
**File:** `src/main.py`, outer loop reconcile (line 1042) and inner loop reconcile (line 2547)
**Issue:** `active_trades` is a plain Python list mutated in-place in both the outer and inner loops without any lock. The outer loop runs `for tr in active_trades[:]` and calls `active_trades.remove(tr)`, then `save_active_trades(active_trades)`. The inner loop simultaneously does the same. `asyncio` cooperative multitasking means two coroutines cannot execute the same line simultaneously, but `asyncio.to_thread()` (used for Kronos inference) runs on a thread pool and could interact with the list if the implementation ever moves list mutations into those threads. Currently safe, but one future `asyncio.to_thread()` wrapping a mutating operation would introduce a hard-to-reproduce race.
**Financial consequence:** If the list is mutated from a thread while an `active_trades.remove(tr)` runs in the event loop, TP/SL tracking entries can be dropped silently, leaving positions unguarded. At 5× leverage, the guardian skipping one cycle for an untracked position = full SL loss.

### BUG-v9-L3 | MEDIUM | Inner Loop Limit Cancel + `continue` Skips `active_trades` Cleanup
**File:** `src/main.py` lines 2839–2845
**Issue:** In the inner loop, when an unfilled limit is cancelled after 3 polls (`_i_filled_qty == 0`), the code does `continue` after the cancel — it never appends to `active_trades` and never calls `state_mgr.record_entry()`. This is correct for the happy path. But if `cancel_order` raises an exception, the except block logs a warning and falls through — the code continues to the `active_trades.append(...)` at line 2889 with `_itp_sl_size = 0.0` (since `_i_filled_qty` is 0). This records a zero-size trade in `active_trades`, which the guardian will then try to protect with zero-size TP/SL orders.
**Financial consequence:** A zero-size entry in `active_trades` that the cancel failed on → guardian tries to place zero-size SL orders repeatedly every outer cycle, each attempt failing. The slot is permanently consumed for that asset (state = ENTERED) preventing any new real trade on that asset.

### BUG-v9-L4 | MEDIUM | Daily Macro Trend Filter Uses 1d ADX — Missing on Low-History Assets
**File:** `src/main.py` lines 2001–2013
**Issue:** `_outer_macro_trending = _outer_adx_1d is not None and float(_outer_adx_1d) > 20`. The daily trend filter (which blocks BUY on BEARISH daily trend) only activates when ADX is available. For new assets or those with sparse 1d history (< 14 candles for ADX), `_outer_adx_1d` is None — the macro filter never activates for these assets. An asset trending strongly bearish on the daily but with insufficient daily candle history passes through the macro filter unconditionally.
**Financial consequence:** Long entry on an asset with a daily downtrend (insufficient 1d ADX data) at 5× leverage, if the daily trend reversal plays out, produces a loss amplified by leverage. No alert that the filter was bypassed.

### BUG-v9-L5 | LOW | Stale Candle Warning — No Halt, Trade Continues on Old Data
**File:** `src/main.py` lines 1460–1471 (CHAOS-3 FIX)
**Issue:** The stale candle watchdog correctly identifies when a candle is >3× interval old and logs a warning. However it does NOT halt trading for that asset — the outer loop continues and computes a signal score on stale data. The `continue` that would skip the asset is absent; only `logging.warning` is emitted.
**Financial consequence:** If Hyperliquid's REST endpoint returns 30-minute-old 5m candles due to a data feed lag, the bot generates and potentially executes a signal on stale market state. At 5× leverage, a 30-minute gap in BTC data can precede a 2–5% move. Executing into the tail of that move produces immediate SL.

---

## 🔐 AGENT 2 — SECURITY ANALYST

### BUG-v9-S1 | CRITICAL | Live Keys in Plaintext `.env` — Still Present
**File:** `.env` lines 9–11
**Issue:** `HYPERLIQUID_PRIVATE_KEY`, `ANTHROPIC_API_KEY`, and `HYPERLIQUID_VAULT_ADDRESS` are live production values in `.env`. The `.gitignore` now properly excludes `.env` (BUG-v8-S6 confirmed fixed). However the file still exists on disk on any machine where it has been opened — including within this Cowork session. The private key grants full trading authority over the Hyperliquid account. This is a persistent reminder: the key should be rotated whenever it is exposed to any new context.
**Financial consequence:** Key compromise = complete and immediate fund drain. No recovery path.

### BUG-v9-S2 | HIGH | Bot Will Not Start — `.env` Misconfiguration Still Unresolved
**File:** `.env` lines 184, 189; `src/main.py` lines 55–64
**Issue:** `.env` still has `API_HOST=0.0.0.0` and `DASHBOARD_TOKEN=` (blank). The BUG-v7-S3 fix adds `sys.exit(1)` in this exact combination. This has been present since v7 and is still present in v9 — the operator has not updated `.env`.
**Financial consequence:** Bot is currently UNRUNNABLE. No automated risk management, no guardian, no KILLSWITCH monitoring. If any position is open on the exchange from a prior session, it is completely unmonitored.
**Required action:** Either set `DASHBOARD_TOKEN=<random_secret>` in `.env`, or change `API_HOST=127.0.0.1`. One-line fix.

### BUG-v9-S3 | HIGH | Telegram Still Unconfigured — All Runtime Alerts Offline
**File:** `.env` lines 197–198
**Issue:** `TELEGRAM_BOT_TOKEN=` and `TELEGRAM_CHAT_ID=` are both blank (unchanged across v7, v8, v9). Every alert — startup, KILLSWITCH, circuit breaker, SL failure, market-close failure, SL orphan — silently does nothing. The `send_alert_sync()` fix in v8 improved delivery reliability, but if the destination is unconfigured, there's nothing to deliver.
**Financial consequence:** Critical events at 3 AM produce no operator notification. Bot can hit 12% drawdown circuit breaker, auto-reset at midnight, and resume trading with no operator awareness.

### BUG-v9-S4 | MEDIUM | Dashboard Exposed on All Interfaces Before Token Set
**File:** `.env` line 184: `API_HOST=0.0.0.0`
**Issue:** Even setting `DASHBOARD_TOKEN` (which would fix BUG-v9-S2 and let the bot start) while leaving `API_HOST=0.0.0.0` means the dashboard is accessible from any network interface on port 3000. The bearer-token auth gate protects the `/diary`, `/live`, `/logs` endpoints, but:
1. The root `/` (dashboard HTML itself) is always allowed without auth
2. Any host on the network can reach the login page and attempt token brute-force
3. If the VPS has an open firewall rule for port 3000 (e.g. cloud security group "all traffic"), the dashboard is internet-accessible
**Financial consequence:** Partial account exposure (dashboard HTML) or full exposure if DASHBOARD_TOKEN is weak/guessable.

---

## 🌪️ AGENT 3 — CHAOS TESTER

### CHAOS-v9-1 | HIGH | Kronos Pre-Warm Uses `asyncio.ensure_future` in Sync Context — Python 3.12+ Crash
**File:** `src/main.py` lines 557–564
**Issue:** The PERF-1 fix added:
```python
async def _prewarm_kronos():
    ...
asyncio.ensure_future(_prewarm_kronos())  # line 564
```
This is inside the synchronous `main()` function, BEFORE `asyncio.run(main_async())` is called at line 3220. There is NO running event loop at line 564. Behavior:
- **Python 3.12+:** `asyncio.ensure_future()` raises `RuntimeError: no running event loop` → bot crashes at startup before any trading begins.
- **Python < 3.12:** `asyncio.get_event_loop()` creates a legacy loop; `ensure_future` schedules the task on that legacy loop. `asyncio.run(main_async())` creates a NEW event loop — the pre-warm task is on the OLD loop and never executes. Kronos cold-loads during the first guardian pass, which was the original problem PERF-1 was meant to fix.
**Financial consequence:** Python 3.12+: Complete startup failure — no trading bot at all. Python < 3.12: The PERF-1 fix silently does nothing; Kronos still cold-loads (30–60s) during the first guardian pass, leaving any open positions without SL re-placement protection for that window.
**Fix direction:** Move the `ensure_future` call inside `main_async()`, after the event loop is running: `asyncio.ensure_future(_prewarm_kronos())` or `asyncio.create_task(_prewarm_kronos())`.

### CHAOS-v9-2 | HIGH | Stale Candle Warning Doesn't Halt — Trading Continues on Old Data
**File:** `src/main.py` lines 1460–1471
**Issue:** Stale candle detection logs a WARNING but does NOT skip the asset's signal computation. The asset proceeds through the full scoring pipeline on potentially 30+ minute old OHLCV data.
**Financial consequence:** Same as BUG-v9-L5. Signal fires on stale data during an exchange data lag.

### CHAOS-v9-3 | MEDIUM | `active_trades.json` Write Race — Two Concurrent Closes
**Scenario:** TP fills at the exact same inner tick that the outer loop's reconcile runs (both are async coroutines on the same event loop but the outer and inner overlap via nested loops). If both loops detect the position as closed and both call `active_trades.remove(tr)` and `save_active_trades(active_trades)`, one remove raises `ValueError` (caught), but both then write to `active_trades.json`. The second write wins and correctly reflects the empty list. Risk is low in normal operation but elevated during rapid fills.
**Financial consequence:** Low severity if state converges correctly. If the ValueError is uncaught or a new entry is appended in between, the file can contain a duplicate or orphan entry, causing the guardian to try to re-place TP/SL on a closed position.

### CHAOS-v9-4 | MEDIUM | `asyncio.to_thread` for Kronos — Shared State Under Load
**File:** `src/main.py` lines 1861, 2708
**Issue:** Claude calls use `asyncio.to_thread(agent.confirm_trade, ...)`. The `TradingAgent.confirm_trade()` function accesses shared state (if any instance variables). If two assets simultaneously call Claude via `to_thread` (4 assets, high-score cycle), two threads run `confirm_trade` concurrently. If `confirm_trade` writes to any shared instance state (e.g. rate-limit counters, token tracking), a data race exists.
**Financial consequence:** Non-deterministic — could cause duplicate Claude calls, incorrect rate-limit tracking, or corrupt token count. Low probability but hard to debug in production.

### CHAOS-v9-5 | LOW | SL Orphan Check Uses `hyperliquid.get_positions()` — Extra Uncounted API Call
**File:** `src/main.py` lines 2444–2450
**Issue:** The SL orphan check runs on every 5-minute inner tick and calls `await hyperliquid.get_positions()` for each pending trade. This is an additional REST call not accounted for in the rate limit budget. With 3 concurrent positions all deferred (e.g. after a restart mid-session), this is 3 extra REST calls every 5 minutes. Hyperliquid rate limits are per-second; during a busy tick, these extra calls can push into the rate limit headroom reserved for exit orders.
**Financial consequence:** Rate limit on exit order execution during the SL orphan check window. Exit delayed 2–8s due to backoff.

---

## ⚡ AGENT 4 — PERFORMANCE ENGINEER

### PERF-v9-1 | HIGH | `asyncio.ensure_future` in Sync Context (Same as CHAOS-v9-1)
**File:** `src/main.py` line 564
**Issue:** Already detailed in CHAOS-v9-1. From a performance perspective: on Python 3.12+, the bot crashes before serving any trades. On Python < 3.12, the pre-warm silently fails — Kronos cold-loads during the first guardian pass (30–60s blocking call inside `asyncio.to_thread` since `get_kronos_modifier` eventually calls `_load_kronos()` which blocks). This 30–60s is NOT blocking the event loop (since the Kronos call in the outer loop wraps via `try/import`) but it means the guardian is delayed for that long on first call.
**Severity:** High — PERF-1 fix is currently non-functional on all Python versions.

### PERF-v9-2 | MEDIUM | SL Orphan Check: Sequential `get_positions()` Per Asset
**File:** `src/main.py` lines 2444–2450
**Issue:** The SL orphan check loops over `active_trades` and calls `await hyperliquid.get_positions()` for each trade that has `sl_oid=None`. With 3 concurrent deferred trades, this is 3 sequential REST calls. A single `get_positions()` call returns all positions — the check should call once and filter, not call once per trade.
**Financial consequence:** Unnecessary API call multiplication; each extra call consumes rate-limit budget.

### PERF-v9-3 | MEDIUM | Outer Loop Fetches 400 5m Candles Per Asset — Network Overhead
**File:** `src/main.py` line 1452
**Issue:** `hyperliquid.get_candles(asset, "5m", 400)` fetches 400 candles × 4 assets = 1600 OHLCV records per outer cycle. This is correct for Kronos (BUG-v8-L2 fix) but introduces ~4× more network data than the previous 100-candle fetch. On a bandwidth-constrained VPS (1 Mbps), 1600 JSON objects with 6 fields each ≈ 96KB per cycle. At 60-minute outer cycles this is negligible, but the INNER loop also fetches 400 5m candles per asset per tick (line 2600) = 1600 records × 11 ticks × 4 assets per hour ≈ 17,600 candle records/hour. The inner-loop fetch is redundant for anything beyond the latest ~5 candles.
**Financial consequence:** Not directly a fund-loss risk, but on a 1 Mbps VPS it can increase tick duration and introduce latency during candle fetches, delaying entry/exit order placement.

### PERF-v9-4 | LOW | `_prewarm_kronos` Uses `asyncio.to_thread` for `_load_kronos`
**File:** `src/main.py` lines 558–563
**Issue:** `await asyncio.to_thread(_load_kronos)` inside `_prewarm_kronos()` is the correct pattern for CPU-blocking tasks in async code. This is fine and will work correctly once the `ensure_future` placement is fixed (BUG-v9-P1 / CHAOS-v9-1). No issue with the internal implementation of the pre-warm itself.

---

## 👤 AGENT 5 — OPERATOR SAFETY

### OPS-v9-1 | CRITICAL | Bot is STILL UNRUNNABLE — .env Unchanged for 3 Audit Passes
**File:** `.env` lines 184, 189
**Issue:** `API_HOST=0.0.0.0` + `DASHBOARD_TOKEN=` (blank). This triggers `sys.exit(1)` (BUG-v7-S3 fix) at startup. This was reported in v7, v8, and now v9. The fix is a one-line change to `.env`. This is the highest-priority operational blocker — nothing else matters until the bot can start.
**Operator consequence:** Bot has been unrunnable since the v7 security fix was applied. If the operator believes the bot is running, it is not. Any positions opened before v7 was deployed are completely unmonitored.

### OPS-v9-2 | HIGH | Kronos Pre-Warm Broken (Python 3.12+: Crash at Startup)
**File:** `src/main.py` line 564
**Issue:** See CHAOS-v9-1. On Python 3.12+ (default on modern Ubuntu 24.04), `asyncio.ensure_future()` in a non-async context raises `RuntimeError`. The bot crashes before the trading loop starts. On older Python, the pre-warm silently never runs. Both outcomes leave Kronos cold-loading during the first guardian pass.
**Operator consequence (Python 3.12+):** Bot crashes silently at startup. Operator sees no output, assumes bot is running, it is not. OPS-v9-1 and OPS-v9-2 together mean the bot cannot start under any reasonable configuration.

### OPS-v9-3 | HIGH | Telegram Still Unconfigured — Three Audit Passes With No Alerts
**File:** `.env` lines 197–198
**Issue:** Unchanged since v7. Every alert-dependent safety feature is effectively disabled:
- KILLSWITCH confirmation not delivered
- Circuit breaker auto-reset not delivered
- SL failure not delivered
- SL orphan detection alert not delivered
- Bot startup/shutdown not delivered
The `send_alert_sync` fix (v8) and `send_alert` throughout the code are fully operational — but they deliver to nowhere.
**Operator consequence:** Bot operates completely silently. Any failure at any hour is invisible.

### OPS-v9-4 | MEDIUM | No Testnet Mode — Three Audit Passes
**File:** `.env`, `src/config_loader.py`
**Issue:** Unchanged since v7. There is no testnet configuration, no paper mode, no dry-run flag. `HYPERLIQUID_NETWORK=mainnet`. Every code change goes directly to live funds. The v9 audit found a new bug (CHAOS-v9-1) that could crash the bot at startup on Python 3.12+ — this would have been caught in 24h of testnet validation.
**Operator consequence:** Any regression in any of the fixes made across v7/v8/v9 executes immediately on real money.

### OPS-v9-5 | MEDIUM | `requirements.txt` Missing Kronos Dependencies
**File:** `requirements.txt`
**Issue:** `requirements.txt` does not include `torch`, `transformers`, or `chronos-forecasting`. Running `pip install -r requirements.txt` installs the bot without Kronos. Kronos gracefully degrades to 0.0 modifier — this is by design. However:
1. The operator may not know that `pip install -r requirements.txt` is insufficient for full Tier 1 Kronos operation
2. `CLAUDE.md` docs mention `pip install ... torch transformers` but `requirements.txt` is the authoritative install manifest
3. If Kronos is intended to be Tier 1 active (MASTER RULE 1), its dependencies should appear in `requirements.txt` (with `extras` or conditional install if optional)
**Financial consequence:** Non-fatal. Kronos modifier = 0.0 → score boundary trades may be affected by missing ±0.5 modifier. At score=5.5 (base), a missing +0.5 from Kronos keeps the trade as HOLD when it should fire. No direct fund loss from this but reduces strategy effectiveness.

### OPS-v9-6 | LOW | BUG-v8-L7 Partial Fix — Trailing Stop Has No Protection After Zero-Size
**File:** `src/main.py` lines 1297–1302
**Issue:** After TP1, if `active_trades["amount"]` rounds to zero due to lot-size constraints, the trailing stop is silently skipped with a warning. No fallback to exchange-queried position size. The position runs without trailing stop protection.
**Operator consequence:** For small-allocation positions on high-price assets (e.g. 0.001 BTC rounded to 0.000 after TP1), the trailing stop is permanently disabled for the life of the trade. Operator sees a warning in logs but no alert (Telegram unconfigured).

---

## 🔄 STEP 3 — FULL FLOW AUDIT

| Flow | Status | Key Issue |
|------|--------|-----------|
| Signal generation | ✅ Volume bonus uses closed candle | Stale candle watchdog warns but doesn't halt |
| Pre-trade risk check | ✅ Balance refreshed before sizing | Daily macro filter bypassed when ADX unavailable |
| Position sizing | ✅ Score-weighted, fees+funding in TP/SL | — |
| Entry order | ✅ LIMIT at 0.15% better, entry_oid captured | — |
| Entry confirmation | ✅ `tp_sl_size = filled_qty` | — |
| SL placement | ✅ 2× retry + market-close fallback | — |
| Position monitoring | ✅ Guardian uses active_trades amount | Trailing stop has no fallback for zero-size after TP1 |
| SL orphan detection | ✅ New in v9 — 5-min inner tick check | Makes extra sequential REST calls per deferred trade |
| Exit signal | ✅ TP1/TP2 partial close confirmed | — |
| WebSocket reconnect | ⚠️ Stale candle watchdog added | Warning only — no trading halt on stale data |
| Drawdown check | ✅ Midnight rollover + alert flag | Alert still unconfigured (Telegram blank) |
| Kill switch | ✅ Both inner and outer: 3× retry | Alert delivery works but no Telegram configured |
| Restart recovery | ✅ state.json corrupt → sys.exit(1) | — |
| Kronos pre-warm | ❌ ensure_future in sync context | Python 3.12+: crash; older: pre-warm silent no-op |
| Bot startup | ❌ .env misconfiguration | sys.exit(1) at startup — bot cannot run |
| Alert dispatch | ⚠️ Fixed technically | Telegram unconfigured — all alerts silently swallowed |
| Backtest | N/A | No backtest engine |

---

## 📊 STEP 4 — EDGE CASE TABLE

| Edge Case | Expected | Actual Risk | Severity |
|-----------|----------|-------------|----------|
| Python 3.12+, bot starts | Normal startup | RuntimeError crash at line 564 | 🔴 CRITICAL |
| .env with API_HOST=0.0.0.0 + no token | Operator warned to fix | sys.exit(1) — bot can't start | 🔴 CRITICAL |
| Telegram unconfigured | Operator configures it | All runtime alerts silent for 3 passes | 🟠 HIGH |
| Kronos pre-warm on Python < 3.12 | Pre-warm fires | Pre-warm scheduled on wrong loop, never runs | 🟠 HIGH |
| Stale candle (30 min lag) | Halt asset signal | Warning logged, signal fires on stale data | 🟠 HIGH |
| TP1 hits, amount rounds to zero | Query exchange position size | Warning + skip; trailing stop permanently absent | 🟡 MEDIUM |
| 3 deferred positions + SL orphan check | Single get_positions() call | 3 sequential REST calls per tick | 🟡 MEDIUM |
| Inner limit cancel fails + fall-through | Clean state | Zero-size entry in active_trades | 🟡 MEDIUM |
| 1d candle history < 14 candles | Macro filter active | Filter bypassed: no ADX = no daily gate | 🟡 MEDIUM |
| Python 3.12+ with fixed .env | Starts normally | Crashes at line 564 (ensure_future) | 🔴 CRITICAL |
| Two Claude calls concurrent (4 assets) | Thread-safe | Potential shared-state race in confirm_trade | 🟡 MEDIUM |
| Outer loop 400 candles × 4 assets | Network acceptable | 17,600 inner candle records/hour on slow VPS | 🔵 LOW |
| Kronos not in requirements.txt | Modifier = 0.0 graceful | Tier 1 component silently inactive from fresh install | 🔵 LOW |

---

## 🔄 STEP 6 — PEER REVIEW

### 🔨 LOGIC BREAKER reviews other agents:
- **From CHAOS:** CHAOS-v9-1 (ensure_future crash) is the most immediately critical finding. It's not a trading logic bug — it's a startup crash that prevents any trading at all. But it lands exactly when the guardian needs to run for any open positions from the prior session.
- **From OPERATOR:** OPS-v9-1 + OPS-v9-2 together mean the bot cannot start under any realistic configuration. These two must be fixed atomically (same `.env` edit session).
- **What all agents missed:** The SL orphan check (new feature) is actually a good fix for the previously-unprotected window after a deferred limit fill. However it only triggers when `sl_oid is None` — it doesn't check if `sl_oid` is set but the order was actually rejected. If the SL placement at entry produced an `sl_oid` but Hyperliquid silently dropped the order (network race), the orphan check will not fire. The guardian covers this eventually, but the inner-tick SL orphan check has a false-negative path.

### 🔐 SECURITY ANALYST reviews other agents:
- **From LOGIC BREAKER:** BUG-v9-L3 (zero-size entry from failed cancel) is a security-adjacent concern: if an attacker could induce repeated exchange timeouts, they could exhaust all asset slots with zero-size zombie entries, permanently blocking real trade execution (DoS via state corruption).
- **From OPERATOR:** The key concern this pass: the `.gitignore` properly excludes `.env`, but the file still exists in this Cowork session's folder context. This is a persistent reminder that once a private key is observed in any session, it should be considered potentially exposed and rotated.
- **One issue that would stop live trading:** OPS-v9-1 — the bot can't start. That's the answer.

### 🌪️ CHAOS TESTER reviews other agents:
- **From PERFORMANCE:** PERF-v9-3 (400 candles × 4 assets × 11 inner ticks) is relevant when both network and processing time compound. On a 1 Mbps VPS during a volatile period with 4 assets, 11 inner ticks per hour, each tick fetching 1600 5m candles: the tick duration approaches the 5-minute tick interval itself. The `_interruptible_sleep(300)` sleep is spent waiting, but if candle fetch + scoring takes 90s on a slow VPS, the inner loop effectively runs every 6.5 minutes instead of 5, missing candle signals.
- **From OPERATOR:** OPS-v9-5 (requirements.txt missing Kronos dependencies): a fresh `pip install -r requirements.txt` followed by running the bot produces a bot that runs with Kronos modifier = 0.0. MASTER RULE 1 says "Kronos-mini is ACTIVE Tier 1 — always attempt load". The intent is always-attempt, graceful degrade. But from `requirements.txt`, it will ALWAYS degrade. The fix-or-document choice belongs to the operator.
- **The scenario all agents missed:** Bot starts (after fixing .env + Python version), Kronos loads, pre-warm fires correctly. But what if the VPS has no internet access to HuggingFace? `ChronosPipeline.from_pretrained("KronosResearch/Kronos-mini")` will hang on a network timeout during the cold load inside `asyncio.to_thread`. The timeout for HuggingFace downloads is not set explicitly — `requests` default is no timeout. The pre-warm thread hangs indefinitely, holding the thread pool. `asyncio.to_thread` has no timeout parameter. The guardian eventually calls `get_kronos_modifier`, the `_load_kronos` short-circuits on `_kronos_failed` (already set?), but if the pre-warm thread is stuck waiting, `_kronos_failed` is never set. The inner-loop `get_kronos_modifier` call tries to load again (since `_kronos_loaded` is False and `_kronos_failed` is False), hangs again. Result: inner loop hangs indefinitely waiting for HuggingFace on an air-gapped VPS.

### ⚡ PERFORMANCE ENGINEER reviews other agents:
- **From CHAOS:** CHAOS-v9-1 is the highest severity finding. From a performance perspective: even if the ensure_future doesn't crash (Python < 3.12), the intended optimization (pre-warm before guardian) doesn't work. The guardian still pays the 30–60s cold-load cost on first execution.
- **From OPERATOR:** OPS-v9-3 (Telegram unconfigured) has a performance angle — if Telegram were configured and alerts were being sent, any latency in `send_alert()` would add to the inner loop tick time. The async `send_alert()` with a 10s timeout is non-blocking for the event loop, but if it times out waiting for Telegram, that's 10s of event-loop-occupied time that delays the next `asyncio.sleep`.
- **What all agents missed:** The `handle_live` dashboard endpoint (called by operators checking the live view) calls `hyperliquid.get_user_state()` synchronously on every HTTP request. These calls are NOT tracked in the bot's rate-limit accounting. If an operator has the dashboard auto-refreshing every 5 seconds, this adds 12 extra `get_user_state()` calls per minute. During a high-volatility period with heavy inner-loop execution, these extra calls can push into the rate-limit headroom reserved for exit orders.

### 👤 OPERATOR SAFETY reviews other agents:
- **From LOGIC BREAKER:** BUG-v9-L5 (stale candle warning without halt) is what I'd call the most insidious remaining issue. The operator sees a WARNING in logs. But since Telegram is unconfigured, the warning never reaches them. The bot enters a trade on stale data, gets stopped out immediately because the market already moved 30 minutes ago, and the operator has no idea why.
- **From CHAOS:** CHAOS-v9-3 (HuggingFace download hang on air-gapped VPS) is a new scenario not previously identified. If the production VPS has egress firewall rules (common in corporate environments), the Kronos load hangs indefinitely. The bot appears unresponsive but is not dead.
- **The one issue I'd refuse to allow live trading over:** OPS-v9-1 + OPS-v9-2 together. The bot cannot start. The PERF-1 fix, if left as-is, will crash Python 3.12+ bots at line 564 even after fixing `.env`. These are the two concrete blockers that need to be addressed before any live trading attempt.

---

## 🏛️ STEP 7 — FINAL REPORT

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
     🏛️  TRADING BOT BUG COUNCIL — v9 FINAL REPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 BOT SUMMARY
   Code-first hybrid perps scalper | Crypto | Hyperliquid mainnet
   5× leverage | LIMIT entry + MARKET exits | Per-asset state machine
   Risk controls FOUND: ATR sizing, drawdown CB (persisted), daily cap
     (persisted), cooldown, max concurrent, mandatory SL (2× retry +
     market-close fallback), ADX regime, BTC correlation, S&R zone,
     time/session gate, OI gate, funding gate, candle completion gate,
     instance lock, stale candle warning, SL orphan detection
   Risk controls MISSING: Testnet, Telegram, .env startup blocker
   DeFi Agent: Not activated | Re-audit pass: v9 (post-v8 fixes)
```

---

### 💀 STARTUP BLOCKERS — Fix Before Bot Can Run At All

**[OPS-v9-1] `.env` Misconfiguration — `sys.exit(1)` at Every Startup**
`API_HOST=0.0.0.0` + blank `DASHBOARD_TOKEN` triggers immediate exit. Present since v7, unchanged through v8 and v9. Reported three times. Fix: in `.env`, set `DASHBOARD_TOKEN=<random_secret>` or change `API_HOST=127.0.0.1`. This is a SINGLE LINE CHANGE.

**[CHAOS-v9-1 / OPS-v9-2] Kronos Pre-Warm Crashes Python 3.12+**
`asyncio.ensure_future(_prewarm_kronos())` at `main.py:564` — called from sync `main()` before event loop exists. Python 3.12+: `RuntimeError` crash. Python < 3.12: pre-warm silently never fires. Fix: move the `ensure_future` / `create_task` call inside `main_async()` after the event loop is running.

---

### 🔴 CRITICAL — Fix Before Live Trading

**[BUG-v9-S1]** Live mainnet private key and API key remain in `.env`. `.gitignore` is now correct, but rotate keys if this session or any prior session exposed the file.

---

### 🟠 HIGH — Fix Before Real Money Exposure

**[OPS-v9-3]** Telegram completely unconfigured — zero runtime alerts. Set `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` in `.env`.

**[CHAOS-v9-2 / BUG-v9-L5]** Stale candle watchdog warns but doesn't halt. Trading continues on >3× stale data.

**[OPS-v9-4]** No testnet mode. Every code change deploys directly to live funds.

---

### 🟡 EDGE CASE — Fix Before Scale

**[BUG-v9-L1]** Trailing stop zero-size after TP1: logs warning, no fallback to exchange position size.
**[BUG-v9-L3]** Inner loop: failed limit cancel fall-through records zero-size entry in active_trades.
**[BUG-v9-L4]** Daily macro trend filter bypassed when 1d ADX unavailable (< 14 candles).
**[PERF-v9-2]** SL orphan check: 3 sequential `get_positions()` calls instead of 1.
**[OPS-v9-5]** `requirements.txt` missing Kronos dependencies — fresh installs never get Kronos.
**[CHAOS-v9-3]** HuggingFace download hang on air-gapped VPS: Kronos load blocks indefinitely.
**[PERF-v9-3]** 400 5m candles × 4 assets × 11 inner ticks = 17,600 candle records/hour.

---

### 🔐 SECURITY

| Severity | Issue |
|----------|-------|
| Critical | BUG-v9-S1 — Live private key + API key in .env (rotate) |
| Critical | OPS-v9-1 — Bot won't start; .env misconfigured |
| High | OPS-v9-2 — Python 3.12+ crash from ensure_future |
| High | BUG-v9-S3 — Telegram unconfigured; all alerts dead |
| Medium | BUG-v9-S4 — Dashboard on 0.0.0.0 after token is set |

---

### ⚡ PERFORMANCE

| Severity | Issue |
|----------|-------|
| High | PERF-v9-1 — Kronos pre-warm non-functional (ensure_future placement) |
| Medium | PERF-v9-2 — Sequential REST calls in SL orphan check |
| Medium | PERF-v9-3 — 400 candles × 11 ticks inner fetch overhead |
| Low | PERF-v9-4 — Dashboard /live makes unaccounted REST calls per HTTP request |

---

### 🛡️ MISSING RISK CONTROLS

- No testnet / paper trading mode (three passes)
- No Telegram alerts (three passes, unconfigured in .env)
- Stale candle warning doesn't halt — trading on old data
- Trailing stop has no protection path when TP1 halves to zero

---

### 📊 MONITORING & OPERATIONAL GAPS

- All 3 startup blockers (API_HOST, DASHBOARD_TOKEN, Python 3.12) remain unresolved
- Kronos not in requirements.txt — always degrades on fresh install
- No emergency shutdown documentation (unchanged)

---

### 📦 TECHNICAL DEBT

- `is_trending_regime()` still dead code (clearly documented INACTIVE — low priority)
- Score 9 base structurally unreachable — noted in docs, no action needed
- `requirements.txt` needs `torch`, `transformers`, or `chronos-forecasting` with optional marker
- No formal test suite (unchanged)

---

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COUNCIL VERDICT: MODERATE RISK (improved from v8 HIGH RISK).
  All 16 v8 bugs confirmed fixed or partially fixed. Bot architecture
  is sound. Two startup blockers (unresolved .env + Python 3.12 crash)
  prevent ANY live trading. After those two fixes, the risk profile
  drops significantly. The code is the most defensive it has been
  across all 9 audit passes.

TOP 3 MUST-FIX (in this order — do not skip):
  1. OPS-v9-1 — Fix .env: set DASHBOARD_TOKEN or change API_HOST=127.0.0.1
     (.env, line 184/189 — one line, 30 seconds)
  2. CHAOS-v9-1 — Move asyncio.ensure_future(_prewarm_kronos()) to
     inside main_async() after event loop starts
     (main.py, ~line 564 → move to inside main_async() ~line 3156)
  3. OPS-v9-3 — Set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID in .env
     (without alerts, every silent failure is invisible at 3 AM)

CAPITAL RECOMMENDATION:
  Before fixes 1+2: $0 — bot cannot start.
  After fixes 1+2 only: $0 — no testnet validation, no Telegram alerts.
  After fixes 1+2+3 + 72h testnet run: max $500 live for 72h observation.
  After all HIGH items resolved + 1-week testnet: scale to full allocation.

GO LIVE: ❌ NOT READY — fix .env (30 seconds), fix ensure_future placement
         (5 minutes), configure Telegram (2 minutes), then run testnet.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

*Generated by Trading Bot Bug Council v9 — 2026-05-28*
*16 prior v8 bugs verified fixed (2 partial). 7 new/remaining issues identified. Risk profile: MODERATE (improved from HIGH in v8).*
