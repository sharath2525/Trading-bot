━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
     🏛️  TRADING BOT BUG COUNCIL — FINAL REPORT v11
         Deepest Re-Audit Pass | 2026-05-29
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 📋 BOT SUMMARY

- **Strategy:** Code-first hybrid perpetual futures (multi-asset, multi-timeframe)
- **Exchange:** Hyperliquid DEX — MAINNET (LIVE FUNDS)
- **Leverage:** Up to 5× — every position risk is multiplied 5×
- **Order types:** LIMIT entry (0.15% better-than-market), LIMIT TP1/TP2, LIMIT SL (market-order fallback)
- **Position model:** Partial close — TP1 (50% at 1×ATR), TP2 (50% at 3×ATR), trailing stop guardian
- **Architecture:** asyncio — outer hourly loop + inner 11×5min loop
- **Claude role:** Structured APPROVE/REJECT gate only (5-factor, TOTAL ≥ 15/25 to APPROVE)
- **Risk controls FOUND:** Circuit breaker (daily loss %), max concurrent positions, max daily trades, per-asset cooldown, ATR position sizing, max leverage cap, min balance reserve, SL orphan recovery, instance lock (port 47293)
- **Risk controls MISSING (in .env):** Telegram alerts = null → zero operator notifications; dashboard exposed on 0.0.0.0 with blank token → startup blocked
- **DeFi Agent:** NOT activated (no on-chain code)
- **Testnet:** Config present but not tested in .env (HYPERLIQUID_NETWORK=mainnet)

---

## V10 FINDINGS STATUS — CORRECTIONS FIRST

The following v10 findings were **already fixed** in the codebase at time of v11 read.
They were erroneously listed as unfixed in v10. They are NOT re-flagged in v11.

| v10 ID | Finding | Status |
|--------|---------|--------|
| LB-v10-1 | Guardian TP2 orphan — bot would never replace missing TP2 | **FIXED** — `_g_tp_count`/`_g_expected_tp_count` logic at main.py:1261–1298 |
| LB-v10-3 | `isBuy`/`is_buy` fill field mismatch — wrong exit-type detection | **FIXED** — dual-lookup at main.py:722 |
| LB-v10-5 | TP1+TP2 simultaneous fill race — TP2 orphaned if both detected same cycle | **FIXED** — TP2 OID check added at main.py:1088–1103 |
| LB-v10-6 | Force-close missing cooldown — ENTERED state persisted after force-close | **FIXED** — `state_mgr.start_cooldown()` at main.py:997–1001 |
| CHAOS-v10-11 | `_save_daily_count()` non-atomic write | **WAS NEVER A BUG** — always used `os.replace()` at line 138; v10 analysis was wrong |
| PERF-v10-8 | Log rotation only at startup — logs could grow unbounded during long runs | **FIXED** — rotation called every outer cycle at main.py:947–951 |

---

## 🔴 CRITICAL — Fix before ANY live trading

### CRIT-1 — STARTUP BLOCKER: Bot cannot start with current .env (4th consecutive audit)
**File:** `.env` + `src/main.py` (security gate ~line 481)
**Finding:** `.env` has `API_HOST=0.0.0.0` and `DASHBOARD_TOKEN=` (blank). The security gate in `main()` calls `sys.exit(1)` when `API_HOST` is not `127.0.0.1` and `DASHBOARD_TOKEN` is falsy. This has been present and unfixed across v8, v9, v10, and v11 audits.
**Financial consequence:** The bot cannot start at all. Any attempted live trading run exits immediately before the first candle is processed. This is the highest-priority blocker — everything else is moot until this is resolved.
**Fix:** Either set `API_HOST=127.0.0.1` OR set a strong `DASHBOARD_TOKEN=<secret>`.

### CRIT-2 — Telegram alerts unconfigured — zero operator notifications (4th consecutive audit)
**File:** `.env` lines for `TELEGRAM_BOT_TOKEN=` and `TELEGRAM_CHAT_ID=`
**Finding:** Both values are blank. Every Telegram send call silently no-ops. This means NO notification for: circuit breaker triggered, SL orphan detected, daily loss limit hit, KILLSWITCH activated, force-close executed, consecutive API failures. The operator has no visibility into bot health during live operation.
**Financial consequence:** At 5× leverage, a runaway loss scenario (circuit breaker hit, bot halts) produces zero notification. Operator only discovers the halt by manually checking logs. If halt is missed overnight, margin erosion continues on open positions that were not closed before halt.
**Fix:** Configure a real Telegram bot token and chat ID, or substitute another alert channel.

---

## 🟠 HIGH — Fix before real money exposure

### HIGH-1 — State file structural corruption silently resets all open positions to IDLE
**File:** `src/trade_state.py` lines 84–92 (`_load()`)
**Finding:** `json.JSONDecodeError` correctly causes `sys.exit(1)` (CHAOS-7 FIX). However, if `state.json` contains valid JSON with corrupted *structure* — e.g., `{"states": null, "cooldown_until": {}, "entry_time": {}}` — then `payload.get("states", {})` returns `None` silently. Iteration over `None` in the expired-cooldown loop at line 89 (`for a, t in self._cooldown_until.items()`) would raise `AttributeError` on the `cooldown_until` dict if also null, but `get("states", {})` returning `None` causes `self._states = None`. Subsequent calls to `get_state()` calling `self._states.get(asset, self.IDLE)` would crash with `AttributeError: 'NoneType'`.

More critically: if `state.json` is written during a partial failure where `states` is a valid dict but missing certain asset keys (e.g., `BTC` key gone but position is open), the bot sees `BTC` as IDLE and re-enters a new position on top of the existing open one.
**Financial consequence:** Double BTC position at 5× leverage = effective 10× leverage exposure. At $50k account, that's $500k notional on a single asset.
**Fix:** Add structural validation after load — check that `_states` is a `dict` and halt if not; add a type check for each restored state value.

### HIGH-2 — Inner loop daily macro filter missing ADX=None bypass (asymmetry with outer loop)
**File:** `src/main.py` line 2845
**Outer loop (FIXED at BUG-v9-L4):**
```python
_outer_macro_trending = (
    (_outer_adx_1d is not None and float(_outer_adx_1d) > 20)
    or (_outer_adx_1d is None and _outer_trend_1d != "UNKNOWN")  # ← BUG-v9-L4 FIX
)
```
**Inner loop (still uses old logic):**
```python
_imacro_trending = _iadx_1d is not None and float(_iadx_1d) > 20  # ← BUG-v9-L4 NOT applied
```
**Finding:** When a new asset has fewer than 14 daily candles (insufficient ADX history), `_iadx_1d` is `None`. The outer loop correctly applies the daily trend filter based on EMA trend alone (`_outer_trend_1d != "UNKNOWN"`). The inner loop silently skips the entire daily filter because `None is not None` is `False`. The inner loop can then execute counter-trend trades on new assets at any of the 11 inner ticks — trades the outer loop's scan would have blocked.
**Financial consequence:** A BUY signal fired by the inner loop on an asset in a clear 1d BEARISH EMA trend (which the outer loop would block) enters without the daily bias gate. At 5× leverage, a 3% move against the trend = 15% position loss before SL.

### HIGH-3 — `validate_trade()` minimum order bump can silently override ATR risk sizing
**File:** `src/risk_manager.py` lines 358–361 and 411–412
**Finding:** Two separate locations bump the allocation to a $11 minimum:
```python
if alloc_usd < 11.0:
    alloc_usd = 11.0       # Line ~360 — bumps calculated allocation
...
if max_alloc < 11.0:
    max_alloc = 11.0       # Lines 411-412 — bumps the ATR-derived ceiling
```
If ATR-based sizing computes a safe allocation of $5 (e.g., very close SL, volatile asset), the first bump makes it $11. If `max_alloc` also computes below $11, the second bump raises the ceiling to $11 as well — creating a situation where the bump **overrides** the ATR risk calculation that said the correct exposure is <$11. The risk cap set by `1% ATR rule` is circumvented silently.
**Financial consequence:** At 5× leverage, $11 minimum → $55 notional. For a micro-cap on a 30% ATR move, the minimum bump could represent 5× the intended risk. The ATR cap is a MASTER RULE 4 requirement and must not be silently overridden.

---

## 🟡 MEDIUM — Fix before sustained operation

### MED-1 — `_code_compute_tpsl()` funding buffer always widens TP regardless of direction
**File:** `src/main.py` lines 297–301 (`_code_compute_tpsl()`)
**Finding:** The comment says "only applied when funding_rate is adverse." The code:
```python
_fr = abs(float(funding_rate or 0))
funding_buffer = entry * _fr if _fr > 0 else 0.0
```
`abs()` discards the sign — making the buffer always positive. For a BUY with **negative** funding (receiving funding, favorable), the buffer still makes TP harder to reach. The code is supposed to distinguish adverse vs favorable funding based on direction, but it doesn't — `abs()` throws away that information.
- BUY + positive funding (longs pay): buffer correctly widens TP ✓
- BUY + negative funding (longs receive): buffer incorrectly widens TP ✗ — should shrink or eliminate
- SELL + negative funding (shorts pay): buffer correctly widens TP (makes TP price lower) ✓
- SELL + positive funding (shorts receive): buffer incorrectly widens TP ✗
**Financial consequence:** On favorable-funding trades, the bot sets TP ~0.01–0.1% further than necessary, reducing win rate on TP1 hits. Persistent misapplication across many trades causes measurable P&L drag. Not fund-loss risk but material over many cycles.

### MED-2 — Inner loop: redundant dual `get_positions()` calls per tick under deferred trades
**File:** `src/main.py` lines ~2557–2612 (inner loop SL orphan check) and ~2597 (pending limit cancel)
**Finding:** When there are deferred (pending) trades, the inner loop executes:
1. `get_positions()` call inside the SL orphan check → fetches ALL positions to find orphaned SL orders
2. `get_positions()` call for each pending limit order → separate per-trade fetch for cancel check

These are two distinct API calls that query the same position data. Under rate limit pressure (Hyperliquid enforces per-IP/per-key limits), consuming rate budget on the duplicate orphan check could starve the cancel check for a losing position. The cancel check is the more time-sensitive operation.
**Financial consequence:** Rate-limit-induced delay on cancel check leaves a filled limit entry order un-cancelled. The position is opened but the cancel confirmation arrives late, causing the entry to be processed twice or the SL placement to be delayed by one tick.

### MED-3 — Sequential RSS feed fetch: 9-second worst-case outer loop delay
**File:** `src/main.py` lines 412–430 (`_fetch_macro_context()`)
**Finding:** Three RSS feeds are fetched SEQUENTIALLY inside a single `ClientSession`:
```python
for _url, _bucket, _prefix in _feeds:   # ForexFactory, CoinDesk, Reuters
    async with _sess.get(_url, timeout=ClientTimeout(total=3)) as _resp:
        ...
```
If all three feeds are slow/unresponsive, the outer loop waits up to 3 × 3 = **9 seconds** before proceeding to signal analysis. The outer loop runs hourly, but this 9s delay is added to the first inner-tick start time. For a bot tuned to 5-minute candle close timing (70% candle-complete gate), a 9s extra delay could push the inner loop past the candle close window on the first tick.
**Financial consequence:** First inner-tick signal consistently fires late, missing the intended entry price window by 9+ seconds. On a 5-minute candle at BTC prices, 9 seconds of extra slippage could be $50–$200 of adverse price movement.

### MED-4 — XML bomb vulnerability via untrusted RSS feeds
**File:** `src/main.py` line 421 (`_fetch_macro_context()`)
**Finding:** `ET.fromstring(_text)` parses XML from three external URLs. Python's built-in `xml.etree.ElementTree` is **not** protected against XML bomb attacks (exponentially nested entities — "billion laughs" attack). A compromised or MITM-intercepted RSS feed could return a crafted XML payload that causes the parser to consume gigabytes of RAM, causing OOM-kill of the trading process.
While true XXE (external entity injection) is not possible with ElementTree (it doesn't resolve external entities), the XML bomb is a real attack vector.
**Financial consequence:** OOM-kill mid-trade leaves open positions unprotected. KILLSWITCH does not execute because the process is killed by the OS. SL orders placed on exchange remain active but TP orders become orphaned until the bot restarts.
**Fix:** Replace `ET.fromstring()` with `defusedxml.ElementTree.fromstring()` (drop-in replacement, available via pip).

---

## 🔵 LOW-MEDIUM — Address before sustained live operation

### LOW-1 — `stats.json` written to relative path (only remaining relative-path file)
**File:** `src/main.py` line 628
```python
stats_path = "stats.json"   # ← relative path
```
All other runtime files (`diary.jsonl`, `decisions.jsonl`, `llm_requests.log`, `prompts.log`, `signals.jsonl`) were fixed to use absolute paths via `_MAIN_PROJECT_ROOT`. `stats.json` was missed. If `python src/main.py` is run from a directory other than the project root (e.g., `python /path/to/bot/src/main.py` from `/`), `stats.json` lands at `/stats.json` or wherever CWD is. The stats will accumulate silently in the wrong location. Dashboard `/stats` endpoint would show zeroed stats.
**Financial consequence:** No direct risk; silently incorrect P&L statistics. Operator making sizing decisions based on dashboard win-rate would see stale/incorrect data.

### LOW-2 — `_sl_cooldown_map` in-memory only — restart clears cooldown alert suppression
**File:** `src/main.py` (in-memory dict, not persisted)
**Finding:** `_sl_cooldown_map[asset]` is used to suppress duplicate Telegram alerts when an SL is hit. It is not persisted to disk. On restart, the map is empty. If the bot restarts immediately after an SL hit (e.g., due to KILLSWITCH + restart), the state machine correctly has the asset in COOLDOWN (persisted to `state.json`), but `_sl_cooldown_map` is empty, so the alert suppression logic fires again — operator receives a duplicate "SL hit" alert after restart. The re-entry gate is correctly blocked (state machine handles it), but alert noise is produced.
**Financial consequence:** No direct financial risk. Alert noise can desensitize operators to SL alerts over time.

### LOW-3 — `diary.jsonl` writes lack explicit asyncio lock guard (code quality)
**File:** `src/main.py` lines 804–806, 2395, 3060
**Finding:** Three separate locations write to `diary_path` using `open(diary_path, "a")` without an asyncio Lock. Currently safe because asyncio is single-threaded and writes complete between `await` points (no interleaving possible in the current code flow). However, there is no defensive guard. If the code is later refactored to run outer/inner loops concurrently (e.g., with `asyncio.gather()`), concurrent diary appends could produce corrupted JSONL (two `json.dumps()` lines merged into one invalid line).
**Financial consequence:** No current risk. Future refactoring risk: corrupted `diary.jsonl` breaks P&L tracking and dashboard `/diary` endpoint.

### LOW-4 — `decisions.jsonl` write similarly lacks asyncio lock (code quality)
**File:** `src/main.py` (outer loop `decisions.jsonl` append)
**Finding:** Same pattern as LOW-3. Safe now, fragile under concurrent refactoring.

---

## 🔐 SECURITY FINDINGS

### SEC-1 — XML bomb / billion-laughs attack surface (see MED-4 above)
Severity: **MEDIUM**. Fix: `defusedxml`.

### SEC-2 — Blank DASHBOARD_TOKEN currently causes sys.exit(1) — but if ever bypassed, API is world-accessible
**File:** `src/main.py` security gate + CORS middleware
**Finding:** The current security gate `sys.exit(1)` prevents the bot from starting, which masks a deeper problem: if `API_HOST=0.0.0.0` and `DASHBOARD_TOKEN` is set to a weak value, the dashboard serves live account state, position data, and full trade history to any IP that can reach the port. CORS middleware correctly reflects only `localhost`/`127.0.0.1` origins, but this only blocks browser-based cross-origin requests — it does not block `curl`, Python `requests`, or any direct HTTP client from the internet.
Severity: **HIGH** (if bot ever starts with `API_HOST=0.0.0.0` and weak token).

### SEC-3 — API key in `.env` — confirm `.env` is git-ignored
**File:** `.env` + `.gitignore`
**Finding:** `.env` contains the live `HYPERLIQUID_PRIVATE_KEY` and `ANTHROPIC_API_KEY`. Confirm `.gitignore` includes `.env` so it cannot be accidentally committed. (Cannot verify `.gitignore` contents in this audit run.)
Severity: **CRITICAL** if `.env` is accidentally committed; **routine checklist** if `.gitignore` is confirmed.

---

## ⚡ PERFORMANCE FINDINGS

### PERF-1 — Sequential RSS fetch (9s worst case) — detailed in MED-3 above
Severity: **Medium**. Fix: fetch feeds concurrently with `asyncio.gather()`.

### PERF-2 — Inner loop dual `get_positions()` — detailed in MED-2 above
Severity: **Medium**. Fix: fetch once per tick, share result with both checks.

### PERF-3 — Kronos 120-second pre-warm timeout blocks startup
**File:** `src/main.py` (startup, `main_async()`)
**Finding:** Kronos-mini pre-warm has a 120-second timeout. On a cold start (first model load from HuggingFace cache), this is appropriate. On a slow network or cold filesystem, it can extend startup time significantly. The outer loop cannot begin until pre-warm completes or times out. If Kronos consistently times out (120s × every restart), repeated restarts under live conditions have a 2-minute delay before first signal scan.
**Financial consequence:** If a position-exit signal fires in the first 2 minutes post-restart, it is missed. Open positions are not monitored during Kronos pre-warm.
Severity: **Low-Medium**. Fix: run Kronos pre-warm in background without blocking the outer loop start.

---

## 🛡️ MISSING RISK CONTROLS (operator-level)

### OPER-1 — No operator kill-switch test documented
The KILLSWITCH file mechanism exists but there is no evidence of it being tested on testnet with live position scenarios (TP1 filled but TP2 + SL still open). A KILLSWITCH test must verify: all open orders cancelled, all positions closed, bot stays halted and does not restart.

### OPER-2 — No reconciliation frequency alert
The reconcile loop runs inside the outer loop (hourly). If a fill occurs during the inner loop that the reconcile doesn't catch (e.g., a TP2 fill that occurs while the bot is in the middle of a Kronos pre-warm), the position state could drift for up to 55 minutes (remaining inner ticks + next outer cycle start).

### OPER-3 — No testnet validation run documented
Four consecutive audits (v8–v11) have noted that the bot is configured for mainnet (`HYPERLIQUID_NETWORK=mainnet`) without evidence of a testnet validation run. MASTER RULE guidance requires 24–72h testnet validation before live operation.

### OPER-4 — `MAX_DAILY_TRADES=20` with 5-minute inner ticks is high
At 11 inner ticks × 5 minutes = 55-minute cycle, and multiple assets, the bot can theoretically enter up to 20 trades in a single day. With 3 concurrent positions (MAX_CONCURRENT_POSITIONS=3) and 30-minute cooldown per asset, the 20-trade cap is the effective governor. On a highly volatile day with many APPROVE signals, 20 entries at 5× leverage is significant exposure. Confirm this cap is intentional.

---

## 📊 MONITORING & OPERATIONAL GAPS

| Gap | Severity | Status |
|-----|----------|--------|
| Telegram alerts blank (all notifications silenced) | Critical | Unresolved v8→v11 |
| `stats.json` relative path | Low | New v11 |
| No outer/inner loop heartbeat metric | Low | Ongoing |
| `llm_requests.log` and `prompts.log` relative paths in `decision_maker.py` | Low-Medium | Check confirmation needed |
| Daily PnL summary never sent | Medium | Ongoing |

---

## 📈 BACKTEST VALIDITY

No backtest engine found in codebase. Strategy is forward-tested only via live signals logged to `signals.jsonl`. Confirm:
- Win-rate analysis from `signals.jsonl` uses actual fill prices (not signal prices)
- Fee + funding accounted for in all P&L calculations in `_update_stats()`
- `_update_stats()` hardcodes `0.00045` fee rate (line 648–649) rather than reading `CONFIG["taker_fee_pct"]` — if fee rate changes, stats will silently use wrong fee

---

## STEP 3 — FULL FLOW AUDIT

| Flow | Result |
|------|--------|
| Signal generation | PASS — 5 base signals + volume/pattern bonuses + Kronos; lookback checked |
| Pre-trade risk check | PARTIAL — inner loop daily macro filter asymmetric with outer (HIGH-2) |
| Position sizing | PARTIAL — ATR sizing can be overridden by $11 minimum bump (HIGH-3) |
| Entry order (LIMIT 0.15%) | PASS — limit order, 3-poll fill confirmation, partial-fill cancel |
| Entry confirmation | PASS — `state_mgr.record_entry()` before `save_active_trades()` |
| SL placement | PASS — 2× retry + market-close fallback; SL confirmed before inner loop |
| Position monitoring | PASS — orphan check every inner tick; 12h timeout enforced |
| TP1 hit detection | PASS — LB-v10-5 fixed; both TP1 and TP2 OID checked |
| TP2 guardian | PASS — LB-v10-1 fixed; `_g_tp_count`/`_g_expected_tp_count` logic |
| Trailing stop | PASS — breakeven at +1×ATR, trail at +1.5×ATR; new SL placed before old cancelled |
| Exit confirmation | PASS — `_log_trade_close()` fetches fill, dual-lookup isBuy/is_buy |
| Restart recovery | PARTIAL — state.json structural corruption not detected (HIGH-1) |
| Drawdown check | PASS — circuit breaker persisted, UTC-midnight reset, alert on reset |
| Kill switch | PASS — file-based, 3× retry in both loops; UNTESTED on testnet (OPER-1) |
| Reconcile (divergence) | PASS — runs every outer cycle; gap up to 55min for inner-tick fills |
| PnL calc | PARTIAL — `_update_stats()` hardcodes fee rate instead of reading config |
| Alert dispatch | FAIL — Telegram unconfigured; all alerts silently dropped (CRIT-2) |

---

## STEP 4 — EDGE CASE TABLE

| Edge Case | Expected | Actual Risk | Severity |
|-----------|----------|-------------|----------|
| `state.json` valid JSON, wrong structure (null states) | sys.exit(1) | Silent IDLE reset → re-entry on open position | HIGH |
| New asset with <14 daily candles, inner loop | Outer's ADX=None bypass applies | Inner skips daily macro filter → counter-trend entry | HIGH |
| Favorable funding rate (receiving) | Tighter TP buffer | Same buffer as adverse → TP harder to reach | MEDIUM |
| All 3 RSS feeds timeout | Proceed with empty macro context | 9s delay before inner loop starts | MEDIUM |
| MITM/compromised RSS feed with XML bomb | Exception caught, bot continues | OOM-kill, open positions unprotected | MEDIUM |
| Bot started from wrong CWD | stats.json in project root | stats.json created at CWD root → dashboard shows wrong data | LOW |
| Restart immediately after SL hit | No duplicate alert | Duplicate Telegram alert (if Telegram were configured) | LOW |
| Rate limit during inner tick with deferred trades | Single position fetch shared | Two get_positions() calls → one may fail → cancel check delayed | MEDIUM |
| ATR-sized safe allocation < $11 | ATR cap honoured | Bumped to $11 overriding ATR cap | MEDIUM |
| Cronos pre-warm takes 120s on restart | Outer loop delayed by 120s | Exit signal missed during pre-warm → open position unmonitored | LOW-MEDIUM |
| `state.json` with extra unknown fields | Gracefully ignored | Handled correctly (get with defaults) | PASS |
| Two bot instances started simultaneously | Second exits (port 47293 in use) | Correctly handled by instance lock | PASS |
| Force-close with open position | COOLDOWN set, no re-entry | Correctly handled (LB-v10-6 FIXED) | PASS |
| Guardian TP2 orphan after TP1 hit | TP2 re-placed | Correctly handled (LB-v10-1 FIXED) | PASS |
| isBuy vs is_buy fill field | Both checked | Correctly handled (LB-v10-3 FIXED) | PASS |
| Daily count.json partial write | Atomic | Was already atomic via os.replace() | PASS |

---

## STEP 6 — PEER REVIEW

### 🔨 LOGIC BREAKER reviews other agents:
Most critical finding from another agent: **HIGH-1** (structural state corruption) from Chaos. This is the most direct path to 10× effective leverage — a bot that re-enters a position it thinks is closed while the exchange already has one open.
Biggest risk all agents missed: The `_update_stats()` fee hardcode at line 648–649 uses `0.00045` directly instead of `CONFIG["taker_fee_pct"]`. If the operator changes fee rate in `.env`, statistics silently track wrong fees.
Would refuse to let this bot trade real money until: **CRIT-1 resolved** (bot literally cannot start).

### 🔐 SECURITY ANALYST reviews other agents:
Most critical from another agent: **CRIT-2** (Telegram unconfigured). Security is meaningless without operational visibility — a compromised or crashed bot is discovered only by manual log inspection.
Biggest risk all agents missed: The `.env` file contains live private keys and is read by `dotenv`. If any log handler accidentally logs full environment variables (e.g., during a config validation error), keys appear in `*.log` files. Confirm no logger ever does `logging.info("CONFIG: %s", CONFIG)`.
Would refuse to let this bot trade real money until: **SEC-3 confirmed** (`.env` gitignored) and **CRIT-2 resolved**.

### 🌪️ CHAOS TESTER reviews other agents:
Most critical from another agent: **HIGH-2** (inner loop daily macro bypass). This is a logic asymmetry that silently fires counter-trend entries on new assets — precisely the scenario where the extra gate matters most (new asset = less price history = higher signal noise).
Biggest risk all agents missed: What happens if `funding_rate` in the `_code_compute_tpsl()` call at line 1858 is `None` vs `0`? `float(None or 0)` = `float(0)` = `0.0` — correctly handled. What about a missing `atr14` for an asset mid-run? `_iout["atr14"] = _iact_ctx.get("long_term_4h", {}).get("atr14")` could return `None`. Then `_code_compute_tpsl(entry, None, ...)` passes `None` as `atr` — `round(entry + 1.0 * None + ...)` raises `TypeError`. While `validate_trade()` checks `atr > 0`, if atr14 is set to None after validate_trade passes, the crash is unguarded.
Would refuse to let this bot trade real money until: **HIGH-1 and HIGH-2 resolved**.

### ⚡ PERFORMANCE ENGINEER reviews other agents:
Most critical from another agent: **MED-3** (9s sequential RSS fetch). Performance is the silent killer — every second of delay between signal and entry is slippage, and the 70% candle-close gate makes timing critical.
Biggest risk all agents missed: The Kronos pre-warm (PERF-3) runs to completion before the outer loop starts. On restart after a loss event (the worst time to be slow), the bot spends up to 120 seconds unmonitored. Combine with CRIT-2 (no Telegram) and an operator may not know the bot was down for 120s post-restart.
Would refuse to let this bot trade real money until: **CRIT-2 resolved** (so restart delays are at least visible).

### 👤 OPERATOR SAFETY reviews other agents:
Most critical from another agent: **HIGH-3** (minimum order bump overrides ATR). The ATR 1% rule is MASTER RULE 4. Any silent override of a MASTER RULE is by definition the most severe operator safety concern — it means the risk model is not actually enforced.
Biggest risk all agents missed: There is no documented emergency procedure if the bot is stuck in a loop (consecutive failures, inner loop hung waiting on API). The KILLSWITCH file mechanism cancels orders but if the main thread is blocked on a hanging API call (despite timeouts), dropping the KILLSWITCH file may not be processed. Operators should have a documented "kill the process" step (e.g., `kill -9 $(cat bot.pid)` or Docker `stop`).
Would refuse to let this bot trade real money until: **CRIT-1, CRIT-2, and OPER-3** (testnet run) resolved.

---

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## 🏛️ TRADING BOT BUG COUNCIL — FINAL VERDICT v11
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Progress Since v10

Significant fixes have been applied. Six v10 findings are confirmed resolved. The bot's core architecture — signal pipeline, TP/SL placement, guardian logic, state machine persistence, instance locking, circuit breaker — is substantially more robust than v8 or v9. The fill-detection dual-lookup, guardian TP count tracking, and force-close cooldown fix are all correct and working.

### What Remains

Three categories of issues survive into v11:

**Unresolved from .env (2 Critical, persistent):** CRIT-1 (startup blocker) and CRIT-2 (zero operator alerts) have now appeared across 4 consecutive audits unchanged. These are not code bugs — they are configuration gaps requiring one env variable change each.

**New logic gaps found in v11 (2 High, 1 Medium):** HIGH-1 (state structural corruption), HIGH-2 (inner loop daily macro asymmetry), and HIGH-3 (ATR minimum bump override) are genuine code defects that require fixes.

**Mathematical inaccuracy (1 Medium):** MED-1 (funding buffer direction) causes measurable TP accuracy degradation but is not a fund-loss risk.

---

### TOP 3 MUST-FIX

1. **CRIT-1** — Fix `.env`: set `API_HOST=127.0.0.1` OR set `DASHBOARD_TOKEN`. The bot cannot run at all without this.

2. **CRIT-2** — Configure Telegram (or equivalent alerting). At 5× leverage, a circuit-breaker halt with no notification is a blind loss event.

3. **HIGH-2** — Add ADX=None bypass to inner loop daily macro filter (mirror the BUG-v9-L4 FIX already in outer loop at lines 2076–2079). One 3-line change eliminates a systematic counter-trend entry risk on new assets.

---

### CAPITAL RECOMMENDATION

**$0 live capital** until CRIT-1 is resolved (bot cannot start).

Once CRIT-1 + CRIT-2 are resolved: **≤ 10% of intended live capital** during testnet/paper validation.

Once HIGH-1 + HIGH-2 + HIGH-3 are resolved: **≤ 25% of intended live capital** for first live week, with manual position reconciliation every 4 hours.

---

### GO LIVE VERDICT

**NOT READY**

Primary blockers: CRIT-1 (startup failure), CRIT-2 (no operator alerts), HIGH-1 (state corruption re-entry risk), HIGH-2 (inner loop filter asymmetry).

The infrastructure is mature and improving with each audit cycle. Fix the 4 items above and this bot clears the minimum threshold for a small-capital testnet-validated live run.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

*Report generated: 2026-05-29 | Audit version: v11 | Files read: main.py (full), risk_manager.py (full), trade_state.py (full), config_loader.py (full), decision_maker.py (prior session)*
