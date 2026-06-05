# 🏛️ TRADING BOT BUG COUNCIL — FULL REPORT v12
### BB + StochRSI Mean Reversion Scalper | Hyperliquid Perpetuals

---

## 📋 BOT SUMMARY

| Field | Value |
|-------|-------|
| Strategy | BB(20,2) + StochRSI(14,14,3,3) Mean Reversion |
| Asset class | Crypto perps + equities/commodities (xyz: HIP-3 markets) |
| Exchange | Hyperliquid only |
| Leverage | 5× (configurable, set on exchange at startup) |
| Order types | LIMIT entry, Market exit/close, Trigger (TP/SL reduce-only) |
| Position model | Per-asset state machine IDLE→ENTERED→COOLDOWN→IDLE |
| Claude role | Anomaly check only (APPROVE/REJECT on >3% moves) |
| DeFi | Not detected — Agent 6 not activated |

**Risk controls FOUND:** Position size cap, total exposure cap, concurrent position limit, daily drawdown circuit breaker, balance reserve, mandatory SL, ATR 1% sizing, force-close on loss%, per-asset SL cooldown, daily trade cap, state gate, spread gate, candle age gate, funding rate gate, regime (ADX) gate, session/weekend gate, time-based exit, trailing stop guardian, TP/SL guardian, instance lock, KILLSWITCH file, restart reconciliation.

**Risk controls MISSING / BROKEN:** Overnight session block ranges, SL price validity check, `time_limit_candles` never used, weekly drawdown halt, operator-confirmed circuit breaker reset, timing-safe auth, no rate-limit header tracking.

**Testnet support:** Present (`HYPERLIQUID_NETWORK` env var). No confirmation gate before live trading.

---

## 💀 FUND-LOSS RISKS — Fix before ANY live trading

---

### 🔴 FL-1 — MASTER_RULE VIOLATION: Time-limit exit uses hours, not candles
**Files:** `src/config_loader.py:84`, `src/main.py:989`

```python
# config_loader.py — default = 1 hour
"max_trade_hours": _get_int("MAX_TRADE_HOURS", 1),

# main.py — exit trigger
_max_hours = int(CONFIG.get("max_trade_hours") or 12)
if state_mgr.is_trade_expired(_asset_name, _max_hours):
```

MASTER_RULE 1 mandates **"Time-limit exit = 8 candles (40 min) if TP not hit."** The code uses `max_trade_hours` (default: 1 hour = 12 candles). The correctly-named `time_limit_candles` key is defined in config_loader.py but **never read anywhere in the codebase**. Trades are held 50% longer than the strategy specifies, increasing adverse-move exposure and funding cost on a mean-reversion strategy where time = risk.

**Financial consequence (5× leverage):** An extra 20 minutes of exposure at 5× on a counter-trend position where the entry signal was already borderline. Every trade stays open ~50% longer than intended by design.

---

### 🔴 FL-2 — Session gate silently broken for cross-midnight block ranges
**File:** `src/strategy.py:183`

```python
if start_block <= h < end_block:   # fails when start_block > end_block
    return False, ...
```

Python's chained comparison `22 <= h < 6` evaluates as `22 <= h AND h < 6` — always False. Setting `SESSION_BLOCK_START_UTC=22`, `SESSION_BLOCK_END_UTC=6` to block 22:00–06:00 UTC produces **zero blocking**. The default (0–6) works only because start < end. Any operator who tries to configure a late-evening to early-morning block (the most common low-liquidity window) has no protection. The weekend gate still works separately.

**Financial consequence:** Trades enter during the operator's intended low-liquidity window with wide spreads and thin order books. For HIP-3 markets (TSLA, SPX, GOLD), this is the exact period where quotes disappear entirely.

---

### 🔴 FL-3 — No validation that computed SL price is positive
**Files:** `src/main.py:1712`, `src/risk_manager.py:237`

```python
_sl = round(_entry - 1.5 * _atr_5m, 6)   # can produce negative value
```

`enforce_stop_loss()` passes through any pre-computed SL without range validation:
```python
if sl_price is not None:
    return sl_price   # returns whatever was passed
```

For low-price assets with high ATR (e.g., a $0.50 token with ATR = $1.00), SL = −$1.00. Hyperliquid rejects the trigger order. The 2-retry SL placement then falls through to `market_close()` — correct behavior but wasteful (valid long signal closes immediately due to price arithmetic). For any new HIP-3 market addition, this is a silent trap.

**Financial consequence:** Valid signal → SL rejected → immediate market close at slippage → entry fee wasted.

---

### 🔴 FL-4 — `time_limit_candles` config key defined but never consumed
**Files:** `src/config_loader.py:99`, searched entire codebase

```python
"time_limit_candles": _get_int("TIME_LIMIT_CANDLES", 8),   # defined
# ... never referenced in main.py, strategy.py, or anywhere else
```

This creates a false sense of compliance with MASTER_RULE 1. An operator reading the `.env` file sees `TIME_LIMIT_CANDLES=8` and believes 8-candle exits are active. They are not. The actual exit gate uses `MAX_TRADE_HOURS`. Dead config keys are operational landmines.

---

## 🔴 CRITICAL — Fix before live trading

---

### C-1 — TP1/TP2 placed at identical price — guardian's tp1_hit logic causes partial-close confusion
**File:** `src/main.py:1713-1722, 2006-2016`

```python
_tp1 = _tp   # SAME price
_tp2 = _tp   # SAME price
```

Then two half-size TP orders are placed at the same price. The guardian's `tp1_hit` detection fires when the TP1 order disappears from open orders while the position still exists. Since TP1 and TP2 are at the same price, Hyperliquid may fill one before the other in a fast move. When TP1 fills first:
- `tp1_hit = True`, `amount` halved
- Guardian re-places TP1 (now at same price as live TP2)
- Result: 3 half-size TP orders in flight for a position that should only have 1 full-size TP

MASTER_RULE 2 says **"single exit — no partial closes."** The split-then-same-price architecture contradicts this at the implementation level while attempting to honor it at the conceptual level.

**Financial consequence:** Duplicate TP orders can overshoot the reduce-only budget, causing one to be rejected silently, leaving the position partially unprotected.

---

### C-2 — StochRSI K uses forming candle while price comparison uses last closed candle
**File:** `src/strategy.py:62-78`

```python
k_vals = [v for v in sr_data["k"] if v is not None]
k_cur  = k_vals[-1]   # forming candle's K (may repaint)
# ...
current_price = float(candles_5m[-2]["close"])  # last CLOSED candle
```

The BB band touch is checked against the last closed candle's price (correct). But the hook detection (`k_cur > k_prev`) uses the **forming candle's** StochRSI K, which can repaint before the candle closes. A 10%-complete candle that temporarily shows oversold bounce triggers a LONG entry using the previous closed candle's price. If the forming candle closes without confirming the hook, the signal was false. The 30%-candle-completion gate (line 1872) reduces (but does not eliminate) this risk.

**Financial consequence (5× leverage):** False signal entry at the BB lower band against a still-falling market. SL placed at 1.5×ATR below entry; position closes at SL.

---

### C-3 — Inner loop SL placement retries once; outer loop retries twice
**File:** `src/main.py:2660-2671`

```python
# INNER LOOP: single try, market-close on any failure
if _i_can_place_tpsl and _iout.get("sl_price"):
    try:
        _isl_res = await hyperliquid.place_stop_loss(...)
        _isl_oid = ...
    except Exception as _isl_err:
        logging.critical("[INNER SL FAIL]...")
        await hyperliquid.market_close(_ia)   # immediate close
```

The outer loop (lines 2026-2059) retries SL placement twice with a 2-second pause. The inner loop retries zero times — any transient API error (connection reset, 429, timeout) forces immediate market close at slippage. Over 11 inner ticks per hour × 3 assets, any momentary API blip causes unnecessary forced exits.

**Financial consequence:** Valid entries with clean signals are market-closed at slippage due to single-attempt SL placement. Fee round-trip on entry + market-close exit = 0.09% wasted per spurious close.

---

### C-4 — `state_mgr.record_entry()` called before diary write — crash window creates ENTERED state with no guardian coverage
**File:** `src/main.py:2096-2134`

```python
state_mgr.record_entry(asset)    # line 2096 — state.json = ENTERED
save_active_trades(active_trades) # line 2097
# ... diary write at line 2108 (later)
```

If the process crashes between line 2097 and 2108:
- `state.json` = ENTERED (blocks new entry) ✓
- `active_trades.json` has the trade ✓
- `diary.jsonl` has no entry ✗
- Guardian falls back to live-price SL (line 1094–1107) ✓ — SL placed
- `diary_index` has no entry ✗ — trailing stop guardian cannot trail

The position is protected by the fallback SL but is invisible to the trailing stop. It runs without stop advancement until the time-based exit or manual intervention.

---

### C-5 — `diary.jsonl` read via `readlines()` on every outer cycle — full file in memory
**File:** `src/main.py:880-887`

```python
with open(diary_path, "r") as f:
    lines = f.readlines()   # reads ENTIRE file into memory
    for line in lines[-10:]:
```

File rotates at 50MB. Just before rotation: every outer loop cycle loads 50MB of JSONL into RAM to access the last 10 lines. At one 60-minute outer cycle per hour this is 50MB × 24 = 1.2GB of memory allocation per day in the worst case. On a VPS with 1–2GB RAM, this causes OOM errors, crashing the bot mid-trade.

**Financial consequence:** Bot crash while positions are open. Guardian and trailing stop stop running. Positions drift with only exchange-side SL protection.

---

## 🟠 HIGH — Fix before real money exposure

---

### H-1 — `enforce_stop_loss()` passes through any pre-computed SL without direction or floor validation
**File:** `src/risk_manager.py:237`

```python
def enforce_stop_loss(self, sl_price, entry_price, is_buy, atr14=None):
    if sl_price is not None:
        return sl_price   # unconditional pass-through
```

For a LONG at entry $100 with SL at $103 (wrong side — above entry), `enforce_stop_loss` returns $103 silently. The TP/SL validity check upstream only checks TP edge, not SL direction. An SL above entry for a long would trigger immediately on entry, force-closing the position within the first tick.

**Financial consequence:** Immediate SL trigger after entry → unnecessary round-trip fee → repeated false entries if the signal persists.

---

### H-2 — PnL calculation charges taker fee on limit (maker-fee=0%) entries
**File:** `src/main.py:607`

```python
_fee = (entry_price + exit_price) * qty * 0.00045
```

Hyperliquid maker fee = 0% for GTC limit orders (the bot's entry order type). The formula charges 0.045% on the entry notional incorrectly. PnL is understated by `entry_price × qty × 0.00045` per trade. Over 40 trades/day × $50 average notional: ~$0.90/day in phantom fees artificially reducing stats. The Sharpe ratio shown on the dashboard is understated, and the daily P&L report misleads the operator into thinking the bot is less profitable than it is.

---

### H-3 — Circuit breaker auto-resets at UTC midnight with no operator confirmation
**File:** `src/risk_manager.py:95`, `src/main.py:2207-2210`

```python
self.circuit_breaker_active = False   # silent reset at midnight
```

An alert is sent but the bot **immediately resumes trading** without waiting for operator acknowledgment. If the 12% daily loss was caused by a systematic bug (wrong signal, data corruption, API misread), the bot will lose another 12% the next day. The circuit breaker is the last line of defense; auto-reset with only a Telegram message negates its protection for operators who aren't watching 24/7.

**Financial consequence:** Systematic losses can compound across multiple days. At 5× leverage, daily 12% drawdown × N days = potential account wipeout without intervention.

---

### H-4 — `.gitignore` not found in project — private key may be in version control
**Observed:** No `.gitignore` in directory listing.

The project root contains `.env` with live `HYPERLIQUID_PRIVATE_KEY`, `ANTHROPIC_API_KEY`, wallet credentials, and Telegram bot tokens. Without `.gitignore`, any `git add .` or `git init` commits these files. If the repo is ever pushed to GitHub, GitLab, or any hosted service — even private — the private key is at risk.

**Financial consequence:** Full wallet drain if key is discovered. Hyperliquid private key = full fund access.

---

### H-5 — API server defaults to `0.0.0.0` in config_loader.py
**File:** `src/config_loader.py:76`

```python
"api_host": _get_env("API_HOST", "0.0.0.0"),
```

Default is all-interfaces, not loopback. The startup warning only fires if `DASHBOARD_TOKEN` is absent AND `API_HOST=0.0.0.0`. A user who sets `DASHBOARD_TOKEN` gets no warning but still exposes the dashboard on all network interfaces. The correct secure default is `127.0.0.1`.

---

### H-6 — `active_trades.json` corruption or deletion leaves ghost positions undetected
**File:** `src/trade_state.py:36-43`, `src/main.py:910-929`

The reconciler removes active_trades entries when neither position nor orders exist on the exchange. It does NOT add entries when the exchange has a position but active_trades doesn't (ghost position after `active_trades.json` deletion). The guardian handles state=ENTERED assets in `args.assets`, but if `active_trades.json` is empty after deletion, the trailing stop guardian skips all assets. The fallback SL placement (line 1083–1107) runs, so protection exists, but without TP or trailing stop.

---

### H-7 — No weekly/total drawdown halt — daily 12% loss possible every day indefinitely
**File:** `src/risk_manager.py` — absent

There is no cumulative stop. A bot losing 12% per day for 5 consecutive days halts at midnight each day and resumes the next. No alert escalates after N consecutive daily breaker trips. No weekly or total-account drawdown limit.

**Financial consequence at 5× leverage:** A bot with systematic issues can lose 60%+ of account value over a week while the operator sees only daily alerts.

---

### H-8 — `_log_trade_close()` hardcodes fee rate instead of reading from CONFIG
**File:** `src/main.py:607`

```python
_fee = (entry_price + exit_price) * qty * 0.00045   # hardcoded
```

CONFIG has `taker_fee_pct` (read correctly in `_update_stats()`). If Hyperliquid changes its fee structure, `_update_stats()` would pick it up from .env but `_log_trade_close()`'s PnL formula would remain wrong silently.

---

### H-9 — Outer loop `state` (account/positions) is up to 60+ seconds stale at execution time
**File:** `src/main.py:771, 1890`

`state` is fetched once at line 771, then used for risk checks (position count, balance) at line 1890. The data-gather phase (27 API calls through semaphore) takes several seconds. During a volatile period, a concurrent liquidation or forced close could change the position count, causing the risk manager to allow one more position than safe. Inner loop refreshes `state` correctly (line 2399), but the outer loop execution block does not re-verify immediately before order submission.

---

## 🟡 EDGE CASE — Fix before scale

---

### E-1 — Cross-midnight `_today_utc` not refreshed in inner loop
**File:** `src/main.py:2336`

```python
_save_daily_count(_daily_trade_count, str(_today_utc))
```

`_today_utc` is set in the outer loop (line 757) and not updated in the inner loop (lines 2155+). A trade that fills in the inner loop after UTC midnight is saved with the previous day's date. On the next outer cycle, the date mismatch resets the counter to 0. The net effect is that a midnight inner-tick trade doesn't count against either day's cap.

---

### E-2 — Timeout/guardian cooldowns hardcoded to 3600s, ignoring COOLDOWN_MINUTES
**Files:** `src/main.py:929, 1015, 1045`

```python
state_mgr.start_cooldown(_asset_name, interval_seconds=3600)   # hardcoded
```

Lines 862 and 2387 correctly read `CONFIG.get("cooldown_minutes")`, but timeout-exit and guardian cooldowns use 3600 seconds unconditionally. An operator who sets `COOLDOWN_MINUTES=15` expects 15-minute cooldowns everywhere. Timeout and guardian cooldowns are 4× longer than expected, blocking valid signals for 45 extra minutes.

---

### E-3 — `check_losing_positions()` skips force-close when price lookup returns 0 (pnl_unknown)
**File:** `src/main.py:839-848`, `src/trading/hyperliquid_api.py:643-645`

When the price lookup fails after all retries, `pnl_unknown=True` and `pnl=0.0`. `check_losing_positions()` sees `pnl=0` (not negative) and doesn't trigger force-close. The alert is sent (line 844) and the operator is notified, but the position remains unprotected by the force-close layer. Only the exchange-side SL trigger protects it.

**Financial consequence:** If price also fails on Hyperliquid's side (e.g., maintenance), the SL trigger may not execute, leaving the position open indefinitely.

---

### E-4 — `diary.jsonl` fill lookup uses last 50 fills — missed fills produce null PnL in diary
**File:** `src/main.py:557`

```python
_fills = await hyperliquid.get_recent_fills(limit=50)
```

High-frequency operation (40 trades/day × 3 assets = 120 fills/day) means fills older than ~20 minutes may not appear in the last 50. When the bot's fill loop misses a closing fill, `exit_price=None` and `realized_pnl=None` are written to diary.jsonl. Stats and Sharpe ratio calculations silently skip these trades.

---

### E-5 — `stoch_rsi()` K/D series padding length may mismatch candle count for non-standard periods
**File:** `src/indicators/local_indicators.py:232-239`

```python
pad_k = max(0, len(rsi_vals) - len(k_line))
full_k = ([None] * pad_k) + k_line
full_k = full_k[:len(rsi_vals)]   # hard trim
```

The hard trim `[:len(rsi_vals)]` silently discards values if `k_line` is longer than `rsi_vals` due to unusual period combinations. This produces a K series that's shorter than expected, causing the `k_vals[-1]` / `k_vals[-2]` lookup in strategy.py to use the wrong bar. Only occurs with non-default StochRSI periods where smooth periods exceed RSI period.

---

### E-6 — `candle_gate` uses `datetime.now()` called twice — minute rollover race
**File:** `src/main.py:1871`

```python
_secs_into_5m = (datetime.now(timezone.utc).minute % 5) * 60 \
                + datetime.now(timezone.utc).second
```

Two separate `datetime.now()` calls. If the minute rolls from 4 to 5 (or 9 to 10, etc.) between the two calls, `minute % 5` returns 0 from the second call, producing `_secs_into_5m = 0 + seconds`. This would pass the 30% gate immediately after a candle boundary — the exact moment of maximum repaint risk.

---

### E-7 — `check_balance_reserve()` reserve floor doesn't scale with account growth
**File:** `src/risk_manager.py:191-201`

```python
min_balance = initial_balance * (self.min_balance_reserve_pct / 100.0)
```

If account grows from $100 → $500, reserve floor stays at $20 (20% of $100). At $500, a $20 reserve is only 4%. A 4% cushion at 5× leverage means the account can be wiped out by a single bad trade before the reserve gate fires.

---

### E-8 — No maximum order size check against asset minimum lot size for small accounts
**File:** `src/main.py:1906`

```python
amount = alloc_usd / current_price
```

After ATR sizing, if ATR cap produces a valid allocation below the exchange minimum ($10), the risk manager bumps to $11 (line 424). But `round_size()` then applies `szDecimals` rounding. For high-precision assets (szDecimals=8), the rounded amount × price may still be below $10 after rounding. Hyperliquid rejects the order silently.

---

## 🔐 SECURITY FINDINGS

**CRITICAL:**
- **SEC-1** — `.gitignore` not present. Private key in `.env` and wallet state in `active_trades.json`, `state.json`, `diary.jsonl` are all committable. Any `git push` exposes wallet credentials.

**HIGH:**
- **SEC-2** — `config_loader.py:76` defaults `API_HOST=0.0.0.0`. Dashboard exposed on all network interfaces by default. Startup warning only fires when `DASHBOARD_TOKEN` is absent, not when it's set but interface is open.
- **SEC-3** — No rate limiting on dashboard auth endpoint. `DASHBOARD_TOKEN` can be brute-forced if dashboard is public-facing.

**MEDIUM:**
- **SEC-4** — `defusedxml` is optional; stdlib `xml.etree` fallback allows XML bomb attacks from compromised RSS feeds. Install `defusedxml` as a hard requirement.
- **SEC-5** — `DASHBOARD_TOKEN` compared with `!=` (not timing-safe). Use `hmac.compare_digest()` to prevent timing oracle attacks.
- **SEC-6** — `CONFIG["hyperliquid_private_key"]` string remains in memory after wallet construction. Not cleared after `Account.from_key()`.

---

## ⚡ PERFORMANCE FINDINGS

**HIGH:**
- **PERF-1** — `diary.jsonl` `readlines()` (line 880) reads the entire file every outer cycle. File rotates at 50MB — potential 50MB read every 60 minutes. Use `open().seek(0, 2)` + read backwards to get last N lines.

**MEDIUM:**
- **PERF-2** — 27+ API calls per outer cycle (3 assets × 9 calls each) through `_read_semaphore(4)` = 7 sequential waves. At 200ms per call, outer cycle minimum is ~1.4s. Acceptable for 5m interval but grows with asset count. Consider caching OI and funding (they change slowly).
- **PERF-3** — `compute_all()` called for 5 timeframes per asset (15 calls per outer cycle) recomputes MACD, ADX, and VWAP from scratch every time. MACD and ADX are incremental — recompute only the last value from the previous result.
- **PERF-4** — `get_meta_and_ctxs()` has a 6-hour TTL but `get_funding_rate()` and `get_open_interest()` call it per-asset per-cycle (not in parallel). Already fetched once in the `asyncio.gather()` block; the subsequent spread calculation (line 1473) calls it again.

**LOW:**
- **PERF-5** — `save_active_trades()` called 5+ times per inner tick per trade (trailing stop stages each call). On SSD this is fine; on NFS or slow disk, I/O accumulates.
- **PERF-6** — No explicit rate-limit header tracking. 429 responses fall into the generic retry path without specific cooldown. Under high load (many assets), rate limits cause cascading retry storms.

---

## 🛡️ MISSING RISK CONTROLS

| Control | Status | Financial Risk |
|---------|--------|----------------|
| SL price > 0 validation | ❌ Missing | SL rejected → market close at slippage |
| SL on correct side of entry | ❌ Missing | Wrong-side SL triggers immediately |
| Weekly/total drawdown halt | ❌ Missing | Systematic losses compound daily |
| Operator gate on circuit breaker reset | ❌ Missing | Auto-resumes after 12% loss each day |
| Overnight session block (start > end) | ❌ Broken | Trades in intended blackout hours |
| `time_limit_candles` enforcement | ❌ Dead config | Trades held 50% longer than intended |
| Rate limit header tracking | ❌ Missing | Critical exit blocked during 429 storm |
| Minimum viable account size check | ❌ Missing | $11 minimum bump overrides 1% risk rule |

---

## 📊 MONITORING & OPERATIONAL GAPS

- **MON-1** — No alert when bot has not traded in N consecutive days (silent idle state).
- **MON-2** — No alert when N consecutive outer cycles complete with zero signals (possible feed issue vs. market silence).
- **MON-3** — No weekly PnL summary — only per-cycle Sharpe from diary.
- **MON-4** — `llm_requests.log` rotation was missing for prompts.log and llm_requests.log paths until the PERF-v10-8 fix; verify both rotate correctly on the current deployment.
- **MON-5** — KILLSWITCH requires filesystem access (SSH/SCP to server). No Telegram `/kill` command. Operators without server access cannot halt the bot remotely.
- **MON-6** — No alert when `consecutive_failures` circuit breaker fires — only sleeping 5 minutes silently (alert is sent at line 786, but if Telegram is not configured, it's invisible).

---

## 🏦 EXCHANGE-SPECIFIC ISSUES

- **EX-1** — `_trigger_order_retry()` idempotency check finds ANY trigger of the same type for the asset, not the specific order. If a trailing stop update leaves a briefly-orphaned old SL, the idempotency guard prevents placing the new SL. The old SL at the wrong price remains active.

- **EX-2** — `cancel_all_orders()` fetches orders then cancels one-by-one sequentially. For an asset with 3–4 trigger orders (TP1, TP2, SL, trailing SL), cancellation takes 3–4 sequential API calls during the already-time-sensitive KILLSWITCH path.

- **EX-3** — `TAKER_FEE_PCT = 0.045` in `RiskManager` class (line 226) is ambiguous — it looks like 4.5% but means 0.045% (represented as a percent value, not a decimal). The `get_risk_summary()` returns this as `0.045`, and the dashboard might display it as 0.045% (correct visually) or 4.5% depending on display formatting. No decimal/percent confusion in the math, but the naming causes maintenance risk.

- **EX-4** — `spot_usdc` is included in `total_value` (line 608 in hyperliquid_api.py). USDC in the spot wallet cannot be used for perp margin without a manual transfer. Including it inflates the available balance seen by the risk manager, potentially allowing a trade that the perp wallet can't actually margin.

---

## 📈 BACKTEST VALIDITY ISSUES

- **BT-1** — `time_limit_candles` = 8 is the documented exit rule but is never used. Any backtest using `max_trade_hours=1` will show slightly different results than a candle-count-based exit. The two are close but not identical (hours don't align with candle boundaries).
- **BT-2** — PnL calculation charges taker fee on limit entries (maker fee = 0%). Backtested results will show lower profitability than live (overcounting fees).
- **BT-3** — No formal backtest engine in the codebase. `signals.jsonl` tracks signal-level data but no fill simulation exists for historical validation.

---

## 📦 TECHNICAL DEBT

- **TD-1** — `time_limit_candles` config key + `TIME_LIMIT_CANDLES` env var should replace `max_trade_hours` entirely, or be documented as the candle-count equivalent.
- **TD-2** — TP1/TP2 at identical price is architecturally confusing. If single-exit is the design, place one full-size TP and remove the partial-close infrastructure (`tp1_hit`, `half_size`, `tp1_oid`, `tp2_oid`).
- **TD-3** — `TAKER_FEE_PCT = 0.045` class constant should be renamed `TAKER_FEE_PCT_PERCENT = 0.045` or converted to decimal `TAKER_FEE_DECIMAL = 0.00045` to avoid confusion.
- **TD-4** — `score: 8.0` placeholder in trade output (lines 1794, 2507) is a remnant of the old scoring system. Dead field, should be removed.
- **TD-5** — Inner loop is 340+ lines of duplicated logic from the outer loop (same signal pipeline, same execution block, same TP/SL placement). Should be extracted to a shared async function.
- **TD-6** — `near_ema_15m` and `near_ema` are computed and passed in market data but are never used as a gate condition in the current BB+StochRSI pipeline. Dead calculation.
- **TD-7** — `adx_trending` field in market data uses threshold 15 (line 1548) but regime gate uses `TREND_PAUSE_ADX=30`. The field is unused in the trade pipeline — dead computation.

---

## 🔨 STEP 6 — PEER REVIEW

### 🔨 LOGIC BREAKER reviews others:

Most critical finding from Security: **SEC-1 (no .gitignore)** — if the private key is in git history, no amount of trading-logic fixes matters. Complete fund loss risk that exists entirely outside the runtime.

Most critical finding from Chaos: **EX-4 (spot USDC in balance)** — this is a subtle sizing bug. The risk manager sizes positions against a balance that includes USDC that cannot actually be used for perp margin without a transfer. The bot can attempt to open a position with more margin than available, leading to rejected orders or forced liquidation at adverse prices.

Issue ALL agents missed: **The inner loop's `asyncio.to_thread(agent.claude_anomaly_check, ...)` call blocks the event loop's thread pool for up to 15 seconds per asset with an anomaly** (timeout=15.0 in decision_maker.py). With 3 assets all triggering anomalies simultaneously, the inner loop could block for 45 seconds before proceeding to the next tick. During those 45 seconds, no position monitoring, trailing stop updates, or reconciliation runs. The SL/TP guardian does not fire. For a leveraged position in a fast-moving market, 45 seconds of monitoring blindness is significant.

**Refuse-to-trade issue:** The session gate bug (FL-2) combined with the time-limit violation (FL-1) means the bot can enter trades in blackout hours AND hold them 50% longer than intended. These two interact to create maximum adverse-move exposure at exactly the worst market times.

---

### 🔐 SECURITY ANALYST reviews others:

Most critical from Logic Breaker: **C-2 (StochRSI K repaint)** — from a security-of-operation perspective, this is the most insidious. False signals that look valid at the time of evaluation but fail after candle close systematically drain the account through repeated SL exits. Not exploitable externally, but causes systematic financial loss at the strategy level.

Most critical from Performance: **PERF-1 (full diary.jsonl read)** — 50MB reads every 60 minutes can cause OOM, crashing the bot mid-trade with positions open. From a fund-safety standpoint, this is a reliability vulnerability with direct financial consequence.

Issue ALL agents missed: **The Telegram `send_alert()` (line 730) is called with `await` from within the KILLSWITCH close loop.** If Telegram is unreachable (network issue), the `await send_alert()` will timeout after 10 seconds. During KILLSWITCH execution where milliseconds matter for position protection, a 10-second Telegram timeout causes the remaining close attempts to be delayed. Critical alert paths should use non-blocking or fire-and-forget delivery.

---

### 🌪️ CHAOS TESTER reviews others:

Most critical from Logic Breaker: **FL-3 (negative SL price)** — for operators adding HIP-3 markets (xyz:GOLD was listed at ~$2600, ATR could be large in proportion), this could silently market-close every entry attempt, draining fees without the operator noticing why.

Most critical from Security: **EX-4 (spot USDC inflates balance)** — chaos scenario: account has $200 in spot USDC (undeployed capital), $100 in perp margin. Risk manager sees $300, sizes a position at 15% of $1500 buying power = $225. The perp wallet only has $100. Hyperliquid either rejects the order or allows it with liquidation risk at much smaller adverse move than expected.

Issue ALL agents missed: **The `_inner_positions` variable in the inner loop (line 2361) is fetched inside the inner reconcile try/except but also used (if already populated) by the pending-limit-cancel block (line 2312). If the inner reconcile's `get_positions()` succeeds but returns stale data (Hyperliquid read from a non-authoritative node), the pending-limit-cancel could falsely confirm a position as open/closed, either missing a cancel or premature clearing of the entry_oid.**

---

### ⚡ PERFORMANCE ENGINEER reviews others:

Most critical from Logic Breaker: **C-5 (diary.jsonl full read)** — confirmed OOM risk. 50MB `readlines()` allocates a Python list in RAM. With async, this blocks the event loop thread for the duration of the read (file I/O isn't awaited — it's synchronous `open()`). During the 50ms+ blocking read, no other coroutines run, including order-placement confirmations.

Most critical from Operator Safety: **H-7 (no weekly drawdown halt)** — operationally, the daily circuit breaker creates a false safety signal. "The bot has circuit breakers" is true but misleading: they reset automatically every midnight.

Issue ALL agents missed: **The trailing stop guardian (lines 1191-1273) calls `hyperliquid.place_stop_loss()` and `hyperliquid.cancel_order()` for EVERY active trade on EVERY outer loop cycle when trail conditions are met.** With 3 active trades all moving favorably, the guardian fires 6 API calls (3 place + 3 cancel) before any other outer-loop processing. These calls are sequential (not gathered). Add the SL cancel-then-place pattern: each trail update uses 2 sequential calls. At 200ms per call × 6 calls = 1.2 seconds of blocking API calls at the start of every outer cycle when positions are profitable. No timeout is enforced on individual trail updates.

---

### 👤 OPERATOR SAFETY reviews others:

Most critical from Logic Breaker: **FL-1 (time_limit_candles never used)** — this is both a rule violation and an operational trust issue. CLAUDE.md, MASTER_RULES.md, and the .env all say "8 candles." The dashboard reports entries and exits. An operator watching their P&L and seeing trades held 12 candles would not know the 8-candle rule is being violated until they correlate log timestamps manually.

Most critical from Security: **SEC-1 (no .gitignore)** — if the project was initialized with git and .env was ever committed, it must be treated as compromised regardless of whether the remote is private.

Issue ALL agents missed: **There is no audit trail for KILLSWITCH activation.** When the KILLSWITCH file is detected and positions are closed, the diary.jsonl `trade_closed` events are NOT written (the close paths at lines 710-712 call `market_close` and `cancel_all_orders` but never call `_log_trade_close()`). KILLSWITCH exits are invisible to the trade history, Sharpe calculation, and stats.json. An operator reviewing weekly performance after a KILLSWITCH event will see a gap in their trade records with no explanation.

---

## FINAL VERDICT

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
     🏛️  TRADING BOT BUG COUNCIL — FINAL REPORT v12
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 BOT: BB+StochRSI Mean Reversion | Hyperliquid Perps
   5× leverage | LIMIT entry | Trigger SL/TP exits
   Risk controls: 20+ found | 8 missing/broken
   DeFi: Not active

💀 FUND-LOSS (4 issues):
   FL-1  MASTER_RULE VIOLATION: time_limit_candles never used
         → trades held 50% longer than 8-candle rule specifies
   FL-2  Session gate broken for overnight block ranges
         → silent trading during intended blackout hours
   FL-3  No SL price > 0 validation
         → SL rejection cascades to forced market close
   FL-4  time_limit_candles config key is dead code
         → operator trust violation

🔴 CRITICAL (5 issues):
   C-1   TP1/TP2 at same price + tp1_hit logic → duplicate TP orders
   C-2   StochRSI K uses forming candle → partial repaint risk
   C-3   Inner loop SL: 0 retries vs outer loop's 2 retries
   C-4   record_entry before diary write → crash window
   C-5   diary.jsonl full readlines() → OOM crash risk

🟠 HIGH (9 issues):
   H-1   enforce_stop_loss passes any SL without validation
   H-2   Fee formula charges taker fee on maker (limit) entries
   H-3   Circuit breaker auto-resets at midnight, no operator gate
   H-4   No .gitignore — private key committable
   H-5   API host defaults to 0.0.0.0 (all interfaces)
   H-6   active_trades.json deletion leaves ghost positions
   H-7   No weekly/total drawdown halt
   H-8   Fee rate hardcoded in PnL formula (not config-driven)
   H-9   Outer loop account state stale at execution time

🟡 EDGE CASE (8 issues):
   E-1   _today_utc not refreshed in inner loop at midnight
   E-2   Timeout/guardian cooldowns hardcoded 3600s
   E-3   check_losing_positions skips pnl_unknown positions
   E-4   get_recent_fills limit=50 may miss closing fills
   E-5   stoch_rsi hard-trim may clip valid K values
   E-6   Candle gate double datetime.now() rollover race
   E-7   Balance reserve floor doesn't scale with account growth
   E-8   No check for sub-minimum-lot allocation after rounding

🔐 SECURITY: Critical(1) High(2) Medium(3)
   SEC-1 CRITICAL: No .gitignore — .env may be in git history
   SEC-2 HIGH: Default API_HOST=0.0.0.0 exposes dashboard
   SEC-3 HIGH: No dashboard auth rate limiting

⚡ PERFORMANCE: High(1) Medium(3) Low(2)
   PERF-1 diary.jsonl full file read every cycle (50MB → OOM)

🛡️ MISSING RISK CONTROLS (8):
   SL price validation, SL direction validation, weekly drawdown
   halt, operator gate on circuit breaker reset, overnight session
   block (cross-midnight), time_limit_candles enforcement,
   rate-limit header tracking, post-rounding minimum size check

📊 MONITORING GAPS (6):
   No trade-silence alert, no consecutive-signal alert,
   KILLSWITCH exits not logged to diary.jsonl,
   Telegram blocking during KILLSWITCH close sequence,
   anomaly check blocks event loop up to 45s with 3 assets,
   no weekly PnL summary

🏦 EXCHANGE ISSUES (4):
   EX-1 Idempotency check blocks new SL if orphaned old SL exists
   EX-2 cancel_all_orders sequential during KILLSWITCH
   EX-3 TAKER_FEE_PCT = 0.045 ambiguous naming (% vs decimal)
   EX-4 spot_usdc inflates balance — not usable for perp margin

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COUNCIL VERDICT: NOT READY FOR LIVE TRADING

PRIMARY CONCERN: 4 fund-loss risks (FL-1 through FL-4) include
a MASTER_RULE violation (time_limit_candles never enforced) and
a silent session gate failure. The bot's documented behavior
does not match its actual behavior in two independently critical
respects. These must be fixed before any capital deployment.

TOP 3 MUST-FIX (in order):
  1. FL-1: Wire time_limit_candles into the time-based exit logic
     (replace max_trade_hours or convert to equivalent candles)
  2. FL-2: Fix session_gate_ok() for cross-midnight block ranges
     using modular hour arithmetic
  3. H-4: Add .gitignore immediately. Treat .env as compromised
     if any git history exists. Rotate all keys.

CAPITAL RECOMMENDATION: $0 live capital until FL-1, FL-2, H-4
are resolved. After those: $50–100 paper trade equivalent on
testnet for 48 hours minimum to validate the corrected exits.

GO LIVE: ❌ NOT READY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

*Report generated: 2026-06-04 | Council version: v12 | Agents: Logic Breaker, Security Analyst, Chaos Tester, Performance Engineer, Operator Safety*
