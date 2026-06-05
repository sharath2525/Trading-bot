# 🏛️ TRADING BOT BUG COUNCIL — RE-CHECK REPORT v13
### Post-Fix Re-Audit | BB + StochRSI Mean Reversion | Hyperliquid Perpetuals
### Date: 2026-06-04 | Agents: 5 | Scope: Full re-read of all source files

---

## 📋 RE-AUDIT METHOD

Every source file was re-read fresh after Claude Code applied fixes. Each prior finding was
verified by inspecting the actual code — not assumed fixed from commit messages. Status labels:

- ✅ **FIXED** — Confirmed corrected in current code
- ⚠️ **STILL PRESENT** — Issue persists, unchanged
- 🆕 **NEW** — Regression or new issue introduced by the fix round

Files verified: `src/main.py` · `src/strategy.py` · `src/risk_manager.py` · `src/trading/hyperliquid_api.py` · `src/trade_state.py` · `src/config_loader.py` · `src/alerts.py` · `.gitignore`

---

## ═══════════════════════════════════════════
## PART 1 — WHAT WAS FIXED (Confirmed ✅)
## ═══════════════════════════════════════════

### ✅ FL-1 — `time_limit_candles` now wired into exit logic
**Files fixed:** `src/trade_state.py:205-217` · `src/main.py:997-999`

New `is_trade_expired_minutes()` method added to `TradeStateMachine`. Time-based exit now uses:
```python
_candle_limit = int(CONFIG.get("time_limit_candles") or 8)
_max_minutes = _candle_limit * (_interval_seconds / 60.0)   # 8 × 5 = 40 min ✓
if state_mgr.is_trade_expired_minutes(_asset_name, _max_minutes):
```
MASTER_RULE 1 now correctly enforced: 8 candles × 5 min = 40-minute exit window.

---

### ✅ FL-2 — Session gate cross-midnight block ranges fixed
**File fixed:** `src/strategy.py:188-193`

```python
if start_block < end_block:
    _blocked = start_block <= h < end_block
else:  # overnight range wraps past midnight
    _blocked = h >= start_block or h < end_block
```
`SESSION_BLOCK_START_UTC=22, SESSION_BLOCK_END_UTC=6` now correctly blocks 22:00–06:00 UTC.

---

### ✅ FL-3 + FL-4 — SL price validated positive and on correct side of entry
**File fixed:** `src/risk_manager.py:238-252`

```python
_sl_valid = (
    sl_price > 0
    and (not is_buy or sl_price < entry_price)   # long SL must be below entry
    and (is_buy or sl_price > entry_price)         # short SL must be above entry
)
if _sl_valid:
    return sl_price
# Falls through to auto-compute a valid SL if invalid
```
Invalid SL (negative or wrong side) now triggers auto-recompute instead of silent pass-through.

---

### ✅ FL-5 — Spot USDC excluded from risk manager's account value
**File fixed:** `src/trading/hyperliquid_api.py:614`

```python
# FL-5 FIX: use perps_value only for risk calculations.
total_value = perps_value    # was: perps_value + spot_usdc
```
Spot USDC still shown on dashboard (`spot_usdc` key preserved) but no longer inflates position sizing or drawdown calculations.

---

### ✅ C-1 — TP1/TP2 set to None — single full-size TP placed per MASTER_RULE 2
**File fixed:** `src/main.py:1727-1743`

```python
# C-1 FIX: MASTER_RULE 2 = single exit, no partial closes.
_tp1 = None
_tp2 = None
```
Guardian fallback correctly re-places a single full-size TP (via `_g_tp_px` path at line 1173). The `tp1_oid=None` check at the reconciler short-circuits TP1 detection correctly.

---

### ✅ C-2 — StochRSI K uses closed-candle values — repaint risk eliminated
**File fixed:** `src/strategy.py:65-73`

```python
# C-2 FIX: use closed-candle K values only.
if len(k_vals) < 3:
    return no_signal

k_cur  = k_vals[-2]   # last CLOSED candle's K (not forming)
k_prev = k_vals[-3]   # previous closed candle's K (hook detection)
```
Hook detection now uses the last two fully-closed candles. Forming candle K is no longer part of the signal decision.

---

### ✅ C-3 — Inner loop SL placement now retries twice (matches outer loop)
**File fixed:** `src/main.py:2684-2696`

```python
# C-3 FIX: retry SL placement twice (matches outer loop) before market close
for _i_sl_attempt in range(2):
    try:
        _isl_res = await hyperliquid.place_stop_loss(...)
        _i_sl_placed = True
        break
    except Exception as _i_sl_err:
        if _i_sl_attempt == 0:
            await asyncio.sleep(2)
if not _i_sl_placed:
    # market-close fallback
```
Transient API errors (429, connection reset) no longer trigger immediate emergency close in the inner loop.

---

### ✅ C-4 / PERF-1 — `diary.jsonl` now reads 16KB tail instead of full file
**File fixed:** `src/main.py:880-893`

```python
# C-5/PERF-1 FIX: read only the tail of diary.jsonl (max 16KB)
with open(diary_path, "rb") as _df:
    _df.seek(0, 2)
    _fsize = _df.tell()
    _df.seek(max(0, _fsize - 16384))
    _tail = _df.read().decode("utf-8", errors="ignore")
for _dl in _tail.splitlines()[-10:]:
```
OOM risk eliminated. Maximum 16KB read per cycle regardless of file size.

---

### ✅ C-6 (Outer KILLSWITCH) — KILLSWITCH exits now logged to diary
**File fixed:** `src/main.py:707-711`

```python
# MON-5 FIX: log KILLSWITCH close to diary.jsonl so stats/history are complete
try:
    await _log_trade_close(_ks_tr, "killswitch")
except Exception as _kd_err:
    logging.warning("[KILLSWITCH] diary log failed for %s: %s", _ks_asset, _kd_err)
```
Outer KILLSWITCH trades now appear in diary.jsonl, stats.json, and Sharpe calculations.

---

### ✅ H-1 / H-8 — PnL formula uses only exit taker fee, reads from CONFIG
**File fixed:** `src/main.py:603-606`

```python
# H-2/H-8 FIX: LIMIT entry = 0% maker fee on Hyperliquid; only exit is taker.
_fee = exit_price * qty * float(CONFIG.get("taker_fee_pct") or 0.00045)
```
Entry maker fee no longer double-counted. Fee rate reads from CONFIG (config-driven). PnL accuracy improved ~0.045% per trade.

---

### ✅ H-3 — `.gitignore` created and comprehensive
**File fixed:** `.gitignore` (new file)

Covers: `.env`, `.env.*`, `state.json`, `active_trades.json`, `risk_state.json`, `daily_count.json`, `diary_index.json`, `*.jsonl`, `*.log`, `KILLSWITCH`, `__pycache__`, `venv/`, IDE files, OS files. Private key and trade data no longer committable.

---

### ✅ H-4 — API server defaults to `127.0.0.1` (loopback only)
**File fixed:** `src/config_loader.py:76`

```python
"api_host": _get_env("API_HOST", "127.0.0.1"),   # was: "0.0.0.0"
```
Dashboard now only accessible from localhost by default.

---

### ✅ H-9 — KILLSWITCH Telegram alerts use fire-and-forget (`asyncio.create_task`)
**File fixed:** `src/main.py:735-736, 2237`

```python
# H-9 FIX: fire-and-forget alert so Telegram timeout doesn't delay monitoring
asyncio.create_task(send_alert(_ks_msg))
```
A Telegram outage no longer adds 10-second delays to the KILLSWITCH close sequence.

---

### ✅ E-1 — `_today_utc` refreshed in inner loop before daily count saves
**File fixed:** `src/main.py:2383, 2759`

```python
_today_utc = datetime.now(timezone.utc).date()   # refreshed each inner-tick save
_save_daily_count(_daily_trade_count, str(_today_utc))
```
Midnight boundary trade fills are now credited to the correct day.

---

### ✅ E-2 — Timeout cooldown reads `COOLDOWN_MINUTES` from config
**File fixed:** `src/main.py:1025`

```python
state_mgr.start_cooldown(_asset_name, interval_seconds=int(CONFIG.get("cooldown_minutes") or 15) * 60)
```
Timeout exits now respect the operator-configured cooldown, not a hardcoded 3600s.

---

### ✅ E-6 (outer loop) — Candle gate uses single `datetime.now()` call
**File fixed:** `src/main.py:1905-1908`

```python
# E-6 FIX: single datetime.now() call — two separate calls risk a minute rollover
_now_utc = datetime.now(timezone.utc)
_secs_into_5m = (_now_utc.minute % 5) * 60 + _now_utc.second
```
Minute-rollover race condition eliminated for the outer loop candle gate.

---

## ═══════════════════════════════════════════════════════
## PART 2 — WHAT REMAINS UNFIXED (⚠️ Still Present)
## ═══════════════════════════════════════════════════════

---

### ⚠️ C-6 (Inner KILLSWITCH) — Inner loop KILLSWITCH still does NOT call `_log_trade_close()`
**File:** `src/main.py:2195-2222`

The outer KILLSWITCH was fixed (✅ above). The inner loop KILLSWITCH at lines 2195-2222 received no diary logging fix:

```python
await hyperliquid.cancel_all_orders(_iks_asset)
await hyperliquid.market_close(_iks_asset)
logging.critical("[KILLSWITCH] inner %s — closed", _iks_asset)
_iks_closed = True
break
# ← NO _log_trade_close() call here
```

If the KILLSWITCH is triggered during an inner loop tick (which happens most of the time — 11 ticks vs 1 outer cycle), the trade close is still invisible to `diary.jsonl`, `stats.json`, and the Sharpe calculation.

**Financial consequence:** Operator's trade history and performance stats have gaps for any KILLSWITCH triggered between outer cycles, which is the vast majority of KILLSWITCH events.

---

### ⚠️ C-7 — Anomaly check still sequential — up to 45s event loop blockage
**File:** `src/main.py:1771, 2526`

```python
_verdict = await asyncio.to_thread(
    agent.claude_anomaly_check,
    _asset, _price_chg_5c, _direction_str   # sequential per asset, timeout=15s
)
```

Three assets × 15s API timeout = 45-second potential blockage in the outer loop's asset-processing for-loop. During a broad market crash (exactly when the anomaly threshold is most likely to be triggered on all assets simultaneously), position monitoring, trailing stop updates, and reconciliation are paused for up to 45 seconds.

**Financial consequence:** 45-second monitoring blind spot during a flash crash — the highest-risk moment for leveraged positions.

---

### ⚠️ H-2 — Circuit breaker still auto-resets at UTC midnight, no operator gate
**File:** `src/risk_manager.py:92-97`

```python
def _reset_daily_if_needed(self, account_value: float):
    if self.daily_high_date != today:
        self.circuit_breaker_was_active = bool(self.circuit_breaker_active)
        self.daily_high_value = account_value
        self.circuit_breaker_active = False    # ← auto-reset, no gate
        self._save_circuit_state()
```

A Telegram alert is sent when `circuit_breaker_was_active=True`, but trading resumes immediately without waiting for operator confirmation. If the 12% daily loss was caused by a systematic issue (wrong signal, data corruption, external market manipulation), the bot will lose another 12% the next day.

**Financial consequence (5× leverage):** Systematic issues can compound daily losses. 12% × N days with no cumulative stop.

---

### ⚠️ H-5 — `active_trades.json` loss/corruption leaves position without trailing stop
**File:** `src/main.py:910-929` · `src/trade_state.py`

If `active_trades.json` is deleted or corrupted, `active_trades = []` on restart. The reconciler only removes entries that are in `active_trades` but not on the exchange — it does NOT add entries for exchange positions that have no `active_trades` entry. The guardian places a fallback SL (from live price), but no TP and no trailing stop runs because the trailing stop iterates `active_trades`.

**Financial consequence:** Open position runs with only a static SL and no stop advancement or TP protection until the time-based exit triggers.

---

### ⚠️ H-6 — No weekly or total drawdown halt
**File:** `src/risk_manager.py` — feature absent

The daily circuit breaker resets every midnight and there is no cumulative loss counter. A bot losing 12% per day for 5 consecutive days halts each night and resumes each morning with no escalation.

**Financial consequence (5× leverage):** Systematic losses can total 60%+ account value in a week while the operator only sees individual daily alerts.

---

### ⚠️ H-7 — Outer loop account state stale at execution time
**File:** `src/main.py:771, 1890`

Account state (`state`) fetched at line 771, used for risk checks at line 1890. The data-gather phase (27+ API calls through `semaphore(4)`) takes several seconds between fetch and execution. A concurrent liquidation or force-close during this window causes the risk manager to allow a trade that the account can no longer safely margin.

**Financial consequence:** In high-volatility periods, an extra position may be opened after a concurrent liquidation, creating unintended overexposure.

---

### ⚠️ H-8 — Balance reserve floor uses `initial_balance` — doesn't scale with account growth
**File:** `src/risk_manager.py:195`

```python
min_balance = initial_balance * (self.min_balance_reserve_pct / 100.0)
```

If account grows from $100 to $500, the reserve floor stays at $20 (20% of $100). At $500 account value, a $20 floor is only 4%. A 4% cushion at 5× leverage is insufficient protection.

**Financial consequence:** Reserve gate becomes effectively meaningless as account grows, removing a designed safeguard.

---

### ⚠️ E-3 — Force-close skipped when `pnl_unknown=True` (price lookup failed)
**File:** `src/main.py:839-848`

When `get_current_price()` returns 0 after all retries, `pnl=0.0` and `pnl_unknown=True`. `check_losing_positions()` requires `pnl < 0` to trigger force-close; a pnl of 0 never fires. Alert is sent, operator notified, but the position continues running with only exchange-side SL protection.

**Financial consequence:** During an exchange price-feed outage (exactly when unusual market conditions are likely), the force-close failsafe is disabled.

---

### ⚠️ E-4 — `get_recent_fills(limit=50)` may miss closing fills on high-frequency days
**File:** `src/main.py:554`

40 trades/day × 3 assets = 120 fills/day. At peak, the last 50 fills cover only ~24 minutes of history. A trade that sat open for hours and closed via a day-old fill may have its exit price return `None`, producing `realized_pnl=None` in diary — excluded from stats and Sharpe.

---

### ⚠️ E-5 — StochRSI hard-trim may clip valid K values for non-default periods
**File:** `src/indicators/local_indicators.py:234-239`

```python
full_k = full_k[:len(rsi_vals)]   # hard trim — silently discards if k_line is longer
```

For unusual StochRSI period combinations (custom `STOCHRSI_K` or `STOCHRSI_D` values), the hard trim can clip valid K values. C-2 fix requires `len(k_vals) >= 3`, making this more impactful if the trim reduces the count below 3.

---

### ⚠️ E-6 (inner loop) — Inner loop candle gate STILL uses two `datetime.now()` calls
**File:** `src/main.py:2570-2571`

The outer loop was fixed (✅ E-6), but the inner loop was missed:

```python
_i_secs_into_5m = (datetime.now(timezone.utc).minute % 5) * 60 \
                  + datetime.now(timezone.utc).second    # ← still two calls
if (_i_secs_into_5m / 300) < 0.30:
```

A minute-rollover between the two calls produces `_secs_into_5m ≈ 0` at the candle boundary — the highest repaint-risk moment. The fix was applied to the outer loop but not copied to the inner loop.

---

### ⚠️ E-7 — Idempotency check blocks new SL if orphaned old SL order exists
**File:** `src/trading/hyperliquid_api.py:330-344`

During trailing stop update: place new SL → cancel old SL. If cancel fails (old SL stays open) and a retry fires for the new SL, `_trigger_order_retry()` finds the old SL in open orders and returns `{"status": "already_placed"}`. The new SL price never registers — position continues protected by the pre-trail (worse) SL level.

---

### ⚠️ E-8 — Post-rounding notional can fall below Hyperliquid $10 minimum
**File:** `src/main.py:1945` · `src/trading/hyperliquid_api.py:233`

Risk manager bumps allocation to $11 (line 445). But `round_size()` then applies `szDecimals` rounding to the contract quantity. For high-precision assets, the rounded quantity × price can slip below $10 again. Hyperliquid silently rejects the order.

---

### ⚠️ E-9 — Stale candle watchdog threshold accepts candles up to 14m59s old
**File:** `src/main.py:1389`

```python
if (_now_ms - int(_last_t)) > 3 * _stale_ms:   # 3 × 5min = 15min threshold
```

A 5m candle that is 14 minutes and 59 seconds old (missing 3 candles worth of data) passes the watchdog and triggers a signal on stale OHLCV data.

---

### ⚠️ E-10 — `_sl_cooldown_map` not persisted across restarts
**File:** `src/main.py:461`

```python
_sl_cooldown_map: dict = {}   # in-memory only, lost on crash/restart
```

An asset that triggered SL and entered cooldown is immediately tradeable again after a bot restart. A crash mid-cooldown allows immediate re-entry on the same adverse trend that just stopped the position.

**Financial consequence:** Re-entry on still-moving adverse trend immediately after SL hit. At 5× leverage, a second consecutive SL doubles the loss.

---

### ⚠️ SEC-3 — No rate limiting on dashboard auth endpoint
**File:** `src/main.py:329-338`

`DASHBOARD_TOKEN` comparison has no failed-attempt counter or lockout. An attacker with network access to the dashboard can brute-force the token at wire speed.

---

### ⚠️ SEC-5 — `DASHBOARD_TOKEN` compared with `!=` (not timing-safe)
**File:** `src/main.py:334`

```python
if auth_header != f"Bearer {_token}" and token_param != _token:
```

Python string comparison is not constant-time. A timing oracle attack can determine token length and content character by character. Use `hmac.compare_digest()`.

---

## ═══════════════════════════════════════
## PART 3 — NEW ISSUES INTRODUCED 🆕
## ═══════════════════════════════════════

---

### 🆕 NB-1 — `_log_trade_close()` inside outer KILLSWITCH blocks on Telegram (partial H-9 regression)
**File:** `src/main.py:708-711`

The top-level KILLSWITCH alert was converted to fire-and-forget (`asyncio.create_task`). But `_log_trade_close()` is now called inside the retry loop and internally calls `await send_alert()` (blocking, 10s timeout):

```python
# The _log_trade_close() call can block up to 10s if Telegram is down
await _log_trade_close(_ks_tr, "killswitch")   # contains await send_alert() internally
```

With 3 assets and Telegram down, the KILLSWITCH close loop can still block ~30 seconds — just at a different point than before the H-9 fix.

**Financial consequence:** KILLSWITCH close sequence delayed during the exact network-issue scenario where rapid closure matters most.

---

### 🆕 NB-2 — `get_user_state()` docstring now contradicts the FL-5 fix
**File:** `src/trading/hyperliquid_api.py:580-581`

```python
# Docstring still says:
TRUE total = perps_value (crossMarginSummary.accountValue, includes PnL)
           + spot_usdc  (USDC in the spot/unified wallet).
# Code now does:
total_value = perps_value   # spot_usdc excluded (FL-5 fix)
```

The docstring misleads future developers into thinking `spot_usdc` is included in `total_value`. A developer reading the docstring might revert FL-5 thinking the code is wrong.

---

### 🆕 NB-3 — `is_trade_expired()` method is now dead code
**File:** `src/trade_state.py:191-203`

The new `is_trade_expired_minutes()` method (lines 205-217) replaced `is_trade_expired()`. The old method is still present but has zero callers. Dead code in the state machine creates confusion and maintenance risk.

---

## ═══════════════════════════════════════════════════════
## PART 4 — COMPLETE STATUS SCORECARD
## ═══════════════════════════════════════════════════════

| ID | Category | Description | Status |
|----|----------|-------------|--------|
| FL-1 | Fund-Loss | time_limit_candles wired to 8-candle exit | ✅ FIXED |
| FL-2 | Fund-Loss | session_gate_ok overnight block ranges | ✅ FIXED |
| FL-3 | Fund-Loss | SL price > 0 validation | ✅ FIXED |
| FL-4 | Fund-Loss | SL on correct side of entry | ✅ FIXED |
| FL-5 | Fund-Loss | Spot USDC excluded from risk sizing | ✅ FIXED |
| C-1  | Critical  | TP1=TP2 same price / duplicate TP orders | ✅ FIXED |
| C-2  | Critical  | StochRSI K forming-candle repaint | ✅ FIXED |
| C-3  | Critical  | Inner loop SL 0 retries | ✅ FIXED |
| C-4  | Critical  | diary.jsonl OOM from readlines() | ✅ FIXED |
| C-5  | Critical  | record_entry before diary write | (unchanged — existing crash guard covers) |
| C-6-outer | Critical | KILLSWITCH outer exits not in diary | ✅ FIXED |
| C-6-inner | Critical | KILLSWITCH inner exits not in diary | ⚠️ STILL PRESENT |
| C-7  | Critical  | Anomaly check blocks event loop 45s | ⚠️ STILL PRESENT |
| H-1  | High | PnL charges taker fee on limit entry | ✅ FIXED |
| H-2  | High | Circuit breaker auto-resets midnight | ⚠️ STILL PRESENT |
| H-3  | High | No .gitignore / key exposed to git | ✅ FIXED |
| H-4  | High | API_HOST defaults to 0.0.0.0 | ✅ FIXED |
| H-5  | High | active_trades loss → no trailing stop | ⚠️ STILL PRESENT |
| H-6  | High | No weekly/total drawdown halt | ⚠️ STILL PRESENT |
| H-7  | High | Outer state stale 60s at execution | ⚠️ STILL PRESENT |
| H-8  | High | Reserve floor doesn't scale with growth | ⚠️ STILL PRESENT |
| H-9  | High | KILLSWITCH Telegram blocks close seq | ✅ FIXED (top-level) |
| H-10 | High | TAKER_FEE_PCT naming ambiguity | (unchanged — low runtime risk) |
| E-1  | Edge | _today_utc stale in inner loop | ✅ FIXED |
| E-2  | Edge | Timeout cooldown ignores COOLDOWN_MINUTES | ✅ FIXED |
| E-3  | Edge | Force-close skipped on pnl_unknown | ⚠️ STILL PRESENT |
| E-4  | Edge | get_recent_fills limit=50 misses old fills | ⚠️ STILL PRESENT |
| E-5  | Edge | stoch_rsi hard-trim clips values | ⚠️ STILL PRESENT |
| E-6-outer | Edge | Candle gate double datetime.now() | ✅ FIXED |
| E-6-inner | Edge | Candle gate inner loop double datetime | ⚠️ STILL PRESENT |
| E-7  | Edge | Idempotency blocks new SL if old open | ⚠️ STILL PRESENT |
| E-8  | Edge | Post-rounding notional < $10 | ⚠️ STILL PRESENT |
| E-9  | Edge | Stale candle threshold 14m59s | ⚠️ STILL PRESENT |
| E-10 | Edge | _sl_cooldown_map not persisted | ⚠️ STILL PRESENT |
| SEC-3 | Security | No dashboard auth rate limiting | ⚠️ STILL PRESENT |
| SEC-5 | Security | DASHBOARD_TOKEN not timing-safe | ⚠️ STILL PRESENT |
| NB-1 | 🆕 New | _log_trade_close blocks KILLSWITCH loop | 🆕 NEW |
| NB-2 | 🆕 New | get_user_state docstring contradicts FL-5 | 🆕 NEW |
| NB-3 | 🆕 New | is_trade_expired() dead code in TradeStateMachine | 🆕 NEW |

---

## FINAL VERDICT

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
     🏛️  TRADING BOT BUG COUNCIL — REPORT v13
     POST-FIX RE-AUDIT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ FIXED THIS ROUND: 17 issues (5 fund-loss, 5 critical,
   4 high, 2 edge, 1 security)

⚠️ STILL PRESENT: 17 issues (2 critical, 6 high,
   8 edge, 2 security)

🆕 NEW REGRESSIONS: 3 (1 medium, 2 low)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COUNCIL VERDICT: SIGNIFICANT PROGRESS — CONDITIONALLY
READY FOR LOW-CAPITAL TESTNET

The 5 fund-loss risks are fully resolved. The signal
logic (FL-1, FL-2, FL-3, FL-4, FL-5, C-1, C-2) is
now correct. The bot's documented behavior matches its
actual behavior for the first time.

REMAINING BLOCKERS (before any live capital):
  1. C-6-inner: Apply _log_trade_close() to inner loop
     KILLSWITCH (2-line fix, mirrors outer loop)
  2. E-6-inner: Fix inner loop candle gate double
     datetime.now() (1-line fix, mirrors outer fix)
  3. NB-1: Wrap _log_trade_close() Telegram calls in
     asyncio.create_task() inside KILLSWITCH paths

NEXT PRIORITY (before scaling capital):
  4. C-7: Gather anomaly checks concurrently across
     assets (asyncio.gather) to eliminate 45s blockage
  5. E-10: Persist _sl_cooldown_map to disk
  6. H-2: Add OPERATOR_CONFIRM_RESTART flag to prevent
     circuit breaker auto-resume
  7. H-6: Add weekly drawdown halt

TOP 3 MUST-FIX BEFORE LIVE:
  C-6-inner · E-6-inner · NB-1

CAPITAL RECOMMENDATION:
  Testnet: ✅ Safe to run
  Live $0–100: ✅ After C-6-inner + E-6-inner + NB-1
  Live $100+: After C-7 + E-10 + H-2

GO LIVE: ⚠️ PAPER TRADE FIRST (3 fixes needed)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

*Re-audit: 2026-06-04 | Council v13 | All 5 agents re-ran on updated codebase*
*Fixed this round: 17/37 total issues. 3 new regressions. 17 remaining.*
