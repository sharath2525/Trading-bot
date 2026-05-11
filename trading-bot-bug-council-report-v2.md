# 🏛️ TRADING BOT BUG COUNCIL — RESCAN REPORT v2
**Date:** 2026-05-10 | **Scope:** Post-fix rescan, bugs only — no improvements

---

## 📋 BOT SUMMARY

| Field | Value |
|-------|-------|
| Strategy | Code-first hybrid perpetual futures |
| Exchange | Hyperliquid mainnet |
| Asset class | Crypto perps (BTC, ETH + 229 others) |
| Leverage | 5× (set via FL-1 fix) |
| Order types | Market entry + reduce-only trigger TP/SL |
| Architecture | Outer loop (1h) + inner loop (11 × 5m ticks) |
| Claude role | APPROVE/REJECT only on score ≥ MIN_AI_SCORE |
| MASTER RULES status | All 4 rules intact — score system, code decisions, Claude role, risk math unchanged |

**Previous fixes verified as correct:** FL-1 through FL-4, C-2 through C-7, H-1 through H-5, E-1 through E-4

**This scan found 7 new bugs.** All were introduced or revealed by the post-fix code state.

---

## 💀 CRITICAL — Fix before any live trading

---

### BUG-1 · SIGTERM/SIGINT signal handler declared but never wired
**File:** `src/main.py` line 39, 608
**Agents:** 🌪️ Chaos + 👤 Operator

```python
# Line 39 — _shutdown flag exists but is NEVER set to True
_shutdown = False  # Comment says "Set to True by SIGTERM/SIGINT handler"

# Line 11 — signal imported but never used
import signal

# Line 608 — check exists but is permanently False
if _shutdown:
    logging.info("[SHUTDOWN] Signal received — exiting loop cleanly")
    break
```

**What is missing:** `signal.signal(signal.SIGTERM, ...)` and `signal.signal(signal.SIGINT, ...)` are never called anywhere in the code. The handler intent is documented but not implemented.

**Financial consequence:** When systemd, Docker, or the OS sends SIGTERM (during a deploy, reboot, or OOM kill), Python's default SIGTERM handler terminates the process immediately. The outer loop's clean-exit `break` never runs. Any mid-cycle execution — including a just-placed market order with no TP/SL yet placed — leaves an open position with no protection. At 5× leverage, slippage on an unexpected close can be severe.

**Fix:** In `main()`, before `asyncio.run()`, add:
```python
import signal

def _handle_shutdown(sig, frame):
    global _shutdown
    _shutdown = True
    logging.info("[SHUTDOWN] %s received — will exit after current cycle", signal.Signals(sig).name)

signal.signal(signal.SIGTERM, _handle_shutdown)
signal.signal(signal.SIGINT,  _handle_shutdown)
```

---

### BUG-2 · Inner loop TP/SL computed from stale outer-loop price, position executed at fresh price
**File:** `src/main.py` lines 1709–1714 (computation) vs 1793–1810 (refresh) vs 1844 (execution)
**Agents:** 🔨 Logic Breaker + ⚡ Performance

```python
# Lines 1709–1714: TP/SL computed BEFORE price refresh using outer-loop _ie
_ie   = float(_iac.get("current_price") or 0)     # ← stale outer-loop price
_iatr = float(_iac.get("long_term_4h", {}).get("atr14") or 0)
_itp, _isl = _code_compute_tpsl(_ie, _iatr, _idir)   # ← wrong base price

# Lines 1793–1810 (C-3/C-4 fix): refreshes account state AND prices
_istate_fresh = await hyperliquid.get_user_state()
_rip = await hyperliquid.get_current_price(_ri_asset)
if _rip > 0:
    asset_prices[_ri_asset] = _rip       # asset_prices updated...
    _rms["current_price"] = _rip          # market_sections updated...

# Line 1844: execution uses fresh price
_iout["current_price"] = _iprice          # ← fresh price (good)
# BUT _iout["tp_price"] and _iout["sl_price"] are still from stale _ie ← BUG
```

**Financial consequence:** If BTC moves from $104,000 (outer loop) to $105,000 (inner tick), `_itp` and `_isl` are computed against $104,000 but the position opens at $105,000. With ATR = $1,000 (≈1%):
- SL should be at $104,000 ($105,000 - $1,000) — but is placed at $103,000 ($104,000 - $1,000)
- The actual SL distance is $2,000 instead of $1,000 — 2× the intended risk
- At 5× leverage on a $75 position: $150 exposure with $3 SL tolerance instead of $1.50

In a downtrend where 1% moves occur within a 5-minute tick, every inner-loop entry has TP/SL misaligned by one full ATR.

**Fix:** After line 1844 (`_iout["current_price"] = _iprice`), recompute TP/SL from the fresh price:
```python
_iout["current_price"] = _iprice
# Recompute TP/SL from fresh execution price (fixes inner-loop stale-price TP/SL bug)
_fresh_iatr = _iact_ctx.get("long_term_4h", {}).get("atr14") or _iout.get("atr14")
if _fresh_iatr and float(_fresh_iatr) > 0 and abs(_iprice - _ie) / _ie > 0.001:
    _fresh_tp, _fresh_sl = _code_compute_tpsl(_iprice, float(_fresh_iatr), _iout["action"])
    _iout["tp_price"] = _fresh_tp
    _iout["sl_price"] = _fresh_sl
```

---

### BUG-3 · Timeout force-close does not cancel TP/SL orders — orphaned orders block reconciliation
**File:** `src/main.py` lines 786–803
**Agents:** 🔨 Logic Breaker + 👤 Operator

```python
# Lines 796–802: TIME-BASED EXIT — missing cancel_all_orders
try:
    await hyperliquid.market_close(_asset_name)          # position closed ✓
    state_mgr.start_cooldown(_asset_name, interval_seconds=3600)  # state = COOLDOWN ✓
    # ← NO cancel_all_orders() here — TP/SL trigger orders remain on exchange
    for _tr in active_trades:
        if _tr.get('asset') == _asset_name:
            _tr['pending_exit_type'] = 'timeout'

# Compare: RISK force-close at line 707–708 correctly cancels
await hyperliquid.market_close(coin)
await hyperliquid.cancel_all_orders(coin)    # ← this is correct
```

**Chain of consequences:**
1. Position is closed by `market_close()`
2. TP/SL reduce-only trigger orders remain active on the exchange
3. Reconcile loop (line 762): `assets_with_orders = {o.get('coin') for o in open_orders}` — asset stays in this set because orphaned TP/SL orders appear in `get_open_orders()`
4. Reconcile condition (line 765): `if asset not in assets_with_positions and asset not in assets_with_orders` — False because of orphaned orders
5. Stale `active_trade` entry is **never removed** from `active_trades.json`
6. On next restart, the ghost entry is reloaded and the guardian sees state=COOLDOWN with no position — it calls `start_cooldown()` again (OK), but the orphaned TP/SL orders on the exchange are never cleaned up until they expire or are manually cancelled

**Financial consequence:** Ghost entries in `active_trades.json` corrupt the trade log. If the cooldown expires and the bot attempts a new entry on the same asset, the old orphaned reduce-only trigger orders may conflict with newly placed ones (duplicate trigger orders). Hyperliquid silently ignores some duplicate triggers but the guardian and reconcile can get confused, potentially missing a real SL order.

**Fix:** Add `cancel_all_orders` to the timeout path:
```python
try:
    await hyperliquid.market_close(_asset_name)
    await hyperliquid.cancel_all_orders(_asset_name)     # ← ADD THIS LINE
    state_mgr.start_cooldown(_asset_name, interval_seconds=3600)
    for _tr in active_trades:
        if _tr.get('asset') == _asset_name:
            _tr['pending_exit_type'] = 'timeout'
except Exception as _te:
    add_event(f"[TIMEOUT] {_asset_name} close error: {_te}")
```

---

## 🔴 HIGH — Fix before real money exposure

---

### BUG-4 · near_ema defaults to True when 15m EMA20 is unavailable — E-1 fix bypassed
**File:** `src/main.py` lines 1093–1098 (outer loop) and `src/strategy.py` line 138
**Agents:** 🔨 Logic Breaker

The E-1 fix in strategy.py correctly defaults to `False` when the `near_ema` **key is absent**. But the key is never absent — it is always populated by the outer loop:

```python
# src/main.py lines 1093–1098: near_ema is ALWAYS set
near_ema_15m = (
    abs(current_price - ema20_15m) / current_price < 0.003
    if (ema20_15m is not None and current_price > 0)
    else True        # ← BUG: when EMA unavailable, sets True instead of False
)
# Line 1167: always pushed into setup_15m dict
"near_ema": near_ema_15m,

# src/strategy.py line 138 (E-1 fix):
near_ema = bool(s15.get("near_ema", False))   # False default never reached —
                                               # key is always present (True) when EMA missing
```

**When does EMA20 become None?** During startup when <20 completed 15m candles are available (first ~5 hours of operation, or after exchange data gaps). In that window, `near_ema = True` is written and `entry_confirmed()` allows entries through the EMA proximity gate without any actual EMA proximity data.

**Financial consequence:** During startup the bot can enter trades that would be blocked by the EMA proximity gate if indicators were fully warmed up. The entry_confirmed() function explicitly requires `near_ema and macd_15m > -threshold` for BUY — this guard is silently bypassed.

**Fix:** In `src/main.py` line 1097, change `else True` to `else False`:
```python
near_ema_15m = (
    abs(current_price - ema20_15m) / current_price < 0.003
    if (ema20_15m is not None and current_price > 0)
    else False    # ← FIX: missing data blocks entry, same as E-1 intent
)
```

---

### BUG-5 · KILLSWITCH not checked inside the 55-minute inner loop
**File:** `src/main.py` lines 1659–1898
**Agents:** 🌪️ Chaos + 👤 Operator

```python
# Line 607–616: KILLSWITCH checked in outer loop only
while True:
    if _shutdown: break
    if os.path.exists(os.path.normpath(_KILLSWITCH_FILE)):
        logging.critical("[KILLSWITCH] KILLSWITCH file detected — halting")
        break
    ...
    # Lines 1659–1898: inner loop — NO KILLSWITCH CHECK
    for _tick in range(11):
        await asyncio.sleep(300)   # 5 minutes
        # ← operator creates KILLSWITCH file here
        # bot continues trading for up to 55 more minutes before checking
```

**Financial consequence:** An operator creating the KILLSWITCH file expects immediate halt. Instead the bot continues for up to 55 minutes (11 ticks × 5 minutes), executing up to 11 additional scoring cycles and potentially entering new positions. With `MAX_CONCURRENT_POSITIONS=2` and `MAX_DAILY_TRADES=10`, multiple trades can still be placed in this window.

**Fix:** Add KILLSWITCH check at the top of each inner tick:
```python
for _tick in range(11):
    await asyncio.sleep(300)
    # C-7 extension: check KILLSWITCH inside inner loop
    if os.path.exists(os.path.normpath(_KILLSWITCH_FILE)):
        logging.critical("[KILLSWITCH] KILLSWITCH detected in inner tick %d — breaking inner loop", _tick + 1)
        break
    if risk_mgr.circuit_breaker_active:
        ...
```

---

### BUG-6 · H-3 startup validation never fires — permissive config defaults are always non-None
**File:** `src/main.py` lines 325–347 (H-3), `src/config_loader.py` lines 96–103
**Agents:** 👤 Operator

```python
# config_loader.py — ALL risk keys have non-None string defaults:
"max_leverage":                  _get_env("MAX_LEVERAGE", "10"),          # default "10"
"max_position_pct":              _get_env("MAX_POSITION_PCT", "20"),       # default "20"
"max_loss_per_position_pct":     _get_env("MAX_LOSS_PER_POSITION_PCT", "20"),
"daily_loss_circuit_breaker_pct":_get_env("DAILY_LOSS_CIRCUIT_BREAKER_PCT","25"),
"max_concurrent_positions":      _get_env("MAX_CONCURRENT_POSITIONS", "10"),
"min_balance_reserve_pct":       _get_env("MIN_BALANCE_RESERVE_PCT", "10"),

# main.py H-3 validation — checks for None, which never happens:
for _env_key, (_cfg_key, _code_default, _safe_value) in _risk_defaults.items():
    _actual = CONFIG.get(_cfg_key)
    if _actual is None:              # ← NEVER True — always returns default string
        _using_dangerous_defaults.append(...)
```

**What happens without .env:** A user who runs `python src/main.py` without a `.env` file gets 10× leverage, 20% position sizing, 25% daily loss tolerance, and 10 concurrent positions — all silently, without any warning. H-3 was designed specifically to catch this but the condition is logically inverted relative to how config_loader works.

Note also: config_loader defaults (10×, 20%, 25%) differ from `.env.example` values (5×, 15%, 12%). This is a secondary documentation bug.

**Fix:** Change the H-3 check to detect when the **env var** is absent (not when CONFIG value is None):
```python
for _env_key, (_cfg_key, _code_default, _safe_value) in _risk_defaults.items():
    # Check if the env var is actually set — CONFIG always returns a default, never None
    if os.getenv(_env_key) is None:
        _using_dangerous_defaults.append(
            f"  {_env_key} not set in .env — using code default {_code_default} (recommended: {_safe_value})"
        )
```

---

## 🟡 EDGE CASE — Fix before scale

---

### BUG-7 · trade_log list grows indefinitely in memory
**File:** `src/main.py` lines 356, 1573
**Agents:** ⚡ Performance

```python
# Line 356: initialized once, never cleared
trade_log = []

# Line 1573: appended on every executed trade (outer loop)
trade_log.append({"type": action, "price": current_price, "amount": tp_sl_size, ...})
# Note: inner-loop trades do NOT append to trade_log (asymmetry)
```

`trade_log` is declared "for Sharpe ratio" but `calculate_sharpe_from_diary()` (line 659) reads from `diary.jsonl` on disk — it never reads `trade_log`. The in-memory list is dead weight that accumulates for the entire process lifetime. At 10 trades/day × 30 days = 300 entries, each ~200 bytes = ~60KB — trivial. At months of runtime on many assets it becomes MB-scale. The asymmetry (inner loop doesn't append) is also a correctness issue if trade_log were ever actually used.

**Fix:** Either bound it or remove it:
```python
trade_log = deque(maxlen=200)   # bound it (already imports deque)
# or simply remove trade_log entirely — calculate_sharpe_from_diary() doesn't use it
```

---

## 🔐 SECURITY

No new security issues found beyond those already addressed. Previous fixes hold.

---

## ⚡ PERFORMANCE

No new performance issues beyond BUG-7.

---

## 📊 MONITORING & OPERATIONAL

- **BUG-1** leaves the process with no clean shutdown path under systemd/Docker
- **BUG-5** means the KILLSWITCH is not a real kill switch — it's a "kill switch within the next hour"

---

## 🔍 PEER REVIEW

### 🔨 Logic Breaker reviewing others:
Most critical finding from other agents: **BUG-1 (signal handler)** — an unhandled SIGTERM mid-execution-path (between `place_buy_order` and `place_stop_loss`) leaves a position with no SL. This is the scenario that ends accounts.

Biggest miss by all agents: The **inner loop allocation** (`_ialloc` at line 1714) is computed using `account_value` from the outer loop. Although `validate_trade()` re-caps it using the refreshed state, the initial `_ialloc` sizing uses the stale pre-refresh `account_value`. If a trade executed during the outer loop reduced the account by $20, the inner loop's `_ialloc` could temporarily compute above what `validate_trade` would allow — validate_trade catches this, so no actual over-allocation occurs, but it's a latent consistency issue.

Issue that would stop me from letting this run: **BUG-3** — stale ghost `active_trade` entries that never get cleaned up corrupt the entire trade tracking system over time.

### 🔐 Security Analyst reviewing others:
Most critical: **BUG-1** — no signal handler means any process manager (systemd, Docker restart policy) can kill the bot mid-trade.

Biggest miss: The KILLSWITCH file (BUG-5) creates a security-adjacent operational gap: a compromised operator who can't SSH but can create files might expect the KILLSWITCH to stop the bot instantly. It won't for 55 minutes.

Issue that would stop me: **BUG-2** — TP/SL at wrong levels means the risk-reward ratio is corrupted on every inner-loop trade.

### 🌪️ Chaos Tester reviewing others:
Most critical: **BUG-3** — scenario: bot times out a trade at hour 12, `market_close()` succeeds, then Hyperliquid connection drops before `cancel_all_orders()` runs. Now the position is flat but 2 trigger orders remain. On reconnect, nothing cleans them up. They sit indefinitely.

Biggest miss by all agents: What happens when `market_close()` succeeds in the timeout path but `state_mgr.start_cooldown()` throws an exception? The position is closed, orders are not cancelled (BUG-3), state stays ENTERED, and the reconcile loop re-enters on the next cycle thinking it's an active trade. The outer loop will try to reconcile a closed position against active trigger orders, potentially re-logging a "stale entry" close.

Issue that would stop me: **BUG-2** — in a fast-moving market where 1% moves happen between ticks, every inner loop trade has TP/SL misaligned by one full ATR distance.

### ⚡ Performance Engineer reviewing others:
Most critical: **BUG-1** — systemd restarts on SIGTERM, which now kills the process hard. With `Restart=always` in the service file (`trading-agent.service` is present in the repo), SIGTERM → immediate kill → restart → startup leverage check → possibly re-entering a position that wasn't properly closed.

Biggest miss: Inner loop runs `asyncio.to_thread(agent.confirm_trade, ...)` at lines 1766–1769 which is a blocking Claude API call. If Claude responds slowly (network latency), this blocks the inner tick for the duration. With `MIN_AI_CALL_GAP_MINUTES=30` this should be rare, but when it fires it delays all execution in that 5-minute window.

Issue that would stop me: **BUG-5** — if the operator creates KILLSWITCH because an account drawdown is occurring, waiting 55 minutes for it to take effect defeats the purpose.

### 👤 Operator Safety reviewing others:
Most critical: **BUG-6** — an operator who forgot to set `.env` is running at 10× leverage with 20% position sizing and no warning. H-3 was the only guard against this and it's broken.

Biggest miss: **Signal handler (BUG-1) combined with Docker `--restart=always`** — the service file uses `Restart=on-failure`. SIGTERM triggers an immediate kill (not a failure), so the bot does NOT auto-restart on SIGTERM — it stays dead with the position open. No alert fires, no close. The combination of no signal handler + no alerting (C-8 was deferred) = silent orphaned position.

Issue that would stop me: **BUG-1 + deferred C-8 (no alerting)** — when the process dies silently, nobody knows. Positions stay open indefinitely.

---

## 🏛️ FINAL REPORT

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
     🏛️  TRADING BOT BUG COUNCIL — FINAL REPORT v2
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Previous council scan: 15 bugs fixed, all verified correct.
This scan (post-fix): 7 new bugs found — 3 Critical, 2 High, 1 Edge, 1 Low.

MASTER RULES: All 4 intact. Score system, code decisions,
Claude role, risk math — all unchanged by this report.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COUNCIL VERDICT: HIGH RISK — 3 Critical bugs remain

TOP 3 MUST-FIX:
  1. BUG-1: Wire SIGTERM/SIGINT handlers (process dies with open positions)
  2. BUG-3: Add cancel_all_orders() to timeout handler (ghost orders)
  3. BUG-2: Recompute TP/SL from fresh price in inner loop (wrong risk levels)

CAPITAL RECOMMENDATION: Max $50 until BUG-1 + BUG-3 fixed.
                        Max $200 until all 3 critical + 2 high fixed.

GO LIVE AT SCALE: NOT READY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 📋 QUICK REFERENCE TABLE

| # | Bug | File | Line(s) | Severity | Financial Risk | Fix Size |
|---|-----|------|---------|----------|----------------|----------|
| BUG-1 | SIGTERM/SIGINT handler declared but never wired — process kills instantly mid-trade | `main.py` | 11, 39, 608 | 💀 CRITICAL | Open position, no SL, full account risk | 6 lines |
| BUG-2 | Inner loop TP/SL computed from stale outer-loop price before C-3/C-4 refresh fires | `main.py` | 1709–1714, 1844 | 💀 CRITICAL | TP/SL off by 1 full ATR, 2× intended risk | 4 lines |
| BUG-3 | Timeout force-close calls `market_close()` but not `cancel_all_orders()` — orphaned TP/SL orders block reconcile | `main.py` | 796–803 | 💀 CRITICAL | Ghost active_trade entries, stale order corruption | 1 line |
| BUG-4 | `near_ema_15m` defaults to `True` when EMA20 is unavailable — E-1 fix bypassed | `main.py` | 1097 | 🔴 HIGH | Entry without EMA confirmation during warmup | 1 character |
| BUG-5 | KILLSWITCH file only checked in outer loop — inner loop runs 55 more minutes | `main.py` | 1659–1898 | 🔴 HIGH | Up to 55 min of uncontrolled trading after kill | 4 lines |
| BUG-6 | H-3 validation checks `CONFIG.get() is None` but config_loader always returns a default string — warning never fires | `main.py` | 335–347 | 🔴 HIGH | Permissive defaults (10×, 20%) used silently if .env missing | 1 line |
| BUG-7 | `trade_log` list grows forever in memory and is never used by Sharpe calculation | `main.py` | 356, 1573 | 🟡 EDGE | Memory leak over weeks/months of runtime | 1 line |

### Fix Priority Order
1. **BUG-1** → wire signal handler (4 lines in `main()` before `asyncio.run()`)
2. **BUG-3** → add `cancel_all_orders` to timeout path (1 line)
3. **BUG-4** → change `else True` to `else False` in `near_ema_15m` (1 character)
4. **BUG-2** → recompute TP/SL at inner loop execution site (4 lines)
5. **BUG-5** → add KILLSWITCH check inside inner loop (3 lines)
6. **BUG-6** → change `CONFIG.get(_cfg_key) is None` to `os.getenv(_env_key) is None` (1 line)
7. **BUG-7** → bound `trade_log` with `deque(maxlen=200)` (1 line)

### Still Deferred (user's decision from previous scan)
| Item | What | Status |
|------|------|--------|
| C-8 | Telegram alerting | Needs `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` in `.env` |
| E-3 | Delete legacy `decide_trade()` | Dead code, safe to remove |
| E-5 | `DRY_RUN=true` paper mode | Wrap order calls behind env flag |

---

*Report generated by 5-agent Trading Bot Bug Council. MASTER RULES verified intact. All findings are bugs requiring code fixes — no strategy changes, no improvements, no score system modifications.*
