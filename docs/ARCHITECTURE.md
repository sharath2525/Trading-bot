## Trading Agent Architecture (High-Level)

> **Architecture version: CODE-FIRST HYBRID (updated 2026-04-30, all 22 fixes applied)**
> Code decides direction, TP, SL, and size. Claude is called only as a final sanity check
> on the highest-confidence (score = 10/10) setups. All signal logic is deterministic.
> Permanent rules governing this architecture: see `MASTER_RULES.md` in the project root.

This document outlines the end-to-end flow of the trading agent at a conceptual level.
It focuses on subsystems, data flows, and guardrails rather than specific functions.

### Subsystems

- **Config/Env**: Centralized runtime settings from `.env` (keys, model, assets, interval).
- **Outer Loop (every 1h)**: Coordinates all subsystems. Fetches full candle history, computes all indicators, handles force-closes, reconciliation, and guardian re-placement of missing TP/SL orders.
- **Inner Loop (every 5m × 12)**: Refreshes 5m candles only and re-runs the full scoring pipeline. 12 ticks per hour = entry opportunities every 5 minutes without waiting for the 1h outer cycle.
- **Code Signal Engine** (`strategy.py` + `main.py`):
  - `_code_decide_direction()` — 4h hard gate. Returns `"buy"`, `"sell"`, or `None` (HOLD). Counter-trend trades are structurally impossible.
  - `compute_signal_score()` — weighted float score 0–10. Score < `MIN_SIGNAL_SCORE` (7) → HOLD.
  - `_code_compute_tpsl()` — TP = entry + 2×ATR, SL = entry − 1×ATR. Never set by Claude.
  - `entry_confirmed()` — final 15m/5m confirmation gate (RSI, ADX, volume, near-EMA).
- **Claude Confirmation Gate** (`confirm_trade()` in `decision_maker.py`):
  - Called ONLY when score == 10.0 (mathematically rare).
  - Receives a ~150-token context: asset, direction, entry, TP, SL, score.
  - Returns exactly one word: `APPROVE` or `REJECT`. `max_tokens=10`.
  - Fails closed — any error, timeout, or unexpected output → `REJECT`.
- **Risk/Collateral Gate** (`risk_manager.py`): Validates all 8 safety checks before execution. Can cap allocation or block trade entirely. Non-bypassable.
- **Execution Layer** (`hyperliquid_api.py`): Places market orders. Immediately places reduce-only TP and SL trigger orders after entry.
- **Reconciliation**: Resolves local intent vs exchange truth (positions/orders/fills), purges stale local state, logs outcomes.
- **Observability**: HTTP API (port 3000) serving `/diary`, `/live`, `/logs`.

### Data Principles

- **Authoritative Source**: Exchange state (positions, orders, fills, mids) always supersedes local intent.
- **Perp-Only Pricing**: Price context comes from Hyperliquid mids; no spot/perp basis mixing.
- **Pre-Computed Signals**: `trend_4h`, `trend_1h` (EMA20 vs EMA50) and all MACD/ATR values are computed locally before any decision logic runs.
- **Two-Config Isolation**: `MIN_TRADE_SCORE` (int, 0–5) feeds `entry_confirmed()` only. `MIN_SIGNAL_SCORE` (float, 0–10) feeds the main loop pre-gate only. These must stay separate.
- **Time Semantics**: All timestamps are UTC ISO-8601.

### Signal Flow

```
Candles (5m × 20, 15m × 30, 1h × 60, 4h × 60, 1d × 30) — per asset
    ↓ compute_all() — local indicator calculation (EMA, RSI, MACD, ATR, ADX, etc.)
    ↓
_code_decide_direction()
    ↓ None → HOLD (conflicting or unknown trend_4h)
    ↓ "buy" or "sell"
compute_signal_score()   [weights: trend_4h=3, trend_1h=2, MACD_15m=2, near_ema=1.5, trigger_5m=1.5]
    ↓ score < 7.0 → HOLD
    ↓ score 7.0–8.5 → proceed to daily cap + SL cooldown + market_filter + entry_confirmed
    ↓ score == 10.0 → confirm_trade() (Claude APPROVE/REJECT, max_tokens=10, fail-closed)
_code_compute_tpsl(entry, atr, direction)
    ↓ tp = entry ± 2×ATR,  sl = entry ∓ 1×ATR
atr_position_size × (score / 10)
    ↓
risk_manager.validate_trade()   [8 guards]
    ↓
Hyperliquid SDK — market order + TP trigger + SL trigger
```

### 4h Hard Gate

`_code_decide_direction()` returns `None` (HOLD) if:
- `trend_4h` is `"UNKNOWN"` (insufficient candles or flat EMA cross)
- `trend_4h == "BULLISH"` but `trend_1h == "BEARISH"` (conflicting trends)
- `trend_4h == "BEARISH"` but `trend_1h == "BULLISH"` (conflicting trends)

When `None` is returned, `compute_signal_score()` is never called and no Claude call is made.
Counter-trend entries are structurally impossible.

### Score Achievable Values

Weights: `trend_4h=3.0, trend_1h=2.0, MACD_15m=2.0, near_ema=1.5, trigger_5m=1.5` (sum = 10.0)

Achievable: **0, 1.5, 2, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 8, 8.5, 10**

Score 9 is mathematically unreachable. Every path to score ≥ 7 requires `trend_4h` to be aligned.

### Per-Asset SL Cooldown

After any SL hit or force-close due to loss on an asset, that asset is blocked from new entries for `COOLDOWN_MINUTES` (default 60). Tracked in `_sl_cooldown_map: dict[str, datetime]` inside `run_loop()`. Prevents revenge-trading into a losing direction.

### Daily Trade Cap

`_daily_trade_count` increments on each executed trade. When `>= MAX_DAILY_TRADES`, all new executions are blocked until UTC midnight reset. Prevents fee bleed on high-signal days.

### Robustness

- **Retry**: Up to 3 attempts with exponential backoff on Hyperliquid API calls.
- **Reconciliation**: Stale `active_trades` entries are pruned each cycle based on live exchange state.
- **TP/SL Guardian**: Every outer cycle re-places missing trigger orders for any ENTERED position.
- **Circuit Breaker**: Daily drawdown limit halts all new trades; resets at UTC midnight.
- **Time-Based Exit**: Trades open > `MAX_TRADE_HOURS` (12h) with no TP hit are force-closed at market to prevent capital lock.
- **Fail-Closed Claude**: `confirm_trade()` catches all exceptions and returns `"REJECT"` — a Claude API outage never causes a bad trade to execute.
