## Trading Agent Architecture

> **Architecture: CODE-FIRST HYBRID — updated 2026-05-25**
> Code decides direction, TP, SL, and size. Claude (Sonnet 4.6) is called as a deep market-analysis
> gate when score ≥ MIN_AI_SCORE (6) AND multi-timeframe confluence is confirmed.
> Permanent rules: see `MASTER_RULES.md`.

---

### Subsystems

- **Config/Env**: All runtime settings from `.env`. `src/config_loader.py` handles type coercion.
- **Outer Loop (every 1h)**: Fetches full candle history, computes all indicators, handles force-closes, trailing-stop guardian, TP/SL guardian, time-based exits, daily counter reset.
- **Inner Loop (every 5m × 12)**: Refreshes 5m candles and re-runs the full scoring pipeline. 12 ticks/hour.
- **Code Signal Engine** (`strategy.py` + `main.py`):
  - `_code_decide_direction()` — 4h EMA trend + 1h ADX≥15 + daily bias (1d candle) + BB width regime. Returns `"buy"`, `"sell"`, or `None` (HOLD). Counter-trend entries are structurally impossible.
  - `compute_signal_score()` — weighted float 0–10 with optional bonuses (volume +1.0, candle pattern +0.5, Kronos ±0.5). Capped at 11.0.
  - `_code_compute_tpsl()` — score-adaptive TP (1.6×/1.8×/2.0×/2.2×/2.5×ATR) + partial-close levels (TP1=1×ATR, TP2=3×ATR) + SL=1×ATR. Fee buffer always included.
  - `entry_confirmed()` — 15m/5m RSI, volume, near-EMA, stale-setup confirmation gate.
  - `market_filter()` — ATR spike, spread, time gate (00:00–06:00 UTC), weekend gate (Fri 20:00→Sun 08:00 UTC), BTC correlation filter, S&R zones, funding rate hard gate.
  - `oi_confirmed()` — OI must be increasing; blocks on OI spike >5%.
  - `is_trending_regime()` — BB width above 20-period median required.
- **Claude Confirmation Gate** (`confirm_trade()` in `decision_maker.py`):
  - Called when `score >= MIN_AI_SCORE (6)` AND `multi_timeframe_confluence()` returns True.
  - Model: `claude-sonnet-4-6`, `max_tokens=4000`, `timeout=30s`.
  - Receives ~1500-token context: full 5-TF indicators, macro events, news headlines, recent trade history.
  - Returns structured 5-factor analysis (each 1–5) with TOTAL/25.
  - `VERDICT: APPROVE` only if TOTAL ≥ 15 and no auto-reject. Anything else → REJECT (fail closed).
  - Verdict cached: 60 min APPROVE / 30 min REJECT per asset.
  - Hard gap: `MIN_AI_CALL_GAP_MINUTES` (30 min) per asset.
- **Risk/Collateral Gate** (`risk_manager.py`): Validates all 8 safety checks before execution. Non-bypassable.
- **Execution Layer** (`hyperliquid_api.py`): Places LIMIT orders (0.15% better than market). Cancels if unfilled after 1 candle. Places TP1 (50% at 1×ATR) + TP2 (50% at 3×ATR) + SL trigger orders.
- **Trailing Stop Guardian** (outer loop): Stage 1 — move SL to breakeven at +1×ATR. Stage 2 — trail SL at 0.5×ATR behind at +1.5×ATR.
- **Observability**: HTTP API (port 3000) serving `/`, `/diary`, `/live`, `/logs`.

---

### Signal Flow

```
Candles (5m × 20+, 15m × 30+, 1h × 60+, 4h × 60+, 1d × 30+) — per asset
    ↓ compute_all() — local indicators (EMA, RSI, MACD, ATR, ADX, BB, OBV, VWAP)
    ↓
_code_decide_direction()
    Gates: 4h EMA trend + 1h ADX≥15 + daily bias + BB width regime
    ↓ None → HOLD
    ↓ "buy" or "sell"
compute_signal_score()
    Base weights: trend_4h=3.0, trend_1h=2.0, MACD_15m=2.0, near_ema=1.5, trigger_5m=1.5
    Bonuses: volume +1.0, candle pattern +0.5, Kronos ±0.5 (optional)
    ↓ score < MIN_SIGNAL_SCORE (6) → HOLD + signal log if score ≥ 6
    ↓ score ≥ 6
Daily cap check, SL cooldown check
multi_timeframe_confluence() — 4h/1h/30m/15m/5m all aligned
    ↓ if False → skip Claude, check MIN_AI_SCORE gate separately
confirm_trade() — Claude Sonnet structured 5-factor analysis
    ↓ VERDICT: REJECT → HOLD
    ↓ VERDICT: APPROVE
market_filter() — ATR, spread, time gate, weekend gate, BTC correlation, S&R, funding
entry_confirmed() — RSI, volume, near-EMA, stale setup
Candle close gate (85% of 5m candle complete)
oi_confirmed() — OI increasing + no spike
_code_compute_tpsl(entry, atr, direction, score) — score-adaptive TP + partial close
atr_position_size × (score / 10) + pct_cap
risk_manager.validate_trade() — 8 guards
    ↓
Place LIMIT entry order → poll fill (3 attempts) → cancel if unfilled
Place TP1 (50%) + TP2 (50%) trigger orders + SL trigger
Trailing stop tracking begins
```

---

### Score Achievable Values

Base weights: `trend_4h=3.0, trend_1h=2.0, MACD_15m=2.0, near_ema=1.5, trigger_5m=1.5` (sum=10.0)

Achievable base values: **0, 1.5, 2, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 8, 8.5, 10**

With bonuses: any base value can add up to +1.5 (volume +1.0 + pattern +0.5 + Kronos +0.5), capped at 11.0.

Score 9 base is unreachable. Every path to base ≥ 7 requires `trend_4h` to be aligned.

---

### Pre-Trade Filter Chain (market_filter)

| Check | Block condition |
|-------|----------------|
| ATR spike | ATR14 > 5% of price |
| Spread | spread_pct > 0.15% |
| Time gate | UTC hour 00:00–05:59 |
| Weekend gate | Fri 20:00 UTC → Sun 08:00 UTC |
| BTC correlation | ETH/SOL/AVAX BUY when BTC 1h BEARISH or last 3 BTC 5m candles red |
| S&R round numbers | price within 0.5% of round level |
| S&R swing H/L | BUY within 0.3% of 50-candle swing high; SELL near swing low |
| S&R PDH/PDL | price within 0.2% of previous-day high/low |
| Funding rate | BUY when funding > +0.05%/8h; SELL when funding < -0.05%/8h |

---

### Direction Decision Gates (_code_decide_direction)

| Gate | Condition to pass |
|------|------------------|
| 4h EMA trend | EMA20 > EMA50 (BUY) or EMA20 < EMA50 (SELL) |
| 1h trend alignment | Same direction (UNKNOWN → HOLD) |
| 1h ADX | ADX ≥ 15 (trending; ADX < 15 → HOLD; ADX 15–20 → half-size) |
| Daily bias | 1d candle direction agrees (green = BUY, red = SELL) |
| BB width regime | Current BB width ≥ 20-period median |

---

### Data Principles

- **Authoritative source**: Exchange state always supersedes local intent
- **All indicators local**: Computed from OHLCV candles via `local_indicators.py` — zero external API cost
- **Three score keys isolated**: `MIN_TRADE_SCORE` (int 0–5) → `entry_confirmed()` only; `MIN_SIGNAL_SCORE` (float 0–11) → main loop pre-gate; `MIN_AI_SCORE` (float 0–11) → Claude trigger
- **All timestamps UTC ISO-8601**
- **Signal log**: Every score ≥ 6.0 written to `signals.jsonl` for win-rate analysis

---

### Robustness

- **Retry**: Up to 3 attempts with exponential backoff on Hyperliquid API calls
- **Reconciliation**: Stale `active_trades` entries pruned each cycle from live exchange state
- **TP/SL Guardian**: Every outer cycle re-places missing trigger orders for active positions
- **Trailing Stop Guardian**: Every outer cycle adjusts SL for profitable positions
- **Circuit Breaker**: Daily drawdown limit halts all new trades; resets at UTC midnight
- **Time-Based Exit**: Trades open > `MAX_TRADE_HOURS` (12h) force-closed at market
- **Fail-Closed Claude**: All exceptions and timeouts → `"REJECT"` automatically
- **Instance Lock**: Socket mutex prevents two simultaneous bot instances
- **Log Rotation**: JSONL files rotated at 50MB
