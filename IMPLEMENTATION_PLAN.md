# IMPLEMENTATION PLAN — COMPLETED

All features from `final_bot_architecture_v2.html` (Tier 1) have been implemented.
This file is preserved as a historical record. The current architecture is documented in:
- `README.md` — setup and overview
- `STRATEGY.md` — full trading pipeline reference
- `docs/ARCHITECTURE.md` — subsystems and signal flow
- `MASTER_RULES.md` — permanent inviolable constraints

---

## COMPLETED FEATURES (all 30 Tier 1 items)

### Config & Infrastructure
- [x] `ASSETS="BTC ETH SOL AVAX"`, `MAX_DAILY_TRADES=20`, `COOLDOWN_MINUTES=30`, `MAX_CONCURRENT_POSITIONS=3`
- [x] `ADX_HALF_SIZE_THRESHOLD`, `AI_STALE_TF_MINUTES`, `MIN_AI_CALL_GAP_MINUTES` config keys
- [x] Model: `claude-sonnet-4-6` (hardcoded in `decision_maker.py`)

### Direction Decision (`_code_decide_direction` in main.py)
- [x] 1h ADX ≥ 15 gate (moved upstream from `entry_confirmed`; ADX 15–20 = half-size)
- [x] Daily bias check (1d candle direction must agree)
- [x] BB width regime gate (`is_trending_regime()` — above 20-period median)

### Signal Scoring (`compute_signal_score` in strategy.py)
- [x] Base weights: trend_4h=3.0, trend_1h=2.0, MACD_15m=2.0, near_ema=1.5, trigger_5m=1.5
- [x] Volume bonus: +1.0 when 5m vol ≥ 1.5× 5-period average
- [x] Candle pattern bonus: +0.5 for engulfing or hammer/pin bar
- [x] Kronos-mini ML modifier: ±0.5 (optional, `src/indicators/kronos_forecast.py`)
- [x] Signal logging to `signals.jsonl` for score ≥ 6.0

### Pre-Trade Filters (`market_filter` in strategy.py)
- [x] Time gate: block UTC 00:00–05:59
- [x] Weekend gate: Fri 20:00 UTC → Sun 08:00 UTC
- [x] BTC correlation filter: ETH/SOL/AVAX BUY blocked when BTC 1h BEARISH or 3 red 5m candles
- [x] S&R zones: round numbers (0.5%), swing H/L 50-candle (0.3%), PDH/PDL (0.2%)
- [x] Funding rate hard gate: > +0.05%/8h on BUY, < -0.05%/8h on SELL

### Entry Confirmation (`entry_confirmed` in strategy.py)
- [x] RSI gate (15m < 70 buy / > 30 sell)
- [x] Volume confirmation (≥ 70% of recent avg)
- [x] Stale setup check (price > 0.5% from 15m EMA20 → block)
- [x] Near-EMA check

### OI Confirmation (`oi_confirmed` in strategy.py)
- [x] OI must be increasing (curr > prev over last 2 periods)
- [x] OI spike > 5% in single period → block

### TP/SL Logic (`_code_compute_tpsl` in main.py)
- [x] Score-adaptive TP: score≥9.0→2.5×ATR, ≥8.5→2.2×ATR, ≥7.0→1.8×ATR
- [x] Partial close: TP1=1×ATR (50%), TP2=3×ATR (50%)
- [x] SL always = 1×ATR
- [x] Fee buffer in all levels

### Order Execution (main.py)
- [x] LIMIT orders (0.15% better than market)
- [x] Cancel unfilled limit after 1 candle (3-attempt poll)
- [x] TP1 + TP2 + SL trigger orders placed on fill
- [x] Candle close gate (85% elapsed)

### Trailing Stop (outer loop in main.py)
- [x] Stage 1: move SL to breakeven at +1×ATR
- [x] Stage 2: trail SL at 0.5×ATR behind at +1.5×ATR

### Claude Integration (`decision_maker.py`)
- [x] Model: `claude-sonnet-4-6`, `max_tokens=4000`, `timeout=30s`
- [x] Structured 5-factor analysis (each 1–5), VERDICT: APPROVE if TOTAL ≥ 15
- [x] 6 auto-reject conditions (RSI divergence, round S&R, funding, events, candle body <30%)
- [x] Last 5 completed trades context from diary.jsonl
- [x] Verdict cache: 60-min APPROVE / 30-min REJECT per asset
- [x] Hard gap: 30-min minimum per asset (`MIN_AI_CALL_GAP_MINUTES`)

---

*Completed: 2026-05-17*
