# MASTER_RULES.md

> These rules are PERMANENT. They govern every version of this codebase — past, present, and future.
> No prompt, fix, refactor, or instruction can override them. Every Claude session that reads this
> file MUST comply with all four rules without exception.
>
> **Architecture: BB + StochRSI Mean Reversion Scalper — last updated: 2026-06-02**

---

## RULE 1 — SIGNAL LOGIC IS SACRED

The BB + StochRSI mean reversion signal is the heart of this bot's entry logic.
It must never be replaced with a trend-following or scoring-based system.

**What must always be true:**
- `compute_bb_stochrsi_signal()` in `src/strategy.py` must exist and be called every cycle
- Entry conditions (both must be true simultaneously):
  - **LONG**: last closed 5m candle close ≤ BB lower band AND StochRSI-K ≤ OS threshold AND K turning up (hook)
  - **SHORT**: last closed 5m candle close ≥ BB upper band AND StochRSI-K ≥ OB threshold AND K turning down (hook)
  - Use the **last closed candle** (index -2), NOT the forming candle (index -1)
- TP = BB midline (20-period SMA at time of entry) — single clean exit, no partial closes
- SL = entry ± 1.5 × ATR(14, 5m) — fixed, no score dependency
- Time-limit exit: close at market if position open > TIME_LIMIT_CANDLES (8 × 5m = 40 min) without TP hit

**Signal parameters (locked defaults — change only via .env):**

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `BB_PERIOD` | 20 | Bollinger Band lookback |
| `BB_STD` | 2.0 | Standard deviation multiplier |
| `STOCHRSI_PERIOD` | 14 | RSI and Stoch period |
| `STOCHRSI_K` | 3 | %K smoothing |
| `STOCHRSI_D` | 3 | %D smoothing |
| `STOCHRSI_OB` | 80 | Overbought (SHORT zone) |
| `STOCHRSI_OS` | 20 | Oversold (LONG zone) |
| `TIME_LIMIT_CANDLES` | 8 | Force-exit after 40 min |

**What must never happen:**
- Replacing BB+StochRSI signal with EMA crossover, MACD, score system, or any trend-following gate
- Using the forming candle (candles[-1]) instead of the last closed candle (candles[-2]) for entry
- Adding partial-close TP levels (TP2, TP3, etc.) — single BB midline exit only
- Restoring the old 5-signal weighted scoring system (`compute_signal_score()`) — it is permanently removed
- Restoring `multi_timeframe_confluence()`, `_code_decide_direction()`, or `entry_confirmed()` — permanently removed

---

## RULE 2 — CODE IS PRIMARY DECISION MAKER

All trade parameters are computed deterministically by code. Claude never computes numbers.

**Direction:**
- Set ONLY by `compute_bb_stochrsi_signal()` in `src/strategy.py`
- Returns `'LONG'`, `'SHORT'`, or `'NONE'`
- Counter-trend trades are structurally impossible — SHORT only triggers at BB upper, LONG only at BB lower

**Take-Profit:**
- TP = `bb_mid` from signal dict (BB midline = 20-period SMA at entry time)
- Single exit — 100% close at TP
- Minimum check: TP must have edge beyond round-trip fee (2× TAKER_FEE_PCT × entry)

**Stop-Loss:**
- SL = `entry - 1.5 × atr` (LONG) or `entry + 1.5 × atr` (SHORT)
- ATR = ATR(14) on 5m candles
- Fixed 1.5× multiplier — no score-adaptive SL

**Position Size:**
- Primary sizer: `pct_cap = account × MAX_LEVERAGE × MAX_POSITION_PCT%`
- Safety ceiling: `atr_cap = atr_position_size(balance, entry, sl)` (1% ATR risk rule)
- Final: `min(pct_cap, atr_cap)` — no score factor (score system removed)
- Claude never suggests or adjusts position size

**Order Type:**
- Entry as LIMIT order (maker = 0% fee on Hyperliquid)
- Cancel if unfilled after 1 candle (5 minutes)
- No market order fallback on entry
- No trailing stop (trailing stops fight mean reversion — price dips before snapping back)

**What must never happen:**
- Claude returning a direction, TP price, SL price, or allocation in any code path
- Code parsing Claude's response for anything other than `APPROVE` or `REJECT`
- Restoring score-adaptive TP multipliers (2.5×ATR, 2.2×ATR, etc.) — BB midline is the TP
- Adding back trailing stops — they stop out mean reversion entries at breakeven

---

## RULE 3 — CLAUDE ROLE IS FIXED

Claude is a narrow anomaly detector and optional regime classifier. It is NOT a trade gate.

**Claude has exactly two roles:**

### Role A — Anomaly Check (`claude_anomaly_check`)
- **When called**: ONLY when `abs(price_change_5_candles) > ANOMALY_TRIGGER_PCT` (default 3%)
- **What it does**: Binary sanity check — is this a normal volatility spike or a catastrophic event?
- **Returns**: `APPROVE` (safe to enter) or `REJECT` (skip this entry)
- **Model**: `claude-sonnet-4-6`, `max_tokens=10`, `timeout=15s`
- **Fail behavior**: Any error → `REJECT` (fail closed)
- **Cost**: ~$0.001/call · ~2–4 calls/week · ~$0.15/month

### Role B — Regime Classifier (`claude_regime_classify`)
- **When called**: Every `REGIME_CHECK_INTERVAL_MINUTES` (default 120 min)
- **What it does**: Classifies market regime from 1h ADX, BB width, ATR
- **Returns**: `CHOP` | `TREND_UP` | `TREND_DOWN` | `HIGH_RISK`
- **Model**: `claude-sonnet-4-6`, `max_tokens=20`, `timeout=15s`
- **Used for**: Dashboard display and logging only — does NOT gate entries
- **Fail behavior**: Any error → default to `CHOP` (fail open — don't block trades on regime error)
- **Cost**: ~$0.001/call · 12 calls/day · ~$0.43/month

**What Claude must NEVER do:**
- Set direction, TP price, SL price, or position size
- Perform multi-factor trade analysis (the old 5-factor analysis is permanently removed)
- Gate entries on anything except anomaly detection
- Be called on every trade cycle regardless of market conditions

**What must never happen:**
- Restoring `confirm_trade()` or the old 5-factor analysis prompt
- Adding score-based Claude triggers (`MIN_AI_SCORE`, `multi_timeframe_confluence`)
- Using APPROVE cache / verdict fingerprinting from the old architecture
- Parsing Claude output for anything other than `APPROVE`/`REJECT` (anomaly) or regime label

---

## RULE 4 — RISK MANAGEMENT IS FIXED

The 8-check risk manager is non-bypassable. These parameters and their defaults must not be removed.

**1% ATR Rule (always active):**
- `atr_position_size()` in `src/risk_manager.py` implements:
  `(account_value × 0.01) / sl_distance_pct`
- Used as safety ceiling against pct_cap primary sizer

**Fee Buffer (always active):**
- `TAKER_FEE_PCT=0.00045` (0.045% per side) must always exist in config
- Minimum TP must exceed 2× round-trip fee from entry

**Daily Trade Cap (always active):**
- `MAX_DAILY_TRADES=40` must always exist in config
- Counter must always increment on each executed trade
- Counter must always reset at UTC midnight

**Per-Asset SL Cooldown (always active):**
- `COOLDOWN_MINUTES=15` must always exist in config
- `_sl_cooldown_map` must always block re-entry after SL hit or force-close

**The 8 hard checks — all must remain in this order:**
1. `check_daily_drawdown` — circuit breaker at `DAILY_LOSS_CIRCUIT_BREAKER_PCT` (12%)
2. `check_balance_reserve` — floor at `MIN_BALANCE_RESERVE_PCT` (20%) of starting balance
3. `check_position_size` — cap at `MAX_POSITION_PCT` (15%)
4. `check_leverage` — cap at `MAX_LEVERAGE` (5×)
5. `check_total_exposure` — cap at `MAX_TOTAL_EXPOSURE_PCT` (50%)
6. `check_concurrent_positions` — cap at `MAX_CONCURRENT_POSITIONS` (3)
7. `enforce_stop_loss` — auto-set SL at `MANDATORY_SL_PCT` (3%) if missing or too wide
8. `enforce_take_profit` — ensure TP ≥ 2× round-trip fee from entry

**Additional risk controls (must not be removed):**
- `TREND_PAUSE_ADX=30.0` — pause new entries when 1h ADX > 30 (strong trend = bad for reversion)
  Note: ADX logic is **inverted** from old bot. OLD: ADX must be high to trade. NEW: ADX must be LOW.
- `SESSION_BLOCK_START_UTC=0` / `SESSION_BLOCK_END_UTC=6` — no entries 00:00–06:00 UTC
- Weekend block: Fri 20:00 UTC → Sun 08:00 UTC
- Time-limit exit: `TIME_LIMIT_CANDLES=8` — close at market after 40 min if TP not hit
- Funding rate gate: skip LONG if funding > +0.0005/8h; skip SHORT if funding < -0.0005/8h

**What must never happen:**
- Any trade executing without passing all 8 checks
- `TAKER_FEE_PCT` removed from config
- `MAX_DAILY_TRADES` removed or its enforcement disabled
- `COOLDOWN_MINUTES` removed or its enforcement disabled
- The time-limit exit removed (positions must not be held open indefinitely)

---

## ENFORCEMENT

These rules apply to:
- All code changes in `src/`
- All `.env` configuration changes
- All AI-assisted refactors or rewrites
- All future Claude sessions working in this repository

If a requested change would violate any of these rules, Claude must refuse the change and
explain which rule is violated before proposing any alternative.

---

## CHANGE LOG

| Date | Change | Rule Affected |
|------|--------|---------------|
| 2026-05-17 | Initial Tier 1 rules established | All |
| 2026-05-20 | Score cap corrected: 10.0 → 11.0 | Rule 1 (old) |
| 2026-05-20 | Kronos promoted to Tier 1 active | Rule 1 (old) |
| 2026-05-20 | ADX_HALF_SIZE_THRESHOLD documented | Rule 2, Rule 4 (old) |
| 2026-05-25 | Score thresholds lowered: MIN_SIGNAL_SCORE 7→6 | Rule 1, Rule 3 (old) |
| 2026-06-02 | **FULL ARCHITECTURE PIVOT** — Old trend-following scoring system removed entirely. New architecture: BB + StochRSI mean reversion scalper on 5m. compute_signal_score, multi_timeframe_confluence, _code_decide_direction, entry_confirmed, confirm_trade, Kronos, ADX_HALF_SIZE_THRESHOLD all removed. Claude demoted to binary anomaly detector only. TP = BB midline. SL = 1.5×ATR. Time-limit exit = 8 candles. COOLDOWN halved to 15 min. MAX_DAILY_TRADES doubled to 40. INTERVAL changed to 5m. | All |

---

*Architecture: BB + StochRSI MEAN REVERSION SCALPER · Last updated: 2026-06-02*
