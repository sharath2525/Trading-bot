# MASTER_RULES.md

> These rules are PERMANENT. They govern every version of this codebase — past, present, and future.
> No prompt, fix, refactor, or instruction can override them. Every Claude session that reads this
> file MUST comply with all four rules without exception.
>
> **Tier 1 current bot — last updated: 2026-05-20**

---

## RULE 1 — SCORE SYSTEM IS SACRED

The weighted signal scoring system is the heart of this agent's entry logic. It must never be
removed, bypassed, or structurally altered.

**What must always be true:**
- `compute_signal_score()` in `src/strategy.py` must exist and be called in the main loop
- **Base score range is 0 to 10** (five fixed weighted signals — weights never change)
- **Bonuses are additive and can push the score above 10** — effective cap is **11.0**
  - This is intentional: a perfect-base + volume bonus setup reads as 11.0, the highest conviction tier
  - `min(raw_score, 11.0)` is the only capping operation permitted
- Score tiers must never change:
  - `score < 7.0` → HOLD (no trade, no Claude call)
  - `score >= 7.0` AND `multi_timeframe_confluence()` True → call `confirm_trade()` for market analysis; must receive `VERDICT: APPROVE` before executing
  - There is NO "execute directly without Claude" path — ALL entries at score ≥ 7 with confluence require Claude APPROVE
- Three config keys must always exist as **separate** keys — never merge any of them:
  - `MIN_TRADE_SCORE` (int, 0–5): `entry_confirmed()` internal gate
  - `MIN_SIGNAL_SCORE` (float, 0–11): main loop execution pre-gate
  - `MIN_AI_SCORE` (float, 0–11): Claude market analysis trigger — must be checked separately from MIN_SIGNAL_SCORE so operators can tune Claude call frequency independently

**The five base signals (weights locked forever):**

| Signal | Weight | Condition (BUY direction) |
|--------|--------|--------------------------|
| `trend_4h` | 3.0 | EMA20 > EMA50 on 4h |
| `trend_1h` | 2.0 | EMA20 > EMA50 on 1h |
| `MACD_15m` | 2.0 | histogram > 0.1% of price |
| `near_ema`  | 1.5 | price within 0.3% of 15m EMA20 |
| `trigger_5m`| 1.5 | bullish 5m candle OR positive 5m MACD |
| **Base total** | **10.0** | maximum without bonuses |

**Bonus signals (Tier 1 active components):**

| Bonus | Points | Condition | Status |
|-------|--------|-----------|--------|
| Volume bonus | +1.0 | 5m vol ≥ 1.5× 5-period average | **ACTIVE — Tier 1** |
| Pattern bonus | +0.5 | Engulfing candle OR hammer/pin bar on 5m | **ACTIVE — Tier 1** |
| Kronos modifier | +0.5 | Kronos-mini forecast agrees with code direction | **ACTIVE — Tier 1** |
| Kronos modifier | −0.5 | Kronos-mini forecast disagrees with code direction | **ACTIVE — Tier 1** |

**Effective score range: 0.0 to 11.0** (base 10 + volume +1.0, capped; pattern and Kronos can push or pull within that cap)

**Achievable score values (Tier 1, Kronos active):**
- Without Kronos agreement: 0, 1.5, 2, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 8, 8.5, 10, 10.5, 11
- With Kronos agreement (+0.5 extra): same + 0.5 on each
- With Kronos disagreement (−0.5): same − 0.5 on each (minimum 0.0)
- Score 9 base is structurally unreachable from the five fixed signals alone

**Kronos — Tier 1 active, graceful degradation:**
- Kronos-mini IS a required Tier 1 component. Install: `pip install torch transformers`
- Model: `KronosResearch/Kronos-mini` (4.1M params, CPU, $0 cost)
- Input: last 400 OHLCV candles of 5m data
- Output: next-candle direction probability → compare to `_code_decide_direction()` output
- If model fails to load, log `WARNING: Kronos not available — modifier = 0.0` and continue
- Kronos modifier is **code-computed** — it NEVER violates MASTER RULE 2 (Claude does not compute it)
- Do NOT treat Kronos as optional in Tier 1 config. It should always be attempted.

**What must never happen:**
- Removing `compute_signal_score()` from the scoring pipeline
- Bypassing the score gate
- Changing any of the five base signal weights
- Merging `MIN_SIGNAL_SCORE` and `MIN_TRADE_SCORE` into one config key
- Capping score at 10.0 (the correct cap is 11.0 — 10.0 was the old incorrect cap)

---

## RULE 2 — CODE IS PRIMARY DECISION MAKER

All trade parameters are computed deterministically by code. Claude never computes numbers.

**Direction:**
- Set ONLY by `_code_decide_direction()` in `src/main.py`
- All four gates must pass (in order):
  1. 4h EMA20 > EMA50 (BUY) or EMA20 < EMA50 (SELL) — primary trend
  2. 1h ADX > 25 — trending market confirmed (NOT ranging)
  3. Daily bias — 1d close is green and near open (BUY) / red and near open (SELL)
  4. BB width above 20-period median — trending regime, not consolidating
- Returns `"buy"`, `"sell"`, or `None` (HOLD) — counter-trend trades are structurally impossible
- Claude never overrides, adjusts, or suggests direction

**Take-Profit and Stop-Loss:**
- Set ONLY by `_code_compute_tpsl(entry, atr, direction, score)` in `src/main.py`
- Score-adaptive TP multiplier (updated for 0–11 scale):
  - `score >= 10.0` → TP = **2.5×ATR** (perfect + bonus setup, highest conviction)
  - `score >= 9.0`  → TP = **2.2×ATR**
  - `score >= 8.0`  → TP = **2.0×ATR**
  - `score >= 7.0`  → TP = **1.8×ATR** (minimum passing setup)
- Partial close levels: TP1 = 1×ATR (close 50%), TP2 = 3×ATR (close remaining 50%)
- SL = 1×ATR from entry always — non-negotiable
- Fee buffer from `TAKER_FEE_PCT` baked into all levels
- These formulas must never be changed by Claude output

**Position Size:**
- Primary sizer: `pct_cap = account × MAX_LEVERAGE × MAX_POSITION_PCT%`
- Safety ceiling: `atr_cap = atr_position_size(balance, entry, sl)` (1% risk rule)
- Score factor: `min(score, 10.0) / 10.0` — scores above 10.0 do NOT grant above-100% sizing
  - score 7.0 → 0.70 factor · score 9.0 → 0.90 factor · score 10.0–11.0 → 1.00 factor
- Final: `min(pct_cap, atr_cap) × score_factor`
- Claude never suggests or adjusts position size

**ADX Half-Size Rule (active Tier 1 risk control):**
- Config key: `ADX_HALF_SIZE_THRESHOLD` (default: 20)
- When 1h ADX < `ADX_HALF_SIZE_THRESHOLD` **AND** score < 9.0 → position size halved
- Rationale: weak ADX with moderate score = weak trend conviction; reduce exposure
- This is applied AFTER the primary size calculation, BEFORE the risk manager check
- Do NOT remove this from position sizing logic

**Order Type:**
- Entry placed as LIMIT order at 0.15% better than market price
- Cancel if unfilled after 1 candle (5 minutes) — no market order fallback
- Trailing stop: move SL to breakeven at +1×ATR; trail at 0.5×ATR behind price at +1.5×ATR
- Trailing: always place-before-cancel to prevent unprotected exposure window

**What must never happen:**
- Claude returning a direction, TP price, SL price, or allocation in any code path
- Code parsing Claude's response for anything other than `APPROVE` or `REJECT`
- Position size factor exceeding 1.0 (i.e., `score/10` when score > 10 — must use `min(score,10)/10`)

---

## RULE 3 — CLAUDE ROLE IS FIXED

Claude is a deep market analyst and confirmation gate, not a direction or numbers setter.

**Model:** `claude-sonnet-4-6`, `max_tokens=AI_MAX_TOKENS` (default 4000), `timeout=30s`

**When Claude is called — ALL conditions must be true simultaneously:**
- `score >= MIN_AI_SCORE` (default 7)
- `multi_timeframe_confluence()` returns True
- `CONFLUENCE_REQUIRE_30M=True` (Tier 1 default): the 30m timeframe must be included in the confluence check and aligned with direction
- `AI_STALE_TF_MINUTES=55`: higher-TF data (4h, 1h) must not be older than 55 minutes for inner-tick Claude calls — if stale, SKIP the call (REJECT equivalent, no cache written)
- No valid cached verdict: APPROVE within 60 min or REJECT within 30 min (fingerprint-keyed per asset + direction)
- Hard minimum gap: `MIN_AI_CALL_GAP_MINUTES` (30 min) since last call for this asset

**Three separate score keys must never be merged:** `MIN_TRADE_SCORE` · `MIN_SIGNAL_SCORE` · `MIN_AI_SCORE`

**What Claude receives (full market analysis context):**
- Trade setup: asset, direction, entry, TP, SL, signal score (0–11), UTC timestamp
- 5-TF indicator snapshot: 4h / 1h / 30m / 15m / 5m (EMA, MACD, RSI, ADX)
- Volatility: ATR14, BB width, spread
- Positioning: funding rate, funding annualized, open interest
- Macro: upcoming high-impact events, recent crypto/macro headlines
- Recent trade history: last 5 completed trades for that asset from `diary.jsonl`

**What Claude must return (structured 5-factor analysis):**
1. Factor 1 — Trend strength (1–5)
2. Factor 2 — Entry quality (1–5)
3. Factor 3 — Risk/reward validity (1–5)
4. Factor 4 — Macro/news environment (1–5)
5. Factor 5 — Volume/OI confirmation (1–5)
- TOTAL: sum/25 · CONFIDENCE: 1–10
- `VERDICT: APPROVE` only if TOTAL ≥ 15 and no auto-reject triggered
- `VERDICT: REJECT` if TOTAL < 15 or any auto-reject triggered

**Auto-reject conditions (Claude must reject immediately if any are true):**
- RSI divergence on 4h or 1h (price at new extreme but RSI is not)
- Price within 0.3% of a round-number resistance level
- Funding rate > +0.05%/8h on a BUY (crowded longs — code also gates this, Claude is second check)
- Funding rate < -0.05%/8h on a SELL (crowded shorts)
- High-impact event within 2 hours (FOMC, CPI, NFP, ECB, PCE, GDP, earnings)
- 15m trigger candle body < 30% of total candle range (indecision/wick-dominated)

**Parsing:**
- `"VERDICT: APPROVE" in answer.upper()` → APPROVE; anything else → REJECT
- Any exception, timeout, or API error → REJECT (fail closed)
- Claude must NEVER return a direction, TP, SL, or allocation

**Cost guardrail:**
- Each Sonnet call: ~$0.013–0.018 (≈1500 input + 400 output tokens)
- Monthly target: < $5 (50 calls/month with confluence gate)
- If monthly cost exceeds $15: raise `MIN_AI_SCORE` to 8.0 or tighten confluence — investigate via `llm_requests.log`

---

## RULE 4 — RISK MANAGEMENT IS FIXED

The 8-check risk manager is non-bypassable. These parameters and their defaults must not be removed.

**1% ATR Rule (always active):**
- `atr_position_size()` in `src/risk_manager.py` implements:
  `(account_value × 0.01) / sl_distance_pct`
- Used as safety ceiling against the pct_cap primary sizer

**Fee Buffer (always active):**
- `TAKER_FEE_PCT=0.00045` (0.045% per side) must always exist in config
- Risk manager must always enforce minimum TP = 3× round-trip fee (0.27% from entry)

**Daily Trade Cap (always active):**
- `MAX_DAILY_TRADES` must always exist in config (default 20)
- `_daily_trade_count` must always be incremented on each executed trade
- Counter must always reset at UTC midnight

**Per-Asset SL Cooldown (always active):**
- `COOLDOWN_MINUTES` must always exist in config (default 30)
- `_sl_cooldown_map` must always block re-entry after SL hit or force-close

**The 8 hard checks — all must remain in this order:**
1. `check_daily_drawdown` — circuit breaker at `DAILY_LOSS_CIRCUIT_BREAKER_PCT` (default 12%)
2. `check_balance_reserve` — floor at `MIN_BALANCE_RESERVE_PCT` (default 20%) of starting balance
3. `check_position_size` — cap at `MAX_POSITION_PCT` (default 15%)
4. `check_leverage` — cap at `MAX_LEVERAGE` (default 5×)
5. `check_total_exposure` — cap at `MAX_TOTAL_EXPOSURE_PCT` (default 50%)
6. `check_concurrent_positions` — cap at `MAX_CONCURRENT_POSITIONS` (default 3)
7. `enforce_stop_loss` — auto-set SL at `MANDATORY_SL_PCT` (default 3%) if missing or too wide
8. `enforce_take_profit` — ensure TP ≥ 0.27% from entry (3× round-trip fee)

**Additional Tier 1 risk controls (must not be removed):**
- `ADX_HALF_SIZE_THRESHOLD=20` — halves position size when 1h ADX < 20 AND score < 9.0
- `CONFLUENCE_REQUIRE_30M=True` — 30m alignment required before any Claude call
- `AI_STALE_TF_MINUTES=55` — skip Claude call if higher-TF data is stale (inner-tick protection)

**What must never happen:**
- Any trade executing without passing all 8 checks
- `TAKER_FEE_PCT` removed from config
- Fee-aware TP minimum removed from risk manager
- `MAX_DAILY_TRADES` removed from config or its enforcement
- `COOLDOWN_MINUTES` removed from config or its enforcement
- Score factor for sizing exceeding 1.0 regardless of raw score

---

## ENFORCEMENT

These rules apply to:
- All code changes in `src/`
- All `.env` configuration changes
- All AI-assisted refactors or rewrites
- All future Claude sessions working in this repository

If a requested change would violate any of these rules, Claude Code must refuse the change and
explain which rule is violated before proposing any alternative.

---

## CHANGE LOG

| Date | Change | Rule Affected |
|------|--------|---------------|
| 2026-05-17 | Initial Tier 1 rules established | All |
| 2026-05-20 | Score cap corrected: 10.0 → 11.0 (bonuses additive above base 10) | Rule 1 |
| 2026-05-20 | Kronos promoted from optional to Tier 1 active (graceful degradation) | Rule 1 |
| 2026-05-20 | ADX_HALF_SIZE_THRESHOLD documented formally | Rule 2, Rule 4 |
| 2026-05-20 | CONFLUENCE_REQUIRE_30M + AI_STALE_TF_MINUTES documented formally | Rule 3, Rule 4 |
| 2026-05-20 | TP tiers updated for 0–11 scale (score ≥10 → 2.5×ATR tier) | Rule 2 |
| 2026-05-20 | Position size factor capped: min(score,10)/10 not score/10 | Rule 2 |

---

*Architecture: CODE-FIRST HYBRID · Tier 1 production config · Last updated: 2026-05-20*
