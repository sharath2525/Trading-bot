# Trading Strategy & Risk Reference Guide

> Personal reference — understand how this bot trades, why it makes decisions,
> how fees affect every trade, and what to realistically expect.
>
> **Architecture: CODE-FIRST HYBRID — updated 2026-05-17**
> Code decides direction, TP, SL, and size. Claude (Sonnet 4.6) only APPROVE/REJECTs when
> score ≥ MIN_AI_SCORE (6) AND multi-timeframe confluence is confirmed.
> Full structured analysis: `max_tokens=4000`. Permanent rules: see `MASTER_RULES.md`.

---

## PART 1 — THE TRADING CYCLE

### Outer Loop (every 1h) + Inner Loop (every 5 minutes × 12 ticks)

**Every outer cycle:**
1. Fetch account state, force-close losers at MAX_LOSS_PER_POSITION_PCT (8%)
2. Trailing stop guardian: advance SL to breakeven (+1×ATR), then trail (at +1.5×ATR)
3. TP/SL guardian: re-place any missing trigger orders for active positions
4. Time-based exit: force-close trades open > MAX_TRADE_HOURS (12h)
5. Fetch 1d/4h/1h/30m/15m/5m candles per asset, compute all indicators locally
6. Reset daily trade counter at UTC midnight

**Every inner 5m tick (12 per hour), per asset, in strict order:**
```
_code_decide_direction()      → 4h + ADX + daily bias + BB width (hard gate)
compute_signal_score()        → weighted float 0–10 + bonuses
  < 7.0    → HOLD (no Claude, no order)
  ≥ 7.0    → continue pipeline
Daily cap + SL cooldown check
multi_timeframe_confluence()  → 4h/1h/30m/15m/5m all aligned
  → if True: confirm_trade() → VERDICT: APPROVE or REJECT
market_filter()               → time/weekend/BTC correlation/S&R/funding
entry_confirmed()             → RSI/volume/near-EMA/stale-setup
Candle close gate             → 85% of 5m candle must be complete
oi_confirmed()                → OI increasing, no spike
_code_compute_tpsl()          → score-adaptive TP + partial close levels
Position size                 → min(pct_cap, atr_cap) × (score/10)
risk_manager.validate_trade() → all 8 guards
LIMIT order + poll fill + cancel if unfilled
TP1 (50%) + TP2 (50%) trigger + SL trigger
```

---

## PART 2 — STEP-BY-STEP TRADE PIPELINE

### STEP 1: FETCH MARKET DATA
```
→ 1d, 4h, 1h, 30m, 15m, 5m candles per asset
→ Current funding rates, open interest, OI series
→ Account state: balance, positions, margin
→ Compute ALL indicators locally (zero external API cost):
    ├─ Trend:      EMA(20), EMA(50) on all timeframes
    ├─ Momentum:   RSI(14), MACD line + histogram + signal
    ├─ Volatility: ATR(14), Bollinger Bands + BB width series, ADX(14)
    └─ Volume:     OBV, VWAP, StochasticRSI
→ Bid/ask spread from exchange metadata cache
```

### STEP 2: DIRECTION GATE — `_code_decide_direction()`
```
trend_4h = "BULLISH" if EMA20_4h > EMA50_4h else "BEARISH"
trend_1h = "BULLISH" if EMA20_1h > EMA50_1h else "BEARISH"

Gate 1 — EMA trends: BUY if trend_4h BULLISH + trend_1h BULLISH
                     SELL if trend_4h BEARISH + trend_1h BEARISH
                     None (HOLD) if trends conflict or either is UNKNOWN

Gate 2 — ADX ≥ 15 on 1h: must show directional movement. HOLD if ADX < 15. ADX 15–20 → half-size.

Gate 3 — Daily bias: 1d candle must agree with direction.
  Green daily candle (close > open) → supports BUY
  Red daily candle → supports SELL
  Disagreement → HOLD

Gate 4 — BB width regime: BB width on 4h must be above its 20-period median.
  Ranging (narrow BB) → HOLD

If None → no score, no Claude, no order for this asset this tick.
```

### STEP 3: OUTER LOOP SAFETY NET
```
Force-close:
  IF position_loss_pct >= MAX_LOSS_PER_POSITION_PCT (8%):
    → Market close + SL cooldown (COOLDOWN_MINUTES = 30)

Trailing stop guardian:
  Stage 1: price moved +1×ATR from entry → move SL to breakeven
  Stage 2: price moved +1.5×ATR from entry → trail SL at 0.5×ATR behind

TP/SL Guardian:
  IF active position has no TP or SL trigger orders → re-place at original prices

Time-based exit:
  IF trade open > MAX_TRADE_HOURS (12h) and TP not hit → force market-close
```

### STEP 4: WEIGHTED SCORE GATE — `compute_signal_score()`
```
BASE WEIGHTS:
  trend_4h   : 3.0  EMA20 > EMA50 on 4h (required — hard gate)
  trend_1h   : 2.0  EMA20 > EMA50 on 1h
  MACD_15m   : 2.0  histogram > 0.1% of price
  near_ema   : 1.5  price within 0.3% of 15m EMA20
  trigger_5m : 1.5  bullish candle OR macd_5m > 0

BONUSES (additive above base, cap 11.0):
  Volume bonus    : +1.0  when 5m trigger volume ≥ 1.5× 5-period average
  Pattern bonus   : +0.5  for bullish engulfing or hammer (BUY), bearish engulfing or pin bar (SELL)
  Kronos modifier : ±0.5  from Kronos-mini ML forecast (optional — 0 if not installed)

Base achievable values: 0, 1.5, 2, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 8, 8.5, 10
Score 9 base is mathematically unreachable.

If score < MIN_SIGNAL_SCORE (6) → HOLD
If score ≥ 6 → signal logged to signals.jsonl regardless of outcome
```

### STEP 5: EXECUTION FILTERS
```
① DAILY CAP:
   If daily_trade_count >= MAX_DAILY_TRADES (20) → HOLD

② SL COOLDOWN:
   If asset SL cooldown active (< COOLDOWN_MINUTES=30 since last SL hit) → HOLD

③ MULTI-TIMEFRAME CONFLUENCE (multi_timeframe_confluence in main.py):
   4h EMA trend aligned
   1h EMA trend aligned
   15m MACD histogram direction aligned
   5m candle or MACD direction aligned
   If all pass AND score ≥ MIN_AI_SCORE → call Claude

④ CLAUDE ANALYSIS (confirm_trade in decision_maker.py):
   Model: claude-sonnet-4-6, max_tokens=4000, timeout=30s
   Verdict cache: 60-min APPROVE / 30-min REJECT per asset
   Hard gap: MIN_AI_CALL_GAP_MINUTES (30) per asset
   Response: structured 5-factor analysis, TOTAL/25
   APPROVE only if TOTAL ≥ 15 and no auto-reject
   Fail closed: any error/timeout → REJECT

⑤ MARKET FILTER (market_filter in strategy.py):
   ATR spike > 5% → HOLD
   Spread > 0.15% → HOLD
   UTC hour 00:00–05:59 → HOLD (low liquidity)
   Fri 20:00 UTC → Sun 08:00 UTC → HOLD (weekend gate)
   ETH/SOL/AVAX BUY when BTC 1h BEARISH → HOLD (correlation)
   ETH/SOL/AVAX BUY when last 3 BTC 5m candles red → HOLD
   Price within 0.5% of round S&R level → HOLD
   BUY within 0.3% of 50-candle swing high → HOLD
   SELL within 0.3% of 50-candle swing low → HOLD
   Price within 0.2% of PDH/PDL → HOLD
   Funding > +0.05%/8h on BUY → HOLD (crowded longs)
   Funding < -0.05%/8h on SELL → HOLD (crowded shorts)

⑥ ENTRY CONFIRMATION (entry_confirmed in strategy.py):
   For BUY — all must be true:
     • inner score ≥ MIN_TRADE_SCORE (2) — 0-5 integer gate
     • 15m RSI < 78 (not extreme overbought)
     • price within 0.5% of 15m EMA20 OR 15m MACD > threshold (OR logic)
     • 5m: bullish candle OR positive MACD
     • 5m trigger volume ≥ 70% of recent average (real pressure)
   For SELL — mirror bearish conditions
   Missing data → False (fail closed)

⑦ CANDLE CLOSE GATE:
   85% of the current 5m candle must be elapsed before entry.

⑧ OI CONFIRMATION (oi_confirmed in strategy.py):
   OI series must show OI increasing (curr > prev over last 2 periods)
   OI spike > 5% in single period → HOLD (manipulation risk)
   Missing OI data → pass (don't block)
```

### STEP 6: CODE COMPUTES TRADE PARAMETERS
```
entry_price = limit price = market price × (1 − 0.0015) for BUY
                          = market price × (1 + 0.0015) for SELL
atr         = ATR14 from 4h candles

Score-adaptive TP:
  score ≥ 10.0 → tp_mult = 2.5×ATR
  score ≥ 9.0  → tp_mult = 2.2×ATR
  score ≥ 8.0  → tp_mult = 2.0×ATR
  score ≥ 7.0  → tp_mult = 1.8×ATR
  score ≥ 6.0  → tp_mult = 1.6×ATR

Partial close levels (both placed as reduce-only triggers):
  TP1 = entry + 1.0×ATR  (close 50% of position — lock profit)
  TP2 = entry + 3.0×ATR  (close remaining 50% — let winner run)

SL = entry − 1.0×ATR (always; MANDATORY_SL_PCT=3% backstop)

All levels include fee buffer: entry × TAKER_FEE_PCT × 2

Position sizing (notional value):
  pct_cap  = account × MAX_LEVERAGE × (MAX_POSITION_PCT / 100)  e.g. $100×5×15% = $75
  atr_cap  = atr_position_size(balance, entry, sl)               1% risk rule ceiling
  alloc    = min(pct_cap, atr_cap) × (score / 10)               scale by signal strength

Example — $100 account, score=10, SL at 1% from entry:
  pct_cap = $75 | atr_cap = $100 → alloc = $75 × 1.0 = $75 notional
Example — $100 account, score=7:
  alloc = $75 × 0.70 = $52.50 notional → ~$10.50 margin at 5×
```

### STEP 7: CLAUDE ANALYSIS GATE — `confirm_trade()`
```
Called when: score ≥ MIN_AI_SCORE (6) AND multi_timeframe_confluence() returns True

Model: claude-sonnet-4-6, max_tokens=4000, timeout=30s

Claude receives:
  - Trade setup (asset, direction, entry, TP, SL, score, UTC time)
  - 5-TF indicator snapshot (4h / 1h / 30m / 15m / 5m indicators)
  - ATR14, BB width, spread
  - Funding rate, funding annualized, open interest
  - Macro: upcoming high-impact events + recent headlines
  - Last 5 completed trades for this asset (from diary.jsonl)

Claude must output structured 5-factor analysis:
  Factor 1 — Trend strength: /5
  Factor 2 — Entry quality: /5
  Factor 3 — Risk/reward validity: /5
  Factor 4 — Macro/news environment: /5
  Factor 5 — Volume/OI confirmation: /5
  TOTAL: /25 · CONFIDENCE: 1–10
  VERDICT: APPROVE (only if TOTAL ≥ 15 and no auto-reject triggered)
  VERDICT: REJECT  (if TOTAL < 15 or any auto-reject)

Auto-reject conditions (immediate REJECT if any true):
  • RSI divergence on 4h or 1h
  • Price within 0.3% of round-number resistance
  • Funding > +0.05%/8h on BUY
  • Funding < -0.05%/8h on SELL
  • High-impact event within 2h (FOMC/CPI/NFP/ECB/PCE/GDP/earnings)
  • 15m trigger candle body < 30% of total range (indecision)

Verdict cache:
  APPROVE cached 60 min per asset (keyed by score bucket + trend + hour)
  REJECT  cached 30 min per asset

Hard gap: MIN_AI_CALL_GAP_MINUTES (30) between Claude calls per asset
```

### STEP 8: RISK MANAGER — `validate_trade()`
```
CHECK 1: Daily loss circuit breaker
  Today's loss ≥ DAILY_LOSS_CIRCUIT_BREAKER_PCT (12%) → BLOCK all new trades

CHECK 2: Balance reserve
  balance < MIN_BALANCE_RESERVE_PCT (20%) of starting balance → BLOCK

CHECK 3: Position size cap
  alloc > pct_cap → cap at pct_cap

CHECK 4: Effective leverage cap
  effective_leverage = alloc / account_value > MAX_LEVERAGE (5) → BLOCK

CHECK 5: Total exposure cap
  total_notional / account > MAX_TOTAL_EXPOSURE_PCT (50%) → BLOCK

CHECK 6: Concurrent positions
  open_positions ≥ MAX_CONCURRENT_POSITIONS (3) → BLOCK

CHECK 7: Stop-loss enforcement
  sl_price missing or too wide → auto-set at MANDATORY_SL_PCT (3%) from entry

CHECK 8: TP fee coverage
  (tp_price − entry) / entry < 0.0027 → adjust TP upward (3× round-trip fee)
```

### STEP 9: EXECUTE ON HYPERLIQUID
```
1. Place LIMIT order (entry price = market ± 0.15%)
2. Poll for fill: 3 attempts × ~3s wait
3. If unfilled after 1 candle (5m): cancel order → HOLD
4. On fill:
   a. Place TP1 trigger order (reduce-only, 50% size at 1×ATR)
   b. Place TP2 trigger order (reduce-only, 50% size at 3×ATR)
   c. Place SL trigger order (reduce-only, 100% size at 1×ATR)
   d. Record tp1_oid, tp2_oid, sl_oid, entry_time in active_trades
   e. Log to diary.jsonl
   f. _daily_trade_count += 1
```

---

## PART 3 — LEVERAGE & EXCHANGE SETUP

5× leverage is set as cross-margin leverage on Hyperliquid at startup and re-verified every outer cycle.

| Account | Buying Power | Target Notional (score=10) | Margin Used |
|---------|-------------|---------------------------|-------------|
| $100 | $500 | $75 | ~$15 |
| $200 | $1,000 | $150 | ~$30 |
| $500 | $2,500 | $375 | ~$75 |
| $1,000 | $5,000 | $750 | ~$150 |

Score scaling reduces notional further: score-7 = 70%, score-8.5 = 85%, score-10 = 100%.

### Risk-Reward
- SL = 1×ATR14 (always)
- TP1 = 1×ATR14 (50% close — 1:1 on first half)
- TP2 = 3×ATR14 (remaining 50% — 1:3 on second half)
- Weighted average R:R ≈ 1:2

---

## PART 4 — FEE MATH

| Order type | Fee |
|-----------|-----|
| Limit order (maker, resting on book) | 0% |
| Market order / trigger fill (taker) | 0.045% |
| Round-trip (open + close, both taker) | 0.09% |

**Minimum TP to cover fees:** TP must be ≥ 0.27% from entry (3× round-trip fee).
Code sets TP via ATR; risk manager enforces the 0.27% minimum.
Limit entry (0% fee) saves ~$0.045/trade vs market — compounds over many trades.

---

## PART 5 — TRADE LIFECYCLE

### Opening a Trade
```
1. _code_decide_direction()         → "buy" or "sell" (4 gates)
2. compute_signal_score()           → float 0–10 + bonuses
3. score < 6 → HOLD
4. Daily cap + SL cooldown
5. multi_timeframe_confluence()
6. confirm_trade() → VERDICT: APPROVE required
7. market_filter()                  → 9 checks
8. entry_confirmed()                → 7 conditions
9. Candle close gate (85%)
10. oi_confirmed()
11. _code_compute_tpsl()            → score-adaptive TP + partial close
12. Position size                   → min(pct_cap, atr_cap) × (score/10)
13. risk_manager.validate_trade()   → 8 guards
14. LIMIT order → poll fill → cancel if unfilled
15. TP1 (50%) + TP2 (50%) + SL triggers placed
```

### During the Trade
- Every outer loop: trailing stop advanced if price moved favorably
- Every outer loop: TP/SL guardian re-places missing orders
- Score checked each 5m — new entries possible on other assets if score qualifies

### Closing
1. **TP1 fills** at +1×ATR — 50% closed; remaining 50% continues
2. **TP2 fills** at +3×ATR — remaining 50% closed at full profit
3. **SL fills** — position closed at loss → 30-min cooldown for that asset
4. **Trailing SL** — breakeven or trailed SL may fill before TP2 (locks partial profit)
5. **Force-close** — MAX_LOSS_PER_POSITION_PCT (8%) → market close → cooldown
6. **Time-based exit** — open > MAX_TRADE_HOURS (12h) → market close

---

## PART 6 — API COST

Claude Sonnet 4.6 pricing: ~$3/M input + $15/M output tokens.
Each full analysis call (~1500 input + 500 output tokens) ≈ $0.012.

| Scenario | Monthly calls | Cost/month |
|----------|--------------|-----------|
| Score rarely reaches 7+ | ~5–20 | ~$0.06–0.24 |
| Moderate setups | ~30–60 | ~$0.36–0.72 |
| Active market | ~100 | ~$1.20 |

**If monthly cost exceeds $30:** Check `MIN_AI_CALL_GAP_MINUTES=30` is set and `MIN_AI_SCORE=6`.
Check `llm_requests.log` for per-call frequency.

---

## PART 7 — SCORE SYSTEM IN DETAIL

### Score Achievable Values (base, before bonuses)

| Conditions met | Base Score | Action |
|----------------|-----------|--------|
| None | 0 | HOLD |
| trigger_5m only | 1.5 | HOLD |
| near_ema only | 1.5 | HOLD |
| trend_1h only | 2.0 | HOLD |
| trend_4h + trigger_5m | 4.5 | HOLD |
| trend_4h + near_ema | 4.5 | HOLD |
| trend_4h + trend_1h | 5.0 | HOLD |
| trend_4h + trend_1h + near_ema | 6.5 | HOLD |
| trend_4h + trend_1h + trigger_5m | 6.5 | HOLD |
| trend_4h + trend_1h + MACD_15m | 7.0 | → pipeline |
| trend_4h + trend_1h + MACD_15m + near_ema | 8.5 | → pipeline |
| trend_4h + trend_1h + MACD_15m + trigger_5m | 8.5 | → pipeline |
| All 5 signals | 10.0 | → pipeline |

With volume bonus (+1.0) or pattern bonus (+0.5): base scores can increase further, capped at 11.0.

### Three Separate Score Keys — Never Merge

| Key | Range | Used by | Purpose |
|-----|-------|---------|---------|
| `MIN_TRADE_SCORE` | 0–5 int | `entry_confirmed()` | Inner 0-5 gate |
| `MIN_SIGNAL_SCORE` | 0–11 float | main loop pre-gate | HOLD/proceed threshold |
| `MIN_AI_SCORE` | 0–11 float | `confirm_trade()` call | When to trigger Claude |

---

## PART 8 — HEALTH CHECKLIST

### Signs the Bot Is Working Well
- Most inner ticks result in HOLD (high hold rate is correct and healthy)
- Each executed trade has TP1, TP2, and SL trigger orders placed
- `[TRADE]` log lines show direction consistent with `trend_4h`
- `[DIRECTION]` blocks from ADX/daily-bias/BB-regime are normal and healthy
- `[ENTRY]` blocks appear when price overextended — correct behavior
- `[SPREAD]` blocks on wide-spread assets — correct behavior
- `[COOLDOWN]` blocks after SL exits — correct behavior
- `[TIME GATE]` blocks during 00:00–06:00 UTC or weekends — correct behavior
- `[WEEKEND]` blocks on Friday evening through Sunday morning — correct behavior
- `[OI]` blocks when OI flat or spiking — correct behavior
- `[DAILY CAP]` blocks once MAX_DAILY_TRADES (20) reached
- `[CIRCUIT BREAKER]` fires when daily loss threshold hit

### Signs Something Is Wrong
- API costs > $30/month → Claude called too frequently; check gap and score settings
- Force-closes frequent → MAX_LOSS_PER_POSITION_PCT too low or leverage too high
- `[TIMEOUT]` every cycle → trades never reaching TP; tighten TP or reduce position size
- Daily cap hitting on day 1 → MAX_DAILY_TRADES too low for signal frequency
- `INVERSION BUG DETECTED` in logs → **stop immediately and report**
- Limit orders never filling → entry offset 0.15% may be too tight; consider widening
