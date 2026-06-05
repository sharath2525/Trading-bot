# Trading Bot — Final Fix Report (v3: Weighted Scoring + Code-First Architecture)

> **Status:** Complete redesign spec. All agreed decisions incorporated.
> **Date:** 2026-04-29
> **Scope:** strategy.py · main.py · decision_maker.py · config_loader.py · .env

---

## 1. What Changed From v2 (Quick Reference)

| Area | Old (v2) | New (v3) |
|------|----------|----------|
| Signal scoring | 0–5 integer, +1 per condition | 0–10 float, weighted (3+2+2+1.5+1.5) |
| 4h trend | One score point among five | Hard gate — blocks entry if misaligned |
| Claude usage | Called every cycle for all assets | Called only when score == 10 |
| Claude role | Full trading brain (action + size + prices) | APPROVE / REJECT only (10 tokens) |
| Config keys | MIN_TRADE_SCORE=7 (conflicts with 0–5 max) | MIN_TRADE_SCORE=3 + MIN_SIGNAL_SCORE=7 (separate systems) |
| Execute ordering | Claude called after placing order (bug) | Claude gates execution (Claude → validate → execute) |
| Inner loop | asyncio.sleep(3600) — misses moves | 5m sub-loop checks score every 5 minutes |
| API cost | ~$13–40 / month (Claude every cycle) | ~$0.01 / month (Claude only on perfect setups) |

---

## 2. Architecture: Data → Decision → Execution → State → Logging

```
┌─────────────────────────────────────────────────────────────────────┐
│  OUTER LOOP  (runs once per hour)                                   │
│                                                                     │
│  1. Fetch account state (balance, positions, fills)                 │
│  2. Force-close any position >= MAX_LOSS_PER_POSITION_PCT           │
│  3. Fetch 5m candles + 4h candles per asset (8 parallel calls)      │
│  4. Compute all indicators locally (EMA, RSI, MACD, ATR, ADX)      │
│  5. Record 1h snapshot (trend_1h, trend_4h, intraday_1h)           │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  INNER LOOP  (runs every 5 minutes for 12 iterations)        │   │
│  │                                                              │   │
│  │  For each asset:                                             │   │
│  │                                                              │   │
│  │  STEP 1 — 4h HARD GATE                                       │   │
│  │    direction = _code_decide_direction(asset_data)            │   │
│  │    if direction != trend_4h alignment → HOLD, next asset     │   │
│  │                                                              │   │
│  │  STEP 2 — WEIGHTED SCORE (0–10)                              │   │
│  │    score = compute_signal_score(asset_data, direction)       │   │
│  │    if score < MIN_SIGNAL_SCORE (7) → HOLD, next asset        │   │
│  │                                                              │   │
│  │  STEP 3 — ENTRY FILTERS                                      │   │
│  │    market_filter() — ATR spike, spread check                 │   │
│  │    entry_confirmed() — RSI, ADX, volume, MIN_TRADE_SCORE=3   │   │
│  │    if any fail → HOLD, next asset                            │   │
│  │                                                              │   │
│  │  STEP 4 — CODE COMPUTES TRADE PARAMETERS                     │   │
│  │    direction = BUY / SELL                                    │   │
│  │    size = risk_manager.atr_position_size() (1% risk rule)    │   │
│  │    tp = entry + (2 x ATR)                                    │   │
│  │    sl = entry - (1 x ATR)                                    │   │
│  │                                                              │   │
│  │  STEP 5 — CLAUDE GATE  (score == 10 ONLY)                    │   │
│  │    if score == 10:                                           │   │
│  │      result = confirm_trade(asset, direction, entry,         │   │
│  │                             tp, sl, score, indicators)       │   │
│  │      if result != "APPROVE" → HOLD (fail closed)            │   │
│  │                                                              │   │
│  │  STEP 6 — RISK MANAGER VALIDATION                            │   │
│  │    risk_manager.validate_trade() — all 8 guards              │   │
│  │    if blocked or capped → adjust or HOLD                     │   │
│  │                                                              │   │
│  │  STEP 7 — EXECUTE                                            │   │
│  │    place market order + TP trigger + SL trigger              │   │
│  │    write to diary.jsonl                                      │   │
│  │                                                              │   │
│  │  STEP 8 — RECONCILE                                          │   │
│  │    clean stale active-trade records                          │   │
│  │    write cycle summary to decisions.jsonl                    │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Sleep 5 minutes → repeat inner loop (up to 12 times)              │
│  After 12 iterations → back to outer loop (next hour)              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. New Scoring System (0–10 Weighted Float)

### 3.1 Score Weights

| Condition | Direction | Weight |
|-----------|-----------|--------|
| trend_4h == direction | BUY: BULLISH · SELL: BEARISH | **3.0** |
| trend_1h == direction | BUY: BULLISH · SELL: BEARISH | **2.0** |
| macd_15m aligned | BUY: histogram > threshold · SELL: < -threshold | **2.0** |
| near_ema == True | Same for both directions | **1.5** |
| trigger_5m aligned | BUY: bull candle OR macd_5m>0 · SELL: bear OR macd_5m<0 | **1.5** |
| **Maximum possible** | All 5 conditions met | **10.0** |

### 3.2 Score Tiers and Behaviour

| Score | Tier | Action |
|-------|------|--------|
| < 7.0 | Weak setup | **HOLD** — no trade, no Claude |
| 7.0 – 8.5 | Strong setup | **Execute directly** — code handles parameters |
| 10.0 | Perfect setup | **Claude APPROVE/REJECT first**, then execute if APPROVE |

> **Note:** Score 9 is mathematically unreachable. With binary conditions and weights 3, 2, 2, 1.5, 1.5
> the achievable values are: 0, 1.5, 2, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 8, 8.5, 10.
> There is no combination that produces a value between 8.5 and 10. Score >= 9 in practice means score == 10 only.

### 3.3 Reachable Score Combinations (All Paths to >= 7)

| trend_4h | trend_1h | MACD_15m | EMA | Trigger | Score |
|----------|----------|----------|-----|---------|-------|
| YES (3) | YES (2) | YES (2) | NO | NO | 7.0 |
| YES (3) | YES (2) | YES (2) | YES (1.5) | NO | 8.5 |
| YES (3) | YES (2) | YES (2) | NO | YES (1.5) | 8.5 |
| YES (3) | YES (2) | YES (2) | YES (1.5) | YES (1.5) | **10.0** |

**Critical observation:** Every path to score >= 7 requires trend_4h (weight 3) to be aligned.
The 4h hard gate and the weight structure together make counter-trend trades impossible above threshold.

### 3.4 The Counter-Trend Bug (Why the 4h Hard Gate Exists)

Without the hard gate, the following combination reaches score 7.0:

```
trend_4h = BEARISH  (counter-trend for a BUY)
trend_1h = BULLISH  → +2
MACD_15m = positive → +2
near_ema = True     → +1.5
trigger_5m = bullish→ +1.5
TOTAL = 7.0  ← would execute, but 4h trend is against us
```

Fix: Before computing the score at all, check if direction aligns with trend_4h.
If not → HOLD immediately. This is enforced in STEP 1 of the inner loop.

---

## 4. Two-Config Solution (Prevents Silent Failure)

### 4.1 The Problem

The old config had one key: MIN_TRADE_SCORE. If this was set to 7 (appropriate for
the new 0–10 system), it was passed into entry_confirmed(), which internally uses
_compute_signal_score() — the old 0–5 system. Since 5 (max possible old score) < 7
(threshold), every trade is silently blocked with zero error messages and zero trades.

### 4.2 The Fix: Two Separate Config Keys

```
MIN_TRADE_SCORE = 3   # Used by entry_confirmed() — old 0–5 integer system
MIN_SIGNAL_SCORE = 7  # Used by pre-gate — new 0–10 weighted float system
```

These operate at different points in the flow:

```
compute_signal_score() → MIN_SIGNAL_SCORE gate → [score >= 7 passes] →
entry_confirmed() uses _compute_signal_score() internally → MIN_TRADE_SCORE gate
```

Both gates must pass independently. Neither replaces the other.

### 4.3 config_loader.py Change Required

One line must be added to the CONFIG dict in src/config_loader.py:

```python
# EXISTING (keep as-is):
"min_trade_score": _get_int("MIN_TRADE_SCORE", 3),

# ADD THIS LINE:
"min_signal_score": _get_int("MIN_SIGNAL_SCORE", 7),
```

Without this addition, MIN_SIGNAL_SCORE=7 in .env has no effect — the key is never read.

---

## 5. Claude's New Role: APPROVE / REJECT Only

### 5.1 When Claude Is Called

| Condition | Claude Called? |
|-----------|---------------|
| Score < 7 | No |
| Score 7–8.5 | No |
| Score == 10 | **Yes — APPROVE/REJECT** |
| Score == 10, Claude errors | No — REJECT (fail closed) |

### 5.2 What Claude Receives

```
System: "You are a trading risk validator. Given a trade setup, respond with 
         exactly one word: APPROVE or REJECT. Nothing else."

User:   "Asset: BTC  Direction: BUY  Entry: 94200  TP: 96400  SL: 93100
         Score: 10/10  trend_4h: BULLISH  trend_1h: BULLISH  
         MACD_15m: +42.3  near_ema: true  RSI: 58  ADX: 34"
```

### 5.3 confirm_trade() Method (add to decision_maker.py)

```python
def confirm_trade(self, asset, direction, entry_price, tp_price, sl_price,
                  score, indicators) -> str:
    """Call Claude Haiku for final APPROVE/REJECT. Fail closed on any error."""
    _haiku = "claude-haiku-4-5-20251001"
    system_prompt = (
        "You are a trading risk validator. Given a trade setup, respond with "
        "exactly one word: APPROVE or REJECT. Nothing else."
    )
    user_msg = (
        f"Asset: {asset}  Direction: {direction}  Entry: {entry_price}  "
        f"TP: {tp_price}  SL: {sl_price}  Score: {score}/10\n"
        f"Indicators: {indicators}"
    )
    try:
        response = self.client.messages.create(
            model=_haiku,
            max_tokens=10,
            system=system_prompt,
            messages=[{"role": "user", "content": user_msg}],
            timeout=15,
        )
        answer = response.content[0].text.strip().upper()
        return "APPROVE" if answer == "APPROVE" else "REJECT"
    except Exception as e:
        logging.warning("[CLAUDE CONFIRM] Error — failing closed: %s", e)
        return "REJECT"
```

### 5.4 Cost Analysis

| Scenario | Claude Calls | Cost Per Call | Monthly Cost |
|----------|-------------|--------------|-------------|
| Old design (every cycle) | ~8,640/month | ~$0.003 | ~$26/month |
| New design (score==10 only) | ~5–20/month | ~$0.000004 | **~$0.01/month** |

Haiku pricing: $0.80/million input tokens. A 150-token call costs $0.00000012.
Even 1,000 confirmations/month = $0.00012.

---

## 6. Corrected Execution Order

### 6.1 The Bug (Old Flow)

```
Step 5: execute_trade(outputs)   ← order placed here
Step 6: Claude decides           ← Claude called AFTER order already in market
```

This was in main.py — Claude's output was used for logging, not actually gating execution.

### 6.2 Fixed Flow

```
Score >= 7
  → entry_confirmed() passes
    → code computes: direction, size, tp, sl
      → [if score == 10] Claude.confirm_trade() → must return "APPROVE"
        → risk_manager.validate_trade() — 8 guards
          → execute_trade()
```

Claude now sits between parameter computation and execution. It cannot be bypassed.
If it returns anything other than "APPROVE" (including errors, timeouts, empty responses),
the trade is blocked.

---

## 7. 5-Minute Inner Loop (Replaces asyncio.sleep(3600))

### 7.1 Why It Matters

The current code does `await asyncio.sleep(3600)` at the end of each cycle. This means
during a strong trending hour, the bot enters once at the start and then sleeps through
all subsequent setups. The 5m inner loop fixes this.

### 7.2 Structure

```python
# Replace the single sleep at end of main loop with:
for _ in range(12):                          # 12 x 5min = 60min (1 outer cycle)
    await asyncio.sleep(300)                 # 5 minutes
    
    for asset in assets:
        # Refresh 5m candles only (cheap, no 4h refetch)
        fresh_data = await fetch_5m_snapshot(asset)
        asset_data[asset].update(fresh_data)
        
        # Re-run the full scoring pipeline (Steps 1–7 from Section 2)
        await evaluate_and_trade(asset, asset_data[asset])
```

The outer 1h loop refreshes 4h candles, computes 1h trend, and resets daily drawdown tracking.
The inner 5m loop only refreshes 5m candles and re-runs the decision pipeline.

---

## 8. Risk Manager — No Changes Required

All 8 guards remain as-is. They execute in validate_trade() after Claude confirmation:

1. Single-position size cap (MAX_POSITION_PCT)
2. Effective leverage cap (MAX_LEVERAGE)
3. Total portfolio exposure cap (MAX_TOTAL_EXPOSURE_PCT)
4. Daily drawdown circuit breaker
5. Concurrent position limit (MAX_CONCURRENT_POSITIONS)
6. Minimum balance reserve (MIN_BALANCE_RESERVE_PCT)
7. Mandatory stop-loss (auto-set if missing, ATR-aware)
8. Force-close at position loss threshold

atr_position_size(), enforce_stop_loss(), and enforce_take_profit() are already implemented
correctly and are used by the code-computes-parameters step.

---

## 9. Files to Change — Complete List

### 9.1 src/config_loader.py — ADD one line

```python
# In the CONFIG dict, after min_trade_score:
"min_signal_score": _get_int("MIN_SIGNAL_SCORE", 7),
```

### 9.2 src/strategy.py — REWRITE scoring function

```python
# Add new public function (keep old _compute_signal_score unchanged for entry_confirmed):

def compute_signal_score(asset_data: dict, direction: str) -> float:
    """Return a weighted score 0–10 counting aligned entry conditions.
    
    Weights: trend_4h=3, trend_1h=2, MACD_15m=2, EMA_15m=1.5, Trigger_5m=1.5
    MIN_SIGNAL_SCORE (default 7) sets the minimum to allow execution.
    This is the new public function used by main.py's pre-gate.
    entry_confirmed() continues to use the old _compute_signal_score() internally.
    """
    s15 = asset_data.get("setup_15m", {})
    t5  = asset_data.get("trigger_5m", {})
    current_price = float(asset_data.get("current_price") or 0)
    macd_threshold = current_price * 0.001 if current_price > 0 else 0.0

    macd_15m = float(s15.get("macd_histogram") or 0)
    near_ema  = bool(s15.get("near_ema", False))
    macd_5m   = float(t5.get("macd_histogram") or 0)
    bull_5m   = bool(t5.get("candle_bullish", False))
    trend_4h  = asset_data.get("trend_4h", "UNKNOWN")
    trend_1h  = asset_data.get("trend_1h", "UNKNOWN")

    score = 0.0
    if direction == "buy":
        if trend_4h == "BULLISH":           score += 3.0
        if trend_1h == "BULLISH":           score += 2.0
        if macd_15m > macd_threshold:       score += 2.0
        if near_ema:                        score += 1.5
        if bull_5m or macd_5m > 0:         score += 1.5
    elif direction == "sell":
        if trend_4h == "BEARISH":           score += 3.0
        if trend_1h == "BEARISH":           score += 2.0
        if macd_15m < -macd_threshold:      score += 2.0
        if near_ema:                        score += 1.5
        if (not bull_5m) or macd_5m < 0:   score += 1.5
    return score
```

Note: Add compute_signal_score as a new function. Do NOT rename or modify _compute_signal_score.
entry_confirmed() must continue to call _compute_signal_score() for the old 0–5 system.

### 9.3 src/main.py — FOUR changes

**Change A: Add helper functions** (insert near top of trading logic section):

```python
def _code_decide_direction(asset_data: dict) -> str | None:
    """Return 'buy', 'sell', or None based on trend alignment."""
    trend_4h = asset_data.get("trend_4h", "UNKNOWN")
    trend_1h = asset_data.get("trend_1h", "UNKNOWN")
    if trend_4h == "BULLISH" and trend_1h in ("BULLISH", "UNKNOWN"):
        return "buy"
    if trend_4h == "BEARISH" and trend_1h in ("BEARISH", "UNKNOWN"):
        return "sell"
    return None  # conflicting trends → no trade

def _code_compute_tpsl(entry: float, atr: float, direction: str) -> tuple[float, float]:
    """Return (tp, sl) using 2:1 ATR ratio."""
    if direction == "buy":
        return entry + 2 * atr, entry - 1 * atr
    return entry - 2 * atr, entry + 1 * atr

def _code_compute_allocation(asset_data: dict, balance: float, risk_manager) -> float:
    """Return position size using ATR 1% risk rule."""
    atr = float(asset_data.get("long_term_4h", {}).get("atr14") or 0)
    price = float(asset_data.get("current_price") or 0)
    if atr <= 0 or price <= 0:
        return 0.0
    return risk_manager.atr_position_size(balance, atr, price)
```

**Change B: Replace unconditional Claude call** (around line 869 — the decide_trade() call):

```python
# REMOVE THIS:
outputs = await asyncio.to_thread(agent.decide_trade, args.assets, context)

# REPLACE WITH (full pipeline per asset):
for asset in assets:
    ad = context.get(asset, {})
    
    # Step 1: 4h hard gate
    direction = _code_decide_direction(ad)
    if direction is None:
        logging.debug("[%s] 4h hard gate: conflicting trends → HOLD", asset)
        continue
    
    # Step 2: weighted score gate
    score = compute_signal_score(ad, direction)
    min_sig = int(CONFIG.get("min_signal_score") or 7)
    if score < min_sig:
        logging.debug("[%s] score %.1f < %.1f → HOLD", asset, score, min_sig)
        continue
    
    # Step 3: entry filters
    allowed, reason = market_filter(ad)
    if not allowed:
        logging.info("[%s] market_filter blocked: %s", asset, reason)
        continue
    if not entry_confirmed(ad, direction):
        logging.debug("[%s] entry_confirmed failed → HOLD", asset)
        continue
    
    # Step 4: code computes parameters
    entry_price = float(ad.get("current_price") or 0)
    atr = float(ad.get("long_term_4h", {}).get("atr14") or 0)
    tp_price, sl_price = _code_compute_tpsl(entry_price, atr, direction)
    allocation = _code_compute_allocation(ad, account_balance, risk_manager)
    if allocation <= 0:
        continue
    
    # Step 5: Claude confirmation (score == 10 ONLY)
    if score >= 10.0:
        indicators_summary = {
            "trend_4h": ad.get("trend_4h"),
            "trend_1h": ad.get("trend_1h"),
            "rsi_15m": ad.get("setup_15m", {}).get("rsi14"),
            "adx_1h": ad.get("intraday_1h", {}).get("adx"),
            "macd_15m": ad.get("setup_15m", {}).get("macd_histogram"),
        }
        verdict = agent.confirm_trade(
            asset, direction, entry_price, tp_price, sl_price,
            score, indicators_summary
        )
        if verdict != "APPROVE":
            logging.info("[%s] Claude rejected score-10 setup", asset)
            continue
    
    # Step 6: risk manager validation
    trade_proposal = {
        "asset": asset, "action": direction,
        "allocation_pct": allocation,
        "stop_loss_pct": abs(entry_price - sl_price) / entry_price * 100,
    }
    validated = risk_manager.validate_trade(trade_proposal, context)
    if not validated.get("allowed"):
        logging.info("[%s] risk_manager blocked: %s", asset, validated.get("reason"))
        continue
    
    # Step 7: execute
    await execute_trade(asset, direction, validated["allocation_pct"],
                        entry_price, tp_price, sl_price)
```

**Change C: Remove retry block** (around line 879–894):

```python
# DELETE the retry block entirely:
# if not outputs or len(outputs) == 0:
#     outputs = await asyncio.to_thread(agent.decide_trade, ...)
```

**Change D: Replace asyncio.sleep(3600) with 5m inner loop** (around line 1190):

```python
# REMOVE:
await asyncio.sleep(_interval_seconds)

# REPLACE WITH:
for _tick in range(12):
    await asyncio.sleep(300)  # 5 minutes
    logging.info("[TICK %d/12] refreshing 5m data", _tick + 1)
    for asset in args.assets:
        fresh_5m = await hyperliquid_api.get_candles(asset, "5m", 20)
        if fresh_5m:
            context[asset]["candles_5m"] = fresh_5m
        # Re-run full evaluation pipeline (same Steps 1–7 as Change B above)
        await evaluate_and_trade(asset, context[asset])
```

### 9.4 src/agent/decision_maker.py — ADD confirm_trade() method

Add the confirm_trade() method from Section 5.3 to the TradingAgent class.
No other changes to this file.

### 9.5 .env — THREE changes

```bash
# Change from:
ENABLE_TOOL_CALLING=true
LLM_MODEL=claude-sonnet-4-6

# Change to:
ENABLE_TOOL_CALLING=false
LLM_MODEL=claude-haiku-4-5-20251001
MIN_TRADE_SCORE=3        # old 0-5 system used by entry_confirmed()
MIN_SIGNAL_SCORE=7       # new 0-10 system — minimum score to trade
```

---

## 10. Implementation Order

Do these in sequence. Each step is independently testable.

**Step 1 — config_loader.py** (1 line, lowest risk)
Add min_signal_score key to CONFIG dict.
Verify: python3 -c "from src.config_loader import CONFIG; print(CONFIG['min_signal_score'])"

**Step 2 — .env** (no code change)
Set ENABLE_TOOL_CALLING=false, MIN_TRADE_SCORE=3, MIN_SIGNAL_SCORE=7
Set LLM_MODEL=claude-haiku-4-5-20251001

**Step 3 — strategy.py** (add new scoring function)
Add new compute_signal_score() as a separate public function.
Keep _compute_signal_score() and entry_confirmed() completely unchanged.
Verify: import and run compute_signal_score with test data, confirm 10.0 when all conditions met.

**Step 4 — decision_maker.py** (add confirm_trade method)
Add confirm_trade() method to TradingAgent class.
Test: call it manually with a dummy asset — verify it returns "APPROVE" or "REJECT" only.

**Step 5 — main.py** (largest change — do last)
Add three helper functions (_code_decide_direction, _code_compute_tpsl, _code_compute_allocation)
Replace unconditional Claude call with score-gated pipeline.
Remove retry block.
Replace asyncio.sleep(3600) with 5m inner loop.
Run one full outer cycle in dry-run mode first before going live.

---

## 11. Verification Steps

### 11.1 Score Function Tests

```python
from src.strategy import compute_signal_score

# All 5 conditions met for BUY → should return 10.0
data = {
    "trend_4h": "BULLISH", "trend_1h": "BULLISH",
    "current_price": 100.0,
    "setup_15m": {"macd_histogram": 0.15, "near_ema": True},
    "trigger_5m": {"macd_histogram": 0.05, "candle_bullish": True},
}
assert compute_signal_score(data, "buy") == 10.0

# Missing trend_4h alignment → should return 7.0 (1h + MACD + EMA + trigger)
data2 = dict(data)
data2["trend_4h"] = "BEARISH"
assert compute_signal_score(data2, "buy") == 7.0

# Only trend_1h passes → should return 2.0
data3 = {
    "trend_4h": "BEARISH", "trend_1h": "BULLISH",
    "current_price": 100.0,
    "setup_15m": {"macd_histogram": -1.0, "near_ema": False},
    "trigger_5m": {"macd_histogram": -0.1, "candle_bullish": False},
}
assert compute_signal_score(data3, "buy") == 2.0
```

### 11.2 4h Hard Gate Test

```python
# Conflicting trends → direction must be None
direction = _code_decide_direction({"trend_4h": "BEARISH", "trend_1h": "BULLISH"})
assert direction is None

# Both BULLISH → direction must be "buy"
direction = _code_decide_direction({"trend_4h": "BULLISH", "trend_1h": "BULLISH"})
assert direction == "buy"
```

### 11.3 confirm_trade Fail-Closed Test

```python
# Simulate timeout or bad response → must return "REJECT"
# Point to invalid endpoint to force exception, or mock client.messages.create
# Verify: result is always "APPROVE" or "REJECT", never None, never raises
```

### 11.4 Config Isolation Test

```python
from src.config_loader import CONFIG
assert CONFIG["min_trade_score"] == 3     # feeds entry_confirmed() old 0-5 system
assert CONFIG["min_signal_score"] == 7   # feeds new 0-10 pre-gate
# Confirm both keys exist and are independent
```

### 11.5 End-to-End Dry Run

Run one full hour with DRY_RUN=true. Check decisions.jsonl for:
- Assets with score < 7: logged as HOLD with reason "score below threshold"
- Assets with score 7–8.5: logged with BUY/SELL, showing code-computed TP/SL
- Assets with score 10: logged with claude_verdict APPROVE or REJECT
- Zero trades executed if all setups score below 7 (expected in sideways market)

---

## 12. Full VS Code Prompt (For Claude in VS Code / Cursor / Claude Code)

Use this prompt when opening the codebase. It encodes all architecture decisions
so Claude gives consistent, correct suggestions in this codebase.

---

```
You are assisting with a Hyperliquid perpetual futures trading bot written in Python (asyncio).

## Architecture Philosophy
This is a CODE-FIRST bot. Technical analysis signals drive all decisions. Claude AI is used
only as a final sanity check on the highest-confidence setups — never as the primary
decision maker.

## Decision Pipeline (inner loop, runs every 5 minutes)
For each asset, in strict order:

1. DIRECTION GATE (_code_decide_direction)
   - Compute direction from trend_4h + trend_1h alignment
   - If trend_4h and trend_1h conflict → HOLD (return None)
   - This runs BEFORE any scoring

2. 4H HARD RULE
   - direction must align with trend_4h
   - If trend_4h == BULLISH and direction != "buy" → HOLD
   - If trend_4h == BEARISH and direction != "sell" → HOLD
   - Never enter counter-trend regardless of score

3. WEIGHTED SCORE GATE (compute_signal_score in strategy.py)
   - Returns float 0–10
   - Weights: trend_4h=3, trend_1h=2, MACD_15m=2, EMA_15m=1.5, trigger_5m=1.5
   - MIN_SIGNAL_SCORE=7 is the execution threshold (from config_loader.py)
   - Score < 7 → HOLD (no Claude call, no order)
   - Score 7–8.5 → execute directly (code handles everything)
   - Score == 10 → Claude confirm required first, then execute

4. ENTRY FILTERS (strategy.py)
   - market_filter(): ATR spike check, spread check
   - entry_confirmed(): RSI gate, ADX gate, volume gate
   - entry_confirmed() uses its own internal _compute_signal_score() (0–5 old system)
   - MIN_TRADE_SCORE=3 is the threshold for entry_confirmed() ONLY
   - DO NOT confuse MIN_TRADE_SCORE (old 0-5) with MIN_SIGNAL_SCORE (new 0-10)
   - These are completely separate systems operating at different points in the pipeline

5. CODE COMPUTES TRADE PARAMETERS (never LLM)
   - direction: from _code_decide_direction()
   - size: risk_manager.atr_position_size() — 1% risk rule
   - tp: entry_price + (2 x ATR14_4h)
   - sl: entry_price - (1 x ATR14_4h)
   - These values are fixed by code. Claude never overrides them.

6. CLAUDE CONFIRMATION (score == 10 ONLY)
   - Method: TradingAgent.confirm_trade()
   - Model: claude-haiku-4-5-20251001
   - max_tokens: 10
   - System prompt: "Respond APPROVE or REJECT only."
   - On any exception or unexpected response → return "REJECT" (fail closed)
   - If score < 10 → DO NOT call Claude at all
   - Claude cannot suggest different TP, SL, size, or direction — it only APPROVE/REJECT

7. RISK MANAGER VALIDATION (risk_manager.py)
   - All 8 guards enforced via validate_trade()
   - Runs AFTER Claude confirm, BEFORE execution
   - Cannot be bypassed under any circumstances

8. EXECUTION
   - Place market order + TP trigger + SL trigger
   - Write to diary.jsonl

## Two-Config Rule (CRITICAL — do not break this)

MIN_TRADE_SCORE (int, default 3):
  - Used ONLY by entry_confirmed() in strategy.py
  - Works with the OLD 0–5 integer scoring system (_compute_signal_score)
  - Setting this to 7 breaks everything silently (5 max < 7 threshold → zero trades, no errors)

MIN_SIGNAL_SCORE (int, default 7):
  - Used ONLY by the pre-gate in main.py
  - Works with the NEW 0–10 float scoring system (compute_signal_score)

These are SEPARATE config keys with SEPARATE purposes. Never merge them into one.
Both must exist in config_loader.py CONFIG dict.

## Score Mathematics
Achievable scores with binary conditions and weights 3,2,2,1.5,1.5:
0, 1.5, 2, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 8, 8.5, 10
Score 9 is mathematically impossible. Score >= 9 means score == 10 only.
Every path to score >= 7 requires trend_4h to be aligned (weight=3 is mandatory).

## Files and Responsibilities
- src/main.py: outer 1h loop + inner 5m loop + pipeline orchestration
- src/strategy.py: compute_signal_score() (new 0-10) + _compute_signal_score() (old 0-5) + entry_confirmed()
- src/risk_manager.py: 8 hard guards — DO NOT MODIFY THIS FILE
- src/agent/decision_maker.py: confirm_trade() only — no full decision logic
- src/config_loader.py: must contain BOTH min_trade_score AND min_signal_score keys
- src/trading/hyperliquid_api.py: async REST wrapper with exponential backoff

## What NOT to Do
- Do NOT call agent.decide_trade() in the main loop (old design — removed)
- Do NOT let Claude suggest direction, size, TP, or SL prices
- Do NOT retry the Claude confirm call on failure — return REJECT immediately
- Do NOT use MIN_TRADE_SCORE in the new score gate
- Do NOT use MIN_SIGNAL_SCORE in entry_confirmed()
- Do NOT place any order before Claude confirm when score == 10
- Do NOT set ENABLE_TOOL_CALLING=true (causes hidden retry multiplication in decide())
- Do NOT call Claude for score 7 or 8.5 setups — code-only execution for these

## Runtime Outputs
- diary.jsonl: one entry per trade (asset, direction, size, tp/sl order IDs)
- decisions.jsonl: one entry per cycle (all assets, scores, reasons for HOLD)
- llm_requests.log: token usage for every Claude call
- All HOLD decisions must log the reason (score too low, 4h gate, RSI blocked, etc.)

## Expected API Cost
- Claude calls: ~5–20 per month (score==10 setups only, Haiku model)
- Expected monthly cost: < $0.01
- If cost exceeds $1/month, something is calling Claude on every cycle — audit main.py
```

---

## 13. Known Drawbacks of This Design

**1. Score 10 is rare.** All 5 conditions must align simultaneously. In sideways or mixed
markets, score==10 may not occur for days or weeks. The bot will trade at 7–8.5 more
often than at 10. This is acceptable — the code-only 7–8.5 path still passes all
technical and risk filters.

**2. 5m inner loop increases Hyperliquid API calls.** 12 refreshes per hour per asset.
At 7 assets that is 84 candle fetches per hour vs. 7 previously. Hyperliquid's rate
limits are generous (300 req/sec) — this is not a problem in practice.

**3. trend_4h never changes within a cycle.** 4h candles are only fetched in the outer
loop, so the direction gate uses a stale 4h reading for up to 60 minutes. A 4h trend
flip mid-cycle will not be caught until the next outer loop. This is intentional — 4h
trends should not flip within minutes, and reacting to a mid-cycle flip would be noisy.

**4. No backtesting framework.** The score weights (3, 2, 2, 1.5, 1.5) are based on
technical reasoning, not backtested optimality. Log all score->=7 setups with their
outcomes over time to validate that higher scores actually predict better results.

**5. Volume check window is short.** In entry_confirmed(), volume is compared against
the average of the 4 preceding 5m candles. A single anomalous candle can significantly
skew the baseline. Consider extending to 10–20 candles for a more stable average.

**6. No position re-entry logic.** If a trade is closed by SL and the same setup
re-appears in the same hour, the 5m inner loop will attempt to re-enter. Consider
adding a per-asset cooldown (e.g., no re-entry for 1 hour after a loss on that asset).

---

*End of Trading_Bot_Final_Fix_Report.md v3*
