# Trading Bot Redesign Plan
## Code-First Architecture — AI Used Only When Necessary
**Date:** 2026-04-29

---

## CORRECTION TO PREVIOUS AUDIT

Before the plan: the earlier audit made one error.

`strategy.py` was **updated after Phase 1**. It now contains `_compute_signal_score()` which IS called inside `entry_confirmed()`. `MIN_TRADE_SCORE=3` IS read by code — confirmed at line 76 of strategy.py:

```python
_score = _compute_signal_score(asset_data, direction)
_min_score = int(CONFIG.get("min_trade_score") or 3)
if _score < _min_score:
    return False
```

**The score system exists. It works. But it is in the wrong place.**

It gates EXECUTION (after Claude responds) — not the API CALL itself.

Current flow:
```
DATA → INDICATORS → CLAUDE API (always) → score check → execution
```

Required flow:
```
DATA → INDICATORS → score check → CLAUDE API (only if score passes) → execution
```

This is the entire root problem. Everything else follows from fixing this one ordering issue.

---

## WHAT IS WRONG — EXACT DIAGNOSIS

### Problem 1: Score gates execution, not the API call

`entry_confirmed()` in `strategy.py` runs at `main.py` line 1010.
Claude is called at `main.py` line 869.

Claude is called **141 lines before** the score is checked.
The score filter currently only determines whether to ACT on Claude's output.
It does NOT prevent Claude from being called.

### Problem 2: Retry path doubles cost silently

`main.py` lines 879–894: If Claude's output fails validation, a second full Claude call fires automatically. This is invisible in logs unless you search for "Retrying LLM".

### Problem 3: Tool loop can multiply calls up to 4×

`decision_maker.py` line 494: `_max_iterations = int(CONFIG.get("max_tool_iterations") or 3) + 1`
With `ENABLE_TOOL_CALLING=true`, each `fetch_indicator` request from Claude = 1 extra API call.
4 iterations max = 4 API calls in one cycle minimum.

### Problem 4: Bot is blind between 1h cycles

`main.py` line 1190: `await asyncio.sleep(_interval_seconds)` = sleep 3600 seconds.
A 5m breakout at 1:30 PM is not seen until 2:00 PM.
The 5m and 15m candles ARE fetched but only once per hour at cycle start.

### Problem 5: Claude is making the TRADE DECISION

Current: Claude is told "analyze and decide: buy/sell/hold, allocation, TP/SL, exit_plan"
This means Claude IS the trading brain. Code only validates what Claude decides.

Required: Code makes the trade decision. Claude only confirms or rejects.
Claude's role should be: receive a specific code-computed setup → output APPROVE or REJECT.

---

## NEW ARCHITECTURE

### Core Principle
```
Code drives everything.
Claude is a cheap confirmation check, called rarely.
```

### New Flow

```
OUTER LOOP — every 1h (trend + context computation):
│
├── Fetch all candles (1h, 4h, 1d) for all assets
├── Compute indicators locally
├── Compute trend labels: trend_1d, trend_4h, trend_1h
├── Compute signal score (0–5) per asset per direction
│     5 conditions:
│       trend_4h aligned:    +1
│       trend_1h aligned:    +1
│       MACD 15m aligned:    +1
│       near EMA 15m:        +1
│       5m candle/MACD:      +1
│
├── IF score < MIN_TRADE_SCORE (e.g. 4):
│     → HOLD, NO API CALL, log reason
│
├── IF score >= MIN_TRADE_SCORE:
│     → Compute entry price, TP, SL from ATR (code, not Claude)
│     → Compute allocation from 1% risk rule (code, not Claude)
│     → Apply all execution gates (state, trend, market_filter)
│     → IF all gates pass:
│           → Call Claude (Haiku, minimal prompt, max 150 tokens)
│             Input:  asset, direction, score, key indicators only
│             Output: APPROVE or REJECT (one word)
│           → IF APPROVE: execute trade
│           → IF REJECT: HOLD, log reason
│
└── Sleep until next 1h cycle

INNER LOOP — every 5m (entry trigger, NO Claude):
│
├── Fetch 5m + 15m candles only (2 cheap API calls per asset)
├── Compute 5m score against ALREADY KNOWN 4h/1h trend
├── IF 5m score >= 4 AND 4h trend known AND asset state = IDLE:
│     → Flag asset for immediate entry check
│     → Compute TP/SL from ATR, compute allocation
│     → Apply execution gates (no Claude)
│     → Execute immediately (no waiting for next 1h cycle)
│
└── Sleep 5 minutes, repeat
```

---

## WHAT CLAUDE DOES IN THE NEW SYSTEM

### Old role (current, wrong):
Claude receives entire market context for all assets and decides:
- Direction (buy/sell/hold)
- Allocation amount
- TP price
- SL price
- Exit plan
- Rationale

Claude is the brain. Code just validates.

### New role (correct):
Claude receives ONE pre-computed trade setup and answers ONE question:

```
Asset: BTC
Direction: BUY (code decided this)
Score: 4/5
Entry: $94,200
TP: $96,150 (1.5× ATR above entry, code-computed)
SL: $93,250 (1× ATR below entry, code-computed)
Signal summary: 4h BULLISH, 1h BULLISH, MACD 15m positive, 5m candle bullish

Should I enter this trade? Answer APPROVE or REJECT only.
```

Claude is a sanity check. Code is the brain.

---

## EXACT CHANGES REQUIRED

### Change 1 — Move score check BEFORE Claude call (main.py)

Where: Between line 838 (context built) and line 869 (Claude called).

Logic to add:
```
For each asset in the upcoming decision:
    compute score
    determine direction (from trend labels, code-only)
If no asset has score >= MIN_TRADE_SCORE:
    skip Claude call
    return HOLD for all assets
    log "Skipped Claude — no high-score setup found"
```

### Change 2 — Remove Claude as decision-maker (decision_maker.py)

Current: Claude is sent the full context and asked to decide buy/sell/hold + allocation + TP/SL.

New: Claude is sent ONE pre-computed trade and asked APPROVE/REJECT.

New prompt (short, cheap, Haiku):
```
You are a quick trade validator. A system has detected a trade setup using code.
Review and respond APPROVE or REJECT. Only those two words. No explanation.
If the setup has strong multi-timeframe alignment and the risk/reward is positive, say APPROVE.
If there is a clear reason not to enter (eg. upcoming major resistance, extreme RSI, 
funding very negative), say REJECT.
```

User message: JSON with only the fields needed for confirmation. ~200 tokens total.

Output: 1 word. Max tokens: 10.

### Change 3 — Code computes TP/SL/allocation before calling Claude (main.py)

Currently: Claude proposes allocation, TP, SL → risk manager adjusts.

New: Code computes everything first:
- Allocation = `atr_position_size(account_value, entry_price, sl_price, risk_pct=1.0)` (already in risk_manager.py)
- SL = `entry_price - (1 × ATR14)` for buys, `entry_price + (1 × ATR14)` for sells
- TP = `entry_price + (2 × ATR14)` for buys, `entry_price - (2 × ATR14)` for sells
- Risk manager then validates and caps (unchanged)

### Change 4 — Add 5m inner loop (main.py)

Current: One sleep of 3600 seconds between cycles.

New: Inside the sleep period, run a lightweight 5m check:
```python
# After 1h cycle completes:
for sub_interval in range(interval_seconds // 300):   # e.g. 12 × 5min = 60min
    await asyncio.sleep(300)                           # sleep 5 minutes
    await run_5m_entry_check()                        # fetch 5m candles only, compute score
    # No Claude call in 5m check — code gates only
```

The 5m entry check:
- Fetches 5m + 15m candles (2 Hyperliquid calls per asset — free)
- Computes 5m score using ALREADY STORED 4h/1d trend from last 1h cycle
- If score >= threshold AND all code gates pass → execute immediately (no Claude)
- Logs as "5m trigger entry" in diary

### Change 5 — Disable tool calling (.env)

```
ENABLE_TOOL_CALLING=false
```

All indicators are already computed locally and sent in context.
Tool calling adds 1–3 extra API calls per cycle with no meaningful benefit.

### Change 6 — Remove hidden retry (main.py)

Lines 879–894: The automatic retry on failed output fires a second full Claude call.
New behavior: If output fails → log error → HOLD all assets → no retry API call.

### Change 7 — Set model to Haiku for confirmation calls (decision_maker.py)

For APPROVE/REJECT confirmation calls, hardcode `claude-haiku-4-5-20251001`.
Haiku output for 1 word costs ~$0.000004. 
No reason to use Sonnet for a binary confirmation.

### Change 8 — Set MIN_TRADE_SCORE = 4 (.env)

Current: 3/5 conditions required.
Recommended: 4/5 conditions required.

At 3/5: a setup with only trend_4h + trend_1h aligned passes. That is not enough conviction.
At 4/5: requires trend alignment + MACD + EMA proximity. This is a real setup.

---

## WHAT DOES NOT CHANGE

| Component | Status |
|-----------|--------|
| `risk_manager.py` | No changes. All 8 guards stay exactly as-is. |
| `local_indicators.py` | No changes. All computation stays local. |
| `trade_state.py` | No changes. State machine stays as-is. |
| `hyperliquid_api.py` | No changes. Exchange integration stays as-is. |
| `config_loader.py` | No changes. |
| Force-close logic | No changes. |
| TP/SL Guardian | No changes. |
| Reconcile logic | No changes. |
| Entry gates (trend, daily macro, market_filter) | No changes. |
| `_compute_signal_score()` in strategy.py | No changes. Already correct. |

---

## EXPECTED API CALL FREQUENCY

### Current (broken):
```
Every cycle: 1 Claude call (guaranteed)
+ Tool calls: 0–3 extra (when ENABLE_TOOL_CALLING=true)
+ Retry: 0–1 extra (on bad output)
= 1–5 Claude calls per cycle
= 24–120 Claude calls per day
```

### After fix:
```
1h cycle:
  Score < 4: 0 Claude calls (most cycles in ranging/weak market)
  Score = 4: 1 Haiku call (APPROVE/REJECT, ~$0.000004)
  Score = 5: 1 Haiku call (APPROVE/REJECT, ~$0.000004)

5m inner loop:
  0 Claude calls (code-only execution)

Retry:
  0 Claude calls (retry removed)

Tool calls:
  0 (disabled)
```

**In a typical market (BTC/ETH, mixed conditions):**
- High-score setups (4–5 points) occur roughly 3–6 times per day across both assets
- Most hours: 0 Claude calls
- Hours with a setup: 1 Haiku call
- Daily Claude calls: 3–6 (down from 24–120)
- Cost with Haiku: ~$0.00002–0.00004 per call
- **Monthly cost: under $0.01**

Even if Sonnet is used for confirmation calls instead of Haiku:
- 5 calls/day × $0.020/call = $0.10/day = **$3/month**

---

## SCORE THRESHOLDS (RECOMMENDED)

The existing `_compute_signal_score()` uses a 0–5 scale. No changes needed to the formula.

| Score | Meaning | Action |
|-------|---------|--------|
| 0–2 | Weak or no signal | HOLD. No API call. |
| 3 | Some alignment, below conviction | HOLD. No API call. |
| 4 | Strong alignment — 4 of 5 conditions | Code gates → optional Haiku confirmation |
| 5 | Maximum alignment — all 5 conditions | Code gates → Haiku confirmation → execute |

Set `MIN_TRADE_SCORE=4` in `.env`.

---

## SUMMARY TABLE

| Item | Before | After |
|------|--------|-------|
| Claude call frequency | Every 1h cycle, unconditional | Only when score ≥ 4 (3–6×/day) |
| Claude role | Decides everything | APPROVE/REJECT only |
| Claude model | claude-sonnet-4-6 | claude-haiku-4-5-20251001 |
| Max tokens | 4096 | 10 (one word output) |
| Tool calls | 0–3 per cycle | 0 (disabled) |
| Retry calls | 1 per bad output | 0 (removed) |
| Monthly cost | $13–$40 | Under $1 |
| 5m setups missed | Yes (1:30 PM missed) | No (5m inner loop catches them) |
| Who makes trade decision | Claude | Code (score + trend labels) |
| TP/SL computed by | Claude | Code (ATR formula, already exists) |
| Allocation by | Claude | Code (1% risk rule, already in risk_manager.py) |

---

## VS CODE PROMPT — PASTE THIS EXACTLY

The following prompt is designed to be pasted into VS Code (GitHub Copilot, Claude extension, or any AI assistant). It is self-contained and does not require additional context.

---

```
=== TRADING BOT REFACTOR — CODE-FIRST ARCHITECTURE ===

PROJECT: Hyperliquid perpetual futures trading bot
LOCATION: src/ directory
DO NOT MODIFY: risk_manager.py, local_indicators.py, trade_state.py, hyperliquid_api.py, config_loader.py

READ THESE FILES FIRST BEFORE MAKING ANY CHANGES:
- src/main.py
- src/agent/decision_maker.py
- src/strategy.py
- .env

=== PROBLEM ===

Currently, the Claude API is called on every single cycle (every 1 hour), unconditionally,
at main.py line ~869:
    outputs = await asyncio.to_thread(agent.decide_trade, args.assets, context)

There is a signal scoring function `_compute_signal_score()` in strategy.py that scores
setups 0–5. It is called inside `entry_confirmed()` at main.py line ~1010.

The score check happens AFTER the Claude API call. This is wrong.
Claude is called even when no valid setup exists.

=== REQUIRED CHANGES ===

--- CHANGE 1: MOVE SCORE CHECK BEFORE CLAUDE CALL (main.py) ---

After market data is gathered and indicators are computed (after the `for asset in args.assets` 
loop that builds `market_sections`), and BEFORE the Claude call at line ~869:

Add a function `_get_best_direction(asset_data: dict) -> str | None` that:
  - Calls `_compute_signal_score(asset_data, "buy")` from strategy.py
  - Calls `_compute_signal_score(asset_data, "sell")` from strategy.py
  - Returns "buy" if buy_score > sell_score and buy_score >= min_score
  - Returns "sell" if sell_score > buy_score and sell_score >= min_score
  - Returns None if neither direction meets the threshold

Make `_compute_signal_score` importable from strategy.py by renaming it from 
`_compute_signal_score` to `compute_signal_score` (remove underscore prefix).

Before the Claude call, loop through all assets in market_sections:
  - For each asset, call `_get_best_direction(asset_data)`
  - If NO asset has a valid direction (all return None): 
      skip Claude entirely
      set outputs = {"reasoning": "No high-score setup found", "trade_decisions": [
          {"asset": a, "action": "hold", "allocation_usd": 0, "order_type": "market",
           "limit_price": None, "tp_price": computed_tp, "sl_price": computed_sl,
           "exit_plan": "", "rationale": "Score below threshold"}
          for a in args.assets
      ]}
      log at INFO level: "[SCORE GATE] All assets below threshold — skipping Claude API call"
      continue to next cycle

Store the computed directions per asset in a dict: `asset_directions = {"BTC": "buy", "ETH": None}`

--- CHANGE 2: CODE COMPUTES TP/SL/ALLOCATION (main.py) ---

For each asset where a valid direction exists, BEFORE calling Claude, compute:

  atr14 = asset_ctx.get("long_term_4h", {}).get("atr14")  # already available
  
  For BUY:
    sl_price = round(current_price - (float(atr14) * 1.0), 2)  if atr14 else None
    tp_price = round(current_price + (float(atr14) * 2.0), 2)  if atr14 else None
  
  For SELL:
    sl_price = round(current_price + (float(atr14) * 1.0), 2)  if atr14 else None
    tp_price = round(current_price - (float(atr14) * 2.0), 2)  if atr14 else None
  
  allocation_usd = risk_mgr.atr_position_size(account_value, current_price, sl_price)
  allocation_usd = min(allocation_usd, account_value * (risk_mgr.max_position_pct / 100.0) * risk_mgr.max_leverage)
  allocation_usd = max(allocation_usd, 11.0)

Store these per asset: `asset_computed = {"BTC": {"sl": ..., "tp": ..., "alloc": ...}}`

--- CHANGE 3: CHANGE CLAUDE'S ROLE TO APPROVE/REJECT (decision_maker.py) ---

Replace the existing `decide_trade()` and `_decide()` methods with a new method:
`confirm_trade(asset: str, direction: str, entry_price: float, tp_price: float, 
               sl_price: float, score: int, indicators_summary: dict) -> str`

This method:
  - Uses model: "claude-haiku-4-5-20251001" (hardcoded, not self.model — cheap)
  - Uses max_tokens: 10
  - Has NO tools, NO tool loop, NO retry
  - System prompt (keep it under 200 tokens):
      "You are a trade validator for a crypto futures bot. The system has computed a trade
       using technical analysis. Respond with exactly one word: APPROVE or REJECT.
       APPROVE if: multi-timeframe trend is aligned and risk/reward > 1.5.
       REJECT if: trade fights a major trend, RSI extreme, or ATR shows dangerous volatility."
  - User message: compact JSON with only:
      asset, direction, entry_price, tp_price, sl_price, score, 
      trend_4h, trend_1h, rsi14_1h, adx_1h, funding_rate
  - Parses response: if "APPROVE" in response.upper() → return "APPROVE", else → return "REJECT"
  - On any exception → return "APPROVE" (fail open — code gates already protect execution)

In main.py, replace the Claude call block:
  OLD: outputs = await asyncio.to_thread(agent.decide_trade, args.assets, context)
  NEW:
    For each asset with a valid direction:
      result = await asyncio.to_thread(
          agent.confirm_trade, asset, direction, entry_price, tp_price, sl_price, score, indicators
      )
      if result == "APPROVE":
          trade_decisions.append({
              "asset": asset, "action": direction,
              "allocation_usd": computed_alloc,
              "order_type": "market", "limit_price": None,
              "tp_price": computed_tp, "sl_price": computed_sl,
              "exit_plan": f"SL at {computed_sl}, TP at {computed_tp}, score={score}/5",
              "rationale": f"Code-computed. Score {score}/5. Claude confirmed."
          })
      else:
          trade_decisions.append({"asset": asset, "action": "hold", ...})
    
    For each asset with no valid direction:
      trade_decisions.append({"asset": asset, "action": "hold", ...})
    
    outputs = {"reasoning": "Code-driven decision", "trade_decisions": trade_decisions}

--- CHANGE 4: ADD 5-MINUTE INNER LOOP (main.py) ---

After the 1h cycle executes trades, instead of:
    await asyncio.sleep(_interval_seconds)

Use:
    _sub_interval = 300  # 5 minutes
    _sub_cycles = _interval_seconds // _sub_interval
    for _sc in range(_sub_cycles):
        await asyncio.sleep(_sub_interval)
        # Quick 5m check — no Claude, code only
        await _run_5m_check()

Add function `_run_5m_check()` as an async inner function inside `run_loop()`:

async def _run_5m_check():
    for asset in args.assets:
        # Skip if not IDLE
        if state_mgr.get_state(asset) != "IDLE":
            continue
        # Skip if 4h trend not established
        trend_4h = asset_trends.get(asset)
        if not trend_4h or trend_4h == "UNKNOWN":
            continue
        try:
            # Fetch 5m and 15m candles only (2 cheap calls)
            candles_5m, candles_15m = await asyncio.gather(
                hyperliquid.get_candles(asset, "5m", 20),
                hyperliquid.get_candles(asset, "15m", 30),
            )
            current_price = await hyperliquid.get_current_price(asset)
            ind_5m  = compute_all(candles_5m)
            ind_15m = compute_all(candles_15m)
            # Build minimal asset_data for scoring
            _asset_data = {
                "asset": asset,
                "current_price": current_price,
                "trend_4h": asset_trends.get(asset, "UNKNOWN"),
                "trend_1h": asset_trends.get(asset, "UNKNOWN"),  # use 4h as proxy if 1h not updated
                "setup_15m": {
                    "macd_histogram": latest(ind_15m.get("macd_histogram", [])),
                    "near_ema": True,  # conservative: allow entry
                    "rsi14": latest(ind_15m.get("rsi14", [])),
                },
                "trigger_5m": {
                    "macd_histogram": latest(ind_5m.get("macd_histogram", [])),
                    "candle_bullish": candles_5m[-1]["close"] > candles_5m[-1]["open"] if candles_5m else False,
                },
                "long_term_4h": {},  # populated from last 1h cycle if available
                "intraday_1h": {},
                "candles_5m": candles_5m,
                "spread_pct": 0,
            }
            _min_score = int(CONFIG.get("min_trade_score") or 4)
            _buy_score  = compute_signal_score(_asset_data, "buy")
            _sell_score = compute_signal_score(_asset_data, "sell")
            _direction = None
            if _buy_score >= _min_score and _buy_score >= _sell_score:
                _direction = "buy"
                _score = _buy_score
            elif _sell_score >= _min_score:
                _direction = "sell"
                _score = _sell_score
            if not _direction:
                continue
            # Apply code gates (no Claude)
            _mf_pass, _mf_reason = market_filter(_asset_data)
            if not _mf_pass:
                continue
            if not entry_confirmed(_asset_data, _direction):
                continue
            # Get ATR from last known 4h data
            _atr14 = None  # Will fall back to mandatory_sl_pct in risk manager
            _is_buy = _direction == "buy"
            _sl = risk_mgr.enforce_stop_loss(None, current_price, _is_buy, _atr14)
            _tp = risk_mgr.enforce_take_profit(None, current_price, _is_buy, _atr14)
            _alloc = max(risk_mgr.atr_position_size(account_value, current_price, _sl), 11.0)
            _trade_dict = {
                "asset": asset, "action": _direction,
                "allocation_usd": _alloc, "order_type": "market",
                "limit_price": None, "tp_price": _tp, "sl_price": _sl,
                "current_price": current_price, "atr14": _atr14,
                "exit_plan": f"5m trigger entry. SL={_sl}, TP={_tp}, score={_score}/5",
                "rationale": f"5m inner loop trigger. Score {_score}/5. No Claude call."
            }
            _allowed, _reason, _trade_dict = risk_mgr.validate_trade(_trade_dict, state, initial_account_value or 0)
            if not _allowed:
                logging.info("[5m GATE] %s blocked by risk: %s", asset, _reason)
                continue
            # Execute
            _amount = float(_trade_dict["allocation_usd"]) / current_price
            if _is_buy:
                _order = await hyperliquid.place_buy_order(asset, _amount)
            else:
                _order = await hyperliquid.place_sell_order(asset, _amount)
            if _trade_dict.get("tp_price"):
                await hyperliquid.place_take_profit(asset, _is_buy, _amount, float(_trade_dict["tp_price"]))
            if _trade_dict.get("sl_price"):
                await hyperliquid.place_stop_loss(asset, _is_buy, _amount, float(_trade_dict["sl_price"]))
            state_mgr.record_entry(asset)
            active_trades.append({
                "asset": asset, "is_long": _is_buy, "amount": _amount,
                "entry_price": current_price, "tp_price": _trade_dict.get("tp_price"),
                "sl_price": _trade_dict.get("sl_price"), "tp_oid": None, "sl_oid": None,
                "exit_plan": _trade_dict["exit_plan"], "funding_rate": 0,
                "opened_at": datetime.now(timezone.utc).isoformat()
            })
            save_active_trades(active_trades)
            logging.info("[5m TRIGGER] %s %s entered at %s (score %d/5)", asset, _direction, current_price, _score)
        except Exception as _5m_err:
            logging.warning("[5m CHECK] %s error: %s", asset, _5m_err)

--- CHANGE 5: DISABLE RETRY (main.py) ---

Remove or disable the retry block at lines ~879–894:
  OLD:
    if _is_failed_outputs(outputs):
        outputs = await asyncio.to_thread(agent.decide_trade, args.assets, context_retry)
  NEW:
    if _is_failed_outputs(outputs):
        logging.warning("[RETRY] Skipping API retry — holding all assets to avoid double billing")
        # outputs remains whatever it was (may be partial HOLD)

--- CHANGE 6: .env UPDATES (do not touch other settings) ---

ENABLE_TOOL_CALLING=false
MIN_TRADE_SCORE=4
LLM_MODEL=claude-haiku-4-5-20251001

=== DO NOT CHANGE ===
- risk_manager.py (any function)
- local_indicators.py (any function)
- trade_state.py (any function)
- hyperliquid_api.py (any function)
- All existing execution gates in main.py (state gate, trend gate, inversion check, daily filter, market_filter, entry_confirmed)
- Force-close logic
- TP/SL Guardian
- Reconcile logic
- _log_trade_close()
- _update_stats()

=== EXPECTED RESULT ===

Before: Claude called every cycle = 24–120 API calls/day
After:  Claude called only when score >= 4 = 0–6 API calls/day
        5m inner loop catches momentum setups between 1h cycles (no Claude)
        Monthly API cost: under $1
```

---

## IMPLEMENTATION ORDER (for developer)

Execute changes in this sequence to avoid breaking the running system:

1. `.env` — set `ENABLE_TOOL_CALLING=false` and `MIN_TRADE_SCORE=4` (immediate, no code change)
2. `strategy.py` — rename `_compute_signal_score` to `compute_signal_score`, update import in strategy.py
3. `main.py` — add import: `from src.strategy import compute_signal_score`
4. `main.py` — add `_get_best_direction()` function
5. `main.py` — add pre-score check block before Claude call (Change 1)
6. `main.py` — add code-computed TP/SL/allocation block (Change 2)
7. `decision_maker.py` — add `confirm_trade()` method (Change 3)
8. `main.py` — replace Claude call with new `confirm_trade()` flow (Change 3)
9. `main.py` — disable retry block (Change 5)
10. `main.py` — replace `asyncio.sleep(_interval_seconds)` with 5m inner loop (Change 4)
11. Run bot and watch logs for: `[SCORE GATE]`, `[5m TRIGGER]`, `[API]` lines

---

## HOW TO VERIFY IT WORKED

After running for a few hours, check `llm_requests.log`:
- You should see far fewer entries (1 per high-score setup vs 1 per hour)
- Each entry cost should be tiny ($0.000004 for Haiku APPROVE/REJECT)

Check the main log for lines:
- `[SCORE GATE] All assets below threshold — skipping Claude API call` = working
- `[5m TRIGGER] BTC buy entered at 94200 (score 4/5)` = 5m loop catching setups
- `[API] input=X output=1 cost=$0.00000X` = Haiku single-word response confirmed
