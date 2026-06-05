# 🏛️ TRADING BOT BUG COUNCIL — V3 FINAL REPORT

**Scan date:** 2026-05-10
**Scope:** Full line-by-line re-scan after VS Code applied all 7 v2 bug fixes
**Bot:** Hyperliquid perpetual futures · Code-first hybrid · Score-gated · Claude as APPROVE/REJECT gate

---

## ✅ V2 FIX VERIFICATION — ALL 7 CONFIRMED

Every fix applied by VS Code was verified against live source code.

| Bug | Description | Status | Evidence |
|-----|-------------|--------|----------|
| BUG-1 | SIGTERM/SIGINT handlers never wired | ✅ FIXED | `signal.signal()` at lines 2156-2157 |
| BUG-2 | Inner loop TP/SL from stale outer price | ✅ FIXED | Fresh `asset_prices.get()` used at line 1731 |
| BUG-3 | Timeout close missing `cancel_all_orders` | ✅ FIXED | Line 802 confirmed present |
| BUG-4 | `near_ema_15m = True` when EMA unavailable | ✅ FIXED | `else False` default at lines 1104-1108 |
| BUG-5 | KILLSWITCH not checked inside inner loop | ✅ FIXED | Check at lines 1678-1682 |
| BUG-6 | H-3 validation checked `CONFIG.get()` (never None) | ✅ FIXED | `os.getenv()` at line 339 |
| BUG-7 | `trade_log` list unbounded | ✅ FIXED | `deque(maxlen=200)` at line 358 |

---

## 💀 NEW CRITICAL BUG — BOT WILL CRASH ON RESTART

### V3-CRITICAL-1 · `UnboundLocalError` in `run_loop()` — Bot Won't Restart
**File:** `src/main.py` · Line 1681 (BUG-5 fix introduces a Python scoping error)

**What happened:** The BUG-5 fix added `_shutdown = True` inside the inner `for _tick` loop at line 1681. However, `_shutdown` is a module-level global. Python's scoping rule is: if a name is **assigned anywhere** in a function without `global` or `nonlocal`, ALL references to it in that function use the local scope.

Because `_shutdown = True` appears inside `run_loop()` with no `global _shutdown` declaration in that function, Python's bytecode compiler treats every reference to `_shutdown` in `run_loop()` as local. This means line 610:

```python
while True:
    if _shutdown:    # ← UnboundLocalError: 'referenced before assignment'
```

will raise `UnboundLocalError` on the very first loop iteration, before any inner tick is ever reached.

**Financial consequence:** The bot cannot be restarted after a process restart, OOM kill, Docker container restart, or server reboot. Open positions from the last session have no protection.

**Fix:** Add one line at the top of `run_loop()`:
```python
async def run_loop():
    global _shutdown   # ← ADD THIS LINE
    ...
```

Note: `_handle_signal()` at line 2150 already correctly declares `global _shutdown` — the same declaration is simply missing from `run_loop()`.

---

## 🔴 NEW HIGH BUGS

### V3-HIGH-1 · `llm_requests.log` grows unbounded — disk-fill risk
**File:** `src/agent/decision_maker.py` · Lines 246-258

The E-4 fix (log rotation) covers `diary.jsonl` and `decisions.jsonl` only. `llm_requests.log` is written by `confirm_trade()` on every Claude call: the full market analysis text (hundreds of tokens), model, cost, timestamps. No size check or rotation exists for this file.

Over weeks of live trading with active score-10 setups, this file grows without limit. A full disk halts all file writes including diary and state — open positions lose P&L tracking and the bot enters an unmonitored state.

**Fix:** Add rotation for `llm_requests.log` in the outer cycle alongside the existing `_rotate_if_needed()` calls:
```python
_rotate_if_needed(diary_path)
_rotate_if_needed(decisions_path)
_rotate_if_needed(os.path.join(_log_base, "llm_requests.log"))  # ADD
```

### V3-HIGH-2 · `context_payload` built and serialized every cycle but never used
**File:** `src/main.py` · Lines 1190-1215

These ~25 lines build a full `context_payload` JSON object (all assets, full indicators, account state, instructions) and serialize it to a string called `context`. This was the payload passed to the old `agent.decide_trade()` call. That call was removed when the score-gated pipeline was introduced. The variable `context` is now used only for a misleading `len()` log message and a `prompts.log` write.

Consequences:
- Wasted CPU/memory every outer cycle (large JSON serialization for nothing)
- `prompts.log` fills with irrelevant data that looks like live Claude inputs — misleads operators trying to debug AI decisions
- Line 1206 logs `"Combined prompt length: X chars"` suggesting Claude receives this — it does not

**Fix:** Delete lines 1190-1215 entirely. The `_ac` dict passed to `confirm_trade()` is the actual context Claude receives, already logged in `prompts.log` by `confirm_trade()` itself.

---

## 🟡 MEDIUM ISSUES

### V3-MEDIUM-1 · `candle_bullish` defaults to `True` in `entry_confirmed()`
**File:** `src/strategy.py` · Line 140

```python
bull_5m = t5.get("candle_bullish", True)   # ← should be False
```

All other missing-indicator defaults in the codebase use `False` to fail closed (E-1, BUG-4). This `True` default means if trigger_5m data is missing during a buy evaluation, the candle direction check passes automatically. In practice `candle_bullish` is always explicitly set in `market_sections` assembly (line 1182), so this never triggers in normal operation — but it's a latent inconsistency that will silently allow entry if the assembly ever changes.

**Fix:** Change `True` to `False` on line 140.

### V3-MEDIUM-2 · Misleading "Combined prompt length" log message every cycle
**File:** `src/main.py` · Line 1206

```python
add_event(f"Combined prompt length: {len(context)} chars for {len(args.assets)} assets")
```

`context` is never sent to Claude. The actual Claude call sends individual `_ac` market section dicts (~2–5 KB). This message implies a multi-asset context is being transmitted, which could cause operators to wrong diagnose "prompt too large" issues or miss the actual prompt size problem.

**This disappears automatically when V3-HIGH-2 is removed.**

---

## 🗑️ DEAD CODE INVENTORY — SAFE TO DELETE

### DEAD-1 · `decide_trade()` + `_decide()` — 525 lines of dead code
**File:** `src/agent/decision_maker.py` · Lines 265-789

These two methods are **never called from the main loop or anywhere in the live code path**. `decide_trade()` (line 265) is a 3-line wrapper that calls `_decide()`. `_decide()` (lines 269-789) is the original full-context Claude decision engine from before the code-first redesign.

Everything inside `_decide()` is dead:
- `system_prompt` (77 lines) — full quantitative trader prompt
- `tools` list (30 lines) — `fetch_indicator` tool definition
- `_call_claude()` inner function — Anthropic API call with caching
- `_handle_tool_call()` inner function — fetch_indicator execution
- `_extract_json_brute_force()` inner function — JSON brace-counting
- `_sanitize_output()` inner function — Claude JSON repair
- `_hold_all()` inner function — fallback hold generator
- Main iteration loop (lines 661-789)

**Risk of keeping this code:** A future developer reading the class docstring (which describes the old Claude-centric architecture — see DEAD-5) might "re-enable" `decide_trade()`, which would violate MASTER RULE 2 (direction must be set by code only). The dead code is an active confusion hazard.

**Safe to delete:** Lines 265-789 in `decision_maker.py`. `confirm_trade()` at line 99 is the **live** method — keep it.

### DEAD-2 · Three unused `__init__` properties in `TradingAgent`
**File:** `src/agent/decision_maker.py` · Lines 89-97

```python
self.model = CONFIG.get("llm_model") or "claude-haiku-4-5-20251001"        # ← dead
self.sanitize_model = CONFIG.get("sanitize_model") or "claude-haiku-..."   # ← dead
self.max_tokens = int(CONFIG.get("max_tokens") or 2500)                    # ← dead
```

These three properties are only referenced inside `_decide()` (dead). `confirm_trade()` hardcodes `_haiku` (intentional per MASTER RULES cost constraint) and reads `ai_max_tokens` config directly. Remove these three assignments from `__init__`. The `logging.info` on lines 96-97 also references dead `self.sanitize_model`.

**Safe to delete:** Lines 89-97 (replace with just `self.hyperliquid = hyperliquid`).

### DEAD-3 · `src/utils/formatting.py` — entire file unused
**File:** `src/utils/formatting.py`

`format_number()` and `format_size()` are never imported in any file in the project. All numeric formatting uses `round_or_none()` from `prompt_utils.py` directly. This file has been superseded and is unreachable.

**Verification:** `grep -r "format_number\|format_size\|from src.utils.formatting"` returns no results outside the file itself.

**Safe to delete:** The entire `src/utils/formatting.py` file.

### DEAD-4 · `context_payload` build block — dead computation
**File:** `src/main.py` · Lines 1190-1215

Covered under V3-HIGH-2. The code block (25 lines) computes a large dict and JSON string that is never sent anywhere. Once V3-HIGH-2 is fixed, this block should be removed entirely.

### DEAD-5 · `TradingAgent` class docstring describes old architecture
**File:** `src/agent/decision_maker.py` · Lines 18-85

The 68-line docstring describes a "Claude-driven" architecture where "Claude analyzes multi-dimensional context" and "chooses ONE action per asset: BUY | SELL | HOLD". This is the **opposite** of the current design and directly contradicts MASTER RULE 2.

**Risk:** Any developer (or Claude in a future session) reading this docstring will try to restore the Claude-driven approach, violating the MASTER RULES.

**Fix:** Replace the docstring with a short accurate description of the current role:
```python
"""Trading agent wrapper. confirm_trade() is the only live method.
Code decides direction/TP/SL/size. Claude returns APPROVE or REJECT only.
decide_trade() and _decide() are removed dead code from the pre-2026-04-30 design.
"""
```

---

## 📊 DEAD CODE REMOVAL TABLE

| Item | File | Lines | LOC to Remove | Safe? |
|------|------|-------|---------------|-------|
| `decide_trade()` + `_decide()` | `decision_maker.py` | 265-789 | 525 | ✅ Yes |
| Dead `__init__` properties | `decision_maker.py` | 89-97 | 9 | ✅ Yes |
| Dead class docstring | `decision_maker.py` | 18-85 | 68 | ✅ Yes |
| `context_payload` build | `main.py` | 1190-1215 | 25 | ✅ Yes |
| `formatting.py` entire file | `utils/formatting.py` | 1-16 | 16 | ✅ Yes |
| **Total removable** | | | **~643 lines** | |

---

## ✅ CONFIRMED CLEAN — NO ISSUES FOUND

These components were re-read and are clean:

- `src/strategy.py` — `compute_signal_score()`, `multi_timeframe_confluence()`, `entry_confirmed()`, `market_filter()` all correct (except MEDIUM-1 above)
- `src/utils/prompt_utils.py` — `json_default`, `safe_float`, `round_or_none`, `round_series` all correct
- `src/main.py` signal pipeline — `_code_decide_direction()`, `_code_compute_tpsl()`, `_build_confluence_fingerprint()` all correct
- `src/main.py` guardian — TP/SL re-placement logic and fallback SL are correct
- `src/main.py` reconciler — stale trade cleanup with traceback logging is correct
- `src/main.py` KILLSWITCH outer loop — file check at line 615 is correct
- `src/main.py` instance lock — TCP socket mutex survives signal and finally block
- `src/main.py` state refresh — C-3/C-4 refresh before inner-loop execution is correct
- `src/main.py` `_log_trade_close()` — PnL calculation, funding cost, exit_type classification all correct
- `src/agent/decision_maker.py` `confirm_trade()` — model hardcoding to Haiku, timeout=30s, fail-closed, verdict cache writes all correct
- `src/indicators/local_indicators.py` — EMA, SMA, RSI (Wilder's smoothing), MACD all standard implementations
- `_fetch_macro_context()` — RSS feeds with 3s timeout, non-blocking, correct

---

## 🏁 PRIORITY ORDER — WHAT TO DO FIRST

### Must fix before next restart:
1. **V3-CRITICAL-1** — Add `global _shutdown` in `run_loop()` or bot crashes on startup

### Fix in this session:
2. **V3-HIGH-1** — Add `llm_requests.log` rotation to `_rotate_if_needed()` calls
3. **V3-HIGH-2** — Remove dead `context_payload` block (lines 1190-1215)
4. **V3-MEDIUM-1** — Change `candle_bullish` default from `True` to `False` in `entry_confirmed()`

### Clean up (can be done incrementally):
5. **DEAD-1** — Remove `decide_trade()` + `_decide()` (525 lines)
6. **DEAD-2** — Remove three dead `__init__` properties
7. **DEAD-3** — Delete `src/utils/formatting.py`
8. **DEAD-5** — Replace `TradingAgent` class docstring

---

## 📋 QUICK REFERENCE — V3 FINDINGS TABLE

| ID | Severity | File | Lines | Issue | Financial Impact |
|----|----------|------|-------|-------|-----------------|
| V3-CRITICAL-1 | 💀 CRITICAL | main.py | 610, 1681 | `_shutdown` scoping → `UnboundLocalError` on restart | Bot can't restart; open positions unmonitored |
| V3-HIGH-1 | 🔴 HIGH | decision_maker.py | 246-258 | `llm_requests.log` unbounded | Disk full → diary/state writes fail |
| V3-HIGH-2 | 🔴 HIGH | main.py | 1190-1215 | Dead `context_payload` build + misleading logs | Wasted CPU; confuses operator debugging |
| V3-MEDIUM-1 | 🟡 MEDIUM | strategy.py | 140 | `candle_bullish` defaults True | Latent: allows entry without candle confirm |
| V3-MEDIUM-2 | 🟡 MEDIUM | main.py | 1206 | "Combined prompt length" log is misleading | Operator confusion only (disappears with HIGH-2) |
| DEAD-1 | ⬛ DEAD | decision_maker.py | 265-789 | `decide_trade()` + `_decide()` never called | Risk: future accidental re-enable violates RULE 2 |
| DEAD-2 | ⬛ DEAD | decision_maker.py | 89-97 | Unused `self.model/sanitize_model/max_tokens` | Confusion only |
| DEAD-3 | ⬛ DEAD | utils/formatting.py | 1-16 | Entire file never imported | None |
| DEAD-4 | ⬛ DEAD | main.py | 1190-1215 | Dead `context_payload` computation | Same as HIGH-2 |
| DEAD-5 | ⬛ DEAD | decision_maker.py | 18-85 | Docstring describes old Claude-driven architecture | Risk: future code changes violate MASTER RULES |

---

## 🏛️ COUNCIL VERDICT

```
COUNCIL VERDICT: CONDITIONALLY READY — one blocking crash bug before restart

TOP 3 MUST-FIX:
  1. V3-CRITICAL-1 — Add `global _shutdown` in run_loop() before any restart
  2. V3-HIGH-1     — Rotate llm_requests.log or disk fills in days of active trading
  3. DEAD-1        — Remove decide_trade()/_decide() to prevent accidental MASTER RULE violation

CAPITAL RECOMMENDATION: Hold at current allocation until V3-CRITICAL-1 is fixed.
  Do NOT restart the process (OOM, deploy, server reboot) without the fix applied.
  Currently running instance is safe; crash only occurs on restart.

GO LIVE STATUS: DO NOT RESTART — apply V3-CRITICAL-1 fix first.
  After fix: CONDITIONALLY READY (apply HIGH and DEAD removals at next maintenance window)
```
