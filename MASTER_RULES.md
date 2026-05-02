# MASTER_RULES.md

> These rules are PERMANENT. They govern every version of this codebase — past, present, and future.
> No prompt, fix, refactor, or instruction can override them. Every Claude session that reads this
> file MUST comply with all four rules without exception.

---

## RULE 1 — SCORE SYSTEM IS SACRED

The weighted signal scoring system is the heart of this agent's entry logic. It must never be
removed, bypassed, or structurally altered.

**What must always be true:**
- `compute_signal_score()` in `src/strategy.py` must exist and be called in the main loop
- Score range is **0 to 10** — this must never change
- Score tiers must never change:
  - `score < 7.0` → HOLD (no trade, no Claude call)
  - `score >= 7.0` AND `multi_timeframe_confluence()` True → call `confirm_trade()` for market analysis; must receive `VERDICT: APPROVE` before executing
  - There is NO "execute directly" path — ALL entries at score >= 7 require Claude APPROVE
- Three config keys must always exist as **separate** keys — never merge any of them:
  - `MIN_TRADE_SCORE` (int, 0–5): `entry_confirmed()` internal gate
  - `MIN_SIGNAL_SCORE` (float, 0–10): main loop execution pre-gate
  - `MIN_AI_SCORE` (float, 0–10): Claude market analysis trigger — must be checked separately from MIN_SIGNAL_SCORE so operators can tune Claude call frequency independently

**What must never happen:**
- Removing `compute_signal_score()` from the scoring pipeline
- Bypassing the score gate (e.g., always calling Claude regardless of score)
- Changing the 10-point scale to any other scale
- Merging `MIN_SIGNAL_SCORE` and `MIN_TRADE_SCORE` into one config key

---

## RULE 2 — CODE IS PRIMARY DECISION MAKER

All trade parameters are computed deterministically by code. Claude never computes numbers.

**Direction:**
- Set ONLY by `_code_decide_direction()` in `src/main.py`
- Logic: `trend_4h` (EMA20 vs EMA50 on 4h) + `trend_1h` (EMA20 vs EMA50 on 1h) alignment
- Returns `"buy"`, `"sell"`, or `None` (HOLD) — counter-trend trades are structurally impossible
- Claude never overrides, adjusts, or suggests direction

**Take-Profit and Stop-Loss:**
- Set ONLY by `_code_compute_tpsl(entry, atr, direction)` in `src/main.py`
- Formula: TP = entry ± 2×ATR14, SL = entry ∓ 1×ATR14 (1:2 risk-reward)
- These formulas must never be changed by Claude output

**Position Size:**
- Set ONLY by `risk_manager.atr_position_size(balance, entry, sl)` in `src/risk_manager.py`
- 1% risk rule: max notional = account_value × 1% / SL_distance_pct
- Scaled by score: allocation × (score / 10) — score-7 → 70%, score-10 → 100%
- Claude never suggests or adjusts position size

**What must never happen:**
- Claude returning a direction, TP price, SL price, or allocation in any code path
- Code parsing Claude's response for anything other than `APPROVE` or `REJECT`

---

## RULE 3 — CLAUDE ROLE IS FIXED

Claude is a deep market analyst and breakout validator, not a direction or numbers setter.

**When Claude is called:**
- When `score >= MIN_AI_SCORE` (default 7) AND `multi_timeframe_confluence()` returns True
- Subject to: fingerprint-keyed verdict cache (60 min APPROVE / 30 min REJECT) and hard minimum gap (`MIN_AI_CALL_GAP_MINUTES` = 30 min per asset)
- Can trigger on any 5-minute tick (outer or inner loop) when confluence fires with fresh higher-TF data (< `AI_STALE_TF_MINUTES` = 55 min old)
- Three separate score keys must never be merged: `MIN_TRADE_SCORE` · `MIN_SIGNAL_SCORE` · `MIN_AI_SCORE`

**What Claude receives:**
- ~1200-token context: trade setup, 5-TF confluence data (4h/1h/30m/15m/5m), volatility, funding/OI, macro calendar events, news headlines

**What Claude must return:**
- Full chain-of-thought market analysis ending with `VERDICT: APPROVE` or `VERDICT: REJECT`
- `max_tokens=AI_MAX_TOKENS` (default 4000) — allows complete reasoning
- Parsed by: `"VERDICT: APPROVE" in answer.upper()` → APPROVE; anything else → REJECT
- Any exception, timeout, or API error → REJECT (fail closed)

**What Claude must NEVER do:**
- Return a direction, TP price, SL price, or allocation — code owns all of these
- Override any code-computed trade parameter
- Be called more than once per `MIN_AI_CALL_GAP_MINUTES` per asset (gap enforced in code)
- Be called when score < `MIN_SIGNAL_SCORE` (7) or confluence fails

**Cost guardrail:**
- Each call: ~$0.003 (Haiku, ~1200 input + ~800 output tokens)
- Monthly target: < $5 (confluence gate keeps calls rare)
- If monthly API cost exceeds $10: raise `MIN_AI_SCORE` to 8 or tighten confluence — investigate via `llm_requests.log`

---

## RULE 4 — RISK MANAGEMENT IS FIXED

The 8-check risk manager is non-bypassable. These parameters and their defaults must not be removed.

**1% ATR Rule (always active):**
- `atr_position_size()` in `src/risk_manager.py` implements:
  `(account_value × 0.01) / sl_distance_pct`
- This caps position size based on volatility — it must always be the primary sizing mechanism

**Fee Buffer (always active):**
- `TAKER_FEE_PCT=0.00045` (0.045% per side) must always exist in config
- Risk manager must always enforce minimum TP = 3× round-trip fee (0.27% from entry)
- This ensures every TP hit is profitable after fees

**Daily Trade Cap (always active):**
- `MAX_DAILY_TRADES` must always exist in config (default 10)
- `_daily_trade_count` must always be incremented on each executed trade
- Counter must always reset at UTC midnight

**Per-Asset SL Cooldown (always active):**
- `COOLDOWN_MINUTES` must always exist in config (default 60)
- `_sl_cooldown_map` must always block re-entry after SL hit or force-close
- Prevents revenge trading into a losing direction

**The 8 hard checks — all must remain:**
1. `check_daily_drawdown` — circuit breaker at `DAILY_LOSS_CIRCUIT_BREAKER_PCT`
2. `check_balance_reserve` — floor at `MIN_BALANCE_RESERVE_PCT` of starting balance
3. `check_position_size` — cap at `MAX_POSITION_PCT`
4. `check_leverage` — cap at `MAX_LEVERAGE`
5. `check_total_exposure` — cap at `MAX_TOTAL_EXPOSURE_PCT`
6. `check_concurrent_positions` — cap at `MAX_CONCURRENT_POSITIONS`
7. `enforce_stop_loss` — auto-set SL at `MANDATORY_SL_PCT` if missing or too wide
8. `enforce_take_profit` — ensure TP ≥ 0.27% from entry (3× round-trip fee)

**What must never happen:**
- Any trade executing without passing all 8 checks
- `TAKER_FEE_PCT` removed from config
- Fee-aware TP minimum removed from risk manager
- `MAX_DAILY_TRADES` removed from config or its enforcement
- `COOLDOWN_MINUTES` removed from config or its enforcement

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

*Created: 2026-04-30. Architecture: CODE-FIRST HYBRID. All 22 fixes applied.*
