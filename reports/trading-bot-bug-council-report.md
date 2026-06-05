# 🏛️ TRADING BOT BUG COUNCIL — FINAL REPORT

**Generated:** 2026-05-10  
**Codebase:** `hyperliquid-trading-agent-master/`  
**Council version:** 5-agent (no DeFi — pure CEX/perp)

---

## 📋 BOT SUMMARY

| Field | Value |
|---|---|
| Strategy type | CODE-FIRST hybrid perpetual futures — code decides direction/TP/SL/size; Claude is APPROVE/REJECT only |
| Asset class | Crypto perpetual futures (BTC, ETH) on Hyperliquid |
| Exchange | Hyperliquid (mainnet) |
| Leverage | **Configured 5×** — **ACTUAL UNKNOWN** (exchange-side leverage never set by bot) |
| Order types | Market orders (entry + TP/SL trigger) — limit orders for some entries |
| Position model | Long or short, max 2 concurrent, 12-hour max hold |
| Risk controls FOUND | ATR position sizing, daily drawdown circuit breaker, max position pct, concurrent position limit, balance reserve, mandatory SL, TP fee floor, force-close at max loss, SL cooldown, daily trade cap |
| Risk controls MISSING | **Exchange leverage enforcement**, remote kill switch, alerting, testnet mode, instance lock |
| Testnet | **ABSENT** — `.env` sets `HYPERLIQUID_NETWORK=mainnet` |
| DeFi Agent | Not activated |
| User's concern | "5× leverage set but trades open at 20×; 15% risk allocation not producing expected $75 sizing on $100 account" |

---

## 💀 FUND-LOSS RISKS — Fix before ANY live trading

### FL-1 · Exchange leverage never set — root cause of reported 5× → 20× bug
**File:** `src/trading/hyperliquid_api.py` — `place_buy_order`, `place_sell_order`, `place_limit_buy`, `place_limit_sell`

There is no call to `exchange.update_leverage()` anywhere in the codebase. Hyperliquid's `Exchange.market_open()` uses whatever leverage the user last set in the Hyperliquid web UI for that asset. If you previously clicked "20×" on the Hyperliquid interface for BTC, every order the bot places opens at 20× leverage regardless of `MAX_LEVERAGE=5` in `.env`. At 20× leverage a 5% adverse move = 100% margin wipe.

**Fix required:** Add `exchange.update_leverage(asset, is_cross=True, leverage=int(max_leverage))` before each market_open call (or once per startup per asset).

---

### FL-2 · `check_leverage()` formula is mathematically wrong — provides zero protection
**File:** `src/risk_manager.py` — `check_leverage()` lines 139–152

The check computes `effective_lev = alloc_usd / account_value`. A $50 notional on a $100 account returns 0.5× — far below `max_leverage=5` — so the check always passes. But it does not reflect the actual exchange leverage (Finding FL-1). This guard protects against nothing. It will approve a $99 allocation on a $100 account at any exchange leverage setting.

---

### FL-3 · Force-close fires too late — threshold measured against notional, not margin
**File:** `src/risk_manager.py` — `check_losing_positions()` lines 295–311

`loss_pct = abs(pnl / notional) * 100` — where `notional = size × entry_price`. With 5× leverage, `MAX_LOSS_PER_POSITION_PCT=8%` of notional = 40% of margin. With 20× leverage it equals 160% of margin — the position would already be liquidated before force-close activates. The force-close guard is essentially disabled at any real leverage level.

---

### FL-4 · `initial_account_value` resets on every restart — reserve floor erodes after losses
**File:** `src/main.py` — `run_loop()` line 536: `if initial_account_value is None: initial_account_value = account_value`

After a 20% drawdown and bot restart, `initial_account_value` = $80. The balance reserve check `balance >= $80 × 20% = $16` always passes. The original intent was to preserve 20% of the starting $100. Every restart silently lowers the safety floor.

---

## 🔴 CRITICAL — Fix before live trading

### C-1 · Position sizing produces $11 minimum trades — not the $75 the user expects
**File:** `src/risk_manager.py` + `src/main.py` lines 1117–1126

**Root cause of the user's sizing bug.** The user expects: $100 account × 5× leverage = $500 buying power → 15% = **$75**. What actually happens:
1. `atr_position_size = $100 × 1% / SL_pct` — with 2% SL this gives $50; with 3% SL this gives $33
2. `alloc = atr_size × (score/10)` — for score 7: `$50 × 0.7 = $35`
3. `alloc × 0.5` if ADX low and score < 9: `$35 × 0.5 = $17.50`
4. Bumped to minimum $11 by risk manager

`atr_position_size()` does not use leverage in the 1% risk calculation. It sizes based on equity only, ignoring the $500 buying power. The `pct_cap = $500 × 15% = $75` ceiling is never reached because ATR sizing dominates at lower values.

---

### C-2 · AI verdict cache cleared every outer cycle — Claude called hourly regardless of cache settings
**File:** `src/main.py` line 495: `_ai_verdict_cache.clear()`

`AI_APPROVE_CACHE_MINUTES=60` is designed to prevent Claude from being called more than once per hour per asset. But `_ai_verdict_cache.clear()` unconditionally wipes it at the start of every outer cycle (which runs hourly). The 60-minute cache has no effect. The only real throttle is `MIN_AI_CALL_GAP_MINUTES=30`. This doubles intended Claude API cost and call frequency.

---

### C-3 · Inner loop uses stale account state — risk checks for inner-tick trades use outer-loop balance
**File:** `src/main.py` — inner loop lines 1518–1724

`account_value` and `state` (positions dict) are not refreshed across the 11 inner ticks. After the outer loop executes a BTC trade consuming $50 of a $100 account, all subsequent inner ticks for ETH size against `account_value = $100`. This defeats `check_total_exposure`, `check_concurrent_positions`, and the ATR sizing for inner-loop trades.

---

### C-4 · Inner loop calculates TP/SL from prices up to 55 minutes old
**File:** `src/main.py` — inner loop line 1565: `_iprice = asset_prices.get(_ia, 0)`

`asset_prices` is populated once per outer cycle. Inner ticks use this for both `_itp, _isl = _code_compute_tpsl(_ie, _iatr, _idir)` and order sizing. If BTC moves 2% in 55 minutes, the SL placed by an inner-tick trade is calculated relative to the wrong price — potentially placing a stop below the current market price for a long, triggering immediately on fill.

---

### C-5 · No instance lock — two simultaneous instances create double positions
**File:** `src/main.py` — no PID file or lock file anywhere

The repo includes a `trading-agent.service` systemd file. If the user starts the bot manually while systemd is also running it, both instances read `state.json` as IDLE and both enter the same position simultaneously — doubling exposure at the exact wrong moment. No `fcntl.flock()`, no PID file, no socket lock.

---

### C-6 · Reconcile block silently swallowed — stale ENTERED state blocks future entries for 13 hours
**File:** `src/main.py` lines 612–640: `except Exception: pass`

The entire reconcile block is wrapped in bare `except Exception: pass`. If it throws (malformed position data, dict key error), stale active_trades are never cleaned. The state machine stays ENTERED for an asset with no open position. Every subsequent cycle skips that asset with `[STATE GATE] skipped — state=ENTERED`. The asset is effectively dead until the state file is manually edited.

---

### C-7 · No kill switch — cannot stop the bot from opening positions remotely
**File:** `src/main.py` — no kill switch endpoint

The only shutdown is `SIGTERM`/`SIGINT` via terminal. There is no Telegram command, no dashboard stop button, no watchdog file. If the bot begins opening positions during a flash crash and the operator is not at their terminal, there is no way to halt it.

---

### C-8 · No alerting system — losses accumulate silently
**Files:** All — no alerting calls anywhere

There are no email, Telegram, SMS, or webhook alerts for: process crash, unhandled exception, daily circuit breaker activation, SL hit, force-close attempted, force-close failed, Claude API errors, fill failures, or balance threshold breach. The bot can trade, lose, and crash with no notification to the operator.

---

## 🟠 HIGH — Fix before real-money exposure

### H-1 · `entry_confirmed()` sell MACD gate is too permissive
**File:** `src/strategy.py` line 169: `macd_15m < macd_threshold` (where threshold is a positive number)

For sell entries, the condition `macd_15m < macd_threshold` passes when MACD = 0 (neutral). The buy condition correctly requires `macd_15m > -macd_threshold`. The sell should require `macd_15m < -macd_threshold` — genuinely negative MACD. Current code allows sell entries with neutral or mildly-positive MACD, creating counter-trend signals that bypass the intended 15m confirmation gate.

---

### H-2 · Fee tracking only counts one side — dashboard understates cumulative costs by ~50%
**File:** `src/main.py` `_update_stats()` lines 319–321

`stats["total_fees"] += exit_price * qty * 0.00045` — only the exit leg is counted. The entry fee (`entry_price * qty * 0.00045`) is never added. After 100 trades the dashboard shows half the real fee cost.

---

### H-3 · Config defaults are dangerously aggressive if `.env` fails to load
**File:** `src/config_loader.py` lines 96–103

If `.env` fails silently: `MAX_LEVERAGE` defaults to 10 (not 5), `MAX_POSITION_PCT` to 20% (not 15%), `MAX_LOSS_PER_POSITION_PCT` to 20% (not 8%), `DAILY_LOSS_CIRCUIT_BREAKER_PCT` to 25% (not 12%), `MAX_CONCURRENT_POSITIONS` to 10 (not 2). No startup validation confirms all risk parameters loaded correctly before trading begins.

---

### H-4 · TP/SL guardian skips all assets when `open_orders` fetch fails — naked positions during outages
**File:** `src/main.py` lines 665–669

`if not open_orders_ok: break` is correct (prevents mass duplicate TP/SL placement). But if the exchange has even one failed `get_open_orders()` call, all ENTERED positions run without TP/SL monitoring for that cycle. No maximum "unguarded cycles" count triggers an escalating alert.

---

### H-5 · Inner loop uses requested quantity for TP/SL, not confirmed fill quantity
**File:** `src/main.py` lines 1690–1697

Unlike the outer loop (which polls fills 3× to get actual `filled_qty`), the inner loop uses `_iamt = allocation / price` directly for TP/SL placement. A partial fill leaves the unfilled portion unprotected until the next outer cycle guardian.

---

### H-6 · Funding cost uses entry-time rate for the entire trade duration
**File:** `src/main.py` `_log_trade_close()` lines 416–424

The logged P&L applies the funding rate from the moment the position was opened across the full hold duration. Hyperliquid funding rates reset every 8 hours. A position held 24 hours has 3 funding periods, but only the first is used. Logged cumulative P&L drifts from actual P&L for every trade held more than 8 hours.

---

### H-7 · `check_losing_positions` PnL calc does not account for leverage
Already covered in FL-3. Flagged here separately because it affects dashboard loss display: the "loss_pct" shown in add_event is relative to notional, not the user's capital at risk.

---

## 🟡 EDGE CASES — Fix before scale

### E-1 · `near_ema` defaults to `True` in `entry_confirmed` — gate passes when data is missing
**File:** `src/strategy.py` line 138: `near_ema = s15.get("near_ema", True)`
Missing 15m EMA data silently opens the near-EMA gate. The function docstring says "block entry rather than allow through with no confirmation" — this default contradicts the stated intent.

---

### E-2 · Sharpe ratio is not a real Sharpe ratio
**File:** `src/main.py` `calculate_sharpe_from_diary()` lines 1914–1937
`mean_pnl / std_pnl` uses raw dollar amounts, not percentage returns. Dimensionally inconsistent and not comparable to standard benchmarks. Also reads the full diary file on every cycle (grows with time).

---

### E-3 · `decide_trade()` / `_decide()` legacy code still alive with rule-violating prompts
**File:** `src/agent/decision_maker.py` lines 265–789
These methods instruct Claude to "choose buy/sell/hold" and set allocation_usd — violating MASTER RULE 2 (code is primary decision maker) and MASTER RULE 3 (Claude role is fixed). While not called from the main loop, the methods exist and could be accidentally invoked. The system prompt explicitly says "Claude chooses ONE action per asset: BUY | SELL | HOLD" which contradicts the CLAUDE.md architecture.

---

### E-4 · `decisions.jsonl` and `diary.jsonl` grow unbounded — eventual disk-full crash
**File:** `src/main.py` — file appends with no rotation
`bot.log` has RotatingFileHandler (5MB). Neither `diary.jsonl` nor `decisions.jsonl` have rotation. When disk fills, all diary writes fail silently (`except Exception: pass`). This destroys the guardian's TP/SL price data source — the guardian falls back to live-price SL calculation, which is derived from entry-time ATR, not the actual TP/SL placed at trade entry.

---

### E-5 · No confirmation gate before mainnet trading
**File:** `src/main.py` lines 226–228
If `HYPERLIQUID_NETWORK` is set or defaults to mainnet, the bot starts trading immediately. No `DRY_RUN=true` flag, no interactive confirmation prompt, no paper trading mode. First accidental run always hits mainnet.

---

## 🔐 SECURITY

| Severity | Finding |
|---|---|
| Critical | `.env` in project root contains private key and API key — must be confirmed gitignored; if ever accidentally committed, treat private key as compromised |
| High | Zero authentication on all HTTP API endpoints (`/live`, `/diary`, `/logs`, `/fills`) — full account state and trade history exposed to any localhost process |
| High | `config_loader.py` dangerous defaults (MAX_LEVERAGE=10) if `.env` fails silently |
| Medium | `prompts.log` and `llm_requests.log` in HTTP allowlist — contain account balances and API costs |
| Medium | No mainnet confirmation gate — any misconfigured deployment goes live immediately |

---

## ⚡ PERFORMANCE

| Severity | Finding |
|---|---|
| High | `diary.jsonl` and `decisions.jsonl` unbounded growth — `calculate_sharpe_from_diary()` reads entire file every cycle |
| High | `confirm_trade()` uses synchronous SDK in `asyncio.to_thread()` with no asyncio-level timeout — hung threads exhaust executor |
| High | Outer-loop asset data gathered sequentially per asset instead of all assets in one `asyncio.gather()` |
| Medium | `TradeStateMachine._save()` called on every state mutation — 11+ disk writes per outer cycle |
| Low | Price history `deque(maxlen=60)` is fine — no memory leak here |

---

## 🛡️ MISSING RISK CONTROLS

1. **Exchange leverage enforcement** — no `update_leverage()` before orders
2. **Remote kill switch** — no way to halt trading without SSH
3. **Alerting** — no alerts for losses, errors, or circuit breaker activation
4. **Instance lock** — no prevention of simultaneous bot instances
5. **Testnet validation** — no testnet run documented or configured
6. **Position size confirmation** — inner-loop trades not reconciled against actual fills before TP/SL placement
7. **Leverage display** — dashboard shows exchange leverage for open positions but not for the next order

---

## 📊 MONITORING & OPERATIONAL GAPS

- No health check endpoint (the API has no `/health` or `/status` route)
- No daily P&L summary report
- No alert when circuit breaker activates
- No alert when force-close fails
- `bot.log` rotates at 5MB but is only accessible via SSH
- Dashboard shows stale data — it reads last cached `decisions.jsonl` entry, not live state

---

## 📈 BACKTEST VALIDITY ISSUES

- No backtesting capability exists — no simulation mode, no historical replay
- All testing appears to have been done live on mainnet
- `calculate_sharpe_from_diary()` is live-trading Sharpe, not a backtested metric
- No look-ahead bias concern (live trading) but no out-of-sample validation documented

---

## 📦 TECHNICAL DEBT

1. `decide_trade()` / `_decide()` in `decision_maker.py` — 500+ lines of legacy code with Claude-as-decision-maker architecture that violates MASTER RULES; should be deleted or clearly disabled
2. Two representations of the same fee: `TAKER_FEE_PCT = 0.045` (percent form, class constant in risk_manager) vs `taker_fee_pct = 0.00045` (decimal form, CONFIG) — confusing and inconsistent, works correctly by accident
3. `files in project root: `0.15%` and `MAX_TRADE_HOURS` appear to be accidentally created files (fragment text written as filenames) — should be deleted
4. Inner loop duplicates the entire outer loop's score pipeline with no shared function — ~200 lines of near-identical code

---

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
     🏛️  TRADING BOT BUG COUNCIL — COUNCIL VERDICT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RISK LEVEL: CRITICAL — Three fund-loss risks active simultaneously

PRIMARY CONCERN: The bot has no mechanism to enforce the exchange-side
leverage setting. MAX_LEVERAGE=5 in config only affects the risk
calculation formula, not the actual Hyperliquid platform leverage.
If the user's account has 20× set, every trade opens at 20×, making
all position sizing and loss calculations wrong by a factor of 4×.

TOP 3 MUST-FIX:

1. ADD exchange.update_leverage(asset, True, max_leverage) before EVERY
   market order placement (hyperliquid_api.py). This fixes the reported
   5×→20× leverage bug and makes all risk calculations meaningful.

2. FIX check_losing_positions to use loss as % of MARGIN (not notional):
   loss_pct = abs(pnl) / (notional / max_leverage) * 100
   Otherwise force-close fires at 40%+ margin loss at 5× (too late).

3. ADD a remote kill switch (Telegram bot, file-flag polling, or
   dashboard button) so trading can be halted without SSH access.
   This is non-negotiable for any unattended deployment.

ADDITIONAL HIGH-PRIORITY (sizing bug fix):
   The $75 sizing the user expects requires changing atr_position_size
   to incorporate leverage: risk_usd = account_value * leverage * (risk_pct/100)
   OR accept that ATR sizing is the conservative primary cap and the
   pct_cap ($75) is only the ceiling — both are working as designed
   but fighting each other. Document the intent clearly.

CAPITAL RECOMMENDATION: Maximum $50 until FL-1, FL-2, FL-3 are fixed.
At current leverage uncertainty, treat any open position as potentially
4× over-leveraged. Do not add capital until exchange leverage is
confirmed and enforced.

GO LIVE: ❌ NOT READY

Bot should not trade live funds until:
□ Exchange leverage enforced before every order (FL-1)
□ check_leverage formula corrected (FL-2)
□ check_losing_positions uses margin-based loss threshold (FL-3)
□ Remote kill switch implemented (C-7)
□ At minimum basic Telegram alert on circuit breaker + force-close (C-8)
□ Instance lock added (C-5)
□ 24-hour testnet validation run completed

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
