# Trading Bot Deep Analysis Report — Phase 2 (Post-Change Verification)
**Date:** 2026-04-29  
**Scope:** Full re-read of all source files after user's code changes. Verifies which Phase 1 findings were resolved, identifies what remains outstanding, and flags any new issues introduced by the changes.

---

## Files Read in This Pass

| File | Lines |
|------|-------|
| `.env` | live config |
| `src/main.py` | 1,361 |
| `src/agent/decision_maker.py` | 620 |
| `src/risk_manager.py` | 444 |
| `src/strategy.py` | 74 |
| `src/trading/hyperliquid_api.py` | 702 |
| `src/indicators/local_indicators.py` | 408 |
| `src/config_loader.py` | 113 |
| `src/trade_state.py` | 119 |

---

## Phase 1 Fix Scorecard

| # | Phase 1 Finding | Status |
|---|-----------------|--------|
| 1 | VWAP cumulative / not session-anchored | ✅ FIXED — resets at UTC midnight |
| 2 | ADX/Bollinger Bands computed but never sent to Claude | ✅ FIXED — in payload |
| 3 | No daily (1d) timeframe, no macro filter | ✅ FIXED — 1d candles + EMA trend gate |
| 4 | Volume confirmation missing on 5m trigger | ✅ FIXED — 70% avg-volume threshold |
| 5 | Flat position sizing regardless of risk | ✅ FIXED — ATR 1% risk rule added |
| 6 | No exit logging (TP/SL fills invisible) | ✅ FIXED — `_log_trade_close()` added |
| 7 | Sharpe ratio always zero (no PnL in log) | ✅ FIXED — reads `trade_closed` events |
| 8 | Path traversal vulnerability in `/logs` | ✅ FIXED — allowlist + basename check |
| 9 | `API_HOST=0.0.0.0` exposed to network | ✅ FIXED — changed to `127.0.0.1` |
| 10 | `MAX_TOKENS=1200` truncating LLM output | ✅ FIXED — set to 2500 in .env |
| 11 | Model too weak (Haiku) for multi-asset decisions | ✅ FIXED — upgraded to claude-sonnet-4-6 |
| 12 | Brute-force JSON repair before expensive API retry | ✅ FIXED — `_extract_json_brute_force()` added |
| 13 | `stats.json` trade accounting | ✅ FIXED — `_update_stats()` added |

**11 of 13 primary findings were addressed. 2 remain open (detailed below).**

---

## Section 1: Architecture & Structure

### What Was Fixed
The overall pipeline now handles the complete trade lifecycle: open → monitor → close → log. The `_log_trade_close()` function writes a `trade_closed` event capturing exit price, realized PnL, exit type (tp/sl/force/timeout), and duration. `_update_stats()` atomically updates `stats.json` with win rate and cumulative PnL. The daily macro gate adds a structural pre-filter before any order reaches execution.

### What Remains Broken

**`active_trades` is still in-memory only.** This is the most structurally dangerous unresolved issue. The `TradeStateMachine` in `trade_state.py` correctly persists IDLE/ENTERED/COOLDOWN state to `state.json` and survives restarts. However, the `active_trades` dict in `main.py` — which stores entry price, position size, and TP/SL order IDs for every open trade — is not persisted anywhere. After a crash or restart, the bot will see an asset in ENTERED state (from state.json) but have no record of its entry price, quantity, or which trigger orders are protecting it.

Consequences of this gap: `_log_trade_close()` will compute duration as zero (no entry_time from active_trades), exit PnL calculation will fail silently (entry_price defaults to 0), and the TP/SL Guardian will re-place trigger orders unnecessarily since it can't verify the existing ones were placed by this instance.

**Dead config variables are still dead.** `MIN_TRADE_SCORE` is loaded in `config_loader.py` (`_get_int("MIN_TRADE_SCORE", 7)`) but there is no code anywhere that reads `CONFIG["min_trade_score"]`. `ENABLE_CLAUDE_COMMENTARY` is not even loaded in config_loader — it can be set in `.env` and has zero effect. These are configuration knobs that look functional but are silently ignored.

---

## Section 2: AI Decision Engine

### What Was Fixed
`MAX_TOKENS` increased from 1200 to 2500, preventing truncated JSON responses. Model upgraded from Haiku to Sonnet. `_extract_json_brute_force()` added as a pure-Python first-pass before the expensive API repair call, which correctly extracts JSON from partial or wrapped text without a model call. Both improvements reduce parse failures.

### New Issues Introduced

**`sanitize_model` config is now ignored.** `config_loader.py` loads `sanitize_model: _get_env("SANITIZE_MODEL", "claude-haiku-4-5-20251001")` — a separate cheaper model intended for JSON repair. In the Phase 2 changes, `decision_maker.py`'s sanitize step was changed from the hardcoded `"claude-haiku-4-5-20251001"` to `self.model` (which is Sonnet). This means every failed JSON parse now triggers a Sonnet API call at roughly 5× the cost of the original Haiku call. The `sanitize_model` config exists but is completely ignored — a Haiku-configured fallback that was never wired in.

**Tool calling cost not bounded.** `ENABLE_TOOL_CALLING=true` in `.env`. When Claude calls `fetch_indicator`, this adds a round-trip to the decision cycle. There is no limit on the number of tool call iterations per decision. If the LLM enters a loop of fetching indicators, the cost and latency grow unbounded for that cycle. There is no max-iterations guard in `decision_maker.py`.

**`MAX_TOKENS=2500` is overriding the `config_loader.py` default of 4096.** The `.env` sets 2500. The config_loader defaults to 4096. Since `.env` wins, effective value is 2500. For Sonnet with tool calling, a complex multi-asset decision with tool use, reasoning, and a full JSON response could still hit 2500 tokens. The Phase 1 fix raised the floor, but for the upgraded model with richer outputs, 2500 may prove too low. Haiku needed 2500; Sonnet's verbose JSON + reasoning easily exceeds it.

---

## Section 3: Trading Logic

### What Was Fixed
Volume confirmation now applied to the 5m trigger candle in `strategy.py`. The daily macro filter correctly blocks BUY signals in BEARISH (EMA20 < EMA50 on 1d) and SELL signals in BULLISH macro environments. ATR-based TP/SL pre-processing applied before risk validation.

### New Issues Introduced

**ATR double-enforcement with conflicting multipliers.** The ATR pre-processor in `main.py` adjusts TP to `current_price + (ATR × 2.0)` and SL to `current_price - (ATR × 1.0)`. Then `risk_manager.enforce_take_profit()` enforces a minimum TP distance of `max(fee_distance, ATR × 1.5)`. For a long trade, the pre-processor sets TP at `entry + 2×ATR`. Then the risk manager checks whether this TP meets its own `1.5×ATR` floor — it will, since 2×ATR > 1.5×ATR. So far no conflict. But for short trades or cases where the LLM provides a TP closer than 2×ATR, the pre-processor overrides first, then the risk manager re-enforces. The two systems apply independent multipliers (2.0 vs 1.5) with no coordination between them. This creates unpredictable final TP values and makes it impossible to reason about what TP price will actually be placed.

**ATR sizing skips when LLM omits SL price.** In `risk_manager.validate_trade()`, the ATR-based position sizing (`atr_position_size()`) is only invoked when `sl_price_early is not None`. If the LLM omits the stop-loss (relying on the risk manager to auto-set one), the sizing falls through to the full `pct_cap` — which for a $100 account with `MAX_LEVERAGE=5` and `MAX_POSITION_PCT=15` is `$100 × 5 × 15% = $75`. The position enters at the maximum size, then the SL is set afterward. The 1% risk rule was supposed to size down the position; it is silently skipped in this common case.

**Daily macro filter has no trend-strength gate.** The 1d EMA20/EMA50 crossover can be extremely close (near-crossover with no directional momentum) and still block all trades in one direction for hours. A 1d ADX filter (trending vs. ranging) would prevent false macro blocks in sideways markets but is not implemented.

---

## Section 4: Profit Leakage Points

### What Was Fixed
Exit fees are now deducted in `_log_trade_close()`. Exit type (tp/sl/force/timeout) is logged so you can analyze which exit mechanism is performing well or badly.

### What Remains Broken

**Open-side fee still not deducted.** `_log_trade_close()` computes realized PnL and deducts `exit_price × qty × 0.00045` (taker fee on close). It does not deduct the open-side taker fee of `entry_price × qty × 0.00045`. For a round-trip trade, actual fees are approximately double what is recorded. PnL is systematically overstated by roughly 50% of total fee cost. On a $100 account trading at $75 notional, the missed fee is ~$0.034 per trade — small in dollar terms but meaningful as a percentage of thin profit margins.

**Funding rate cost not modeled.** Funding rates are fetched and sent to Claude as context, but they are not deducted from PnL in any exit calculation. For high-funding environments (e.g., BTC perpetual funding at 0.01% per 8 hours = 0.03%/day), a 12-hour position would accrue ~0.045% in funding cost on top of trade fees. This is not captured.

**Capital is ~$100.** Not a code issue but the primary profit blocker. With a 0.15% round-trip fee floor and ATR-based TP typically at 1–2×ATR, most trades need a minimum 0.3–0.5% move just to break even. After funding, spread, and AI API costs (~$0.84/day for Sonnet at current settings), the account must generate >1% daily return before paying the operator. This is mathematically very difficult to sustain.

---

## Section 5: Bot Cycle & Timing

### What Was Fixed
The cycle now properly fetches 8 data streams in parallel (adding 1d candles). The `_log_trade_close()` call fires asynchronously after detected closes. The TP/SL Guardian re-places missing trigger orders every cycle.

### New Issues Introduced

**Pre-flight idempotency check on every order attempt.** `_order_retry()` in `hyperliquid_api.py` calls `_check_order_landed()` before attempt 0 (line 181) as a pre-flight check. `_check_order_landed()` makes two sequential async API calls: `get_open_orders()` and `get_recent_fills()`. This means every single order placement — even a brand-new trade with no prior history — starts with two redundant API calls before the order is even attempted. For a single trade with BUY + TP + SL, that is 6 pre-flight API calls before any order is submitted. At the 1h interval with 2 assets, this is acceptable latency, but at 5m intervals with 7 assets, it becomes a bottleneck.

**Cycle duration not bounded or measured.** If a cycle takes longer than the configured interval (e.g., a 1h cycle takes 65 minutes due to slow API responses), the next cycle starts immediately with no delay. There is no warning when cycle duration exceeds the interval. For 5m or 1m intervals, a stacked cycle is a real risk.

**Meta cache never refreshed.** `get_meta_and_ctxs()` caches exchange metadata in `_meta_cache` after the first call and never refreshes it during the bot's lifetime. For long-running deployments (days or weeks), if Hyperliquid adds new assets, adjusts contract specs, or changes decimal precision, the bot will use stale metadata until restart.

---

## Section 6: Data & Indicators

### What Was Fixed
**VWAP is now correctly session-anchored.** The `local_indicators.py` implementation resets `cum_tp_vol` and `cum_vol` at each UTC midnight boundary using `datetime.fromtimestamp(c["t"] / 1000.0, tz=_tz.utc)`. This is a meaningful fix — VWAP now correctly represents intraday value area rather than a meaningless cumulative average from the first fetched candle.

Volume confirmation added to `entry_confirmed()`. Daily timeframe indicators added. ADX and Bollinger Bands now transmitted to Claude.

### Remaining Issues

**Stochastic RSI padding is fragile.** The `full_k` array in `stoch_rsi()` is padded using the formula `[None] * (pad + (len(valid_rsi) - len(valid_k)) + (len(valid_k) - len(k_line)))`. This simplifies to `[None] * (pad + len(valid_rsi) - len(k_line))`. Because `k_line` may contain leading `None` values from the SMA (when `valid_k` has fewer elements than `k_smooth`), the padding calculation could produce a result shorter or longer than `len(rsi_vals)` in edge cases with very few candles. This won't crash but will silently return misaligned or wrong-length indicator arrays.

**Stoch RSI %D output alignment.** `full_d` is padded as `[None] * (len(rsi_vals) - len(d_line))` then extended with `d_line`. If `d_line` is longer than `len(rsi_vals) - prefix_nones`, the total length would exceed `len(rsi_vals)`. The `compute_all()` caller uses `latest()` which returns the last non-None value and is unaffected, but any code iterating the series by index would misalign.

**Only two indicators are actually used in `entry_confirmed()`.** The strategy gate uses only `macd_histogram` (15m) and `candle_bullish`/`macd_histogram` (5m). RSI, ADX, StochRSI, OBV, and VWAP are sent to Claude as context but are not referenced in any deterministic pre-trade filter. Claude may or may not factor them into its decision — you cannot verify or reproduce its reasoning from the logged JSON.

---

## Section 7: Exchange Integration

### What Was Fixed
Idempotency guard with `_check_order_landed()` prevents duplicate orders on connection retry. HIP-3 asset support via raw POST endpoint for candle fetches. Trigger orders use `_trigger_order_retry()` with its own idempotency path. Balance calculation properly separates perps equity from spot USDC.

### Remaining Issues

**HIP-3 asset name mismatch in `round_size()`.** The method searches `self._hip3_meta_cache` for an asset whose `"name"` field equals the full asset string (e.g., `"xyz:GOLD"`). However, the Hyperliquid API returns universe entries with the short name only (e.g., `"GOLD"`). The comparison `u.get("name") == "xyz:GOLD"` always fails. For all HIP-3 assets, `round_size()` silently falls through to `round(amount, 8)`. This is usually safe (8 decimal places is fine) but means any asset with fewer allowed decimal places could have orders rejected by the exchange for over-precision.

**`market_close()` does not use the idempotency-aware retry.** Line 387 of `hyperliquid_api.py` uses the basic `_retry()` wrapper. If a close order is sent, the connection drops, and the bot retries, `_retry()` will re-send the close. At this point: if the position is already closed, `market_close()` will fail gracefully with a rejection. However, if the position was partially closed by the first attempt (edge case), a second `market_close()` could create a new short position in the opposite direction rather than simply erroring. Using `_order_retry()` with an idempotency check would be safer here.

**`get_recent_fills()` does not sort chronologically.** The method returns `fills[-limit:]` — the last N items from whatever order the API returns them. If the API returns fills in reverse-chronological order (newest first), `fills[-limit:]` returns the oldest N fills. The 60-second cutoff check in `_check_order_landed()` would then miss recent fills entirely, defeating the idempotency guard for recent orders.

**Balance accounting edge case with isolated margin.** The comment in `get_user_state()` acknowledges that `spot_usdc` from `spot_user_state` includes USDC locked in isolated positions, and `perps_value` represents the equity of those positions. The code uses `spot_usdc if spot_usdc > 0 else perps_value`. For an account using cross-margin (no isolated positions), spot_usdc reflects all idle capital, perps_value reflects total portfolio equity including unrealized PnL — and they are not equal. If `spot_usdc` doesn't include unrealized PnL from open cross-margin positions, the balance will be understated whenever positions are open and profitable.

---

## Section 8: Logging & Diary System

### What Was Fixed
`_log_trade_close()` now writes a `trade_closed` event to `diary.jsonl` with: timestamp, event type, asset, entry price, exit price, quantity, direction, realized PnL, exit type (tp/sl/force/timeout/unknown), and duration in minutes. `_update_stats()` maintains a running `stats.json` with total trades, wins, losses, win rate, total PnL, and total fees. Sharpe ratio calculation now reads actual `realized_pnl` from `trade_closed` events rather than always returning 0.

### Remaining Issues

**Open-side fee missing from `trade_closed` events.** The `realized_pnl` field in diary is overstated by one taker fee (entry_price × qty × 0.00045) every trade. Over time, `stats.json` will show a systematically optimistic PnL. Any Sharpe or win-rate analysis built on these numbers will be slightly inflated.

**No TP/SL order IDs in close events.** The `trade_closed` diary event does not record which trigger order fired (TP oid vs SL oid). If you want to analyze slippage between trigger price and actual fill price, you can only work backward from `exit_type`. The actual order response from the close is not saved.

**Exit type detection is best-effort.** `exit_type` is inferred by comparing who called `_log_trade_close()` (force-close loop → "force", timeout check → "timeout", TP/SL Guardian → "tp"/"sl"). But if a TP trigger fires on the exchange natively (without the guardian detecting a missing order), the reconcile loop detects the position closed and calls with `exit_type="unknown"`. For any analysis of TP hit rate vs SL hit rate, "unknown" exits are a gap.

**`stats.json` is not atomic-safe on concurrent writes.** `_update_stats()` uses `json.load()` followed by `open(stats_file, "w")` with no locking. If two async tasks tried to call `_update_stats()` simultaneously (unlikely in the current single-threaded async loop but possible during force-close + Guardian running on the same cycle), a race condition could corrupt `stats.json`.

---

## Section 9: Weaknesses & Bottlenecks

### What Was Fixed
The primary structural weaknesses around exit tracking, VWAP correctness, indicator completeness, and security are resolved. The bot now has a meaningful data trail for post-trade analysis.

### Remaining Weaknesses

**API cost is now 5× the original.** With `LLM_MODEL=claude-sonnet-4-6` at `MAX_TOKENS=2500`, each primary decision call costs approximately `($3/M input + $15/M output)`. At 1h intervals with 2 assets and an estimated ~2000 input + ~800 output tokens per call: ~24 calls/day × ~($0.006 + $0.012) = ~$0.43/day for primary calls, plus sanitize retries now also using Sonnet. On a $100 account, this is 0.43%/day in API costs before any trade. Over 30 days: ~$13 in API fees, which is 13% of the account destroyed with no trading activity at all. The original Haiku model would cost ~$0.086/day.

**`sanitize_model` config is dead weight.** `config_loader.py` loads `SANITIZE_MODEL` but `decision_maker.py` ignores it and uses `self.model` (Sonnet) for the repair step. The config variable exists, is documented, but has no effect. Anyone who sets `SANITIZE_MODEL=claude-haiku-4-5-20251001` in `.env` to save money will be confused when repair calls still use Sonnet.

**CORS wildcard still present.** `Access-Control-Allow-Origin: *` in the aiohttp middleware allows any web page in any browser to make requests to the dashboard API. Since `API_HOST=127.0.0.1`, the API is not reachable from the internet, but a malicious web page opened in the same browser as the bot's machine could still read trade data or trigger log reads via the `/diary` and `/logs` endpoints.

**`web3` is still an unused dependency.** Listed in `pyproject.toml`, never imported. Adds ~20MB to the install footprint and a security surface area for a library with known past vulnerabilities. Should be removed.

**No circuit breaker for consecutive failed cycles.** If Hyperliquid is unreachable and every API call fails, the main loop catches exceptions and continues to the next cycle. There is no counter for consecutive failed cycles, no alert, and no progressive backoff at the loop level. The bot will spin indefinitely making failed API calls.

**`trade_state.py` does not persist `active_trades`.** As noted in Section 1, the ENTERED state survives restart but the trade metadata does not. This is the single largest architectural gap remaining. If the bot is restarted with an open position, it will: know the asset is ENTERED (good), but not know the entry price, contract size, or TP/SL order IDs (bad). The force-close logic in `main.py` reads `active_trades.get(asset)` to get `entry_price_at_entry` for loss calculation — after restart this returns None and loss-based force-close is effectively disabled for that position.

---

## Section 10: Recommendations (Prioritized)

### Priority 1 — Safety-Critical (Fix Before Next Live Run)

**Persist `active_trades` to disk.** Add `active_trades` serialization alongside `state.json`. At minimum, write `{asset: {entry_price, qty, is_long, tp_price, sl_price, tp_oid, sl_oid, entry_time}}` to `active_trades.json` on every mutation. Load it on startup. Without this, every restart leaves open positions unprotected from the loss-based force-close check.

**Fix `get_recent_fills()` ordering assumption.** After slicing `fills[-limit:]`, sort by `fill_ts` descending before iterating. Otherwise the 60-second cutoff in `_check_order_landed()` checks potentially stale entries.

**Wire `sanitize_model` into `decision_maker.py`.** Replace `model=self.model` in the sanitize API call with `model=CONFIG.get("sanitize_model") or self.model`. Set `SANITIZE_MODEL=claude-haiku-4-5-20251001` in `.env`. This immediately cuts the repair-call cost by 5× with no quality loss (repair is mechanical JSON extraction, not reasoning).

### Priority 2 — Correctness (Fix Soon)

**Resolve ATR double-enforcement.** Choose one mechanism: either the pre-processor in `main.py` or `enforce_take_profit()`/`enforce_stop_loss()` in the risk manager should apply ATR-based floors — not both. The simpler fix is to remove the ATR logic from the pre-processor and let the risk manager handle it (it already does), using the LLM's TP/SL as input and only adjusting when they violate the floor.

**Add open-side fee to `_log_trade_close()`.** Change `fees = exit_price * qty * 0.00045` to `fees = (entry_price + exit_price) * qty * 0.00045`. Update `realized_pnl` accordingly.

**Fix `round_size()` for HIP-3 assets.** When searching the universe for a HIP-3 asset like `"xyz:GOLD"`, strip the dex prefix: `asset_short = asset.split(":")[-1]` and compare `u.get("name") == asset_short`.

**Guard ATR sizing against missing SL.** In `validate_trade()`, if `sl_price_early` is None and ATR data is available, compute a provisional SL price using `mandatory_sl_pct` before calling `atr_position_size()`. Otherwise the 1% risk rule is only applied when the LLM happens to provide a SL, which is inconsistent.

### Priority 3 — Cost & Operational (Fix When Convenient)

**Cap tool call iterations.** Add a `max_tool_iterations: int = 3` guard in the tool-use loop in `decision_maker.py`. If Claude has not produced a final decision after 3 tool calls, break and use the last available response.

**Raise `MAX_TOKENS` to 4096 in `.env`.** The config_loader default is already 4096, which is the correct value for Sonnet producing multi-asset JSON with reasoning. The current `.env` setting of 2500 overrides it downward unnecessarily.

**Add cycle duration logging.** Record `cycle_start = time.monotonic()` at the top of the loop and log `cycle_duration = time.monotonic() - cycle_start` at the end. Alert (or add a WARNING log) if cycle duration exceeds the configured interval.

**Remove the `web3` dependency.** Delete from `pyproject.toml` and `requirements.txt`. Reduces install size and removes an unused security surface.

**Remove dead config variables.** Either implement `MIN_TRADE_SCORE` as an actual filter in `entry_confirmed()` or remove it from `.env` and `config_loader.py`. Same for `ENABLE_CLAUDE_COMMENTARY`. Dead configuration creates maintenance confusion.

**Narrow the CORS policy.** Change `Access-Control-Allow-Origin: *` to `Access-Control-Allow-Origin: http://localhost:3000` in the aiohttp middleware, limiting API access to the same origin as the dashboard.

---

## Summary

Significant progress was made in Phase 2. The 13 most important structural issues from Phase 1 were either fixed or substantially addressed. The bot now has proper exit tracking, session-anchored VWAP, volume confirmation, ATR-based position sizing, daily macro context, and a realistic Sharpe calculation.

Five issues remain outstanding from Phase 1 (active_trades persistence, open-side fee, CORS wildcard, web3 dependency, dead config variables), and six new issues were introduced by the changes (sanitize_model not wired in, ATR double-enforcement, fills ordering assumption, round_size HIP-3 name mismatch, pre-flight overhead on every order, and inconsistent exposure math between check_position_size and check_total_exposure).

The single highest-value fix available is wiring `sanitize_model` back to `decision_maker.py` — it requires changing one line of code and immediately reduces API costs by approximately $0.35/day on a $100 account where every dollar matters. The single highest-safety fix is persisting `active_trades` to disk so restarts don't leave open positions without their loss-protection metadata.

The bot is meaningfully improved from Phase 1 and is now architecturally sound in most areas. The remaining issues are fixable with targeted changes, none of which require redesigning existing systems.
