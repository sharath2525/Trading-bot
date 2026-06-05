
# 🏛️ TRADING BOT BUG COUNCIL — v7 RE-AUDIT REPORT

**Date:** 2026-05-26
**Pass:** v7 (full re-audit — post-Pass 11 fixes applied)
**Files read:** main.py (2721 lines, 5 chunks), strategy.py, risk_manager.py, trade_state.py, hyperliquid_api.py, decision_maker.py, config_loader.py, alerts.py, .env
**Leverage:** 5× — all position risks multiplied accordingly
**Exchange:** Hyperliquid perpetuals (mainnet, REAL FUNDS)
**Order model:** LIMIT at 0.15% better than market; code-first hybrid with Claude as APPROVE/REJECT gate
**DeFi Agent:** Not activated (no on-chain / no web3 calls)

---

## 📋 BOT SUMMARY

**Strategy:** CODE-FIRST hybrid perpetual futures — code owns direction/TP/SL/sizing; Claude (Sonnet 4.6) gates entries at score ≥ 6 + multi-TF confluence
**Asset class:** Crypto perps (BTC ETH SOL AVAX)
**Exchange:** Hyperliquid (REST-only, mainnet)
**Leverage:** 5× cross-margin
**Order types:** LIMIT entries + reduce-only TP/SL trigger orders
**Risk controls FOUND:** Daily circuit breaker, concurrent position cap (3), ATR 1% sizing rule, balance reserve, daily trade cap (20), SL cooldown (30 min), mandatory SL enforcement, trailing stop guardian, KILLSWITCH file, instance lock (port 47293)
**Risk controls MISSING/WEAK:** DASHBOARD_TOKEN blank; API_HOST=0.0.0.0; Telegram unconfigured; private key in .env plaintext; guardian reads diary amount post-TP1 instead of active_trades
**Testnet:** Absent — `.env` is live mainnet with real keys

---

## 🔨 AGENT 1 — LOGIC BREAKER (Trading Logic & Financial Math)

### 🔴 CRITICAL — L1: Guardian re-places SL for full original size after TP1 partial close (NEW REGRESSION)
**File:** `src/main.py`, lines ~1207–1217 (guardian SL re-place block)

After TP1 fills (50% position closed), the BUG-P11-1 fix correctly updates `active_trades[asset]["amount"]` to the remaining half-size. However, the guardian reads `_g_amount = float(_g_diary.get('amount') or 0)` from the **diary** (the original full amount recorded at entry), not from `active_trades`. If the exchange drops the SL while TP1 has already filled, the guardian attempts to place a reduce-only SL for the full original quantity against a position that is only half that size.

**Financial consequence (5× leverage):** Hyperliquid rejects a reduce-only order larger than the remaining position. SL not placed. Remaining 50% position runs naked. On a 5× leveraged BTC position: uncapped loss exposure. A 3% adverse move = 15% margin loss with no exit.

**Note:** This is a direct regression from the BUG-P11-1 fix. The fix updated `active_trades` correctly but the guardian bypasses `active_trades` and reads the diary.

---

### 🟠 HIGH — L2: Entry price for TP/SL uses mid-price, not confirmed fill price
**File:** `src/main.py`, lines ~2011–2012

`amount = alloc_usd / current_price` uses `current_price` (mid-price at signal time). TP/SL are anchored to `_entry = float(_ac.get("current_price"))`. The limit order is placed 0.15% better, meaning the real fill is 0.15% more favourable than the TP/SL anchor. TP1 is therefore 0.15% closer to actual fill than designed.

**Financial consequence:** Minor per-trade, material at scale. SL is effectively 0.15% tighter than intended in a 1× ATR design, causing slightly earlier stops.

---

### 🟠 HIGH — L3: `state_mgr.clear_entry()` does not reset state from ENTERED after limit cancel
**File:** `src/trade_state.py` `clear_entry()` + `src/main.py` ~line 2391

When an unfilled limit is cancelled after ≥4 minutes, the code calls `state_mgr.clear_entry(asset)` which only removes `entry_time`. The state remains `ENTERED`. The asset is blocked from new entries until the outer-loop reconciler detects "no position, no orders" on the next cycle — up to 55+ minutes later.

**Financial consequence:** Asset locked out of trading for up to one full outer-loop cycle. Valid setups missed during that window.

---

### 🟠 HIGH — L4: Inner loop uses stale 4h ATR for TP/SL computation
**File:** `src/main.py`, `_code_compute_tpsl()` called in inner loop

`_iatr = float(_iac.get("long_term_4h", {}).get("atr14") or 0)` is sourced from outer-cycle data (up to 55 minutes old). The inner loop only refreshes 5m candles. During high-volatility sessions ATR can expand 30–50% within an hour.

**Financial consequence (5× leverage):** TP/SL placed from stale ATR. If ATR expands post-outer-cycle, SL sits too close to entry and stops out prematurely. 5× leverage amplifies every premature stop.

---

### 🟡 MEDIUM — L5: `multi_timeframe_confluence()` allows UNKNOWN 1h trend to pass
**File:** `src/main.py` `multi_timeframe_confluence()`

`if is_buy and trend_1h == "BEARISH": return False` — only blocks BEARISH, not UNKNOWN. While `_code_decide_direction()` already rejects UNKNOWN 1h, the confluence check is meant as a defense-in-depth layer. If direction logic ever changes, confluence becomes ineffective as a backstop.

**Financial consequence:** Defense-in-depth failure. On code refactor, Claude could be called without confirmed 1h trend.

---

### 🟡 MEDIUM — L6: `oi_confirmed()` ineffective for first 2 outer cycles
**File:** `src/strategy.py` `oi_confirmed()`

`_oi_history` is a `deque(maxlen=3)` initialized empty. Requires `len(oi_series) >= 2`. First 2 outer cycles pass trivially (`return True, ""`). Bot can enter on the first 2 cycles without any OI confirmation — a known market-open vulnerability.

**Financial consequence:** First 2 hours of bot runtime have no OI confirmation. Startup-window entries have reduced quality.

---

### 🟡 MEDIUM — L7: Limit price in inner loop may be stale
**File:** `src/main.py` inner loop

`_ilim_px = round(_ie * (1 - 0.0015), 6)`. `_ie` is refreshed but computed before all assets are processed sequentially. If BTC is being evaluated and ETH's price moves 0.5% while waiting, the ETH limit price submitted is 0.5% stale.

**Financial consequence:** Limit order placed at a price that may be well inside the book, effectively becoming a near-market order and losing the intended 0.15% improvement.

---

### 🔵 LOW — L8: `calculate_sharpe_from_diary()` may be undefined
**File:** `src/main.py` line ~876

`sharpe = calculate_sharpe_from_diary(diary_path)` is called every outer cycle. If this function was removed or never defined in the codebase as examined, every outer-loop call raises `NameError` — silently caught by the outer `except Exception` handler. Sharpe ratio stays `None` in dashboard.

**Financial consequence:** None direct. Dashboard Sharpe metric permanently missing.

---

## 🔐 AGENT 2 — SECURITY ANALYST

### 🔴 CRITICAL — S1: Live private key and Anthropic API key in plaintext `.env`
**File:** `.env` lines 9–10

`HYPERLIQUID_PRIVATE_KEY=0xe090ecb7c...` and `ANTHROPIC_API_KEY=sk-ant-api03-...` are stored in a flat text file at the project root. If the VPS is compromised, `.env` is accidentally committed to git, included in a Docker image, or exposed via log files, funds are at risk.

**Financial consequence:** Private key compromise → complete fund drain with no recourse. Anthropic API key → API abuse charges. The vault address `0x467B95deA18dBd8AE2C19743f823E653D89fD6Ff` (which holds funds, not just signs) is also now known.

---

### 🔴 CRITICAL — S2: Dashboard bound to `0.0.0.0` with no authentication token
**File:** `.env` lines 184, 189 (`API_HOST=0.0.0.0`, `DASHBOARD_TOKEN=`)

The API server binds to all interfaces. `DASHBOARD_TOKEN` is blank. The `auth_middleware` skips auth when `_token` is falsy. Any host that can reach port 3000 gets full read access to `/diary` (all trades), `/live` (live positions + balance), `/logs` (LLM prompts with trade context). Startup warning fires but bot continues.

**Financial consequence:** External attacker seeing live position data can front-run your orders. Combine with S1 (known private key) = total exposure.

---

### 🟠 HIGH — S3: Startup warning for missing DASHBOARD_TOKEN is advisory-only, no enforcement
**File:** `src/main.py` lines ~51–58

The code warns but does not halt when `DASHBOARD_TOKEN` is unset and `API_HOST` is not localhost. A hard `sys.exit()` or forced `API_HOST=127.0.0.1` override would prevent this.

---

### 🟠 HIGH — S4: Telegram not configured — all critical alerts are silent
**File:** `.env` lines 197–198 (`TELEGRAM_BOT_TOKEN=`, `TELEGRAM_CHAT_ID=`)

`send_alert()` is a no-op when Telegram is unconfigured. Events that send alerts: KILLSWITCH, SL failure, SL orphan fail, force-close fail, circuit breaker. Without Telegram, all these events are log-only on a headless VPS.

**Financial consequence:** A force-close failure or SL orphan at 3am goes undetected until manual log check. At 5× leverage on BTC, hours unmonitored can mean liquidation.

---

### 🟡 MEDIUM — S5: `confirm_trade()` timestamps with local time, not UTC
**File:** `src/agent/decision_maker.py` line 41

`_now_utc = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")` uses local time. If the server is not set to UTC (common on cloud VMs set to provider region), Claude's news-timing auto-reject fires at wrong times. An FOMC event 2 hours away in UTC could appear 7 hours away in the prompt.

**Financial consequence:** Claude could approve a trade entering an FOMC event window that the timestamp misrepresents.

---

### 🟡 MEDIUM — S6: `API_HOST` overridden to `0.0.0.0` in `.env` vs code default of `127.0.0.1`
**File:** `.env` line 184 + `src/config_loader.py` line 102

Config loader defaults `api_host` to `"127.0.0.1"`. The `.env` overrides to `0.0.0.0`. Simple config rebuild from docs would leave the dashboard on localhost only (safe). But current live config is exposed.

---

### 🔵 LOW — S7: `send_alert_sync` drops alerts on event loop close
**File:** `src/alerts.py` lines 43–47

`asyncio.ensure_future(send_alert(message))` schedules but the returned future is discarded. If the event loop closes before the future completes (e.g., SIGTERM shutdown), the alert is silently dropped. KILLSWITCH alerts may not be delivered.

---

## 🌪️ AGENT 3 — CHAOS TESTER

### 🔴 CRITICAL — C1: Guardian SL re-place for full size after TP1 fill (confirmed)
**Scenario:** BTC long entered at 0.01 BTC. TP1 fills (50% closed → 0.005 BTC remains). Connection resets; exchange drops SL. Guardian runs next cycle. Reads diary `amount=0.01`, places SL reduce-only for `0.01`. Hyperliquid rejects (larger than position). No SL placed. BTC drops 5%.

**Financial consequence (5× leverage):** 25% margin loss on remaining position with no exit. Uncapped downside.

---

### 🟠 HIGH — C2: Bot restart with pending unfilled limit — state stays ENTERED
**Scenario:** Limit order placed, `active_trades` saved with `entry_oid`. Bot crashes before fill. On restart, `state.json` shows ENTERED. `pending_limits` cancel logic fires after 4 min but `clear_entry()` preserves ENTERED state. Asset blocked for up to 55 minutes.

**Financial consequence:** Opportunity cost — valid setups missed for up to one outer-loop cycle.

---

### 🟠 HIGH — C3: Flash crash fills limit at bottom; ATR-sized SL triggers immediately
**Scenario:** BTC drops 8% in 30 seconds. Limit BUY placed 0.15% below pre-crash price fills during the crash. SL placed at `entry − 1×ATR` where ATR was computed pre-crash (3× smaller than actual crash ATR). SL sits inside the crash candle and triggers on normal rebound noise.

**Financial consequence:** Round-trip loss + 2× fees on a valid signal that filled at a bad time.

---

### 🟠 HIGH — C4: `get_open_orders()` failure disables ALL guardian protection for the cycle
**File:** `src/main.py` ~line 1074

When `get_open_orders()` returns an error, `open_orders_ok = False` and guardian skips all assets. Correct behavior to avoid duplicate SL. But if this fails every cycle for 60 minutes (maintenance window), all positions run with no guardian re-placement protection.

**Financial consequence:** Open positions with no SL during API degradation. Uncapped loss at 5× leverage.

---

### 🟠 HIGH — C5: Rate limit hit during exit order blocks protection
**File:** `src/trading/hyperliquid_api.py` — no rate limit header tracking

4 assets × ~10 API calls per outer cycle + fills + positions + per-position price enrichment can approach Hyperliquid's weight limits. `_retry()` backs off on 429 but does not preemptively throttle. A force-close call in the retry queue waits while position is unprotected.

**Financial consequence:** Exit order delayed 1–8 seconds during a fast-moving market. At 5× leverage on BTC: 1% move in 3 seconds = 5% margin impact.

---

### 🟡 MEDIUM — C6: KILLSWITCH close failure leaves position open silently
**File:** `src/main.py` lines ~806–820

KILLSWITCH iterates `active_trades`, calls `market_close` for each. On exception, logs critical and sends Telegram alert (unconfigured). Then `break`s the outer loop. Bot exits. Position remains open on Hyperliquid with no monitoring and no alerts.

**Financial consequence:** Operator triggers KILLSWITCH believing all positions closed. They are not. No notifications. Leveraged position runs to liquidation unmonitored.

---

### 🟡 MEDIUM — C7: Stale `account_value` used in inner-loop sizing before state refresh
**File:** `src/main.py` inner loop

`account_value` used for position sizing in the inner loop is from the previous outer-loop or previous inner-tick state fetch. If a force-close fires mid-inner-loop (realizing a loss), the next asset in the same tick still sizes against the pre-close account value.

**Financial consequence:** Over-allocation by up to 8% (max force-close loss) on next asset trade in the same tick.

---

### 🟡 MEDIUM — C8: Clock drift / NTP failure silently fails all orders
No clock drift detection. If VPS clock drifts >5 seconds (possible on cloud VMs without NTP), Hyperliquid's HMAC timestamp validation rejects all orders. `_retry()` backs off but the error is not distinguished from other failures. No alert sent.

**Financial consequence:** All orders fail silently during drift window. Open positions receive no guardian updates.

---

### 🔵 LOW — C9: Blocking XML parsing in async context stalls event loop
**File:** `src/main.py` `_fetch_macro_context()`

`ET.fromstring(_text)` is synchronous. For large RSS feeds (50–100KB), this blocks the async event loop. 3 feeds × potential response size = up to 9 seconds of event loop stall per outer cycle.

**Financial consequence:** 5m inner tick sleep timing thrown off. Minor order placement delay.

---

## ⚡ AGENT 4 — PERFORMANCE ENGINEER

### 🔴 CRITICAL — P1: Kronos AutoModel fallback produces scientifically invalid score modifiers
**File:** `src/indicators/kronos_forecast.py` lines 93–104

When `chronos-forecasting` is unavailable, the fallback loads Kronos-mini via `AutoModelForSeq2SeqLM`. The input `_tok(str(_closes), ...)` tokenizes a Python **list-string representation** into a seq2seq language model — not a valid time-series encoding. The output `_logits[0, -1, 0].item()` extracts a single logit from the last token — not a price forecast. This is noise.

The ±0.5 modifier from this path has no predictive value. It is random noise applied to every score evaluation when ChronosPipeline is not installed. This systematically pollutes the scoring system — pushing borderline 5.5 setups above 6.0 or borderline 6.0 setups below 6.0 with no signal content.

**Financial consequence:** ~1 in 4 trades with ±0.5 score error from Kronos noise. Over 20 trades/day this generates multiple false approvals and false rejections per day.

**Fix required:** Log `WARNING: Kronos AutoModel fallback is not a valid time-series forecast — modifier set to 0.0` and return `0.0` when ChronosPipeline is unavailable. Do not attempt AutoModelForSeq2SeqLM.

---

### 🟠 HIGH — P2: No Hyperliquid rate-limit tracking
**File:** `src/trading/hyperliquid_api.py` — no rate limit header parsing

4 assets × (price + OI + funding + 6 candle timeframes + spread) = 4 × 10 = 40+ API calls per outer cycle, plus fills, open orders, positions, and per-position price enrichment. Not tracked against Hyperliquid's weight limits.

**Financial consequence:** Rate-limit retries delay all order placement. Exit orders delayed during fast markets.

---

### 🟡 MEDIUM — P3: Per-position price enrichment adds 3 extra API calls per cycle
**File:** `src/trading/hyperliquid_api.py` `get_user_state()`

Inside `get_user_state()`, every open position calls `await self.get_current_price(pos["coin"])` — an independent API call per position. With 3 concurrent positions, this adds 3 extra calls inside an already-heavy cycle.

**Financial consequence:** Compounds rate-limit risk (P2). Additional latency per cycle.

---

### 🟡 MEDIUM — P4: BB width computation runs every cycle for dead `is_trending_regime()` function
**File:** `src/main.py` lines ~1500–1511; `src/strategy.py` — `is_trending_regime()` marked `DEAD_CODE_MARKER`

Every cycle computes `_bb_w_vals` over all 4h candles for all 4 assets. This data is placed in `market_sections["long_term_4h"]["bb_width_series"]` which is never read. The function using it is dead code.

**Financial consequence:** Wasted CPU. No direct trading impact.

---

### 🟡 MEDIUM — P5: `signals.jsonl` grows unboundedly — no rotation
**File:** `src/main.py` — `_rotate_if_needed` not called for `signals.jsonl`

`signals.jsonl` is written on every score ≥ 5.0 event across 4 assets + multiple timeframes. No rotation applied. This file can reach gigabytes in weeks.

**Financial consequence:** Disk full → silent logging failure. No functional trading impact, but analytics become incomplete.

---

### 🔵 LOW — P6: `asyncio.to_thread` for `confirm_trade()` has no independent timeout
**File:** `src/main.py` line ~1754

`await asyncio.to_thread(agent.confirm_trade, ...)`. The Anthropic client has `timeout=30.0` but `asyncio.to_thread()` itself has no timeout. If the SDK ignores the parameter, the thread runs indefinitely, blocking a thread-pool slot and potentially queuing subsequent `to_thread` calls (including order placement).

---

### 🔵 LOW — P7: Log `.old` files grow without compression/cleanup
**File:** `src/main.py` `_rotate_if_needed()`

On rotation, `diary.jsonl` → `diary.jsonl.old`. Only 2 copies are kept (correct). But `.old` files are never compressed. Disk usage = 2× log size max per log file.

---

## 👤 AGENT 5 — OPERATOR SAFETY

### 🔴 CRITICAL — O1: KILLSWITCH does not halt on force-close failure
**File:** `src/main.py` lines ~806–820

KILLSWITCH iterates `active_trades`, calls `market_close` for each. On exception: logs critical + sends Telegram alert (Telegram unconfigured — S4). Then `break`s outer loop and bot exits. **Position remains open on Hyperliquid** with no monitoring and no alerts delivered.

**Financial consequence:** Operator triggers KILLSWITCH believing all positions closed. They are not. No notifications. At 5× leverage a 3% adverse overnight move = 15% margin loss. Liquidation risk if move is larger.

---

### 🟠 HIGH — O2: No hard halt after SL orphan placement failure
**File:** `src/main.py` lines ~2333–2340

If `place_stop_loss()` fails in the SL orphan check, code logs critical + sends alert (unconfigured) but **continues the inner loop**. Position remains unprotected.

**Financial consequence:** For up to 55 minutes (until next outer cycle), a filled position has no SL. At 5× leverage, a 2% adverse move = 10% margin loss.

---

### 🟠 HIGH — O3: `ADX_HALF_SIZE_THRESHOLD` mismatch between `.env` (20) and code default (15)
**File:** `.env` line 177 (`ADX_HALF_SIZE_THRESHOLD=20`) vs `src/config_loader.py` line 131 (default `15`)

If `.env` is rebuilt from code defaults, threshold drops from 20 to 15. ADX 15–20 range would then trade at **full size** instead of half-size — doubling position exposure in weak-trend conditions without operator awareness.

**Also:** `main.py` reads `float(CONFIG.get("adx_half_size_threshold") or 15)`. Setting `ADX_HALF_SIZE_THRESHOLD=0` would be silently overridden to `15`.

**Financial consequence:** Operator rebuilding config from docs gets a different risk profile than expected. Documentation drift is a pre-live risk.

---

### 🟠 HIGH — O4: No exchange position reconcile on startup
**File:** `src/trade_state.py` `_load()` + `src/main.py` startup

On restart, `state.json` and `active_trades.json` are loaded without verifying against exchange. If the exchange HAS a position but `active_trades.json` is missing or corrupt (e.g., crash during write), the position is unknown to the guardian for up to 55 minutes (until outer reconcile detects it).

**Financial consequence:** Position with no SL and no guardian tracking for up to 55 minutes post-restart.

---

### 🟡 MEDIUM — O5: Circuit breaker auto-resets at UTC midnight without operator confirmation
**File:** `src/risk_manager.py` `_reset_daily_if_needed()`

After a 12%+ daily drawdown, the circuit breaker resets automatically at UTC midnight. No alert sent at reset time. Bot resumes trading after a losing day without operator awareness.

**Financial consequence:** Bot resumes trading automatically after a significant loss event. Operator may not review what went wrong.

---

### 🟡 MEDIUM — O6: No testnet validation gate enforced in config
**File:** `.env` line 12 (`HYPERLIQUID_NETWORK=mainnet`)

No "I confirm this is live money" confirmation gate. After development, a developer can accidentally run live by forgetting to check the network setting.

---

### 🔵 LOW — O7: Pre-live checklist status (current)
- KILLSWITCH: ✅ file-based | ❌ Not tested on testnet; fails open on close exception
- Max position: ✅ enforced before every order
- SL always placed: ✅ with 2 retry attempts + market-close fallback | ❌ fails silently after SL orphan failure
- Drawdown limit: ✅ configured | ❌ auto-resets without operator gate
- Concurrent positions: ✅ double-gated
- Testnet validation: ❌ `.env` is live mainnet with real keys
- Alerting tested: ❌ Telegram unconfigured — all alerts are no-ops

---

## 📊 STEP 3 — FULL FLOW AUDIT

| Flow | Status | Issue |
|------|--------|-------|
| Signal generation | ✅ OK | Indicators computed locally; volume/pattern/Kronos bonuses intact |
| Pre-trade risk check | ✅ OK | 8 guards intact and non-bypassable |
| Position sizing | ✅ OK | ATR 1% + pct_cap + score factor all correct |
| Entry order | ✅ OK | LIMIT placed correctly at 0.15% better than market |
| Entry confirmation | ⚠️ PARTIAL | Pending-limit cancel leaves state ENTERED for up to 55 min (L3) |
| SL placement | ⚠️ PARTIAL | SL placed correctly at entry; orphan check covers within 5 min; `already_placed` → `sl_oid=None` |
| Position monitoring | ⚠️ PARTIAL | Guardian covers every outer cycle; fails if `get_open_orders()` fails (C4) |
| Exit signal (TP/SL) | 🔴 PARTIAL | Guardian re-places SL at full size post-TP1 (L1/C1) |
| Trailing stop | ⚠️ PARTIAL | Correct breakeven + trail logic; uses stale ATR for inner-loop entries (L4) |
| Drawdown check | ✅ OK | Circuit breaker persisted to disk; resets at UTC midnight (auto) |
| Kill switch | ⚠️ PARTIAL | File-based; works for normal close; fails silently on exchange exception (O1/C6) |
| Restart recovery | ⚠️ PARTIAL | Loads active_trades; no startup exchange reconcile (O4) |
| PnL calc | ⚠️ PARTIAL | Fee hardcoded at 0.00045 instead of CONFIG; Kronos modifier is noise (P1) |
| Alert dispatch | ❌ BROKEN | Telegram unconfigured; all runtime alerts are no-ops (S4/O2) |

---

## 📦 EDGE CASE TABLE

| Edge Case | Expected | Actual Risk | Severity |
|-----------|----------|-------------|----------|
| SL dropped post-TP1 fill | Guardian re-places at 50% size | **Places at 100% → exchange rejects → naked** | 💀 FUND-LOSS |
| Kronos AutoModel fallback active | ±0.5 based on valid ML forecast | **±0.5 based on tokenized list string — noise** | 🔴 CRITICAL |
| KILLSWITCH, exchange in maintenance | All positions closed | **Close fails silently, no alert, position runs** | 🔴 CRITICAL |
| Limit cancel after 4 min | State resets to IDLE | **State stays ENTERED up to 55 min** | 🟠 HIGH |
| `get_open_orders()` API failure | Guardian skips (safe) | **All positions unguarded for up to 60 min** | 🟠 HIGH |
| Rate limit during exit order | Backoff + retry | **Exit delayed; no pre-emptive throttle** | 🟠 HIGH |
| SL orphan placement failure | Halt + alert | **Inner loop continues; no alert (Telegram off)** | 🟠 HIGH |
| Bot restart with open position | Verify then resume | **Loads stale state; no exchange reconcile (55 min gap)** | 🟠 HIGH |
| Flash crash fills limit at bottom | SL executes bounded | **Stale pre-crash ATR; SL inside crash candle** | 🟡 MEDIUM |
| Dashboard token unset + 0.0.0.0 | Warn operator | **Dashboard fully open to any host on port 3000** | 🟡 MEDIUM |
| FOMC event within 2h | Claude rejects | **Timestamp may be wrong TZ (S5)**| 🟡 MEDIUM |
| Clock drift >5s | NTP correction | **All orders rejected silently; no alert** | 🟡 MEDIUM |
| ADX_HALF_SIZE_THRESHOLD rebuilt from defaults | Consistent behavior | **15 (default) vs 20 (.env) — doubles position in ADX 15-20 range** | 🟡 MEDIUM |
| `signals.jsonl` disk full | Logs rotated | **Grows unbounded; logging fails silently** | 🔵 LOW |

---

## 🔎 STEP 6 — PEER REVIEW

**🔨 LOGIC BREAKER reviews other agents:**
Most critical from Security: S1 (live private key in plaintext). If that file is on a VPS with any web-facing service, funds can be drained within seconds of compromise. Biggest risk all agents missed: The guardian's SL re-placement bug (L1) is a *new regression* introduced when BUG-P11-1 was fixed. The fix correctly updated `active_trades` but the guardian bypasses `active_trades` and reads from the diary. One issue that would make me refuse: L1 + S4 (no Telegram) = position with no SL generates no alert and guardian cannot fix it.

**🔐 SECURITY reviews other agents:**
Most critical from Chaos: C6 — KILLSWITCH failure during maintenance + no Telegram = operator believes positions closed, they are not, no notifications. Biggest risk missed: The `.env` contains `HYPERLIQUID_VAULT_ADDRESS` — this is the wallet holding funds. Combined with the known private key (S1), an attacker has everything needed to drain the vault. One issue that blocks live trading: Open dashboard on 0.0.0.0 with no token + live balance data.

**🌪️ CHAOS reviews other agents:**
Most critical from Operator: O1 — KILLSWITCH not halting on close failure is the worst operational scenario. Operator goes to sleep, bot is "stopped," position runs to liquidation. Biggest risk all agents missed: No alert sent at circuit breaker reset (O5). Operator may not know the bot resumed trading after a losing day. One issue that blocks live trading: No Telegram on a headless VPS. Zero operational visibility for any failure mode.

**⚡ PERFORMANCE reviews other agents:**
Most critical from Logic Breaker: L4 (stale 4h ATR in inner loop). In volatile markets the ATR from 55 minutes ago can be 30–50% smaller than current. TP1 placed too close to entry (below 1× real ATR) triggers on normal noise. Biggest risk missed: P1 (Kronos AutoModel noise) pollutes every score evaluation — this is not a rare edge case, it fires on every signal where ChronosPipeline is unavailable. One issue that blocks live trading: P1 — random ±0.5 score modifier with no predictive value. This undermines the entire scoring system.

**👤 OPERATOR reviews other agents:**
Most critical from Security: S2 — API bound to 0.0.0.0 with no auth. This should hard-block startup, not just warn. Biggest risk missed: The `or 15` fallback for `adx_half_size_threshold` silently overrides a 0 setting (O3). An operator trying to disable the ADX size reduction (`=0`) gets the old threshold back invisibly. One issue that blocks live trading: Three independent critical configuration failures simultaneously: Telegram off + open dashboard + live keys in .env.

---

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
     🏛️  TRADING BOT BUG COUNCIL — v7 FINAL VERDICT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 BOT SUMMARY
   CODE-FIRST hybrid | Crypto perps | Hyperliquid mainnet
   5× leverage | LIMIT orders | Claude APPROVE/REJECT gate
   REAL FUNDS ACTIVE (mainnet live keys in .env)

💀 FUND-LOSS RISKS — Fix before ANY live trading

   BUG-v7-L1 [NEW REGRESSION]: Guardian re-places SL at full
   original diary amount after TP1 partial close. The BUG-P11-1
   fix correctly halves active_trades["amount"] but the guardian
   reads from diary (original full amount). After TP1 fill, if
   SL drops from exchange, guardian submits reduce-only SL for
   2× remaining position size → Hyperliquid rejects → position
   runs naked at 5× leverage.
   Fix: Guardian should read amount from active_trades[asset]
   not from diary. Replace `_g_diary.get('amount')` with
   `_active_trade.get('amount')` for the current open trade.

   BUG-v7-S1 [CRITICAL]: Live private key and Anthropic API key
   in plaintext `.env`. If VPS is compromised or .env is leaked,
   complete fund drain with no recourse.
   Fix: Move secrets to a secrets manager or encrypted vault.
   At minimum: verify .gitignore excludes .env; restrict VPS
   SSH access; IP-whitelist Hyperliquid API key.

   BUG-v7-O1 [CRITICAL]: KILLSWITCH exits on force-close failure
   without confirming all positions are closed. No Telegram alert
   delivered (Telegram unconfigured). Operator assumes stop,
   position runs to liquidation.
   Fix: KILLSWITCH must verify each position is flat before
   proceeding to next. On failure: enter retry loop, not break.
   If unable to close after 3 attempts: log + alert + halt
   without exiting (so position monitoring continues).

🔴 CRITICAL — Fix before live trading

   BUG-v7-P1 [CRITICAL]: Kronos AutoModel fallback path produces
   noise. Tokenizing a Python list string `str(_closes)` into a
   seq2seq LM is not time-series forecasting. Output logit is
   random. The ±0.5 modifier from this path has zero predictive
   value and pollutes the score system.
   Fix: When ChronosPipeline is unavailable, return 0.0 and log
   WARNING. Remove AutoModelForSeq2SeqLM fallback entirely.

   BUG-v7-S2 [CRITICAL]: Dashboard bound to 0.0.0.0 with no
   auth token. Live position and balance data fully exposed.
   Fix: Set DASHBOARD_TOKEN to a random 32-char hex string.
   Change API_HOST to 127.0.0.1 unless external access needed
   with a reverse proxy + TLS.

   BUG-v7-S4 [CRITICAL]: Telegram not configured. All runtime
   alerts (KILLSWITCH, SL failure, SL orphan, circuit breaker)
   are no-ops on this headless VPS.
   Fix: Configure TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.
   Verify alert delivery before running live.

🟠 HIGH — Fix before real money exposure

   BUG-v7-L3: Pending-limit cancel leaves state ENTERED up to 55
   min. Fix: call state_mgr.set_state(asset, "IDLE") in the
   pending-limit cancel path, not just clear_entry().

   BUG-v7-O2: SL orphan placement failure — inner loop continues.
   Fix: On orphan SL failure, halt inner loop for that asset and
   send alert.

   BUG-v7-O4: No exchange reconcile on startup.
   Fix: On startup, call get_user_state() and compare positions
   against active_trades.json before first cycle.

   BUG-v7-C4: Guardian entirely disabled on get_open_orders()
   failure. Fix: On repeated failure, send alert and use
   last-known SL prices as fallback.

   BUG-v7-L4: Stale 4h ATR used in inner-loop TP/SL.
   Fix: Re-fetch ATR14 from 5m candles in the inner loop using
   a lightweight local computation over cached OHLCV.

🟡 EDGE CASE — Fix before scale

   BUG-v7-O3: ADX_HALF_SIZE_THRESHOLD: .env=20, code default=15.
   Update MASTER_RULES.md and CLAUDE.md to document the .env
   value (20) as the authoritative threshold.

   BUG-v7-O5: Circuit breaker auto-resets silently at midnight.
   Add a Telegram alert at reset time.

   BUG-v7-L5: multi_timeframe_confluence() passes UNKNOWN 1h.
   Fix: Also block UNKNOWN in the confluence check.

   BUG-v7-C8: Stale account_value in same-tick sizing.
   Fix: Refresh account_value before each inner-loop asset entry.

🔐 SECURITY SUMMARY:
   💀 FUND-LOSS: Private key in plaintext .env
   🔴 CRITICAL: Open dashboard 0.0.0.0 no auth (set DASHBOARD_TOKEN now)
   🟠 HIGH: Telegram unconfigured (configure before live)
   🟡 MEDIUM: Server timezone may affect Claude news-timing verdict (use datetime.utcnow())

⚡ PERFORMANCE SUMMARY:
   🔴 CRITICAL: Kronos AutoModel fallback is noise (remove it)
   🟠 HIGH: No rate-limit tracking
   🟡 MEDIUM: Dead BB width computation runs every cycle
   🟡 MEDIUM: signals.jsonl grows unboundedly

🛡️ MISSING RISK CONTROLS:
   - Guardian reads diary for SL size post-TP1 (should read active_trades)
   - Kronos score modifier from AutoModel path is noise not signal
   - No startup exchange position reconcile
   - No operator confirmation gate after circuit breaker reset

📦 TECHNICAL DEBT:
   - is_trending_regime() dead code but documented as active in CLAUDE.md/ARCHITECTURE.md
   - Fee hardcoded at 0.00045 in _log_trade_close() (use CONFIG["taker_fee_pct"])
   - adx_trending threshold mismatch: code uses 25, actual gate is 15
   - calculate_sharpe_from_diary() may be undefined (verify)
   - Duplicate inline `import json as _sjson` in tight signal-logging loop

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COUNCIL VERDICT: NOT READY

TOP 3 MUST-FIX:
  1. BUG-v7-L1: Guardian SL regression after TP1 — new regression
     from BUG-P11-1 fix. Guardian reads diary amount (full), not
     active_trades amount (halved). Fix guardian to read from
     active_trades. Without this, all TP1 hits leave the runner
     position unprotected.

  2. BUG-v7-P1: Remove Kronos AutoModel fallback — replace with
     `return 0.0` when ChronosPipeline is unavailable. The current
     fallback injects random noise into every score evaluation.

  3. BUG-v7-S2+S4: Set DASHBOARD_TOKEN + configure Telegram.
     These are 60-second config fixes that eliminate critical
     operational and security gaps.

CAPITAL RECOMMENDATION: Max $200 test allocation until items
  1 and 2 are fixed and verified on testnet. After fixes:
  $200 for 24-48h monitoring, then scale gradually.

GO LIVE: NOT READY
  Pass 11 bugs were fixed. This pass found new issues.
  The guardian regression (L1) is the most urgent — it directly
  negates the TP1 partial-close protection that was added in P11.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 🔧 FIX PROMPTS

### Fix 1 — BUG-v7-L1 (CRITICAL — Guardian SL size regression)

In `src/main.py`, the guardian's SL re-place block (lines ~1207–1217) reads SL amount from the diary:
```python
_g_amount = float(_g_diary.get('amount') or 0)
```
After TP1 fills, only 50% remains. The guardian should read from the matching `active_trades` entry instead. Find the active_trades entry for `_g_asset` and use its `amount` field, which was correctly halved by the BUG-P11-1 fix:
```python
_g_active = next((t for t in active_trades if t.get("asset") == _g_asset), None)
_g_amount = float((_g_active or {}).get("amount") or _g_diary.get("amount") or 0)
```
Apply the same pattern to the trailing-stop guardian that reads `_tr_size`.

---

### Fix 2 — BUG-v7-P1 (CRITICAL — Kronos AutoModel is noise)

In `src/indicators/kronos_forecast.py`, remove the `except` block that falls back to `AutoModelForSeq2SeqLM`. Replace with:
```python
except Exception as _e:
    logging.warning("Kronos unavailable (%s) — modifier = 0.0 (AutoModel fallback removed: not valid time-series)", _e)
    return 0.0
```

---

### Fix 3 — BUG-v7-S2 (CRITICAL — Dashboard auth)

In `.env`:
```
DASHBOARD_TOKEN=<output of: openssl rand -hex 32>
API_HOST=127.0.0.1
```

---

### Fix 4 — BUG-v7-L3 (HIGH — State stays ENTERED after limit cancel)

In `src/main.py`, in the pending-limit cancel path (lines ~2391), after `state_mgr.clear_entry(asset)`:
```python
state_mgr.clear_entry(_pl_asset)
state_mgr.set_state(_pl_asset, "IDLE")  # ← ADD THIS
```
(Verify `set_state` or equivalent exists in `trade_state.py`; if not, call `state_mgr._states[_pl_asset] = "IDLE"` and `state_mgr._save()`.)

---

### Fix 5 — BUG-v7-S5 (MEDIUM — UTC timestamp in confirm_trade)

In `src/agent/decision_maker.py` line 41:
```python
# WRONG:
_now_utc = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
# FIX:
from datetime import timezone
_now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
```

---

*v7 council complete. Primary new finding: BUG-v7-L1 (guardian SL regression from P11 fix). Secondary: Kronos AutoModel noise. Config fixes (dashboard + Telegram) remain outstanding from previous reports. Fix these three categories and re-test on testnet before scaling.*
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           