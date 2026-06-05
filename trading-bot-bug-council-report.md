# 🏛️ TRADING BOT BUG COUNCIL — FINAL REPORT
**Date:** 2026-06-02 | **Bot:** BB+StochRSI Mean Reversion Scalper | **Exchange:** Hyperliquid Perpetuals

---

## 📋 BOT SUMMARY
- **Strategy:** BB(20,2) + StochRSI(14,14,3,3) mean reversion on 5m candles
- **Assets:** BTC, ETH, SOL (perpetual futures)
- **Leverage:** 5× (set via API per asset)
- **Order types:** LIMIT entry, trigger TP/SL orders
- **Position model:** Single TP at BB midline, SL at 1.5×ATR, time-limit exit
- **Claude role:** Binary anomaly check on >3% price moves only
- **Risk controls FOUND:** Circuit breaker, balance reserve, position cap, leverage cap, exposure cap, concurrent positions cap, SL enforcement, TP minimum, ATR sizing, cooldown
- **Risk controls MISSING:** See below
- **Testnet:** Not configured / not present

---

## 💀 FUND-LOSS RISKS — Fix before ANY live trading

### BUG-C1 | `src/main.py` ~line 950-992 | **PARTIAL CLOSE LOGIC ACTIVE ON SINGLE-TP SYSTEM**
The TP1/TP2 partial close code (50%+50%) from the old architecture is still fully active in the reconciler, TP/SL guardian, and trailing stop code. The new bot sets `_tp1 = _tp` and `_tp2 = _tp` (both equal to BB midline), but the position management code still tracks two separate TP orders, halves position size at "TP1", and monitors a remaining 50% for "TP2". On a real fill, the reconciler will close 50% at BB midline and leave the other 50% OPEN with no TP order, running naked until time-limit. The trailing stop guardian then attempts to manage the remaining 50% as if it were still in a different strategy. **At 5× leverage, a naked 50% position drifting past the mean can turn a winning trade into a loss.**

### BUG-C2 | `src/main.py` ~line 1001, `config_loader.py` line 100 | **TIME_LIMIT_CANDLES NEVER ENFORCED**
`TIME_LIMIT_CANDLES=8` is in config and CONFIG dict, but `is_trade_expired()` uses `max_trade_hours=1` (60 minutes). The 40-minute exit is documented, intended, and configured — but the actual exit fires at 60 minutes. This 20-minute gap means a failed reversion holds 33% longer than intended. At 5× leverage with a trend running against the position, the extra 20 min can consume an additional 2-5% of margin before forced exit.

### BUG-C3 | `src/main.py` ~line 1794 | **SHORT LIMIT ORDER PLACED ABOVE MARKET (NEVER FILLS)**
For a SHORT signal, the limit sell is placed at `entry + 0.15%`. On Hyperliquid, a limit SELL above current price is a passive resting order that only fills if price rises to that level. But a SHORT mean reversion signal fires when price is at BB upper band — the expected move is DOWN, not further up. The sell limit at +0.15% above current price will never fill unless price moves further into overbought territory. Result: SHORT signals produce unfilled limit orders that expire after 5 minutes, wasting every SHORT opportunity. Only LONG trades actually execute. **This effectively disables 50% of the strategy's signals.**

---

## 🔴 CRITICAL — Fix before live trading

### BUG-H1 | `src/main.py` ~line 1626-1641 | **DEAD BTC CORRELATION FETCH**
When BTC is not in `--assets`, the bot fetches BTC 1h and 5m candles, computes EMA trend, and injects `trend_1h` into `market_sections`. But `trend_1h` is never read by the BB+StochRSI pipeline — `is_trend_regime_active()` reads `intraday_1h.adx`, not `trend_1h`. This is a leftover from the old `market_filter()`. With BTC in assets (current config: `BTC ETH SOL`), this code doesn't run. But it's misleading dead code with an incorrect comment claiming it's a correlation filter.

### BUG-H2 | `src/main.py` ~line 1368-1481 | **DEAD 4H/1H EMA TREND COMPUTATION**
Every cycle, the bot fetches 4H and 1H candles for all assets, computes EMA20/EMA50 on both timeframes, builds `asset_trends` and `asset_trends_1d` dicts. These are used ONLY for the `[TRADE] BTC action=hold | 4h_trend=BEARISH` log line. Zero trading decisions depend on them. This is 6+ unnecessary API calls per cycle (3 assets × 2 timeframes), increasing rate-limit exposure and adding ~2 seconds of latency per cycle.

### BUG-H3 | `src/main.py` ~line 1550-1626 | **MASSIVE DEAD MARKET CONTEXT PAYLOAD**
`market_sections` is built with a large context dict per asset including: trend_4h, trend_1h, macd_15m, near_ema, trigger_5m, spread_pct, candles_4h, candles_1h, daily_1d, and more. The BB+StochRSI pipeline only uses: `candles_5m`, `funding_rate`, `current_price`, `intraday_1h.adx`. All other fields are leftover from the old 5-factor Claude analysis and consume significant memory each cycle.

### BUG-H4 | `src/main.py` ~line 48-60 | **DASHBOARD TOKEN SECURITY WEAKENED**
The original code correctly blocked startup when `API_HOST=0.0.0.0` and `DASHBOARD_TOKEN` was empty. This was weakened to a warning only. The dashboard at `http://209.38.120.100:3000` now exposes live account balance, positions, TP/SL prices, and trade history to any public internet user. Set `DASHBOARD_TOKEN` to a strong secret, or change `API_HOST` back to `127.0.0.1` and use SSH tunneling.

---

## 🟠 HIGH — Fix before real money exposure

### BUG-M1 | `src/main.py` ~line 1738-1753 | **TP VALIDITY CHECK WRONG FOR SHORT**
For a SHORT: `if _tp >= _entry - _fee_buf: continue`. But `_tp` is `bb_mid` (BB midline). If the asset is at the BB upper band and the midline is below entry (as expected), `bb_mid < entry` so the check passes. But if somehow `bb_mid >= entry - fee_buf` (extremely narrow bands), the trade is skipped silently. This edge case is low-frequency but could cause the bot to miss valid trades in tight-band environments without logging why.

### BUG-M2 | `src/trade_state.py` ~line 110-130 | **CORRUPT STATE.JSON CAUSES sys.exit ON RESTART**
If `state.json` contains valid JSON with wrong types, the bot calls `sys.exit(1)`. This is intentional (prevents re-entry on open positions) but means a corrupt state file permanently bricks the bot until manually repaired. There's no auto-recovery or operator alert. On a server with no Telegram configured, the bot silently stops and doesn't restart.

### BUG-M3 | `src/risk_manager.py` ~line 420 | **MINIMUM ORDER BUMP OVERRIDES 1% ATR RULE**
When ATR sizing produces `< $11`, the bot bumps to `$11` minimum (Hyperliquid requirement). On a $100 account with a tight SL, ATR sizing might compute `$8` for 1% risk. The bump to $11 increases risk to 1.375% — silent rule violation. The code logs a warning but still executes. Over many trades on a small account, this systematically exceeds the 1% rule.

### BUG-M4 | `src/main.py` ~line 2290-2355 | **PENDING LIMIT CANCEL USES 4 MIN NOT 1 CANDLE**
The bot documentation says "cancel if unfilled after 1 candle (5 min)". The actual implementation uses `_pl_age_s >= 4 * 60` (4 minutes). Minor inconsistency but means a short-fired signal at minute 0 can still cancel at minute 4, leaving a 1-minute gap where a new signal could fire on the same asset before the old limit is confirmed cancelled.

### BUG-M5 | `src/main.py` ~line 1715-1716 | **DIRECTION CONVERSION BUG IF SIGNAL IS NEITHER LONG NOR SHORT**
`_direction_str = "buy" if _direction == "LONG" else "sell"`. If `_direction` is somehow not `"LONG"` (e.g. a bug returns `"long"` lowercase), this evaluates to `"sell"` silently. Should be an explicit dict lookup with a guard: `{"LONG": "buy", "SHORT": "sell"}.get(_direction)`.

---

## 🟡 EDGE CASES — Fix before scale

### BUG-E1 | `src/strategy.py` ~line 77 | **USES candles[-2] BUT FETCHES ONLY 100**
Uses the last CLOSED candle (index -2). If exactly 50 candles are returned (minimum warmup), `candles[-2]` is index 48 — the second-to-last warming candle. At exactly the warmup boundary this could use a warm-up candle. Low frequency but worth a guard: `if len(candles_5m) < min_candles + 1: return no_signal`.

### BUG-E2 | `src/main.py` ~line 1987-1995 | **PARTIAL FILL CANCEL ON ENTRY**
When a limit entry partially fills (e.g. 37 of 100 units), the bot cancels the remaining unfilled portion. This is correct. But it then continues to place TP/SL for only the filled portion. If the position is very small (e.g. $3 notional on ETH), Hyperliquid may reject the TP/SL order due to min-size. No guard for this case.

### BUG-E3 | `src/indicators/local_indicators.py` | **STOCHRSI K SMOOTHING BUG**
`stoch_rsi()` computes the raw %K as `(rsi - rsi_min) / (rsi_max - rsi_min + 1e-10) * 100`. Then it applies SMA smoothing to get smoothed %K. The `hook` detection in `strategy.py` compares `k_vals[-1] > k_vals[-2]` (turning up). But `k_vals` filters `[v for v in sr_data["k"] if v is not None]`. If smoothed K has trailing None values, the last 2 non-None values could span several bars (not just 1 bar apart), making the hook detection look at stale comparison. This is a subtle lookahead-adjacent bug.

### BUG-E4 | `src/main.py` ~line 2301-2355 | **LIMIT CANCEL READS STALE `active_trades` LIST**
The pending limit cancel iterates `active_trades` (the in-memory list). If `active_trades` wasn't updated between the outer loop and this check (e.g. during inner loop ticks), stale entries might cause a cancel attempt on an already-filled order. Minor — the cancel will just fail gracefully.

---

## 🔐 SECURITY

| Severity | Issue |
|----------|-------|
| **Critical** | Live private key (`0xe090...`) and Anthropic API key visible in `.env` in this Cowork session. Key should be rotated. |
| **High** | Dashboard (`http://209.38.120.100:3000`) publicly accessible without token — live balance and positions exposed |
| **High** | `DASHBOARD_TOKEN` security block weakened from fatal to warning — previous protection removed |
| **Medium** | Server runs bot as `root` (in `trading-bot.service`) — any code execution bug = full server compromise |
| **Medium** | No IP whitelist on Hyperliquid agent key |

---

## ⚡ PERFORMANCE

| Severity | Issue |
|----------|-------|
| **High** | 6+ unnecessary API calls/cycle for dead EMA trend data (4H+1H candles per asset, never used) |
| **High** | `market_sections` dict holds entire candle history per cycle — memory grows with assets |
| **Medium** | Outer loop and inner loop both run full BB+StochRSI pipeline — state gate prevents double-entry but doubles API calls |
| **Medium** | `asset_candles_5m` dict persists across cycles holding 100 candles × 3 assets in memory permanently |
| **Low** | `round_series()` called on large candle arrays for logging — unnecessary serialization overhead |

---

## 🛡️ MISSING RISK CONTROLS

| Control | Status |
|---------|--------|
| Testnet mode | ❌ Not implemented — bot goes live immediately |
| Kill switch (no-SSH) | ❌ No Telegram kill command, no web UI button, no file-flag check |
| State reconciliation on startup | ✅ Present (`active_trades.json` + state machine) |
| Instance lock | ✅ Present (socket mutex port 47293) |
| Daily trade cap | ✅ Present (40/day) |
| Per-asset cooldown | ✅ Present (15 min) |
| Circuit breaker | ✅ Present (12% drawdown) |

---

## 📊 MONITORING & OPERATIONAL GAPS

- No Telegram configured → circuit breaker fires silently, bot stops, no alert
- No daily P&L summary sent
- `signals.jsonl` written but no win-rate reporting tool
- `llm_requests.log` tracks Claude cost but no monthly cost alert threshold

---

## 📦 DEAD CODE TO REMOVE

| File | Dead Code | Safe to Remove |
|------|-----------|----------------|
| `src/main.py` | `asset_trends`, `asset_trends_1d`, 4H/1H EMA compute | ✅ |
| `src/main.py` | BTC correlation fetch block (lines 1626-1654) | ✅ |
| `src/main.py` | `market_filter()` comment references | ✅ |
| `src/main.py` | `"score": 8.0` placeholder field | ✅ |
| `src/indicators/kronos_forecast.py` | Entire file (never called) | ✅ |
| `src/main.py` | Trailing stop guardian code (TP1 = TP2 = same price, no partial close) | ⚠️ After fixing BUG-C1 |

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## COUNCIL VERDICT
**Risk Level: HIGH — 3 fund-loss bugs active**

**TOP 3 MUST-FIX (in order):**
1. **BUG-C3** — SHORT limit orders never fill (placed above market). Half your signals don't execute.
2. **BUG-C1** — Partial close logic still active. 50% of every trade runs naked after BB midline hit.
3. **BUG-C2** — TIME_LIMIT_CANDLES=8 (40 min) not enforced. Bot exits at 60 min, not 40.

**CAPITAL RECOMMENDATION:** $10–20 max until BUG-C1 and BUG-C3 fixed. These are not edge cases — they affect every SHORT trade and every TP fill.

**GO LIVE STATUS:** ⚠️ CONDITIONALLY READY for LONG-only testing with small capital. NOT READY for full operation until BUG-C1 + BUG-C3 resolved.
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
