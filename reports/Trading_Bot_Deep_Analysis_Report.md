# Hyperliquid AI Trading Bot — Comprehensive Technical Analysis Report
**Generated:** 2026-04-28  
**Analyst:** Claude (Cowork)  
**Codebase:** `hyperliquid-trading-agent-master/`  
**Verdict Summary:** Architecturally solid, operationally dangerous at current capital scale. The bot will not generate meaningful profit in its current configuration — not because of bad code, but because of fatal math at the account size and model tier being used.

---

> ⚠️ **CRITICAL SECURITY ALERT — Read First**
> Your live `.env` file contains **plaintext private keys and API keys committed to disk**:
> - `ANTHROPIC_API_KEY=sk-ant-api03-4MyGH...` (live Anthropic key)
> - `HYPERLIQUID_PRIVATE_KEY=0xe090ecb7...` (live EVM private key — can drain your wallet)
> - `HYPERLIQUID_VAULT_ADDRESS=0x467B95...` (your main wallet address)
>
> If this project folder is ever synced to GitHub, shared, or accessed by anyone else, your funds and API access can be stolen immediately. Rotate these keys now if this repo has ever been pushed publicly.

---

## 1. ARCHITECTURE & STRUCTURE

### Complete File Tree with Purpose

```
hyperliquid-trading-agent-master/
│
├── src/
│   ├── main.py                   ← Entry point + full trading loop (1,082 lines)
│   ├── config_loader.py          ← .env parser → CONFIG dict (113 lines)
│   ├── risk_manager.py           ← 8-guard safety layer before every execution (373 lines)
│   ├── strategy.py               ← Pre-trade filters: spread + 15m/5m entry confirmation (56 lines)
│   ├── trade_state.py            ← Per-asset state machine (IDLE/ENTERED/COOLDOWN) (119 lines)
│   │
│   ├── agent/
│   │   └── decision_maker.py     ← Claude API orchestration + tool use loop (573 lines)
│   │
│   ├── trading/
│   │   └── hyperliquid_api.py    ← All exchange I/O: orders, candles, positions (702 lines)
│   │
│   ├── indicators/
│   │   └── local_indicators.py   ← Pure-Python indicator library (408 lines)
│   │
│   └── utils/
│       ├── formatting.py         ← Number rounding helpers (16 lines)
│       └── prompt_utils.py       ← JSON serialization for LLM context (43 lines)
│
├── dashboard.html                ← Browser-based monitoring dashboard (standalone HTML)
├── .env                          ← Live secrets + config (⚠ contains real private keys)
├── pyproject.toml                ← Poetry deps (Python ≥3.12, hyperliquid-sdk, anthropic, aiohttp)
├── Dockerfile                    ← Container build spec
├── STRATEGY.md                   ← Strategy documentation
├── DEVNOTES.md                   ← Private operational reference
├── decisions.jsonl               ← Runtime: per-cycle Claude decisions + account snapshots
├── llm_requests.log              ← Runtime: token usage + cost per Claude call
├── prompts.log                   ← Runtime: full JSON payloads sent to Claude
│
└── docs/
    └── ARCHITECTURE.md           ← Architecture diagram text
```

### How Components Connect

```
CLI / .env
    ↓
CONFIG (config_loader.py)
    ↓
main.py::run_loop()  ────────────────────────────────┐
    │                                                 │
    ├──► HyperliquidAPI.get_user_state()              │ aiohttp web server
    ├──► RiskManager.check_losing_positions()         │ (parallel)
    ├──► HyperliquidAPI.get_candles() × 4 intervals  │ GET /diary
    ├──► local_indicators.compute_all()               │ GET /logs
    ├──► strategy.market_filter()                     │ GET /live
    ├──► strategy.entry_confirmed()                   │ GET /fills
    ├──► TradingAgent.decide_trade()  ──► Claude API │ GET /
    ├──► RiskManager.validate_trade()                 │
    ├──► HyperliquidAPI.place_buy/sell_order()        │
    ├──► HyperliquidAPI.place_take_profit()           │
    ├──► HyperliquidAPI.place_stop_loss()             │
    ├──► TradeStateMachine.record_entry()             │
    └──► diary.jsonl, decisions.jsonl ◄───────────────┘
```

### Data Flow: Start to Finish

```
TRIGGER: asyncio.sleep(interval) expires (default 1h)
    ↓
ACCOUNT FETCH:  get_user_state() → balance, positions, PnL, open orders, fills
    ↓
RISK PRE-SCREEN: check_losing_positions() → force-close if > MAX_LOSS_PER_POSITION_PCT
TIMEOUT CHECK:  if ENTERED trade open > 12h → market_close()
GUARDIAN:       if TP/SL orders missing for ENTERED position → re-place them
    ↓
DATA GATHER (per asset, in parallel):
    get_current_price() | get_open_interest() | get_funding_rate()
    get_candles(1h,60)  | get_candles(4h,60)  | get_candles(15m,30) | get_candles(5m,20)
    compute_all() → EMA20/50, RSI14, MACD, ATR14, BBands, ADX, OBV, VWAP, StochRSI
    compute trend labels: trend_4h, trend_1h, momentum_4h
    spread from cached impactPxs
    ↓
LLM CALL: single Claude call with ALL assets in one JSON payload
    ← returns { reasoning, trade_decisions: [{ asset, action, allocation_usd,
               order_type, limit_price, tp_price, sl_price, exit_plan, rationale }] }
    ↓
PER-DECISION FILTERS:
    [STATE GATE]    skip if ENTERED or COOLDOWN
    [TREND GUARD]   skip if trend_4h = UNKNOWN
    [INVERSION CHECK] raise if BULLISH+sell or BEARISH+buy
    [MARKET FILTER] skip if ATR > 5% or spread > 0.15%
    [ENTRY CONFIRM] skip if 15m/5m not aligned with direction
    ↓
RISK VALIDATION (validate_trade()):
    1. Daily drawdown circuit breaker
    2. Balance reserve
    3. Position size cap (cap, not reject)
    4. Total exposure
    5. Leverage
    6. Concurrent position count
    7. Mandatory SL enforcement
    8. Fee-aware TP minimum
    ↓
EXECUTION:
    place_buy/sell_order() → market or limit
    poll fills 3× at 1s intervals → filled_qty
    place_take_profit() | place_stop_loss()
    state_mgr.record_entry()
    ↓
LOGGING:
    diary.jsonl (per-trade)
    decisions.jsonl (per-cycle summary + Claude reasoning)
    llm_requests.log (token usage, cost)
    ↓
asyncio.sleep(interval) → next cycle
```

---

## 2. AI DECISION ENGINE

### Model in Use

- **Primary model:** `claude-haiku-4-5-20251001` (hardcoded default, configurable)
- **Sanitize model:** `claude-haiku-4-5-20251001` (same cheap model used to normalize malformed JSON)
- **Max tokens:** 1,200 (too low for complex multi-asset analysis — see Weakness section)
- **Tool calling:** **DISABLED** (`ENABLE_TOOL_CALLING=false` in .env)
- **Extended thinking:** **DISABLED** (`THINKING_ENABLED=false`)
- **System prompt caching:** Enabled via `cache_control: ephemeral` — saves ~80% on system prompt tokens from call 2 onward

### What Data Is Fed to Claude

Claude receives one JSON payload per cycle containing:

```
invocation:     minutes_since_start, current_time, cycle count
account:        balance, account_value, Sharpe ratio, positions (with PnL),
                active_trades, open_orders, recent_diary (last 10 entries),
                recent_fills (last 20)
risk_limits:    all 8 risk parameters + circuit breaker status + fee rates
market_data:    per asset:
    current_price, trend_4h, trend_1h, momentum_4h (pre-computed labels)
    intraday_1h: EMA20, EMA50, MACD, histogram, signal, RSI14 + 3-bar series
    long_term_4h: EMA20, EMA50, ATR14, MACD, histogram, signal, RSI14 + 3-bar series
    open_interest, funding_rate, funding_annualized_pct
    recent_mid_prices (last 10 price history samples from bot's in-memory buffer)
    setup_15m: EMA20, MACD histogram, RSI14, near_ema flag
    trigger_5m: MACD histogram, RSI14, candle_bullish flag
    spread_pct
```

**Observed payload size from logs:** ~4,373–4,478 input tokens for 2 assets, costing ~$0.007–$0.008 per call. This is appropriate.

### How Decisions Are Parsed and Executed

1. Claude is called in a loop of up to 6 iterations (to handle tool use rounds)
2. JSON is stripped of markdown fences and parsed directly
3. If JSON is malformed, a second Haiku call (`_sanitize_output`) attempts to normalize it
4. Normalized output: `{ reasoning: str, trade_decisions: [{ asset, action, ... }] }`
5. If parse fails completely → all assets get `action: "hold"` as a safe fallback
6. If the LLM output is all-hold with "parse error" in rationale → one retry with a stricter instruction prefix

### Weaknesses in the Prompt Logic

**Weakness 1 — Model tier mismatch.** Claude Haiku is a lightweight model optimized for speed and cost, not multi-step quantitative reasoning. It is being asked to simultaneously analyze 2 assets, weigh 4 timeframes, evaluate 9 indicators per asset, enforce hysteresis rules, respect cooldowns from previous cycles, and account for funding costs — all in a single inference with max_tokens=1200. Real observed reasoning (from decisions.jsonl) shows Claude produces coherent surface-level summaries but does not actually enforce its own cooldown logic across cycles (it has no persistent memory — every call is stateless).

**Weakness 2 — Stateless Claude.** The system prompt instructs Claude to "honor your own cooldowns" and "respect prior exit_plans." This cannot work. Claude has no memory between API calls. The bot does pass `active_trades` and `recent_diary` in context, but Claude Haiku at 1,200 max_tokens doesn't have sufficient context window use to reliably track multi-cycle commitments. The `TradeStateMachine` enforces hard ENTERED/COOLDOWN gates correctly in Python, but the LLM's self-described cooldowns in `exit_plan` strings are purely advisory and are never programmatically enforced by the code.

**Weakness 3 — Output format fragility.** Despite the sanitization fallback, the primary output is plain JSON requested from a model that routinely wraps responses in markdown. The `_sanitize_output` function adds another API call cost and latency when triggered, and itself can fail if Haiku returns invalid JSON in the second pass.

**Weakness 4 — Tool calling disabled.** The `fetch_indicator` tool is well-designed, but `ENABLE_TOOL_CALLING=false` means Claude never has access to additional indicators. Every call is a single-pass guess from pre-computed data only.

**Weakness 5 — Haiku `max_tokens=1200` is borderline.** The observed output from the log was 943–1,061 tokens, meaning the model is routinely generating near the cap. This risks truncated responses, causing JSON parse failures and unnecessary sanitization calls.

**Weakness 6 — No position closing logic from Claude.** The system prompt tells Claude to "close winners at high-quality opportunities," but there is no mechanism for Claude to close an existing position. `active_trades` is passed as context, but to actually close a position, Claude would need to output a direction-reversed trade for the same asset. The code has no explicit `close` action type — it only supports `buy`, `sell`, or `hold`. Closing must happen through TP/SL triggers or the force-close mechanisms, not through Claude deciding to exit.

---

## 3. TRADING LOGIC

### Entry and Exit Conditions

**Entry — Multi-layer confirmation required (all must pass):**
1. State machine must be IDLE (not ENTERED or COOLDOWN)
2. `trend_4h` must not be UNKNOWN (must have sufficient candle history)
3. Trend direction must match action (BULLISH+buy or BEARISH+sell — hard assertion)
4. `market_filter()`: ATR(14) on 4h < 5% of price AND spread < 0.15%
5. `entry_confirmed()`: price within 0.3% of 15m EMA20 AND (bullish 5m candle OR positive 5m MACD histogram) for buys; inverse for sells
6. Risk manager: 8 sequential checks (circuit breaker, reserve, size, exposure, leverage, concurrent count, SL enforcement, TP minimum)

**Exit conditions:**
- TP trigger order fills automatically on exchange (primary exit — profit)
- SL trigger order fills automatically on exchange (primary exit — controlled loss)
- Force-close: `check_losing_positions()` triggers if unrealized loss ≥ `MAX_LOSS_PER_POSITION_PCT` (8%) — fires every cycle before Claude is even called
- Time-based exit: `is_trade_expired()` triggers if trade open > `MAX_TRADE_HOURS` (12h) with no TP hit
- Guardian re-places missing TP/SL orders every cycle if exchange dropped them
- Claude reversals: theoretically possible but not explicitly supported by the `close` action type

### Position Sizing Logic

The formula is entirely LLM-driven, then capped:

```
Claude proposes → allocation_usd (free-form)
RiskManager caps → min(allocation_usd, account_value × MAX_POSITION_PCT / 100)
                   minimum: $11 (Hyperliquid minimum order)
actual amount   → allocation_usd / current_price (in asset units)
amount          → round_size(asset, amount)  (rounded to szDecimals)
```

**At current account size ($101.92) with MAX_POSITION_PCT=15%:**
- Max allocation per trade: $101.92 × 15% = **$15.29**
- At 3x leverage (MAX_LEVERAGE=3): **$45.87 notional** maximum
- Observed fills: BTC at 0.00014–0.00015 BTC = ~$11–12 notional (minimum order)
- Observed fills: ETH at 0.0048 ETH = ~$11.09 notional (minimum order)

The bot is trading at literal minimum size. This is financially non-viable (see Section 4).

### Leverage Settings

- **`MAX_LEVERAGE=3`** (hard cap via `check_leverage()`: `alloc_usd / balance ≤ 3`)
- Note: The leverage check compares `allocation_usd / balance` — this is the allocation-to-balance ratio, which is NOT the same as true perpetual futures leverage. True leverage = notional / margin used. The check is mathematically conservative but also somewhat misaligned with how Hyperliquid reports leverage (the ETH position shows `"value": 20` in the fills log, suggesting cross-margin account with 20x default leverage, while the bot cap computes something very different).
- Observed position: ETH -0.0048 at 2310.7 with `leverage.value: 20` — the account is running at 20x cross leverage at the exchange level, but the bot's check didn't catch this because the formula `alloc_usd / balance` ≈ `11 / 101 = 0.11x` passes fine. The exchange's leverage setting and the bot's leverage check are measuring completely different things.

### Risk Management (Stop Loss / Take Profit / Drawdown)

**Stop Loss:**
- `MANDATORY_SL_PCT=3`: auto-set 3% from entry if Claude omits it
- Intended SL from STRATEGY.md: `entry - 1× ATR14` (but this is a guideline for Claude, not code-enforced)
- The risk manager auto-enforces a 3% SL — this is the actual hard floor

**Take Profit:**
- `enforce_take_profit()`: TP must be ≥ 0.27% from entry (= 3× round-trip taker fee)
- Intended TP from STRATEGY.md: `entry + 3× ATR14` (again, guideline for Claude, not enforced)
- On a $11 position at 0.27% TP: **$0.0297 gross profit** before fees

**Drawdown Limits:**
- Daily circuit breaker: `DAILY_LOSS_CIRCUIT_BREAKER_PCT=12%` (fires if account drops 12% from day's high)
- Max position loss: `MAX_LOSS_PER_POSITION_PCT=8%` (force-close at 8% position loss)
- Min balance reserve: `MIN_BALANCE_RESERVE_PCT=20%` (stop trading if balance < 20% of starting balance)
- Total exposure cap: `MAX_TOTAL_EXPOSURE_PCT=50%` (max 50% of account in open positions)
- Concurrent position limit: `MAX_CONCURRENT_POSITIONS=2`

### Assets Traded

Currently configured: **BTC and ETH only** (from `.env: ASSETS="BTC ETH"`).

The codebase supports any of Hyperliquid's 229+ perp markets including crypto (SOL, ARB, etc.), commodities (OIL, GOLD, SILVER), and indices (SPX) via HIP-3 dex prefix format (`xyz:GOLD`). The asset selection rationale in the code is: "whatever is in the ASSETS env variable." There is no asset-selection logic — the user manually decides.

---

## 4. PROFIT LEAKAGE POINTS — BRUTALLY HONEST MATH

### The Core Problem: The Math Doesn't Work at $100

This is the single most important finding in this report. **The bot cannot generate meaningful returns at current account size, regardless of how well the strategy works.**

**Concrete breakdown at current settings:**
```
Account value:       $101.92
Max position size:   $15.29 (15% of account)
Taker fee per fill:  0.045% of notional
Round-trip fee:      0.09% of notional

Scenario: $15 position, 3x leverage = $45 notional
Open fee:            $45 × 0.045% = $0.0203
Close fee:           $45 × 0.045% = $0.0203
Total fees:          $0.0405 per trade

Minimum TP (0.27%):  $45 × 0.27% = $0.1215 gross
Net profit at min TP: $0.1215 - $0.0405 = $0.081 per winning trade

Required win rate to break even (ignoring SL losses): >33%
SL loss at 3% (no leverage adjustment): $45 × 3% = $1.35

Win:Loss ratio at 1:3 (ATR-based):
  Win: $0.12 gross, $0.08 net
  Loss: $1.35 gross, $1.39 with fees
  Break-even win rate: 1.39 / (0.08 + 1.39) = 94.5%
```

**Even with a theoretically correct strategy, you need to win 94.5% of the time just to break even at these position sizes.** That is not achievable with any trading strategy.

The ATR-based 1:3 risk:reward from the docs assumes the TP is 3× ATR and SL is 1× ATR. But the risk manager overrides Claude's SL to a minimum 3% regardless of ATR. On BTC at $80,000, 1× ATR14 on 1h might be $400–600 (0.5–0.75%), meaning the risk manager is placing SLs much wider than ATR and TPs much narrower than 3× ATR. The actual RR ratio in practice is far worse than 1:3.

### Fee Analysis

From the runtime log:
- Claude API: ~$0.007–0.008 per cycle, $0.168–0.192/day at 1h intervals
- Monthly Claude cost: ~$5–6
- Observed trades: BTC at 0.00014 BTC (~$11) = $0.005 open fee + $0.005 close fee = **$0.01 round-trip**
- With 24 cycles/day and estimated 5% trade rate = 1.2 trades/day × $0.01 = **$0.012/day in exchange fees**
- Claude API cost ($0.17/day) is **14× bigger than the trading fee cost**

At this account size, the Anthropic API bill is the primary cost, not trading fees.

### Are Positions Held Too Long or Too Short?

- MAX_TRADE_HOURS=12: trades are force-closed after 12h if no TP hit. This is reasonable for 1h timeframe trading.
- The actual ETH position observed (entry 2310.7, SL 2348, TP 2297) had been open for at least 2 days based on fill timestamps (opened 2026-04-24, observed 2026-04-26). The 12h timeout should have fired but didn't in that cycle — possibly because the state machine was not in ENTERED state (the trade may have been entered before this code version was deployed).
- The cooldown of 3600s (1h) after position close means at most 1 trade per 2 hours per asset — appropriate.

### Is the Bot Over-Trading or Under-Trading?

The entry confirmation filters (15m/5m layer + spread filter + trend guard + state gate) are aggressive. Based on the system design, the expected hold rate is 60–80% per cycle. At 1h interval for 2 assets = 48 potential decisions/day, with 20–40% trade rate = 10–20 trade attempts/day. Of those, the risk manager blocks some. Actual fill rate is much lower.

The bot is **structurally under-trading** at current settings — which is intentional but means even fewer opportunities to generate the returns needed to be meaningful at $100.

### Is Position Sizing Too Small?

**Yes, critically.** The minimum viable position size to generate meaningful returns:
- At 0.27% TP (minimum): need at least $200 notional to earn $0.54 net per trade
- To make $100/month: need ~185 winning trades per month at $0.54/trade = 6 wins/day
- At $100 account with 15% position size and 3x leverage: max notional = $45
- At $45 notional and 0.27% TP: gross $0.12/trade, net $0.08/trade — need 1,250 wins/month to make $100
- **The math is not viable below $5,000 account minimum.** Realistically, to see meaningful returns, the account needs $10,000+ with current risk parameters, or $5,000+ with MAX_POSITION_PCT raised to 25–30% and MAX_LEVERAGE raised to 5x.

---

## 5. BOT CYCLE & TIMING

### How Often Does the Bot Run?

- Default: `INTERVAL=1h` — one full cycle per hour
- Configurable: 1m, 5m, 15m, 1h, 4h, 1d (any Hyperliquid candle interval)
- At 1h: 24 cycles/day, 168/week, 720/month

### How Long Does Each Cycle Take?

**Breakdown of cycle time:**
```
get_user_state():           ~0.2–0.5s (2 API calls: user_state + spot_user_state)
Force-close checks:         ~0.1s (synchronous math)
get_open_orders():          ~0.2s
Guardian TP/SL check:       ~0.1–0.3s (reads diary, checks orders)
get_recent_fills():         ~0.2s
Per asset (parallel):
  7 concurrent API calls:   ~0.5–1.5s total (asyncio.gather)
  compute_all() × 4:       ~0.05s (pure Python, fast)
LLM call (Haiku):           ~2–8s (network + inference)
Trade execution:
  place order:              ~0.5s
  fill polling (3× 1s):    ~3s
  place TP:                 ~0.5s
  place SL:                 ~0.5s
Logging:                    ~0.05s
asyncio.sleep(3600):        3600s
```

**Total active time per cycle (excluding sleep): ~7–15 seconds** for 2 assets. This is well within a 1h interval.

### Signal-to-Execution Delay

From Claude decision to order placement: ~3–8s (LLM latency) + ~0.5s (order placement) = **3.5–8.5 seconds total lag** from signal to open order.

For a 1h candle strategy, this lag is completely immaterial.

For a 5m candle strategy, an 8.5s lag represents 2.8% of the candle — could cause meaningful slippage on fast-moving markets, but still acceptable.

---

## 6. DATA & INDICATORS

### What Market Data Is Fetched

Per cycle per asset:
- Current mid-price (from `allMids` endpoint)
- Open interest (from cached `metaAndAssetCtxs`)
- Funding rate (from cached `metaAndAssetCtxs`)
- Candles: 1h (60 bars), 4h (60 bars), 15m (30 bars), 5m (20 bars)
- Bid/ask spread estimate from `impactPxs` in cached metadata (zero extra API calls)

**Not fetched:**
- Real-time order book (depth, bid/ask walls)
- Liquidation data
- Long/short ratio from open interest (only raw OI, not directional breakdown)
- Volume profile (VPOC, VWAP by session)
- Perpetual basis (spot vs perp price divergence)
- Hyperliquid's own mark price vs index price divergence
- Trader sentiment / social signals

### Technical Indicators Computed

All computed locally in pure Python from OHLCV candles — zero external API cost:

| Indicator | Implementation | Notes |
|-----------|---------------|-------|
| EMA(20), EMA(50) | Standard EMA formula | Correct implementation |
| RSI(14) | Wilder smoothing | Correct — uses Wilder's modified MA, not simple |
| MACD (12/26/9) | EMA difference + signal EMA | Correct |
| ATR(14) | True range Wilder average | Correct |
| Bollinger Bands (20, 2σ) | SMA + std dev | Correct (uses population variance, minor pedantry) |
| ADX(14) | Directional movement | Correct Wilder implementation |
| OBV | Volume accumulation | Correct |
| VWAP | Cumulative since bot start | **WRONG** — see below |
| Stochastic RSI | RSI → Stoch → smoothed | Correct but padding logic is fragile |

**VWAP Bug:** The VWAP implementation is **cumulative from the first candle in the fetched series** (not session-based). Every time the bot restarts or candles are re-fetched, VWAP resets. In live trading, VWAP is meaningful only when anchored to session open (midnight UTC or exchange open). The current implementation produces a meaningless rolling VWAP that drifts with the candle window. This value is included in the indicator suite (`compute_all()`) but not currently passed to Claude in the main payload — it's only available if tool calling fires `fetch_indicator('vwap', ...)`. Low impact but worth fixing.

**ADX is computed but never sent to Claude.** The ADX series is included in `compute_all()` but is not included in the `intraday_1h` or `long_term_4h` sections of the context payload in `main.py`. Claude never sees ADX values. This is a missed signal — ADX > 25 is a strong trend strength confirmation that would improve entry quality.

**Bollinger Bands and Stochastic RSI are also computed but not passed to Claude** in the main context. They are only available via the tool call mechanism (which is disabled).

### What Timeframes Are Used

| Timeframe | Bars Fetched | Usage |
|-----------|-------------|-------|
| 4h | 60 | Structural trend (EMA20/50, MACD, RSI) |
| 1h | 60 | Intraday direction (EMA20/50, MACD, RSI) |
| 15m | 30 | Entry setup (EMA20 proximity, MACD histogram) |
| 5m | 20 | Entry trigger (candle direction, MACD histogram) |

Missing: daily (1d) timeframe for macro bias. On a 4h timeframe, a "BULLISH" EMA cross may be a minor retracement within a larger downtrend. The daily trend is never checked.

### What Data Is Missing That Would Improve Decisions

- **EMA200** on 4h: The system prompt mentions EMA200 (e.g., "EMA20 > EMA50 > EMA200 = strong BULLISH") but EMA200 is never computed or sent. The candle history fetched (60 bars of 4h) is insufficient to compute a reliable EMA200 anyway — you'd need at least 200 bars.
- **Daily candle context**: Never fetched. No macro trend filter.
- **Long/short ratio**: OI is fetched as a number but the directional breakdown (how many contracts are long vs short) is not fetched. Hyperliquid provides this.
- **Volume confirmation**: The 15m/5m entry filter checks MACD and candle direction but never checks if the volume on the trigger candle is above average. Low-volume reversals are false signals.
- **Recent high/low levels**: No support/resistance detection. Claude cannot identify key structural levels unless it infers them from the price series, which Haiku is not reliable at doing.
- **Unrealized PnL as % of position**: Currently tracked, but not factored into entry decisions for the same asset.

---

## 7. EXCHANGE INTEGRATION (HYPERLIQUID)

### How Orders Are Placed

- **Market orders**: `exchange.market_open(asset, is_buy, amount, None, slippage=0.01)` — 1% slippage tolerance hardcoded
- **Limit orders**: `exchange.order(asset, is_buy, amount, limit_price, {"limit": {"tif": "Gtc"}})` — Good-til-canceled by default
- **TP trigger**: `exchange.order(asset, not_is_buy, amount, tp_price, {"trigger": {"triggerPx": tp_price, "isMarket": True, "tpsl": "tp"}}, reduce_only=True)`
- **SL trigger**: Same as TP with `"tpsl": "sl"`
- **Market close**: `exchange.market_close(asset, None, slippage=0.01)` — closes full position

All orders go through `_order_retry()` with idempotency guard (`_check_order_landed()`) to prevent duplicate orders on connection failures.

### How Positions Are Tracked

Two parallel tracking systems exist, which is a source of potential inconsistency:

1. **Exchange-side** (authoritative): `get_user_state()` returns `assetPositions` with real positions. The bot queries this every cycle.
2. **Bot-side** (`active_trades` list in memory + `state.json` on disk): Maintained by the bot. Used for guardian TP/SL re-placement and timeout tracking.

These can diverge. The reconciler in `run_loop()` detects when an asset appears in `active_trades` but has no position and no orders on the exchange, and removes the stale record. However, the reverse case — position on exchange but not in `active_trades` (e.g., manually placed trade, or position from before bot restart) — is handled partially by the guardian and state machine, but not fully.

### How Balance and Equity Are Calculated

The balance computation has a known complexity: Hyperliquid uses a "unified account" model where:
- `perps_value` = `marginSummary.accountValue` (includes unrealized PnL on perps)
- `spot_usdc` = USDC from `spot_user_state` (includes idle capital + isolated margin)

The code uses `spot_usdc if spot_usdc > 0 else perps_value` as the `total_value`. This is a simplification that can lead to double-counting in edge cases (e.g., isolated margin positions where USDC is locked in both `spot_usdc` and counted in `perps_value`).

The comment in the code acknowledges this: "spot_user_state 'total' includes USDC locked in isolated positions, so perps_value (isolated margin equity) is already counted inside spot_usdc." — but the code then just uses one or the other, not the correct combination.

**From the runtime log:** Cycle 1 shows `balance: 0.0, account_value: 1.05` — the balance was showing zero while there was a live $1.05 position. This is exactly the bug described above.

### API Errors and Rate Limiting

**Rate limiting:** No explicit rate limit handling. The exponential backoff (0.5s, 1s, 2s) in `_retry()` is for connection failures, not HTTP 429 rate limit responses. Hyperliquid's public API has rate limits, and if the bot were running many assets rapidly, it could hit them without specific handling.

**Connection error handling:** Good — `WebSocketConnectionClosedException`, `aiohttp.ClientError`, `ConnectionError`, `TimeoutError`, `socket.timeout` are all caught and trigger client reset + retry.

**Timeout on Claude API:** 45s timeout hardcoded. If Claude is slow, this blocks the entire trading loop for up to 45s. Since this runs in `asyncio.to_thread()`, the event loop is not blocked, but the cycle is delayed.

**Missing SDK version pinning risk:** `hyperliquid-python-sdk (>=0.20.0,<0.21.0)` is locked to a minor version range. The SDK is actively developed and breaking changes are possible within this range. The `user_fills` vs `fills` method check (`hasattr(self.info, 'user_fills')`) is already a symptom of SDK version instability.

---

## 8. LOGGING & DIARY SYSTEM

### What Gets Logged Per Cycle

**`diary.jsonl`** (per-trade events):
```json
{
  "timestamp", "asset", "action",
  "order_type", "limit_price",
  "allocation_usd", "amount", "filled_qty", "requested_qty",
  "entry_price", "tp_price", "tp_oid", "sl_price", "sl_oid",
  "exit_plan", "rationale", "order_result",
  "opened_at", "filled": bool
}
```
Also logs: `risk_blocked`, `risk_force_close`, `reconcile_close` events.

**`decisions.jsonl`** (per-cycle summary):
```json
{
  "timestamp", "cycle", "reasoning" (truncated to 2000 chars),
  "decisions" (asset + action + allocation + rationale),
  "account_value", "balance", "perps_value", "spot_usdc", "withdrawable",
  "positions", "open_orders", "recent_fills", "positions_count"
}
```

**`llm_requests.log`** (per-LLM-call):
- Model name, messages count, last message preview (500 chars)
- Stop reason, token usage (input/output/cache_read), cost estimate

**`prompts.log`** (per-cycle full context):
- Full JSON payload sent to Claude, timestamped
- Rotated at 10MB (renamed to `.old`)

### What Is NOT Being Logged That Should Be

**Missing from diary.jsonl:**
- **Exit price and exit timestamp** — when a TP/SL fires, there is no diary entry for the close. The bot only logs the entry. You cannot reconstruct P&L from diary alone.
- **Actual realized P&L per trade** — no closed PnL is written to diary. The `/fills` endpoint has this data but it's not captured in the diary.
- **Fee amounts paid** — no fee tracking per trade.
- **Reason for every HOLD decision** — only buy/sell/block events are logged; HOLD decisions that don't result in any action are only in `decisions.jsonl` (per-cycle), not `diary.jsonl`.
- **Position close events from TP/SL** — when a trigger order fires, the bot has no hook to detect it and log the outcome. The reconciler detects the position is gone, but doesn't log which exit mechanism fired.
- **Guardian re-placement events** — these are logged to `add_event()` (application log) but not to diary.jsonl.
- **Cumulative win rate and P&L** — no running total maintained.

**Missing from decisions.jsonl:**
- **Actual trade outcome** — was the trade profitable? No feedback loop.
- **Indicator values that drove the decision** — Claude's reasoning text is logged but the raw indicator numbers aren't.

**Sharpe Ratio Calculation is Broken:**
```python
def calculate_sharpe(returns):
    vals = [r.get('pnl', 0) if 'pnl' in r else 0 for r in returns]
```
The `trade_log` list only ever contains entries from current session (not persisted). Each entry is added at trade open (not close), and contains no `pnl` key. The dictionary added at line 827 is `{"type", "price", "amount", "exit_plan", "filled"}` — no `pnl` field. So `vals` is always a list of zeros, and Sharpe ratio is always 0.0. This is displayed on the dashboard and is meaningless.

---

## 9. WEAKNESSES & BOTTLENECKS — EVERY IDENTIFIED ISSUE

### Fatal Issues (Block Profitability)

**F1 — Account size too small.** At $100, no mechanical advantage exists. The minimum viable account for this bot's parameters is $5,000–$10,000. This is not a code issue — it's a capital sizing issue.

**F2 — Model too weak for the task.** Claude Haiku cannot reliably perform multi-step quantitative reasoning across 2 assets, 4 timeframes, 9 indicators, and multi-cycle position tracking in a single stateless call. The model is fine for simple classification but this system expects expert-level trading analysis.

**F3 — No P&L feedback loop.** The bot has no way to know if its trades are profitable. Trade outcomes (fill prices, realized PnL from TP/SL) are not logged back to the diary. There is no mechanism to detect "this strategy is losing money" and adapt.

**F4 — Broken Sharpe ratio.** The dashboard displays Sharpe = 0.0 always. The calculation has a bug that makes it always return zero. This is the only "performance metric" visible to the operator.

### Significant Issues (Degrade Performance)

**S1 — VWAP is computed wrong.** Cumulative since first candle, not session-based. A VWAP support/resistance concept doesn't apply to a running cumulative average.

**S2 — ADX, Bollinger Bands, Stochastic RSI are computed but not sent to Claude.** Three valuable indicators are computed every cycle and thrown away. Only EMA20, EMA50, MACD histogram, RSI14, and ATR14 are passed to Claude.

**S3 — No EMA200 or daily timeframe.** The prompt references EMA200 as a filter but it's never computed (insufficient candle history would be fetched anyway — only 60 bars of 4h are fetched).

**S4 — Leverage check is measuring the wrong thing.** `alloc_usd / balance` is not perpetual futures leverage. The exchange uses cross-margin with default 20x, which the bot's check doesn't control.

**S5 — Account balance calculation can double-count.** The `spot_usdc if spot_usdc > 0 else perps_value` logic is a simplification that can show incorrect total_value in unified accounts with both spot USDC and active perp positions.

**S6 — No volume confirmation in entry filter.** The 15m/5m entry filter checks price proximity to EMA20 and MACD histogram direction but not volume. A low-volume reversal candle is a false signal.

**S7 — Claude's "cooldowns" are advisory only.** The prompt instructs Claude to "honor your own cooldowns from previous cycles" but Claude has no memory between calls. Cooldowns are only enforced by the Python state machine, not by Claude's reasoning.

**S8 — Limit orders have no fill-confirmation TP/SL placement.** When a limit order is placed (not filled yet), TP/SL are deferred to the guardian. The guardian places them the next cycle — up to 1h later if the limit fills quickly. A position can run naked for up to 1h with no SL.

**S9 — Position close events generate no diary entry.** When TP or SL fires on the exchange, the bot detects the position is gone via reconciliation, logs a `reconcile_close` event, but doesn't capture the exit price, PnL, or which mechanism fired (TP vs SL vs stop). P&L reconstruction is impossible from diaries alone.

### Minor Issues (Polish and Robustness)

**M1 — `MAX_TOKENS=1200` risks truncated JSON.** Observed output is 943–1,061 tokens. Two more assets would routinely exceed the cap.

**M2 — Prompts.log rotation.** The log rotates at 10MB by renaming to `.old` — only one generation of backup. A second rotation will delete the `.old` file. No compression, no multi-generation rotation.

**M3 — `active_trades` is in-memory only.** Bot restart loses the `active_trades` list. The state machine (ENTERED/COOLDOWN) persists via `state.json`, but the detailed trade info (entry price, amounts, TP/SL OIDs) is lost. The guardian will attempt to reconstruct from `diary.jsonl`, but if the diary is also cleared, the guardian cannot function.

**M4 — `handle_logs` allows arbitrary file path access.** The `/logs?path=...` endpoint accepts a `path` parameter without sanitization. An attacker who can reach port 3000 can read any file on the server: `GET /logs?path=/etc/passwd`.

**M5 — Port 3000 is bound to `0.0.0.0` by default.** Anyone who can reach the server's IP can hit the dashboard and log endpoints. Combined with M4, this is a serious security issue on a cloud server.

**M6 — `web3` is listed as a dependency but never used.** The pyproject.toml includes `web3 (>=7.14.0,<8.0.0)` — a heavy Ethereum library. It's not imported anywhere in the codebase. Dead dependency that slows installs.

**M7 — `ENABLE_CLAUDE_COMMENTARY` and `MIN_TRADE_SCORE` in .env are dead.** These variables are referenced in `.env` comments and DEVNOTES but have no corresponding logic in any Python file (there is no commentary scoring system in the code). They're left over from an earlier design.

**M8 — Inversion assertion raises an uncaught exception mid-loop.** `raise ValueError(f"INVERSION BUG DETECTED...")` will propagate up to the per-asset `try/except` block in the execution loop, logging a traceback and skipping that trade — correct behavior. But if this fires in production, a position may be open with no TP/SL (if the inversion was on a re-entry attempt). The guardian covers this next cycle, but there's a 1-cycle gap.

**M9 — No graceful shutdown.** There is no SIGTERM handler. If the bot is killed mid-order-placement (e.g., systemd stop during `place_buy_order()`), the position may exist on the exchange with no TP/SL and no record in `active_trades`. The guardian will catch this on next start if `state.json` shows ENTERED, but the state machine may not reflect the correct state if the kill happened before `record_entry()`.

**M10 — Dashboard HTML file access reads arbitrary text.** The `/` endpoint serves `dashboard.html` via `read_text()`. If `dashboard.html` is replaced with a malicious file, it's served directly.

---

## 10. RECOMMENDATIONS

### Priority 1 — Capital (Prerequisite to Everything Else)

The bot's logic is structurally sound. The problem is not the code. **Fund the account to at least $5,000** before running this bot in production. With $5,000:
- Max position ($750) at 3x leverage = $2,250 notional
- 0.27% TP = $6.08 gross, $4.05 net per winning trade
- $100/month requires ~25 winning trades — achievable at 1–2 trades per week

### Priority 2 — Upgrade the Model

Switch `LLM_MODEL=claude-sonnet-4-6` (or at minimum `claude-sonnet-4-5`). Increase `MAX_TOKENS=2500`. The cost increase is justified: Sonnet is ~5x more expensive than Haiku but has dramatically better multi-step reasoning. For 24 calls/day on Sonnet at current token counts: ~$0.15/day vs $0.03/day on Haiku — a $3.60/month increase. At $5,000 capital, this is trivial.

Alternatively, keep Haiku but enable extended thinking (`THINKING_ENABLED=true`, `THINKING_BUDGET_TOKENS=5000`) for significantly better reasoning at lower cost than Sonnet.

### Priority 3 — Fix P&L Tracking

Add a trade close logging mechanism:
- When `reconcile_close` fires, fetch the fill that closed the position from `get_recent_fills()` and capture: exit price, realized PnL, which exit mechanism fired (TP/SL/timeout/force-close)
- Add a `cumulative_pnl`, `win_count`, `loss_count`, `total_fees` tracker persisted to a `stats.json` file
- Fix the Sharpe ratio calculation to use actual realized trade returns, not the always-empty `pnl` field

### Priority 4 — Add Missing Context to Claude

Pass these to Claude in the market context payload:
- **ADX value** (from already-computed `compute_all()` — zero additional cost): ADX > 25 confirms a strong trend, ADX < 20 suggests ranging — critical filter
- **Bollinger Band width** as a volatility proxy
- **Long/short ratio** from Hyperliquid OI data
- **Daily candle trend** (1 additional API call per asset — fetch 50 daily bars): EMA20/50 on daily for macro bias

Fetch 200+ bars of 4h data (increase from 60 to 250) to enable proper EMA200 computation.

### Priority 5 — Enable Tool Calling

Set `ENABLE_TOOL_CALLING=true`. The `fetch_indicator` tool is well-implemented and gives Claude the ability to pull additional data before committing to a decision. The cost is one extra API round-trip when Claude uses it — negligible versus the benefit of better decisions.

### Priority 6 — Fix VWAP

Replace the cumulative VWAP with a session-anchored VWAP that resets at UTC midnight (or at each candle window boundary). Use the session start candle's time as the anchor:

```python
def vwap_anchored(candles, session_start_hour=0):
    # Reset cumulative when hour == session_start_hour
```

### Priority 7 — Position Close Logging

Add an explicit close detection hook:
```python
# After reconcile, check recent fills for the closed position
# Log exit_price, realized_pnl, exit_type (tp/sl/force/timeout)
```

### Priority 8 — Security Fixes

1. **Rotate all secrets in .env immediately** if this repo was ever pushed to GitHub
2. **Restrict API server to localhost**: `API_HOST=127.0.0.1` — access via SSH tunnel
3. **Sanitize the `path` parameter** in `handle_logs()` to prevent directory traversal:
   ```python
   allowed_files = {'llm_requests.log', 'prompts.log', 'diary.jsonl', 'decisions.jsonl'}
   if os.path.basename(path) not in allowed_files:
       return web.Response(text="Forbidden", status=403)
   ```
4. **Remove `web3` from dependencies** (unused, slow install)

### Priority 9 — Volume Confirmation in Entry Filter

Add to `entry_confirmed()`:
```python
# Require above-average volume on trigger candle
avg_vol_5m = sum(c['volume'] for c in candles_5m[:-1]) / max(len(candles_5m)-1, 1)
vol_ok = candles_5m[-1]['volume'] > avg_vol_5m * 0.8
```

### Priority 10 — Better Position Sizing Formula

Replace flat USD allocation with ATR-based position sizing:
```
risk_per_trade = account_value × 0.01  (1% risk per trade)
sl_distance    = ATR14 × 1.0           (1× ATR stop)
position_size  = risk_per_trade / sl_distance  (in asset units)
allocation_usd = position_size × current_price
```

This produces positions that are sized to risk exactly 1% of account per trade regardless of volatility — the correct approach for a systematic strategy. Currently the bot sizes by % of account without accounting for the actual SL distance, meaning high-volatility assets risk more per trade than intended.

---

## Summary Scorecard

| Category | Score | Notes |
|----------|-------|-------|
| Code architecture | 8/10 | Clean, modular, well-commented |
| Risk management | 7/10 | Solid 8-guard system; leverage check formula is wrong |
| Exchange integration | 8/10 | Retry logic, idempotency, guardian are production-grade |
| AI integration | 5/10 | Haiku too weak; stateless claude can't honor its own cooldowns |
| Data pipeline | 6/10 | Good indicators; ADX/BBands/StochRSI unused; VWAP wrong; no daily TF |
| Logging/observability | 4/10 | No exit events, no realized PnL, Sharpe always 0 |
| Security | 3/10 | Plaintext keys in .env; open API server; path traversal in /logs |
| Profitability at current settings | 1/10 | Mathematically impossible at $100 account |
| Profitability at $5k+ account | 5/10 | Strategy logic is reasonable; needs better model + volume filter |

---

*End of Report — All files reviewed: main.py, decision_maker.py, risk_manager.py, hyperliquid_api.py, local_indicators.py, config_loader.py, strategy.py, trade_state.py, formatting.py, prompt_utils.py, .env, pyproject.toml, STRATEGY.md, DEVNOTES.md, dashboard.html, decisions.jsonl, llm_requests.log*
