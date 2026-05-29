# Hyperliquid AI Trading Agent

> **Architecture: CODE-FIRST HYBRID** — Technical analysis signals drive all decisions.
> Claude AI (Sonnet 4.6) is called as a deep market-analysis gate when signal score ≥ 7 AND
> multi-timeframe confluence is confirmed. All direction, sizing, TP, and SL are set by code.

A perpetual futures trading bot running on Hyperliquid with a 1h outer loop / 5m inner loop.
Code makes every trade decision. Claude performs full structured market analysis (5-factor,
25-point) and returns `VERDICT: APPROVE` or `VERDICT: REJECT` — anything else is REJECT (fail-closed).

---

## What It Does

**Outer loop (every 1h):**
1. Fetch account state (balance, open positions, recent fills)
2. Force-close any position at or beyond `MAX_LOSS_PER_POSITION_PCT` (8%) loss
3. Trailing stop guardian — advance SL to breakeven (+1×ATR), then trail behind (+1.5×ATR)
4. TP/SL guardian — re-place missing trigger orders for active positions
5. Time-based exit — force-close trades open > `MAX_TRADE_HOURS` (12h)
6. Fetch 1d/4h/1h/30m/15m/5m candles per asset, compute all indicators locally
7. Reset daily trade counter at UTC midnight

**Inner loop (12 ticks/hour at 5-minute intervals):**

For each asset on every tick, in strict order:
1. `_code_decide_direction()` — 4h EMA + 1h ADX≥15 + daily bias + BB width regime; returns `"buy"`, `"sell"`, or None
2. `compute_signal_score()` — weighted float 0–10 + bonuses (volume +1.0, candle pattern +0.5, Kronos ±0.5); HOLD if < 6
3. Daily cap check — HOLD if `_daily_trade_count >= MAX_DAILY_TRADES` (20)
4. SL cooldown check — HOLD if within `COOLDOWN_MINUTES` (30) of last SL hit
5. `multi_timeframe_confluence()` + `confirm_trade()` — Claude structured analysis, APPROVE if ≥15/25
6. `market_filter()` — ATR, spread, time gate (00:00–06:00 UTC), weekend gate, BTC correlation, S&R, funding
7. `entry_confirmed()` — 15m/5m RSI, ADX, volume, near-EMA, stale-setup check
8. Candle close gate — 85% of 5m candle must be elapsed
9. `oi_confirmed()` — OI must be increasing, no spike >5%
10. `_code_compute_tpsl()` — score-adaptive TP + partial close (TP1=1×ATR, TP2=3×ATR) + SL=1×ATR
11. Position sizing — `min(pct_cap, atr_cap) × (score / 10)`
12. `risk_manager.validate_trade()` — all 8 guards
13. **Execute** — LIMIT order → poll fill → cancel if unfilled after 1 candle → TP1 + TP2 + SL triggers

---

## Signal Logic (Weighted Score 0–10)

Direction is determined entirely by code. Claude never sets direction.

| Signal | Weight | BUY condition | SELL condition |
|--------|--------|---------------|----------------|
| `trend_4h` (hard gate) | 3.0 | EMA20 > EMA50 on 4h | EMA20 < EMA50 on 4h |
| `trend_1h` | 2.0 | EMA20 > EMA50 on 1h | EMA20 < EMA50 on 1h |
| `MACD_15m` | 2.0 | histogram > 0.1% of price | histogram < -0.1% of price |
| `near_ema` | 1.5 | price within 0.3% of 15m EMA20 | same |
| `trigger_5m` | 1.5 | bullish candle OR macd_5m > 0 | bearish OR macd_5m < 0 |
| Volume bonus | +1.0 | 5m vol ≥ 1.5× 5-period avg | same |
| Candle pattern | +0.5 | bullish engulfing or hammer | bearish engulfing or pin bar |
| Kronos modifier | ±0.5 | Kronos-mini agrees (optional) | — |

**Score tiers (all entries at ≥6 require Claude APPROVE when confluence fires):**
- `< 6.0` → HOLD (no trade, no Claude)
- `≥ 6.0` without multi-TF confluence → continue to market_filter/entry_confirmed directly
- `≥ MIN_AI_SCORE (6.0)` AND confluence True → Claude full analysis gate first

**Achievable base values:** 0, 1.5, 2, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 8, 8.5, 10
(Score 9 base is mathematically unreachable. Any ≥7 requires `trend_4h` aligned.)

---

## Direction Decision Gates

`_code_decide_direction()` returns None (HOLD) unless ALL four pass:

| Gate | Condition |
|------|-----------|
| 4h EMA trend | EMA20 aligned with direction |
| 1h ADX ≥ 15 | Market is trending (ADX < 15 → HOLD; ADX 15–20 → half-size) |
| Daily bias | 1d candle direction agrees |
| BB width regime | BB width above 20-period median |

---

## TP/SL Logic

Code sets all levels — Claude never touches them.

| Level | Formula | Purpose |
|-------|---------|---------|
| TP1 | entry ± 1×ATR | Close 50% (lock profit) |
| TP2 | entry ± 3×ATR | Close remaining 50% (let winner run) |
| tp_main | score-adaptive: 1.6–2.5×ATR (score≥6→1.6×, ≥7→1.8×, ≥8→2.0×, ≥9→2.2×, ≥10→2.5×) | Primary trigger order multiplier |
| SL | entry ∓ 1×ATR | Always 1:1 risk per trade |
| Trailing | Breakeven at +1×ATR; trail at +1.5×ATR | Protect open profit |

---

## Position Sizing

```
pct_cap  = account × MAX_LEVERAGE × (MAX_POSITION_PCT / 100)  # $100 × 5 × 15% = $75
atr_cap  = atr_position_size(balance, entry, sl)               # 1% risk rule ceiling
alloc    = min(pct_cap, atr_cap) × (score / 10)               # scale by signal strength
```

| Account | Buying Power | Score=10 | Score=7 |
|---------|-------------|---------|---------|
| $100 | $500 | $75 notional | $52.50 notional |
| $200 | $1,000 | $150 notional | $105 notional |
| $500 | $2,500 | $375 notional | $262.50 notional |

---

## Risk Guards

| Guard | Default | Description |
|-------|---------|-------------|
| `MAX_POSITION_PCT` | 15% | Single position cap (of buying power) |
| `MAX_LEVERAGE` | 5× | Hard leverage cap |
| `MAX_TOTAL_EXPOSURE_PCT` | 50% | All positions combined |
| `DAILY_LOSS_CIRCUIT_BREAKER_PCT` | 12% | Stops trading at 12% daily drawdown |
| `MANDATORY_SL_PCT` | 3% | Auto-sets SL if missing |
| `MAX_LOSS_PER_POSITION_PCT` | 8% | Force-close at 8% loss |
| `MAX_CONCURRENT_POSITIONS` | 3 | Concurrent position limit |
| `MIN_BALANCE_RESERVE_PCT` | 20% | Stop trading below 20% of initial balance |

---

## Setup

### Prerequisites
- Python 3.12+
- Anthropic API key
- Hyperliquid wallet (agent wallet as signer + main wallet holding funds)

### Agent Wallet Setup

1. Go to `app.hyperliquid.xyz` → Settings → API Wallets
2. Add your agent/API wallet address as an authorized signer
3. `HYPERLIQUID_PRIVATE_KEY` = agent wallet private key (signer only, cannot withdraw)
4. `HYPERLIQUID_VAULT_ADDRESS` = main wallet address (holds all funds)

### Install & Run

```bash
pip install hyperliquid-python-sdk anthropic python-dotenv aiohttp requests web3 rich

cp .env.example .env
# fill in ANTHROPIC_API_KEY, HYPERLIQUID_PRIVATE_KEY, HYPERLIQUID_VAULT_ADDRESS

python src/main.py --assets "BTC ETH SOL AVAX" --interval 1h
```

Or with Docker:
```bash
docker build -t trading-agent .
docker run --env-file .env -p 3000:3000 trading-agent
```

### Required .env Variables

```env
# ── REQUIRED ──────────────────────────────────────────────────
ANTHROPIC_API_KEY=sk-ant-...
HYPERLIQUID_PRIVATE_KEY=0x...          # Agent wallet private key (signer only)
HYPERLIQUID_VAULT_ADDRESS=0x...        # Main wallet address (holds funds)
HYPERLIQUID_NETWORK=mainnet

# ── TRADING ───────────────────────────────────────────────────
ASSETS="BTC ETH SOL AVAX"
INTERVAL=1h
MAX_TRADE_HOURS=12

# ── SCORE SYSTEM (THREE SEPARATE KEYS — never merge) ──────────
MIN_TRADE_SCORE=2        # Used ONLY by entry_confirmed() — 0-5 integer system
MIN_SIGNAL_SCORE=6       # Used ONLY by main loop pre-gate — 0-11 weighted float
MIN_AI_SCORE=6           # Minimum score to trigger Claude structured analysis

# ── AI ANALYSIS CONTROLS ──────────────────────────────────────
AI_MAX_TOKENS=4000              # Max tokens for Claude 5-factor analysis
AI_APPROVE_CACHE_MINUTES=60     # Cache APPROVE verdicts 60 min per asset
AI_REJECT_CACHE_MINUTES=30      # Cache REJECT verdicts 30 min
MIN_AI_CALL_GAP_MINUTES=30      # Hard gap between Claude calls per asset
AI_STALE_TF_MINUTES=55          # Skip Claude if higher-TF data older than this
NEWS_FETCH_ENABLED=true         # Fetch macro events + headlines for Claude context
ADX_HALF_SIZE_THRESHOLD=20      # Half position size if 1h ADX below this (no-trade gate: ADX < 15)

# ── EXECUTION CONTROLS ────────────────────────────────────────
TAKER_FEE_PCT=0.00045   # 0.045% per side — used in TP/SL fee buffer + risk manager
COOLDOWN_MINUTES=30     # Block asset for 30 min after SL hit
MAX_DAILY_TRADES=20     # Hard cap per UTC calendar day

# ── RISK MANAGEMENT ───────────────────────────────────────────
MAX_POSITION_PCT=15
MAX_LEVERAGE=5
MAX_TOTAL_EXPOSURE_PCT=50
MAX_LOSS_PER_POSITION_PCT=8
DAILY_LOSS_CIRCUIT_BREAKER_PCT=12
MANDATORY_SL_PCT=3
MAX_CONCURRENT_POSITIONS=3
MIN_BALANCE_RESERVE_PCT=20

# ── SERVER ────────────────────────────────────────────────────
API_HOST=0.0.0.0
APP_PORT=3000
```

---

## Dashboard & API

The agent starts an HTTP server on `$APP_PORT` (default 3000):

| Endpoint | Description |
|----------|-------------|
| `GET /` | Dashboard HTML |
| `GET /diary` | Per-cycle trade entries from `decisions.jsonl` |
| `GET /live` | Live account state from Hyperliquid |
| `GET /logs` | LLM request log (token usage, cost per call) |
| `GET /meta` | Bot metadata (network, auth status) |

Open `dashboard.html` in a browser — it auto-connects to `http://localhost:3000`.

---

## Project Structure

```
src/
  main.py                  # Entry point; outer + inner loops; direction, TP/SL, sizing
  config_loader.py         # All env vars with type coercion and conservative defaults
  risk_manager.py          # 8 hard guards; atr_position_size(); circuit breaker
  strategy.py              # compute_signal_score(), entry_confirmed(), market_filter(),
                           # is_trending_regime(), oi_confirmed()
  trade_state.py           # Per-asset state machine; atomic state saves
  agent/
    decision_maker.py      # confirm_trade() — Claude Sonnet structured 5-factor gate
  indicators/
    local_indicators.py    # EMA, SMA, RSI, MACD, ATR, BBands, ADX, OBV, VWAP, StochRSI
    kronos_forecast.py     # Optional Kronos-mini ML score modifier (±0.5)
  trading/
    hyperliquid_api.py     # Async Hyperliquid REST wrapper; exponential-backoff retry
  utils/
    prompt_utils.py        # JSON serialization helpers
```

## Runtime Files

| File | Content |
|------|---------|
| `diary.jsonl` | Per-trade: asset, action, allocation, TP/SL order IDs |
| `decisions.jsonl` | Per-cycle: score, decisions, account value, positions |
| `signals.jsonl` | Per-signal: every score ≥ 7 logged for win-rate analysis |
| `llm_requests.log` | Model, token usage, cost per Claude call |
| `prompts.log` | Full context payloads sent to Claude |
| `state.json` | Active trade tracking + per-asset SL cooldowns |
| `risk_state.json` | Circuit breaker state (persists across restarts) |
| `stats.json` | Win rate + cumulative PnL |

All runtime files are gitignored.

---

## API Cost

Claude Sonnet 4.6: ~$3/M input + $15/M output tokens.
Each analysis call (~1500 input + 500 output tokens) ≈ $0.012.

| Scenario | Monthly calls | Cost/month |
|----------|--------------|-----------|
| Score rarely reaches 7+ | ~5–20 | ~$0.06–0.24 |
| Moderate setups | ~30–60 | ~$0.36–0.72 |
| Active market | ~100 | ~$1.20 |

If cost exceeds $30/month — Claude being called too frequently.
Check `MIN_AI_CALL_GAP_MINUTES` is enforced and review `llm_requests.log`.

---

## License

Use at your own risk. No guarantee of returns. This code has not been audited.
