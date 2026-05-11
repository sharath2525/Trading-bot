# Hyperliquid AI Trading Agent

> **Architecture: CODE-FIRST HYBRID** — Technical analysis signals drive all decisions.
> Claude AI is called as a deep market-analysis gate when signal score ≥ 7 AND multi-timeframe
> confluence is confirmed. All direction, sizing, TP, and SL are set by code.

A perpetual futures trading bot running on Hyperliquid with a 1h outer loop / 5m inner loop.
Code makes every trade decision. Claude performs full market analysis (macro, volatility, breakout
validity) and returns `VERDICT: APPROVE` or `VERDICT: REJECT` — anything else is treated as
REJECT (fail-closed).

---

## What It Does

**Outer loop (every 1h):**
1. Fetch account state (balance, open positions, recent fills)
2. Force-close any position at or beyond `MAX_LOSS_PER_POSITION_PCT` loss
3. TP/SL guardian — re-place missing trigger orders for active positions
4. Time-based exit — force-close trades open > `MAX_TRADE_HOURS` (12h)
5. Fetch 1d/4h/1h/30m/15m/5m candles per asset, compute all indicators locally
6. Reset daily trade counter at UTC midnight

**Inner loop (runs 11 more times at 5-minute intervals = 12 ticks/hour):**

For each asset on every 5m tick, in strict order:
1. `_code_decide_direction()` — 4h hard gate; returns `"buy"`, `"sell"`, or None (HOLD)
2. `compute_signal_score()` — weighted float 0–10; HOLD if score < `MIN_SIGNAL_SCORE` (7)
3. Daily cap check — HOLD if `_daily_trade_count >= MAX_DAILY_TRADES`
4. SL cooldown check — HOLD if asset is within `COOLDOWN_MINUTES` of last SL hit
5. `market_filter()` — HOLD if ATR spike > 5% or spread > 0.15%
6. `entry_confirmed()` — HOLD if 15m/5m RSI/ADX/volume/near-EMA fail
7. `_code_compute_tpsl()` — code sets TP = entry + 2×ATR, SL = entry − 1×ATR
8. `atr_position_size()` — size = min(pct_cap, atr_cap) × (score / 10); see **Position Sizing** below
9. `confirm_trade()` — **score ≥ MIN_AI_SCORE (7) AND confluence confirmed**: Claude full market
   analysis → `VERDICT: APPROVE` or `VERDICT: REJECT` (`max_tokens=4000`, fail-closed)
10. `risk_manager.validate_trade()` — all 8 guards
11. **Execute** — market order + TP trigger + SL trigger

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

**Score tiers:**
- `< 7.0` → HOLD (no trade, no Claude)
- `≥ 7.0` without multi-TF confluence → execute directly after risk checks
- `≥ MIN_AI_SCORE (7.0)` AND confluence confirmed → Claude full market analysis gate

**Achievable score values:** 0, 1.5, 2, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 8, 8.5, 10
Score 9 is mathematically unreachable. Any score ≥ 7 requires `trend_4h` to be aligned.

---

## Position Sizing

The bot uses **percentage-of-buying-power** as the primary sizer, with the ATR 1% risk rule as a
safety ceiling:

```
buying_power  = account_balance × MAX_LEVERAGE        # e.g. $100 × 5 = $500
pct_cap       = buying_power × (MAX_POSITION_PCT / 100) # $500 × 15% = $75 notional
atr_cap       = atr_position_size(balance, entry, sl)  # 1% risk rule ceiling
alloc         = min(pct_cap, atr_cap)                  # start at $75, reduce if too risky
alloc         = alloc × (score / 10)                   # scale by signal strength
```

Examples with $100 account, 5× leverage, MAX_POSITION_PCT=15%, score=10:
- Default: $500 buying power → $75 notional → ~$15 margin on exchange
- Score 7: $75 × 0.70 = $52.50 notional → ~$10.50 margin
- $200 account: $1000 buying power → $150 notional → ~$30 margin

The ATR cap only reduces size when volatility makes the full pct_cap allocation too risky
(i.e., SL would cost more than 1% of account). TP = entry + 2×ATR, SL = entry − 1×ATR (1:2 RR).

---

## Risk Guards

All enforced by code before execution:

| Guard | Default | Description |
|-------|---------|-------------|
| `MAX_POSITION_PCT` | 15% | Single position cap (of buying power) |
| `MAX_LEVERAGE` | 5× | Hard leverage cap — set on exchange at startup |
| `MAX_TOTAL_EXPOSURE_PCT` | 50% | All positions combined |
| `DAILY_LOSS_CIRCUIT_BREAKER_PCT` | 12% | Stops new trades at 12% daily drawdown |
| `MANDATORY_SL_PCT` | 3% | Auto-sets SL if missing |
| `MAX_LOSS_PER_POSITION_PCT` | 8% | Force-closes positions at 8% loss |
| `MAX_CONCURRENT_POSITIONS` | 2 | Concurrent position limit |
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
# Install dependencies
pip install hyperliquid-python-sdk anthropic python-dotenv aiohttp requests web3 rich

# Copy and fill in config
cp .env.example .env

# Run
python3 src/main.py --assets "BTC ETH" --interval 1h
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
ASSETS="BTC ETH"
INTERVAL=1h
MAX_TRADE_HOURS=12

# ── SCORE SYSTEM (keep these separate — never merge) ──────────
MIN_TRADE_SCORE=3       # Used ONLY by entry_confirmed() — 0-5 integer system
MIN_SIGNAL_SCORE=7      # Used ONLY by main loop pre-gate — 0-10 float system

# ── AI ANALYSIS CONTROLS ──────────────────────────────────────
MIN_AI_SCORE=7                  # Score threshold to trigger Claude analysis
AI_MAX_TOKENS=4000              # Max tokens for Claude market analysis response
AI_APPROVE_CACHE_MINUTES=60     # Cache APPROVE verdicts for 60 min per asset
AI_REJECT_CACHE_MINUTES=30      # Cache REJECT verdicts for 30 min (re-checks sooner)
MIN_AI_CALL_GAP_MINUTES=30      # Hard minimum gap between Claude calls per asset
CONFLUENCE_REQUIRE_30M=true     # Require 30m TF in confluence gate

# ── EXECUTION CONTROLS ────────────────────────────────────────
TAKER_FEE_PCT=0.00045   # 0.045% per side
COOLDOWN_MINUTES=60
MAX_DAILY_TRADES=10

# ── RISK MANAGEMENT ───────────────────────────────────────────
MAX_POSITION_PCT=15
MAX_LEVERAGE=5
MAX_TOTAL_EXPOSURE_PCT=50
MAX_LOSS_PER_POSITION_PCT=8
DAILY_LOSS_CIRCUIT_BREAKER_PCT=12
MANDATORY_SL_PCT=3
MAX_CONCURRENT_POSITIONS=2
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
Set `DASHBOARD_TOKEN` in `.env` to enable Bearer-token auth on the dashboard.

---

## Project Structure

```
src/
  main.py                  # Entry point; outer + inner loops; direction, TP/SL, sizing
  config_loader.py         # All env vars with type coercion and conservative defaults
  risk_manager.py          # 8 hard guards; atr_position_size(); circuit breaker
  strategy.py              # compute_signal_score(), entry_confirmed(), market_filter()
  trade_state.py           # Per-asset state machine; atomic state saves
  agent/
    decision_maker.py      # confirm_trade() — Claude APPROVE/REJECT gate
  indicators/
    local_indicators.py    # EMA, SMA, RSI, MACD, ATR, BBands, ADX, OBV, VWAP, StochRSI
  trading/
    hyperliquid_api.py     # Async Hyperliquid REST wrapper; exponential-backoff retry
  utils/
    formatting.py          # Number formatting helpers
    prompt_utils.py        # JSON serialization helpers
```

## Runtime Files

| File | Content |
|------|---------|
| `diary.jsonl` | Per-trade: asset, action, allocation, TP/SL order IDs |
| `decisions.jsonl` | Per-cycle: score, decisions, account value, positions |
| `llm_requests.log` | Model, stop reason, token usage per Claude call |
| `prompts.log` | Full context payloads sent to Claude (can grow large) |
| `state.json` | Active trade tracking + per-asset SL cooldowns |
| `risk_state.json` | Circuit breaker state (persists across restarts) |
| `stats.json` | Win rate + cumulative PnL |

All runtime files are gitignored.

---

## API Cost

Claude Haiku pricing: ~$0.80/M input + $4/M output tokens.
Each full analysis call (~800 input + 200 output tokens) ≈ $0.0014.

| Scenario | Monthly calls | Cost/month |
|----------|--------------|-----------|
| Rarely reaches score 7+ | ~5–20 | ~$0.003–0.03 |
| Moderate setups | ~30–60 | ~$0.05–0.10 |
| Active market | ~100 | ~$0.20 |

If cost exceeds $5/month — something is calling Claude every cycle.
Check `MIN_AI_CALL_GAP_MINUTES` is enforced and review `llm_requests.log`.

---

## License

Use at your own risk. No guarantee of returns. This code has not been audited.
