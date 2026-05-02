# Hyperliquid AI Trading Agent

> **Architecture: CODE-FIRST HYBRID** — Technical analysis signals drive all decisions. Claude AI is called only as a final sanity check on the highest-confidence (score = 10/10) setups. All direction, sizing, TP, and SL are set by code.
>
> All 22 fixes applied as of 2026-04-30. Permanent architecture rules: see `MASTER_RULES.md`.

A code-first perpetual futures trading bot on Hyperliquid with a 1h outer loop / 5m inner loop. Claude Haiku is called only when the weighted signal score == 10 and returns a single word (APPROVE or REJECT).

## What It Does

1. **Outer loop (every 1h):** fetches account state, 4h/1h/15m/5m candles, computes all indicators locally
2. Computes trend labels locally: `EMA20 > EMA50` on 4h = BULLISH, `EMA20 < EMA50` = BEARISH
3. **Inner loop (every 5m × 12):** refreshes 5m candles, re-runs the full scoring pipeline
4. **4h hard gate:** direction must align with `trend_4h` — counter-trend trades never happen
5. **Weighted score gate (0–10):** score < 7 → HOLD; score 7–8.5 → execute directly; score == 10 → Claude APPROVE/REJECT
6. **Code computes all trade parameters:** direction, TP = entry + 2×ATR, SL = entry − 1×ATR, size = 1% risk rule
7. **Claude (score == 10 only):** receives a ~150-token context, returns APPROVE or REJECT — nothing else
8. Executes trades with take-profit and stop-loss trigger orders placed immediately after entry
9. **Time-based exit** force-closes trades open > 12h with no progress (prevents capital lock)
10. Hard-coded safety guards enforce position limits, leverage caps, daily cap, and loss protection

## Signal Logic (Code-First, 0–10 Weighted Score)

Direction is determined entirely by code. Claude never sets direction.

| Signal | BULLISH | BEARISH |
|--------|---------|---------|
| `trend_4h` (hard gate + weight 3.0) | EMA20 > EMA50 on 4h | EMA20 < EMA50 on 4h |
| `trend_1h` (weight 2.0) | EMA20 > EMA50 on 1h | EMA20 < EMA50 on 1h |
| `MACD_15m` (weight 2.0) | histogram > 0.1% of price | histogram < -0.1% of price |
| `near_ema` (weight 1.5) | price within 0.3% of 15m EMA20 | same |
| `trigger_5m` (weight 1.5) | bullish candle OR macd_5m > 0 | bearish OR macd_5m < 0 |

**Score tiers:** `< 7.0` → HOLD (no trade, no Claude) · `7.0–8.5` → execute directly · `== 10.0` → Claude APPROVE/REJECT first

**4h hard gate:** if `trend_4h` conflicts with intended direction → HOLD immediately (score not even computed).

**Score 9 is mathematically unreachable** with the above weights. Achievable values: 0, 1.5, 2, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 8, 8.5, 10.

**Allocation scales with score:** `size = 1%-risk-rule × (score / 10)` — score-7 → 70%, score-10 → 100%.

## Safety Guards

All enforced in code before execution. Configured via `.env`:

| Guard | Value | Description |
|-------|-------|-------------|
| Max Position Size | 15% | Single position capped at 15% of portfolio |
| Max Leverage | 3× | Hard leverage cap |
| Total Exposure | 50% | All positions combined capped at 50% |
| Daily Circuit Breaker | 12% | Stops new trades at 12% daily drawdown |
| Mandatory Stop-Loss | 3% | Auto-set SL if LLM omits one |
| Force Close | 8% | Auto-close positions at 8% loss (`MAX_LOSS_PER_POSITION_PCT`) |
| Max Positions | 2 | Concurrent position limit |
| Balance Reserve | 20% | Don't trade below 20% of initial balance |

TP/SL sizing: SL = 1.0 × ATR14, TP = 3.0 × ATR14 (1:3 risk-reward).

## Setup

### Prerequisites
- Python 3.12+
- Anthropic API key
- Hyperliquid wallet (agent wallet as signer + main wallet with funds)

### Configuration

```bash
cp .env.example .env
# Edit .env with your keys
```

Required environment variables:
- `ANTHROPIC_API_KEY` — Claude API key
- `HYPERLIQUID_PRIVATE_KEY` — Agent/API wallet private key (signer only)
- `HYPERLIQUID_VAULT_ADDRESS` — Main wallet address (holds funds)
- `ASSETS` — Space-separated assets to trade (e.g. `BTC ETH`)
- `INTERVAL` — Trading loop interval (`1h` recommended)

Recommended `.env` for a small account:
```env
ASSETS="BTC ETH SOL"
INTERVAL=1h
LLM_MODEL=claude-haiku-4-5-20251001
MAX_TOKENS=1200
ENABLE_TOOL_CALLING=false
MAX_TRADE_HOURS=12

# Score system (two separate keys — do not merge)
MIN_TRADE_SCORE=3        # Used ONLY by entry_confirmed() — old 0-5 integer system
MIN_SIGNAL_SCORE=7       # Used ONLY by main loop pre-gate — new 0-10 weighted float system

# Execution controls (new)
TAKER_FEE_PCT=0.00045    # 0.045% per side (Hyperliquid taker fee)
COOLDOWN_MINUTES=60      # Minutes to block re-entry after SL hit on any asset
MAX_DAILY_TRADES=10      # Hard cap: max executed trades per UTC calendar day

# Risk management
MAX_POSITION_PCT=15
MAX_LEVERAGE=3
MAX_TOTAL_EXPOSURE_PCT=50
MAX_LOSS_PER_POSITION_PCT=8
DAILY_LOSS_CIRCUIT_BREAKER_PCT=12
MANDATORY_SL_PCT=3
MAX_CONCURRENT_POSITIONS=3
MIN_BALANCE_RESERVE_PCT=20
```

### Install & Run

```bash
pip install hyperliquid-python-sdk anthropic python-dotenv aiohttp requests
python3 src/main.py
```

Or with CLI args:
```bash
python3 src/main.py --assets "BTC ETH" --interval 1h
```

### Agent Wallet Setup

1. Go to app.hyperliquid.xyz → Settings → API Wallets
2. Add your agent wallet address as an authorized signer
3. Set `HYPERLIQUID_VAULT_ADDRESS` to your main wallet address in `.env`

The agent wallet signs trades on behalf of your main wallet. It cannot withdraw funds.

## Project Structure

```
src/
  main.py                  # Entry point, trading loop, HTTP API server
  config_loader.py         # Environment config with typed defaults
  risk_manager.py          # Safety guards (position limits, loss protection)
  strategy.py              # Rule-based scoring, spread filter, entry confirmation (15m/5m)
  trade_state.py           # Per-asset state machine, trade timeout tracking
  agent/
    decision_maker.py      # Claude API integration, system prompt, tool calling
  indicators/
    local_indicators.py    # EMA, RSI, MACD, ATR, BBands, ADX, OBV, VWAP
    taapi_client.py        # Unused legacy file
  trading/
    hyperliquid_api.py     # Order execution, candles, state queries, market_close
  utils/
    formatting.py          # Number formatting helpers
    prompt_utils.py        # JSON serialization helpers
```

## Trading Loop

### Outer Loop (every 1 hour)
1. Fetch account state (balance, positions, fills)
2. Force-close any position at ≥ MAX_LOSS_PER_POSITION_PCT loss
3. TP/SL guardian — re-place missing trigger orders for ENTERED positions
4. Time-based exit — force-close trades open > MAX_TRADE_HOURS (12h)
5. Fetch 1d + 4h + 1h + 15m + 5m candles per asset, compute all indicators locally
6. Reset daily trade counter at UTC midnight

### Inner Loop (runs 11 more times at 5m intervals = 12 ticks/hour total)
For each asset on every 5m tick:
1. **4h hard gate** (`_code_decide_direction`) — HOLD if trend_4h conflicts or UNKNOWN
2. **Weighted score gate** (`compute_signal_score`) — HOLD if score < MIN_SIGNAL_SCORE (7)
3. **Daily cap** — HOLD if `_daily_trade_count >= MAX_DAILY_TRADES`
4. **SL cooldown** — HOLD if asset is within COOLDOWN_MINUTES of last stop-loss hit
5. **Market filter** — HOLD if ATR spike > 5% or spread > 0.15%
6. **Entry confirmation** — HOLD if 15m/5m don't confirm (RSI, ADX, volume, MIN_TRADE_SCORE)
7. **Code computes parameters** — TP = entry + 2×ATR, SL = entry − 1×ATR, size = 1%-risk × (score/10)
8. **Claude confirm** (score == 10 ONLY) — calls `confirm_trade()`, must return APPROVE
9. **Risk manager** — validates all 8 guards via `validate_trade()`
10. **Execute** — market order + TP trigger + SL trigger

### Execution Order (strict)
```
4h hard gate → score gate → daily cap → SL cooldown → market_filter
  → entry_confirmed → code computes params → [score==10] Claude APPROVE/REJECT
    → risk_manager.validate_trade → execute_trade
```

## API Endpoints

When running, serves a local HTTP API on port 3000:
- `GET /` — Dashboard HTML
- `GET /diary` — Cycle decision log (JSON)
- `GET /live` — Live account state from Hyperliquid
- `GET /logs` — LLM request log with per-call token counts and cost

## API Cost

**Before redesign (Claude every cycle, all assets):**
- ~720 calls/month at ~$0.005/call = **~$3.60–7.20/month**

**After redesign (Claude only for score == 10):**
- ~5–20 confirmations/month at ~$0.00016/call = **~$0.003/month**
- Model: `claude-haiku-4-5-20251001`, `max_tokens=10`
- If monthly API cost exceeds $1, something is calling Claude every cycle — check main.py

## License

Use at your own risk. No guarantee of returns. This code has not been audited.
