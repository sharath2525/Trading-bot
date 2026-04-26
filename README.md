# Hyperliquid AI Trading Agent

An AI-powered perpetual futures trading bot that uses Claude (Haiku) to analyze markets and execute trades on Hyperliquid. Runs on a 1-hour decision cycle trading BTC and ETH.

## What It Does

1. Fetches 1h + 4h + **15m + 5m** candle data and computes all technical indicators locally
2. Computes trend labels locally: `EMA20 > EMA50` on 4h = BULLISH, `EMA20 < EMA50` = BEARISH
3. Sends compact market context to Claude, which decides buy/sell/hold for each asset
4. **Spread filter** blocks entry if bid/ask spread > 0.15% (illiquid market protection)
5. **15m/5m entry confirmation** blocks overextended entries — waits for pullback to EMA20 and 5m trigger
6. Executes trades with take-profit and stop-loss trigger orders placed immediately after entry
7. **Time-based exit** force-closes trades open > 12h with no progress (prevents capital lock)
8. Hard-coded safety guards enforce position limits, leverage caps, and loss protection

## Signal Logic

Trend direction is determined locally before Claude sees the data:

| Signal | BULLISH | BEARISH |
|--------|---------|---------|
| `trend_4h` (primary) | EMA20 > EMA50 on 4h | EMA20 < EMA50 on 4h |
| `trend_1h` (confirmation) | EMA20 > EMA50 on 1h | EMA20 < EMA50 on 1h |
| `momentum_4h` | MACD histogram > 0 | MACD histogram < 0 |
| RSI14 | > 50 | < 50 |

Both 4h trend and 1h trend must agree to enter. Counter-trend trades require strong contrary evidence.

## Safety Guards

All enforced in code before execution. Configured via `.env`:

| Guard | Value | Description |
|-------|-------|-------------|
| Max Position Size | 15% | Single position capped at 15% of portfolio |
| Max Leverage | 3× | Hard leverage cap |
| Total Exposure | 50% | All positions combined capped at 50% |
| Daily Circuit Breaker | 12% | Stops new trades at 12% daily drawdown |
| Mandatory Stop-Loss | 3% | Auto-set SL if LLM omits one |
| Force Close | 20% | Auto-close positions at 20% loss |
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
ASSETS="BTC ETH"
INTERVAL=1h
LLM_MODEL=claude-haiku-4-5-20251001
MAX_TOKENS=1200
ENABLE_TOOL_CALLING=false
ENABLE_CLAUDE_COMMENTARY=false
MIN_TRADE_SCORE=7
MAX_TRADE_HOURS=12
MAX_POSITION_PCT=15
MAX_LEVERAGE=3
MAX_TOTAL_EXPOSURE_PCT=50
MAX_LOSS_PER_POSITION_PCT=8
DAILY_LOSS_CIRCUIT_BREAKER_PCT=12
MANDATORY_SL_PCT=3
MAX_CONCURRENT_POSITIONS=2
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

Each 1h iteration:
1. Fetch account state (balance, positions, PnL)
2. Force-close any position at ≥ 8% loss
3. **Time-based exit**: force-close any trade open > 12h with no TP hit
4. Fetch 1h + 4h + **15m + 5m** candles, compute indicators locally
5. Calculate spread from cached metadata (no extra API call)
6. Send compact context to Claude (≈1,000 tokens, cached system prompt)
7. Claude returns buy/sell/hold with allocation, TP/SL
8. **Spread filter**: block if spread > 0.15%
9. **Entry confirmation**: block if 15m/5m don't confirm the 1h direction
10. Risk manager validates each trade (caps allocation, enforces SL/TP)
11. Execute approved trades (limit orders preferred, TP+SL placed immediately)
12. Record entry time in state machine for timeout tracking

## API Endpoints

When running, serves a local HTTP API on port 3000:
- `GET /` — Dashboard HTML
- `GET /diary` — Cycle decision log (JSON)
- `GET /live` — Live account state from Hyperliquid
- `GET /logs` — LLM request log with per-call token counts and cost

## API Cost

- Model: `claude-haiku-4-5-20251001`
- Tokens per call: ~1,000 input (with prompt cache hit), ~400 output
- Cost per call: ~$0.0016
- Cost at 1h interval, 2 assets: ~$0.03/day (~$1/month)
- System prompt is cached (ephemeral cache_control) — effective from 2nd call onward

## License

Use at your own risk. No guarantee of returns. This code has not been audited.
