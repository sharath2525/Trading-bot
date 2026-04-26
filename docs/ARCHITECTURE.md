## Trading Agent Architecture (High-Level)

This document outlines the end-to-end flow of the trading agent at a conceptual level. It focuses on subsystems, data flows, and guardrails rather than specific functions.

### Subsystems
- **Config/Env**: Centralized runtime settings from `.env` (keys, model, assets, interval).
- **Agent Runtime Loop**: 1h cycle. Coordinates all subsystems in sequence.
- **Context Builder**: Prepares the prompt context with exchange state, locally-computed trend labels (trend_4h, trend_1h, momentum_4h), indicator values, active trades, and recent fills.
- **Decision Engine**:
  - Primary LLM (Haiku): Produces structured trade decisions for all assets using canonical indicator rules in system prompt.
  - Sanitizer LLM: Fast schema-enforcing post-processor that coerces malformed outputs into valid JSON.
- **Risk/Collateral Gate**: Validates all 8 safety checks before execution. Can cap allocation or block trade entirely.
- **Execution Layer**: Places market or limit orders. Immediately places reduce-only TP and SL trigger orders after entry.
- **Reconciliation**: Resolves local intent vs exchange truth (positions/orders/fills), purges stale local state, logs outcomes.
- **Observability**: HTTP API (port 3000) serving `/diary`, `/live`, `/fills`, `/logs`.

### Data Principles
- **Authoritative Source**: Exchange state (positions, orders, fills, mids) always supersedes local intent.
- **Perp-Only Pricing**: Price context comes from Hyperliquid mids; no spot/perp basis mixing.
- **Pre-Computed Signals**: trend_4h and trend_1h (EMA20 vs EMA50) and momentum_4h (MACD histogram sign) are computed locally before Claude sees the data — removes ambiguity.
- **Compact Context**: ~1,000 effective tokens per call (with prompt caching). 1h + 4h indicators, last 3 values of each series.
- **Time Semantics**: All timestamps are UTC ISO-8601.

### Signal Flow
```
Candles (1h × 50, 4h × 30)
    ↓ compute_all() — local indicator calculation
    ↓ EMA20 vs EMA50 comparison
trend_4h / trend_1h / momentum_4h  (BULLISH / BEARISH / NEUTRAL)
    ↓ JSON context payload
Claude Haiku — decides buy / sell / hold
    ↓ trade_decisions[]
Risk Manager — 8 checks, cap or block
    ↓ validated trade
Hyperliquid SDK — limit/market order + TP + SL trigger orders
```

### Inversion Safety
The execution loop contains a mandatory assertion: if `trend_4h == "BULLISH"` and Claude outputs `action == "sell"`, a `ValueError` is raised and the trade is skipped. This catches any future regression where Claude's decision contradicts the locally-computed trend.

### Robustness
- **Retry**: Up to 3 attempts with exponential backoff on Hyperliquid API calls.
- **JSON Sanitization**: Malformed Claude output is corrected by a second Haiku call before being discarded.
- **Reconciliation**: Stale `active_trades` entries are pruned each cycle based on live exchange state.
- **Circuit Breaker**: Daily drawdown limit halts all new trades; resets at UTC midnight.
